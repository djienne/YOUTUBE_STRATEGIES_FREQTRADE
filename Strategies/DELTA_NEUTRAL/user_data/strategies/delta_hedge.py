"""Reconcile a dedicated standard Hyperliquid account's spot/perp hedge.

Spot and perp orders are not atomic. Persist intent before sending a spot order
or transfer; an ambiguous response blocks further actions across restarts.
"""
import hashlib
import json
import logging
import math
import os
import time
import uuid
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

SPOT_TOKENS = {
    'BTC': 'UBTC', 'ETH': 'UETH', 'SOL': 'USOL', 'HYPE': 'HYPE',
    'PUMP': 'UPUMP', 'FARTCOIN': 'UFART', 'PURR': 'PURR',
}
MIN_NOTIONAL = 10.0
logger = logging.getLogger(__name__)


def number(value, *, signed=False):
    value = float(value)
    if not math.isfinite(value) or (not signed and value < 0):
        raise ValueError('Non-finite or invalid negative exchange value')
    return value


def floor_size(value, decimals):
    return float(Decimal(str(number(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_DOWN))


def check_signer(info, address, signer):
    # account_address on the SDK client is not an order-routing field: the
    # signing key determines which account receives the signed action.
    role = info.post('/info', {'type':'userRole','user':address})
    if role.get('role') != 'user':
        raise ValueError('walletAddress must be the actual master trading account')
    if signer.lower() != address.lower():
        agent = info.post('/info', {'type':'userRole','user':signer})
        if agent.get('role') != 'agent' or agent.get('data',{}).get('user','').lower() != address.lower():
            raise ValueError('API signing key is not an agent for the configured account')


class SpotHedge:
    def __init__(self, config, state_path, *, info=None, exchange=None, transfer_exchange=None):
        if config.get('dry_run', True):
            raise ValueError('SpotHedge is live-only; never construct it in dry-run')
        credentials = config['exchange']
        self.address = credentials['walletAddress']
        self.path = Path(state_path)
        self._storage_failed = False
        self.account_id = hashlib.sha256(self.address.lower().encode()).hexdigest()
        self.slippage = number(config.get('delta_neutral_spot_slippage', 0.005))
        if not 0 < self.slippage <= 0.02:
            raise ValueError('Spot slippage must be > 0 and <= 2%')
        if info is None:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils.constants import MAINNET_API_URL
            from eth_account import Account
            info = Info(MAINNET_API_URL, skip_ws=True, timeout=10)
            signer = Account.from_key(credentials['privateKey'])
            check_signer(info,self.address,signer.address)
            exchange = Exchange(signer, MAINNET_API_URL,
                                account_address=self.address, timeout=10)
            if credentials.get('privateKeyEthWallet'):
                wallet = Account.from_key(credentials['privateKeyEthWallet'])
                if wallet.address.lower() != self.address.lower():
                    raise ValueError('USDC transfers require the configured master account signer')
                transfer_exchange = Exchange(wallet, MAINNET_API_URL, account_address=self.address, timeout=10)
        self.info, self.exchange, self.transfer_exchange = info, exchange, transfer_exchange
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'account': self.account_id, 'pending': None}
        if self.state.get('account') != self.account_id or 'pending' not in self.state:
            raise ValueError('Invalid hedge state or state belongs to another account')
        # This strategy assumes separate spot/perp USDC balances and isolated 1x perps.
        # Default/unified/portfolio-margin modes need a different collateral model.
        self._check_account_mode()
        meta = info.spot_meta()
        tokens = {t['index']: t for t in meta['tokens']}
        self.markets = {}
        for pair in credentials['pair_whitelist']:
            coin, suffix = pair.split('/', 1)
            if suffix != 'USDC:USDC' or coin not in SPOT_TOKENS:
                raise ValueError('Unsupported perp/spot hedge pair: '+pair)
            matches = []
            for market in meta['universe']:
                base, quote = [tokens[i] for i in market['tokens']]
                if base['name'] == SPOT_TOKENS[coin] and quote['name'] == 'USDC' and quote['index'] == 0:
                    matches.append((market['name'], base['index'], base['szDecimals']))
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous USDC spot market for '+coin)
            self.markets[coin] = matches[0]
        self._refresh_fee()
        self.ready = False
        self.prices, self.spot_free = {}, 0.0
        self._save()  # prove the journal is writable before allowing a perp entry

    def _check_account_mode(self):
        if self.info.post('/info', {'type': 'userAbstraction', 'user': self.address}) != 'disabled':
            raise ValueError('Only explicitly disabled account abstraction is supported')

    def _refresh_fee(self):
        self.fee = number(self.info.user_fees(self.address)['userSpotCrossRate'])
        if self.fee >= 0.01:
            raise ValueError('Unexpected spot taker fee')
        self._fee_updated = time.monotonic()

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            with tmp.open('w', encoding='utf-8') as handle:
                json.dump(self.state, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except Exception:
            self._storage_failed = True
            self.ready = False
            raise

    def _pending(self, value):
        self.state['pending'] = value
        self._save()

    def _balances(self):
        balances = {}
        for row in self.info.spot_user_state(self.address)['balances']:
            token = int(row['token'])
            if token in balances:
                raise ValueError('Duplicate spot balance token')
            total, hold = number(row['total']), number(row['hold'])
            if hold > total + 1e-8:
                raise ValueError('Spot hold exceeds total')
            balances[token] = (total, max(0.0, total-hold))
        return balances

    def _settled(self, balances):
        pending = self.state['pending']
        if pending is None:
            return True
        # A timeout or unrecognized response has no proven outcome. Never guess
        # that it failed, and never issue a replacement. Inspect the recorded cloid.
        if 'expected' not in pending:
            raise RuntimeError('Ambiguous external action: reconcile delta_neutral_state.json manually')
        actual = balances.get(pending['token'], (0.0, 0.0))[0]
        lower, upper = pending['expected']
        if not lower <= actual <= upper:
            logger.warning('Awaiting confirmed %s settlement; new actions blocked', pending['kind'])
            return False  # acknowledged fill/transfer not reflected in balances yet
        self._pending(None)
        return True

    def _submit(self, coin, is_buy, size, before, price):
        from hyperliquid.utils.types import Cloid
        market, token, decimals = self.markets[coin]
        cloid = Cloid.from_str('0x'+uuid.uuid4().hex)
        pending = {'kind': 'spot', 'coin': coin, 'token': token, 'cloid': cloid.to_raw(),
                   'is_buy': is_buy, 'size': size, 'before': before}
        self._pending(pending)  # write-ahead: a crash must not trigger a blind retry
        result = self.exchange.market_open(market, is_buy, size, price, self.slippage, cloid=cloid)
        if result.get('status') == 'err':
            self._pending(None)
            raise RuntimeError('Spot order rejected')
        if result.get('status') != 'ok':
            raise RuntimeError('Unrecognized spot response')
        statuses = result.get('response', {}).get('data', {}).get('statuses', [])
        if len(statuses) != 1:
            raise RuntimeError('Unexpected spot order statuses')
        status = statuses[0]
        if isinstance(status, dict) and 'error' in status:
            self._pending(None)
            raise RuntimeError('Spot IOC rejected')
        filled = status.get('filled', {}) if isinstance(status, dict) else {}
        amount = number(filled.get('totalSz', 0))
        if not 0 < amount <= size + 10**(-decimals):
            raise RuntimeError('Unconfirmed or invalid spot fill')
        # Partial IOC fills are terminal, but not proof of a complete hedge.
        # Observe the actual balance before computing the remaining quantity.
        lot = 10**(-decimals)
        expected = ([before+amount*(1-self.fee)-lot, before+amount+lot] if is_buy
                    else [before-amount-lot, before-amount+lot])
        pending['expected'] = expected
        pending['oid'] = filled['oid']
        self._pending(pending)
        logger.info('Spot IOC %s: filled %s of %s; awaiting balance confirmation', coin, amount, size)

    def _plan(self, coin, target, total, free):
        """One sizing/precision/minimum check for both preflight and execution."""
        _, _, decimals = self.markets[coin]
        delta, price = target-total, self.prices[coin]
        tolerance = max(10**(-decimals), target*self.fee, 0.01/price)
        if abs(delta) <= tolerance or (target == 0 and total*price < MIN_NOTIONAL):
            return None
        is_buy = delta > 0
        size = floor_size(delta/(1-self.fee) if is_buy else min(-delta,free), decimals)
        if size <= 0 or size*price*(1-self.slippage) < MIN_NOTIONAL:
            raise RuntimeError('Material hedge gap cannot meet spot minimum size/notional')
        if is_buy and size*price*(1+self.slippage)*(1+self.fee) > self.spot_free:
            raise RuntimeError('Insufficient free spot USDC to hedge the filled perp')
        return is_buy, size, price

    def reconcile(self, expected=None):
        self.ready = False
        if self._storage_failed:
            raise RuntimeError('Hedge journal write failed; repair storage and restart after reconciliation')
        self._check_account_mode()
        balances = self._balances()
        if not self._settled(balances):
            return False
        if time.monotonic()-self._fee_updated >= 3600:
            self._refresh_fee()
        # Never overlap manual orders, Freqtrade's in-flight perp orders or our IOC.
        if self.info.open_orders(self.address):
            return False
        state = self.info.user_state(self.address)
        targets = {coin: 0.0 for coin in self.markets}
        for row in state['assetPositions']:
            position = row['position']
            amount = number(position['szi'], signed=True)
            if amount == 0:
                continue
            coin = position['coin']
            if coin not in targets or amount > 0:
                raise ValueError('Unexpected or long perpetual position on dedicated account')
            if position['leverage']['type'] != 'isolated' or number(position['leverage']['value']) != 1:
                raise ValueError('Only isolated 1x short positions are supported')
            targets[coin] = -amount
        for coin, amount in (expected or {}).items():
            if coin not in targets or not math.isclose(targets[coin], amount, rel_tol=1e-8, abs_tol=1e-10):
                return False  # wait for the known Freqtrade fill to reach account state
        mids = self.info.all_mids()
        self.prices = {coin: number(mids[market]) for coin,(market,_,_) in self.markets.items()}
        if any(price <= 0 for price in self.prices.values()):
            raise ValueError('Missing spot valuation')
        known_tokens = {token for _,token,_ in self.markets.values()} | {0}
        if any(token not in known_tokens and total > 0 for token,(total,_) in balances.items()):
            raise ValueError('Unsupported spot holdings on dedicated account')
        self.spot_free = balances.get(0, (0.0,0.0))[1]
        self.balances = balances
        for coin, (_,token,decimals) in self.markets.items():
            target = targets[coin]
            total, free = balances.get(token, (0.0,0.0))
            plan = self._plan(coin,target,total,free)
            if plan is None:
                continue
            is_buy, size, price = plan
            self._submit(coin,is_buy,size,total,price)
            return False  # next snapshot confirms balance and any residual gap
        self.ready = True
        return True

    def can_fund(self, coin, size):
        if not self.ready or coin not in self.prices:
            return False
        try:
            token = self.markets[coin][1]
            target = number(size)
            if target <= 0:
                return False
            self._plan(coin,target,*self.balances.get(token,(0.0,0.0)))
            return True
        except (ValueError, RuntimeError):
            return False

    def rebalance(self):
        if not self.ready or self.transfer_exchange is None or self.state['pending']:
            return False
        self._check_account_mode()
        if self.info.open_orders(self.address):
            return False
        balances = self._balances()
        spot_total, spot_free = balances.get(0,(0.0,0.0))
        # accountValue includes collateral/unrealized P&L, not transferable funds.
        perp_free = number(self.info.user_state(self.address)['withdrawable'])
        total = spot_free+perp_free
        if total < 22 or abs(spot_free-perp_free)/total < 0.005:
            return False
        amount = floor_size(abs(spot_free-perp_free)/2,2)
        if amount <= 0:
            return False
        to_perp = spot_free > perp_free
        pending = {'kind':'transfer','token':0,'amount':amount,'to_perp':to_perp}
        self.ready = False
        self._pending(pending)
        result = self.transfer_exchange.usd_class_transfer(amount,to_perp)
        if result.get('status') == 'err':
            self._pending(None)
            raise RuntimeError('USDC transfer rejected')
        if result.get('status') != 'ok' or result.get('response',{}).get('type') != 'default':
            raise RuntimeError('Unconfirmed USDC transfer')
        expected = spot_total + (-amount if to_perp else amount)
        pending['expected'] = [expected-0.005, expected+0.005]
        self._pending(pending)
        return True

    def equity(self):
        mids = self.info.all_mids()
        prices = {token:number(mids[market]) for market,token,_ in self.markets.values()}
        prices[0] = 1.0
        total = number(self.info.user_state(self.address)['marginSummary']['accountValue'],signed=True)
        for token,(amount,_) in self._balances().items():
            if amount == 0: continue
            if token not in prices or prices[token] <= 0:
                raise ValueError('Cannot value spot holding; refusing a misleading equity total')
            total += amount*prices[token]
        return number(total, signed=True)
