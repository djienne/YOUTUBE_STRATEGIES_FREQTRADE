import pandas as pd
from datetime import datetime, timezone, timedelta
from freqtrade.strategy import IStrategy, Order
from freqtrade.persistence import Trade
import logging
import math
import os
import json
from pathlib import Path
from delta_hedge import SpotHedge, SPOT_TOKENS, number

logger = logging.getLogger(__name__)

def write_log(message):
    logger.info("%s", message)

def _iso_ms(dt: datetime) -> str:
    """ISO-8601 with milliseconds and timezone offset, e.g. 2025-08-06T04:00:00.000+00:00"""
    # Ensure timezone-aware UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="milliseconds")

def _to_hour_start(dt: datetime) -> datetime:
    """Floor to the start of the hour."""
    return dt.replace(minute=0, second=0, microsecond=0)

def load_db(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("Funding DB must be an object")
    return result  # Never overwrite a corrupt database with an empty one.


def save_db(path: str, db: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(db, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def update_funding_db(
    db_path: str,
    records: list[dict],
    market_key: str,  # e.g. "BTC/USDC:USDC"
    value_transform=lambda s: float(s),  # how to turn fundingRate string into number
    align_to_hour: bool = True,
) -> dict:
    """
    Update the DB with records like:
    {"coin": "BTC", "fundingRate": "0.0000125", "time": 1753862400004}

    - market_key is where the value is stored, e.g. "BTC/USDC:USDC"
    - An old forecast row marked harvested_datetime is discarded, not promoted
    - Settled exchange records replace any earlier value for the same hour
    - Returns the updated DB (also saved to db_path)
    """
    db_path=str(db_path)
    db = load_db(db_path)

    for r in records:
        # 1) Convert ms epoch -> UTC datetime
        dt = datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc)
        if align_to_hour:
            dt = _to_hour_start(dt)

        ts_key = _iso_ms(dt)  # e.g. 2025-08-06T04:00:00.000+00:00
        val = number(value_transform(r["fundingRate"]), signed=True)*24.0*365.0*100.0

        # 2) Ensure record exists for this timestamp
        if ts_key not in db or not isinstance(db[ts_key], dict) or 'harvested_datetime' in db[ts_key]:
            db[ts_key] = {}

        db[ts_key][market_key] = val

    save_db(db_path, db)
    return db

def get_funding_history(db_path, coin, days_interval=8, info=None, now_utc=None):
    """Fetch settled hourly funding, paginating with the last timestamp + 1 ms."""
    if days_interval <= 0:
        raise ValueError("Funding lookback must be positive")
    if info is None:
        from hyperliquid.info import Info
        from hyperliquid.utils.constants import MAINNET_API_URL
        info = Info(MAINNET_API_URL, skip_ws=True, timeout=10)
    now = now_utc or datetime.now(timezone.utc)
    end = int(now.timestamp()*1000)
    cursor = int((now-timedelta(days=days_interval)).timestamp()*1000)
    records = []
    while cursor <= end:
        batch = info.funding_history(coin, startTime=cursor, endTime=end)
        if not batch:
            break
        times = [int(row['time']) for row in batch]
        if min(times) < cursor or max(times) > end:
            raise ValueError("Funding API returned records outside requested interval")
        if any(row.get('coin') != coin for row in batch):
            raise ValueError("Funding API returned the wrong market")
        records.extend(batch)
        cursor = max(times)+1
    update_funding_db(db_path, records, f"{coin}/USDC:USDC")
    return True


def _funding_samples(file_path=None, now_utc=None, hours=24):
    if not isinstance(hours, int) or isinstance(hours, bool) or hours <= 0:
        raise ValueError("Funding window must be a positive integer number of hours")
    file_path = Path(file_path) if file_path else Path(__file__).with_name('settled_funding_rates.json')
    now = now_utc or datetime.now(timezone.utc)
    now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    start = now-timedelta(hours=hours)
    samples = {}
    for timestamp, pairs in load_db(file_path).items():
        if not isinstance(pairs,dict) or 'harvested_datetime' in pairs:
            continue  # old forecast observations are not settled payments
        try:
            when = datetime.fromisoformat(timestamp.replace('Z','+00:00'))
            when = when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue
        if not start < when <= now:
            continue
        for pair, value in pairs.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                samples.setdefault(pair,{})[_to_hour_start(when)] = float(value)
    return samples


def avg_funding_last_hours(pair, printt=False, nb_hours=24, file_path=None, now_utc=None) -> float:
    """Mean settled funding APR; require every requested hourly observation."""
    values = _funding_samples(file_path, now_utc, nb_hours).get(pair,{})
    if len(values) != nb_hours:
        raise ValueError(f"Incomplete settled funding coverage for {pair}: {len(values)}/{nb_hours}")
    average = sum(values.values())/len(values)
    if printt:
        write_log(f"Settled funding APR on {pair}, {nb_hours}h: {average:.3f}%")
    return average


def funding_negative_last_Xhours(pair, printt=False, nb_hours=24, file_path=None, now_utc=None):
    return avg_funding_last_hours(pair, printt, nb_hours, file_path, now_utc) < 0


def log_equity(
    total_equity: float,
    min_interval_minutes: int = 59,
):
    """
    Logs `total_equity` for `address` to CSV, with abs/rel change vs first recorded value.
    Appends only if last row is older than `min_interval_minutes`.

    CSV columns:
      timestamp, total_equity_usdc, abs_change_from_first, rel_change_from_first
    """

    # --- imports inside for easy copy/paste ---
    from datetime import datetime, timezone
    from decimal import Decimal, InvalidOperation
    import csv, os
    from typing import Optional

    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def _read_last_timestamp(csv_path: str) -> Optional[datetime]:
        if not os.path.exists(csv_path):
            return None
        with open(csv_path, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode("utf-8").strip()
        if not last_line or last_line.startswith("timestamp"):
            return None
        try:
            return datetime.fromisoformat(last_line.split(",")[0])
        except ValueError:
            return None

    def _read_first_total(csv_path: str) -> Optional[Decimal]:
        if not os.path.exists(csv_path):
            return None
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            try:
                _ = next(f)
            except StopIteration:
                return None
            for row in r:
                if not row or row[0].strip().lower() == "timestamp":
                    continue
                if len(row) >= 2 and row[1].strip():
                    try:
                        return Decimal(row[1].strip())
                    except InvalidOperation:
                        return None
                break
        return None

    def _write_sample(csv_path: str, when: datetime, equity: Decimal, first_total: Optional[Decimal]) -> None:
        file_exists = os.path.exists(csv_path)
        abs_change, rel_change = "", ""
        if first_total is not None:
            try:
                abs_change_val = equity - first_total
                abs_change = str(abs_change_val)
                if first_total != 0:
                    rel_change = str((equity / first_total) - Decimal("1"))
            except InvalidOperation:
                pass
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow([
                    "timestamp",
                    "total_equity_usdc",
                    "abs_change_from_first",
                    "rel_change_from_first"
                ])
            w.writerow([when.isoformat(), str(equity), abs_change, rel_change])

    # Build file path
    filename = f"equity_track.csv"
    here = Path(__file__).resolve().parent
    csv_path = os.path.join(here, filename)

    # Decide whether to append
    when = _now_utc()
    last_ts = _read_last_timestamp(csv_path)
    if last_ts is None or (when - last_ts).total_seconds() >= min_interval_minutes * 60:
        equity_decimal = Decimal(str(total_equity))
        first_total = _read_first_total(csv_path)
        _write_sample(csv_path, when, equity_decimal, first_total)

def record_hourly_funding_by_pair(pair, funding_val, tz=None, file_path=None):
    """Keep observed predictions separate from settled exchange funding."""
    file_path = Path(file_path) if file_path else Path(__file__).with_name('funding_observations.json')
    now = datetime.now(tz or timezone.utc).astimezone(timezone.utc)
    db = load_db(file_path)
    row = db.setdefault(_iso_ms(_to_hour_start(now)),{})
    row.update(harvested_datetime=_iso_ms(now))
    row[pair] = number(funding_val, signed=True)
    save_db(file_path, db)


def average_funding_last_7_days(file_path=None, now_utc=None):
    return {pair:sum(values.values())/len(values)
            for pair,values in _funding_samples(file_path,now_utc,24*7).items() if values}


def print_average_funding_last_7_days(file_path=None):
    write_log(f"Settled funding APR averages: {average_funding_last_7_days(file_path)}")


class DELTA_NEUTRAL(IStrategy):
    INTERFACE_VERSION = 3
    minimal_roi = {"0": 5000.0}
    stoploss = -0.90
    timeframe = '5m'
    startup_candle_count = 0
    can_short = True
    process_only_new_candles = False
    position_adjustment_enable = False
    MINIMUM_FUNDING_APR_pc = 10
    MINIMUM_VOLUME_usdc = 2_500_000
    MINIMUM_TIME_TO_KEEP_POSITION_hour = 24*30
    FORCE_EXIT = False
    order_types = {'entry':'market','exit':'market','stoploss':'market','stoploss_on_exchange':False}
    order_time_in_force = {'entry':'gtc','exit':'gtc'}
    _initialized = False
    _trading_ready = False

    def _mode(self):
        mode = self.config.get('runmode')
        return getattr(mode,'value',mode)

    def _max_positions(self):
        count = self.config.get('max_open_trades')
        if not isinstance(count,int) or isinstance(count,bool) or count <= 0:
            raise ValueError('max_open_trades must be a finite positive integer')
        return count

    def select_BEST_PAIRS(self, quote_volumes, fundings, max_pairs=2):
        return sorted((p for p in fundings if p in quote_volumes
                       and fundings[p]>self.MINIMUM_FUNDING_APR_pc
                       and quote_volumes[p]>self.MINIMUM_VOLUME_usdc),
                      key=lambda p:(fundings[p],quote_volumes[p]),reverse=True)[:max_pairs]

    def PAIR_SHOULD_BE_REPLACED(self, current_pair):
        # Unknown/stale data cannot be evidence that another pair is better.
        return bool(self.BEST_PAIRS and current_pair not in self.BEST_PAIRS
                    and max(self.FUNDINGS[p] for p in self.BEST_PAIRS) > self.FUNDINGS.get(current_pair,float('inf')))

    def bot_start(self, **kwargs):
        self._initialized = self._trading_ready = False
        self.FUNDINGS, self.QUOTE_VOLUMES, self.BEST_PAIRS = {}, {}, []
        self.CURRENT_POSITION_PAIRS = []
        self._expected_positions = {}
        self._funding_updated = self._equity_logged = None
        self._rebalance_needed = True
        self.nb_loop = 0
        self._hedge = None
        self._max_positions()
        if self._mode() not in ('live','dry_run'):
            raise ValueError('This live funding strategy does not implement candle backtests; use funding_db_and_backtest')
        if (self.config.get('trading_mode') != 'futures' or self.config.get('margin_mode') != 'isolated'
                or self.config.get('stake_currency') != 'USDC' or self.config['exchange']['name'] != 'hyperliquid'):
            raise ValueError('Requires Hyperliquid isolated USDC futures')
        self._pairs = list(self.config['exchange']['pair_whitelist'])
        if not self._pairs or any(p.split('/')[0] not in SPOT_TOKENS or not p.endswith('/USDC:USDC') for p in self._pairs):
            raise ValueError('Unsupported or empty hedge pair list')
        self._funding_path = Path(__file__).with_name('settled_funding_rates.json')
        if self._mode() == 'live':
            self._hedge = SpotHedge(self.config,Path(__file__).with_name('delta_neutral_state.json'))
            self._info = self._hedge.info
        else:
            from hyperliquid.info import Info
            from hyperliquid.utils.constants import MAINNET_API_URL
            self._info = Info(MAINNET_API_URL,skip_ws=True,timeout=10)
        self._initialized = True

    def bot_loop_start(self, current_time, **kwargs):
        self._trading_ready = False
        self.FUNDINGS, self.QUOTE_VOLUMES, self.BEST_PAIRS = {}, {}, []
        if not self._initialized:
            return
        self.nb_loop += 1
        try:
            trades = Trade.get_trades_proxy(is_open=True)
            self.CURRENT_POSITION_PAIRS = [t.pair for t in trades]
            if len(trades)>self._max_positions() or any(not t.is_short or t.leverage!=1 for t in trades):
                raise ValueError('Unexpected Freqtrade position count, side or leverage')
            if self._hedge is not None:
                if not self._hedge.reconcile(self._expected_positions):
                    return
                self._expected_positions.clear()
                if self._rebalance_needed and len(trades)<self._max_positions():
                    if self._hedge.rebalance():
                        return
                    self._rebalance_needed = False
            if self._funding_updated is None or _to_hour_start(current_time)>_to_hour_start(self._funding_updated):
                # Mark unavailable before any request; failed refreshes cannot reuse eligibility.
                self._funding_updated = None
                for pair in self._pairs:
                    get_funding_history(self._funding_path,pair.split('/')[0],8,self._info,current_time)
                    avg_funding_last_hours(pair,nb_hours=3,file_path=self._funding_path,now_utc=current_time)
                self._funding_updated = current_time
            self._trading_ready = True
        except Exception as exc:
            write_log(f'New entries blocked: {type(exc).__name__}: {exc}')
        # A reporting failure must not interrupt order management.
        if self._hedge is not None and (self._equity_logged is None or current_time-self._equity_logged>=timedelta(hours=1)):
            try:
                log_equity(self._hedge.equity())
                self._equity_logged = current_time
            except Exception as exc:
                write_log(f'Equity unavailable: {type(exc).__name__}')

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['entry_signal'] = 0
        pair = metadata['pair']
        if not self._initialized or self.FORCE_EXIT:
            return df
        self.FUNDINGS.pop(pair,None)
        self.QUOTE_VOLUMES.pop(pair,None)
        try:
            if not self._trading_ready:
                return df
            ticker = self.dp.ticker(pair)
            apr = number(ticker['info']['funding'],signed=True)*24*365*100
            volume = number(ticker['quoteVolume'])
            record_hourly_funding_by_pair(pair,apr)
            average = avg_funding_last_hours(pair,nb_hours=3,file_path=self._funding_path)
            self.FUNDINGS[pair], self.QUOTE_VOLUMES[pair] = average, volume
            if average>self.MINIMUM_FUNDING_APR_pc and volume>self.MINIMUM_VOLUME_usdc:
                if not df.empty:
                    df.loc[df.index[-1],'entry_signal'] = -1
        except Exception as exc:
            write_log(f'No usable funding signal for {pair}: {type(exc).__name__}')
        self.BEST_PAIRS = self.select_BEST_PAIRS(self.QUOTE_VOLUMES,self.FUNDINGS,self._max_positions())
        return df

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_short'] = 0
        dataframe['enter_long'] = 0
        if not self.FORCE_EXIT and self._trading_ready:
            dataframe.loc[dataframe['entry_signal']==-1,'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe['exit_short'] = int(self.FORCE_EXIT)
        dataframe['exit_long'] = 0
        return dataframe

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        # Never place a spot order before the perp is confirmed filled.
        try:
            if (not self._trading_ready or self.FORCE_EXIT or side!='short'
                    or pair not in self.BEST_PAIRS or pair in self.CURRENT_POSITION_PAIRS
                    or Trade.get_open_trade_count()>=self._max_positions() or number(amount)<=0):
                return False
            if self._mode()=='live' and not self._hedge.can_fund(pair.split('/')[0],amount):
                return False
            if self._mode() not in ('live','dry_run'):
                return False
            self._trading_ready = False  # serialize entry attempts until the next reconciliation
            return True
        except Exception:
            return False  # Freqtrade's wrapper otherwise defaults to approving entry on exceptions

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force,
                           exit_reason, current_time, **kwargs):
        # A spot failure must never veto a stoploss or other risk-reducing perp exit.
        self._trading_ready = False
        return True

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if current_profit>0.50: return 'upper_limit_rebalance'
        if current_profit<-0.50: return 'lower_limit_rebalance'
        if current_time-trade.open_date_utc < timedelta(hours=self.MINIMUM_TIME_TO_KEEP_POSITION_hour):
            return None
        if self._trading_ready and self.PAIR_SHOULD_BE_REPLACED(pair):
            return 'timeout_and_better'
        try:
            if funding_negative_last_Xhours(pair,nb_hours=24,file_path=self._funding_path,now_utc=current_time):
                return 'timeout_and_negative_fundings_avg24h'
        except (OSError,ValueError):
            write_log(f'Funding-based exit unavailable for {pair}')
        return None

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        try:
            slots = self._max_positions()-Trade.get_open_trade_count()
            if not self._trading_ready or slots<=0 or leverage!=1 or side!='short':
                return 0.0
            stake = max(0.0,number(max_stake)/slots-0.5)
            if self._mode()=='live':
                hedge = self._hedge
                price = hedge.prices[pair.split('/')[0]]
                spot_cap = hedge.spot_free*number(current_rate)/price*(1-hedge.fee)/((1+hedge.slippage)*(1+hedge.fee))
                stake = min(stake,spot_cap)
            return stake if stake>=number(min_stake or 0) else 0.0
        except Exception:
            return 0.0  # Never fall back to Freqtrade's proposed stake after a sizing error.

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs):
        if self._mode()!='live' or self._hedge is None or order.safe_filled<=0:
            return
        self._trading_ready = False
        self._rebalance_needed = True
        # Freqtrade 2025.10 updates trade.amount/is_open before this callback.
        # Keep this expectation until the exchange snapshot reflects the fill.
        self._expected_positions[pair.split('/')[0]] = number(trade.amount) if trade.is_open else 0.0
        try:
            self._hedge.reconcile(self._expected_positions)
        except Exception as exc:
            write_log(f'Hedge requires attention; new entries blocked: {type(exc).__name__}: {exc}')

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage,
                 entry_tag, side, **kwargs):
        return 1.0
