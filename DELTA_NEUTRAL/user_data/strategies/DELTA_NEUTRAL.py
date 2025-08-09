import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative, Order)
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.persistence import Trade
from freqtrade.configuration import Configuration
import logging
import sys
import os
import json
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

GLOBAL_ADDRESS = None
PRIVATE_HL = None
PRIVATE_EVM_WALLET = None

def write_log(message):
    """
    Writes a log message to the log file delta_neutral.log.
    
    Args:
        message (str): The log message to write.
    """
    from datetime import datetime

    here = Path(__file__).resolve().parent
    filename = here / "delta_neutral.log"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as file:
        file.write(f"[{timestamp}] {message}\n")

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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(path: str, db: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            db,
            f,
            ensure_ascii=False,
            indent=2,            # Pretty-print
            sort_keys=True       # Keep keys sorted
        )
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
    - harvested_datetime is ignored (left as-is if present)
    - Existing (timestamp, market_key) values are NOT overwritten
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
        val = round(value_transform(r["fundingRate"])*24.0*365.0*100.0,2)

        # 2) Ensure record exists for this timestamp
        if ts_key not in db or not isinstance(db[ts_key], dict):
            db[ts_key] = {}

        # 3) Only set if not already present (no overwrite)
        if market_key not in db[ts_key]:
            db[ts_key][market_key] = val
        # else: skip (do not overwrite)

    save_db(db_path, db)
    return db

def get_funding_history(db_path, coin, days_interval=14):
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    """
    Retrieves funding historical data and put it in the db db_path
    """
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    # Current time minus 7 days
    seven_days_ago = datetime.now() - timedelta(days=days_interval)
    # Convert to timestamp in milliseconds
    timestamp_ms = int(seven_days_ago.timestamp() * 1000)

    json_out = info.funding_history(coin,startTime=timestamp_ms)

    #print(json_out)

    funding_dict = {}

    for entry in json_out:
        # Convert ms timestamp to datetime in UTC
        dt = datetime.fromtimestamp(entry["time"] / 1000, tz=timezone.utc)
        
        # Format as ISO 8601 with milliseconds and timezone offset
        iso_str = dt.isoformat(timespec='milliseconds')
        
        # Store fundingRate as float (optional: keep string if you prefer)
        funding_dict[iso_str] = round(float(entry["fundingRate"])*24.0*365.0*100.0,2)

    # for k, v in list(funding_dict.items()):
    #     print(k, v)

    update_funding_db(db_path, json_out, f"{coin}/USDC:USDC")

    return True

def funding_negative_last_Xhours(
    pair: str,
    printt = False,
    nb_hours = 24,
    file_path = None,
    now_utc = None,
) -> bool:
    """
    Return True if *pair* has a negative average funding APR over the
    most‑recent nb_hours; False otherwise.

    Parameters
    ----------
    pair      : str
        The trading‑pair key in your JSON (e.g. "BTC/USDC:USDC").
    file_path : str | Path | None, optional
        Path to *funding_rates.json*.  Defaults to the same directory that
        contains this source file.
    now_utc   : datetime | None, optional
        Override the "current time" (mainly for tests).  Defaults to utcnow().

    Raises
    ------
    ValueError
        If the JSON file contains no entries for *pair* in the last day.

    Notes
    -----
    * Non‑numeric fields like "harvested_datetime" are ignored.
    * If fewer than nb_hours hourly records exist (e.g. fresh database),
      the function averages whatever data *is* present in that window.
    """
    # 1) resolve path
    file_path = Path(file_path) if file_path else Path(__file__).resolve().parent / "historical_funding_rates_DB.json"
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # 2) load JSON
    db: dict[str, dict[str, float | str]] = json.loads(file_path.read_text())

    # 3) time window
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    window_start = now_utc - timedelta(hours=nb_hours)

    total, count = 0.0, 0

    for ts_str, pairs in db.items():
        # parse the timestamp key
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue                        # malformed → skip

        if ts < window_start:               # outside last 24 h
            continue

        val = pairs.get(pair)
        if isinstance(val, (int, float)):   # ignore metadata strings
            total += float(val)
            count += 1

    if count == 0:
        raise ValueError(f"No data for '{pair}' in the last 24 hours.")

    avg = total / count
    if printt:
        mystr = " > 0" if avg>0 else " < 0"
        write_log(f"Average Funding APR on {pair} over last {nb_hours} hours: {avg:.1f}" + mystr)
    if avg < 0.0:
        write_log(f"It was negative, position should be cut or prevented to open.")
    return avg < 0.0

def REBALANCE_PERP_SPOT():
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    import math
    global GLOBAL_ADDRESS
    global PRIVATE_EVM_WALLET

    def _get_spot_account_available_USDC(info, address):
        spot_user_state = info.spot_user_state(address)
        for balance in spot_user_state["balances"]:
            if balance["coin"] == "USDC":
                return float(balance["total"])
        return 0

    def _get_perp_account_available_USDC(info, address):
        user_state = info.user_state(address)
        perp_user_state = account_balance_available = float(user_state['crossMarginSummary'].get('accountValue', 0))
        total_account_balance = float(user_state['marginSummary'].get('accountValue', 0))
        return account_balance_available, total_account_balance
    
    if GLOBAL_ADDRESS is None or PRIVATE_EVM_WALLET is None:
        config = Configuration.from_files(["user_data/config.json", "user_data/config-private.json"])
        ex = config.get("exchange", {})
        GLOBAL_ADDRESS = ex.get("walletAddress")
        PRIVATE_EVM_WALLET  = ex.get("privateKeyEthWallet")

    # Initialize exchange
    account: LocalAccount = eth_account.Account.from_key(PRIVATE_EVM_WALLET)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=GLOBAL_ADDRESS)

    # Fetch spot metadata
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    spot_usdc_available = _get_spot_account_available_USDC(info, GLOBAL_ADDRESS)
    perp_usdc_available, _ = _get_perp_account_available_USDC(info, GLOBAL_ADDRESS)

    amt_avilable_total = spot_usdc_available + perp_usdc_available
    target_each = amt_avilable_total / 2.0

    # deviation in percent
    def pct_diff(current, target):
        return abs(current - target) / target * 100

    spot_pct = pct_diff(spot_usdc_available, target_each)
    perp_pct = pct_diff(perp_usdc_available, target_each)

    write_log(f"Spot: {spot_usdc_available:.4f}, Perp: {perp_usdc_available:.4f}, target {target_each:.4f}")
    write_log(f"Deviations – spot: {spot_pct:.2f}%, perp: {perp_pct:.2f}%")

    amt_avilable_total = spot_usdc_available+perp_usdc_available

    # If deviation below threshold: skip
    if spot_pct < 0.5 and perp_pct < 0.5:
        write_log("Balances within 0.5% threshold — no rebalance needed")
        return False
    else:
        # decide which direction to transfer
        if spot_usdc_available > target_each:
            amt = spot_usdc_available - target_each
            amt = math.floor(amt*100.0)/100.0
            write_log(f"Considering transferring {amt:.2f} USDC from spot to perp")
            if amt_avilable_total < 22.0:
                write_log(f"But is very little USDC available ({round(amt_avilable_total,2)}), there may be a position open. No rebalacing.")
                return False
            transfer_result = exchange.usd_class_transfer(amt, True)
            write_log(f"Transfer result: {transfer_result}")
            return True
        else:
            amt = perp_usdc_available - target_each
            amt = math.floor(amt*100.0)/100.0
            write_log(f"Considering transferring {amt:.2f} USDC from perp to spot")
            if amt_avilable_total < 22.0:
                write_log(f"But is very little USDC available ({round(amt_avilable_total,2)}), there may be a position open. No rebalacing.")
                return False
            transfer_result = exchange.usd_class_transfer(amt, False)
            write_log(f"Transfer result: {transfer_result}")
            return True

def _get_spot_price(coin_name):
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    mid_price = info.all_mids()
    for key, value in mid_price.items():
        if key == coin_name:
            return float(value)
    return 0

def GET_NUMBER_SPOT_POSITION():
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    global GLOBAL_ADDRESS

    def count_spot_position(info, address):
        # Fetch spot metadata
        spot_user_state = info.spot_user_state(address)
        #print(spot_user_state)
        count = sum(
            1 for item in spot_user_state['balances']
            if item['coin'] != 'USDC' and float(item['entryNtl']) > 10
        )
        return count

    if GLOBAL_ADDRESS is None:
        config = Configuration.from_files(["user_data/config.json", "user_data/config-private.json"])
        ex = config.get("exchange", {})
        GLOBAL_ADDRESS = ex.get("walletAddress")

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    return count_spot_position(info, GLOBAL_ADDRESS)

def get_coin_info(coin):
    
    def count_decimal_digits(num):
        import math
        return math.log10(num)*-1
    
    match coin:
        case "BTC":
            size_tick = 0.00001
            price_tick = 1
            return {
                "HL_spot_pair": "UBTC/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "ETH":
            size_tick = 0.0001
            price_tick = 0.1
            return {
                "HL_spot_pair": "UETH/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "SOL":
            size_tick = 0.001
            price_tick = 0.01
            return {
                "HL_spot_pair": "USOL/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "HYPE":
            size_tick = 0.01
            price_tick = 0.001
            return {
                "HL_spot_pair": "HYPE/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "PUMP":
            size_tick = 1
            price_tick = 0.0000001
            return {
                "HL_spot_pair": "UPUMP/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "FARTCOIN":
            size_tick = 0.1
            price_tick = 0.0001
            return {
                "HL_spot_pair": "UFART/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case "PURR":
            size_tick = 1
            price_tick = 0.00001
            return {
                "HL_spot_pair": "PURR/USDC",
                "size_tick": size_tick,
                "size_decimal_digits": count_decimal_digits(size_tick),
                "price_tick": price_tick,
                "price_decimal_digits": count_decimal_digits(price_tick)
            }
        case _:
            raise ValueError(f"{coin} is unsupported coin")

def get_account_total_equity():
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    import math
    global GLOBAL_ADDRESS

    def get_spot_token_price_usdc(token_symbol: str) -> float:
        """
        Get the price of a spot token in USDC using Hyperliquid SDK
        
        Args:
            token_symbol: The token symbol (e.g., 'UETH', 'UFART', 'BTC', 'ETH', etc.)
                        Can also handle numerical IDs like '@1', '@100', etc.
        
        Returns:
            Price in USDC as float
        """

        # Initialize the Info client
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        if token_symbol=='UFART':
            token_symbol='FARTCOIN'
        elif token_symbol=='UBTC':
            token_symbol='BTC'
        elif token_symbol=='UETH':
            token_symbol='ETH'
        elif token_symbol=='USOL':
            token_symbol='SOL'
        elif token_symbol=='PURR':
            token_symbol='PURR'
        elif token_symbol=='HYPE':
            token_symbol='HYPE'
        elif token_symbol=='UPUMP':
            token_symbol='PUMP'
        
        # Get all mids (market prices) for all actively traded coins
        all_mids = info.all_mids()
        
        # Find the price for the specific token
        if token_symbol in all_mids:
            price = float(all_mids[token_symbol])
            return price
        else:
            return 0.0

    def sum_entry_ntl_and_usdc_total(data):
        from decimal import Decimal, InvalidOperation

        total = 0.0

        balances = data.get("balances", [])
        for b in balances:
            try:
                # Sum all entryNtl values
                # print(b.get("coin", "0"))
                # print(b.get("total", "0"))
                if b.get("coin", "0")!='USDC':
                    total += float(b.get("total", "0"))*float(get_spot_token_price_usdc(b.get("coin", "0")))
                else:
                    total += float(b.get("total", "0"))
            except (InvalidOperation, TypeError):
                pass

        return float(total)

    if GLOBAL_ADDRESS is None:
        config = Configuration.from_files(["user_data/config.json", "user_data/config-private.json"])
        ex = config.get("exchange", {}) 
        GLOBAL_ADDRESS = ex.get("walletAddress")

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    perp_user_state = info.user_state(GLOBAL_ADDRESS)
    spot_user_state = info.spot_user_state(GLOBAL_ADDRESS)

    perp_account = float(perp_user_state['marginSummary']['accountValue'])
    spot_account = sum_entry_ntl_and_usdc_total(spot_user_state)
    return perp_account+spot_account

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

def HL_buy_spot_market(coin, spot_size):
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    global GLOBAL_ADDRESS
    global PRIVATE_HL

    def get_spot_position_size(info, address, wanted_coin_name):
        if wanted_coin_name=='PUMP':
            wanted_coin_name = 'UPUMP'
        if wanted_coin_name=='FARTCOIN':
            wanted_coin_name = 'UFART'
        spot_user_state = info.spot_user_state(address)
        for balance in spot_user_state.get("balances", []):
            if float(balance["total"]) > 0:
                coin_name = balance["coin"]
                if wanted_coin_name in coin_name:
                    return float(balance["total"])
        return 0.0

    def floor_to_n_digits(value, n):
        import math
        factor = 10.0 ** n
        return math.floor(value * factor) / factor
    
    def round_to_n_digits(value, n):
        factor = 10.0 ** n
        return round(value * factor) / factor
    
    coin_info = get_coin_info(coin)
    HL_spot_pair = coin_info['HL_spot_pair']
    size_decimal_digits = coin_info['size_decimal_digits']
    # price_decimal_digits = coin_info['price_decimal_digits']

    if GLOBAL_ADDRESS is None or PRIVATE_HL is None:
        config = Configuration.from_files(["user_data/config.json", "user_data/config-private.json"])
        ex = config.get("exchange", {})
        GLOBAL_ADDRESS = ex.get("walletAddress")
        PRIVATE_HL    = ex.get("privateKey")
        
    account: LocalAccount = eth_account.Account.from_key(PRIVATE_HL)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=GLOBAL_ADDRESS)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    #last_price = _get_spot_price(coin)
    rounded_spot_buy_size = floor_to_n_digits(spot_size, size_decimal_digits)
    write_log(rounded_spot_buy_size)
    spot_size_DUST = get_spot_position_size(info, GLOBAL_ADDRESS, coin) # if there is already a dust
    write_log(spot_size_DUST)
    rounded_spot_buy_size = round_to_n_digits(rounded_spot_buy_size-spot_size_DUST, size_decimal_digits) 
    write_log(rounded_spot_buy_size)
    rounded_spot_buy_size = round_to_n_digits(rounded_spot_buy_size*(1.0+0.065/100.0), size_decimal_digits) # 0.065 depends on your spot taker fees 
    write_log(rounded_spot_buy_size)
    #limit_buy_price = floor_to_n_digits(last_price*1.05, price_decimal_digits)
    # True -> buy
    spot_order_result = exchange.market_open(HL_spot_pair, True, rounded_spot_buy_size, None, 0.10)
    if spot_order_result["status"] == "ok":
        for status in spot_order_result["response"]["data"]["statuses"]:
            try:
                filled = status["filled"]
                write_log(f'Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                return True
            except Exception as e:
                write_log(f'Error: {e}')
                return False
    else:
        write_log(spot_order_result)
        return False

def HL_sell_spot_market(coin):
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    global GLOBAL_ADDRESS
    global PRIVATE_HL
    
    def get_spot_position_size(info, address, wanted_coin_name):
        if wanted_coin_name=='PUMP':
            wanted_coin_name = 'UPUMP'
        if wanted_coin_name=='FARTCOIN':
            wanted_coin_name = 'UFART'
        spot_user_state = info.spot_user_state(address)
        for balance in spot_user_state.get("balances", []):
            if float(balance["total"]) > 0:
                coin_name = balance["coin"]
                if wanted_coin_name in coin_name:
                    return float(balance["total"])
        return 0.0

    def round_to_n_digits(value, n):
        factor = 10 ** n
        return round(value * factor) / factor
    
    coin_info = get_coin_info(coin)
    HL_spot_pair = coin_info['HL_spot_pair']
    size_decimal_digits = coin_info['size_decimal_digits']
    price_decimal_digits = coin_info['price_decimal_digits']

    if GLOBAL_ADDRESS is None or PRIVATE_HL is None:
        config = Configuration.from_files(["user_data/config.json", "user_data/config-private.json"])
        ex = config.get("exchange", {})
        GLOBAL_ADDRESS = ex.get("walletAddress")
        PRIVATE_HL    = ex.get("privateKey")

    account: LocalAccount = eth_account.Account.from_key(PRIVATE_HL)
    exchange = Exchange(account, constants.MAINNET_API_URL, account_address=GLOBAL_ADDRESS)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    spot_size = get_spot_position_size(info, GLOBAL_ADDRESS, coin)
    last_price = _get_spot_price(coin)

    rounded_spot_sell_size = round_to_n_digits(spot_size, size_decimal_digits)
    limit_sell_price = round_to_n_digits(last_price*0.95, price_decimal_digits)
    # False -> sell
    spot_order_result = exchange.order(HL_spot_pair, False, rounded_spot_sell_size, limit_sell_price, {"limit": {"tif": "Ioc"}})
    write_log(spot_order_result)

    if spot_order_result["status"] == "ok":
        for status in spot_order_result["response"]["data"]["statuses"]:
            try:
                filled = status["filled"]
                write_log(f'Order #{filled["oid"]} filled {filled["totalSz"]} @{filled["avgPx"]}')
                return True
            except Exception as e:
                write_log(f'Error: {status["error"]}')
                return False
    else:
        return False

def record_hourly_funding_by_pair(
    pair: str,
    funding_val: float,
    tz: timezone | None = None,
) -> None:
    """
    Persist `funding_val` for a given trading `pair` with one‑hour resolution.

    ▶ Call this once for every pair whose funding you just calculated.
      It will *overwrite* the value for the current hour if it already exists,
      or append a fresh entry when the hour rolls over.

    Parameters
    ----------
    pair        : str
        Trading pair symbol (e.g. "BTC/USDT"); becomes a key inside each hour.
    funding_val : float
        Funding APR you just computed for that pair.
    file_path   : str | Path, optional
        Location of the JSON “database”. Defaults to *funding_rates.json*.
    tz          : datetime.timezone | None, optional
        Clock to use when stamping the hour. If None, the system local tz is used.
    """
    here = Path(__file__).resolve().parent
    file_path = here / "historical_funding_rates_DB.json"

    # 1) Current hour (truncate to 00:00 in minutes/seconds)
    now = datetime.now(tz=tz)
    nowplus1 = datetime.now(tz=tz) + timedelta(hours=1) # because fundigns values are actually estimated values for the next hour change (the one just before hour change is closest to real one, probably the same)
    hour_key = nowplus1.replace(minute=0, second=0, microsecond=0).isoformat(timespec="milliseconds")

    harvested_hour_key = now.isoformat(timespec="milliseconds")

    # 2) Load or create the DB structure:  {hour: {pair: funding_val}}
    if file_path.exists():
        try:
            db: dict[str, dict[str, float]] = json.loads(file_path.read_text())
        except (json.JSONDecodeError, OSError):
            db = {}              # corrupt / missing → start fresh
    else:
        db = {}

    # 3) Ensure we have an inner dict for this hour, then upsert the pair
    db.setdefault(hour_key, {})['harvested_datetime'] = harvested_hour_key
    db.setdefault(hour_key, {})[pair] = round(float(funding_val), 6)

    # 4) Atomic save (write‑then‑rename)
    tmp = file_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(db, indent=2))
    tmp.replace(file_path)

def average_funding_last_7_days(
    file_path: Union[str, Path, None] = None,
    now_utc: datetime | None = None,
) -> dict[str, float]:
    """
    Compute the average funding APR for each trading pair over the last 7 days
    (or for the full history if < 7 days of data).

    Ignores non‑numeric metadata fields such as 'harvested_datetime'.

    Returns
    -------
    {pair: average_apr_float}
    """
    # ── locate JSON beside this .py file unless overridden ─────────────
    file_path = Path(file_path) if file_path else Path(__file__).resolve().parent / "historical_funding_rates_DB.json"
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    db: dict[str, dict[str, float | str]] = json.loads(file_path.read_text())

    # ── time window ────────────────────────────────────────────────────
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    window_start = now_utc - timedelta(days=7)

    sums, counts = {}, {}

    for ts_str, pairs in db.items():
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue                       # malformed timestamp key

        if ts < window_start:              # too old → skip
            continue

        for pair, val in pairs.items():
            # skip metadata such as 'harvested_datetime'
            if not isinstance(val, (int, float)):
                continue

            sums[pair]   = sums.get(pair, 0.0) + float(val)
            counts[pair] = counts.get(pair, 0) + 1

    return {pair: round(sums[pair] / counts[pair], 6)
            for pair in sums}

# ────────────────────────────────────────────────────────────────────────────────
# Convenience wrapper that prints nicely
def print_average_funding_last_7_days(file_path = None):
    averages = average_funding_last_7_days(file_path=file_path)
    if not averages:
        write_log("No data within the last 7 days.")
        return
    write_log("Average funding APR (last 7 days):")
    for pair, avg in sorted(averages.items()):
        write_log(f"  {pair:<10}  {avg:.1f} %")

def avg_funding_last_hours(
    pair: str,
    printt = False,
    nb_hours = 24,
    file_path = None,
    now_utc = None,
) -> bool:
    
    # 1) resolve path
    file_path = Path(file_path) if file_path else Path(__file__).resolve().parent / "historical_funding_rates_DB.json"
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    # 2) load JSON
    db: dict[str, dict[str, float | str]] = json.loads(file_path.read_text())

    # 3) time window
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    window_start = now_utc - timedelta(hours=nb_hours)

    total, count = 0.0, 0

    for ts_str, pairs in db.items():
        # parse the timestamp key
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue                        # malformed → skip

        if ts < window_start:               # outside last 24 h
            continue

        val = pairs.get(pair)
        if isinstance(val, (int, float)):   # ignore metadata strings
            total += float(val)
            count += 1

    if count == 0:
        raise ValueError(f"No data for '{pair}' in the last {nb_hours} hours.")

    avg = total / count
    if printt:
        write_log(f"Average Funding APR on {pair} over last {nb_hours} hours: {avg:.1f}")
    return avg

# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────

class DELTA_NEUTRAL(IStrategy):
    minimal_roi = {
        "0": 5000.0  # Effectively disables ROI
    }
    stoploss = -0.90
    timeframe = '1m'
    startup_candle_count: int = 0
    can_short: bool = True
    process_only_new_candles: bool = False

    # Tunable parameters
    MINIMUM_FUNDING_APR_pc = 10
    MINIMUM_VOLUME_usdc = 2_500_000
    MINIMUM_TIME_TO_KEEP_POSITION_hour = 24*30 # minimum time for a delta neutral position to be kept openned to have a good chance it compensates the opening+closing fees of the spot and futures trades
                                               # here 30 days
    MAX_POSITIONS = 2  # Maximum number of open positions

    # State variables (do not touch)
    has_looped_once = False
    nb_loop = 1
    FUNDINGS = {}
    QUOTE_VOLUMES = {}
    BEST_PAIRS = []  # list of best pairs
    CURRENT_POSITION_PAIRS = []  # list
    order_just_filled = False
    rebalancing_done = True

    # DEBUG PARAMETERS
    FORCE_EXIT = False # for debug only, forces exit of delta neutral position (To close an open position (debug, tests), put FORCE_EXIT to `True` and restart [commands `docker compose down` then `docker compose up`])

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    def select_BEST_PAIRS(self, quote_volumes: dict, fundings: dict, max_pairs: int = 2) -> list[str]:
        """
        Return up to k pairs sorted by funding APR desc, then quote volume desc.
        Assumes both APR and volume thresholds were already applied upstream.
        """
        if not quote_volumes or not fundings:
            return []
        
        candidates = [s for s in fundings.keys() if s in quote_volumes]

        # Sort: primary by funding APR desc, secondary by volume desc
        ranked = sorted(
            candidates,
            key=lambda s: (fundings.get(s, float("-inf")), quote_volumes.get(s, 0.0)),
            reverse=True,
        )

        return ranked[:max_pairs]


    def PAIR_SHOULD_BE_REPLACED(self, current_pair: str):
        """Check if there's a better opportunity than the current pair"""

        if current_pair not in self.CURRENT_POSITION_PAIRS:
            write_log(f"The Pair {current_pair}, checked  to be replaced (or not), should be in the CURRENT_POSITION_PAIRS list. Something went wrong. Aborting.")
            sys.exit()

        if current_pair not in self.FUNDINGS: # Current pair is no longer viable because of funding APR too low, negative, or too low volume
            return True
            
        if not self.BEST_PAIRS or len(self.BEST_PAIRS)==0: # there are no best pairs (it is empty)
            return False
            
        # Check if current pair is still in top pairs
        if current_pair in self.BEST_PAIRS:
            return False
            
        # Check if any of the best pairs has significantly better funding
        current_funding = self.FUNDINGS.get(current_pair, 0)
        best_funding = max(self.FUNDINGS.get(pair, 0) for pair in self.BEST_PAIRS)
        
        return best_funding > current_funding

    def bot_start(self, **kwargs) -> None:
        """
        Called only once after bot instantiation.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        write_log("NEW START")
        self.has_looped_once = False
        if self.config["runmode"].value in ('live', 'dry_run'):
            
            # retrive historical fundings for the last 30 days and update the funding database (in case there would be missing data)
            write_log("Updating fundings database historical_funding_rates_DB.json with historical data from API.")
            here = Path(__file__).resolve().parent
            db_path = here / "historical_funding_rates_DB.json"
            get_funding_history(db_path,"BTC",8) # 8 days
            get_funding_history(db_path,"ETH",8)
            get_funding_history(db_path,"SOL",8)
            get_funding_history(db_path,"HYPE",8)
            get_funding_history(db_path,"PUMP",8)
            get_funding_history(db_path,"FARTCOIN",8)
            get_funding_history(db_path,"PURR",8)
            write_log("Done updating fundings database.")

            open_perp_count = Trade.get_open_trade_count() # freqtrade only manages perp positions here, spot position management is done with custom code.
            write_log(f'Number of open perp positions: [{open_perp_count}]')

            if self.config["runmode"].value not in ('dry_run'):
                open_spot_count = GET_NUMBER_SPOT_POSITION()
                write_log(f"Number of open spot positions: [{open_spot_count}]")
                if open_perp_count!=open_spot_count:
                    write_log(f'WARNING: The number of spot and perp positions should be the same ! Check if everyting is fine.')
                    sys.exit()
            
            if open_perp_count!=self.MAX_POSITIONS:
                if self.config["runmode"].value in ('live'):
                    REBALANCE_PERP_SPOT()


    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop). For each loop, it will run populate_indicators on all pairs.
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """

        if self.config["runmode"].value in ('live', 'dry_run'):
            write_log(f'Loop #{self.nb_loop}')

            log_equity(get_account_total_equity())

            if self.nb_loop>=2:
                self.has_looped_once = True
            self.nb_loop = self.nb_loop+1

            open_count = Trade.get_open_trade_count() # perp positions as freqtrade only manages perp positions here, spot position management is done with custom code.

            nb_spot_position = GET_NUMBER_SPOT_POSITION()

            if self.config["runmode"].value not in ('dry_run'):
                if open_count != nb_spot_position:
                        write_log(f'WARNING: The number of spot and perp positions should be the same ! Check if everyting is fine.')
                        sys.exit()

             # rebalance to have 50/50 perp/spot USDC repartition  if several conditions are met
            if self.order_just_filled:
                self.order_just_filled = False
                if open_count!=self.MAX_POSITIONS and open_count==nb_spot_position:
                    try: 
                        if self.config["runmode"].value in ('live') and not self.rebalancing_done:
                            REBALANCE_PERP_SPOT()
                            self.rebalancing_done = True
                    except Exception as e: # abort if failed to rebalance
                        write_log(f'There was an error while rebalancing perp and spot accounts. ABORTING.')
                        write_log(str(e))
                        sys.exit()

            if open_count == 0:
                self.CURRENT_POSITION_PAIRS = []
            else:
                # Retrieve a list of open positions
                open_trades = Trade.get_trades_proxy(is_open=True)
                open_pairs = [t.pair for t in open_trades]
                leverages = [t.leverage for t in open_trades]

                if len(open_pairs) > self.MAX_POSITIONS: # abort if there are more positions than allowed
                    write_log(f'Should not have more than {self.MAX_POSITIONS} positions opened! ABORTING.')
                    sys.exit()

                for lev in leverages:
                    if lev!=1:
                        write_log(f'Leverage should always be 1. ABORTING.')
                        sys.exit()

                self.CURRENT_POSITION_PAIRS = open_pairs
                
                write_log(f'Open Positions: {open_pairs};    Leverages: {leverages};    Fundings 12h average: {[round(avg_funding_last_hours(op, False, 12),2) for op in open_pairs]};    Fundings instant: {[round(avg_funding_last_hours(op, False, 1),2) for op in open_pairs]}')

            self.BEST_PAIRS = []
            if self.has_looped_once:
                self.BEST_PAIRS = self.select_BEST_PAIRS(self.QUOTE_VOLUMES, self.FUNDINGS, self.MAX_POSITIONS)
                # log some information
                if self.BEST_PAIRS:
                    if self.nb_loop == 3 or self.nb_loop%10==0:
                        write_log(f"Current Funding rates (APR %): {self.FUNDINGS}")
                        write_log(f"Current QUOTE_VOLUMES (USDC): {self.QUOTE_VOLUMES}")
                        write_log(f"List of {len(self.BEST_PAIRS)} / {self.config["max_open_trades"]}max best pair(s): ")
                        for i, pair in enumerate(self.BEST_PAIRS, 1):
                            write_log(f"  BEST_PAIR_{i}: {pair} ({self.FUNDINGS[pair]:.1f}% ; {self.QUOTE_VOLUMES[pair]:.1f})")
                        write_log(f"Number of open positions: [{open_count}]")
                        print_average_funding_last_7_days()

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp.runmode.value in ('live', 'dry_run'):

            if self.FORCE_EXIT:
                return df

            df['entry_signal'] = 0
            df['exit_signal'] = 0

            cPAIR = metadata['pair']
            open_count = Trade.get_open_trade_count()

            # retrieve funding rate APR, volume and add to dictionaries and json database
            ticker = self.dp.ticker(cPAIR)
            funding_val = round(float(ticker['info']['funding'])*24.0*365.0*100.0, 3)
            vol = float(ticker['quoteVolume'])

            record_hourly_funding_by_pair(cPAIR, funding_val, tz=timezone.utc)

            hours_avg = 3
            funding_val_avg = avg_funding_last_hours(cPAIR, False, hours_avg)

            if not self.has_looped_once:
                write_log(f"Initial Funding Rate {cPAIR} : {funding_val:.1f}% APR")
                write_log(f"Last {hours_avg}h average on  {cPAIR} : {funding_val_avg:.1f}% APR")

            # append "global" dictionaries containing the Funding APR% and Volume values for each pair
            _ = self.FUNDINGS.pop(cPAIR, None)
            _ = self.QUOTE_VOLUMES.pop(cPAIR, None)
            if funding_val_avg > self.MINIMUM_FUNDING_APR_pc and vol > self.MINIMUM_VOLUME_usdc:
                self.FUNDINGS[cPAIR] = funding_val_avg
                self.QUOTE_VOLUMES[cPAIR] = vol
            else:
                write_log(f"{cPAIR} is rejected because of low Funding APR [{funding_val_avg:.1f}], or low volume [{vol:.1f}].")

            if not self.has_looped_once:
                df['entry_signal'] = 0
            else:
                self.BEST_PAIRS = self.select_BEST_PAIRS(self.QUOTE_VOLUMES, self.FUNDINGS, self.MAX_POSITIONS)
                # Signal to open short if:
                # 1. We have fewer than max positions open
                # 2. bot has already done one loop, i.e. we are in loop #2 or more (i.e. we have already grathered current fundings for all pairs)
                # 3. Current pair is in best pairs
                # 4. Current pair is not already open
                if (open_count < self.MAX_POSITIONS and 
                    self.has_looped_once and 
                    cPAIR in self.BEST_PAIRS and 
                    cPAIR not in self.CURRENT_POSITION_PAIRS):
                    df['entry_signal'] = -1  # means open short
                else:
                    df['entry_signal'] = 0
        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp.runmode.value in ('live', 'dry_run'):

            if self.FORCE_EXIT: # used for debug only
                dataframe.loc[:, 'enter_short'] = 0
                return dataframe
            
            current_pair = metadata['pair']
            open_count = Trade.get_open_trade_count()
            if (current_pair in self.BEST_PAIRS and 
                open_count<self.MAX_POSITIONS and
                current_pair not in self.CURRENT_POSITION_PAIRS):
                dataframe.loc[dataframe['entry_signal'] == -1, 'enter_short'] = 1

            return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if self.dp.runmode.value in ('live', 'dry_run'):

            if self.FORCE_EXIT: # used for debug only
                dataframe.loc[:, 'exit_short'] = 1
                return dataframe

            return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        is_OK = False
        if self.dp.runmode.value in ('dry_run'): # skip opening spot position if dry-run
            if pair not in self.CURRENT_POSITION_PAIRS:
                self.CURRENT_POSITION_PAIRS.append(pair)
            return True
        elif self.dp.runmode.value in ('live'):
            coin_name = pair.replace("/USDC:USDC","")
            try:
                # attempt Hyperliquid spot call
                write_log(f"Freqtrade size: {amount}")
                is_OK = HL_buy_spot_market(coin_name, amount)
            except ModuleNotFoundError as e:
                write_log(f"Hyperliquid module missing: {str(e)}. Skipping entry.")
                return False
            except Exception as e:
                write_log(f"Error calling HL_buy_spot_market: {str(e)}. Skipping entry.")
                return False
            if not is_OK:
                write_log('Error placing spot buy order; not opening short')
                return False
            if pair not in self.CURRENT_POSITION_PAIRS:
                self.CURRENT_POSITION_PAIRS.append(pair)
            
        return is_OK

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        is_OK = False
        if self.dp.runmode.value in ('dry_run'): # skip if dry-run
            if pair in self.CURRENT_POSITION_PAIRS:
                self.CURRENT_POSITION_PAIRS.remove(pair)
            return True
        elif self.dp.runmode.value in ('live'):
            coin_name = pair.replace("/USDC:USDC","")
            try:
                # attempt Hyperliquid spot call
                is_OK = HL_sell_spot_market(coin_name)
            except ModuleNotFoundError as e:
                write_log(f"Hyperliquid module missing: {str(e)}. Skipping exit.")
                return False
            except Exception as e:
                write_log(f"Error calling HL_sell_spot_market: {str(e)}. Skipping exit.")
                return False
            
            if pair in self.CURRENT_POSITION_PAIRS:
                self.CURRENT_POSITION_PAIRS.remove(pair)

        return is_OK

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        # Above 50% profit or below 50% loss, exit the position
        if current_profit > 0.50:
            return "upper_limit_rebalance"
        if current_profit < -0.50:
            return "lower_limit_rebalance"
        
        write_log(f"Hours since open on {pair}: {(current_time - trade.open_date_utc).total_seconds()/3600:.2f} (limit to consider changing pair: {self.MINIMUM_TIME_TO_KEEP_POSITION_hour})")
        
        # Check if the minimum time for position has passed, and if there is a better opportunity
        min_holding_time_passed = (current_time - trade.open_date_utc).total_seconds()/3600 >= self.MINIMUM_TIME_TO_KEEP_POSITION_hour

        if min_holding_time_passed and self.PAIR_SHOULD_BE_REPLACED(pair):
            return "timeout_and_better"

        is_funding_negative_last_Xhours = funding_negative_last_Xhours(pair=pair, printt=True, nb_hours = 24)
        if min_holding_time_passed and is_funding_negative_last_Xhours:
            return "timeout_and_negative_fundings_avg24h" 

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs) -> float:
        # self.wallets.get_total_stake_amount() gives the "available_capital" in the config.json
        open_count = Trade.get_open_trade_count()

        dust_USDC = 0.5

        returned_val = 0.0

        if self.config["max_open_trades"]==3:
            if open_count==0:
                returned_val = max_stake/3.0-dust_USDC
            elif open_count==1:
                returned_val = max_stake/2.0-dust_USDC
            elif open_count==2:
                returned_val = max_stake-dust_USDC
        elif self.config["max_open_trades"]==2:
            if open_count==0:
                returned_val = max_stake/2.0-dust_USDC
            elif open_count==1:
                returned_val = max_stake-dust_USDC
        elif self.config["max_open_trades"]==1:
            if open_count==0:
                returned_val = max_stake-dust_USDC
        else:
            write_log("ERROR: max_open_trades larger than 3 is not implemented. ABORTING.")
            sys.exit()

        if leverage!=1:
            write_log("ERROR: Leverage must be 1. Something went wrong. ABORTING.")
            sys.exit()
        #write_log(f"self.wallets.get_total_stake_amount() : {self.wallets.get_total_stake_amount()}")
        #write_log(f"max_stake : {max_stake}")
        #self.config["max_open_trades"]
        #self.config["stake_amount"]

        write_log(f"Using stake amount for {pair} : {returned_val}")
        return returned_val

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Called right after an order fills. 
        Will be called for all order types (entry, exit, stoploss, position adjustment).
        :param pair: Pair for trade
        :param trade: trade object.
        :param order: Order object.
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        # Skip if dry-run
        if self.dp.runmode.value in ('dry_run'):
            return None
        # Trigger a spot/perp rebalance to 50/50 after any order fill
        time.sleep(3.0) # wait for 3 seconds just in case to make sure both spot and perp orders were filled
        self.rebalancing_done = False
        open_count = Trade.get_open_trade_count() # (number of perp positions)
        nb_spot_position = GET_NUMBER_SPOT_POSITION()
        if open_count==nb_spot_position and open_count!=self.MAX_POSITIONS:
            REBALANCE_PERP_SPOT()
            self.rebalancing_done = True
        if open_count!=nb_spot_position:
            write_log("WARNING: in order_filled, the number of spot and perp positions should be equal. Will also try later to rebalance.")
        self.order_just_filled = True
        return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        lev = 1
        write_log(f"Using leverage: {lev}. Should not be changed.")
        return lev
