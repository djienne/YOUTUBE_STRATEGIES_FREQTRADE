from hyperliquid.info import Info
from hyperliquid.utils import constants
from datetime import datetime, timedelta, timezone
import time
import os
import json
from typing import List, Dict, Callable, Optional

# ----------------------------
# Helpers & persistence
# ----------------------------

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
            sort_keys=True       # Sort keys at all levels (timestamps sorted)
        )
    os.replace(tmp, path)

# ----------------------------
# DB updater
# ----------------------------

def update_funding_db(
    db_path: str,
    records: List[Dict],
    market_key: str,                              # e.g. "BTC/USDC:USDC"
    value_transform: Callable[[str], float] = lambda s: float(s),  # turn fundingRate string into number
    align_to_hour: bool = True,
    debug: bool = False,
) -> dict:
    """
    Update the DB with records like:
      {"coin": "BTC", "fundingRate": "0.0000125", "time": 1753862400004}

    - market_key is where the value is stored, e.g. "BTC/USDC:USDC"
    - harvested_datetime is ignored (left as-is if present)
    - Existing (timestamp, market_key) values are NOT overwritten
    - Returns the updated DB (also saved to db_path)
    """
    db = load_db(db_path)

    for r in records:
        # Safety: skip malformed items
        if "time" not in r or "fundingRate" not in r:
            if debug:
                print(f"[SKIP] malformed record: {r}")
            continue

        # 1) Convert ms epoch -> UTC datetime
        dt = datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc)
        if align_to_hour:
            dt = _to_hour_start(dt)

        ts_key = _iso_ms(dt)  # e.g. 2025-08-06T04:00:00.000+00:00

        # Example: annualized percent (24 * 365 * 100); round to 2 decimals
        val = round(value_transform(r["fundingRate"]) * 24.0 * 365.0 * 100.0, 2)

        # 2) Ensure record exists for this timestamp
        if ts_key not in db or not isinstance(db[ts_key], dict):
            db[ts_key] = {}

        # 3) Only set if not already present (no overwrite)
        if market_key not in db[ts_key]:
            db[ts_key][market_key] = val
            if debug:
                print(f"[NEW]  {ts_key} -> {market_key} = {val}")
        else:
            if debug:
                print(f"[SKIP] {ts_key} -> {market_key} already set to {db[ts_key][market_key]}")

    save_db(db_path, db)
    return db

# ----------------------------
# Rolling-window fetch
# ----------------------------

def _filter_window(records: List[Dict], start_ms: int, end_ms: int) -> List[Dict]:
    """Keep entries whose time is in [start_ms, end_ms)."""
    out = []
    for r in records:
        t = r.get("time")
        if isinstance(t, (int, float)) and start_ms <= int(t) < end_ms:
            out.append(r)
    return out

def get_funding_history_rolling(
    db_path: str,
    coin: str,
    total_days: int = 365,
    window_days: int = 5,
    align_to_hour: bool = True,
    sleep_seconds: float = 0.0,
    debug: bool = False,
) -> bool:
    """
    Fetch historical funding using a rolling window to avoid holes for long ranges.

    - total_days: how far back to go (e.g., 365)
    - window_days: window size per API call (e.g., 5)
    - align_to_hour: bucket timestamps to the start of the hour in the DB
    - sleep_seconds: optional small delay between window calls
    - debug: print progress and no-overwrite decisions

    The DB is updated with key = ISO ms timestamp, and the value stored under market_key
    like "BTC/USDC:USDC" using annualized percent scaling (by default).
    """
    if total_days <= 0:
        raise ValueError("total_days must be > 0")
    if window_days <= 0:
        raise ValueError("window_days must be > 0")

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=total_days)

    market_key = f"{coin}/USDC:USDC"

    # Walk windows: [win_start, win_end) ... up to now
    win_start = start_utc
    while win_start < now_utc:
        win_end = min(win_start + timedelta(days=window_days), now_utc)

        start_ms = int(win_start.timestamp() * 1000)
        end_ms = int(win_end.timestamp() * 1000)

        if debug:
            print(f"[FETCH] {coin} {win_start.isoformat()} -> {win_end.isoformat()} (UTC)")

        # API: request from this window's start onward; we locally filter to end_ms
        # NOTE: some APIs optionally support `endTime`, but we stick to startTime for compatibility.
        try:
            raw = info.funding_history(coin, startTime=start_ms)
        except Exception as e:
            print(f"[ERROR] funding_history({coin}, startTime={start_ms}) failed: {e}")
            # continue to next window (or break) depending on preference
            win_start = win_end
            if sleep_seconds:
                time.sleep(sleep_seconds)
            continue

        # Filter records strictly to [start_ms, end_ms)
        window_recs = _filter_window(raw, start_ms, end_ms)

        # Preview (optional)
        if debug:
            for r in window_recs[:5]:
                dt = datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc)
                if align_to_hour:
                    dt = _to_hour_start(dt)
                print(" ", _iso_ms(dt), r.get("fundingRate"))

        # Update DB (no overwrite)
        update_funding_db(
            db_path=db_path,
            records=window_recs,
            market_key=market_key,
            value_transform=lambda s: float(s),
            align_to_hour=align_to_hour,
            debug=debug,
        )

        # Move to next window
        win_start = win_end
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return True

# ----------------------------
# Example batch usage
# ----------------------------

if __name__ == "__main__":
    db_path = "funding_db_for_backtest.json"

    coins = ["BTC", "ETH", "SOL", "HYPE", "PUMP", "FARTCOIN", "PURR"]

    # Fetch last 365 days using 5-day rolling window
    print(f"Updating {db_path}...")
    for c in coins:
        print(f"  Doing {c}...")
        get_funding_history_rolling(
            db_path=db_path,
            coin=c,
            total_days=365, # from today
            window_days=5,
            align_to_hour=True,
            sleep_seconds=0.03,   # add a small delay if you hit rate limits
            debug=False          # set True to see detailed logs and no-overwrite skips
        )
        print(f"  Done.")
    
    print(f"Done Updating {db_path}.")
