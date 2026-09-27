from hyperliquid.info import Info
from hyperliquid.utils import constants
from datetime import datetime, timedelta, timezone
import time
import os
import json
from typing import List, Dict, Callable, Optional, Any

from funding_utils import load_backtest_config, parse_iso_utc, parse_iso_utc_optional

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

def _latest_market_timestamp(db: dict, market_key: str) -> Optional[datetime]:
    latest = None
    for ts, payload in db.items():
        if not isinstance(payload, dict) or market_key not in payload:
            continue
        try:
            dt = parse_iso_utc(ts)
        except Exception:
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest

# ----------------------------
# Rate limiting & backoff
# ----------------------------

MIN_REQUEST_INTERVAL = 0.1
BACKOFF_BASE_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 10.0


class RateLimiter:
    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._last_call = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


def _is_rate_limit_error(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if status == 429:
        return True
    args = getattr(err, "args", ())
    if args:
        first = args[0]
        if isinstance(first, int) and first == 429:
            return True
        if isinstance(first, tuple) and first and first[0] == 429:
            return True
    return False


def _call_with_backoff(
    fn: Callable[[], Any],
    limiter: RateLimiter,
    backoff_base: float,
    backoff_max: float,
    debug: bool,
    call_label: str,
) -> Any:
    attempt = 0
    while True:
        limiter.wait()
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            attempt += 1
            sleep_for = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
            if debug:
                print(f"[RATE_LIMIT] {call_label} hit 429, sleeping {sleep_for:.2f}s (attempt {attempt})")
            time.sleep(sleep_for)

# ----------------------------
# DB updater
# ----------------------------

def update_funding_db(
    db_path: str,
    records: List[Dict],
    market_key: str,                              # e.g. "BTC/USDC:USDC"
    value_transform: Callable[[str], float] = lambda s: float(s),  # turn fundingRate string into number
    align_to_hour: bool = True,
    allow_overwrite: bool = False,
    debug: bool = False,
) -> dict:
    """
    Update the DB with records like:
      {"coin": "BTC", "fundingRate": "0.0000125", "time": 1753862400004}

    - market_key is where the value is stored, e.g. "BTC/USDC:USDC"
    - harvested_datetime is ignored (left as-is if present)
    - Existing (timestamp, market_key) values are NOT overwritten unless allow_overwrite is True
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
        existed = market_key in db[ts_key]
        if not existed or allow_overwrite:
            db[ts_key][market_key] = val
            if debug:
                action = "UPD" if existed and allow_overwrite else "NEW"
                print(f"[{action}]  {ts_key} -> {market_key} = {val}")
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
    sleep_seconds: float = MIN_REQUEST_INTERVAL,
    backoff_base: float = BACKOFF_BASE_SECONDS,
    backoff_max: float = BACKOFF_MAX_SECONDS,
    overlap_days: float = 2.0,
    start_time_utc: Optional[str] = None,
    end_time_utc: Optional[str] = None,
    info: Optional[Info] = None,
    limiter: Optional[RateLimiter] = None,
    debug: bool = False,
) -> bool:
    """
    Fetch historical funding using a rolling window to avoid holes for long ranges.

    - total_days: how far back to go (e.g., 365)
    - window_days: window size per API call (e.g., 5)
    - align_to_hour: bucket timestamps to the start of the hour in the DB
    - sleep_seconds: minimum seconds between API calls (rate limiter, enforced at >= 0.1s)
    - backoff_base: starting delay for exponential backoff on 429s
    - backoff_max: maximum delay for exponential backoff on 429s
    - overlap_days: always re-download this many days to avoid gaps and refresh recent data
    - start_time_utc: optional ISO timestamp to override total_days
    - end_time_utc: optional ISO timestamp to override "now"
    - info: optional shared Info instance (reuses meta calls across coins)
    - limiter: optional shared rate limiter (enforce spacing across coins)
    - debug: print progress and no-overwrite decisions

    The DB is updated with key = ISO ms timestamp, and the value stored under market_key
    like "BTC/USDC:USDC" using annualized percent scaling (by default).
    """
    if total_days <= 0:
        raise ValueError("total_days must be > 0")
    if window_days <= 0:
        raise ValueError("window_days must be > 0")

    min_interval = max(MIN_REQUEST_INTERVAL, sleep_seconds)
    if limiter is None:
        limiter = RateLimiter(min_interval)
    elif limiter.min_interval < min_interval:
        limiter.min_interval = min_interval

    if info is None:
        info = _call_with_backoff(
            lambda: Info(constants.MAINNET_API_URL, skip_ws=True),
            limiter=limiter,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            debug=debug,
            call_label="info.meta",
        )

    market_key = f"{coin}/USDC:USDC"
    now_utc = parse_iso_utc_optional(end_time_utc) or datetime.now(timezone.utc)
    base_start_utc = parse_iso_utc_optional(start_time_utc) or (now_utc - timedelta(days=total_days))
    if align_to_hour:
        base_start_utc = _to_hour_start(base_start_utc)
        now_utc = _to_hour_start(now_utc)
    if base_start_utc >= now_utc:
        raise ValueError("start_time_utc must be earlier than end_time_utc")
    start_utc = base_start_utc
    if overlap_days < 0:
        raise ValueError("overlap_days must be >= 0")
    db = load_db(db_path)
    last_dt = _latest_market_timestamp(db, market_key)
    if last_dt is not None:
        if align_to_hour:
            last_dt = _to_hour_start(last_dt)
        overlap_start = last_dt - timedelta(days=overlap_days)
        if start_utc < overlap_start:
            start_utc = overlap_start
    if start_utc >= now_utc:
        if debug:
            print(f"[SKIP] {coin} no new range to fetch.")
        return True

    # Walk windows: [win_start, win_end) ... up to now
    win_start = start_utc
    while win_start < now_utc:
        win_end = min(win_start + timedelta(days=window_days), now_utc)

        start_ms = int(win_start.timestamp() * 1000)
        end_ms = int(win_end.timestamp() * 1000)

        if debug:
            print(f"[FETCH] {coin} {win_start.isoformat()} -> {win_end.isoformat()} (UTC)")

        # API: request data for this window; endTime supported by SDK.
        try:
            raw = _call_with_backoff(
                lambda: info.funding_history(coin, startTime=start_ms, endTime=end_ms),
                limiter=limiter,
                backoff_base=backoff_base,
                backoff_max=backoff_max,
                debug=debug,
                call_label=f"funding_history:{coin}",
            )
        except Exception as e:
            print(f"[ERROR] funding_history({coin}, {start_ms}-{end_ms}) failed: {e}")
            # continue to next window (or break) depending on preference
            win_start = win_end
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

        # Update DB (dedup; may overwrite in overlap window)
        update_funding_db(
            db_path=db_path,
            records=window_recs,
            market_key=market_key,
            value_transform=lambda s: float(s),
            align_to_hour=align_to_hour,
            allow_overwrite=overlap_days > 0,
            debug=debug,
        )

        # Move to next window
        win_start = win_end

    return True

# ----------------------------
# Example batch usage
# ----------------------------

if __name__ == "__main__":
    config = load_backtest_config()
    db_path = config.get("db_path", "funding_db_test.json")
    coins = config.get("coins", ["BTC", "ETH", "SOL", "HYPE", "PUMP", "FARTCOIN", "PURR"])
    dl_cfg = config.get("downloader", {})
    min_interval = max(MIN_REQUEST_INTERVAL, dl_cfg.get("rate_limit_seconds", 0.1))
    limiter = RateLimiter(min_interval)
    info = _call_with_backoff(
        lambda: Info(constants.MAINNET_API_URL, skip_ws=True),
        limiter=limiter,
        backoff_base=dl_cfg.get("backoff_base_seconds", BACKOFF_BASE_SECONDS),
        backoff_max=dl_cfg.get("backoff_max_seconds", BACKOFF_MAX_SECONDS),
        debug=False,
        call_label="info.meta",
    )

    # Fetch last 365 days using 5-day rolling window
    print(f"Updating {db_path}...")
    for c in coins:
        print(f"  Doing {c}...")
        get_funding_history_rolling(
            db_path=db_path,
            coin=c,
            total_days=dl_cfg.get("total_days", 365), # from today if start_time_utc unset
            window_days=dl_cfg.get("window_days", 5),
            align_to_hour=True,
            sleep_seconds=min_interval,   # minimum delay between API calls
            backoff_base=dl_cfg.get("backoff_base_seconds", BACKOFF_BASE_SECONDS),
            backoff_max=dl_cfg.get("backoff_max_seconds", BACKOFF_MAX_SECONDS),
            overlap_days=dl_cfg.get("overlap_days", 2.0),
            start_time_utc=dl_cfg.get("start_time_utc"),
            end_time_utc=dl_cfg.get("end_time_utc"),
            info=info,
            limiter=limiter,
            debug=False          # set True to see detailed logs and no-overwrite skips
        )
        print(f"  Done.")
    
    print(f"Done Updating {db_path}.")
