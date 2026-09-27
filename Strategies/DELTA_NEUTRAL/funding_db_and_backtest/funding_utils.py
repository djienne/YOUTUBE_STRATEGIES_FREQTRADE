import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional


def load_db(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_iso_utc(ts: str) -> datetime:
    """Parse ISO like 2025-07-27T08:00:00.000+00:00 to aware UTC."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def floor_to_hour(dt: datetime) -> datetime:
    """Normalize to the hour to align funding timestamps with hourly buckets."""
    return dt.replace(minute=0, second=0, microsecond=0)


def iso_no_tz(dt: datetime) -> str:
    """Format as 'YYYY-MM-DDTHH:MM:SS' in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def parse_iso_utc_optional(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return parse_iso_utc(value)

def load_backtest_config(path: str = "config_backtest.json") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_coin_series(db: Dict, coin: str, quote: str = "USDC", settle: str = "USDC") -> Dict[datetime, float]:
    """Return {timestamp(datetime UTC): annualized_pct} for a coin."""
    key = f"{coin}/{quote}:{settle}"
    out: Dict[datetime, float] = {}
    for ts, payload in db.items():
        if isinstance(payload, dict) and key in payload:
            try:
                ann_pct = float(payload[key])
                out[floor_to_hour(parse_iso_utc(ts))] = ann_pct
            except (ValueError, TypeError):
                pass
    return out


def filter_series_by_time(
    series_by_dt: Dict[datetime, float],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> Dict[datetime, float]:
    if not series_by_dt:
        return series_by_dt
    if start_dt:
        start_dt = floor_to_hour(start_dt)
    if end_dt:
        end_dt = floor_to_hour(end_dt)
    if start_dt is None and end_dt is None:
        return series_by_dt
    out: Dict[datetime, float] = {}
    for dt, val in series_by_dt.items():
        if start_dt is not None and dt < start_dt:
            continue
        if end_dt is not None and dt > end_dt:
            continue
        out[dt] = val
    return out


def annualized_pct_to_hourly_fraction(ann_pct: float, funding_efficiency: float = 0.5) -> float:
    """
    Convert annualized percent to hourly fraction, accounting for funding efficiency.
    """
    return (ann_pct / 100.0) / (365.0 * 24.0) * funding_efficiency
