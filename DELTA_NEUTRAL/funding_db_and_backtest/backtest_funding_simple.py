# backtest_funding.py
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import pandas as pd

# ---------- DB I/O ----------
def load_db(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Time helpers ----------
def _parse_iso_utc(ts: str) -> datetime:
    """Parse ISO 8601 like 2025-07-27T08:00:00.000+00:00 to aware UTC datetime."""
    dt = datetime.fromisoformat(ts)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _iso_no_tz(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DDTHH:MM:SS in UTC, without ms or +00:00."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

# ---------- Data extraction ----------
def _sorted_coin_series(db: Dict, market_key: str) -> List[Tuple[datetime, float]]:
    """
    Extract (timestamp, annualized_pct) for market_key, sorted by time.
    Only includes timestamps that contain that market_key.
    """
    rows: List[Tuple[datetime, float]] = []
    for ts, payload in db.items():
        if isinstance(payload, dict) and market_key in payload:
            try:
                ann_pct = float(payload[market_key])  # stored annualized percent
                rows.append((_parse_iso_utc(ts), ann_pct))
            except (ValueError, TypeError):
                pass
    rows.sort(key=lambda x: x[0])
    return rows

# ---------- Backtest ----------
def backtest_funding(
    db_path: str,
    coin: str,
    initial_balance: float = 1000.0,
    quote_key: str = "USDC",
    settle_key: str = "USDC",
    compounding: bool = True,
    gap_behavior: str = "skip",  # "skip" or "zero"
) -> Dict:
    """
    Backtest funding P&L for a coin using DB values (annualized percent per hour bucket).
    """
    if gap_behavior not in ("skip", "zero"):
        raise ValueError("gap_behavior must be 'skip' or 'zero'")

    db = load_db(db_path)
    market_key = f"{coin}/{quote_key}:{settle_key}"
    series = _sorted_coin_series(db, market_key)

    # Return a consistent schema even when no data is found
    if not series:
        return {
            "coin": coin,
            "market_key": market_key,
            "start": None,
            "end": None,
            "interval_days": 0.0,
            "hours_in_span": 0,
            "hours_applied": 0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "error": "No data",
        }

    start_dt, end_dt = series[0][0], series[-1][0]
    ann_pct_by_dt = {dt: ann for dt, ann in series}

    balance = float(initial_balance)
    num_hours = 0
    applied_hours = 0

    cur = start_dt
    while cur <= end_dt:
        num_hours += 1
        ann_pct = ann_pct_by_dt.get(cur)

        if ann_pct is None and gap_behavior == "skip":
            cur += timedelta(hours=1)
            continue

        hourly_rate = 0.0 if ann_pct is None else (ann_pct / 100.0) / (365.0 * 24.0)
        if ann_pct is not None:
            applied_hours += 1

        if compounding:
            balance *= (1.0 + hourly_rate)
        else:
            balance += initial_balance * hourly_rate

        cur += timedelta(hours=1)

    total_return = (balance / initial_balance) - 1.0
    elapsed_days = max((end_dt - start_dt).total_seconds() / 86400.0, 1e-9)
    ann_factor = 365.0 / elapsed_days
    annualized_return = (1.0 + total_return) ** ann_factor - 1.0

    return {
        "coin": coin,
        "market_key": market_key,
        "start": _iso_no_tz(start_dt),
        "end": _iso_no_tz(end_dt),
        "interval_days": round(elapsed_days, 2),
        "hours_in_span": num_hours,
        "hours_applied": applied_hours if gap_behavior == "skip" else num_hours,
        "total_return_pct": round(total_return * 100.0, 4),
        "annualized_return_pct": round(annualized_return * 100.0, 4),
        "error": None,
    }

# ---------- Run & summary ----------
if __name__ == "__main__":
    db_path = "funding_db_for_backtest.json"  # change if needed
    coins = ["BTC", "ETH", "SOL", "HYPE", "PUMP", "FARTCOIN", "PURR"]

    results = []
    for c in coins:
        res = backtest_funding(
            db_path=db_path,
            coin=c,
            initial_balance=1000.0,
            gap_behavior="skip",
            compounding=True
        )
        results.append(res)

    # Build summary table (no compounding, gap_behavior, initial/final balances)
    cols = [
        "coin", "market_key", "start", "end", "interval_days",
        "hours_in_span", "hours_applied",
        "total_return_pct", "annualized_return_pct", "error"
    ]
    df = pd.DataFrame(results).reindex(columns=cols)

    # Pretty print
    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(df.to_string(index=False))

    # Optional: save CSV
    out_csv = "backtest_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}")
