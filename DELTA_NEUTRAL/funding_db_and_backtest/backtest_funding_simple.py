# backtest_funding.py
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd

from funding_utils import (
    annualized_pct_to_hourly_fraction,
    filter_series_by_time,
    extract_coin_series,
    iso_no_tz,
    load_backtest_config,
    load_db,
    parse_iso_utc_optional,
)

# ---------- Backtest ----------
def backtest_funding(
    db_path: str,
    coin: str,
    initial_balance: float = 1000.0,
    quote_key: str = "USDC",
    settle_key: str = "USDC",
    compounding: bool = True,
    gap_behavior: str = "skip",  # "skip" or "zero"
    funding_efficiency: float = 0.5,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> Dict:
    """
    Backtest funding P&L for a coin using DB values (annualized percent per hour bucket).
    """
    if gap_behavior not in ("skip", "zero"):
        raise ValueError("gap_behavior must be 'skip' or 'zero'")

    db = load_db(db_path)
    market_key = f"{coin}/{quote_key}:{settle_key}"
    series_by_dt = extract_coin_series(db, coin, quote_key, settle_key)
    series_by_dt = filter_series_by_time(series_by_dt, start_dt, end_dt)

    # Return a consistent schema even when no data is found
    if not series_by_dt:
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

    ts_sorted = sorted(series_by_dt.keys())
    start_dt, end_dt = ts_sorted[0], ts_sorted[-1]
    ann_pct_by_dt = series_by_dt

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

        hourly_rate = 0.0 if ann_pct is None else annualized_pct_to_hourly_fraction(
            ann_pct, funding_efficiency
        )
        if ann_pct is not None:
            applied_hours += 1

        if compounding:
            balance *= (1.0 + hourly_rate)
        else:
            balance += initial_balance * hourly_rate

        cur += timedelta(hours=1)

    total_return = (balance / initial_balance) - 1.0
    hours_applied = applied_hours if gap_behavior == "skip" else num_hours
    elapsed_days = max(hours_applied / 24.0, 1e-9)
    ann_factor = 365.0 / elapsed_days
    annualized_return = (1.0 + total_return) ** ann_factor - 1.0

    return {
        "coin": coin,
        "market_key": market_key,
        "start": iso_no_tz(start_dt),
        "end": iso_no_tz(end_dt),
        "interval_days": round(elapsed_days, 2),
        "hours_in_span": num_hours,
        "hours_applied": hours_applied,
        "total_return_pct": round(total_return * 100.0, 4),
        "annualized_return_pct": round(annualized_return * 100.0, 4),
        "error": None,
    }

# ---------- Run & summary ----------
if __name__ == "__main__":
    config = load_backtest_config()
    db_path = config.get("db_path", "funding_db_test.json")
    coins = config.get("coins", ["BTC", "ETH", "SOL", "HYPE", "PUMP", "FARTCOIN", "PURR"])
    simple_cfg = config.get("simple_backtest", {})

    start_dt = parse_iso_utc_optional(simple_cfg.get("start_time_utc"))
    end_dt = parse_iso_utc_optional(simple_cfg.get("end_time_utc"))
    initial_balance = simple_cfg.get("initial_balance", 1000.0)
    gap_behavior = simple_cfg.get("gap_behavior", "skip")
    compounding = simple_cfg.get("compounding", True)
    funding_efficiency = simple_cfg.get("funding_efficiency", 0.5)

    results = []
    for c in coins:
        res = backtest_funding(
            db_path=db_path,
            coin=c,
            initial_balance=initial_balance,
            gap_behavior=gap_behavior,
            compounding=compounding,
            funding_efficiency=funding_efficiency,
            start_dt=start_dt,
            end_dt=end_dt
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
