# multi_slot_funding_backtest.py
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timezone, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# DB I/O
# ----------------------------

def load_db(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# Time helpers
# ----------------------------

def parse_iso_utc(ts: str) -> datetime:
    """Parse ISO like 2025-07-27T08:00:00.000+00:00 or 2025-07-27T08:00:00 to aware UTC."""
    dt = datetime.fromisoformat(ts)
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def iso_no_tz(dt: datetime) -> str:
    """Format as 'YYYY-MM-DDTHH:MM:SS' in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

# ----------------------------
# Data shaping
# ----------------------------

def extract_coin_series(db: Dict, coin: str, quote="USDC", settle="USDC") -> Dict[datetime, float]:
    """
    Return {timestamp(datetime UTC): annualized_pct} for a coin.
    """
    key = f"{coin}/{quote}:{settle}"
    out: Dict[datetime, float] = {}
    for ts, payload in db.items():
        if isinstance(payload, dict) and key in payload:
            try:
                ann_pct = float(payload[key])
                out[parse_iso_utc(ts)] = ann_pct
            except (ValueError, TypeError):
                pass
    return out

def continuous_hour_grid(series_by_coin: Dict[str, Dict[datetime, float]]) -> List[datetime]:
    """Continuous hourly grid from the earliest timestamp across all coins to the latest."""
    mins = []
    maxs = []
    for s in series_by_coin.values():
        if s:
            ts_sorted = sorted(s.keys())
            mins.append(ts_sorted[0])
            maxs.append(ts_sorted[-1])
    if not mins:
        raise ValueError("No timestamps found for any coin.")
    start = min(mins)
    end = max(maxs)
    grid = []
    cur = start.replace(minute=0, second=0, microsecond=0)
    end = end.replace(minute=0, second=0, microsecond=0)
    while cur <= end:
        grid.append(cur)
        cur += timedelta(hours=1)
    return grid

def add_option_c_score(
    df: pd.DataFrame,
    w_ret: float = 0.6,   # weight for annualized return
    w_slots: float = 0.25,# weight for more slots
    w_hold: float = 0.15, # weight for shorter holding days
    eps: float = 1e-9
) -> pd.DataFrame:
    df = df.copy()

    # Normalize annualized return (higher is better)
    r_min, r_max = df["annualized_return_pct"].min(), df["annualized_return_pct"].max()
    df["ret_norm"] = (df["annualized_return_pct"] - r_min) / max(r_max - r_min, eps)

    # Normalize slots (higher is better)
    s_min, s_max = df["slots"].min(), df["slots"].max()
    df["slots_norm"] = (df["slots"] - s_min) / max(s_max - s_min, eps)

    # Normalize min_hold_days so that smaller is better
    h_min, h_max = df["min_hold_days"].min(), df["min_hold_days"].max()
    df["hold_norm"] = (h_max - df["min_hold_days"]) / max(h_max - h_min, eps)

    # Weighted geometric mean
    df["score_c"] = (
        (df["ret_norm"] + eps) ** w_ret *
        (df["slots_norm"] + eps) ** w_slots *
        (df["hold_norm"] + eps) ** w_hold
    )

    return df

# ----------------------------
# Backtest engine
# ----------------------------

@dataclass
class Trade:
    coin: str
    open_time: datetime
    open_equity: float
    close_time: Optional[datetime] = None
    close_equity: Optional[float] = None
    duration_hours: Optional[int] = None
    pnl_pct: Optional[float] = None  # (close/open - 1)

@dataclass
class SlotState:
    coin: Optional[str] = None
    equity: float = 0.0
    hours_held: int = 0
    total_switches: int = 0
    trade_log: List[Trade] = field(default_factory=list)

@dataclass
class BacktestParams:
    initial_balance: float = 1000.0
    slots: int = 2
    min_hold_days: int = 15
    open_fee_rate: float = 0.0005  # 0.05%
    close_fee_rate: float = 0.0005  # 0.05%
    quote_key: str = "USDC"
    settle_key: str = "USDC"
    apply_final_close_fee: bool = False  # if True, close open trades at the very end with a fee
    funding_efficiency: float = 0.5  # NEW: Only 50% of capital earns funding rate (due to hedging)

@dataclass
class BacktestResult:
    start: str
    end: str
    interval_days: float
    initial_balance: float
    final_balance: float
    total_return_pct: float
    annualized_return_pct: float
    total_switches: int
    avg_hold_days: float
    per_slot_summary: List[Dict]
    per_slot_trades: Dict[int, pd.DataFrame]
    equity_curve: Optional[pd.DataFrame] = None  # columns: ['timestamp','equity']

def annualized_pct_to_hourly_fraction(ann_pct: float, funding_efficiency: float = 0.5) -> float:
    """
    Convert annualized percent to hourly fraction, accounting for funding efficiency.
    
    Args:
        ann_pct: Annualized funding rate percentage
        funding_efficiency: Fraction of capital that actually earns funding rate (default 0.5 for hedged strategy)
    """
    return (ann_pct / 100.0) / (365.0 * 24.0) * funding_efficiency

def rank_top_n(ann_by_coin: Dict[str, Optional[float]], n: int) -> List[str]:
    """
    Return top-n coins by annualized percent for this hour.
    Ignore coins with None.
    """
    ranked = sorted(
        [(c, v) for c, v in ann_by_coin.items() if v is not None],
        key=lambda x: x[1],
        reverse=True
    )
    return [c for c, _ in ranked[:n]]

def backtest_multi_slot(
    db_path: str,
    coins: List[str],
    params: BacktestParams,
    return_curve: bool = True,
    debug: bool = False
) -> BacktestResult:
    db = load_db(db_path)

    # Load series for each coin (timestamp -> ann_pct)
    series_by_coin: Dict[str, Dict[datetime, float]] = {
        c: extract_coin_series(db, c, params.quote_key, params.settle_key) for c in coins
    }
    # Remove coins with no data
    series_by_coin = {c: s for c, s in series_by_coin.items() if len(s) > 0}
    if not series_by_coin:
        raise ValueError("No coin has data in the DB for the provided list.")

    # Continuous hourly grid
    grid = continuous_hour_grid(series_by_coin)
    start_dt, end_dt = grid[0], grid[-1]
    min_hold_hours = params.min_hold_days * 24

    # Initialize slots
    slot_equity_init = params.initial_balance / params.slots
    slots: List[SlotState] = [
        SlotState(coin=None, equity=slot_equity_init, hours_held=0, total_switches=0, trade_log=[])
        for _ in range(params.slots)
    ]

    portfolio_equity_curve: List[Tuple[datetime, float]] = []

    def portfolio_equity() -> float:
        return sum(s.equity for s in slots)

    total_switches = 0
    hold_hours_accumulated = 0
    hold_events = 0

    # Iterate hour by hour
    for t in grid:
        # Current funding snapshot (annualized pct) for ranking
        ann_by_coin: Dict[str, Optional[float]] = {
            c: series_by_coin[c].get(t) for c in series_by_coin.keys()
        }

        # Determine target = top-N by funding this hour
        target = rank_top_n(ann_by_coin, params.slots)

        # 1) Accrue funding to currently held positions, age them
        for s in slots:
            if s.coin is not None:
                ann_pct = ann_by_coin.get(s.coin)
                # UPDATED: Apply funding efficiency (only 50% of capital earns funding rate)
                hourly = annualized_pct_to_hourly_fraction(ann_pct, params.funding_efficiency) if ann_pct is not None else 0.0
                s.equity *= (1.0 + hourly)
                s.hours_held += 1

        # 2) Fill empty slots with needed coins first
        held: Set[str] = set([s.coin for s in slots if s.coin is not None])
        needed: List[str] = [c for c in target if c not in held]

        for s in slots:
            if not needed:
                break
            if s.coin is None:
                new_coin = needed.pop(0)
                # Open -> fee
                s.equity *= (1.0 - params.open_fee_rate)
                s.coin = new_coin
                s.hours_held = 0
                s.total_switches += 1
                total_switches += 1
                # Start trade record
                s.trade_log.append(Trade(
                    coin=new_coin,
                    open_time=t,
                    open_equity=s.equity
                ))
                if debug:
                    print(f"{iso_no_tz(t)} OPEN {new_coin}")

        # 3) For occupied slots not in target, if min-hold satisfied, switch to needed coins
        held = set([s.coin for s in slots if s.coin is not None])  # refresh
        needed = [c for c in target if c not in held]

        for s in slots:
            if not needed:
                break
            if s.coin is None or s.coin in target:
                continue
            if s.hours_held >= min_hold_hours:
                # Close current -> fee
                s.equity *= (1.0 - params.close_fee_rate)
                # Log close of existing trade
                if s.trade_log and s.trade_log[-1].close_time is None:
                    tr = s.trade_log[-1]
                    tr.close_time = t
                    tr.close_equity = s.equity
                    tr.duration_hours = s.hours_held
                    tr.pnl_pct = (s.equity / tr.open_equity) - 1.0
                    # hold stats
                    hold_hours_accumulated += s.hours_held
                    hold_events += 1

                # Switch
                new_coin = needed.pop(0)
                s.equity *= (1.0 - params.open_fee_rate)
                s.coin = new_coin
                s.hours_held = 0
                s.total_switches += 1
                total_switches += 1
                # New trade record
                s.trade_log.append(Trade(
                    coin=new_coin,
                    open_time=t,
                    open_equity=s.equity
                ))
                if debug:
                    print(f"{iso_no_tz(t)} SWITCH -> {new_coin}")

        # 4) Record equity
        if return_curve:
            portfolio_equity_curve.append((t, portfolio_equity()))

    # End-of-test handling: optionally close open trades with a fee
    if params.apply_final_close_fee:
        for s in slots:
            if s.coin is not None:
                s.equity *= (1.0 - params.close_fee_rate)
            if s.trade_log and s.trade_log[-1].close_time is None:
                tr = s.trade_log[-1]
                tr.close_time = end_dt
                tr.close_equity = s.equity
                tr.duration_hours = s.hours_held
                tr.pnl_pct = (s.equity / tr.open_equity) - 1.0
                hold_hours_accumulated += s.hours_held
                hold_events += 1
                s.coin = None
                s.hours_held = 0

    # If not closing, still count active holds for avg stats
    for s in slots:
        if s.coin is not None and s.hours_held > 0:
            hold_hours_accumulated += s.hours_held
            hold_events += 1

    elapsed_days = max((end_dt - start_dt).total_seconds() / 86400.0, 1e-9)
    init = params.initial_balance
    fin = sum(s.equity for s in slots)
    total_ret = (fin / init) - 1.0
    ann_factor = 365.0 / elapsed_days
    ann_ret = (1.0 + total_ret) ** ann_factor - 1.0

    avg_hold_days = (hold_hours_accumulated / max(hold_events, 1)) / 24.0

    curve_df = None
    if return_curve:
        curve_df = pd.DataFrame(
            [(iso_no_tz(ts), eq) for ts, eq in portfolio_equity_curve],
            columns=["timestamp", "equity"]
        )

    # Per-slot summaries and trade logs
    per_slot_summary = []
    per_slot_trades: Dict[int, pd.DataFrame] = {}
    for i, s in enumerate(slots, 1):
        per_slot_summary.append({
            "slot": i,
            "current_coin": s.coin,
            "equity": round(s.equity, 6),
            "switches": s.total_switches,
            "hours_held_current": s.hours_held
        })
        # Build trade DataFrame
        rows = []
        for tr in s.trade_log:
            rows.append({
                "coin": tr.coin,
                "open_time": iso_no_tz(tr.open_time),
                "close_time": iso_no_tz(tr.close_time) if tr.close_time else None,
                "duration_hours": tr.duration_hours,
                "open_equity": round(tr.open_equity, 6),
                "close_equity": round(tr.close_equity, 6) if tr.close_equity is not None else None,
                "pnl_pct": round(tr.pnl_pct * 100.0, 4) if tr.pnl_pct is not None else None
            })
        per_slot_trades[i] = pd.DataFrame(rows, columns=[
            "coin","open_time","close_time","duration_hours","open_equity","close_equity","pnl_pct"
        ])

    return BacktestResult(
        start=iso_no_tz(start_dt),
        end=iso_no_tz(end_dt),
        interval_days=round(elapsed_days, 2),
        initial_balance=round(init, 6),
        final_balance=round(fin, 6),
        total_return_pct=round(total_ret * 100.0, 4),
        annualized_return_pct=round(ann_ret * 100.0, 4),
        total_switches=total_switches,
        avg_hold_days=round(avg_hold_days, 2),
        per_slot_summary=per_slot_summary,
        per_slot_trades=per_slot_trades,
        equity_curve=curve_df
    )

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    db_path = "funding_db_test.json"   # <- path to your DB
    coins = ["BTC", "ETH", "SOL", "HYPE", "PUMP", "FARTCOIN", "PURR"]

    params = BacktestParams(
        initial_balance=1000.0,
        slots=2,                # try 1 first
        min_hold_days=30,
        open_fee_rate=0.0006,
        close_fee_rate=0.0006,
        apply_final_close_fee=False,  # set True if you want to force-close at end
        funding_efficiency=0.5        # NEW: 50% capital efficiency for hedged strategy
    )

    res = backtest_multi_slot(
        db_path=db_path,
        coins=coins,
        params=params,
        return_curve=True,
        debug=False
    )

    print("\n=== Portfolio Summary ===")
    summary = {
        "start": res.start,
        "end": res.end,
        "interval_days": res.interval_days,
        "initial_balance": res.initial_balance,
        "final_balance": res.final_balance,
        "total_return_pct": res.total_return_pct,
        "annualized_return_pct": res.annualized_return_pct,
        "total_switches": res.total_switches,
        "avg_hold_days": res.avg_hold_days,
        "funding_efficiency": params.funding_efficiency,  # NEW: Show efficiency in output
    }
    for k, v in summary.items():
        print(f"{k:22s} {v}")

    print("\n=== Per-slot Summary (current state) ===")
    print(pd.DataFrame(res.per_slot_summary).to_string(index=False))

    print("\n=== Per-slot Trade History ===")
    for slot_id, df_trades in res.per_slot_trades.items():
        print(f"\n-- Slot {slot_id} --")
        if df_trades.empty:
            print("(no trades)")
        else:
            print(df_trades.to_string(index=False))

    print(f"\n=== Note: Results account for {params.funding_efficiency*100}% capital efficiency ===")
    print("This reflects that in a hedged funding rate strategy, only the perpetual position")
    print("earns funding rates while the spot hedge position does not.")

    ## Run on a grid of # of slots and # of minimum holding days
    results_grid = []

    for slots in range(1, 4):  # N = 1 to 3
        for min_hold in range(7, 101):  # min_hold_days = 7 to 100
            params = BacktestParams(
                initial_balance=1000.0,
                slots=slots,
                min_hold_days=min_hold,
                open_fee_rate=0.0006,
                close_fee_rate=0.0006,
                apply_final_close_fee=False,
                funding_efficiency=0.5  # NEW: 50% efficiency
            )
            res = backtest_multi_slot(
                db_path=db_path,
                coins=coins,
                params=params,
                return_curve=False,
                debug=False
            )
            results_grid.append({
                "slots": slots,
                "min_hold_days": min_hold,
                "final_balance": res.final_balance,
                "total_return_pct": res.total_return_pct,
                "annualized_return_pct": res.annualized_return_pct,
                "total_switches": res.total_switches,
                "avg_hold_days": res.avg_hold_days
            })

    df_grid = pd.DataFrame(results_grid)
    df_grid.to_csv("grid_results.csv", index=False)
    print(df_grid)
    print("\nSaved grid results to grid_results.csv")

    # Apply, sort, and show the top combinations
    df_scored = add_option_c_score(df_grid, w_ret=0.6, w_slots=0.20, w_hold=0.20)
    df_scored = df_scored.sort_values("score_c", ascending=False)

    print("\nTop 15 parameter sets by Option C score:")
    print(df_scored[["slots","min_hold_days","annualized_return_pct","total_return_pct","score_c"]].head(15).to_string(index=False))

    # Pivot table for heatmap
    pivot = df_grid.pivot(index="slots", columns="min_hold_days", values="annualized_return_pct")

    plt.figure(figsize=(14, 6))
    heatmap = plt.imshow(pivot, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(heatmap, label="Annualized Return %")

    # Axis labels and ticks
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns, rotation=90)
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.xlabel("Minimum Holding Days")
    plt.ylabel("Number of Slots")
    plt.title("Annualized Return Heatmap (slots vs min_hold_days) - 50% Capital Efficiency")

    plt.tight_layout()
    plt.show()
