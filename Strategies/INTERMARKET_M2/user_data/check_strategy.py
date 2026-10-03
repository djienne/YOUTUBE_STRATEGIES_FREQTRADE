"""Offline signal checks: docker compose run --rm --entrypoint python freqtrade user_data/check_strategy.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "strategies"))
import INTERMARKET_M2 as strategy


def main():
    # Threshold crossing opens a state; crossing zero closes it. NaN holds it.
    values = [0.2, 0.05, -0.01, -0.3, 0.0, 0.5, -0.6, np.nan, -0.05]
    np.testing.assert_array_equal(
        strategy.threshold_revert_signal(values, 0.1), [1, 1, 0, -1, 0, 1, -1, -1, -1]
    )
    np.testing.assert_array_equal(strategy.threshold_revert_signal([0.1, -0.1], 0.1), [0, 0])

    # A unit ramp has TR = ATR = 1, and close - SMA(4) = 1.5 after warmup.
    ramp = pd.DataFrame({name: np.arange(200.0) for name in ("high", "low", "close")})
    cmma = strategy.cmma(ramp, 4, 10).to_numpy()
    assert np.isnan(cmma[:10]).all()
    np.testing.assert_allclose(cmma[10:], 0.75)
    # Changing billions to dollars must not change this dimensionless indicator.
    np.testing.assert_allclose(strategy.cmma(ramp * 1e9, 4, 10), cmma, equal_nan=True)

    dates = pd.date_range("2026-09-20", periods=12, freq="12h", tz="UTC")
    weeks = pd.date_range("2025-07-28", periods=60, freq="7D", tz="UTC")
    vintages = {
        pd.Timestamp("2026-09-21"): pd.Series(1000.0 + np.arange(60), index=weeks),
        pd.Timestamp("2026-09-23"): pd.Series(1000.0 - np.arange(60), index=weeks),
    }
    pair = np.zeros(len(dates))
    full = strategy.replayed_signal(dates, pair, vintages, 6, 35, 0.05, 720, 24)
    closes = dates + pd.Timedelta(hours=12)
    first, second = pd.Timestamp("2026-09-22", tz="UTC"), pd.Timestamp("2026-09-24", tz="UTC")
    assert np.isnan(full[closes < first]).all()
    assert (full[(closes >= first) & (closes < second)] == 1).all()
    assert (full[closes >= second] == -1).all()
    for end in range(1, len(dates) + 1):
        known = {v: m2 for v, m2 in vintages.items()
                 if v.tz_localize("UTC") + pd.Timedelta(hours=24) <= closes[end - 1]}
        prefix = strategy.replayed_signal(dates[:end], pair[:end], known, 6, 35, 0.05, 720, 24)
        np.testing.assert_allclose(prefix, full[:end], equal_nan=True)
    print("PASS: threshold states, CMMA units, release timing and prefix causality")


if __name__ == "__main__":
    main()
