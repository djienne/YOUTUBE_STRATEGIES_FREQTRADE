# pragma pylint: disable=missing-docstring, invalid-name
"""
INTERMARKET_M2: Bitcoin and Ethereum against the US money supply (M2).

Idea, indicator and signal rule: neurotrader, "Intramarket Indicator Differences | Algorithmic
Crypto Trading Strategy in Python", https://www.youtube.com/watch?v=n2mY86S01fg. His repository
(https://github.com/neurotrader888/IntramarketDifference) has no licence, so none of its code is
copied here: cmma and threshold_revert_signal below are written from the formula and the rule he
presents. neurotrader compares ETH with BTC; here the other market is M2, the US money supply (FRED series WM2NS, billions of
dollars, not seasonally adjusted, one value per week ending Monday).

    cmma = (close - SMA(close, n)) / (ATR(m) * sqrt(n))
        distance to the moving average in volatility units, so that a weekly money-supply
        series and a crypto pair become comparable.
    diff = cmma(M2) - cmma(pair)
    state = -1 once diff < -threshold, +1 once diff > threshold, back to 0 when diff
        crosses zero.
    flip = -1: buy when the pair's cmma is above M2's by more than the threshold
        (state -1), sell when the state is back to 0. Each pair is traded on its own.
Windows are given in weeks (M2 is weekly) and converted to candles: 6 weeks = 84 candles
of 12h. The parameters in INTERMARKET_M2.json come from a Hyperopt run in March 2025.

M2 is published by the Federal Reserve (H.6 release) with a delay: 10 days until
February 2021 (weekly releases), 22 to 50 days since (monthly release, 4th Tuesday).
Live, the bot reads the series as currently published (FRED CSV). A backtest must not use
a week before it was published, so backtests replay every published version of the
series ("vintages" from ALFRED, see download_m2_vintages.py): each candle gets the signal
a live bot would have computed at its close, with the M2 data published by then.
"""
import logging
import os
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WM2NS"
WEEK_MINUTES = 7 * 24 * 60


def cmma(ohlc, lookback, atr_lookback):
    """Distance of the close to its moving average, in volatility units of ATR * sqrt(lookback)."""
    close = ohlc["close"]
    unit = ta.atr(ohlc["high"], ohlc["low"], close, length=atr_lookback) * np.sqrt(lookback)
    return (close - close.rolling(lookback, min_periods=1).mean()) / unit


def threshold_revert_signal(ind, threshold):
    """State +1 or -1 once ind leaves [-threshold, threshold] on that side; back to 0 once ind
    reaches zero or crosses it (state * ind <= 0). NaN keeps the state."""
    states = np.zeros(len(ind))
    state = 0
    for i, x in enumerate(np.asarray(ind, dtype=float)):
        if x > threshold:
            state = 1
        elif x < -threshold:
            state = -1
        elif state * x <= 0:
            state = 0
        states[i] = state
    return states


def m2_cmma(m2, lookback, atr_lookback):
    """cmma of the weekly M2 series (one value per week, so high = low = close)."""
    return cmma(pd.DataFrame({"high": m2, "low": m2, "close": m2}), lookback, atr_lookback)


def intermarket_signal(dates, pair_cmma, m2_weekly_cmma, threshold):
    """State machine on diff. Each candle takes the latest M2 week dated at or before it,
    so live the current candle uses the last week published."""
    m2 = m2_weekly_cmma.reindex(dates, method="ffill").to_numpy()
    return threshold_revert_signal(m2 - pair_cmma, threshold)


def read_m2_csv(source):
    frame = pd.read_csv(source)
    if list(frame.columns) != ["observation_date", "WM2NS"]:
        raise ValueError(f"unexpected FRED columns {list(frame.columns)}")
    m2 = pd.Series(pd.to_numeric(frame["WM2NS"], errors="raise").to_numpy(),
                   index=pd.DatetimeIndex(pd.to_datetime(frame["observation_date"], utc=True)))
    if not m2.index.is_monotonic_increasing or (m2 <= 0).any():
        raise ValueError("M2 series must be increasing in time and positive")
    return m2


def fred_m2(path, max_age_hours=4):
    """Weekly M2 as published today (FRED), cached; the last good copy is used if FRED fails."""
    path = Path(path)
    if not path.exists() or time.time() - path.stat().st_mtime > max_age_hours * 3600:
        try:
            # Default User-Agent: FRED left requests with a custom one unanswered (Sep 2026).
            response = requests.get(FRED_CSV, timeout=30)
            response.raise_for_status()
            read_m2_csv(StringIO(response.text))  # validate before replacing the cache
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(response.text, encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            if not path.exists():
                raise
            logger.warning("M2 download from FRED failed (%s); using the cached copy %s", exc, path)
    return read_m2_csv(path)


def load_vintages(path):
    """{release date: weekly M2 as published that day}, file from download_m2_vintages.py."""
    table = pd.read_csv(path, parse_dates=["vintage_date", "observation_date"])
    table["observation_date"] = table["observation_date"].dt.tz_localize("UTC")
    return {v: g.set_index("observation_date")["value"] for v, g in table.groupby("vintage_date")}


def replayed_signal(dates, pair_cmma, vintages, lookback, atr_lookback, threshold, tf_minutes, delay_hours):
    """Signal a live bot would have computed at each candle close, with the M2 then published.

    Live, the whole dataframe is evaluated with the latest release, so the state at a
    candle depends on that release: each release is replayed on the candles that close
    while it is the latest one. A release is usable delay_hours after 00:00 UTC of its
    release day (published 13:00 or 16:30 New York time, so 24 = first close after it).
    """
    closes = dates + pd.Timedelta(minutes=tf_minutes)
    releases = sorted(vintages)
    usable = pd.DatetimeIndex(releases).tz_localize("UTC") + pd.Timedelta(hours=delay_hours)
    latest = usable.searchsorted(closes, side="right") - 1
    signal = np.full(len(dates), np.nan)  # no signal before the first release
    for k in np.unique(latest[latest >= 0]):
        rows = np.flatnonzero(latest == k)
        end = rows[-1] + 1
        m2 = m2_cmma(vintages[releases[k]], lookback, atr_lookback)
        signal[rows] = intermarket_signal(dates[:end], pair_cmma[:end], m2, threshold)[rows]
    return signal


class INTERMARKET_M2(IStrategy):

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "12h"
    process_only_new_candles = True

    # Exits come from the signal: no ROI, no trailing, a stoploss that never triggers in
    # practice (-0.95 in INTERMARKET_M2.json).
    minimal_roi = {"0": 500.0}
    stoploss = -0.85
    trailing_stop = False

    # Live, 3000 startup candles make Freqtrade keep about 4000 candles of 12h: the
    # 490-candle ATR then forgets where the data starts (weight about 0.1 %). Backtests
    # replay the whole history, so they can start as soon as the ATR exists (490 candles
    # after the first candle): set "startup_candle_count" in the backtest config.
    startup_candle_count = 3000

    order_types = {"entry": "market", "exit": "market", "stoploss": "market", "stoploss_on_exchange": False}

    lookback = IntParameter(5, 200, default=24, space="buy", optimize=True)  # weeks
    atr_lookback = IntParameter(5, 200, default=168, space="buy", optimize=True)  # weeks
    threshold = DecimalParameter(0.02, 0.99, decimals=2, default=0.10, space="buy", optimize=True)
    flip = CategoricalParameter([-1, 1], default=-1, space="buy", optimize=True)

    _vintages = None

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        minutes = timeframe_to_minutes(self.timeframe)
        if WEEK_MINUTES % minutes:
            raise ValueError(f"timeframe {self.timeframe} does not divide a week")
        fact = WEEK_MINUTES // minutes  # candles per week
        lookback, atr_lookback = self.lookback.value, self.atr_lookback.value
        pair = cmma(df, lookback * fact, atr_lookback * fact).to_numpy()
        dates = pd.DatetimeIndex(df["date"])
        m2_dir = Path(self.config["user_data_dir"]) / "m2"
        if self.dp.runmode in (RunMode.LIVE, RunMode.DRY_RUN):
            m2 = m2_cmma(fred_m2(m2_dir / "WM2NS_fred.csv"), lookback, atr_lookback)
            df["signal"] = intermarket_signal(dates, pair, m2, self.threshold.value)
        else:
            if INTERMARKET_M2._vintages is None:
                INTERMARKET_M2._vintages = load_vintages(m2_dir / "WM2NS_vintages.csv.gz")
            df["signal"] = replayed_signal(dates, pair, INTERMARKET_M2._vintages, lookback, atr_lookback,
                                           self.threshold.value, minutes, self.config.get("m2_delay_hours", 24))
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[df["signal"] == self.flip.value, "enter_long"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # As in the dry run, a long exits when the state is back to 0: a direct jump from
        # -1 to +1 (sharp drop of the pair) keeps it open. Exiting on +1 as well was tested
        # (same data, same parameters): +95 % instead of +328 % over 2018-2025.
        df.loc[df["signal"] == 0, "exit_long"] = 1
        return df
