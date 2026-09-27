# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""Long-only Donchian strategy gated by hypothetical opposite-leg losses.

The outcome filter is reconstructed from each input dataframe, not from the
executed trade database. Signals can therefore depend on history length.
"""
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
import math
from pandas import DataFrame
from functools import reduce
from typing import Optional
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative, Order)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from pathlib import Path
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
import warnings
import logging
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
logger = logging.getLogger(__name__)

def donchian_breakout(df: pd.DataFrame, lookback: int):
    """Add close-based bands and a persistent +1/-1 breakout signal in place.

    Bands use up to lookback - 1 preceding closes (min_periods=1), excluding
    the current close. The signal remains NaN until the first breakout.
    """
    df['upper'] = df['close'].rolling(lookback - 1, center=False, min_periods=1).max().shift(1)
    df['lower'] = df['close'].rolling(lookback - 1, center=False, min_periods=1).min().shift(1)
    df['signal'] = np.nan
    df.loc[df['close'] > df['upper'], 'signal'] = 1
    df.loc[df['close'] < df['lower'], 'signal'] = -1
    df['signal'] = df['signal'].ffill()

def last_trade_adj_signal(ohlc: pd.DataFrame, signal: np.array, last_winner: bool = False):
    """Gate each direction on the last completed hypothetical opposite leg.

    Use row-aligned signals (+1/-1, with zero allowed during initialization)
    and row closes as transition prices. Select a losing leg by default or
    a winning leg with last_winner=True; ties and unknown outcomes return zero.
    Fees, actual fills and the trade database are not used. State resets on
    every call, so changing the dataframe's start can change later signals.
    """

    last_type = -1
    if last_winner:
        last_type = 1
    
    close = ohlc['close'].to_numpy()
    mod_signal = np.zeros(len(signal))

    long_entry_p = np.nan
    short_entry_p = np.nan
    last_long = np.nan
    last_short = np.nan

    last_sig = 0.0
    for i in range(len(close)):
        if signal[i] == 1.0 and last_sig != 1.0: # Long entry
            long_entry_p = close[i]
            if not np.isnan(short_entry_p):
                last_short = np.sign(short_entry_p - close[i])
                short_entry_p = np.nan

        if signal[i] == -1.0  and last_sig != -1.0: # Short entry
            short_entry_p = close[i]
            if not np.isnan(long_entry_p):
                last_long = np.sign(close[i] - long_entry_p)
                long_entry_p = np.nan
        
        last_sig = signal[i]
        
        if signal[i] == 1.0 and last_short == last_type:
            mod_signal[i] = 1.0
        if signal[i] == -1.0 and last_long == last_type:
            mod_signal[i] = -1.0
        
    return mod_signal

class donchian(IStrategy):
    """Enter on last_lose=+1; exit on zero or -1. Never open actual shorts."""

    INTERFACE_VERSION = 3

    max_entry_position_adjustment = 0
    can_short: bool = False
    use_custom_stoploss: bool = False
    position_adjustment_enable: bool  = False
    process_only_new_candles = True

    # A 500x return threshold effectively disables ordinary ROI exits.
    minimal_roi = {
        "0": 500.00  # Ratio units: 50,000%.
    }

    # Config or exported strategy parameters can override this -85% stoploss.
    stoploss = -0.85

    # Trailing stoploss
    trailing_stop = False

    # Hourly signal timeframe.
    timeframe = '1h'

    # The adjacent donchian.json currently overrides the default with 185.
    lookback = IntParameter(10, 200, default=120, space='buy', optimize=True)

    # Warmup requested from Freqtrade; it does not ensure history-independent
    # outcome-filter state (see the September 2026 comparison report).
    startup_candle_count: int = 205

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

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """Shift the breakout signal one row, then evaluate legs at row closes."""
        lookback = self.lookback.value
        data = df.copy()
        
        # Generate Donchian signals
        donchian_breakout(data, lookback)
        
        # Shift the signal, not the prices. The filter below reads current closes.
        data['signal_shifted'] = data['signal'].shift(1)  # Use previous candle's signal
        
        # Filter using hypothetical leg outcomes, not executed-trade P&L.
        data['last_lose'] = last_trade_adj_signal(data, data['signal_shifted'].fillna(0).to_numpy(), last_winner=False)
        
        df = pd.merge(df, data[['date', 'last_lose']], on='date', how='left')
        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe[f'last_lose'] == 1.0),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe[f'last_lose'] == -1.0) | (dataframe[f'last_lose'] == 0.0)),
            'exit_long'
        ] = 1
        return dataframe
