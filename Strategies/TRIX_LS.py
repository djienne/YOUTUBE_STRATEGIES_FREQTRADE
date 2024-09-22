# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np
from numpy import NaN  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import copy

from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter)

# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import warnings

warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

# --------------------------------

# This class is a sample. Feel free to customize it.

class TRIX_LS(IStrategy):

    USE_TALIB = False

    df_list = {}

    current_positions = {}

    def custom_stochRSI(self, close, length=14, rsi_length=14):
        # Results between 0 and 1
        """Indicator: Stochastic RSI Oscillator (STOCHRSI)
        Should be similar to TradingView's calculation"""
        # Calculate Result
        rsi_ = pta.rsi(close, length=rsi_length, talib=self.USE_TALIB)
        lowest_rsi = rsi_.rolling(length).min()
        highest_rsi = rsi_.rolling(length).max()
        stochrsi = 100.0 * (rsi_ - lowest_rsi) / pta.non_zero_range(highest_rsi, lowest_rsi)
        return (stochrsi/100.0).round(4)
    
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True
    use_custom_stoploss: bool = False
    
    # Optimal timeframe for the strategy.
    timeframe = '1h'
        
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.75

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 500.00
    }

    stochLength = IntParameter(7, 21, default=18, space="buy", optimize=True)
    rsiLength = IntParameter(7, 21, default=16, space="buy", optimize=True)
    stochOverSold = DecimalParameter(0.1, 0.5, decimals=1, default=0.5, space="buy", optimize=True)
    stochOverBought = DecimalParameter(0.5, 0.9, decimals=1, default=0.9, space="buy", optimize=True)
    EMA_length = IntParameter(5, 600, default=556, space="buy", optimize=True)
    trixLength = IntParameter(5, 600, default=6, space="buy", optimize=True)
    trixSignal = IntParameter(5, 600, default=9, space="buy", optimize=True)

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 10

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        if self.dp.runmode.value in ('live','dry_run'):
            self.USE_TALIB = False # we do not use TA-LIB for live trading because sometimes it bugged
        else :
            self.USE_TALIB = True # we used TA-LIB when running backtest and hyperoptimisation because it runs faster
            
        dataframe['EMA'] = pta.ema(dataframe['close'], length=int(self.EMA_length.value), talib=self.USE_TALIB)
        tmp_df = pd.DataFrame()
        tmp_df['TRIX'] = pta.ema(pta.ema(pta.ema(dataframe['close'], length=int(self.trixLength.value), talib=self.USE_TALIB), length=int(self.trixLength.value), talib=self.USE_TALIB), length=int(self.trixLength.value), talib=self.USE_TALIB)
        tmp_df['TRIX_PCT'] = tmp_df["TRIX"].pct_change()*100.0
        tmp_df['TRIX_SIGNAL'] = pta.sma(tmp_df['TRIX_PCT'], length=int(self.trixSignal.value), talib=self.USE_TALIB)
        dataframe['TRIX_HISTO'] = tmp_df['TRIX_PCT'] - tmp_df['TRIX_SIGNAL']
        dataframe['STOCH_RSI'] = self.custom_stochRSI(close=dataframe['close'], length=self.stochLength.value, rsi_length=self.rsiLength.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['EMA'])
                &
                (dataframe['TRIX_HISTO'] > 0)
                &
                (dataframe['STOCH_RSI'] <= self.stochOverBought.value)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['EMA'])
                &
                (dataframe['TRIX_HISTO'] < 0)
                &
                (dataframe['STOCH_RSI'] >= self.stochOverSold.value)
            ),
            'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[
            (
                (dataframe['TRIX_HISTO'] < 0)
                &
                (dataframe['STOCH_RSI'] >= self.stochOverSold.value)
            ),
            'exit_long'] = 1
        dataframe.loc[
            (
                (dataframe['TRIX_HISTO'] > 0)
                &
                (dataframe['STOCH_RSI'] <= self.stochOverBought.value)
            ),
            'exit_short'] = 1
        return dataframe
        
