# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np
from numpy import NaN  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from datetime import datetime
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy, IntParameter, stoploss_from_open, informative, DecimalParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import ta as ttaa
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

# --------------------------------

# This class is a sample. Feel free to customize it.


class TRIX_spot(IStrategy):

    def custom_stochRSI_TravingView_Style(self, close, length=14, rsi_length=14, k=3, d=3):
        # Results between 0 and 1
        """Indicator: Stochastic RSI Oscillator (STOCHRSI)
        Should be similar to TradingView's calculation"""
        if k < 0:
            raise Exception("k cannot be negative")
        if d < 0:
            raise Exception("d cannot be negative")
        # Calculate Result
        rsi_ = pta.rsi(close, length=rsi_length, talib=False)
        lowest_rsi = rsi_.rolling(length).min()
        highest_rsi = rsi_.rolling(length).max()
        stochrsi = 100.0 * (rsi_ - lowest_rsi) / pta.non_zero_range(highest_rsi, lowest_rsi)
        if k > 0:
            stochrsi_k = pta.ma('sma', stochrsi, length=k, talib=False)
            stochrsi_d = pta.ma('sma', stochrsi_k, length=d, talib=False)
        else:
            stochrsi_k = None
            stochrsi_d = None
        return (stochrsi/100.0).round(4), (stochrsi_k/100.0).round(4), (stochrsi_d/100.0).round(4)

    """
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False
    use_custom_stoploss: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 5000.00
    }

    stochLength = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsiLength = IntParameter(7, 21, default=14, space="buy", optimize=True)
    
    stochOverSold = DecimalParameter(0.1, 0.9, decimals=1, default=0.2, space="buy", optimize=True)
    stochOverBought = DecimalParameter(0.1, 0.9, decimals=1, default=0.8, space="buy", optimize=True)
    EMA_length = IntParameter(3, 600, default=284, space="buy", optimize=True)
    trixLength = IntParameter(2, 200, default=6, space="buy", optimize=True)
    trixSignal = IntParameter(2, 200, default=9, space="buy", optimize=True)
    
    SRSI_K = IntParameter(2, 6, default=3, space="buy", optimize=True)
    SRSI_D = IntParameter(2, 6, default=3, space="buy", optimize=True)

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.95

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 600

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

    def informative_pairs(self):
        """
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe['EMA'] = ta.EMA(dataframe['close'], timeperiod=int(self.EMA_length.value))

        dataframe['TRIX'] = ta.EMA(ta.EMA(ta.EMA(dataframe['close'], timeperiod=int(self.trixLength.value)), timeperiod=int(self.trixLength.value)), timeperiod=int(self.trixLength.value))
        dataframe['TRIX_PCT'] = dataframe["TRIX"].pct_change()*100.0
        dataframe['TRIX_SIGNAL'] = ta.SMA(dataframe['TRIX_PCT'], timeperiod=int(self.trixSignal.value))
        dataframe['TRIX_HISTO'] = dataframe['TRIX_PCT'] - dataframe['TRIX_SIGNAL']

        dataframe['STOCH_RSI'], _, _ = self.custom_stochRSI_TravingView_Style(close=dataframe['close'], length=self.stochLength.value, rsi_length=self.rsiLength.value, k=self.SRSI_K.value, d=self.SRSI_D.value)

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

    # def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
    #                     current_rate: float, current_profit: float, **kwargs) -> float:
    #     """
    #     """
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = dataframe.iloc[-1].squeeze()
    #     if last_candle['TRIX_HISTO'] < 0 and last_candle['STOCH_RSI'] >= self.stochOverSold.value :
    #         return -0.0003
    #     return -0.95
