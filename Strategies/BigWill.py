# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from datetime import datetime
# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# This class is a sample. Feel free to customize it.

class BigWill(IStrategy):

    can_short: bool = False
    USE_TALIB: bool = False

    def custom_stochRSI_TravingView_Style(self, close, length=14, rsi_length=14, k=3, d=3):
        # Results between 0 and 1
        """Indicator: Stochastic RSI Oscillator (STOCHRSI)
        Should be similar to TradingView's calculation"""
        if k < 0:
            raise Exception("k cannot be negative")
        if d < 0:
            raise Exception("d cannot be negative")
        # Calculate Result
        rsi_ = pta.rsi(close, length=rsi_length, talib=self.USE_TALIB)
        lowest_rsi = rsi_.rolling(length).min()
        highest_rsi = rsi_.rolling(length).max()
        stochrsi = 100.0 * (rsi_ - lowest_rsi) / pta.non_zero_range(highest_rsi, lowest_rsi)
        if k > 0:
            stochrsi_k = pta.ma('sma', stochrsi, length=k, talib=self.USE_TALIB)
            stochrsi_d = pta.ma('sma', stochrsi_k, length=d, talib=self.USE_TALIB)
        else:
            stochrsi_k = None
            stochrsi_d = None
        return (stochrsi/100.0).round(4), (stochrsi_k/100.0).round(4), (stochrsi_d/100.0).round(4)

    """
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3
    
    HARD_TP_PC = DecimalParameter(0.10, 0.20, decimals=2, default=0.15, space="buy", optimize=True)

    stochWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsiWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)
    willWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)

    willOverSold = IntParameter(-99, -85, default=-85, space="buy", optimize=True)
    willOverBought = IntParameter(-86, 0, default=-10, space="buy", optimize=True)
    stochOverSold = DecimalParameter(0.1, 0.9, decimals=1, default=0.3, space="buy", optimize=True)
    stochOverBought = DecimalParameter(0.1, 0.9, decimals=1, default=0.7, space="buy", optimize=True)

    aoParam1 = IntParameter(3, 100, default=6, space="buy", optimize=True)
    aoParam2 = IntParameter(3, 100, default=27, space="buy", optimize=True)
    emaf = IntParameter(3, 599, default=3, space="buy", optimize=True)
    emas = IntParameter(20, 599, default=190, space="buy", optimize=True)

    use_custom_stoploss: bool = True
    process_only_new_candles: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 500.0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.75

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe['AO'] = pta.ao(dataframe['high'], dataframe['low'], fast=int(self.aoParam1.value), slow=int(self.aoParam2.value))

        dataframe['EMAf'] = pta.ema(dataframe['close'], length=int(self.emaf.value),talib=self.USE_TALIB)
        dataframe['EMAs'] = pta.ema(dataframe['close'], length=int(self.emas.value),talib=self.USE_TALIB)

        dataframe['STOCH_RSI'], _, _ = self.custom_stochRSI_TravingView_Style(close=dataframe['close'], length=self.stochWindow.value, rsi_length=self.rsiWindow.value, k=3, d=3)

        dataframe['WillR'] = pta.willr(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], length=self.willWindow.value, talib=self.USE_TALIB)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """

        conditions = []
        conditions.append(dataframe['AO'] >= 0)
        conditions.append(dataframe['AO'] < dataframe['AO'].shift(periods=1))
        conditions.append(dataframe['WillR'] < float(self.willOverSold.value))
        conditions.append(dataframe['EMAf'] > dataframe['EMAs'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        conditions = []
        conditions.append((dataframe['AO'] < 0) | (dataframe['WillR'] > float(self.willOverBought.value)))
        conditions.append((dataframe['STOCH_RSI'] > self.stochOverSold.value) | (dataframe['WillR'] > float(self.willOverBought.value)))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe

    # USED FOR HARD TAKE PROFIT
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        """
        if current_profit>self.HARD_TP_PC.value:
            return -0.0001
        return -0.75
