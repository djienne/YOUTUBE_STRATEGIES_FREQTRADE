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
from datetime import datetime
from freqtrade.persistence import Trade
from technical.util import (resample_to_interval, resampled_merge)
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_open, informative)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
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
class SuperReversal_mtf(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True
    process_only_new_candles: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 500.00
    }

    st_short_atr_window = IntParameter(8, 20, default=10, space="buy", optimize=True)
    st_short_atr_multiplier = DecimalParameter(1.0, 6.0, decimals=1, default=5.5, space="buy", optimize=True)

    short_ema_window = IntParameter(3, 600, default=30, space="buy", optimize=True)
    long_ema_window = IntParameter(10, 600, default=545, space="buy", optimize=True)

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.75

    # Trailing stoploss
    trailing_stop: bool = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        st = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'],
                            length=int(self.st_short_atr_window.value), multiplier=self.st_short_atr_multiplier.value)

        dataframe['super_trend_direction'] = st['SUPERTd_' + str(self.st_short_atr_window.value) + "_" + str(self.st_short_atr_multiplier.value)]

        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=int(self.short_ema_window.value))

        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=int(self.long_ema_window.value))

        #print(dataframe.tail(2))

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_short_val"] = int(self.short_ema_window.value)
        dataframe["ema_long_val"] = int(self.long_ema_window.value)

        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:

        if min_stake>proposed_stake:
            proposed_stake = min_stake

        return proposed_stake

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe["super_trend_direction_1h"] == 1)
                    &
                    (dataframe["ema_short_1h"] > dataframe["ema_long_1h"])
                    &
                    (dataframe["high"] > dataframe[f"ema_short_1h"])
                    &
                    (dataframe["low"] < dataframe[f"ema_short_1h"])
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                    (dataframe["super_trend_direction_1h"] == -1)
                    &
                    (dataframe["ema_short_1h"] < dataframe["ema_long_1h"])
                    &
                    (dataframe["high"] > dataframe[f"ema_short_1h"])
                    &
                    (dataframe["low"] < dataframe[f"ema_short_1h"])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    ((dataframe["super_trend_direction_1h"] == -1) | (dataframe["ema_short_1h"] < dataframe["ema_long_1h"]))
                    &
                    (dataframe["high"] > dataframe["ema_short_1h"])
                    &
                    (dataframe["low"] < dataframe["ema_short_1h"])
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                    ((dataframe["super_trend_direction_1h"] == 1) | (dataframe["ema_short_1h"] > dataframe["ema_long_1h"]))
                    &
                    (dataframe["high"] > dataframe["ema_short_1h"])
                    &
                    (dataframe["low"] < dataframe["ema_short_1h"])
            ),
            'exit_short'] = 1

        return dataframe


################################################################################################

class SuperReversal_mtf_5min(SuperReversal_mtf):
    timeframe = '5m'
