# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
from typing import Dict, List
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from datetime import datetime
# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# SIMPLE STRATEGY THAT ENTERS A TRADE WITH RSI IS LARGER THAN A GIVEN VALUE AND USES FOLLOWING ROI AND STOPLOSS:
#    "roi": {
#      "0": 0.646,
#      "10535": 0.323,
#      "23703": 0.106,
#      "33290": 0
#    },
#    "stoploss": {
#      "stoploss": -0.148
#    },

# This class is a sample. Feel free to customize it.

class SimpleRSI(IStrategy):

    can_short: bool = False
    
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3
    
    lev = DecimalParameter(1.0, 4.0, decimals=0, default=1.0, space="buy", optimize=True)
    
    rsiWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)
    minRSI = DecimalParameter(1, 99, decimals=0, default=80, space="buy", optimize=True)

    use_custom_stoploss: bool = False
    process_only_new_candles: bool = True

    position_adjustment_enable: bool = False
    # max_entry_position_adjustment = 1

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 500.0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1d'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

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
        dataframe['RSI'] = ta.RSI(dataframe, timeperiod=int(self.rsiWindow.value))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """

        conditions = []
        conditions.append(dataframe['RSI'] >= self.minRSI.value)
        conditions.append(dataframe['RSI'].shift(1) < self.minRSI.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        conditions = []
        conditions.append(dataframe['RSI'] >= 105) # never exits

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe
        
    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str, **kwargs) -> float:
        val = self.lev.value
        if val > max_leverage:
            val = max_leverage
        return val 
    
    @property
    def protections(self):
        return  [
            {
                "method": "CooldownPeriod",
                "stop_duration": 10080
            }
        ]
    
    class HyperOpt:
        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [Integer(6, 50, name='max_open_trades'),]
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.95, -0.05, decimals=2, name='stoploss')]
               
    # def adjust_trade_position(self, trade: Trade, current_time: datetime,
    #                             current_rate: float, current_profit: float,
    #                             min_stake: float, max_stake: float,
    #                             current_entry_rate: float, current_exit_rate: float,
    #                             current_entry_profit: float, current_exit_profit: float,
    #                             **kwargs) -> float:
    #     """
    #     """
    #     # if current_profit > 0.5 and trade.nr_of_successful_exits == 0:
    #     #     return -(trade.stake_amount * 0.666)
    #     return current_rate
        
