# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
import math
from pandas import DataFrame
from functools import reduce
from typing import Optional
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade, Order
from datetime import datetime, timedelta
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
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

class Market_Making(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    min_spread = 0.3/100.0 # minimum spread to avoid insane backtest results

    SPREAD_UP1 = DecimalParameter(0.05, 0.5, decimals=2, default=0.31, space="buy", optimize=True)
    SPREAD_DOWN1 = DecimalParameter(0.05, 1.40, decimals=2, default=0.35, space="buy", optimize=True)
    nper1 = IntParameter(2, 200, default=20, space="buy", optimize=True)
    multiplicator_UP1 = DecimalParameter(0.1, 0.5, decimals=1, default=1.0, space="buy", optimize=True)
    multiplicator_DOWN1 = DecimalParameter(0.1, 2.0, decimals=1, default=1.0, space="buy", optimize=True)
    atr_or_std1 = CategoricalParameter(["atr", "std"], default="std", space="buy", optimize=True)

    SPREAD_UP2 = DecimalParameter(0.05, 0.5, decimals=2, default=0.31, space="buy", optimize=True)
    SPREAD_DOWN2 = DecimalParameter(0.05, 1.40, decimals=2, default=0.35, space="buy", optimize=True)
    nper2 = IntParameter(2, 200, default=20, space="buy", optimize=True)
    multiplicator_UP2 = DecimalParameter(0.1, 0.5, decimals=1, default=1.0, space="buy", optimize=True)
    multiplicator_DOWN2 = DecimalParameter(0.1, 2.0, decimals=1, default=1.0, space="buy", optimize=True)
    atr_or_std2 = CategoricalParameter(["atr", "std"], default="std", space="buy", optimize=True)

    st_short_atr_window = IntParameter(2, 49, default=10, space="buy", optimize=True)
    st_short_atr_multiplier = DecimalParameter(1.0, 6.0, decimals=1, default=5.6, space="buy", optimize=True)

    # Can this strategy go short?
    can_short: bool = False
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = True
    position_adjustment_enable: bool = False
    max_entry_position_adjustment = 0

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": -1
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
    timeframe = '1m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'limit',
        "emergency_exit": "limit",
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

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        st = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=int(self.st_short_atr_window.value), multiplier=self.st_short_atr_multiplier.value)

        dataframe['super_trend_direction'] = st['SUPERTd_' + str(self.st_short_atr_window.value)+"_"+str(self.st_short_atr_multiplier.value)]

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe[f'STD1'] = dataframe['close'].rolling(int(self.nper1.value)).std()
        dataframe[f'ATR1'] = ta.ATR(dataframe, timeperiod=int(self.nper1.value))

        dataframe[f'STD2'] = dataframe['close'].rolling(int(self.nper2.value)).std()
        dataframe[f'ATR2'] = ta.ATR(dataframe, timeperiod=int(self.nper2.value))

        #st = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=int(self.st_short_atr_window.value), multiplier=self.st_short_atr_multiplier.value)
        #dataframe['super_trend'] = st['SUPERT_' + str(self.st_short_atr_window.value)+"_"+str(self.st_short_atr_multiplier.value)]
        #dataframe['super_trend_direction'] = st['SUPERTd_' + str(self.st_short_atr_window.value)+"_"+str(self.st_short_atr_multiplier.value)]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
        
    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: str, side: str, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if dataframe['super_trend_direction_15m'].iat[-2] == 1:
            if self.atr_or_std1.value=='std':
                calculated_spread = self.SPREAD_DOWN1.value/100.0 + dataframe[f'STD1'].iat[-2]/dataframe['close'].iat[-2]*self.multiplicator_DOWN1.value
                effective_spread = max(calculated_spread, self.min_spread)
                new_entryprice = proposed_rate*(1.0 - effective_spread)
            elif self.atr_or_std1.value=='atr':
                calculated_spread = self.SPREAD_DOWN1.value/100.0 + dataframe[f'ATR1'].iat[-2]/dataframe['close'].iat[-2]*self.multiplicator_DOWN1.value
                effective_spread = max(calculated_spread, self.min_spread)
                new_entryprice = proposed_rate*(1.0 - effective_spread)
        else:
            if self.atr_or_std2.value=='std':
                calculated_spread = self.SPREAD_DOWN2.value/100.0 + dataframe[f'STD2'].iat[-2]/dataframe['close'].iat[-2]*self.multiplicator_DOWN2.value
                effective_spread = max(calculated_spread, self.min_spread)
                new_entryprice = proposed_rate*(1.0 - effective_spread)
            elif self.atr_or_std2.value=='atr':
                calculated_spread = self.SPREAD_DOWN2.value/100.0 + dataframe[f'ATR2'].iat[-2]/dataframe['close'].iat[-2]*self.multiplicator_DOWN2.value
                effective_spread = max(calculated_spread, self.min_spread)
                new_entryprice = proposed_rate*(1.0 - effective_spread)
        return new_entryprice

    def custom_exit_price(self, pair: str, trade: Trade,
                        current_time: datetime, proposed_rate: float,
                        current_profit: float, exit_tag: str, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            if trade_candle['super_trend_direction_15m'] == 1:
                if self.atr_or_std1.value=='std':
                    calculated_spread = self.SPREAD_UP1.value/100.0 + trade_candle[f'STD1']/trade_candle['close']*self.multiplicator_UP1.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_exitprice = proposed_rate*(1.0 + effective_spread)
                elif self.atr_or_std1.value=='atr':
                    calculated_spread = self.SPREAD_UP1.value/100.0 + trade_candle[f'ATR1']/trade_candle['close']*self.multiplicator_UP1.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_exitprice = proposed_rate*(1.0 + effective_spread)
            else:
                if self.atr_or_std2.value=='std':
                    calculated_spread = self.SPREAD_UP2.value/100.0 + trade_candle[f'STD2']/trade_candle['close']*self.multiplicator_UP2.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_exitprice = proposed_rate*(1.0 + effective_spread)
                elif self.atr_or_std2.value=='atr':
                    calculated_spread = self.SPREAD_UP2.value/100.0 + trade_candle[f'ATR2']/trade_candle['close']*self.multiplicator_UP2.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_exitprice = proposed_rate*(1.0 + effective_spread)
            return new_exitprice
        else:
            return proposed_rate

    def adjust_entry_price(self, trade: Trade, order: Order, pair: str,
                            current_time: datetime, proposed_rate: float, current_order_rate: float,
                            entry_tag: str, side: str, **kwargs) -> float:
                            
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            if trade_candle['super_trend_direction_15m'] == 1:
                if self.atr_or_std1.value=='std':
                    calculated_spread = self.SPREAD_DOWN1.value/100.0 + trade_candle[f'STD1']/trade_candle['close']*self.multiplicator_DOWN1.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_entryprice = proposed_rate*(1.0 - effective_spread)
                elif self.atr_or_std1.value=='atr':
                    calculated_spread = self.SPREAD_DOWN1.value/100.0 + trade_candle[f'ATR1']/trade_candle['close']*self.multiplicator_DOWN1.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_entryprice = proposed_rate*(1.0 - effective_spread)
            else:
                if self.atr_or_std2.value=='std':
                    calculated_spread = self.SPREAD_DOWN2.value/100.0 + trade_candle[f'STD2']/trade_candle['close']*self.multiplicator_DOWN2.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_entryprice = proposed_rate*(1.0 - effective_spread)
                elif self.atr_or_std2.value=='atr':
                    calculated_spread = self.SPREAD_DOWN2.value/100.0 + trade_candle[f'ATR2']/trade_candle['close']*self.multiplicator_DOWN2.value
                    effective_spread = max(calculated_spread, self.min_spread)
                    new_entryprice = proposed_rate*(1.0 - effective_spread)
            return new_entryprice
        else:
            return proposed_rate

    @property
    def protections(self):
        return [
            {
                "method": "MaxDrawdown",
                "lookback_period": 10080,  # 1 week
                "trade_limit": 0,  # Evaluate all trades since the bot started
                "stop_duration_candles": 10000000,  # Stop trading indefinitely
                "max_allowed_drawdown": 0.05  # Maximum drawdown of 5% before stopping
            },
        ]