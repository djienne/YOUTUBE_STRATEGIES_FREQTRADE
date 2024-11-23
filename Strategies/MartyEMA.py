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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

class MartyEMA(IStrategy):

    USE_TALIB = True

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

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    position_adjustment_enable: bool  = True
    max_entry_position_adjustment = 4
    can_short: bool = False
    use_custom_stoploss: bool = True
    process_only_new_candles = True

    LEV = DecimalParameter(1.0, 4.0, decimals=0, default=1.0, space="buy", optimize=True)

    nb_levels = IntParameter(2, 4, default=2, space="buy", optimize=True)
    dca_factor = DecimalParameter(0.0, 1.0, decimals=1, default=0.0, space="buy", optimize=True)

    stochWindow = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_length_p = IntParameter(7, 21, default=14, space="buy", optimize=True)
    atr_per = IntParameter(7, 21, default=14, space="buy", optimize=True)

    stochOverSold = DecimalParameter(0.1, 0.5, decimals=1, default=0.1, space="buy", optimize=True)
    stochOverBought = DecimalParameter(0.5, 0.9, decimals=1, default=0.8, space="buy", optimize=True)

    UP = IntParameter(3, 15, default=12, space="buy", optimize=True)
    DOWN = IntParameter(3, 15, default=10, space="buy", optimize=True)

    ema1 = IntParameter(5, 150, default=32, space="buy", optimize=True)
    delta_ema2 = IntParameter(5, 300, default=100, space="buy", optimize=True)
    delta_ema3 = IntParameter(5, 300 , default=100, space="buy", optimize=True)

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 5000.0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.90

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 751

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

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe['EMA1'] = pta.ema(dataframe['close'], length=int(self.ema1.value), talib=self.USE_TALIB)
        dataframe['EMA2'] = pta.ema(dataframe['close'], length=int(self.ema1.value + self.delta_ema2.value), talib=self.USE_TALIB)
        dataframe['EMA3'] = pta.ema(dataframe['close'], length=int(self.ema1.value + self.delta_ema2.value + self.delta_ema3.value), talib=self.USE_TALIB)
        
        _, dataframe['K'], dataframe['D'] = self.custom_stochRSI_TravingView_Style(close=dataframe['close'], length=int(self.stochWindow.value), rsi_length=int(self.rsi_length_p.value), k=3, d=3)
        
        dataframe['ATR'] = pta.atr(dataframe['high'],dataframe['low'],dataframe['close'], length=int(self.atr_per.value), talib=self.USE_TALIB)

        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        conditions = []
        conditions.append(dataframe['EMA1_15m'] >= dataframe['EMA2_15m'])
        conditions.append(dataframe['EMA2_15m'] >= dataframe['EMA3_15m'])
        conditions.append(dataframe['close'] >= dataframe['EMA1_15m'])
        conditions.append(dataframe['K_15m'] < self.stochOverSold.value)
        conditions.append(dataframe['D_15m'] < self.stochOverSold.value)
        conditions.append(dataframe['K_15m'].shift(1) > dataframe['D_15m'].shift(1))
        conditions.append(dataframe['K_15m'] <= dataframe['D_15m'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        # dataframe.loc[:, 'exit_short'] = 0
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        # We need to leave most of the funds for possible further DCA orders
        # This also applies to fixed stakes
        if self.nb_levels.value<=1:
            proposed_stake2 = proposed_stake
        elif self.nb_levels.value==2:
            proposed_stake2 = proposed_stake/( 1.0 + (1.0 + self.dca_factor.value) )
        elif self.nb_levels.value==3:
            proposed_stake2 = proposed_stake/( 1.0 + (1.0 + self.dca_factor.value) + (1.0 + 2.0*self.dca_factor.value) )
        elif self.nb_levels.value==4:
            proposed_stake2 = proposed_stake/( 1.0 + (1.0 + self.dca_factor.value) + (1.0 + 2.0*self.dca_factor.value) + (1.0 + 3.0*self.dca_factor.value) )
        return proposed_stake2
        
    # USED FOR STOP LOSS AND TAKE PROFIT
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]
        # SL
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            target_loss = trade_candle['ATR_15m']*self.DOWN.value/trade.open_rate
            c2 = current_profit < -1.0*target_loss
            count_of_entries = int(trade.nr_of_successful_entries)
            if c2 and count_of_entries>=int(self.nb_levels.value): # stop loss only if the safety orders have been done
                return 0.0
        # TP
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            target_profit = trade_candle['ATR_15m']*self.UP.value/trade.open_rate
            c1 = current_profit > target_profit
            if c1:
                return 0.0

        return self.stoploss

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag,
                 side: str, **kwargs) -> float:
        val = self.LEV.value
        if val > max_leverage:
            val = max_leverage
        return val 
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
        """

        count_of_entries = int(trade.nr_of_successful_entries)

        if count_of_entries >= int(self.nb_levels.value):
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)

        # check if PnL is low enough to do the DCA (safety) order
        try:
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_candle = dataframe.loc[dataframe['date'] == trade_date]
            if not trade_candle.empty:
                trade_candle = trade_candle.squeeze()
                target_loss = trade_candle['ATR_15m']*self.DOWN.value / trade.open_rate
            else:
                return None
            
            if current_profit > -1.0*target_loss:
                return None
            
        except Exception as e:
            return None
        
        # Only buy when not actively falling price.
        try:
            last_candle = dataframe.iloc[-1].squeeze()
            previous_candle = dataframe.iloc[-2].squeeze()
            if last_candle['close'] < previous_candle['close']:
                return None
        except Exception as e:
            return None

        try:
            filled_entries = trade.select_filled_orders(trade.entry_side)
            # This returns first order stake size
            stake_amount_initial = filled_entries[0].stake_amount
            # This then calculates current safety order size
            stake_amount = stake_amount_initial * (1.0 + float(count_of_entries)*float(self.dca_factor.value))
            #stake_amount = stake_amount_initial
            return stake_amount*0.99
        except Exception as e:
            return None
        
