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
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List

import warnings
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

class Trump_LIM(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    max_entry_position_adjustment = 10
    can_short: bool = False
    use_custom_stoploss: bool = True
    position_adjustment_enable: bool  = True
    process_only_new_candles = True

    nb_levels = IntParameter(8, 10, default=10, space="buy", optimize=True)
    dca_factor = DecimalParameter(1.0, 2.0, decimals=1, default=1.0, space="buy", optimize=True)
    target_loss = DecimalParameter(0.04, 0.09, decimals=2, default=0.08, space="buy", optimize=True)

    vol_periods = IntParameter(5, 120, default=51, space="buy", optimize=True)
    vol_factor = IntParameter(2, 10, default=2, space="buy", optimize=True)

    loss_sep_factor = DecimalParameter(0.1, 1.0, decimals=1, default=0.3, space="buy", optimize=True)

    min_candle_variation = DecimalParameter(0.001, 1.000, decimals=3, default=0.027, space="buy", optimize=True)

    trailing_val = DecimalParameter(0.001, 0.050, decimals=3, default=0.003, space="buy", optimize=True)

    HARD_TP = DecimalParameter(0.01, 0.16, decimals=2, default=0.12, space="buy", optimize=True)

    min_stake_amount = 11.0 # USDT, change for other pairs

    # ignore roi parameter
    minimal_roi = {
        "0": 500.00 # 50000 % i.e. useless
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.85

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 121

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
        return dataframe

    @informative('5m')
    def populate_indicators_5m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        for p in self.vol_periods.range:
            dataframe[f'volume_ma_{p}'] = dataframe['volume'].rolling(window=p).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Détecte les conditions d'entrée en position basées sur un volume anormal
        """
        conditions = []
        
        # Condition 1: Le volume actuel est X fois supérieur à la moyenne mobile
        p = self.vol_periods.value
        conditions.append(dataframe['volume_5m'] > (dataframe[f'volume_ma_{p}_5m'] * float(self.vol_factor.value)))

        # Condition 2: Vérification que la baisse est significative
        conditions.append(
            (dataframe['close_5m'] - dataframe['open_5m']) / dataframe['open_5m'] < -1.0*self.min_candle_variation.value
        )
        
        # Vérifie si toutes les conditions sont remplies
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        """
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
    
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        """
        This decides how much stake to use on the VERY FIRST entry of the trade.
        We'll only do the fraction needed such that the sum of geometric series
        equals the total stake.

        Also ensures a minimum stake of self.min_stake_amount USDT.
        """
        if self.nb_levels.value <= 1:
            # If only 1 level, stake everything at once (but at least self.min_stake_amount USDT)
            return max(self.min_stake_amount, proposed_stake)

        r = self.dca_factor.value  # The geometric ratio
        n = float(self.nb_levels.value)
        
        # Avoid division-by-zero if r==1
        if abs(r - 1.0) < 1e-6:
            # If ratio ~ 1, the sum is basically n
            sum_of_series = n
        else:
            sum_of_series = (1.0 - (r ** n)) / (1.0 - r)

        # The fraction of the total capital to be used in the *first* entry
        stake_for_first_entry = proposed_stake / sum_of_series

        #print(stake_for_first_entry)

        # Enforce a minimum of self.min_stake_amount USDT
        return max(self.min_stake_amount, stake_for_first_entry)

    def custom_stoploss( # can also be stop profit
        self,
        pair: str,
        trade: 'Trade',
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        if current_profit > self.HARD_TP.value:
            return -1.0 * self.trailing_val.value

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs
    ) -> Optional[float]:
        """
        :return float: Additional stake to add to the existing position.
                    Return None for no action.
        """
        # Stop if we've reached the max number of entries
        current_entries = int(trade.nr_of_successful_entries)  # is always >=1
        max_allowed_entries = int(self.nb_levels.value)
        if current_entries > max_allowed_entries:
            return None

        # Determine the target loss threshold based on the current number of entries
        required_loss = 0.0
        for i in range(0, current_entries):
            factor = 1.0 + (float(i) * self.loss_sep_factor.value)
            required_loss = required_loss + self.target_loss.value * factor
        # 5% 11% 18% ...

        # How much stake has already been used?
        filled_entry_orders = trade.select_filled_orders(trade.entry_side)

        initial_stake = filled_entry_orders[0].stake_amount
        #last_entry_price = filled_entry_orders[-1].price
        initial_entry_price = filled_entry_orders[0].price

        change_from_initial_entry_price = (current_rate-initial_entry_price)/initial_entry_price

        # Check if current loss meets threshold
        if change_from_initial_entry_price > -1.0*required_loss:
            return None

        # The geometric multiplier for the current (k-th) entry is r^(current_entries).
        dca_multiplier = self.dca_factor.value ** float(current_entries)

        # Proposed new stake amount based on DCA progression.
        new_stake_amount = initial_stake * dca_multiplier

        # Check how much capital is left before hitting max_stake.
        #    If leftover < self.min_stake_amount, we skip (return None).
        used_so_far = sum(o.stake_amount for o in filled_entry_orders)
        leftover = max_stake - used_so_far

        if leftover < self.min_stake_amount:
            # Not enough left to place at least self.min_stake_amount
            return None

        # Clamp the new stake to leftover (we cannot exceed leftover).
        # Then ensure it is at least self.min_stake_amount.
        new_stake_amount = min(new_stake_amount, leftover)

        if new_stake_amount < self.min_stake_amount:
            return None  # If after clamping it's still < self.min_stake_amount, skip

        # Return the final stake amount.
        return new_stake_amount

################################################################################################
