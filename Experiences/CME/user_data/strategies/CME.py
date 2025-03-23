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
                                IStrategy, IntParameter, stoploss_from_absolute, informative, Order)
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
# --------------------------------
# Add your lib to import here
from pathlib import Path
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
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

class CME(IStrategy):

    INTERFACE_VERSION = 3

    max_entry_position_adjustment = 0
    can_short: bool = False
    use_custom_stoploss: bool = False
    position_adjustment_enable: bool  = False
    process_only_new_candles = True

    # min time after friday close and time window where it is allowed to open position
    min_hours_after_friday_close = IntParameter(1, 72, default=64, space='buy', optimize=True)
    window_duration_hours = IntParameter(2, 96, default=84, space='buy', optimize=True)

    # maximum time in hours to keep current trade open if gap is not filled
    give_up_time_delta_hours = IntParameter(24, 168, default=157, space='buy', optimize=True)

    # minimum required gap percent below last week CME close to enter position
    min_gap_percent = DecimalParameter(0.02, 0.15, default=0.07, decimals=2, space='buy', optimize=True)
    # trailing stop as fraction of min_gap_percent
    trailing_stop_fraction = DecimalParameter(0.01, 0.70, default=0.01, decimals=2, space='buy', optimize=True)
    stop_loss_pc = DecimalParameter(0.01, 0.70, default=0.01, decimals=2, space='buy', optimize=True)

    # ignore roi parameter
    minimal_roi = {
        "0": 500.00 # 50000 % i.e. useless
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.95

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 24

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

    def get_btc_cme_weekly_data(self):
        """
        Loads Bitcoin futures data from CME weekly candles from a CSV file.
        
        Returns:
            pd.DataFrame: DataFrame with the data.
        """

        # Determine the base directory.
        base_dir = Path(__file__).resolve().parent.parent if '__file__' in globals() else Path.cwd()
        file_path = base_dir / "cme_data" / "BTC1_weekly_data.csv"

        # re-download CME data if live run (dry-run or real money)
        if self.dp.runmode.value in ('live', 'dry_run'):
            try:
                # Initialize the tvDatafeed instance with your TradingView credentials
                tv = TvDatafeed()
                print("Downloading BTC1! weekly data...")

                # Get historical data:
                # - symbol: 'BTC1!'
                # - exchange: 'CME' (adjust if your symbol is on a different exchange)
                # - interval: Interval.in_weekly downloads weekly bars
                # - n_bars: number of bars to download (e.g., 52 for about 1 year)
                # - fut_contract=1: used for continuous futures contracts (if applicable)
                data = tv.get_hist(symbol='BTC1!',
                                exchange='CME',
                                interval=Interval.in_weekly,
                                n_bars=10000)
                
                # Check if data was returned successfully
                if data is None or data.empty:
                    print("No data was returned. Please check the symbol/exchange credentials.")
                    return

                # Save the data to CSV
                #print(data)
                data.to_csv(file_path)
                print(f"Data saved successfully to {file_path}")

            except Exception as e:
                print("An error occurred while downloading data:", str(e))
            
        # Load the CSV data.
        df = pd.read_csv(file_path)
        
        # Convert the 'datetime' column from string to datetime.
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # If today is a weekday (Monday=0 to Friday=4), drop the last row,
        # assuming the last candle is incomplete.
        if datetime.now().weekday() < 5:
            df = df.drop(df.index[-1])
            
        return df

    def add_time_window_check(self, df):
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Constants for the CME close time:
        # Assume CME closes at 18:00 UTC on Friday, and we wait 30 minutes.
        CME_CLOSE_BASE = pd.Timedelta(hours=23)       # CME close at 23:00 UTC after maintenance
        CME_CLOSE_DELAY = pd.Timedelta(hours=1)      # delay after close
        
        # Compute the candidate Friday for each candle.
        # This finds the most recent Friday by subtracting the appropriate number of days.
        candidate_friday = df_copy['date'] - pd.to_timedelta((df_copy['date'].dt.weekday - 4) % 7, unit='d')
        # Set the reference time: Friday at 18:00 UTC plus the delay → Friday at 18:30 UTC.
        friday_cme_close = candidate_friday.dt.normalize() + CME_CLOSE_BASE + CME_CLOSE_DELAY
        
        # Define the opening window relative to Friday's reference time.
        window_start = friday_cme_close + pd.Timedelta(hours=self.min_hours_after_friday_close.value)
        window_end = window_start + pd.Timedelta(hours=self.window_duration_hours.value)
        
        # Mark each row as True if its timestamp falls within the entry window.
        df['is_in_opening_window'] = (df_copy['date'] >= window_start) & (df_copy['date'] <= window_end)
        
        return df

    def add_historical_last_cme_closes(self, df, df_cme):
        # Used only for backtests
        # Work on copies so that we don't alter the originals.
        df_copy = df.copy()
        df_cme_copy = df_cme.copy()
        
        # Convert the 'date' column in df to datetime (naive).
        df_copy['_date_temp'] = pd.to_datetime(df_copy['date']).dt.tz_localize(None)
        
        # Convert the 'datetime' column in df_cme to datetime (naive) and shift it back by 5 days.
        df_cme_copy['datetime'] = pd.to_datetime(df_cme_copy['datetime']).dt.tz_localize(None) + pd.Timedelta(days=5)
        # Rename the 'close' column to avoid conflicts.
        df_cme_copy = df_cme_copy[['datetime', 'close']].rename(columns={'close': 'cme_close'})
        
        # Use merge_asof to merge on the unshifted left date.
        merged_df = pd.merge_asof(
            df_copy,
            df_cme_copy,
            left_on='_date_temp',
            right_on='datetime',
            direction='backward'
        )
        
        # Remove temporary columns.
        merged_df.drop(columns=['datetime', '_date_temp'], inplace=True)

        return merged_df

    def get_signals_trailing(self, dataframe: DataFrame):
        # signal = 1 for enter long and 0 for exit
        # same get_signals but with a trailing stop take profit and fixed stop loss
        df_copy = dataframe.copy()
        df_copy = self.add_historical_last_cme_closes(df_copy, self.get_btc_cme_weekly_data())
        df_copy = self.add_time_window_check(df_copy)

        # Pre-initialize a NumPy array for signals with only zeros (exit).
        signals = np.zeros(len(df_copy), dtype=int)

        position_open = False
        open_time = None        # When the position was opened.
        open_cme_close = None   # The CME close value used at open.
        trailing_stop_active = False  # Flag to indicate if trailing stop management is active.
        best_close = None       # Highest close since trailing stop activated.
        position_open_close_price = None

        for ii, row in enumerate(df_copy.itertuples()):
            current_time = row.date

            if not position_open:
                # Open condition: if within the allowed window and the candle's close is at least 
                # self.min_gap_percent.value below the current cme_close.
                if row.is_in_opening_window and row.close < row.cme_close * (1.0 - self.min_gap_percent.value):
                    position_open = True
                    open_time = current_time
                    open_cme_close = row.cme_close
                    trailing_stop_active = False
                    best_close = row.close  # Initialize best_close; trailing stop not active yet.
                    position_open_close_price = row.close
                    signals[ii] = 1
                else:
                    signals[ii] = 0
            else:
                hours_open = (current_time - open_time).total_seconds() / 3600.0
                signals[ii] = 1

                # Always check if the position has exceeded the allowed holding time.
                # we let the trailing stop profit trigger for potentially more time than give_up_time_delta_hours.
                if hours_open >= self.give_up_time_delta_hours.value and not trailing_stop_active:
                    position_open = False
                    open_time = None
                    open_cme_close = None
                    trailing_stop_active = False
                    best_close = None
                    position_open_close_price = None
                    signals[ii] = 0
                    continue

                # Check fixed stop loss only when trailing stop is not activated.
                if not trailing_stop_active:
                    stop_loss_level = position_open_close_price * (1.0 - self.stop_loss_pc.value)
                    if row.close < stop_loss_level:
                        position_open = False
                        open_time = None
                        open_cme_close = None
                        trailing_stop_active = False
                        best_close = None
                        position_open_close_price = None
                        signals[ii] = 0
                        continue

                # Activate trailing stop once the price has reached/exceeded the open CME close.
                if row.close >= open_cme_close:
                    if not trailing_stop_active:
                        trailing_stop_active = True
                        best_close = row.close
                    else:
                        best_close = max(best_close, row.close)

                    # Calculate the trailing stop level.
                    trailing_stop_gap = open_cme_close * self.min_gap_percent.value * self.trailing_stop_fraction.value
                    trailing_stop_level = best_close - trailing_stop_gap

                    # Exit the position if the price falls below the trailing stop level.
                    if row.close < trailing_stop_level:
                        position_open = False
                        open_time = None
                        open_cme_close = None
                        trailing_stop_active = False
                        best_close = None
                        position_open_close_price = None
                        signals[ii] = 0
                        continue
                    else:
                        signals[ii] = 1
                        continue
                else:
                    # Before reaching the open CME close, hold the position.
                    signals[ii] = 1
                    continue

        dataframe['signal'] = signals
        return dataframe
        
    def get_signals(self, dataframe: DataFrame):
        df_copy = dataframe.copy()
        df_copy = self.add_historical_last_cme_closes(df_copy, self.get_btc_cme_weekly_data())
        df_copy = self.add_time_window_check(df_copy)

        # Pre-initialize a NumPy array with zeros for every row in df.
        signals = np.zeros(len(df_copy), dtype=int)

        position_open = False
        open_time = None      # Will hold the datetime when the position was opened.
        open_cme_close = None # Will hold the cme_close used at open.

        for ii, row in enumerate(df_copy.itertuples()):
            current_time = row.date  # Assuming 'date' is the datetime column.
            
            if not position_open:
                # Check if we're allowed to open a position:
                # The candle's close is at least self.min_gap_percent.value % below the current cme_close
                # and the market is in an opening window.
                if row.is_in_opening_window and row.close < row.cme_close * (1.0 - self.min_gap_percent.value):
                    # Open a position: record the open time and the CME close value used at open.
                    position_open = True
                    open_time = current_time
                    open_cme_close = row.cme_close
                    signals[ii] = 1
                    continue
                else:
                    signals[ii] = 0
                    continue
            else:
                signals[ii] = 1
                # When a position is open, calculate the hours elapsed since opening.
                hours_open = (current_time - open_time).total_seconds() / 3600.0
                
                # Check closing conditions:
                # 1. The gap has been filled: the candle's close is >= the stored open_cme_close.
                # 2. The "give up time" has been reached.
                if row.close >= open_cme_close or hours_open >= self.give_up_time_delta_hours.value:
                    position_open = False
                    open_time = None
                    open_cme_close = None
                    signals[ii] = 0
                    continue
                else:
                    signals[ii] = 1
                    continue

        # Assign the computed signals to a new column in the DataFrame.
        dataframe['signal'] = signals

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        #dataframe = self.get_signals(dataframe)
        dataframe = self.get_signals_trailing(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['signal'] == 1)
        
        # Vérifie si toutes les conditions sont remplies
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['signal'] == 0)
        
        # Vérifie si toutes les conditions sont remplies
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'
            ] = 1

        return dataframe

    class HyperOpt:
            # Define a custom stoploss space.
            def stoploss_space():
                return [SKDecimal(-0.85, -0.01, decimals=2, name='stoploss')]
################################################################################################
