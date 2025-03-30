# NO REPAINT VERSION OF THE HMM STRATEGY
# CODE MAIN CHANGE TO AVOID REPAINTING:
        # # --- Predict market regimes on the daily test data ---
        # market_regime_predictions = []
        # for i in range(len(ft_data)):
        #     # Use rows up to and including the current row for prediction
        #     current_row_features = ft_data[features].iloc[:i+1].values
        #     current_prediction = hmm_model.predict(current_row_features)
        #     market_regime_predictions.append(current_prediction[-1])
        # ft_data['market_regime'] = market_regime_predictions
        # ft_data = assign_market_signals(ft_data)

# New Free Parameters (to optimize with hyperopt):

# momentum_signal_delay (IntParameter):
# Sets the delay (in number of candles) used to compute momentum indicators. 
#  This delay is applied when calculating logarithmic momentum values, influencing how past price changes contribute to the HMM features. Range: 2–100, Default: 10.

# number_of_momentums_to_use (IntParameter):
# Determines the count of momentum features to include in the HMM model. For each feature, a momentum value is calculated based on the set delay,
#  allowing the model to capture multiple time horizons. Range: 1–20, Default: 1.

# n_components (CategoricalParameter):
# Specifies the number of hidden states (components) for the Gaussian HMM.
#  These states represent different market regimes; choosing among 2, 3, or 4 affects the granularity of regime detection. Options: [2, 3, 4], Default: 2.

import os
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from hmmlearn.hmm import GaussianHMM

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.optimize.space import SKDecimal
from typing import Dict, Optional, Union, Tuple

# Configure logging and warnings
logger = logging.getLogger(__name__)
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def get_data(coin) -> pd.DataFrame:
    """
    Get coin historical data from a local CSV file.

    """

    csv_path = os.path.join("user_data", "yf_data",f"{coin}_daily_data.csv")
    if not os.path.exists(csv_path):
        logger.error("CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV, skipping the first two rows so that the third row becomes the header.
    data = pd.read_csv(
        csv_path,
        index_col=0,             # Use the first column ('Date') as index
        parse_dates=True,        # Automatically parse dates
    )
    
    # Ensure the index is in datetime format
    data.index = pd.to_datetime(data.index)

    return data

def assign_market_signals(df):
    """
    Assigns trading signals based on the average log returns of each market regime.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns 'log_returns' and 'market_regime'.
        The 'market_regime' column should contain values (e.g., 0, 1, 2).

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an additional column 'signal':
            - 1 for the regime with the highest average log_return (long)
            - -1 for the regime with the lowest average log_return (short)
            - 0 for the remaining regime (range)
    """
    # Calculate the average log_returns for each regime
    regime_means = df.groupby('market_regime')['log_returns'].mean()

    # Identify the long (best average) and short (worst average) regimes
    long_regime = regime_means.idxmax()
    short_regime = regime_means.idxmin()

    # Create a mapping: long -> 1, short -> -1, others -> 0 (range)
    signal_map = {}
    for regime in regime_means.index:
        if regime == long_regime:
            signal_map[regime] = 1
        elif regime == short_regime:
            signal_map[regime] = -1
        else:
            signal_map[regime] = 0

    # Map the 'market_regime' to the 'signal'
    df['signal'] = df['market_regime'].map(signal_map)
    return df

class HMMv3(IStrategy):
    # Strategy parameters
    minimal_roi = {"0": 5000}
    stoploss = -0.95
    timeframe = '1d'
    startup_candle_count = 100
    can_short: bool = True
    process_only_new_candles: bool = True

    momentum_signal_delay = IntParameter(2, 100, default=10, space='buy', optimize=True)
    number_of_momentums_to_use = IntParameter(1, 20, default=1, space='buy', optimize=True)
    n_components = CategoricalParameter([2, 3, 4], default=2, space="buy", optimize=True)

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

    def convert_days_to_timeframe(self, days: int, timeframe: str) -> int:
        """
        Convert a number of days into an approximate number of candles based on the timeframe.
        """
        conversions = {
            "1d": days,
            "1h": days * 24,
            "12h": days * 2,         # 2 candles per day.
            "4h": days * 6,          # 6 candles per day.
            "15m": (days * 24 * 60) // 15,
            "5m": (days * 24 * 60) // 5,
            "1m": days * 24 * 60,
            "1w": days // 7,         # One candle per week.
        }
        result = conversions.get(timeframe)
        if result is None:
            raise ValueError("Invalid timeframe")
        return result

    def calculate_signal(self, dataframe: pd.DataFrame, pair) -> pd.DataFrame:
        """
        Generate a trading signal using a sliding window Gaussian Hidden Markov Model (HMM).

        The HMM is trained on daily BTC data up to 2021-01-01. For rows in the input dataframe 
        (which is expected to have a 'date' column) with dates after the training period,
        daily predictions are computed and merged back into the original dataframe.
        The predicted market regime is translated into a signal:
            1: bullish signal (when regime == 0)
            0: otherwise
            There is actually a bearish regime (regime == 2), but ignored since we do not short.
        """
        # --- Get daily BTC data and compute technical indicators ---
        coin = pair.split('/')[0]
        daily_data = get_data(coin)  # Assumed to return daily data with a tz-naive DatetimeIndex.
        daily_data['log_returns'] = np.log(daily_data['close'] / daily_data['close'].shift(1))
        for ii in range(self.number_of_momentums_to_use.value+1):
            daily_data[f'momentum_{ii}'] = np.log(daily_data['close'] / daily_data['close'].shift(self.momentum_signal_delay.value+ii))

        daily_data.dropna(inplace=True)

        # --- Define training period (using daily data) ---
        if 'BTC' in pair:
            data_train = daily_data.loc['2015-01-01':'2021-01-01'].copy()
        else:
            raise ValueError('This strategy is only for BTC.')

        # --- Train the HMM model ---
        np.random.seed(42)
        features = ['log_returns'] + [f'momentum_{ii}' for ii in range(self.number_of_momentums_to_use.value+1)]
        hmm_model = GaussianHMM(n_components=self.n_components.value, covariance_type="full", n_iter=10000, tol=1e-4, algorithm='map')
        hmm_model.fit(data_train[features].values)

        # --- Save the model to pickle if not already saved ---
        pickle_filename = os.path.join("user_data", "yf_data",f"hmm_model_{coin}_{self.momentum_signal_delay.value}_{self.number_of_momentums_to_use.value}.pkl")
        if not os.path.exists(pickle_filename):
            with open(pickle_filename, 'wb') as f:
                pickle.dump(hmm_model, f)
            logger.info("Saved HMM model to pickle file: %s", pickle_filename)
        else:
            logger.info("Pickle file already exists, not overwriting: %s", pickle_filename)

        # --- Prepare test data based on the input dataframe ---
        ft_data = dataframe.copy()
        ft_data['log_returns'] = np.log(ft_data['close'] / ft_data['close'].shift(1))
        for ii in range(self.number_of_momentums_to_use.value+1):
            ft_data[f'momentum_{ii}'] = np.log(ft_data['close'] / ft_data['close'].shift(self.momentum_signal_delay.value+ii))

        ft_data.dropna(inplace=True)

        # --- Predict market regimes on the daily test data ---
        market_regime_predictions = []
        for i in range(len(ft_data)):
            # Use rows up to and including the current row for prediction
            current_row_features = ft_data[features].iloc[:i+1].values
            current_prediction = hmm_model.predict(current_row_features)
            market_regime_predictions.append(current_prediction[-1])
        ft_data['market_regime'] = market_regime_predictions
        ft_data = assign_market_signals(ft_data)

        # --- Merge the computed signals back to the original dataframe using 'date' column ---
        copy_df = dataframe.copy()
        merged = copy_df.merge(ft_data[['date', 'signal']], on='date', how='left')
        dataframe['signal'] = merged['signal']

        with open('processed.txt', 'w') as f:
            f.write(ft_data.to_string(index=False))

        with open('output.txt', 'w') as f:
            f.write(dataframe.to_string(index=False))

        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Compute HMM-based signals and add them to the dataframe.
        """
        if self.timeframe != '1d':
            raise ValueError("timeframe must be 1d")
        return self.calculate_signal(dataframe, metadata['pair'])

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Set the entry signal based on the HMM signal.
        dataframe.loc[dataframe['signal'] == 1, 'enter_long'] = 1
        dataframe.loc[dataframe['signal'] == -1, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Set the exit signal based on the HMM signal.
        dataframe.loc[dataframe['signal'] == 0, 'exit_long'] = 1
        dataframe.loc[dataframe['signal'] == 0, 'exit_short'] = 1
        dataframe.loc[dataframe['signal'] == -1, 'exit_long'] = 1
        dataframe.loc[dataframe['signal'] == 1, 'exit_short'] = 1
        return dataframe

    class HyperOpt:
        @staticmethod
        def stoploss_space():
            return [SKDecimal(-0.95, -0.01, decimals=2, name='stoploss')]
