import os
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from typing import List
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
from hmmlearn.hmm import GaussianHMM

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.optimize.space import SKDecimal
from typing import Dict, Optional, Union, Tuple
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# Configure logging and warnings
logger = logging.getLogger(__name__)
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def download_btc_daily():
    try:
        # Set the full path for the output file
        output_file = os.path.join("user_data", "yf_data", "BTC_daily_data.csv")
        
        # Check if the file already exists
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping download.")
            return
            
        # Initialize the tvDatafeed instance
        tv = TvDatafeed()
        print("Downloading BTCUSD daily data...")
        
        # Get historical data
        data = tv.get_hist(symbol='BTCUSD',
                           exchange='CRYPTO',
                           interval=Interval.in_daily,
                           n_bars=10000
        )
        
        # Check if data was returned successfully
        if data is None or data.empty:
            print("No data was returned. Please check the symbol/exchange credentials.")
            return

        # Reset index to move 'datetime' from index to column
        data = data.reset_index()

        # Rename the index column from 'datetime' to 'date'
        data = data.rename(columns={'datetime': 'date'})

        # Remove 'symbol' column if it exists
        data = data.drop(columns=['symbol'], errors='ignore')

        # Save the processed data to CSV
        #print(data)
        data.to_csv(output_file, index=False)
        print(f"Data saved successfully to {output_file}")

    except Exception as e:
        print("An error occurred while downloading data:", str(e))

def get_data(coin) -> pd.DataFrame:
    """
    Get coin historical data from a local CSV file.

    """
    download_btc_daily()

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
    stoploss = -0.10
    timeframe = '1d'
    startup_candle_count = 1
    can_short = True
    process_only_new_candles = True

    momentum_signal_delay = IntParameter(7, 20, default=10, space='buy', optimize=True)

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

    def calculate_signal(self, dataframe: pd.DataFrame) -> pd.DataFrame:
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
        df_in = dataframe.copy()
        # --- Get daily BTC data and compute technical indicators ---
        daily_data = get_data('BTC')  # Assumed to return daily data with a tz-naive DatetimeIndex.
        daily_data['log_returns'] = np.log(daily_data['close'] / daily_data['close'].shift(1))
        daily_data['momentum'] = np.log(daily_data['close'] / daily_data['close'].shift(self.momentum_signal_delay.value))
        daily_data.fillna(-1, inplace=True)

        # --- Define training period (using daily data) ---
        data_train = daily_data.loc['2015-01-01':'2021-01-01'].copy()

        #print(data_train)

        # --- Train the HMM model ---
        np.random.seed(42)
        features = ['log_returns', 'momentum']
        hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=10000, tol=1e-4, algorithm='map')
        hmm_model.fit(data_train[features].values)

        # --- Save the model to pickle if not already saved ---
        pickle_filename = os.path.join("user_data", "yf_data",f"hmm_model_BTC.pkl")
        if not os.path.exists(pickle_filename):
            with open(pickle_filename, 'wb') as f:
                pickle.dump(hmm_model, f)
            logger.info("Saved HMM model to pickle file: %s", pickle_filename)
        else:
            logger.info("Pickle file already exists, not overwriting: %s", pickle_filename)

        # --- Prepare test data based on the input dataframe ---
        ft_data = dataframe.copy()
        ft_data['log_returns'] = np.log(ft_data['close'] / ft_data['close'].shift(1))
        ft_data['momentum'] = np.log(ft_data['close'] / ft_data['close'].shift(self.momentum_signal_delay.value))
        ft_data.fillna(-1, inplace=True)

        # --- Predict market regimes on the daily test data ---
        ft_data.loc[:, 'market_regime'] = hmm_model.predict(ft_data[features].values)
        ft_data = assign_market_signals(ft_data)

        # --- Merge the computed signals back to the original dataframe using 'date' column ---
        copy_df = dataframe.copy()
        merged = copy_df.merge(ft_data[['date', 'signal']], on='date', how='left')
        dataframe['signal'] = merged['signal']

        if len(df_in) != len(dataframe):
            raise ValueError('dataframe length should not have been changed.')

        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Compute HMM-based signals and add them to the dataframe.
        """
        if self.timeframe != '1d':
            raise ValueError("timeframe must be 1d")
        return self.calculate_signal(dataframe)

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Set the entry signal based on the HMM signal.
        dataframe.loc[dataframe['signal'] == 1, 'enter_long'] = 1
        dataframe.loc[dataframe['signal'] == -1, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Set the exit signal based on the HMM signal.
        dataframe.loc[dataframe['signal'] == 0, 'exit_long'] = 1
        dataframe.loc[dataframe['signal'] == 0, 'exit_short'] = 1
        return dataframe

    class HyperOpt:

        def stoploss_space():
            return [SKDecimal(-0.80, -0.30, decimals=2, name='stoploss')]
