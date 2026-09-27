"""
Bitcoin Intermarket Strategy with NASDAQ, S&P 500, M2, and DXY - Improved Version

This strategy is based on the intramarket differencing technique presented by Neurotrader in his
video. I would like to thank Neurotrader for sharing this algorithm and his insights into
intermarket analysis.

The original strategy compared ETH to BTC, while this extended version compares BTC to four
different markets: NASDAQ, S&P 500, M2 Money Supply, and the Dollar Index (DXY). This allows
for a more comprehensive analysis of BTC's behavior relative to traditional financial markets.

This strategy is implemented for use with the Freqtrade trading bot framework.
"""

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from warnings import simplefilter
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
from typing import Optional
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.persistence import Trade
# --------------------------------
# Add your lib to import here
import pandas_ta as ta
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime
import time
import warnings
import logging
import os
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Constants for data caching
DATA_CACHE_HOURS = 24  # How many hours to cache the external data
DEFAULT_DATA_DIR = "./user_data/tv_data"  # Default directory for data storage


def ensure_directory_exists(directory_path):
    """
    Ensure the directory exists, create it if it doesn't
    
    Parameters:
    -----------
    directory_path : str
        Path to directory
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info(f"Directory '{directory_path}' created successfully")
        except Exception as e:
            logger.error(f"Error creating directory '{directory_path}': {e}")
            raise

def threshold_revert_signal(ind_in, threshold):
    """
    Generate signal based on threshold crossings with reversion to zero
    
    Parameters:
    -----------
    ind_in : Series
        Input indicator values
    threshold : float
        Threshold for signal generation
        
    Returns:
    --------
    numpy.ndarray
        Signal values (-1, 0, or 1)
    """
    ind = ind_in.copy()
    ind = ind.reset_index(drop=True)
    # Initialize the signal array and current position.
    signal = np.zeros(len(ind))
    position = 0
    # Iterate over the indicator values to assign signals.
    for i in range(len(ind)):
        if ind[i] > threshold:
            position = 1
        if ind[i] < -threshold:
            position = -1
        # Revert to 0 when the indicator crosses back through zero.
        if position == 1 and ind[i] <= 0:
            position = 0
        if position == -1 and ind[i] >= 0:
            position = 0
        signal[i] = position
    return signal

def cmma(ohlc, lookback, atr_lookback):
    """
    Compute a normalized indicator: (close - moving average) / (ATR * sqrt(lookback))
    
    Parameters:
    -----------
    ohlc : DataFrame
        OHLC data with 'high', 'low', 'close' columns
    lookback : int
        Lookback period for the moving average
    atr_lookback : int
        Lookback period for the ATR calculation
        
    Returns:
    --------
    Series
        Normalized indicator values
    """
    # Compute ATR using pandas_ta with the given lookback length.
    atr = ta.atr(ohlc['high'], ohlc['low'], ohlc['close'], length=atr_lookback)
    # Compute the simple moving average of the close prices.
    ma = ohlc['close'].rolling(lookback, center=False, min_periods=1).mean()
    # Calculate the indicator: (close - moving average) normalized by (ATR * sqrt(lookback))
    ind = (ohlc['close'] - ma) / (atr * np.sqrt(lookback))
    return ind

def download_process_from_tradingview(symbol, exchange, interval, tf, n_bars, output_file, lookback, atr_lookback, data_dir=DEFAULT_DATA_DIR):
    """
    Download and process data from TradingView
    
    Parameters:
    -----------
    symbol : str
        Symbol to download
    exchange : str
        Exchange for the symbol
    interval : Interval
        TradingView interval
    tf : str
        Timeframe for the resulting data (timeframe used by the Freqtrade strategy)
    n_bars : int
        Number of bars to download
    output_file : str
        Path to save the downloaded data
    lookback : int
        Lookback period for the CMMA calculation
    atr_lookback : int
        Lookback period for the ATR calculation
    data_dir : str
        Directory to store data
        
    Returns:
    --------
    DataFrame
        Processed data with CMMA indicator
    """
    # Construct full output path
    full_output_path = os.path.join(data_dir, output_file)
    folder_path = os.path.dirname(full_output_path)
    
    # Check if the file exists and when it was last modified
    use_existing_file = False
    if os.path.exists(full_output_path):
        file_mod_time = os.path.getmtime(full_output_path)
        current_time = time.time()
        time_diff = current_time - file_mod_time
        
        # If file was modified less than DATA_CACHE_HOURS ago, use the existing file
        if time_diff < (DATA_CACHE_HOURS * 3600):  # Convert hours to seconds
            use_existing_file = True

    # Download fresh data if needed
    if not use_existing_file:
        # Initialize the tvDatafeed instance
        tv = TvDatafeed()
        # Get historical data
        data = tv.get_hist(symbol=symbol,
                          exchange=exchange,
                          interval=interval,
                          n_bars=n_bars, extended_session=True)
        #print(data.head())
        data.reset_index(drop=False, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime'] )
        data['datetime'] = data['datetime'].dt.tz_localize('UTC')
        data = data.rename(columns={'datetime': 'date'})
        # Save the data to CSV
        data.to_csv(full_output_path)
    
    # Process data regardless of source
    out_data = pd.read_csv(full_output_path) # gets loaded with a date column that is datetime UTC because format is like 2025-03-14 01:00:00+00:00
    out_data['cmma'] = cmma(out_data, lookback, atr_lookback)
    out_data.set_index('date', inplace=True)
    out_data.index = pd.to_datetime(out_data.index)
    
    # Keep only the cmma column
    out_data = out_data[['cmma']]
    
    # Create full index for the period and forward fill missing values
    full_index = pd.date_range(start=out_data.index.min(), end=out_data.index.max(), freq=tf)
    out_data = out_data.reindex(full_index).ffill()
    
    return out_data

def download_process_from_m2_weekly(tf, n_bars, output_file, lookback, atr_lookback, data_dir=DEFAULT_DATA_DIR):
    """
    Download and process M2 Money Supply data from FRED
    
    Parameters:
    -----------
    tf : str
        Timeframe for the resulting data
    n_bars : int
        Number of bars to download
    output_file : str
        Path to save the downloaded data
    lookback : int
        Lookback period for the CMMA calculation
    atr_lookback : int
        Lookback period for the ATR calculation
    data_dir : str
        Directory to store data
        
    Returns:
    --------
    DataFrame
        Processed M2 Money Supply data with CMMA indicator
    """
    # Calculate adjusted lookback periods for weekly data
    # Use timeframe_to_seconds to get seconds in the timeframe
    seconds_in_tf = timeframe_to_seconds(tf)
    seconds_in_week = 7 * 24 * 60 * 60
    
    # Calculate the ratio of timeframe to weekly
    tf_to_weekly_ratio = seconds_in_tf / seconds_in_week
    
    # Adjust lookback periods
    adjusted_lookback = max(4, int(lookback * tf_to_weekly_ratio))
    adjusted_atr_lookback = max(4, int(atr_lookback * tf_to_weekly_ratio))
    
    return download_process_from_tradingview(
        "WM2NS", "FRED", Interval.in_weekly, tf, n_bars, 
        output_file, adjusted_lookback, adjusted_atr_lookback, data_dir
    )

def download_process_nasdaq(tf, n_bars, output_file, lookback, atr_lookback, data_dir=DEFAULT_DATA_DIR):
    """
    Download and process NASDAQ data from TradingView
    
    Parameters:
    -----------
    tf : str
        Timeframe for the resulting data
    n_bars : int
        Number of bars to download
    output_file : str
        Path to save the downloaded data
    lookback : int
        Lookback period for the CMMA calculation
    atr_lookback : int
        Lookback period for the ATR calculation
    data_dir : str
        Directory to store data
        
    Returns:
    --------
    DataFrame
        Processed NASDAQ data with CMMA indicator
    """
    # Calculate adjusted lookback periods for daily data
    # Use timeframe_to_seconds to get seconds in the timeframe
    seconds_in_tf = timeframe_to_seconds(tf)
    seconds_in_day = 24 * 60 * 60
    
    # Calculate the ratio of timeframe to daily
    tf_to_daily_ratio = seconds_in_tf / seconds_in_day
    
    # Adjust lookback periods
    adjusted_lookback = max(4, int(lookback * tf_to_daily_ratio))
    adjusted_atr_lookback = max(4, int(atr_lookback * tf_to_daily_ratio))
    
    return download_process_from_tradingview(
        "NQ1!", "CME_MINI", Interval.in_daily, tf, n_bars, 
        output_file, adjusted_lookback, adjusted_atr_lookback, data_dir
    )

def download_process_sp500(tf, n_bars, output_file, lookback, atr_lookback, data_dir=DEFAULT_DATA_DIR):
    """
    Download and process S&P 500 data from TradingView
    
    Parameters:
    -----------
    tf : str
        Timeframe for the resulting data
    n_bars : int
        Number of bars to download
    output_file : str
        Path to save the downloaded data
    lookback : int
        Lookback period for the CMMA calculation
    atr_lookback : int
        Lookback period for the ATR calculation
    data_dir : str
        Directory to store data
        
    Returns:
    --------
    DataFrame
        Processed S&P 500 data with CMMA indicator
    """
    # Calculate adjusted lookback periods for daily data
    # Use timeframe_to_seconds to get seconds in the timeframe
    seconds_in_tf = timeframe_to_seconds(tf)
    seconds_in_day = 24 * 60 * 60
    
    # Calculate the ratio of timeframe to daily
    tf_to_daily_ratio = seconds_in_tf / seconds_in_day
    
    # Adjust lookback periods
    adjusted_lookback = max(4, int(lookback * tf_to_daily_ratio))
    adjusted_atr_lookback = max(4, int(atr_lookback * tf_to_daily_ratio))
    
    return download_process_from_tradingview(
        "ES1!", "CME_MINI", Interval.in_daily, tf, n_bars, 
        output_file, adjusted_lookback, adjusted_atr_lookback, data_dir
    )

def download_process_dxy(tf, n_bars, output_file, lookback, atr_lookback, data_dir=DEFAULT_DATA_DIR):
    """
    Download and process Dollar Index (DXY) data from TradingView
    
    Parameters:
    -----------
    tf : str
        Timeframe for the resulting data
    n_bars : int
        Number of bars to download
    output_file : str
        Path to save the downloaded data
    lookback : int
        Lookback period for the CMMA calculation
    atr_lookback : int
        Lookback period for the ATR calculation
    data_dir : str
        Directory to store data
        
    Returns:
    --------
    DataFrame
        Processed Dollar Index data with CMMA indicator
    """
    # Calculate adjusted lookback periods for daily data
    # Use timeframe_to_seconds to get seconds in the timeframe
    seconds_in_tf = timeframe_to_seconds(tf)
    seconds_in_day = 24 * 60 * 60
    
    # Calculate the ratio of timeframe to daily
    tf_to_daily_ratio = seconds_in_tf / seconds_in_day
    
    # Adjust lookback periods 
    adjusted_lookback = max(4, int(lookback * tf_to_daily_ratio))
    adjusted_atr_lookback = max(4, int(atr_lookback * tf_to_daily_ratio))
    
    return download_process_from_tradingview(
        "DX1!", "ICEUS", Interval.in_daily, tf, n_bars, 
        output_file, adjusted_lookback, adjusted_atr_lookback, data_dir
    )

def reindex_asof(df, new_index):
    """
    Aligns df to new_index by performing an asof merge.
    For each timestamp in new_index, the last available row in df 
    that is less than or equal to that timestamp is used.
    """
    df = df.sort_index()
    new_index_df = pd.DataFrame({'date': new_index})
    df.index.name = 'date'
    df = df.reset_index(drop=False)
    aligned = pd.merge_asof(new_index_df, df, on='date', direction='backward')
    aligned.set_index('date', inplace=True)
    return aligned

class QuatreMousquetaires(IStrategy):
    """
    Bitcoin Intermarket Strategy with NASDAQ, S&P 500, M2, and DXY ("Quatre Mousquetaires")
    
    This strategy compares BTC to four different markets (NASDAQ, S&P 500, M2 Money Supply, DXY)
    and generates trading signals based on the differences between them.
    
    WARNING: This strategy is primarily designed for daily timeframes.
    While efforts have been made to adapt it to other timeframes, optimal performance
    is expected when using the 1d timeframe.
    """
    INTERFACE_VERSION = 3

    max_entry_position_adjustment = 100  # Allow adjustments (for 25%, 50%, 75%, 100%)
    position_adjustment_enable: bool = True  # Enable position adjustment
    can_short: bool = False  # Disable shorting - long only
    use_custom_stoploss: bool = False
    process_only_new_candles = True

    debug_mode = False

    # Ignore ROI parameter
    minimal_roi = {
        "0": 500.00  # 50000% i.e., useless
    }

    # Stoploss designed for the strategy
    stoploss = -0.95

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy
    timeframe = '1d'
    
    # Number of candles the strategy requires before producing valid signals
    # This is increased to accommodate the indicators calculation requirements
    startup_candle_count: int = 205

    # Order type mapping
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # Strategy parameters
    lookback_btc = IntParameter(14, 100, default=53, space='buy', optimize=True)
    lookback_nasdaq = IntParameter(14, 100, default=53, space='buy', optimize=True)
    lookback_sp500 = IntParameter(14, 100, default=53, space='buy', optimize=True)
    lookback_m2 = IntParameter(14, 100, default=53, space='buy', optimize=True)
    lookback_dxy = IntParameter(14, 100, default=53, space='buy', optimize=True)

    atr_lookback = IntParameter(14, 200, default=168, space='buy', optimize=True)

    threshold_nasdaq = DecimalParameter(0.02, 1.00, decimals=2, default=0.52, space='buy', optimize=True)
    threshold_sp500 = DecimalParameter(0.02, 1.00, decimals=2, default=0.52, space='buy', optimize=True)
    threshold_m2 = DecimalParameter(0.02, 1.00, decimals=2, default=0.52, space='buy', optimize=True)
    threshold_dxy = DecimalParameter(0.02, 1.00, decimals=2, default=0.52, space='buy', optimize=True)

    flip = CategoricalParameter([-1, 1], default=1, space='buy', optimize=True)

    fraction_from_number_signals = CategoricalParameter([0.25, 0.5], default=0.25, space='buy', optimize=True)
    
    # Market enable/disable parameters
    use_nasdaq = BooleanParameter(default=True, space='sell', optimize=False)
    use_sp500 = BooleanParameter(default=True, space='sell', optimize=False)
    use_m2 = BooleanParameter(default=True, space='sell', optimize=False)
    use_dxy = BooleanParameter(default=True, space='sell', optimize=False)

    def bot_start(self, **kwargs) -> None:
        """
        Called at bot initialization and startup. Used for one-time tasks.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters
        """
        # Override data directory if provided in the config
        self.data_directory = self.config.get('user_data_dir', DEFAULT_DATA_DIR)
        self.data_directory = os.path.join(self.data_directory, 'tv_data')
        
        # Create data directory if it doesn't exist
        ensure_directory_exists(self.data_directory)
        
        # Log warning if not using daily timeframe
        if self.timeframe != '1d':
            raise ValueError(
                f"strategy is for daily timeframe (1d). "
                f"Current timeframe is set to {self.timeframe}."
            )

    def debug_write_dataframe(self, dataframe, pair, stage):
        """
        Write dataframe to CSV file for debugging
        
        Parameters:
        -----------
        dataframe : DataFrame
            Dataframe to write
        pair : str
            Pair name
        stage : str
            Stage name for the file (e.g., "initial", "processed", "final")
        """
        if not self.debug_mode:
            return
        
        if not self.config['runmode'].value == 'backtest':
            return
            
        try:
            # Create a safe filename from the pair
            safe_pair = pair.replace('/', '_')
            
            # Create the filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.data_directory, f"{safe_pair}_{stage}_{timestamp}.csv")
            
            # Save dataframe to CSV
            dataframe.to_csv(filename)
            
            logger.info(f"Debug data written to {filename}")
        except Exception as e:
            logger.error(f"Error writing debug data: {e}")

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators for the strategy
        
        Parameters:
        -----------
        df : DataFrame
            OHLC data for the current pair
        metadata : dict
            Pair metadata
            
        Returns:
        --------
        DataFrame
            OHLC data with added indicators
        """
        # Create a copy of the dataframe to avoid modifying the original
        data = df.copy()

        # Create a working copy with DatetimeIndex for calculations
        working_df = data.copy()
        
        # Calculate CMMA for BTC
        btc_cmma = cmma(working_df, self.lookback_btc.value, self.atr_lookback.value)
        working_df['cmma'] = btc_cmma

        working_df.set_index('date', inplace=True)
        
        # Download and process data for the four markets
        n_bars = 10000  # Large enough to be sure to cover the backtest period

        # NASDAQ data
        nasdaq_data = download_process_nasdaq(
            self.timeframe, n_bars, 
            'NASDAQ_daily_data.csv', 
            self.lookback_nasdaq.value, self.atr_lookback.value,
            self.data_directory
        )
        
        # S&P 500 data
        sp500_data = download_process_sp500(
            self.timeframe, n_bars, 
            'SP500_daily_data.csv', 
            self.lookback_sp500.value, self.atr_lookback.value,
            self.data_directory
        )
        
        # M2 Money Supply data
        m2_data = download_process_from_m2_weekly(
            self.timeframe, int(n_bars/7), 
            'M2_weekly_data.csv', 
            self.lookback_m2.value, self.atr_lookback.value,
            self.data_directory
        )
        
        # Dollar Index data
        dxy_data = download_process_dxy(
            self.timeframe, n_bars, 
            'DXY_daily_data.csv', 
            self.lookback_dxy.value, self.atr_lookback.value,
            self.data_directory
        )

        working_df.fillna(0, inplace=True)
        
        # Align data (reindex) with the Freqtrade dataframe with date as index (working_df, daily index)
        # shift 1 day to protect against potential forward looking and it should not affect the performance
        nasdaq_data_aligned = reindex_asof(nasdaq_data, working_df.index)
        nasdaq_data_aligned = nasdaq_data_aligned.shift(1)

        sp500_data_aligned = reindex_asof(sp500_data, working_df.index)
        sp500_data_aligned = sp500_data_aligned.shift(1)

        m2_data_aligned = reindex_asof(m2_data, working_df.index)
        m2_data_aligned = m2_data_aligned.shift(7)
        
        dxy_data_aligned = reindex_asof(dxy_data, working_df.index)
        dxy_data_aligned = dxy_data_aligned.shift(1)
        
        # Calculate intermarket differences and signals
        if self.use_nasdaq.value:
            nasdaq_diff = nasdaq_data_aligned['cmma'] - working_df['cmma']
            working_df['nasdaq_signal'] = self.flip.value * threshold_revert_signal(nasdaq_diff, self.threshold_nasdaq.value)
        else:
            working_df['nasdaq_signal'] = 0
        
        if self.use_sp500.value:
            sp500_diff = sp500_data_aligned['cmma'] - working_df['cmma']
            working_df['sp500_signal'] = self.flip.value * threshold_revert_signal(sp500_diff, self.threshold_sp500.value)
        else:
            working_df['sp500_signal'] = 0
        
        if self.use_m2.value:
            m2_diff = m2_data_aligned['cmma'] - working_df['cmma']
            working_df['m2_signal'] = self.flip.value * threshold_revert_signal(m2_diff, self.threshold_m2.value)
        else:
            working_df['m2_signal'] = 0

        if self.use_dxy.value:
            dxy_diff = dxy_data_aligned['cmma'] - working_df['cmma']
            working_df['dxy_signal'] = -1.0 * self.flip.value * threshold_revert_signal(dxy_diff, self.threshold_dxy.value)
        else:
            working_df['dxy_signal'] = 0
        
        # Count active long signals only
        working_df['active_long_signals'] = ((working_df['nasdaq_signal'] > 0).astype(int) + 
                                    (working_df['sp500_signal'] > 0).astype(int) + 
                                    (working_df['m2_signal'] > 0).astype(int) + 
                                    (working_df['dxy_signal'] > 0).astype(int))
        
        # Signal is active if we have any active long signals
        working_df['signal'] = np.zeros(len(working_df))
        working_df.loc[working_df['active_long_signals'] > 0, 'signal'] = 1

        data.set_index('date', inplace=True)
        
        # Copy all new columns from working dataframe to the original dataframe
        for col in working_df.columns:
            if col not in data.columns or col in ['cmma', 'nasdaq_signal', 'sp500_signal', 'm2_signal', 
                                            'dxy_signal', 'active_long_signals', 'signal']:
                data[col] = working_df[col]
        
        # Restore the original index type
        data.reset_index(drop=False, inplace=True)
        
        # Debug output
        self.debug_write_dataframe(data, metadata['pair'], "final_output")
        
        return data

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals
        
        Parameters:
        -----------
        dataframe : DataFrame
            OHLC data with indicators
        metadata : dict
            Pair metadata
            
        Returns:
        --------
        DataFrame
            OHLC data with entry signals
        """
        dataframe.loc[
            (dataframe['signal'] == 1),
            'enter_long'
        ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals
        
        Parameters:
        -----------
        dataframe : DataFrame
            OHLC data with indicators
        metadata : dict
            Pair metadata
            
        Returns:
        --------
        DataFrame
            OHLC data with exit signals
        """
        dataframe.loc[
            (dataframe['signal'] == 0),
            'exit_long'
        ] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str,
                           **kwargs) -> float:
        """
        Determine initial position size based on number of active signals.
        
        Parameters:
        -----------
        pair : str
            Current pair
        current_time : datetime
            Current date and time
        current_rate : float
            Current rate for the pair
        proposed_stake : float
            Stake amount proposed by the bot
        min_stake : float or None
            Minimum stake amount
        max_stake : float
            Maximum stake amount
        leverage : float
            Leverage being used
        entry_tag : str or None
            Entry tag for the trade
        side : str
            Side of the trade ('long' or 'short')
        **kwargs : 
            Other arguments
            
        Returns:
        --------
        float
            Stake amount to use
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"Empty dataframe for {pair}, using proposed stake")
            return proposed_stake
        
        if self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-2].squeeze()
        else:
            last_candle = dataframe.iloc[-1].squeeze()
            
        active_signals = last_candle['active_long_signals']

        position_pct = min(active_signals * self.fraction_from_number_signals.value, 1.0)
        
        # Calculate stake amount
        stake_amount = max_stake * position_pct
            
        return stake_amount
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, 
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Adjust position size based on number of active signals.
        
        Parameters:
        -----------
        trade : Trade
            Current trade object
        current_time : datetime
            Current date and time
        current_rate : float
            Current rate for the pair
        current_profit : float
            Current profit/loss ratio
        min_stake : float or None
            Minimum stake amount
        max_stake : float
            Maximum stake amount
        current_entry_rate : float
            Current entry rate
        current_exit_rate : float
            Current exit rate
        current_entry_profit : float
            Current entry profit
        current_exit_profit : float
            Current exit profit
        **kwargs : 
            Other arguments
            
        Returns:
        --------
        float or None
            Adjustment to stake amount (positive to increase, negative to decrease, None for no adjustment)
        """
        if not trade.is_open:
            return None
            
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"Empty dataframe for {trade.pair}, no position adjustment")
            return None
            
        if self.config['runmode'].value in ('live', 'dry_run'):
            last_candle = dataframe.iloc[-2].squeeze()
        else:
            last_candle = dataframe.iloc[-1].squeeze()
            
        active_signals = last_candle['active_long_signals']
        
        # Calculate target stake amount based on number of active signals
        position_pct = min(active_signals * self.fraction_from_number_signals.value, 1.0)
        target_stake = max_stake * position_pct
        
        # Calculate adjustment needed
        current_stake = trade.stake_amount
        adjustment = target_stake - current_stake

        if adjustment < min_stake*1.05:
            return None

        return adjustment