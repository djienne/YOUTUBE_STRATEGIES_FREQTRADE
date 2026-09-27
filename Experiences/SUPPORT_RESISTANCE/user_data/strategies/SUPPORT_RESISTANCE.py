import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative, Order)
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
import scipy
import logging
import warnings

# Suppress common warnings that don't affect strategy performance
warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# based on original idea from Neurotrader: https://www.youtube.com/watch?v=mNWPSFOVoYA

def find_levels(price: np.array, atr: float, 
                first_w: float = 0.1, 
                atr_mult: float = 3.0, 
                prom_thresh: float = 0.25):
    """
    Identify support and resistance levels using weighted kernel density estimation.
    
    This function uses a Gaussian kernel to create a market profile from historical
    price data, with more recent prices weighted higher. Significant peaks in the 
    density represent potential support/resistance levels.
    
    Args:
        price (np.array): Array of log closing prices
        atr (float): Average True Range (log scale) for bandwidth scaling
        first_w (float): Weight for oldest price data (default: 0.1)
        atr_mult (float): Multiplier for ATR bandwidth scaling (default: 3.0)
        prom_thresh (float): Minimum prominence threshold for peak detection (default: 0.25)
    
    Returns:
        tuple: (levels, peaks, props, price_range, pdf, weights)
            - levels: List of identified support/resistance levels
            - peaks: Peak indices in the density function
            - props: Peak properties from scipy.signal.find_peaks
            - price_range: Price range used for density calculation
            - pdf: Probability density function values
            - weights: Weight array used for recent price emphasis
    """
    # Create linearly increasing weights (recent prices weighted more heavily)
    last_w = 1.0
    weight_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * weight_step
    weights[weights < 0] = 0.0

    # Generate kernel density estimation with ATR-scaled bandwidth
    kernel = scipy.stats.gaussian_kde(price, bw_method=atr * atr_mult, weights=weights)

    # Create market profile over price range
    min_price = np.min(price)
    max_price = np.max(price)
    price_step = (max_price - min_price) / 200
    price_range = np.arange(min_price, max_price, price_step)
    pdf = kernel(price_range)  # Market profile density

    # Identify significant peaks representing support/resistance levels
    pdf_max = np.max(pdf)
    prominence_minimum = pdf_max * prom_thresh

    peaks, peak_properties = scipy.signal.find_peaks(pdf, prominence=prominence_minimum)
    
    # Convert log prices back to actual price levels
    levels = []
    for peak in peaks:
        levels.append(np.exp(price_range[peak]))

    return levels, peaks, peak_properties, price_range, pdf, weights


def support_resistance_levels(data_in: pd.DataFrame, lookback: int, 
                             first_w: float = 0.01, 
                             atr_mult: float = 3.0, 
                             prom_thresh: float = 0.25):
    """
    Calculate support and resistance levels for each candle in the dataset.
    
    For each candle, this function analyzes the previous 'lookback' candles
    to identify current support and resistance levels using kernel density estimation.
    
    Args:
        data_in (pd.DataFrame): OHLC price data
        lookback (int): Number of previous candles to analyze
        first_w (float): Weight for oldest price data
        atr_mult (float): ATR multiplier for bandwidth scaling
        prom_thresh (float): Prominence threshold for peak detection
    
    Returns:
        list: List of support/resistance levels for each candle (None for insufficient data)
    """
    data = data_in.copy()
    
    # Calculate Average True Range on log scale for bandwidth estimation
    log_high = np.log(data['high'])
    log_low = np.log(data['low'])
    log_close = np.log(data['close'])
    atr = ta.atr(log_high, log_low, log_close, lookback)

    # Initialize levels array
    all_levels = [None] * len(data)
    
    # Calculate levels for each candle with sufficient lookback data
    for i in range(lookback, len(data)):
        start_index = i - lookback
        # Extract log closing prices for the lookback period
        log_closes = np.log(data.iloc[start_index + 1: i + 1]['close'].to_numpy())
        
        levels, _, _, _, _, _ = find_levels(
            log_closes, atr.iloc[i], first_w, atr_mult, prom_thresh
        )
        all_levels[i] = levels
        
    return all_levels


def sr_penetration_signal(data_in: pd.DataFrame, levels: list):
    """
    Generate trading signals based on support/resistance level penetrations.
    
    Creates buy signals when price closes above a resistance level and
    sell signals when price closes below a support level.
    
    Args:
        data_in (pd.DataFrame): OHLC price data
        levels (list): Support/resistance levels for each candle
    
    Returns:
        pd.DataFrame: Original data with added 'signal' column
            - 1.0: Buy signal (resistance breakout)
            - -1.0: Sell signal (support breakdown)
            - 0.0: No signal
    """
    data = data_in.copy()
    signal = np.zeros(len(data))
    current_signal = 0.0
    close_prices = data['close'].to_numpy()
    
    # Analyze each candle for level penetrations
    for i in range(1, len(data)):
        if levels[i] is None:
            continue

        previous_close = close_prices[i - 1]
        current_close = close_prices[i]
        
        # Check each support/resistance level for penetrations
        for level in levels[i]:
            # Resistance breakout (bullish signal)
            if current_close > level and previous_close <= level:
                current_signal = 1.0
            # Support breakdown (bearish signal)
            elif current_close < level and previous_close >= level:
                current_signal = -1.0

        signal[i] = current_signal

    # Add signal to output dataframe
    data_out = data_in.copy()
    data_out['signal'] = signal
    return data_out


class SUPPORT_RESISTANCE(IStrategy):
    """
    Support and Resistance Trading Strategy
    
    This strategy identifies dynamic support and resistance levels using kernel
    density estimation and generates trading signals when price penetrates these levels.
    
    Key Features:
    - Uses weighted kernel density to identify key price levels
    - ATR-based bandwidth scaling for adaptive level detection
    - Long-only strategy focusing on resistance breakouts
    - Exit signals on support breakdowns
    """
    
    # Strategy configuration
    minimal_roi = {"0": 5000}  # Hold positions indefinitely (5000% ROI target)
    stoploss = -0.85  # very large, never hit
    timeframe = '12h'  # 12-hour candles
    startup_candle_count: int = 370  # Minimum candles needed before trading
    can_short: bool = False  # Long-only strategy
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = True

    # Optimization parameters
    first_w = DecimalParameter(
        1.0, 2.0, decimals=1, default=1.6, space="buy", optimize=True,
        load=True
    )
    atr_mult = DecimalParameter(
        2.0, 4.0, decimals=1, default=3.2, space="buy", optimize=True,
        load=True
    )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate all technical indicators and support/resistance levels.
        
        Args:
            dataframe (pd.DataFrame): Raw OHLC data
            metadata (dict): Pair metadata
            
        Returns:
            pd.DataFrame: Dataframe with indicators and trading signals
        """
        # Calculate support/resistance levels using 365-candle lookback
        levels = support_resistance_levels(
            data_in=dataframe, 
            lookback=365, 
            first_w=self.first_w.value, 
            atr_mult=self.atr_mult.value
        )

        # Generate penetration signals based on calculated levels
        dataframe = sr_penetration_signal(data_in=dataframe, levels=levels)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define entry conditions for long positions.
        
        Entry signal: Resistance level breakout (signal == 1)
        
        Args:
            dataframe (pd.DataFrame): Dataframe with indicators
            metadata (dict): Pair metadata
            
        Returns:
            pd.DataFrame: Dataframe with entry signals
        """
        dataframe.loc[
            (dataframe['signal'] == 1),
            'enter_long'
        ] = 1
        
        # Short entries disabled for this long-only strategy
        # dataframe.loc[
        #     (dataframe['signal'] == -1),
        #     'enter_short'
        # ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define exit conditions for positions.
        
        Exit signal: Support level breakdown (signal == -1)
        
        Args:
            dataframe (pd.DataFrame): Dataframe with indicators
            metadata (dict): Pair metadata
            
        Returns:
            pd.DataFrame: Dataframe with exit signals
        """
        dataframe.loc[
            (dataframe['signal'] == -1),
            'exit_long'
        ] = 1
        
        # Short exits disabled for this long-only strategy
        # dataframe.loc[
        #     (dataframe['signal'] == 0),
        #     'exit_short'
        # ] = 1
        
        return dataframe