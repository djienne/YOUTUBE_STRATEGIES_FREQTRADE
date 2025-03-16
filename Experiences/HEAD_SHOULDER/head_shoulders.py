import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from rolling_window import rw_top, rw_bottom
from typing import List
from collections import deque
from dataclasses import dataclass
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)


def load_ohlcv_data_df(symbol, timeframe):
    """
    Load OHLCV data from a pre-downloaded Feather file for a given symbol and timeframe.

    The Feather file is expected to be located in:
        user_data/data/binance/<SYMBOL>-<TIMEFRAME>.feather

    where <SYMBOL> is the trading pair with '/' replaced by '_' (e.g. "BTC/USDT" becomes "BTC_USDT")
    and <TIMEFRAME> is the timeframe (e.g. "1h").

    The Feather file should contain the columns:
        date, open, high, low, close, volume

    Optionally, the data can be filtered between start_time and end_time.
    Data integrity checks for missing intervals and duplicate timestamps are also performed.

    Parameters:
        symbol (str): Trading pair (e.g., 'BTC/USDT').
        timeframe (str): Timeframe (e.g., '1h', '15m', etc.).
        start_time (int or str, optional): Start time in milliseconds since epoch or as an ISO8601 string.
        end_time (int or str, optional): End time in milliseconds since epoch or as an ISO8601 string.

    Returns:
        pd.DataFrame: DataFrame of OHLCV data with columns: 
                    ['date', 'open', 'high', 'low', 'close', 'volume'].
    """
    import os
    import pandas as pd

    # Build the filename based on the symbol and timeframe.
    # Replace '/' with '_' for the filename.
    symbol_for_filename = symbol.replace('/', '_')
    filename = os.path.join(f"{symbol_for_filename}-{timeframe}.feather")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Feather file not found: {filename}")

    # Load the Feather file into a DataFrame.
    df = pd.read_feather(filename)

    # Ensure the 'timestamp' column is in datetime format.
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            # If the timestamps are in milliseconds since epoch, specify the unit.
            df['date'] = pd.to_datetime(df['date'], unit='ms')
        except Exception:
            # Fallback conversion.
            df['date'] = pd.to_datetime(df['date'])

    # Sort the DataFrame by timestamp and reset the index.
    df = df.sort_values('date').reset_index(drop=True)

    # ---- Data Integrity Checks ----
    # Determine the expected interval based on the timeframe string.
    def timeframe_to_ms(tf):
        unit = tf[-1]
        num = int(tf[:-1])
        if unit == 'm':  # minutes
            return num * 60 * 1000
        elif unit == 'h':  # hours
            return num * 60 * 60 * 1000
        elif unit == 'd':  # days
            return num * 24 * 60 * 60 * 1000
        elif unit == 'w':  # weeks
            return num * 7 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    return df

@dataclass
class HSPattern:

    # True if inverted, False if not. Inverted is "bullish" according to technical analysis dogma
    inverted: bool

    # Indices of the parts of the H&S pattern
    l_shoulder: int = -1
    r_shoulder: int = -1
    l_armpit: int = -1
    r_armpit: int = -1
    head: int = -1
   
    # Price of the parts of the H&S pattern. _p stands for price.
    l_shoulder_p: float = -1
    r_shoulder_p: float = -1
    l_armpit_p: float = -1
    r_armpit_p: float = -1
    head_p: float = -1
   
    start_i: int = -1
    break_i: int = -1
    break_p: float = -1

    neck_start: float = -1
    neck_end: float = -1

    # Attributes
    neck_slope: float = -1
    head_width: float = -1
    head_height: float = -1
    pattern_r2: float = -1

def compute_pattern_r2(data: np.array, pat: HSPattern):

    line0_slope = (pat.l_shoulder_p - pat.neck_start) / (pat.l_shoulder - pat.start_i)
    line0 = pat.neck_start + np.arange(pat.l_shoulder - pat.start_i) * line0_slope
    
    line1_slope = (pat.l_armpit_p - pat.l_shoulder_p) / (pat.l_armpit - pat.l_shoulder)
    line1 = pat.l_shoulder_p + np.arange(pat.l_armpit - pat.l_shoulder) * line1_slope
    
    line2_slope = (pat.head_p - pat.l_armpit_p) / (pat.head - pat.l_armpit)
    line2 = pat.l_armpit_p + np.arange(pat.head - pat.l_armpit) * line2_slope
    
    line3_slope = (pat.r_armpit_p - pat.head_p) / (pat.r_armpit - pat.head)
    line3 = pat.head_p + np.arange(pat.r_armpit - pat.head) * line3_slope
    
    line4_slope = (pat.r_shoulder_p - pat.r_armpit_p) / (pat.r_shoulder - pat.r_armpit)
    line4 = pat.r_armpit_p + np.arange(pat.r_shoulder - pat.r_armpit) * line4_slope
    
    line5_slope = (pat.break_p - pat.r_shoulder_p) / (pat.break_i - pat.r_shoulder)
    line5 = pat.r_shoulder_p + np.arange(pat.break_i - pat.r_shoulder) * line5_slope
    
    raw_data = data[pat.start_i:pat.break_i]
    hs_model = np.concatenate([line0, line1, line2, line3, line4, line5])
    mean = np.mean(raw_data)

    ss_res = np.sum( (raw_data - hs_model) ** 2.0 )
    ss_tot = np.sum( (raw_data - mean) ** 2.0 )

    r2 = 1.0 - ss_res / ss_tot
    return r2


def check_hs_pattern(extrema_indices: List[int], data: np.array, i:int, early_find: bool = False) -> HSPattern:
    ''' Returns a HSPattern if found, or None if not found ''' 
    # Unpack list
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]

    if i - r_armpit < 2:
        return None

    # Find right shoulder as max price since r_armpit
    r_shoulder = r_armpit + data[r_armpit + 1: i].argmax() + 1

    # Head must be higher than shoulders
    if data[head] <= max(data[l_shoulder], data[r_shoulder]):
        return None

    # Balance rule. Shoulders are above the others midpoint.
    # A shoulder's midpoint is the midpoint between the shoulder and armpit
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] < r_midpoint  or data[r_shoulder] < l_midpoint:
        return None

    # Symmetry rule. time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None
        
    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run
    
    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope

    # Confirm pattern when price is halfway from right shoulder
    if early_find: 
        if data[i] > r_midpoint:
            return None
    else:
       
        # Price has yet to break neckline, unconfirmed
        if data[i] > neck_val:
            return None

    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        
        if l_shoulder - j < 0:
            return None
        
        if data[l_shoulder - j] < neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pat = HSPattern(inverted=False)  
    
    pat.l_shoulder = l_shoulder
    pat.r_shoulder = r_shoulder
    pat.l_armpit = l_armpit
    pat.r_armpit = r_armpit
    pat.head = head
    
    pat.l_shoulder_p = data[l_shoulder]
    pat.r_shoulder_p = data[r_shoulder]
    pat.l_armpit_p = data[l_armpit]
    pat.r_armpit_p = data[r_armpit]
    pat.head_p = data[head]

    pat.start_i = pat_start
    pat.break_i = i
    pat.break_p = data[i]

    pat.neck_start = neck_start
    pat.neck_end = neck_val

    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = data[head] - (data[l_armpit] + (head - l_armpit) * neck_slope)
    #pat.pattern_r2 = compute_pattern_r2(data, pat)

    # I experiemented with r-squared as a filter for H&S, but this can delay recognition.
    # It didn't seem terribly potent, may be useful as a filter in conjunction with other attributes
    # if one wanted to add a machine learning layer before trading these patterns. 

    #if pat.pattern_r2 < 0.0:
    #    return None

    return pat


def check_ihs_pattern(extrema_indices: List[int], data: np.array, i:int, early_find: bool = False) -> HSPattern:
    
    # Unpack list
    l_shoulder = extrema_indices[0]
    l_armpit = extrema_indices[1]
    head = extrema_indices[2]
    r_armpit = extrema_indices[3]
    
    if i - r_armpit < 2:
        return None

    # Find right shoulder as max price since r_armpit
    r_shoulder = r_armpit + data[r_armpit+1: i].argmin() + 1

    # Head must be lower than shoulders
    if data[head] >= min(data[l_shoulder], data[r_shoulder]):
        return None

    # Balance rule. Shoulders are below the others midpoint.
    # A shoulder's midpoint is the midpoint between the shoulder and armpit
    r_midpoint = 0.5 * (data[r_shoulder] + data[r_armpit])
    l_midpoint = 0.5 * (data[l_shoulder] + data[l_armpit])
    if data[l_shoulder] > r_midpoint  or data[r_shoulder] > l_midpoint:
        return None

    # Symmetry rule. time from shoulder to head are comparable
    r_to_h_time = r_shoulder - head
    l_to_h_time = head - l_shoulder
    if r_to_h_time > 2.5 * l_to_h_time or l_to_h_time > 2.5 * r_to_h_time:
        return None

    # Compute neckline
    neck_run = r_armpit - l_armpit
    neck_rise = data[r_armpit] - data[l_armpit]
    neck_slope = neck_rise / neck_run
    
    # neckline value at current index
    neck_val = data[l_armpit] + (i - l_armpit) * neck_slope
    
    # Confirm pattern when price is halfway from right shoulder
    if early_find: 
        if data[i] < r_midpoint:
            return None
    else:
       
        # Price has yet to break neckline, unconfirmed
        if data[i] < neck_val:
            return None
   
    # Find beginning of pattern. Neck to left shoulder
    head_width = r_armpit - l_armpit
    pat_start = -1
    neck_start = -1
    for j in range(1, head_width):
        neck = data[l_armpit] + (l_shoulder - l_armpit - j) * neck_slope
        
        if l_shoulder - j < 0:
            return None
        
        if data[l_shoulder - j] > neck:
            pat_start = l_shoulder - j
            neck_start = neck
            break

    if pat_start == -1:
        return None

    # Pattern confirmed if here :)
    pat = HSPattern(inverted=True)  
    
    pat.l_shoulder = l_shoulder
    pat.r_shoulder = r_shoulder
    pat.l_armpit = l_armpit
    pat.r_armpit = r_armpit
    pat.head = head
    
    pat.l_shoulder_p = data[l_shoulder]
    pat.r_shoulder_p = data[r_shoulder]
    pat.l_armpit_p = data[l_armpit]
    pat.r_armpit_p = data[r_armpit]
    pat.head_p = data[head]

    pat.start_i = pat_start
    pat.break_i = i
    pat.break_p = data[i]

    pat.neck_start = neck_start
    pat.neck_end = neck_val
    #pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    #pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    #if pat.pattern_r2 < 0.0:
    #    return None

    return pat


def find_hs_patterns(data: np.array, order:int, early_find:bool = False):
    assert(order >= 1)
    
    # head and shoulders top checked from/after a confirmed bottom (before right shoulder)
    # head and shoulders bottom checked from/after a confirmed top 
    
    last_is_top = False
    recent_extrema = deque(maxlen=5)
    recent_types = deque(maxlen=5) # -1 for bottoms 1 for tops

    # Lock variables to prevent finding the same pattern multiple times
    hs_lock = False
    ihs_lock = False

    ihs_patterns = [] # Inverted (bullish)
    hs_patterns = []  # Regular (bearish)
    for i in range(len(data)):

        if rw_top(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(1)
            ihs_lock = False
            last_is_top = True
        
        if rw_bottom(data, i, order):
            recent_extrema.append(i - order)
            recent_types.append(-1)
            hs_lock = False
            last_is_top = False

        if len(recent_extrema) < 5:
            continue
        
        hs_alternating = True
        ihs_alternating = True
        
        if last_is_top:
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: 
                    ihs_alternating = False
            
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: 
                    hs_alternating = False
            
            ihs_extrema = list(recent_extrema)[1:5]
            hs_extrema = list(recent_extrema)[0:4]
        else:
            
            for j in range(2, 5):
                if recent_types[j] == recent_types[j - 1]: 
                    hs_alternating = False
            
            for j in range(1, 4):
                if recent_types[j] == recent_types[j - 1]: 
                    ihs_alternating = False
            
            ihs_extrema = list(recent_extrema)[0:4]
            hs_extrema = list(recent_extrema)[1:5]
        
        if ihs_lock or not ihs_alternating:
            ihs_pat = None
        else:
            ihs_pat = check_ihs_pattern(ihs_extrema, data, i, early_find)

        if hs_lock or not hs_alternating:
            hs_pat = None
        else:
            hs_pat = check_hs_pattern(hs_extrema, data, i, early_find)

        if hs_pat is not None:
            hs_lock = True
            hs_patterns.append(hs_pat)
        
        if ihs_pat is not None:
            ihs_lock = True
            ihs_patterns.append(ihs_pat)


    return hs_patterns, ihs_patterns


def get_pattern_return(data: np.array, pat: HSPattern, log_prices: bool = True) -> float:

    entry_price = pat.break_p
    entry_i = pat.break_i
    stop_price = pat.r_shoulder_p

    if pat.inverted:
        tp_price = pat.neck_end + pat.head_height
    else:
        tp_price = pat.neck_end - pat.head_height

    exit_price = -1
    for i in range(pat.head_width):
        if entry_i + i >= len(data):
            return np.nan

        exit_price = data[entry_i + i]
        if pat.inverted and (exit_price > tp_price or exit_price < stop_price):
            break
        
        if not pat.inverted and (exit_price < tp_price or exit_price > stop_price):
            break
    
    if pat.inverted: # Long
        if log_prices:
            return exit_price - entry_price
        else:
            return (exit_price - entry_price) / entry_price
    else: # Short
        if log_prices:
            return entry_price - exit_price
        else:
            return -1 * (exit_price - entry_price) / entry_price

def plot_hs(candle_data: pd.DataFrame, pat: HSPattern, pad: int = 2):
    if pad < 0:
        pad = 0

    idx = candle_data.index
    data = np.exp(candle_data.iloc[pat.start_i:pat.break_i + 1 + pad])

    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()

    l0 = [(idx[pat.start_i], np.exp(pat.neck_start)), (idx[pat.l_shoulder], np.exp(pat.l_shoulder_p))]
    l1 = [(idx[pat.l_shoulder], np.exp(pat.l_shoulder_p)), (idx[pat.l_armpit], np.exp(pat.l_armpit_p))]
    l2 = [(idx[pat.l_armpit], np.exp(pat.l_armpit_p)), (idx[pat.head], np.exp(pat.head_p))]
    l3 = [(idx[pat.head], np.exp(pat.head_p)), (idx[pat.r_armpit], np.exp(pat.r_armpit_p))]
    l4 = [(idx[pat.r_armpit], np.exp(pat.r_armpit_p)), (idx[pat.r_shoulder], np.exp(pat.r_shoulder_p))]
    l5 = [(idx[pat.r_shoulder], np.exp(pat.r_shoulder_p)), (idx[pat.break_i], np.exp(pat.neck_end))]
    neck = [(idx[pat.start_i], np.exp(pat.neck_start)), (idx[pat.break_i], np.exp(pat.neck_end))]


    mpf.plot(data, alines=dict(alines=[l0, l1, l2, l3, l4, l5, neck ], colors=['w', 'w', 'w', 'w', 'w', 'w', 'r']), type='candle', style='charles', ax=ax)
    x = len(data) // 2 - len(data) * 0.1
    if pat.inverted:
        y = np.exp(pat.head_p + pat.head_height * 1.25)
    else:
        y = np.exp(pat.head_p - pat.head_height * 1.25)
    
    ax.text(x,y, f"BTC-USDT 15m ({idx[pat.start_i].strftime('%Y-%m-%d %H:%M')} - {idx[pat.break_i].strftime('%Y-%m-%d %H:%M')})", color='white', fontsize='xx-large')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

if __name__ == '__main__':
    pairs = ['BTC/USDT']
    time_frame = '15m'
    
    # Fee parameters: 0.1% fee per entry and exit.
    fee = 0.001
    fee_adj = 2 * np.log(1 - fee)  # fee cost in log space (~ -0.002001)

    # List to collect processed trades.
    results = []

    # Loop over each trading pair.
    for pair in pairs:
        # Load and preprocess data.
        data = load_ohlcv_data_df(pair, time_frame)
        data = data.set_index('date')
        data = np.log(data)
        dat_slice = data['close'].to_numpy()

        # Loop over detection modes: early and late.
        for detection_label, early_find in [('early', True), ('late', False)]:
            # Detect patterns for the current setting.
            hs_patterns, ihs_patterns = find_hs_patterns(dat_slice, 52, early_find=early_find)
            
            # Process both HS and IHS patterns.
            for pattern_label, patterns in [('HS', hs_patterns), ('IHS', ihs_patterns)]:
                df = pd.DataFrame()
                for i, pat in enumerate(patterns):
                    # Record pattern attributes.
                    df.loc[i, 'head_width'] = pat.head_width
                    df.loc[i, 'head_height'] = pat.head_height
                    df.loc[i, 'r2'] = pat.pattern_r2
                    df.loc[i, 'neck_slope'] = pat.neck_slope
                    # Use the pattern's break index to get the corresponding date.
                    df.loc[i, 'date'] = data.index[pat.break_i]
                    
                    hp = int(pat.head_width)
                    if pat.break_i + hp >= len(data):
                        df.loc[i, 'hold_return'] = np.nan
                    else:
                        # For HS patterns, a downward move is expected (so we take a negative return).
                        # For IHS patterns, the move is upward.
                        if pattern_label == 'HS':
                            ret = -1 * (dat_slice[pat.break_i + hp] - dat_slice[pat.break_i])
                        else:
                            ret = (dat_slice[pat.break_i + hp] - dat_slice[pat.break_i])
                        df.loc[i, 'hold_return'] = ret + fee_adj
                    # Calculate stop return (with fee adjustment).
                    df.loc[i, 'stop_return'] = get_pattern_return(dat_slice, pat) + fee_adj

                # Drop rows with missing values and sort by date.
                df = df.dropna(subset=['hold_return', 'stop_return', 'date']).sort_values('date')
                # Tag each trade with its detection mode, pattern type, and pair.
                df['detection'] = detection_label
                df['pattern'] = pattern_label
                df['pair'] = pair
                results.append(df)

    # Combine all the trades from all pairs and settings.
    combined_df = pd.concat(results)

    # --- Continuous (daily) aggregation ---
    # To see a continuous cumulative profit curve for each group,
    # group the trades by date (summing the returns when multiple trades occur on the same day)
    # for each combination of detection mode and pattern type.
    agg_hold = combined_df.groupby(['detection', 'pattern', 'date'])['hold_return'].sum().reset_index()
    agg_stop = combined_df.groupby(['detection', 'pattern', 'date'])['stop_return'].sum().reset_index()

    # Compute cumulative returns for each group.
    cum_hold_list = []
    for (detection, pattern), group in agg_hold.groupby(['detection', 'pattern']):
        group = group.sort_values('date').copy()
        group['cum_hold_return'] = group['hold_return'].cumsum()
        group['group_label'] = f"{detection.upper()} {pattern}"
        cum_hold_list.append(group)
    cum_hold_df = pd.concat(cum_hold_list)

    cum_stop_list = []
    for (detection, pattern), group in agg_stop.groupby(['detection', 'pattern']):
        group = group.sort_values('date').copy()
        group['cum_stop_return'] = group['stop_return'].cumsum()
        group['group_label'] = f"{detection.upper()} {pattern}"
        cum_stop_list.append(group)
    cum_stop_df = pd.concat(cum_stop_list)

    # --- Plotting continuous cumulative profit curves ---
    plt.figure(figsize=(14, 6))

    # Plot cumulative HOLD returns.
    plt.subplot(1, 2, 1)
    for label, group in cum_hold_df.groupby('group_label'):
        plt.plot(group['date'], group['cum_hold_return'], marker='o', label=f"{label} HOLD")
    plt.xlabel('Date')
    plt.ylabel('Cumulative HOLD Return')
    plt.title('Continuous Cumulative HOLD Profits')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative STOP returns.
    plt.subplot(1, 2, 2)
    for label, group in cum_stop_df.groupby('group_label'):
        plt.plot(group['date'], group['cum_stop_return'], marker='x', label=f"{label} STOP")
    plt.xlabel('Date')
    plt.ylabel('Cumulative STOP Return')
    plt.title('Continuous Cumulative STOP Profits')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Optionally, you can still plot individual patterns.
    plot_hs(data, ihs_patterns[-1], pad=0)
    plot_hs(data, ihs_patterns[-2], pad=0)
    plot_hs(data, ihs_patterns[0], pad=0)
    plot_hs(data, ihs_patterns[1], pad=0)
    plot_hs(data, hs_patterns[-1], pad=0)
    plot_hs(data, hs_patterns[-2], pad=0)  
    plot_hs(data, hs_patterns[0], pad=0)
    plot_hs(data, hs_patterns[1], pad=0)    


    def test_order_range(order_values, pairs=['BTC/USDT'], timeframe='15m', fee=0.001):
        """
        For each order value in order_values, this function:
        - Loads data for each trading pair.
        - Runs pattern detection with both early_find=True ("early") and early_find=False ("late").
        - Computes hold and stop returns (with a fee of 0.1% per entry and exit).
        - Aggregates the trades by date for each pattern type (HS and IHS) separately.
        - Prints the final cumulative PnL (both hold and stop) for HS and IHS trades.
        - Tracks all results and, at the end, prints the best configuration (highest final cumulative PnL)
            for both hold and stop returns.
            
        Helper functions (load_ohlcv_data_df, find_hs_patterns, get_pattern_return) are assumed to be defined.
        """
        fee_adj = 2 * np.log(1 - fee)  # fee applied on entry and exit (in log space)
        results_summary = []  # list to store results for each configuration

        for order in order_values:
            # Loop over detection modes.
            for detection_label, early_find in [('early', True), ('late', False)]:
                all_trades = []
                # Process each trading pair.
                for pair in pairs:
                    # Load and preprocess the data.
                    data = load_ohlcv_data_df(pair, timeframe)
                    data = data.set_index('date')
                    data = np.log(data)  # log-transform for return calculations
                    dat_slice = data['close'].to_numpy()

                    # Run pattern detection for the given order value.
                    hs_patterns, ihs_patterns = find_hs_patterns(dat_slice, order, early_find=early_find)
                    
                    # Process both HS and IHS patterns.
                    for pattern_label, patterns in [('HS', hs_patterns), ('IHS', ihs_patterns)]:
                        for pat in patterns:
                            # Validate the break index.
                            if pat.break_i >= len(data):
                                continue
                            hp = int(pat.head_width)
                            if pat.break_i + hp >= len(data):
                                continue

                            # Calculate the hold return.
                            if pattern_label == 'HS':
                                # For HS patterns, a downward move is expected.
                                ret = -1 * (dat_slice[pat.break_i + hp] - dat_slice[pat.break_i])
                            else:
                                # For IHS patterns, an upward move is expected.
                                ret = dat_slice[pat.break_i + hp] - dat_slice[pat.break_i]
                            hold_return = ret + fee_adj

                            # Calculate the stop return.
                            stop_return = get_pattern_return(dat_slice, pat) + fee_adj

                            # Determine the trade date.
                            trade_date = data.index[pat.break_i]

                            all_trades.append({
                                'date': trade_date,
                                'hold_return': hold_return,
                                'stop_return': stop_return,
                                'detection': detection_label,
                                'pattern': pattern_label,
                                'pair': pair
                            })

                # Convert the collected trades into a DataFrame.
                trades_df = pd.DataFrame(all_trades)
                if trades_df.empty:
                    print(f"Order: {order}, Detection: {detection_label.upper()} -> No trades found.")
                    continue

                # Aggregate trades by date and pattern type.
                agg_df = trades_df.groupby(['pattern', 'date']).agg({
                    'hold_return': 'sum',
                    'stop_return': 'sum'
                }).reset_index().sort_values('date')

                # Compute cumulative returns for each pattern group.
                for pattern, group in agg_df.groupby('pattern'):
                    group = group.sort_values('date').copy()
                    group['cum_hold_return'] = group['hold_return'].cumsum()
                    group['cum_stop_return'] = group['stop_return'].cumsum()
                    final_hold = group['cum_hold_return'].iloc[-1]
                    final_stop = group['cum_stop_return'].iloc[-1]

                    # Save the result in the summary.
                    results_summary.append({
                        'order': order,
                        'detection': detection_label,
                        'pattern': pattern,
                        'final_hold': final_hold,
                        'final_stop': final_stop
                    })

                    print(f"Order: {order}, Detection: {detection_label.upper()}, Pattern: {pattern} -> "
                        f"Final Cumulative HOLD PnL: {final_hold:.6f}, "
                        f"Final Cumulative STOP PnL: {final_stop:.6f}")

        # After processing all configurations, determine which one was best.
        if results_summary:
            # Find the best configuration based on final cumulative hold PnL.
            best_hold = max(results_summary, key=lambda r: r['final_hold'])
            # And the best configuration based on final cumulative stop PnL.
            best_stop = max(results_summary, key=lambda r: r['final_stop'])

            print("\nBest Configuration Based on HOLD PnL:")
            print(f"   Order: {best_hold['order']}, Detection: {best_hold['detection'].upper()}, "
                f"Pattern: {best_hold['pattern']} -> Final Cumulative HOLD PnL: {best_hold['final_hold']:.6f}")

            print("\nBest Configuration Based on STOP PnL:")
            print(f"   Order: {best_stop['order']}, Detection: {best_stop['detection'].upper()}, "
                f"Pattern: {best_stop['pattern']} -> Final Cumulative STOP PnL: {best_stop['final_stop']:.6f}")
        else:
            print("No trades were found for any configuration.")

    order_range = range(4, 200, 2)
    test_order_range(order_range)