import numpy as np
import pandas as pd
from datetime import datetime
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter, stoploss_from_absolute, informative, Order)
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from typing import List, Optional, Tuple
from typing import List, Dict
from typing import List
from collections import deque
from dataclasses import dataclass
import logging
import warnings

warnings.filterwarnings(
    'ignore', message='The objective has been evaluated at this point before.')
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# Checks if there is a local top detected at curr index
def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    top = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break
    
    return top

# Checks if there is a local top detected at curr index
def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False

    bottom = True
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            bottom = False
            break
    
    return bottom

def rw_extremes(data: np.array, order:int):
    # Rolling window local tops and bottoms
    tops = []
    bottoms = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            # top[0] = confirmation index
            # top[1] = index of top
            # top[2] = price of top
            top = [i, i - order, data[i - order]]
            tops.append(top)
        
        if rw_bottom(data, i, order):
            # bottom[0] = confirmation index
            # bottom[1] = index of bottom
            # bottom[2] = price of bottom
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)
    
    return tops, bottoms

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
    pat.pattern_r2 = compute_pattern_r2(data, pat)

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
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    
    pat.neck_slope = neck_slope
    pat.head_width = head_width
    pat.head_height = (data[l_armpit] + (head - l_armpit) * neck_slope) - data[head]
    pat.pattern_r2 = compute_pattern_r2(data, pat)
    
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


def get_pattern_signals_ihs(data: pd.DataFrame, pat: HSPattern) -> pd.DataFrame:
    """
    Generate trading signals for head and shoulders pattern.
    
    Args:
        data: DataFrame containing price data
        pat: Head and shoulders pattern object
        
    Returns:
        DataFrame with added 'signal' column containing 1 (entry) and -1 (exit)
    """
    # Create copy of dataframe to avoid modifying original
    result = data.copy()
    
    entry_i = pat.break_i
    stop_price = pat.r_shoulder_p
    tp_price = pat.neck_end + pat.head_height
    
    # Set entry signal
    result.loc[entry_i, 'signal'] = 1
    
    # Find exit point
    exit_found = False
    reached_time_limit = False

    for i in range(pat.head_width):
        current_i = entry_i + i

        result.loc[current_i, 'signal'] = 1
        
        # Check if we've reached the end of the data
        if current_i >= len(result):
            reached_time_limit = True
            break
            
        current_price = result.iloc[current_i]['close']  # Assuming 'close' is the price column
        
        # Check if stop loss or take profit is hit
        if (current_price > tp_price or current_price < stop_price):
            result.loc[current_i, 'signal'] = 0
            exit_found = True
            break
    
    # If no exit signal was generated due to stop/tp, exit at pattern width
    if not exit_found or reached_time_limit:
        result.loc[entry_i + pat.head_width, 'signal'] = 1
        
    return result

def generate_signals(df, order=74, early_find=False, pattern_type="both", hold_or_tpsl="tpsl"):
    """
    Generate a signal-only DataFrame from input OHLCV data.

    The input df is assumed to have a datetime index and a 'close' column.
    This function:
      - Converts the close prices to log-space.
      - Uses the existing head–shoulders (HS/IHS) pattern detector (find_hs_patterns)
        to identify potential trade setups.
      - For each pattern, marks an entry (1 for long on inverted HS, -1 for short on HS)
        at the pattern’s break index.
      - Simulates an exit by scanning forward (up to pat.head_width bars) until either
        the take-profit or stop condition is met. (At the exit bar the position returns to 0.)
      - Does not allow overlapping trades (later patterns that occur before a previous trade’s exit are skipped).
      - Allows the user to choose to use only inverted HS ("ihs"), regular HS ("hs"), or both ("both").

    Parameters:
        df (pd.DataFrame): Input DataFrame with a datetime index and a 'close' column.
        order (int): Rolling window order for pattern detection.
        early_find (bool): Whether to use early detection.
        pattern_type (str): One of "hs", "ihs", or "both" to select which pattern(s) to use.

    Returns:
        signal_df (pd.DataFrame): A DataFrame with the same index as df and a single column
                                  "signal" containing:
                                    1 for long positions,
                                   -1 for short positions,
                                    0 when not in a trade.
    """
    import numpy as np

    # Validate pattern_type input.
    pattern_type = pattern_type.lower()
    if pattern_type not in ("both", "hs", "ihs"):
        raise ValueError("pattern_type must be one of 'both', 'hs', or 'ihs'.")

    # Convert close prices to log prices (as used in the original script)
    prices = np.log(df["close"].to_numpy())
    
    # Run pattern detection (assumes find_hs_patterns and required helpers are defined)
    hs_patterns, ihs_patterns = find_hs_patterns(prices, order, early_find)
    
    # Build a combined list of detected trades based on the chosen pattern_type.
    trades = []
    if pattern_type in ("both", "hs"):
        # For regular HS patterns, the expected move is downward (signal = -1)
        for pat in hs_patterns:
            trades.append((pat.break_i, pat, -1))
    if pattern_type in ("both", "ihs"):
        # For inverted HS patterns, the expected move is upward (signal = 1)
        for pat in ihs_patterns:
            trades.append((pat.break_i, pat, 1))
    
    # Sort trades by their break index (chronological order)
    trades.sort(key=lambda x: x[0])
    
    # Initialize a signal array with zeros (no position)
    signals = np.zeros(len(prices))
    last_trade_exit = -1  # to avoid overlapping trades

    # For each detected trade, determine the exit index and fill in the position
    for entry, pat, pos in trades:
        # Skip if this trade starts before the previous one has exited
        if entry <= last_trade_exit:
            continue

        # Define exit conditions:
        # For an inverted pattern (long trade):
        #   take-profit (tp) is defined as neck_end + head_height, and stop is the right shoulder price.
        # For a regular HS (short trade):
        #   take-profit is neck_end - head_height, and stop is the right shoulder price.

        if "tpsl" in hold_or_tpsl:
            entry_idx = pat.break_i
            stop_price = pat.r_shoulder_p
            if pat.inverted:
                tp_price = pat.neck_end + pat.head_height
            else:
                tp_price = pat.neck_end - pat.head_height
            
            # Scan forward from the entry to determine the exit index.
            exit_idx = None
            for offset in range(int(pat.head_width)):
                idx = entry_idx + offset
                if idx >= len(prices):
                    break  # reached end of available data
                price = prices[idx]
                if pat.inverted:
                    # For a long trade: exit if price exceeds tp or falls below stop.
                    if price > tp_price or price < stop_price:
                        exit_idx = idx
                        break
                else:
                    # For a short trade: exit if price falls below tp or rises above stop.
                    if price < tp_price or price > stop_price:
                        exit_idx = idx
                        break
            # If no exit index was found within the allowed window, exit at end of pat.head_width.
            if exit_idx is None:
                exit_idx = entry_idx + offset
        elif "hold" in hold_or_tpsl: # if hold, exit at end of pat.head_width.
            entry_idx = entry_idx = pat.break_i
            exit_idx = pat.break_i + int(pat.head_width)

        # Mark the trade: from the entry bar until (but not including) the exit bar,
        # assign the trade's position.
        signals[entry_idx:exit_idx] = pos
        # The exit bar remains 0 (no position)
        last_trade_exit = exit_idx

    df["signal"] = signals
    return df

class HEAD_SHOULDER(IStrategy):
    
    # Paramètres de la stratégie
    minimal_roi = {"0": 5000}
    stoploss = -0.85
    timeframe = '15m'
    startup_candle_count: int = 10
    can_short: bool = True
    use_custom_stoploss: bool = False
    process_only_new_candles: bool = True

    order = IntParameter(3, 200, default=52, space='buy', optimize=True)
    early_find = BooleanParameter(default=False, space="buy", optimize=True)
    pattern_type = CategoricalParameter(['hs', 'ihs', 'both'], default='both', space='buy', optimize=True)
    hold_or_tpsl = CategoricalParameter(['hold', 'tpsl'], default='hold', space='buy', optimize=True)

    LEV = IntParameter(1, 5, default=5, space='buy', optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        dataframe = generate_signals(dataframe, order = self.order.value, early_find=self.early_find.value, pattern_type=self.pattern_type.value, hold_or_tpsl=self.hold_or_tpsl.value)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe[f'signal'] == 1),
            'enter_long'
        ] = 1
        dataframe.loc[
            (dataframe[f'signal'] == -1),
            'enter_short'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # exit done with custom_stoploss
        dataframe.loc[
            (dataframe[f'signal'] == 0),
            'exit_long'
        ] = 1
        dataframe.loc[
            (dataframe[f'signal'] == 0),
            'exit_short'
        ] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: "long" or "short" - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return self.LEV.value
