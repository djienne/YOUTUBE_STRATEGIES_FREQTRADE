"""
FLAG/PENNANT PATTERN DETECTION TRADING STRATEGY

This strategy identifies flag and pennant chart patterns using two different methods:
1. Perceptually important points (PIPs) for pattern recognition
2. Trendline-based pattern recognition

It generates buy/sell signals based on these patterns and supports both long and short positions.
"""

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

# Suppress various warnings to keep logs clean
warnings.filterwarnings(
'ignore', message='The objective has been evaluated at this point before.')
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


def find_pips(data: np.array, n_pips: int, dist_measure: int):
    """
    Find Perceptually Important Points (PIPs) in price data.
    
    This algorithm identifies significant points in a price series by iteratively
    finding points with maximum distance from a line formed by adjacent points.
    
    Parameters:
        data: Array of price values
        n_pips: Number of important points to find
        dist_measure: Method to calculate distance
            1 = Euclidean Distance
            2 = Perpendicular Distance
            3 = Vertical Distance
    
    Returns:
        pips_x: Indices of the important points
        pips_y: Price values of the important points
    """
    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max distance
        md_i = -1 # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope;

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                
                d = 0.0 # Distance
                if dist_measure == 1: # Euclidean distance
                    d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                    d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                elif dist_measure == 2: # Perpendicular distance
                    d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5
                else: # Vertical distance    
                    d = abs( (slope * i + intercept) - data[i] )

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y


def rw_top(data: np.array, curr_index: int, order: int) -> bool:
    """
    Check if there is a local top detected at curr_index - order.
    
    A local top exists if the point at curr_index - order is higher than
    'order' number of points on both sides.
    
    Parameters:
        data: Array of price values
        curr_index: Current index to check (peak will be at curr_index - order)
        order: Number of points to check on each side
    
    Returns:
        bool: True if a local top is detected, False otherwise
    """
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


def rw_bottom(data: np.array, curr_index: int, order: int) -> bool:
    """
    Check if there is a local bottom detected at curr_index - order.
    
    A local bottom exists if the point at curr_index - order is lower than
    'order' number of points on both sides.
    
    Parameters:
        data: Array of price values
        curr_index: Current index to check (bottom will be at curr_index - order)
        order: Number of points to check on each side
    
    Returns:
        bool: True if a local bottom is detected, False otherwise
    """
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


def rw_extremes(data: np.array, order: int):
    """
    Find all local tops and bottoms using a rolling window approach.
    
    Parameters:
        data: Array of price values
        order: Number of points to check on each side for extremes
    
    Returns:
        tops: List of top points with format [confirmation_index, top_index, price]
        bottoms: List of bottom points with format [confirmation_index, bottom_index, price]
    """
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


def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    """
    Check if a trend line is valid and calculate error from actual prices.
    
    For support lines, all prices must be above the line.
    For resistance lines, all prices must be below the line.
    
    Parameters:
        support: True if checking a support line, False for resistance
        pivot: Index of the pivot point the line passes through
        slope: Slope of the line
        y: Array of price values
    
    Returns:
        float: Sum of squared differences between line and prices if valid,
              negative value if invalid
    """
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    """
    Optimize the slope of a trend line to minimize error while maintaining validity.
    
    Uses numerical optimization to find the best slope that:
    - For support lines: stays below all price points
    - For resistance lines: stays above all price points
    
    Parameters:
        support: True if optimizing a support line, False for resistance
        pivot: Index of the pivot point the line passes through
        init_slope: Initial slope to start optimization from
        y: Array of price values
    
    Returns:
        tuple: (optimized_slope, intercept) of the best trend line
    """
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 

    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative

    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    """
    Fit support and resistance trend lines to a single price series.
    
    First finds a basic line of best fit, then identifies upper and lower
    pivot points and optimizes separate support and resistance lines.
    
    Parameters:
        data: Array of price values
    
    Returns:
        tuple: (support_line_coefs, resistance_line_coefs) where each is (slope, intercept)
    """
    # find line of best fit (least squared)
    # coefs[0] = slope,  coefs[1] = intercept
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 

    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    """
    Fit support and resistance trend lines using high, low, and close prices.
    
    Uses close prices to find the initial line of best fit,
    then uses high prices for resistance and low prices for support.
    
    Parameters:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
    
    Returns:
        tuple: (support_line_coefs, resistance_line_coefs) where each is (slope, intercept)
    """
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


@dataclass
class FlagPattern:
    """
    Represents a flag or pennant chart pattern with its characteristics.
    
    Flag and pennant patterns are continuation patterns consisting of:
    - A strong, nearly straight-line move (the pole)
    - A consolidation period (the flag/pennant)
    - A breakout in the direction of the original trend
    
    Pennants are triangular consolidations, while flags are parallelogram-shaped.
    """
    base_x: int         # Start of the trend index, base of pole
    base_y: float       # Start of trend price

    tip_x: int   = -1       # Tip of pole, start of flag
    tip_y: float = -1.

    conf_x: int   = -1      # Index where pattern is confirmed
    conf_y: float = -1.      # Price where pattern is confirmed

    pennant: bool = False      # True if pennant, false if flag

    flag_width: int    = -1    # Width of the flag portion in bars
    flag_height: float = -1.   # Height of the flag portion in price

    pole_width: int    = -1    # Width of the pole portion in bars
    pole_height: float = -1.   # Height of the pole portion in price

    # Upper and lower trendlines for the flag, intercept is at tip_x
    support_intercept: float = -1.  # Intercept of the support trendline
    support_slope: float = -1.      # Slope of the support trendline
    resist_intercept: float = -1.   # Intercept of the resistance trendline
    resist_slope: float = -1.       # Slope of the resistance trendline


def check_bear_pattern_pips(pending: FlagPattern, data: np.array, i: int, order: int):
    """
    Check if a bearish flag/pennant pattern is present and confirmed using PIPs method.
    
    For bearish patterns, we look for:
    1. A downward pole (prior trend)
    2. A consolidation period (flag/pennant)
    3. A downward breakout (confirmation)
    
    Parameters:
        pending: FlagPattern object with base point initialized
        data: Array of price values
        i: Current index being checked
        order: Order parameter for local extremes detection
    
    Returns:
        bool: True if pattern is confirmed, False otherwise
        Side effect: Updates the pending FlagPattern if confirmed
    """
    # Find max price since local bottom, (top of pole)
    data_slice = data[pending.base_x: i + 1] # i + 1 includes current price
    min_i = data_slice.argmin() + pending.base_x # Min index since local top

    if i - min_i < max(5, order * 0.5): # Far enough from max to draw potential flag/pennant
        return False

    # Test flag width / height 
    pole_width = min_i - pending.base_x
    flag_width = i - min_i
    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    pole_height = pending.base_y - data[min_i] 
    flag_height = data[min_i:i+1].max() - data[min_i] 
    if flag_height > pole_height * 0.5: # Flag should smaller vertically than preceding trend
        return False

    # If here width/height are OK.

    # Find perceptually important points from pole to current time
    pips_x, pips_y = find_pips(data[min_i:i+1], 5, 3) # Finds pips between max and current index (inclusive)

    # Check center pip is less than two adjacent. /\/\ 
    if not (pips_y[2] < pips_y[1] and pips_y[2] < pips_y[3]):
        return False

    # Find slope and intercept of flag lines
    # intercept is at the max value (top of pole)
    support_rise = pips_y[2] - pips_y[0]
    support_run = pips_x[2] - pips_x[0]
    support_slope = support_rise / support_run
    support_intercept = pips_y[0] 

    resist_rise = pips_y[3] - pips_y[1]
    resist_run = pips_x[3] - pips_x[1]
    resist_slope = resist_rise / resist_run
    resist_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * resist_slope

    # Find x where two lines intersect.
    #print(pips_x[0], resist_slope, support_slope)
    if resist_slope != support_slope: # Not parallel
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
        #print("Intersects at", intersection)
    else:
        intersection = -flag_width * 100

    # No intersection in flag area
    if intersection <= pips_x[4] and intersection >= 0:
        return False

    # Check if current point has a breakout of flag. (confirmation)
    support_endpoint = pips_y[0] + support_slope * pips_x[4]
    if pips_y[4] > support_endpoint:
        return False

    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # Filter harshly diverging lines
    if intersection < 0 and intersection > -flag_width:
        return False

    # Store pattern details in the pending object
    pending.tip_x = min_i
    pending.tip_y = data[min_i]
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def check_bull_pattern_pips(pending: FlagPattern, data: np.array, i: int, order: int):
    """
    Check if a bullish flag/pennant pattern is present and confirmed using PIPs method.
    
    For bullish patterns, we look for:
    1. An upward pole (prior trend)
    2. A consolidation period (flag/pennant)
    3. An upward breakout (confirmation)
    
    Parameters:
        pending: FlagPattern object with base point initialized
        data: Array of price values
        i: Current index being checked
        order: Order parameter for local extremes detection
    
    Returns:
        bool: True if pattern is confirmed, False otherwise
        Side effect: Updates the pending FlagPattern if confirmed
    """
    # Find max price since local bottom, (top of pole)
    data_slice = data[pending.base_x: i + 1] # i + 1 includes current price
    max_i = data_slice.argmax() + pending.base_x # Max index since bottom
    pole_width = max_i - pending.base_x

    if i - max_i < max(5, order * 0.5): # Far enough from max to draw potential flag/pennant
        return False

    flag_width = i - max_i
    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    pole_height = data[max_i] - pending.base_y 
    flag_height = data[max_i] - data[max_i:i+1].min()
    if flag_height > pole_height * 0.5: # Flag should smaller vertically than preceding trend
        return False

    # Find perceptually important points in the flag/pennant portion
    pips_x, pips_y = find_pips(data[max_i:i+1], 5, 3) # Finds pips between max and current index (inclusive)

    # Check center pip is greater than two adjacent. \/\/  
    if not (pips_y[2] > pips_y[1] and pips_y[2] > pips_y[3]):
        return False
        
    # Find slope and intercept of flag lines
    # intercept is at the max value (top of pole)
    resist_rise = pips_y[2] - pips_y[0]
    resist_run = pips_x[2] - pips_x[0]
    resist_slope = resist_rise / resist_run
    resist_intercept = pips_y[0] 

    support_rise = pips_y[3] - pips_y[1]
    support_run = pips_x[3] - pips_x[1]
    support_slope = support_rise / support_run
    support_intercept = pips_y[1] + (pips_x[0] - pips_x[1]) * support_slope

    # Find x where two lines intersect.
    if resist_slope != support_slope: # Not parallel
        intersection = (support_intercept - resist_intercept) / (resist_slope - support_slope)
    else:
        intersection = -flag_width * 100

    # No intersection in flag area
    if intersection <= pips_x[4] and intersection >= 0:
        return False

    # Filter harshly diverging lines
    if intersection < 0 and intersection > -1.0 * flag_width:
        return False

    # Check if current point has a breakout of flag. (confirmation)
    resist_endpoint = pips_y[0] + resist_slope * pips_x[4]
    if pips_y[4] < resist_endpoint:
        return False

    # Pattern is confirmed, fill out pattern details in pending
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # Store pattern details in the pending object
    pending.tip_x = max_i
    pending.tip_y = data[max_i]
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def find_flags_pennants_pips(data: np.array, order: int):
    """
    Find all bull and bear flag/pennant patterns using the PIPs method.
    
    This uses perceptually important points (PIPs) to identify key points in the
    potential flag/pennant patterns and confirm breakouts.
    
    Parameters:
        data: Array of price values
        order: Order parameter for local extremes detection
    
    Returns:
        tuple: (bull_flags, bear_flags, bull_pennants, bear_pennants)
               Lists of confirmed patterns of each type
    """
    assert(order >= 3)
    pending_bull = None # Pending pattern
    pending_bear = None # Pending pattern

    bull_pennants = []
    bear_pennants = []
    bull_flags = []
    bear_flags = []
    for i in range(len(data)):

        # Initialize new pattern when we detect a local top or bottom
        if rw_top(data, i, order):
            pending_bear = FlagPattern(i - order, data[i - order])
        
        if rw_bottom(data, i, order):
            pending_bull = FlagPattern(i - order, data[i - order])

        # Check if pending patterns are confirmed
        if pending_bear is not None:
            if check_bear_pattern_pips(pending_bear, data, i, order):
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)
                else:
                    bear_flags.append(pending_bear)
                pending_bear = None

        if pending_bull is not None:
            if check_bull_pattern_pips(pending_bull, data, i, order):
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)
                else:
                    bull_flags.append(pending_bull)
                pending_bull = None

    return bull_flags, bear_flags, bull_pennants, bear_pennants


def check_bull_pattern_trendline(pending: FlagPattern, data: np.array, i: int, order: int):
    """
    Check if a bullish flag/pennant pattern is present and confirmed using trendlines.
    
    This uses optimized trendlines to identify the flag/pennant pattern 
    and confirm a breakout.
    
    Parameters:
        pending: FlagPattern object with base and tip points initialized
        data: Array of price values
        i: Current index being checked
        order: Order parameter for local extremes detection
    
    Returns:
        bool: True if pattern is confirmed, False otherwise
        Side effect: Updates the pending FlagPattern if confirmed
    """
    # Check if data max less than pole tip 
    if data[pending.tip_x + 1 : i].max() > pending.tip_y:
        return False

    flag_min = data[pending.tip_x:i].min()

    # Find flag/pole height and width
    pole_height = pending.tip_y - pending.base_y
    pole_width = pending.tip_x - pending.base_x

    flag_height = pending.tip_y - flag_min
    flag_width = i - pending.tip_x

    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    if flag_height > pole_height * 0.75: # Flag should smaller vertically than preceding trend
        return False

    # Find trendlines going from flag tip to the previous bar (not including current bar)
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]

    # Check for breakout of upper trendline to confirm pattern
    current_resist = resist_intercept + resist_slope * (flag_width + 1)
    if data[i] <= current_resist:
        return False

    # Pattern is confirmed, fill out pattern details in pending
    if support_slope > 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # Store pattern details in the pending object
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def check_bear_pattern_trendline(pending: FlagPattern, data: np.array, i: int, order: int):
    """
    Check if a bearish flag/pennant pattern is present and confirmed using trendlines.
    
    This uses optimized trendlines to identify the flag/pennant pattern 
    and confirm a breakout.
    
    Parameters:
        pending: FlagPattern object with base and tip points initialized
        data: Array of price values
        i: Current index being checked
        order: Order parameter for local extremes detection
    
    Returns:
        bool: True if pattern is confirmed, False otherwise
        Side effect: Updates the pending FlagPattern if confirmed
    """
    # Check if data max less than pole tip
    if data[pending.tip_x + 1 : i].min() < pending.tip_y:
        return False

    flag_max = data[pending.tip_x:i].max()

    # Find flag/pole height and width
    pole_height = pending.base_y - pending.tip_y
    pole_width = pending.tip_x - pending.base_x

    flag_height = flag_max - pending.tip_y
    flag_width = i - pending.tip_x

    if flag_width > pole_width * 0.5: # Flag should be less than half the width of pole
        return False

    if flag_height > pole_height * 0.75: # Flag should smaller vertically than preceding trend
        return False

    # Find trendlines going from flag tip to the previous bar (not including current bar)
    support_coefs, resist_coefs = fit_trendlines_single(data[pending.tip_x:i])
    support_slope, support_intercept = support_coefs[0], support_coefs[1]
    resist_slope, resist_intercept = resist_coefs[0], resist_coefs[1]

    # Check for breakout of lower trendline to confirm pattern
    current_support = support_intercept + support_slope * (flag_width + 1)
    if data[i] >= current_support:
        return False

    # Pattern is confirmed, fill out pattern details in pending
    if resist_slope < 0:
        pending.pennant = True
    else:
        pending.pennant = False

    # Store pattern details in the pending object
    pending.conf_x = i
    pending.conf_y = data[i]
    pending.flag_width = flag_width
    pending.flag_height = flag_height
    pending.pole_width = pole_width
    pending.pole_height = pole_height
    pending.support_slope = support_slope
    pending.support_intercept = support_intercept
    pending.resist_slope = resist_slope
    pending.resist_intercept = resist_intercept

    return True


def find_flags_pennants_trendline(data: np.array, order: int):
    """
    Find all bull and bear flag/pennant patterns using the trendline method.
    
    This uses optimized support and resistance trendlines to identify
    the consolidation pattern and confirm breakouts.
    
    Parameters:
        data: Array of price values
        order: Order parameter for local extremes detection
    
    Returns:
        tuple: (bull_flags, bear_flags, bull_pennants, bear_pennants)
               Lists of confirmed patterns of each type
    """
    assert(order >= 3)
    pending_bull = None # Pending pattern
    pending_bear = None  # Pending pattern

    last_bottom = -1
    last_top = -1

    bull_pennants = []
    bear_pennants = []
    bull_flags = []
    bear_flags = []
    for i in range(len(data)):

        # Track local tops and bottoms to identify potential patterns
        if rw_top(data, i, order):
            last_top = i - order
            if last_bottom != -1:
                pending = FlagPattern(last_bottom, data[last_bottom])
                pending.tip_x = last_top
                pending.tip_y = data[last_top]
                pending_bull = pending
        
        if rw_bottom(data, i, order):
            last_bottom = i - order
            if last_top != -1:
                pending = FlagPattern(last_top, data[last_top])
                pending.tip_x = last_bottom
                pending.tip_y = data[last_bottom]
                pending_bear = pending

        # Check if pending patterns are confirmed
        if pending_bear is not None:
            if check_bear_pattern_trendline(pending_bear, data, i, order):
                if pending_bear.pennant:
                    bear_pennants.append(pending_bear)
                else:
                    bear_flags.append(pending_bear)
                pending_bear = None
        
        if pending_bull is not None:
            if check_bull_pattern_trendline(pending_bull, data, i, order):
                if pending_bull.pennant:
                    bull_pennants.append(pending_bull)
                else:
                    bull_flags.append(pending_bull)
                pending_bull = None

    return bull_flags, bear_flags, bull_pennants, bear_pennants


def generate_signals(dataframe: pd.DataFrame, order: int = 7, pattern_type='both', hold_mult=1.0):
    """
    Generate trading signals based on detected flag/pennant patterns.
    
    Parameters:
        dataframe: DataFrame with OHLC price data
        order: Order parameter for local extremes detection
        pattern_type: Type of patterns to generate signals for:
                     'bull_flag', 'bear_flag', or 'both'
        hold_mult: Multiplier for signal hold period relative to flag width
    
    Returns:
        DataFrame: Original dataframe with 'signal' column added
                  (1 for buy, -1 for sell, 0 for no action)
    """
    data = dataframe.copy()
    data = data.set_index('date')
    close_prices_log = np.log(data['close'].to_numpy())
    signals = np.zeros(len(close_prices_log))
    bull_flags, bear_flags, bull_pennants, bear_pennants = find_flags_pennants_pips(close_prices_log, order)

    # Log number of patterns found
    logger.debug(f"Bull flags: {len(bull_flags)}, Bear flags: {len(bear_flags)}")

    # Generate buy signals for bullish patterns
    if pattern_type in ('bull_flag','both'):
        for flag in bull_flags:
            hp = int(flag.flag_width * hold_mult)
            end = min(flag.conf_x + hp, len(signals))
            signals[flag.conf_x:end] = 1

    # Generate sell signals for bearish patterns
    if pattern_type in ('bear_flag','both'):
        for flag in bear_flags:
            hp = int(flag.flag_width * hold_mult)
            end = min(flag.conf_x + hp, len(signals))
            signals[flag.conf_x:end] = -1

    dataframe['signal'] = signals
    return dataframe


class FLAGS(IStrategy):
    """
    Freqtrade strategy that trades flag and pennant chart patterns.
    
    This strategy identifies flag and pennant patterns, which are continuation
    patterns that occur after a strong price move. It generates buy signals
    for bullish patterns and sell signals for bearish patterns.
    
    The pattern detection is based on perceptually important points (PIPs)
    and can be configured to look for bull patterns, bear patterns, or both.
    """
    minimal_roi = {
        "0": 5000.0,  # Effectively disable ROI (let pattern completion determine exits)
    }
    stoploss = -0.85  # Wide stoploss to allow patterns to develop
    timeframe = '1h'
    startup_candle_count: int = 10  # Lowered for faster startup
    can_short: bool = True  # Allow short positions
    process_only_new_candles: bool = False  # Process every tick for debugging

    # Strategy parameters
    order = IntParameter(3, 200, default=189, space='buy', optimize=False)  # Pattern detection order
    pattern_type = CategoricalParameter(['bull_flag', 'bear_flag', 'both'], default='both', space='buy', optimize=False)
    hold_mult = DecimalParameter(0.1, 2.0, decimals=1, default=1.9, space="buy", optimize=False)  # Hold duration multiplier
    LEV = IntParameter(1, 5, default=4, space='buy', optimize=False)  # Leverage to use

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Generate pattern detection signals and add them to the dataframe.
        """
        df = generate_signals(
            dataframe,
            order=self.order.value,
            pattern_type=self.pattern_type.value,
            hold_mult=self.hold_mult.value
        )
        # Debug logging of latest signal
        if not df.empty:
            logger.debug(f"Latest signal: {df['signal'].iat[-1]}")
        return df

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define entry conditions based on pattern signals.
        """
        dataframe.loc[dataframe['signal'] == 1, 'enter_long'] = 1
        dataframe.loc[dataframe['signal'] == -1, 'enter_short'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Define exit conditions based on pattern signals.
        """
        dataframe.loc[dataframe['signal'] == 0, 'exit_long'] = 1
        dataframe.loc[dataframe['signal'] == 0, 'exit_short'] = 1
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str,
                 **kwargs) -> float:
        """
        Define the leverage to use for trades.
        
        Parameters:
            pair: Trading pair
            current_time: Current time
            current_rate: Current exchange rate
            proposed_leverage: Proposed leverage
            max_leverage: Maximum allowed leverage by exchange
            entry_tag: Entry tag (if any)
            side: Trade side ('long' or 'short')
            **kwargs: Additional arguments
        
        Returns:
            float: Leverage to use
        """
        # Clamp leverage to exchange max
        lev = min(self.LEV.value, max_leverage)
        logger.debug(f"Using leverage: {lev} (max allowed: {max_leverage})")
        return lev