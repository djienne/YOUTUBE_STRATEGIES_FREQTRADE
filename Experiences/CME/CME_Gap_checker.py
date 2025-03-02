import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Configure pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.options.mode.chained_assignment = None

@dataclass
class Gap:
    """Class for representing a weekend gap between Friday CME close and Sunday open."""
    index: int  # Sequential index number
    friday_date: datetime
    friday_close: float
    sunday_date: datetime
    sunday_open: float
    gap_pct: float
    gap_direction: str  # 'up', 'down', or 'flat'
    max_deviation_pct: Optional[float] = None  # New attribute: maximum deviation % from Friday close

    def __repr__(self):
        # Choose color for gap percentage based on direction
        if self.gap_direction.lower() == "up":
            pct_color = GREEN
        elif self.gap_direction.lower() == "down":
            pct_color = RED
        else:
            pct_color = YELLOW
        rep = (f"Gap #{self.index}: {self.gap_direction.upper()} {pct_color}{self.gap_pct:.2f}%{RESET} " 
               f"({self.friday_date.strftime('%Y-%m-%d')} Friday close: {CYAN}{self.friday_close:.2f}{RESET} → "
               f"{self.sunday_date.strftime('%Y-%m-%d')} Sunday open: {CYAN}{self.sunday_open:.2f}{RESET})")
        if self.max_deviation_pct is not None:
            rep += f" | Max Deviation (from Friday close): {MAGENTA}{self.max_deviation_pct:.2f}%{RESET}"
        return rep
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Gap object to dictionary for easy DataFrame creation."""
        return {
            'gap_index': self.index,
            'friday_date': self.friday_date,
            'friday_close': self.friday_close,
            'sunday_date': self.sunday_date,
            'sunday_open': self.sunday_open,
            'gap_pct': self.gap_pct,
            'gap_direction': self.gap_direction,
            'max_deviation_pct': self.max_deviation_pct
        }

def load_ohlcv_data_df(symbol, timeframe):
    symbol_for_filename = symbol.replace('/', '_')
    filename = os.path.join(f"{symbol_for_filename}-{timeframe}.feather")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Feather file not found: {filename}")
    df = pd.read_feather(filename)
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        try:
            df['date'] = pd.to_datetime(df['date'], unit='ms')
        except Exception:
            df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('America/Chicago')
    else:
        df['date'] = df['date'].dt.tz_convert('America/Chicago')
    return df

def get_btc_cme_weekly_data():
    """
    Loads Bitcoin futures data from CME weekly candles from a CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with the data.
    """
    file_path = "BTC1_weekly_data.csv"

    if os.path.exists(file_path):
        last_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        current_time = datetime.now()
        time_difference = current_time - last_modified_time
        if time_difference < timedelta(days=1):
            print("Loading data from the existing CSV file...")
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
            return df

    print("Downloading BTC1! weekly data...")
    tv = TvDatafeed()
    data = tv.get_hist(
        symbol='BTC1!',
        exchange='CME',
        interval=Interval.in_weekly,
        n_bars=10000
    )

    if data is None or data.empty:
        print("No data was returned. Please check the symbol/exchange credentials.")
        return

    data.to_csv(file_path)
    print(f"Data saved successfully to {file_path}")
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
    if datetime.now().weekday() < 5:
        df = df.drop(df.index[-1])
    return df

def add_historical_last_cme_closes(df, df_cme):
    df_copy = df.copy()
    df_cme_copy = df_cme.copy()
    df_copy['_date_temp'] = pd.to_datetime(df_copy['date']).dt.tz_localize(None)
    df_cme_copy['datetime'] = pd.to_datetime(df_cme_copy['datetime']).dt.tz_localize(None) + pd.Timedelta(days=5) - pd.Timedelta(hours=1)
    df_cme_copy = df_cme_copy[['datetime', 'close']].rename(columns={'close': 'cme_close'})
    merged_df = pd.merge_asof(
        df_copy,
        df_cme_copy,
        left_on='_date_temp',
        right_on='datetime',
        direction='backward'
    )
    merged_df.drop(columns=['datetime', '_date_temp'], inplace=True)
    return merged_df

def extract_weekend_gaps(df: pd.DataFrame) -> List[Gap]:
    """
    Extract weekend gaps between Friday CME close and Sunday open.
    
    Args:
        df: DataFrame with date, cme_close, and other columns
        
    Returns:
        List of Gap objects representing each weekend gap
    """
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert('UTC')
    if 'sunday_open' not in df.columns:
        df['sunday_open'] = np.nan
        sunday_11pm_mask = (df['date'].dt.dayofweek == 6) & (df['date'].dt.hour == 23) & (df['date'].dt.minute == 0)
        df.loc[sunday_11pm_mask, 'sunday_open'] = df.loc[sunday_11pm_mask, 'close']
    df['friday_close'] = np.nan
    mask = (df['cme_close'] != df['cme_close'].shift().fillna(method='bfill'))
    df.loc[mask, 'friday_close'] = df.loc[mask, 'cme_close']
    gaps = []
    friday_closes = df.loc[mask].copy()
    gap_count = 0
    for idx, friday_row in friday_closes.iterrows():
        friday_date = friday_row['date']
        next_sunday = friday_date + pd.Timedelta(days=2)
        while next_sunday.dayofweek != 6:
            next_sunday += pd.Timedelta(days=1)
        next_sunday = next_sunday.replace(hour=23, minute=0, second=0)
        sunday_rows = df[(df['date'].dt.year == next_sunday.year) & 
                         (df['date'].dt.month == next_sunday.month) & 
                         (df['date'].dt.day == next_sunday.day) & 
                         (df['date'].dt.hour == 23) & 
                         (df['date'].dt.minute == 0)]
        if not sunday_rows.empty:
            sunday_row = sunday_rows.iloc[0]
            friday_close = friday_row['cme_close']
            sunday_open = sunday_row['sunday_open']
            if not pd.isna(friday_close) and not pd.isna(sunday_open):
                gap_pct = ((sunday_open / friday_close) - 1) * 100
                gap_direction = 'up' if gap_pct > 0 else 'down' if gap_pct < 0 else 'flat'
                gap_count += 1
                gap = Gap(
                    index=gap_count,
                    friday_date=friday_date,
                    friday_close=friday_close,
                    sunday_date=sunday_row['date'],
                    sunday_open=sunday_open,
                    gap_pct=gap_pct,
                    gap_direction=gap_direction
                )
                gaps.append(gap)
    return gaps

def analyze_gap_fills(df: pd.DataFrame, gaps: List[Gap]) -> List[Gap]:
    """
    Analyze whether each gap has been filled after Sunday open.
    
    For down gaps: Check if any hourly high price reaches the Friday close
    For up gaps: Check if any hourly low price reaches the Friday close
    
    Additionally, for filled gaps, calculate the maximum deviation % from Friday close 
    until the gap is filled.
    
    Args:
        df: DataFrame with hourly price data
        gaps: List of Gap objects
        
    Returns:
        Updated list of Gap objects with fill and maximum deviation information
    """
    df_copy = df.copy()
    if not hasattr(Gap, 'filled'):
        Gap.filled = False
        Gap.fill_date = None
        Gap.days_to_fill = None
        Gap.candles_to_fill = None
    for gap in gaps:
        after_sunday_mask = df_copy['date'] > gap.sunday_date
        df_after_sunday = df_copy[after_sunday_mask].copy()
        if df_after_sunday.empty:
            continue
        if gap.gap_direction == 'down':
            fill_condition = df_after_sunday['high'] >= gap.friday_close
            fill_row = df_after_sunday[fill_condition]
        else:  # 'up' or 'flat'
            fill_condition = df_after_sunday['low'] <= gap.friday_close
            fill_row = df_after_sunday[fill_condition]
        if not fill_row.empty:
            gap.filled = True
            gap.fill_date = fill_row.iloc[0]['date']
            time_to_fill = gap.fill_date - gap.sunday_date
            gap.days_to_fill = time_to_fill.total_seconds() / (60 * 60 * 24)
            gap.candles_to_fill = len(df_copy[(df_copy['date'] > gap.sunday_date) & 
                                               (df_copy['date'] <= gap.fill_date)])
            # Calculate the maximum deviation % from Friday close until the gap is filled.
            # For UP gaps: (max(high) during [Sunday open, fill_date] / friday_close - 1) * 100
            # For DOWN gaps: (min(low) during [Sunday open, fill_date] / friday_close - 1) * 100
            df_interval = df_copy[(df_copy['date'] > gap.sunday_date) & (df_copy['date'] <= gap.fill_date)]
            if not df_interval.empty:
                if gap.gap_direction.lower() == 'up':
                    max_dev = (df_interval['high'].max() / gap.friday_close - 1) * 100
                else:  # down gap
                    max_dev = (df_interval['low'].min() / gap.friday_close - 1) * 100
                gap.max_deviation_pct = max_dev
            else:
                gap.max_deviation_pct = 0
        else:
            gap.filled = False
            gap.fill_date = None
            gap.days_to_fill = None
            gap.candles_to_fill = None
            gap.max_deviation_pct = None
    return gaps

def generate_gap_fill_report(gaps: List[Gap]) -> pd.DataFrame:
    """
    Generate a report on gap fills.
    
    Args:
        gaps: List of Gap objects with fill information
        
    Returns:
        DataFrame with gap fill statistics and gap details
    """
    gap_dicts = []
    for gap in gaps:
        gap_dict = gap.to_dict()
        gap_dict['filled'] = getattr(gap, 'filled', False)
        gap_dict['fill_date'] = getattr(gap, 'fill_date', None)
        gap_dict['days_to_fill'] = getattr(gap, 'days_to_fill', None)
        gap_dict['candles_to_fill'] = getattr(gap, 'candles_to_fill', None)
        gap_dicts.append(gap_dict)
    gaps_df = pd.DataFrame(gap_dicts)
    report = {}
    total_gaps = len(gaps)
    filled_gaps = sum(1 for gap in gaps if getattr(gap, 'filled', False))
    unfilled_gaps = total_gaps - filled_gaps
    report['total_gaps'] = total_gaps
    report['filled_gaps'] = filled_gaps
    report['unfilled_gaps'] = unfilled_gaps
    report['fill_percentage'] = (filled_gaps / total_gaps * 100) if total_gaps > 0 else 0
    up_gaps = [gap for gap in gaps if gap.gap_direction == 'up']
    down_gaps = [gap for gap in gaps if gap.gap_direction == 'down']
    up_gaps_filled = sum(1 for gap in up_gaps if getattr(gap, 'filled', False))
    down_gaps_filled = sum(1 for gap in down_gaps if getattr(gap, 'filled', False))
    report['up_gaps_total'] = len(up_gaps)
    report['up_gaps_filled'] = up_gaps_filled
    report['up_gaps_fill_percentage'] = (up_gaps_filled / len(up_gaps) * 100) if len(up_gaps) > 0 else 0
    report['down_gaps_total'] = len(down_gaps)
    report['down_gaps_filled'] = down_gaps_filled
    report['down_gaps_fill_percentage'] = (down_gaps_filled / len(down_gaps) * 100) if len(down_gaps) > 0 else 0
    filled_gap_objs = [gap for gap in gaps if getattr(gap, 'filled', False)]
    if filled_gap_objs:
        days_to_fill = [gap.days_to_fill for gap in filled_gap_objs]
        candles_to_fill = [gap.candles_to_fill for gap in filled_gap_objs]
        report['avg_days_to_fill'] = sum(days_to_fill) / len(days_to_fill)
        report['median_days_to_fill'] = sorted(days_to_fill)[len(days_to_fill) // 2]
        report['max_days_to_fill'] = max(days_to_fill)
        report['min_days_to_fill'] = min(days_to_fill)
        report['avg_candles_to_fill'] = sum(candles_to_fill) / len(candles_to_fill)
        report['median_candles_to_fill'] = sorted(candles_to_fill)[len(candles_to_fill) // 2]
        report['max_candles_to_fill'] = max(candles_to_fill)
        report['min_candles_to_fill'] = min(candles_to_fill)
    report_df = pd.DataFrame([report])
    return gaps_df, report_df

def plot_gap_fill_histogram(gaps: List[Gap]):
    """
    Plot histograms of the number of days it takes to fill a gap,
    separated between UP and DOWN gaps, with both the x-axis and bin sizes on a logarithmic scale.
    Also adds mean and median vertical dotted lines.
    """
    up_days = [gap.days_to_fill for gap in gaps 
               if getattr(gap, 'filled', False) and gap.gap_direction.lower() == 'up' and gap.days_to_fill > 0]
    down_days = [gap.days_to_fill for gap in gaps 
                 if getattr(gap, 'filled', False) and gap.gap_direction.lower() == 'down' and gap.days_to_fill > 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    if up_days:
        up_min = min(up_days)
        up_max = max(up_days)
        bins_up = np.logspace(np.log10(up_min), np.log10(up_max), 100)
        ax1.hist(up_days, bins=bins_up, color='green', alpha=0.7)
        up_mean = np.mean(up_days)
        up_median = np.median(up_days)
        ax1.axvline(up_mean, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {up_mean:.2f} days')
        ax1.axvline(up_median, color='purple', linestyle=':', linewidth=1.5, label=f'Median: {up_median:.2f} days')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No data", transform=ax1.transAxes, ha='center')
    ax1.set_title('UP Gaps')
    ax1.set_xlabel('Days to Fill')
    ax1.set_ylabel('Frequency')
    ax1.set_xscale('log')
    if down_days:
        down_min = min(down_days)
        down_max = max(down_days)
        bins_down = np.logspace(np.log10(down_min), np.log10(down_max), 100)
        ax2.hist(down_days, bins=bins_down, color='red', alpha=0.7)
        down_mean = np.mean(down_days)
        down_median = np.median(down_days)
        ax2.axvline(down_mean, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {down_mean:.2f} days')
        ax2.axvline(down_median, color='purple', linestyle=':', linewidth=1.5, label=f'Median: {down_median:.2f} days')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No data", transform=ax2.transAxes, ha='center')
    ax2.set_title('DOWN Gaps')
    ax2.set_xlabel('Days to Fill')
    ax2.set_xscale('log')
    plt.suptitle('Histogram of Days to Fill Gaps (Log Scale)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_gap_percentage_histogram(gaps: List[Gap]):
    """
    Plot histograms of gap percentages, separated between UP and DOWN gaps,
    with UP gaps on the left in green and DOWN gaps on the right in red.
    """
    up_percentages = [gap.gap_pct for gap in gaps if gap.gap_direction.lower() == 'up']
    down_percentages = [abs(gap.gap_pct) for gap in gaps if gap.gap_direction.lower() == 'down']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    if up_percentages:
        bins_up = np.linspace(min(up_percentages), max(up_percentages), 20)
        ax1.hist(up_percentages, bins=bins_up, color='green', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(up_percentages), color='darkgreen', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {np.mean(up_percentages):.2f}%')
        ax1.axvline(np.median(up_percentages), color='lime', linestyle='dotted', linewidth=2, 
                   label=f'Median: {np.median(up_percentages):.2f}%')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No UP gaps data", transform=ax1.transAxes, ha='center')
    ax1.set_title('UP Gaps Percentages')
    ax1.set_xlabel('Gap Percentage (%)')
    ax1.set_ylabel('Frequency')
    if down_percentages:
        bins_down = np.linspace(min(down_percentages), max(down_percentages), 20)
        ax2.hist(down_percentages, bins=bins_down, color='red', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(down_percentages), color='darkred', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {np.mean(down_percentages):.2f}%')
        ax2.axvline(np.median(down_percentages), color='salmon', linestyle='dotted', linewidth=2, 
                   label=f'Median: {np.median(down_percentages):.2f}%')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No DOWN gaps data", transform=ax2.transAxes, ha='center')
    ax2.set_title('DOWN Gaps Percentages (Absolute Values)')
    ax2.set_xlabel('Gap Percentage (Absolute %)')
    plt.suptitle('Histogram of Weekend Gap Percentages')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_max_deviation_histogram(gaps: List[Gap]):
    """
    Plot histograms of the maximum deviation % from Friday close until the gap is filled,
    separated between UP and DOWN gaps.
    
    For UP gaps, the deviation is the maximum percentage increase above Friday close.
    For DOWN gaps, the deviation is the maximum percentage drop (plotted as an absolute value).
    """
    up_max_devs = [gap.max_deviation_pct for gap in gaps 
                   if getattr(gap, 'filled', False) and gap.gap_direction.lower() == 'up' and gap.max_deviation_pct is not None]
    down_max_devs = [abs(gap.max_deviation_pct) for gap in gaps 
                     if getattr(gap, 'filled', False) and gap.gap_direction.lower() == 'down' and gap.max_deviation_pct is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    if up_max_devs:
        bins_up = np.linspace(min(up_max_devs), max(up_max_devs), 100)
        ax1.hist(up_max_devs, bins=bins_up, color='green', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(up_max_devs), color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {np.mean(up_max_devs):.2f}%')
        ax1.axvline(np.median(up_max_devs), color='purple', linestyle='dotted', linewidth=2, 
                    label=f'Median: {np.median(up_max_devs):.2f}%')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No UP gaps data", transform=ax1.transAxes, ha='center')
    ax1.set_title('UP Gaps Maximum Deviation (%)')
    ax1.set_xlabel('Max Deviation (%)')
    ax1.set_ylabel('Frequency')
    if down_max_devs:
        bins_down = np.linspace(min(down_max_devs), max(down_max_devs), 100)
        ax2.hist(down_max_devs, bins=bins_down, color='red', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(down_max_devs), color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Mean: {np.mean(down_max_devs):.2f}%')
        ax2.axvline(np.median(down_max_devs), color='purple', linestyle='dotted', linewidth=2, 
                    label=f'Median: {np.median(down_max_devs):.2f}%')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No DOWN gaps data", transform=ax2.transAxes, ha='center')
    ax2.set_title('DOWN Gaps Maximum Deviation (%)')
    ax2.set_xlabel('Max Deviation (%)')
    plt.suptitle('Histogram of Maximum Deviation (%) from Friday Close to Gap Fill')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Logger class to duplicate prints to terminal and file,
# filtering out ANSI escape codes for file output.
class Logger:
    # Compile regex to remove ANSI escape sequences.
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        # Remove ANSI escape codes for file output.
        clean_message = self.ansi_escape.sub('', message)
        self.log.write(clean_message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    # Redirect prints to both terminal and file
    sys.stdout = Logger("gaps_full_information.txt")
    
    df = load_ohlcv_data_df('BTC/USDT', '1h')
    print("Latest 100 rows of OHLCV data:")
    print(df.tail(100))
    
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert('UTC')
    df = add_historical_last_cme_closes(df, get_btc_cme_weekly_data())
    gaps = extract_weekend_gaps(df)
    print(f"\nFound {len(gaps)} weekend gaps:")
    for gap in gaps:
        print(gap)
    gaps = analyze_gap_fills(df, gaps)
    gaps_df, report_df = generate_gap_fill_report(gaps)
    
    plot_gap_fill_histogram(gaps)
    plot_gap_percentage_histogram(gaps)
    plot_max_deviation_histogram(gaps)
    
    print("\nGap Fill Information:")
    filled_gaps = [gap for gap in gaps if getattr(gap, 'filled', False)]
    unfilled_gaps = [gap for gap in gaps if not getattr(gap, 'filled', False)]
    
    print(f"\nFilled Gaps ({len(filled_gaps)}/{len(gaps)}, {len(filled_gaps)/len(gaps)*100:.2f}%):")
    for gap in filled_gaps:
        friday_date_str = gap.friday_date.strftime("%Y-%m-%d %H:%M") if gap.friday_date else "N/A"
        fill_date_str = gap.fill_date.strftime("%Y-%m-%d %H:%M") if gap.fill_date else "N/A"
        pct_color = GREEN if gap.gap_direction.lower() == "up" else RED if gap.gap_direction.lower() == "down" else YELLOW
        max_dev_str = f"{MAGENTA}{gap.max_deviation_pct:.2f}%{RESET}" if gap.max_deviation_pct is not None else "N/A"
        print(f"Gap #{gap.index}: {gap.gap_direction.upper()} {pct_color}{gap.gap_pct:.2f}%{RESET} - "
              f"Filled in {BLUE}{gap.days_to_fill:.2f}{RESET} days ({MAGENTA}{gap.candles_to_fill}{RESET} hours after Sunday Open) | "
              f"Friday Close: {CYAN}{gap.friday_close}{RESET} on {friday_date_str} | Gap Closed: {fill_date_str} | "
              f"Max Deviation: {max_dev_str}")
    
    print(f"\nUnfilled Gaps ({len(unfilled_gaps)}/{len(gaps)}, {len(unfilled_gaps)/len(gaps)*100:.2f}%):")
    for gap in unfilled_gaps:
        friday_date_str = gap.friday_date.strftime("%Y-%m-%d %H:%M") if gap.friday_date else "N/A"
        # Calculate days since the gap (using the gap's Sunday open date)
        days_since = (datetime.now(tz=gap.sunday_date.tzinfo) - gap.sunday_date).total_seconds() / (60 * 60 * 24)
        pct_color = GREEN if gap.gap_direction.lower() == "up" else RED if gap.gap_direction.lower() == "down" else YELLOW
        print(f"Gap #{gap.index}: {gap.gap_direction.upper()} {pct_color}{gap.gap_pct:.2f}%{RESET} - Not filled yet (Days since gap: {days_since:.2f}) | "
              f"Friday Close: {CYAN}{gap.friday_close}{RESET} on {friday_date_str}")
    
    print("\nGap Fill Statistics:")
    for col in report_df.columns:
        value = report_df[col].iloc[0]
        if isinstance(value, float):
            print(f"{col}: {BLUE}{value:.2f}{RESET}")
        else:
            print(f"{col}: {value}")
    
    gaps_df.to_csv("btc_weekend_gaps_with_fills.csv", index=False)
    report_df.to_csv("btc_weekend_gaps_fill_report.csv", index=False)
    print("\nGap data saved to CSV files")
    
    # Restore standard stdout if desired:
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    
    return df, gaps, gaps_df, report_df

if __name__ == "__main__":
    df, gaps, gaps_df, report_df = main()
    
