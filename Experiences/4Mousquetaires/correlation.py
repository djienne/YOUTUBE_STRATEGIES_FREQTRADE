import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tvDatafeed import TvDatafeed, Interval
from xicorrelation import xicorr  # pip install xicorrelation
from scipy.stats import t
import matplotlib.pyplot as plt

# --- Helper Functions ---
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def download_data(symbol, exchange, interval, n_bars, output_file, data_dir):
    """
    Downloads OHLC data from TradingView and saves it to a CSV file.
    Returns the data as a DataFrame with a datetime index.
    """
    ensure_directory(data_dir)
    full_path = os.path.join(data_dir, output_file)
    use_existing = False
    if os.path.exists(full_path):
        mod_time = os.path.getmtime(full_path)
        # Use cached file if modified within the last 24 hours
        if time.time() - mod_time < 24 * 3600:
            use_existing = True
    if not use_existing:
        tv = TvDatafeed()
        data = tv.get_hist(symbol=symbol,
                           exchange=exchange,
                           interval=interval,
                           n_bars=n_bars,
                           extended_session=True)
        data.reset_index(drop=False, inplace=True)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['datetime'] = data['datetime'].dt.tz_localize('UTC')
        data = data.rename(columns={'datetime': 'date'})
        data.to_csv(full_path, index=False)
    df = pd.read_csv(full_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def reindex_asof(df, new_index):
    """
    Aligns df to new_index by performing an asof merge.
    For each timestamp in new_index, the last available row in df 
    that is less than or equal to that timestamp is used.
    This prevents any forward-looking bias.
    """
    df = df.sort_index()
    df_reset = df.reset_index()
    new_index_df = pd.DataFrame({'date': new_index})
    aligned = pd.merge_asof(new_index_df, df_reset, on='date', direction='backward')
    aligned.set_index('date', inplace=True)
    return aligned

def pearson_corr_and_p(x, y):
    """
    Calculate the Pearson correlation coefficient and its two-tailed p-value using numpy.
    Parameters:
        x (array-like): First set of observations.
        y (array-like): Second set of observations.
    Returns:
        r (float): Pearson correlation coefficient.
        p_value (float): Two-tailed p-value.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    
    n = len(x)
    if n < 3:
        raise ValueError("Not enough data points to compute correlation and p-value")
    
    r = np.corrcoef(x, y)[0, 1]
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 2))
    
    return r, p_value

# --- Parameters and Data Directory ---
data_dir = "./data"
n_bars = 10000  # Number of bars to download
daily_interval = Interval.in_daily
weekly_interval = Interval.in_weekly

# --- Download Data for Each Market ---
btc = download_data("BTCUSDT", "BINANCE", daily_interval, n_bars, "BTC_daily.csv", data_dir)
full_index = pd.date_range(start=btc.index.min(), end=btc.index.max(), freq='D')

nasdaq = download_data("NQ1!", "CME_MINI", daily_interval, n_bars, "NASDAQ_daily.csv", data_dir)
nasdaq = reindex_asof(nasdaq, full_index)

sp500 = download_data("ES1!", "CME_MINI", daily_interval, n_bars, "SP500_daily.csv", data_dir)
sp500 = reindex_asof(sp500, full_index)

dxy = download_data("DX1!", "ICEUS", daily_interval, n_bars, "DXY_daily.csv", data_dir)
dxy = reindex_asof(dxy, full_index)

m2 = download_data("WM2NS", "FRED", weekly_interval, n_bars, "M2_weekly.csv", data_dir)
m2 = reindex_asof(m2, full_index)

# --- Calculate Daily Log Returns ---
btc['log_return'] = np.log(btc['close'] / btc['close'].shift(1))
nasdaq['log_return'] = np.log(nasdaq['close'] / nasdaq['close'].shift(1))
sp500['log_return'] = np.log(sp500['close'] / sp500['close'].shift(1))
dxy['log_return'] = np.log(dxy['close'] / dxy['close'].shift(1))
m2['log_return'] = np.log(m2['close'] / m2['close'].shift(1))

returns_df = pd.DataFrame({
    'BTC': btc['log_return'],
    'NASDAQ': nasdaq['log_return'],
    'SP500': sp500['log_return'],
    'DXY': dxy['log_return'],
    'M2': m2['log_return']
}).dropna()

# --- Calculate Correlations ---
# Pearson using numpy method
r_nasdaq, p_nasdaq = pearson_corr_and_p(returns_df['BTC'].values, returns_df['NASDAQ'].values)
r_sp500,  p_sp500  = pearson_corr_and_p(returns_df['BTC'].values, returns_df['SP500'].values)
r_dxy,    p_dxy    = pearson_corr_and_p(returns_df['BTC'].values, returns_df['DXY'].values)
r_m2,     p_m2     = pearson_corr_and_p(returns_df['BTC'].values, returns_df['M2'].values)

# Xi correlations
xi_nasdaq, pvalue_nasdaq = xicorr(returns_df['BTC'].values, returns_df['NASDAQ'].values)
xi_sp500,  pvalue_sp500  = xicorr(returns_df['BTC'].values, returns_df['SP500'].values)
xi_dxy,    pvalue_dxy    = xicorr(returns_df['BTC'].values, returns_df['DXY'].values)
xi_m2,     pvalue_m2     = xicorr(returns_df['BTC'].values, returns_df['M2'].values)

# --- Summary Plot ---
markets = ['NASDAQ', 'SP500', 'DXY', 'M2']
pearson_corrs = [r_nasdaq, r_sp500, r_dxy, r_m2]
xi_corrs = [xi_nasdaq, xi_sp500, xi_dxy, xi_m2]
pearson_pvals = [p_nasdaq, p_sp500, p_dxy, p_m2]
xi_pvals = [pvalue_nasdaq, pvalue_sp500, pvalue_dxy, pvalue_m2]

x = np.arange(len(markets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

bars1 = ax.bar(x - width/2, pearson_corrs, width, label='Pearson', edgecolor='black')
bars2 = ax.bar(x + width/2, xi_corrs, width, label='xi', edgecolor='black')

# Define a constant y-coordinate for p-value annotations (all aligned below y=0)
annot_y = -0.3

# Annotate each bar with its p-value in scientific notation at a fixed y position below y=0
for i in range(len(markets)):
    ax.text(x[i] - width/2, annot_y, f'p={pearson_pvals[i]:.2e}', ha='center', va='top', fontsize=10, color='blue')
    ax.text(x[i] + width/2, annot_y, f'p={xi_pvals[i]:.2e}', ha='center', va='top', fontsize=10, color='red')

ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_title('Correlation between BTC and Other Markets (based on daily log-returns)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(markets, fontsize=12)
ax.legend(fontsize=12)

# Adjust y-axis limits to clearly show p-value annotations below 0
ax.set_ylim([-1.2, 1])
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
