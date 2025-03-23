import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from scipy.stats import johnsonsu, skew, kurtosis


def assign_market_signals(df):
    """
    Assign trading signals based on the average log returns of each market regime.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the columns 'log_returns' and 'market_regime'.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an additional column 'signal':
            - 1 for the regime with the highest average log_return (long)
            - -1 for the regime with the lowest average log_return (short)
            - 0 for the remaining regime (range)
    """
    regime_means = df.groupby('market_regime')['log_returns'].mean()
    long_regime = regime_means.idxmax()
    short_regime = regime_means.idxmin()
    signal_map = {
        regime: 1 if regime == long_regime else -1 if regime == short_regime else 0
        for regime in regime_means.index
    }
    df['signal'] = df['market_regime'].map(signal_map)
    return df


def get_btc_data(csv_path='BTC_daily_data.csv'):
    """
    Get BTC historical data from a local CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with BTC data.
    """
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        data.index = pd.to_datetime(data.index)
        return data
    else:
        raise ImportError('BTC_daily_data.csv not found')


def backtest_strategy(data, long_only=False, fee=0.001):
    """
    Backtest a trading strategy based on market regime signals.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the market data with signal column.
    long_only : bool, default=False
        If True, only long positions are taken (short signals become flat/0).
    fee : float, default=0.001
        Transaction fee (0.1%).
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with backtest results.
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Assign trading signals based on market regimes
    df = assign_market_signals(df)
    
    # If long_only is True, convert all -1 signals to 0 (no short positions)
    if long_only:
        df['signal'] = df['signal'].apply(lambda x: max(0, x))
    
    # Calculate transaction fees
    df['shifted_signal'] = df['signal'].shift(1)
    df['fee'] = np.where(df['signal'] != df['shifted_signal'], fee, 0)
    
    # Calculate strategy returns
    df['backtest'] = df['forward_returns'] * df['signal'] - df['fee']
    
    return df


def run_analysis_with_momentum(data, momentum_period):
    """
    Run the market regime detection and trading strategy with a specific momentum period.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the market data
    momentum_period : int
        Lookback period for momentum calculation
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with backtest results for the long+short strategy
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Calculate necessary columns
    df['dollar_volume'] = df['volume'] * df['close']
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['forward_returns'] = df['log_returns'].shift(-1)
    df['momentum'] = np.log(df['close'] / df['close'].shift(momentum_period))
    df.dropna(inplace=True)
    
    # Split into train and test
    data_train = df.loc['2015-01-01':'2021-01-01'].copy()
    data_test = df.loc['2021-01-02':].copy()
    
    # Initialize and train the HMM model
    np.random.seed(42)
    features = ['log_returns', 'momentum']
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, tol=0.1, algorithm='map')
    hmm_model.fit(data_train[features].values)
    
    # Predict market regimes
    data_train['market_regime'] = hmm_model.predict(data_train[features].values)
    data_test['market_regime'] = hmm_model.predict(data_test[features].values)
    
    # Backtest the strategy (long+short)
    data_test_long_short = backtest_strategy(data_test, long_only=False, fee=0.001)
    
    return data_test_long_short


# Get data
data = get_btc_data()

# Run strategies with different momentum periods
momentum_periods = range(7, 19)  # 7 to 18 inclusive
results = {}

for period in momentum_periods:
    results[period] = run_analysis_with_momentum(data, period)
    print(f"Completed analysis for momentum period {period}")

# Calculate buy & hold for reference
buy_hold = data.copy()
buy_hold['log_returns'] = np.log(buy_hold['close'] / buy_hold['close'].shift(1))
buy_hold['forward_returns'] = buy_hold['log_returns'].shift(-1)
buy_hold.dropna(inplace=True)
buy_hold_test = buy_hold.loc['2021-01-02':].copy()

# Plot comparison of strategies
plt.figure(figsize=(14, 10))
plt.title('Strategy Comparison - Momentum Periods 7-18 Days', fontsize=16)

# Use a colormap to differentiate the lines
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(momentum_periods)))

for i, period in enumerate(momentum_periods):
    result = results[period]
    plt.plot(
        np.exp(result['backtest'].cumsum()).mul(1000), 
        label=f'Period={period}',
        color=colors[i],
        linewidth=1.5
    )

# Plot Buy & Hold with a distinctive color and thicker line
plt.plot(
    np.exp(buy_hold_test['forward_returns'].cumsum()).mul(1000), 
    label='Buy & Hold',
    color='red',
    linewidth=2.5,
    linestyle='--'
)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylabel('Portfolio Value (starting at 1000)', fontsize=14)
plt.xlabel('Date', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('strategy_comparison_momentum_periods_grid.png', dpi=300)
plt.show()

# Calculate performance metrics for each momentum period
def calculer_metrics_portefeuille(returns_series, market_returns=None, risk_free_rate=0.05, capital=10000, trading_days=252):
    """
    Calculate portfolio performance metrics including CAPM alpha and beta.

    Parameters
    ----------
    returns_series : pd.Series
        Series of portfolio returns.
    market_returns : pd.Series, optional
        Series of market returns for calculating beta and alpha.
    risk_free_rate : float, default=0.05
        Annual risk-free rate (5% default).
    capital : float, default=10000
        Initial capital.
    trading_days : int, default=252
        Number of trading days in a year (252 is standard for most markets).

    Returns
    -------
    dict
        Dictionary of performance metrics.
    """
    returns_array = returns_series.values if isinstance(returns_series, pd.Series) else np.array(returns_series)
    
    # Annualize volatility using the square root of time rule
    daily_std = returns_array.std()
    sigma = daily_std * np.sqrt(trading_days)
    
    # Annualize mean return (simple multiplication by trading days)
    daily_mean = returns_array.mean()
    mu = daily_mean * trading_days
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/trading_days) - 1
    
    # Calculate excess returns for Sharpe ratio and CAPM
    excess_returns = returns_array - daily_rf
    
    # Sharpe ratio using annualized values
    sharpe = (mu - risk_free_rate) / sigma if sigma != 0 else np.nan
    
    # Portfolio value and drawdown calculations
    pf = capital * np.exp(returns_series).cumprod()
    max_dd = -min(pf / pf.expanding(1).max() - 1)
    calmar = mu / max_dd if max_dd != 0 else np.inf
    
    # Sortino ratio calculations (downside risk)
    neg_returns = returns_array[returns_array < daily_rf]  # Using risk-free as threshold
    if len(neg_returns) > 0:
        # Annualized downside deviation
        downside_deviation = np.sqrt(np.mean((neg_returns - daily_rf)**2) * trading_days)
        sortino = (mu - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.inf
    else:
        sortino = np.inf
    
    # Calculate CAPM beta and alpha if market returns are provided
    beta = np.nan
    alpha_annualized = np.nan
    
    if market_returns is not None and len(market_returns) == len(returns_array):
        # Calculate beta (covariance / variance)
        market_excess = market_returns - daily_rf
        covariance = np.cov(excess_returns, market_excess)[0, 1]
        market_variance = np.var(market_excess)
        beta = covariance / market_variance if market_variance != 0 else np.nan
        
        # Calculate alpha using the CAPM formula
        market_mean = market_returns.mean() * trading_days
        market_excess_return = market_mean - risk_free_rate
        expected_return = risk_free_rate + beta * market_excess_return
        alpha_annualized = mu - expected_return

    metrics = {
        'Sharpe Ratio': sharpe,
        'Capital Final': pf.iloc[-1],
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Sortino Ratio': sortino,
        'Beta': beta,
        'Alpha (annual)': alpha_annualized
    }
    
    # Format numerical values
    formatted_metrics = {}
    for k, v in metrics.items():
        if k == 'Capital Final':
            formatted_metrics[k] = f"{v:.2f}"
        elif k == 'Beta' or k == 'Alpha (annual)':
            formatted_metrics[k] = f"{v:.4f}" if not np.isnan(v) else "N/A"
        else:
            formatted_metrics[k] = f"{v:.2f}"
            
    return formatted_metrics

# Calculate and visualize performance metrics
risk_free_rate = 0.05
market_returns = buy_hold_test['forward_returns']
trading_days = 365

# Store metrics in a dictionary for plotting
metrics_data = {
    'Period': [],
    'Sharpe Ratio': [],
    'Max Drawdown': [],
    'Final Capital': [],
    'Calmar Ratio': [],
    'Sortino Ratio': []
}

# Calculate metrics for each period
for period, result in results.items():
    metrics = calculer_metrics_portefeuille(
        result['backtest'], 
        market_returns=market_returns, 
        risk_free_rate=risk_free_rate, 
        trading_days=trading_days
    )
    
    metrics_data['Period'].append(period)
    metrics_data['Sharpe Ratio'].append(float(metrics['Sharpe Ratio']))
    metrics_data['Max Drawdown'].append(float(metrics['Max Drawdown']))
    metrics_data['Final Capital'].append(float(metrics['Capital Final']))
    metrics_data['Calmar Ratio'].append(float(metrics['Calmar Ratio']))
    metrics_data['Sortino Ratio'].append(float(metrics['Sortino Ratio']))

# Convert to pandas DataFrame for easy plotting
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)

# Create subplots for key metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Metrics Across Momentum Periods', fontsize=16)

# Sharpe Ratio
axes[0, 0].plot(metrics_df['Period'], metrics_df['Sharpe Ratio'], 'o-', color='blue')
axes[0, 0].set_title('Sharpe Ratio')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(metrics_df['Period'])

# Final Capital
axes[0, 1].plot(metrics_df['Period'], metrics_df['Final Capital'], 'o-', color='green')
axes[0, 1].set_title('Final Capital')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(metrics_df['Period'])

# Max Drawdown
axes[1, 0].plot(metrics_df['Period'], metrics_df['Max Drawdown'], 'o-', color='red')
axes[1, 0].set_title('Maximum Drawdown')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(metrics_df['Period'])

# Calmar Ratio
axes[1, 1].plot(metrics_df['Period'], metrics_df['Calmar Ratio'], 'o-', color='purple')
axes[1, 1].set_title('Calmar Ratio')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(metrics_df['Period'])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('momentum_periods_metrics.png', dpi=300)
plt.show()

# Create a heatmap of the metrics to visualize the best periods
plt.figure(figsize=(12, 8))
sns.heatmap(
    metrics_df.set_index('Period')[['Sharpe Ratio', 'Calmar Ratio', 'Sortino Ratio', 'Max Drawdown']],
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=.5
)
plt.title('Performance Metrics Heatmap by Momentum Period', fontsize=16)
plt.tight_layout()
plt.savefig('momentum_periods_heatmap.png', dpi=300)
plt.show()

# Print summary of the best periods for each metric
print("\nBest Momentum Periods by Metric:")
best_sharpe = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax()]
best_calmar = metrics_df.loc[metrics_df['Calmar Ratio'].idxmax()]
best_sortino = metrics_df.loc[metrics_df['Sortino Ratio'].idxmax()]
best_capital = metrics_df.loc[metrics_df['Final Capital'].idxmax()]
lowest_drawdown = metrics_df.loc[metrics_df['Max Drawdown'].idxmin()]

print(f"Best Sharpe Ratio: Period {int(best_sharpe['Period'])} (Sharpe: {best_sharpe['Sharpe Ratio']:.2f})")
print(f"Best Calmar Ratio: Period {int(best_calmar['Period'])} (Calmar: {best_calmar['Calmar Ratio']:.2f})")
print(f"Best Sortino Ratio: Period {int(best_sortino['Period'])} (Sortino: {best_sortino['Sortino Ratio']:.2f})")
print(f"Best Final Capital: Period {int(best_capital['Period'])} (Capital: {best_capital['Final Capital']:.2f})")
print(f"Lowest Max Drawdown: Period {int(lowest_drawdown['Period'])} (Drawdown: {lowest_drawdown['Max Drawdown']:.2f})")