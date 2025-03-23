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


# Get data and calculate technical indicators
data = get_btc_data()
print(data)

data['dollar_volume'] = data['volume'] * data['close']
data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
data['forward_returns'] = data['log_returns'].shift(-1)
data['momentum'] = np.log(data['close'] / data['close'].shift(10))
data.dropna(inplace=True)

data_train = data.loc['2015-01-01':'2021-01-01'].copy()
data_test = data.loc['2021-01-02':].copy()

# Initialize and train the HMM model
np.random.seed(42)
features = ['log_returns', 'momentum']
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, tol=0.1, algorithm='map')
hmm_model.fit(data_train[features].values)
print("Model Score:", hmm_model.score(data_train[features].values))

data_train['market_regime'] = hmm_model.predict(data_train[features].values)
data_test['market_regime'] = hmm_model.predict(data_test[features].values)

# Plot Bitcoin closing prices colored by market regime
plt.figure(figsize=(12, 8))
plt.scatter(data_test.index, data_test['close'], c=data_test['market_regime'], cmap='viridis')
plt.title('Bitcoin and Market Regimes')
plt.colorbar(label='Market Regime')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.savefig('btc_market_regimes.png', dpi=300)
plt.show()

# Plot distributions of log-returns for each market regime
long_market = data_test[data_test['market_regime'] == 0]
range_market = data_test[data_test['market_regime'] == 1]
short_market = data_test[data_test['market_regime'] == 2]

sns.set_theme(style="dark", rc={"axes.grid": False})
plt.style.use('dark_background')
plt.figure(figsize=(12, 8))
sns.kdeplot(long_market['log_returns'], label='Long Market', color='lavender')
sns.kdeplot(range_market['log_returns'], label='Range Market', color='lightcoral')
sns.kdeplot(short_market['log_returns'], label='Short Market', color='darkviolet')
plt.axvline(long_market['log_returns'].mean(), color='lavender', linestyle='--')
plt.axvline(range_market['log_returns'].mean(), color='lightcoral', linestyle='--')
plt.axvline(short_market['log_returns'].mean(), color='darkviolet', linestyle='--')

y_coord, delta = 0.90, 0.07
for market, color, label in [(long_market, 'lavender', 'Long'),
                             (range_market, 'lightcoral', 'Range'),
                             (short_market, 'darkviolet', 'Short')]:
    mean_val = market['log_returns'].mean()
    std_dev = market['log_returns'].std()
    skewness_val = skew(market['log_returns'])
    kurt_val = kurtosis(market['log_returns'])
    plt.annotate(f'{label} Market Mean: {mean_val:.4f}', xy=(0.05, y_coord), xycoords='axes fraction', color=color)
    y_coord -= delta
    plt.annotate(f'{label} Market Std Dev: {std_dev:.4f}', xy=(0.05, y_coord), xycoords='axes fraction', color=color)
    y_coord -= delta
    plt.annotate(f'{label} Market Skewness: {skewness_val:.2f}', xy=(0.05, y_coord), xycoords='axes fraction', color=color)
    y_coord -= delta
    plt.annotate(f'{label} Market Kurtosis: {kurt_val:.2f}', xy=(0.05, y_coord), xycoords='axes fraction', color=color)
    y_coord -= delta

plt.legend()
plt.title('Distribution of Returns in Different Market Regimes')
plt.savefig('Dist_log_rets.png', dpi=300)
plt.show()

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

# Backtest both strategies
fee = 0.001  # Transaction fee (0.1%)

# Backtest with both long and short positions
data_test_long_short = backtest_strategy(data_test, long_only=False, fee=fee)

# Backtest with long-only positions
data_test_long_only = backtest_strategy(data_test, long_only=True, fee=fee)

# Calculate performance metrics for both strategies
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

# Calculate CAPM metrics with BTC as the market and 5% risk-free rate
risk_free_rate = 0.05  # 5% annual risk-free rate
market_returns = data_test['forward_returns']  # Using BTC returns as the market
trading_days = 365  # Using 365 days for crypto which trades 24/7

print("Long+Short Strategy Metrics:")
print(calculer_metrics_portefeuille(data_test_long_short['backtest'], market_returns=market_returns, risk_free_rate=risk_free_rate, trading_days=trading_days))
print("\nLong-Only Strategy Metrics:")
print(calculer_metrics_portefeuille(data_test_long_only['backtest'], market_returns=market_returns, risk_free_rate=risk_free_rate, trading_days=trading_days))

# Plot comparison of strategies
plt.figure(figsize=(12, 8))
plt.title('Strategy Comparison')
plt.plot(np.exp(data_test_long_short['backtest'].cumsum()).mul(1000), label='Long+Short Strategy')
plt.plot(np.exp(data_test_long_only['backtest'].cumsum()).mul(1000), label='Long-Only Strategy')
plt.plot(np.exp(data_test['forward_returns'].cumsum()).mul(1000), label='Buy & Hold')
plt.legend()
plt.savefig('strategy_comparison.png', dpi=300)
plt.show()

# Fit Johnson SU distribution to backtest PnL data for the preferred strategy
# Choose which strategy to analyze for VaR (can be changed to long_only if preferred)
np.random.seed(42)
pnl_values = data_test_long_short['backtest'].values


def neg_log_likelihood(params, data):
    a, b, loc, scale = params
    if b <= 0 or scale <= 0:
        return np.inf
    pdf_vals = johnsonsu.pdf(data, a, b, loc=loc, scale=scale)
    return -np.sum(np.log(pdf_vals + 1e-10))


def refine_bounds(params, factor=2):
    a, b, loc, scale = params
    a_bounds = (-1, 1) if np.abs(a) < 1e-6 else (a - np.abs(a), a + np.abs(a))
    b_bounds = (max(b / factor, 1e-10), b * factor)
    loc_bounds = (-1, 1) if np.abs(loc) < 1e-6 else (loc - np.abs(loc), loc + np.abs(loc))
    scale_bounds = (max(scale / factor, 1e-10), scale * factor)
    return [a_bounds, b_bounds, loc_bounds, scale_bounds]


def sample_random_in_bounds(bounds):
    return [np.random.uniform(low, high) for (low, high) in bounds]


def fit_johnsonsu(data, n_trials=10, method='L-BFGS-B', refine=True, refine_factor=2):
    best_obj = np.inf
    best_params = None
    broad_bounds = [(-np.inf, np.inf), (1e-10, np.inf), (-np.inf, np.inf), (1e-10, np.inf)]
    fixed_guess = [1, 1, np.mean(data), np.std(data)]
    trial_guesses = [fixed_guess]

    for _ in range(n_trials - 1):
        a_guess = np.random.uniform(-2, 2)
        b_guess = np.random.uniform(0.1, 2)
        loc_guess = np.random.uniform(np.min(data), np.max(data))
        scale_guess = np.random.uniform(np.std(data) * 0.5, np.std(data) * 1.5)
        trial_guesses.append([a_guess, b_guess, loc_guess, scale_guess])

    for guess in trial_guesses:
        result = minimize(
            neg_log_likelihood,
            x0=guess,
            args=(data,),
            method=method,
            bounds=broad_bounds,
            options={'maxiter': 10000, 'disp': False}
        )
        if np.isfinite(result.fun) and result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x

    if best_params is None or not np.isfinite(best_obj):
        raise RuntimeError("Initial broad search did not converge.")

    if refine:
        refined_bounds = refine_bounds(best_params, factor=refine_factor)
        refined_best_obj = best_obj
        refined_best_params = best_params
        refined_trial_guesses = [best_params]

        for _ in range(n_trials - 1):
            guess = sample_random_in_bounds(refined_bounds)
            refined_trial_guesses.append(guess)

        for guess in refined_trial_guesses:
            result = minimize(
                neg_log_likelihood,
                x0=guess,
                args=(data,),
                method=method,
                bounds=refined_bounds,
                options={'maxiter': 10000, 'disp': False}
            )
            if np.isfinite(result.fun) and result.fun < refined_best_obj:
                refined_best_obj = result.fun
                refined_best_params = result.x
        best_params = refined_best_params

    return best_params


def jitter_data(data, threshold=1e-9, jitter_scale=0.5e-2):
    data = np.array(data)
    jittered = data.copy()
    mask = np.abs(data) < threshold
    jittered[mask] += np.random.uniform(-jitter_scale, jitter_scale, size=np.sum(mask))
    return jittered


pnl_values = jitter_data(pnl_values)
params = fit_johnsonsu(pnl_values, n_trials=20, method='L-BFGS-B', refine=True, refine_factor=20)
print(params)

x = np.linspace(np.min(pnl_values) - 0.01, np.max(pnl_values) + 0.01, 1000)
pdf_fitted = johnsonsu.pdf(x, *params)
alpha_risk = 0.95
VAR_95_1j = -johnsonsu.ppf(1 - alpha_risk, *params)
approx_VAR_95_10j = VAR_95_1j * np.sqrt(10)

VAR_results = {
    'VaR 95% 1 Jour (%)': VAR_95_1j * 100,
    'VaR 95% 10 Jours (%)': approx_VAR_95_10j * 100
}
formatted_results = {k: f"{v:.2f}%" for k, v in VAR_results.items()}
print(formatted_results)

plt.figure(figsize=(8, 5))
plt.hist(pnl_values, bins=50, density=True, alpha=0.5, label='Filtered Daily PnL')
plt.plot(x, pdf_fitted, 'r-', lw=3, label='Fitted Johnson SU PDF')
plt.xlabel('Daily PnL')
plt.ylabel('Density')
plt.title('Johnson SU Fit on Filtered Daily PnL Data')
plt.legend()
plt.show()