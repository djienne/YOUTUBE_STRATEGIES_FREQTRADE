import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from scipy.stats import johnsonsu, skew, kurtosis


# ------------------ Data & Signal Functions ------------------

def assign_market_signals(df):
    """
    Assign trading signals based on the average log returns of each market regime.
    The regime with the highest average log_return is assigned 1 (long),
    and the regime with the lowest average log_return is assigned -1 (short).
    """
    regime_means = df.groupby('market_regime')['log_returns'].mean()
    long_regime = regime_means.idxmax()
    short_regime = regime_means.idxmin()
    signal_map = {regime: 1 if regime == long_regime else -1 for regime in regime_means.index}
    df['signal'] = df['market_regime'].map(signal_map)
    return df


def get_SPXTR_data(csv_path='SPXTR_daily_data.csv'):
    """
    Get SPXTR historical data from a local CSV file.
    """
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        data.index = pd.to_datetime(data.index)
        return data
    else:
        raise ImportError('SPXTR_daily_data.csv not found')


# ------------------ Data Loading & Preprocessing ------------------

MOM_PERIOD = 10

data = get_SPXTR_data()
print(data)

data['dollar_volume'] = data['volume'] * data['close']
data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
data['forward_returns'] = data['log_returns'].shift(-1)
data['momentum'] = np.log(data['close'] / data['close'].shift(MOM_PERIOD))

import talib

# Method 2: Alternative syntax
data['rsi'] = talib.RSI(data['close'], timeperiod=14)

data.dropna(inplace=True)

data_train = data.loc['2000-01-01':'2015-01-01'].copy()
data_test = data.loc['2015-01-02':].copy()


# ------------------ HMM Model ------------------

np.random.seed(42)
features = ['log_returns','momentum']
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=10000, tol=0.1, algorithm='map')
hmm_model.fit(data_train[features].values)
print("Model Score:", hmm_model.score(data_train[features].values))

# For training data, assign market regimes using vectorized prediction
data_train['market_regime'] = hmm_model.predict(data_train[features].values)

# ------------------ Market Regime Assignment for Test Data ------------------

# Create a copy for fast (vectorized) market regime predictions
data_test_fast = data_test.copy()
data_test_fast['market_regime'] = hmm_model.predict(data_test_fast[features].values)

# Create a copy for slow (row-by-row) market regime predictions to avoid forward-looking bias
data_test_slow = data_test.copy()
market_regime_predictions = []
for i in range(len(data_test_slow)):
    # Extract features for rows up to the current row only
    current_row_features = data_test_slow[features].iloc[:i+1].values
    current_prediction = hmm_model.predict(current_row_features)
    market_regime_predictions.append(current_prediction[-1])
data_test_slow['market_regime'] = market_regime_predictions


# ------------------ Plot: Market Regimes (Fast Prediction) ------------------

# Use discrete colors for regimes (0 and 1)
cmap = mcolors.ListedColormap(['skyblue', 'lightcoral'])
bounds = [-0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(12, 8))
plt.scatter(data_test_fast.index, data_test_fast['close'], c=data_test_fast['market_regime'], cmap=cmap, norm=norm)
plt.title('S&P 500 Total Return and Market Regimes (Fast Prediction, may repaint)')
cbar = plt.colorbar(ticks=[0, 1])
cbar.set_label('Market Regime')
cbar.set_ticklabels(['Regime 0', 'Regime 1'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.savefig('sp500nr_market_regimes_fast.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(data_test_slow.index, data_test_slow['close'], c=data_test_slow['market_regime'], cmap=cmap, norm=norm)
plt.title('S&P 500 Total Return and Market Regimes (Slow NO-REPAINT GURANTEED Prediction)')
cbar = plt.colorbar(ticks=[0, 1])
cbar.set_label('Market Regime')
cbar.set_ticklabels(['Regime 0', 'Regime 1'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.savefig('sp500nr_market_regimes_slow.png', dpi=300)
plt.show()


# ------------------ Plot: Log-Return Distributions ------------------

# For this plot, we assume regimes 0 and 1 from the fast prediction
# market0 = data_test_fast[data_test_fast['market_regime'] == 0]
# market1 = data_test_fast[data_test_fast['market_regime'] == 1]

# sns.set_theme(style="dark", rc={"axes.grid": False})
# plt.style.use('dark_background')
# plt.figure(figsize=(12, 8))
# sns.kdeplot(market0['log_returns'], label='Market Regime 0', color='skyblue')
# sns.kdeplot(market1['log_returns'], label='Market Regime 1', color='lightcoral')
# plt.axvline(market0['log_returns'].mean(), color='skyblue', linestyle='--')
# plt.axvline(market1['log_returns'].mean(), color='lightcoral', linestyle='--')
# plt.legend()
# plt.title('Distribution of Log Returns by Market Regime (Fast Prediction)')
# plt.savefig('Dist_log_rets_fast.png', dpi=300)
# plt.show()


# ------------------ Backtesting ------------------

def backtest_strategy(data, long_only=False, fee=0.01, funding_fee_annual=0.08):
    """
    Backtest a trading strategy based on market regime signals.
    Short positions incur an annual funding fee (applied daily) and transaction fees are charged on signal changes.
    """
    df = data.copy()
    df = assign_market_signals(df)
    if long_only:
        df['signal'] = df['signal'].apply(lambda x: max(0, x))
    df['shifted_signal'] = df['signal'].shift(1)
    df['fee'] = np.where(df['signal'] != df['shifted_signal'], fee, 0)
    daily_funding_fee = funding_fee_annual / 365
    df['funding_fee'] = np.where(df['signal'] == -1, daily_funding_fee, 0)
    df['backtest'] = df['forward_returns'] * df['signal'] - df['fee'] - df['funding_fee']
    return df

fee = 0.001  # 0.1% transaction fee

# Backtesting Long+Short strategy using fast regime predictions (for reference)
data_test_long_short = backtest_strategy(data_test_fast, long_only=False, fee=fee)

# ------------------ Backtesting: Long-Only Strategy with Fast vs Slow Regime Predictions ------------------

data_test_long_only_fast = backtest_strategy(data_test_fast, long_only=True, fee=fee)
data_test_long_only_slow = backtest_strategy(data_test_slow, long_only=True, fee=fee)


# ------------------ Performance Metrics (Optional) ------------------
def calculer_metrics_portefeuille(returns_series, market_returns=None, risk_free_rate=0.05,
                                  capital=10000, trading_days=252):
    """
    Calculate portfolio performance metrics including CAPM alpha, beta, and CAGR.
    "Max Drawdown" is returned as a decimal and will be formatted as a percentage.
    """
    returns_array = returns_series.values if isinstance(returns_series, pd.Series) else np.array(returns_series)
    daily_std = returns_array.std()
    sigma = daily_std * np.sqrt(trading_days)
    daily_mean = returns_array.mean()
    mu = daily_mean * trading_days
    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = returns_array - daily_rf
    sharpe = (mu - risk_free_rate) / sigma if sigma != 0 else np.nan
    pf = capital * np.exp(returns_series).cumprod()
    max_dd = -min(pf / pf.expanding(1).max() - 1)
    calmar = mu / max_dd if max_dd != 0 else np.inf
    neg_returns = returns_array[returns_array < daily_rf]
    if len(neg_returns) > 0:
        downside_deviation = np.sqrt(np.mean((neg_returns - daily_rf) ** 2) * trading_days)
        sortino = (mu - risk_free_rate) / downside_deviation if downside_deviation != 0 else np.inf
    else:
        sortino = np.inf
    beta = np.nan
    alpha_annualized = np.nan
    if market_returns is not None and len(market_returns) == len(returns_array):
        market_excess = market_returns - daily_rf
        covariance = np.cov(excess_returns, market_excess)[0, 1]
        market_variance = np.var(market_excess)
        beta = covariance / market_variance if market_variance != 0 else np.nan
        market_mean = market_returns.mean() * trading_days
        market_excess_return = market_mean - risk_free_rate
        expected_return = risk_free_rate + beta * market_excess_return
        alpha_annualized = mu - expected_return
    total_days = len(returns_series)
    total_years = total_days / trading_days
    cagr = (pf.iloc[-1] / capital) ** (1 / total_years) - 1

    metrics = {
        'Sharpe Ratio': sharpe,
        'Capital Final': pf.iloc[-1],
        'Max Drawdown': max_dd,  # as a decimal (will be formatted as %)
        'Calmar Ratio': calmar,
        'Sortino Ratio': sortino,
        'Beta': beta,
        'Alpha (annual)': alpha_annualized,
        'CAGR (%)': cagr * 100  # percentage
    }
    
    formatted_metrics = {}
    for k, v in metrics.items():
        if k == 'Capital Final':
            formatted_metrics[k] = f"{v:.2f}"
        elif k in ['Beta', 'Alpha (annual)']:
            formatted_metrics[k] = f"{v:.4f}" if not np.isnan(v) else "N/A"
        elif k == 'Max Drawdown':
            formatted_metrics[k] = f"{v * 100:.2f}%"  # convert to percentage
        elif k == 'CAGR (%)':
            formatted_metrics[k] = f"{v:.2f}%"
        else:
            formatted_metrics[k] = f"{v:.2f}"
    return formatted_metrics

# ------------------ Plot: Long-Only Performance Comparison ------------------
# Compare the long-only performance using fast (vectorized) vs slow (row-by-row) market regime predictions

plt.figure(figsize=(12, 8))
plt.title('S&P 500 Total Return Long-Only Strategy Performance: Fast (May Repaint) vs Slow (No-Repaint) Market Regime Predictions')
plt.plot(np.exp(data_test_long_only_fast['backtest'].cumsum()).mul(1000), label='Long-Only (Fast)')
plt.plot(np.exp(data_test_long_only_slow['backtest'].cumsum()).mul(1000), label='Long-Only (Slow Row-by-Row NO REPAINT GUARANTEED)')
plt.plot(np.exp(data_test['forward_returns'].cumsum()).mul(1000), label='Buy & Hold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.savefig('long_only_comparison.png', dpi=300)
plt.show()


# ------------------ VaR Metrics Function and Supporting Functions ------------------

def calculate_var_metrics(pnl_values, alpha_risk=0.95, n_trials=20, method='L-BFGS-B',
                          refine=True, refine_factor=20):
    """
    Calculate VaR metrics using a Johnson SU fit on pnl data.
    Returns VaR 95% for 1 day and 10 days (as percentages) rounded to 2 decimals.
    """
    pnl_values = jitter_data(pnl_values)
    params = fit_johnsonsu(pnl_values, n_trials=n_trials, method=method,
                           refine=refine, refine_factor=refine_factor)
    VAR_95_1j = -johnsonsu.ppf(1 - alpha_risk, *params)
    approx_VAR_95_10j = VAR_95_1j * np.sqrt(10)
    return {'VaR 95% 1 Day (%)': round(VAR_95_1j * 100, 2),
            'VaR 95% 10 Days (%)': round(approx_VAR_95_10j * 100, 2)}

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


# # ------------------ Plot: Johnson SU Fit for PnL ------------------

# x = np.linspace(np.min(jitter_data(data_test_long_short['backtest'].values)) - 0.01,
#                 np.max(jitter_data(data_test_long_short['backtest'].values)) + 0.01, 1000)
# params = fit_johnsonsu(jitter_data(data_test_long_short['backtest'].values), n_trials=20, method='L-BFGS-B', refine=True, refine_factor=20)
# pdf_fitted = johnsonsu.pdf(x, *params)
# plt.figure(figsize=(8, 5))
# plt.hist(jitter_data(data_test_long_short['backtest'].values), bins=50, density=True, alpha=0.5, label='Filtered Daily PnL')
# plt.plot(x, pdf_fitted, 'r-', lw=3, label='Fitted Johnson SU PDF')
# plt.xlabel('Daily PnL')
# plt.ylabel('Density')
# plt.title('Johnson SU Fit on Filtered Daily PnL Data')
# plt.legend()
# plt.show()


# ------------------ Portfolio Metrics for Slow Long-Only Strategy ------------------
# Calculate and print portfolio metrics for the slow long-only backtest

risk_free_rate = 0.05  # 5% annual risk-free rate
trading_days = 252   # Using 252 trading days per year
market_returns = data_test['forward_returns']

fast_long_only_metrics = calculer_metrics_portefeuille(data_test_long_only_fast['backtest'],
                                                       market_returns=market_returns,
                                                       risk_free_rate=risk_free_rate,
                                                       trading_days=trading_days)

print("\nPortfolio Metrics for Fast May-Repaint Long-Only Strategy:")
for key, value in fast_long_only_metrics.items():
    print(f"{key}: {value}")

slow_long_only_metrics = calculer_metrics_portefeuille(data_test_long_only_slow['backtest'],
                                                       market_returns=market_returns,
                                                       risk_free_rate=risk_free_rate,
                                                       trading_days=trading_days)

print("\nPortfolio Metrics for Slow No-Repaint Long-Only Strategy:")
for key, value in slow_long_only_metrics.items():
    print(f"{key}: {value}")


risk_free_rate = 0.05  # 5% annual risk-free rate
trading_days = 252   # Using 252 trading days per year
market_returns = data_test['forward_returns']
