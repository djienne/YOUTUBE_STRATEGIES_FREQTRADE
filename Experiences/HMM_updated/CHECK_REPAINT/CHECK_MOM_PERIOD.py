import os
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from scipy.stats import johnsonsu, skew, kurtosis
import talib

mom_period_range = range(2, 100, 2)  # MOM_PERIOD values from 5 to 20
nb_to_use_range = range(1, 10, 1)

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

# ------------------ Main Routine: Vary MOM_PERIOD ------------------

# Load the original data once
data_original = get_SPXTR_data()
print(data_original.head())

# Parameters for the experiment

starting_capital = 1000
fee = 0.001  # 0.1% transaction fee

# Dictionaries to store results
mom_results = {}         # Final portfolio value for each (MOM_PERIOD, nb_to_use)
equity_curves = {}       # Equity curve (time series) for each combination
buy_hold_curves = {}     # Buy and hold equity curve for each combination

for nb_to_use in nb_to_use_range:
    for MOM_PERIOD in mom_period_range:
        # Work on a fresh copy for each combination
        data = data_original.copy()
        
        # Calculate additional columns
        data['dollar_volume'] = data['volume'] * data['close']
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['forward_returns'] = data['log_returns'].shift(-1)
        for ii in range(nb_to_use+1):
            data[f'momentum_{ii}'] = np.log(data['close'] / data['close'].shift(MOM_PERIOD+ii))
        data['rsi'] = talib.RSI(data['close'], timeperiod=14)
        data.dropna(inplace=True)
        
        # Split into training and test sets
        data_train = data.loc['2000-01-01':'2015-01-01'].copy()
        data_test = data.loc['2015-01-02':].copy()
        
        # ------------------ HMM Model ------------------
        np.random.seed(42)
        features = ['log_returns'] + [f'momentum_{ii}' for ii in range(nb_to_use+1)]
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=10000, tol=0.1, algorithm='map')
        hmm_model.fit(data_train[features].values)
        print(f"MOM_PERIOD {MOM_PERIOD}  nb_to_use {nb_to_use}: Model Score:", 
              hmm_model.score(data_train[features].values))
        
        # For training data, assign market regimes (for model understanding)
        data_train['market_regime'] = hmm_model.predict(data_train[features].values)
        
        # ------------------ Market Regime Assignment for Test Data (Slow Prediction) ------------------
        data_test_slow = data_test.copy()
        market_regime_predictions = []
        for i in range(len(data_test_slow)):
            # Use rows up to and including the current row for prediction
            current_row_features = data_test_slow[features].iloc[:i+1].values
            current_prediction = hmm_model.predict(current_row_features)
            market_regime_predictions.append(current_prediction[-1])
        data_test_slow['market_regime'] = market_regime_predictions
        
        # ------------------ Backtesting with Slow (Row-by-Row) Regime Predictions ------------------
        data_test_long_only_slow = backtest_strategy(data_test_slow, long_only=True, fee=fee)
        
        # Compute cumulative portfolio value using the backtest returns
        eq_curve = starting_capital * np.exp(data_test_long_only_slow['backtest'].cumsum())
        portfolio_value = eq_curve.iloc[-1]
        mom_results[(MOM_PERIOD, nb_to_use)] = portfolio_value
        equity_curves[(MOM_PERIOD, nb_to_use)] = eq_curve
        
        # Compute the buy-and-hold equity curve using the test set close prices
        buy_hold = starting_capital * (data_test['close'] / data_test['close'].iloc[0])
        buy_hold_curves[(MOM_PERIOD, nb_to_use)] = buy_hold
        
        print(f"MOM_PERIOD {MOM_PERIOD} nb_to_use {nb_to_use} -> Final Portfolio Value: {portfolio_value:.2f}")

# ------------------ Final Plot: Performance vs MOM_PERIOD (Line Plot for a Fixed nb_to_use) ------------------
# For the line plot, we demonstrate performance for a fixed nb_to_use (e.g., the smallest value)
selected_nb = nb_to_use_range[0]
mom_period_values = []
portfolio_values_line = []
for (MOM_PERIOD, nb_to_use), portfolio_value in mom_results.items():
    if nb_to_use == selected_nb:
        mom_period_values.append(MOM_PERIOD)
        portfolio_values_line.append(portfolio_value)

plt.figure(figsize=(10, 6))
plt.plot(mom_period_values, portfolio_values_line, marker='o', linestyle='-')
plt.title(f'Final Portfolio Value vs MOM_PERIOD (nb_to_use = {selected_nb})')
plt.xlabel('MOM_PERIOD')
plt.ylabel('Final Portfolio Value (Starting Capital = 1000)')
plt.grid(True)
plt.savefig('portfolio_vs_mom_period.png', dpi=300)
plt.show()

# ------------------ Heatmap: Final Portfolio Value vs MOM_PERIOD and nb_to_use ------------------
mom_periods = sorted(set(key[0] for key in mom_results.keys()))
nb_values = sorted(set(key[1] for key in mom_results.keys()))
heatmap_data = np.zeros((len(mom_periods), len(nb_values)))
for i, mp in enumerate(mom_periods):
    for j, nb in enumerate(nb_values):
        heatmap_data[i, j] = mom_results[(mp, nb)]

plt.figure(figsize=(10, 8))
im = plt.imshow(heatmap_data, aspect='auto', origin='lower', interpolation='nearest')
plt.colorbar(im, label='Final Portfolio Value')
plt.xticks(ticks=np.arange(len(nb_values)), labels=nb_values)
plt.yticks(ticks=np.arange(len(mom_periods)), labels=mom_periods)
plt.xlabel('nb_to_use')
plt.ylabel('MOM_PERIOD')
plt.title('Heatmap: Final Portfolio Value vs MOM_PERIOD and nb_to_use')
plt.savefig('portfolio_heatmap.png', dpi=300)
plt.show()

# ------------------ Find and Print the Best Parameter Combination ------------------
best_params = max(mom_results, key=mom_results.get)
best_value = mom_results[best_params]
print("\nBest Combination:")
print(f"  MOM_PERIOD: {best_params[0]}, nb_to_use: {best_params[1]}")
print(f"  Final Portfolio Value: {best_value:.2f}")

# ------------------ Plot Equity Curve vs Buy-and-Hold for the Best Combination ------------------
best_equity_curve = equity_curves[best_params]
best_buy_hold_curve = buy_hold_curves[best_params]

plt.figure(figsize=(12, 7))
plt.plot(best_equity_curve.index, best_equity_curve.values, label='Strategy Equity Curve')
plt.plot(best_buy_hold_curve.index, best_buy_hold_curve.values, label='Buy and Hold Equity Curve', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.title(f'Equity Curve Comparison\n(MOM_PERIOD: {best_params[0]}, nb_to_use: {best_params[1]})')
plt.legend()
plt.grid(True)
plt.savefig('equity_curve_best_vs_buy_and_hold.png', dpi=300)
plt.show()
