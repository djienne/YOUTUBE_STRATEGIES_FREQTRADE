"""
class BestHyperOptLoss(IHyperOptLoss):
This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
import math
from datetime import datetime

from pandas import DataFrame, date_range
from freqtrade.data.metrics import calculate_underwater
from freqtrade.optimize.hyperopt import IHyperOptLoss
from typing import Dict

def calc_sharpe(results: DataFrame, min_date: datetime, max_date: datetime) -> float:
    resample_freq = '1D'
    slippage_per_trade_ratio = 0.0005
    days_in_year = 365
    annual_risk_free_rate = 0.0
    risk_free_rate = annual_risk_free_rate / days_in_year

    # apply slippage per trade to profit_ratio
    results.loc[:, 'profit_ratio_after_slippage'] = \
        results['profit_ratio'] - slippage_per_trade_ratio

    # create the index within the min_date and end max_date
    t_index = date_range(start=min_date, end=max_date, freq=resample_freq,
                            normalize=True)

    sum_daily = (
        results.resample(resample_freq, on='close_date').agg(
            {"profit_ratio_after_slippage": 'sum'}).reindex(t_index).fillna(0)
    )

    total_profit = sum_daily["profit_ratio_after_slippage"] - risk_free_rate
    expected_returns_mean = total_profit.mean()
    up_stdev = total_profit.std()

    if up_stdev != 0:
        sharp_ratio = expected_returns_mean / up_stdev * math.sqrt(days_in_year)
    else:
        # Define high (negative) sharpe ratio to be clear that this is NOT optimal.
        sharp_ratio = -20.
    
    return sharp_ratio

class BestHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation takes only absolute profit into account, not looking at any other indicator.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime, config: Dict,
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results.
        """

        nb_trades = float(len(results['profit_abs']))
        nb_wins = float(sum(results['profit_abs'] > 0))
        total_profit2 = results['profit_abs'].sum()
        win_rate = nb_wins/nb_trades

        try:
            drawdown_df = calculate_underwater(
                results,
                value_col='profit_abs',
                starting_balance=config['dry_run_wallet']
            )
            max_drawdown = abs(min(drawdown_df['drawdown']))
            max_relative_drawdown = max(drawdown_df['drawdown_relative'])  # between 0 and 1
        except ValueError:
            max_relative_drawdown = 0

        starting_balance = float(config['dry_run_wallet'])

        profit_pc = total_profit2/starting_balance  # pc between 0 and 1

        # print(max_relative_drawdown)
        # adjusted drawdown function to account for how hard it is to compensate the loss (-10% -> +11.11%, -50% -> +100%)
        DDC = (1.0 / (1.0 - max_relative_drawdown) - 1.0)
        if DDC == 0:
            return 50
        # DDC and max_relative_drawdown are between 0 and 1

        sharpe = calc_sharpe(results, min_date, max_date)

        if profit_pc<200:
            return -1.0*profit_pc/DDC*sharpe
        else :
            return 50
        