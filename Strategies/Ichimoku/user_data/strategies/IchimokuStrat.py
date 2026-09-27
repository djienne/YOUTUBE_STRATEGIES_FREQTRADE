# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional, Union  # noqa
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Ichimoku(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '4h'
    USE_TALIB = False

    # Can this strategy go short?
    can_short: bool = False

    minimal_roi = {
        "0": 5000.0
    }

    stoploss = -0.75
    trailing_stop = False
    process_only_new_candles: bool = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    use_custom_stoploss: bool = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 250

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    # These are the 'Crypto style values' (uncomment if you want to use these)
    # TS = 20
    TS = IntParameter(10, 40, default=20, space="buy", optimize=True)
    KS = IntParameter(30, 120, default=60, space="buy", optimize=True)
    SS = IntParameter(60, 240, default=120, space="buy", optimize=True)
    # SS = 120
    # CS = 60
    ATR_length = IntParameter(7, 21, default=14, space="buy", optimize=True)
    ATR_Multip = DecimalParameter(1.0, 6.0, decimals=1, default=4, space="buy", optimize=True)
    rr = DecimalParameter(1.0, 4.0, decimals=1, default=4, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        if self.dp.runmode.value in ('live', 'dry_run'):
            self.USE_TALIB = False
        else:
            self.USE_TALIB = True

        ichimo = pta.ichimoku(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'],
                              tenkan=int(self.TS.value), kijun=int(self.KS.value), senkou=int(self.SS.value),
                              include_chikou=True)[0]
        # print(ichimo)
        dataframe['tenkan'] = ichimo[f'ITS_{int(self.TS.value)}'].copy()
        dataframe['kijun'] = ichimo[f'IKS_{int(self.KS.value)}'].copy()
        dataframe['senkanA'] = ichimo[f'ISA_{int(self.TS.value)}'].copy()
        dataframe['senkanB'] = ichimo[f'ISB_{int(self.KS.value)}'].copy()
        dataframe['chiko'] = ichimo[f'ICS_{int(self.KS.value)}'].copy()

        dataframe['ATR'] = pta.atr(dataframe['high'], dataframe['low'], dataframe['close'],
                                   length=int(self.ATR_length.value), talib=self.USE_TALIB)
        # print(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['senkanA'])
                    &
                    (dataframe['close'] > dataframe['senkanB'])
                    &
                    (dataframe['close'] > dataframe['tenkan'])
                    &
                    (dataframe['senkanB'] > dataframe['senkanA'])
                    &
                    (dataframe['tenkan'] > dataframe['kijun'])
            ),
            'enter_long'] = 1
            
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['senkanA'])
                    &
                    (dataframe['close'] < dataframe['senkanB'])
                    &
                    (dataframe['close'] < dataframe['tenkan'])
                    &
                    (dataframe['senkanB'] < dataframe['senkanA'])
                    &
                    (dataframe['tenkan'] < dataframe['kijun'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['high'] < dataframe['tenkan'])
            ),
            'exit_long'] = 1
        dataframe.loc[
            (
                (dataframe['low'] > dataframe['tenkan'])
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe['date'] == trade_date]

        # SL
        c2 = False
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            if not trade.is_short:
                c2 = current_rate < trade.open_rate - trade_candle['ATR'] * float(self.ATR_Multip.value)
            else:
                c2 = current_rate > trade.open_rate + trade_candle['ATR'] * float(self.ATR_Multip.value)
            if c2:
                return -0.0001

        # TP
        c1 = False
        if not trade_candle.empty:
            trade_candle = trade_candle.squeeze()
            dist = trade_candle['ATR'] * self.ATR_Multip.value
            if not trade.is_short:
                c1 = current_rate > trade.open_rate + dist * float(self.rr.value)
            else:
                c1 = current_rate < trade.open_rate - dist * float(self.rr.value)
            if c1:
                return -0.0001

        return -0.75
