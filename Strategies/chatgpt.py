import pandas as pd
import numpy as np
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import pandas_ta as ta
from freqtrade.strategy import CategoricalParameter, IntParameter, DecimalParameter

class chatgpt(IStrategy):
    """
    Stratégie Freqtrade personnalisée nommée 'chatgpt' générée par ChatGPT
    Utilise les indicateurs STC, CCI et Williams %R pour une période d'une heure.
    """

    # Paramètres de période et de stop-loss / ROI
    timeframe = '1h'
    stoploss = -0.75
    minimal_roi = {
        "0": 500.0  # ROI très élevé
    }
    
    # Définir les types d'ordres comme ordres de marché pour toutes les actions
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
    }

    # Paramètres optimisables
    stc_period = IntParameter(10, 90, default=23, space="buy", optimize=True)
    cci_period = IntParameter(10, 90, default=20, space="buy", optimize=True)
    wr_period = IntParameter(10, 90, default=14, space="buy", optimize=True)

    will_buy_level = DecimalParameter(-50.0, -10.0, default=-20, decimals=0, space='buy', optimize=True)
    will_sell_level = DecimalParameter(-130.0, -51.0, default=-80, decimals=0, space='buy', optimize=True)
    
    stc_thres = DecimalParameter(0.1, 0.9, default=0.5, decimals=1, space='buy', optimize=True)
    cci_thres = IntParameter(-70, 50, default=0, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Remplit le dataframe avec des indicateurs.
        - Schaff Trend Cycle (STC)
        - Commodity Channel Index (CCI)
        - Williams %R
        """
        # STC (Schaff Trend Cycle)
        sstc = ta.stc(dataframe['close'], length=self.stc_period.value, fast=23, slow=50)
        #print(sstc)
        dataframe['stc'] = sstc[list(sstc.items())[0][0]]

        # CCI (Commodity Channel Index)
        dataframe['cci'] = ta.cci(dataframe['high'], dataframe['low'], dataframe['close'], length=self.cci_period.value)

        # Williams %R
        dataframe['williamsr'] = ta.willr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.wr_period.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Définir les conditions d'entrée en utilisant STC, CCI et Williams %R.
        Entrer dans une position si :
        - STC est au-dessus de 0,5 (tendance haussière)
        - CCI est positif (au-dessus de 0)
        - Williams %R est en dessous de -20 (non suracheté)
        """
        dataframe.loc[
            (dataframe['stc'] > self.stc_thres.value) &
            (dataframe['cci'] > self.cci_thres.value) &
            (dataframe['williamsr'] < self.will_buy_level.value),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Conditions de sortie :
        Sortir si :
        - STC est en dessous de 0,5 (tendance baissière)
        - CCI est négatif (en dessous de 0)
        - Williams %R est au-dessus de -80 (non survendu)
        """
        dataframe.loc[
            (dataframe['stc'] < self.stc_thres.value) &
            (dataframe['cci'] < self.cci_thres.value) &
            (dataframe['williamsr'] > self.will_sell_level.value),
            'exit_long'] = 1

        return dataframe