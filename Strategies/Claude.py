# Importations nécessaires
import numpy as np
import pandas as pd
import pandas_ta as ta
from functools import reduce 
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, CategoricalParameter

# Définition de la classe de stratégie
class Claude(IStrategy):
    """
    Stratégie optimisée pour Freqtrade
    Utilise plusieurs indicateurs avec des paramètres optimisables via hyperopt
    Timeframe: 1h
    """
    
    # Paramètres généraux de la stratégie
    timeframe = '1h'
    stoploss = -0.32  # Stop loss à -10%
    trailing_stop = True
    trailing_stop_positive = 0.314
    trailing_stop_positive_offset = 0.395
    trailing_only_offset_is_reached = True

    # Optional order type mapping.
    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}
    
    # Paramètres optimisables pour les indicateurs
    rsi_period = IntParameter(10, 30, default=11, space='buy', optimize=True)
    rsi_oversold = IntParameter(10, 40, default=26, space='buy', optimize=True)
    rsi_overbought = IntParameter(60, 90, default=80, space='sell', optimize=True)
    
    macd_fast = IntParameter(5, 20, default=20, space='buy', optimize=True)
    macd_slow = IntParameter(21, 40, default=20, space='buy', optimize=True)
    macd_signal = IntParameter(5, 15, default=5, space='buy', optimize=True)
    
    ema_short = IntParameter(5, 20, default=11, space='buy', optimize=True)
    ema_long = IntParameter(20, 40, default=30, space='buy', optimize=True)
    
    atr_period = IntParameter(10, 30, default=29, space='buy', optimize=True)
    atr_multiplier = DecimalParameter(1.0, 5.0, default=1.1, decimals=1, space='buy', optimize=True)
    
    # Paramètre pour choisir la méthode d'entrée
    entry_method = CategoricalParameter(['rsi', 'macd', 'ema_cross'], default='macd', space='buy', optimize=True)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calcule les indicateurs pour la stratégie
        """
        # RSI
        dataframe['rsi'] = ta.rsi(dataframe['close'], length=self.rsi_period.value)
        
        # MACD
        macd = ta.macd(dataframe['close'], fast=self.macd_fast.value, slow=self.macd_slow.value, signal=self.macd_signal.value)
        dataframe['macd'] = macd['MACD_' + str(self.macd_fast.value) + '_' + str(self.macd_slow.value) + '_' + str(self.macd_signal.value)]
        dataframe['macdsignal'] = macd['MACDs_' + str(self.macd_fast.value) + '_' + str(self.macd_slow.value) + '_' + str(self.macd_signal.value)]
        
        # EMA
        dataframe['ema_short'] = ta.ema(dataframe['close'], length=self.ema_short.value)
        dataframe['ema_long'] = ta.ema(dataframe['close'], length=self.ema_long.value)
        
        # ATR pour le trailing stop
        dataframe['atr'] = ta.atr(dataframe['high'], dataframe['low'], dataframe['close'], length=self.atr_period.value)
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Définit les conditions d'entrée en position
        """
        conditions = []
        
        if self.entry_method.value == 'rsi':
            conditions.append(dataframe['rsi'] > self.rsi_overbought.value)
        elif self.entry_method.value == 'macd':
            conditions.append((dataframe['macd'] > dataframe['macdsignal']) & 
                              (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)))
        elif self.entry_method.value == 'ema_cross':
            conditions.append((dataframe['ema_short'] > dataframe['ema_long']) & 
                              (dataframe['ema_short'].shift(1) <= dataframe['ema_long'].shift(1)))
        
        conditions.append(dataframe['volume'] > 0)  # S'assurer qu'il y a du volume
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Définit les conditions de sortie de position
        """
        conditions = []
        
        conditions.append(dataframe['rsi'] < self.rsi_oversold.value)
        conditions.append(dataframe['volume'] > 0)  # S'assurer qu'il y a du volume
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: 'datetime',
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Implémente un stop loss personnalisé basé sur l'ATR
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        # Calcul du stop loss basé sur l'ATR
        atr_stop = last_candle['close'] - (last_candle['atr'] * self.atr_multiplier.value)
        
        # Convertir en pourcentage
        atr_stop_percent = (atr_stop - trade.open_rate) / trade.open_rate
        
        # Retourner le maximum entre le stop loss par défaut et le stop loss ATR
        return max(self.stoploss, atr_stop_percent)

# Note: La fonction reduce est importée de functools, assurez-vous de l'ajouter aux importations si nécessaire