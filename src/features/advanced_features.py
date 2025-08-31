from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import ta
from ..utils.logging import get_logger

logger = get_logger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for Indian volatility prediction."""
    
    def __init__(self):
        self.feature_names = []
    
    def create_technical_indicators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators."""
        features = pd.DataFrame(index=ohlcv.index)
        
        # Volatility indicators
        features['atr_14'] = ta.volatility.average_true_range(ohlcv['high'], ohlcv['low'], ohlcv['close'], window=14)
        features['atr_21'] = ta.volatility.average_true_range(ohlcv['high'], ohlcv['low'], ohlcv['close'], window=21)
        features['bb_width'] = ta.volatility.bollinger_wband(ohlcv['close'], window=20, window_dev=2)
        features['kc_width'] = ta.volatility.keltner_channel_wband(ohlcv['high'], ohlcv['low'], ohlcv['close'], window=20)
        
        # Momentum indicators
        features['rsi_14'] = ta.momentum.rsi(ohlcv['close'], window=14)
        features['rsi_21'] = ta.momentum.rsi(ohlcv['close'], window=21)
        features['macd'] = ta.trend.macd_diff(ohlcv['close'])
        features['macd_signal'] = ta.trend.macd_signal(ohlcv['close'])
        features['stoch_k'] = ta.momentum.stoch(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        features['stoch_d'] = ta.momentum.stoch_signal(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        
        # Trend indicators
        features['sma_10'] = ta.trend.sma_indicator(ohlcv['close'], window=10)
        features['sma_20'] = ta.trend.sma_indicator(ohlcv['close'], window=20)
        features['sma_50'] = ta.trend.sma_indicator(ohlcv['close'], window=50)
        features['ema_12'] = ta.trend.ema_indicator(ohlcv['close'], window=12)
        features['ema_26'] = ta.trend.ema_indicator(ohlcv['close'], window=26)
        
        # Price relative to moving averages
        features['price_sma20_ratio'] = ohlcv['close'] / features['sma_20']
        features['price_sma50_ratio'] = ohlcv['close'] / features['sma_50']
        
        # Volume indicators
        features['vwap'] = ta.volume.volume_weighted_average_price(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
        features['volume_sma'] = ta.volume.volume_sma(ohlcv['close'], ohlcv['volume'], window=20)
        features['volume_ratio'] = ohlcv['volume'] / features['volume_sma']
        
        return features
    
    def create_volatility_estimators(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Create various volatility estimators."""
        features = pd.DataFrame(index=ohlcv.index)
        
        # Close-to-close volatility
        returns = ohlcv['close'].pct_change()
        for window in [5, 10, 20, 60]:
            features[f'cc_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson estimator
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(ohlcv['high'] / ohlcv['low']) ** 2).rolling(20).mean() * 252
        )
        
        # Garman-Klass estimator
        features['gk_vol'] = np.sqrt(
            (0.5 * (np.log(ohlcv['high'] / ohlcv['low']) ** 2) -
             (2 * np.log(2) - 1) * (np.log(ohlcv['close'] / ohlcv['open']) ** 2)).rolling(20).mean() * 252
        )
        
        # Rogers-Satchell estimator
        features['rs_vol'] = np.sqrt(
            (np.log(ohlcv['high'] / ohlcv['close']) * np.log(ohlcv['high'] / ohlcv['open']) +
             np.log(ohlcv['low'] / ohlcv['close']) * np.log(ohlcv['low'] / ohlcv['open'])).rolling(20).mean() * 252
        )
        
        # Yang-Zhang estimator
        prev_close = ohlcv['close'].shift(1)
        overnight_ret = np.log(ohlcv['open'] / prev_close)
        rs_component = (np.log(ohlcv['high'] / ohlcv['close']) * np.log(ohlcv['high'] / ohlcv['open']) +
                       np.log(ohlcv['low'] / ohlcv['close']) * np.log(ohlcv['low'] / ohlcv['open']))
        open_close_ret = np.log(ohlcv['close'] / ohlcv['open'])
        
        k = 0.34 / (1.34 + (len(ohlcv) + 1) / (len(ohlcv) - 1))
        features['yz_vol'] = np.sqrt(
            (overnight_ret.rolling(20).var() + k * open_close_ret.rolling(20).var() + 
             (1 - k) * rs_component.rolling(20).mean()) * 252
        )
        
        return features
    
    def create_regime_features(self, data: pd.DataFrame, vix_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create market regime indicators."""
        features = pd.DataFrame(index=data.index)
        
        if 'close' in data.columns:
            returns = data['close'].pct_change()
            
            # Volatility clustering
            abs_returns = returns.abs()
            features['vol_clustering'] = abs_returns.rolling(20).corr(abs_returns.shift(1))
            
            # Momentum regimes
            features['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
            features['momentum_60d'] = data['close'] / data['close'].shift(60) - 1
            
            # Trend strength
            sma_20 = data['close'].rolling(20).mean()
            sma_60 = data['close'].rolling(60).mean()
            features['trend_strength'] = (sma_20 - sma_60) / sma_60
            
            # Volatility regimes
            vol_20d = returns.rolling(20).std() * np.sqrt(252)
            vol_60d = returns.rolling(60).std() * np.sqrt(252)
            features['vol_regime'] = vol_20d / vol_60d
        
        # VIX-based regime features
        if vix_data is not None and not vix_data.empty:
            vix_aligned = vix_data.reindex(data.index, method='ffill')
            if 'close' in vix_aligned.columns:
                vix_close = vix_aligned['close']
                features['vix_level'] = vix_close
                features['vix_low_regime'] = (vix_close < 15).astype(int)
                features['vix_high_regime'] = (vix_close > 20).astype(int)
                features['vix_extreme_regime'] = (vix_close > 30).astype(int)
                features['vix_ma_ratio'] = vix_close / vix_close.rolling(20).mean()
        
        return features
    
    def create_interaction_features(self, base_features: pd.DataFrame, vix_features: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between market factors."""
        features = pd.DataFrame(index=base_features.index)
        
        # VIX-return interactions
        if 'close' in base_features.columns and 'vix_level' in vix_features.columns:
            returns = base_features['close'].pct_change()
            vix_level = vix_features['vix_level']
            
            features['vix_return_interaction'] = returns * vix_level
            features['vix_return_regime'] = (
                (vix_level > vix_level.quantile(0.7)) & 
                (returns.abs() > returns.abs().quantile(0.7))
            ).astype(int)
        
        # Volume-volatility interactions
        if 'volume' in base_features.columns and 'cc_vol_20d' in base_features.columns:
            volume_ma = base_features['volume'].rolling(20).mean()
            volume_ratio = base_features['volume'] / volume_ma
            volatility = base_features['cc_vol_20d']
            
            features['vol_volume_interaction'] = volatility * volume_ratio
            features['high_vol_high_volume'] = (
                (volatility > volatility.quantile(0.7)) & 
                (volume_ratio > 1.5)
            ).astype(int)
        
        return features
    
    def create_calendar_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create Indian market specific calendar features."""
        features = pd.DataFrame(index=data.index)
        
        # Basic calendar features
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        features['is_month_end'] = (data.index == data.index.to_period('M').end_time.date).astype(int)
        features['is_quarter_end'] = (data.index == data.index.to_period('Q').end_time.date).astype(int)
        
        # Indian market specific
        # Monsoon season (June-September)
        features['monsoon_season'] = ((data.index.month >= 6) & (data.index.month <= 9)).astype(int)
        
        # Festival season (October-November, typically high volatility)
        features['festival_season'] = ((data.index.month >= 10) & (data.index.month <= 11)).astype(int)
        
        # Budget month (February/March, policy uncertainty)
        features['budget_season'] = ((data.index.month >= 2) & (data.index.month <= 3)).astype(int)
        
        # Year-end effects (March - Indian fiscal year end)
        features['fiscal_year_end'] = (data.index.month == 3).astype(int)
        
        return features
    
    def create_cross_asset_features(self, equity_data: pd.DataFrame, 
                                  currency_data: Optional[pd.DataFrame] = None,
                                  commodity_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create cross-asset spillover features."""
        features = pd.DataFrame(index=equity_data.index)
        
        # USD-INR volatility spillover
        if currency_data is not None and not currency_data.empty:
            currency_aligned = currency_data.reindex(equity_data.index, method='ffill')
            if 'close' in currency_aligned.columns:
                usd_inr_returns = currency_aligned['close'].pct_change()
                usd_inr_vol = usd_inr_returns.rolling(20).std() * np.sqrt(252)
                features['usd_inr_volatility'] = usd_inr_vol
                
                # Correlation with equity returns
                if 'close' in equity_data.columns:
                    equity_returns = equity_data['close'].pct_change()
                    features['equity_fx_correlation'] = equity_returns.rolling(60).corr(usd_inr_returns)
        
        # Commodity spillovers (if available)
        if commodity_data is not None and not commodity_data.empty:
            commodity_aligned = commodity_data.reindex(equity_data.index, method='ffill')
            if 'close' in commodity_aligned.columns:
                commodity_returns = commodity_aligned['close'].pct_change()
                commodity_vol = commodity_returns.rolling(20).std() * np.sqrt(252)
                features['commodity_volatility'] = commodity_vol
        
        return features
    
    def create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lagged features."""
        features = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                for lag in lags:
                    features[f'{col}_lag{lag}'] = data[col].shift(lag)
        
        return features
    
    def build_comprehensive_features(self, 
                                   ohlcv_data: pd.DataFrame,
                                   vix_data: Optional[pd.DataFrame] = None,
                                   currency_data: Optional[pd.DataFrame] = None,
                                   commodity_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Build comprehensive feature set."""
        logger.info("Building comprehensive feature set")
        
        all_features = pd.DataFrame(index=ohlcv_data.index)
        
        # Core OHLCV features
        all_features['open'] = ohlcv_data['open']
        all_features['high'] = ohlcv_data['high']
        all_features['low'] = ohlcv_data['low']
        all_features['close'] = ohlcv_data['close']
        all_features['volume'] = ohlcv_data['volume']
        all_features['returns'] = ohlcv_data['close'].pct_change()
        
        # Technical indicators
        technical_features = self.create_technical_indicators(ohlcv_data)
        all_features = all_features.join(technical_features, how='left')
        
        # Volatility estimators
        vol_features = self.create_volatility_estimators(ohlcv_data)
        all_features = all_features.join(vol_features, how='left')
        
        # Regime features
        regime_features = self.create_regime_features(ohlcv_data, vix_data)
        all_features = all_features.join(regime_features, how='left')
        
        # Calendar features
        calendar_features = self.create_calendar_features(ohlcv_data)
        all_features = all_features.join(calendar_features, how='left')
        
        # Cross-asset features
        cross_asset_features = self.create_cross_asset_features(ohlcv_data, currency_data, commodity_data)
        all_features = all_features.join(cross_asset_features, how='left')
        
        # VIX-based interaction features
        if vix_data is not None and not vix_data.empty:
            vix_features_df = pd.DataFrame(index=ohlcv_data.index)
            vix_aligned = vix_data.reindex(ohlcv_data.index, method='ffill')
            if 'close' in vix_aligned.columns:
                vix_features_df['vix_level'] = vix_aligned['close']
                interaction_features = self.create_interaction_features(all_features, vix_features_df)
                all_features = all_features.join(interaction_features, how='left')
        
        # Lag features (only for key variables to avoid curse of dimensionality)
        key_columns = ['returns', 'cc_vol_20d', 'atr_14', 'rsi_14', 'vix_level']
        lag_data = all_features[key_columns].dropna(axis=1)
        if not lag_data.empty:
            lag_features = self.create_lag_features(lag_data, lags=[1, 2, 3, 5])
            all_features = all_features.join(lag_features, how='left')
        
        # Store feature names
        self.feature_names = all_features.columns.tolist()
        
        logger.info(f"Created {len(self.feature_names)} features")
        return all_features.dropna()  # Remove rows with any NaN values
