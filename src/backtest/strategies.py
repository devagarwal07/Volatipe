from __future__ import annotations
from typing import Dict, List
import pandas as pd
import numpy as np
from ..utils.logging import get_logger

logger = get_logger(__name__)

class VolatilityTargetingStrategy:
    """Volatility targeting strategy for Indian markets."""
    
    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback
        self.positions = {}
    
    def generate_signals(self, data: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
        """Generate position weights based on volatility forecasts."""
        signals = {}
        
        for symbol in data.columns:
            if symbol in predictions.columns:
                # Get predicted volatility
                pred_vol = predictions[symbol].iloc[-1]
                
                # Calculate position size based on volatility targeting
                if pred_vol > 0:
                    target_weight = self.target_vol / pred_vol
                    # Cap maximum position
                    target_weight = min(target_weight, 1.0)
                    signals[symbol] = target_weight
                else:
                    signals[symbol] = 0.0
        
        return signals

class RegimeSwitchingStrategy:
    """Regime-based allocation strategy."""
    
    def __init__(self, regime_weights: Dict[str, Dict[str, float]]):
        self.regime_weights = regime_weights
    
    def generate_signals(self, data: pd.DataFrame, predictions: pd.DataFrame, current_regime: str) -> Dict[str, float]:
        """Generate signals based on current market regime."""
        if current_regime not in self.regime_weights:
            current_regime = 'normal'  # Default fallback
        
        return self.regime_weights[current_regime]

class MeanReversionStrategy:
    """Mean reversion strategy based on volatility deviations."""
    
    def __init__(self, lookback: int = 60, threshold: float = 2.0):
        self.lookback = lookback
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
        """Generate mean reversion signals."""
        signals = {}
        
        for symbol in data.columns:
            if symbol in predictions.columns and len(data) >= self.lookback:
                # Calculate z-score of current volatility vs historical
                recent_vol = data[symbol].pct_change().rolling(20).std().iloc[-1]
                hist_vol = data[symbol].pct_change().rolling(self.lookback).std()
                z_score = (recent_vol - hist_vol.mean()) / hist_vol.std()
                
                # Mean reversion logic
                if z_score > self.threshold:
                    signals[symbol] = -0.5  # Short high volatility
                elif z_score < -self.threshold:
                    signals[symbol] = 0.5   # Long low volatility
                else:
                    signals[symbol] = 0.0   # Neutral
        
        return signals


def volatility_targeting_strategy(row: pd.Series, predictions: pd.Series, positions: Dict[str, float], target_vol: float = 0.15) -> Dict[str, float]:
    """Simple volatility targeting strategy function."""
    signals = {}
    
    for symbol in row.index:
        if symbol in predictions.index:
            pred_vol = predictions[symbol]
            if pred_vol > 0:
                weight = min(target_vol / pred_vol, 1.0)
                signals[symbol] = weight
            else:
                signals[symbol] = 0.0
    
    return signals
