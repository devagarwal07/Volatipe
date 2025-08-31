from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from ..utils.logging import get_logger

logger = get_logger(__name__)

class DataDriftDetector:
    """Detect data drift in financial time series."""
    
    def __init__(self, reference_period: int = 252, significance_level: float = 0.05):
        self.reference_period = reference_period
        self.significance_level = significance_level
        self.reference_stats = {}
    
    def fit_reference(self, data: pd.DataFrame) -> None:
        """Fit reference statistics on training data."""
        logger.info("Fitting reference statistics for drift detection")
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) > 0:
                self.reference_stats[column] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'skew': stats.skew(series),
                    'kurtosis': stats.kurtosis(series),
                    'min': series.min(),
                    'max': series.max(),
                    'percentiles': {
                        'p25': series.quantile(0.25),
                        'p50': series.quantile(0.50),
                        'p75': series.quantile(0.75)
                    }
                }
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Dict]:
        """Detect drift in new data compared to reference."""
        drift_results = {}
        
        for column in new_data.columns:
            if column not in self.reference_stats:
                continue
                
            series = new_data[column].dropna()
            if len(series) == 0:
                continue
            
            ref_stats = self.reference_stats[column]
            drift_results[column] = {}
            
            # Kolmogorov-Smirnov test for distribution drift
            try:
                # Create reference sample from stored stats (normal approximation)
                ref_sample = np.random.normal(
                    ref_stats['mean'], 
                    ref_stats['std'], 
                    size=min(len(series), 1000)
                )
                ks_stat, ks_pvalue = stats.ks_2samp(ref_sample, series.values)
                drift_results[column]['ks_test'] = {
                    'statistic': ks_stat,
                    'p_value': ks_pvalue,
                    'drift_detected': ks_pvalue < self.significance_level
                }
            except Exception as e:
                logger.warning(f"KS test failed for {column}: {e}")
            
            # Statistical tests for individual metrics
            current_stats = {
                'mean': series.mean(),
                'std': series.std(),
                'skew': stats.skew(series),
                'kurtosis': stats.kurtosis(series)
            }
            
            drift_results[column]['statistical_drift'] = {}
            for stat_name, current_value in current_stats.items():
                ref_value = ref_stats[stat_name]
                pct_change = abs((current_value - ref_value) / ref_value) if ref_value != 0 else float('inf')
                
                drift_results[column]['statistical_drift'][stat_name] = {
                    'reference': ref_value,
                    'current': current_value,
                    'pct_change': pct_change,
                    'significant_drift': pct_change > 0.2  # 20% threshold
                }
        
        return drift_results
    
    def detect_concept_drift(self, predictions: pd.Series, actuals: pd.Series, window_size: int = 30) -> Dict:
        """Detect concept drift using prediction accuracy."""
        if len(predictions) != len(actuals) or len(predictions) < window_size * 2:
            return {'error': 'Insufficient data for concept drift detection'}
        
        # Calculate rolling accuracy metrics
        errors = np.abs(predictions - actuals)
        rolling_mae = pd.Series(errors).rolling(window_size).mean()
        
        # Compare recent vs historical performance
        recent_mae = rolling_mae.iloc[-window_size:].mean()
        historical_mae = rolling_mae.iloc[:-window_size].mean()
        
        # Statistical test for significant change
        recent_errors = errors[-window_size:]
        historical_errors = errors[:-window_size]
        
        t_stat, p_value = stats.ttest_ind(recent_errors, historical_errors)
        
        return {
            'recent_mae': recent_mae,
            'historical_mae': historical_mae,
            'mae_ratio': recent_mae / historical_mae if historical_mae > 0 else float('inf'),
            't_statistic': t_stat,
            'p_value': p_value,
            'concept_drift_detected': p_value < self.significance_level and recent_mae > historical_mae
        }

class ModelPerformanceMonitor:
    """Monitor model performance over time."""
    
    def __init__(self, metrics_window: int = 30):
        self.metrics_window = metrics_window
        self.performance_history = []
    
    def update_performance(self, predictions: pd.Series, actuals: pd.Series, timestamp: datetime) -> Dict:
        """Update performance metrics with new data."""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Directional accuracy (for volatility, check if both increase/decrease)
        pred_diff = predictions.diff().dropna()
        actual_diff = actuals.diff().dropna()
        directional_accuracy = np.mean(np.sign(pred_diff) == np.sign(actual_diff))
        
        metrics = {
            'timestamp': timestamp,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'sample_size': len(predictions)
        }
        
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.metrics_window:
            self.performance_history = self.performance_history[-self.metrics_window:]
        
        return metrics
    
    def get_performance_trend(self) -> Dict:
        """Analyze performance trend over time."""
        if len(self.performance_history) < 5:
            return {'status': 'insufficient_data'}
        
        df = pd.DataFrame(self.performance_history)
        
        # Calculate trends
        mae_trend = np.polyfit(range(len(df)), df['mae'], 1)[0]
        rmse_trend = np.polyfit(range(len(df)), df['rmse'], 1)[0]
        accuracy_trend = np.polyfit(range(len(df)), df['directional_accuracy'], 1)[0]
        
        return {
            'mae_trend': mae_trend,
            'rmse_trend': rmse_trend,
            'directional_accuracy_trend': accuracy_trend,
            'current_mae': df['mae'].iloc[-1],
            'avg_mae': df['mae'].mean(),
            'current_directional_accuracy': df['directional_accuracy'].iloc[-1],
            'avg_directional_accuracy': df['directional_accuracy'].mean(),
            'performance_declining': mae_trend > 0 and rmse_trend > 0
        }
    
    def check_performance_alerts(self, thresholds: Dict[str, float]) -> List[str]:
        """Check for performance degradation alerts."""
        alerts = []
        
        if not self.performance_history:
            return alerts
        
        latest = self.performance_history[-1]
        
        # Check against thresholds
        if latest['mae'] > thresholds.get('max_mae', 0.1):
            alerts.append(f"High MAE: {latest['mae']:.4f}")
        
        if latest['directional_accuracy'] < thresholds.get('min_directional_accuracy', 0.6):
            alerts.append(f"Low directional accuracy: {latest['directional_accuracy']:.2%}")
        
        # Check trends
        if len(self.performance_history) >= 5:
            trend = self.get_performance_trend()
            if trend.get('performance_declining', False):
                alerts.append("Performance declining trend detected")
        
        return alerts
