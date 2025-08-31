from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """Collect and aggregate model and system metrics."""
    
    def __init__(self):
        self.prediction_metrics = []
        self.system_metrics = []
        self.api_metrics = []
    
    def log_prediction(self, model_name: str, prediction: float, actual: Optional[float] = None, 
                      timestamp: Optional[datetime] = None) -> None:
        """Log prediction metrics."""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = {
            'timestamp': timestamp,
            'model_name': model_name,
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) if actual is not None else None
        }
        
        self.prediction_metrics.append(metric)
    
    def log_api_request(self, endpoint: str, response_time: float, status_code: int,
                       timestamp: Optional[datetime] = None) -> None:
        """Log API request metrics."""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = {
            'timestamp': timestamp,
            'endpoint': endpoint,
            'response_time': response_time,
            'status_code': status_code,
            'success': status_code < 400
        }
        
        self.api_metrics.append(metric)
    
    def log_system_metric(self, metric_name: str, value: float, unit: str = '',
                         timestamp: Optional[datetime] = None) -> None:
        """Log system performance metrics."""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = {
            'timestamp': timestamp,
            'metric_name': metric_name,
            'value': value,
            'unit': unit
        }
        
        self.system_metrics.append(metric)
    
    def get_prediction_summary(self, hours: int = 24) -> Dict:
        """Get prediction metrics summary for last N hours."""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        recent_predictions = [
            m for m in self.prediction_metrics 
            if m['timestamp'] > cutoff and m['actual'] is not None
        ]
        
        if not recent_predictions:
            return {'status': 'no_data'}
        
        errors = [m['error'] for m in recent_predictions]
        predictions = [m['prediction'] for m in recent_predictions]
        actuals = [m['actual'] for m in recent_predictions]
        
        return {
            'total_predictions': len(recent_predictions),
            'mae': np.mean(errors),
            'rmse': np.sqrt(np.mean([e**2 for e in errors])),
            'mean_prediction': np.mean(predictions),
            'mean_actual': np.mean(actuals),
            'correlation': np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        }
    
    def get_api_summary(self, hours: int = 24) -> Dict:
        """Get API metrics summary for last N hours."""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        recent_requests = [
            m for m in self.api_metrics 
            if m['timestamp'] > cutoff
        ]
        
        if not recent_requests:
            return {'status': 'no_data'}
        
        response_times = [m['response_time'] for m in recent_requests]
        success_count = sum(1 for m in recent_requests if m['success'])
        
        return {
            'total_requests': len(recent_requests),
            'success_rate': success_count / len(recent_requests),
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99)
        }
    
    def get_system_summary(self, hours: int = 24) -> Dict:
        """Get system metrics summary for last N hours."""
        cutoff = datetime.now() - pd.Timedelta(hours=hours)
        recent_metrics = [
            m for m in self.system_metrics 
            if m['timestamp'] > cutoff
        ]
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        # Group by metric name
        grouped = {}
        for metric in recent_metrics:
            name = metric['metric_name']
            if name not in grouped:
                grouped[name] = []
            grouped[name].append(metric['value'])
        
        summary = {}
        for name, values in grouped.items():
            summary[name] = {
                'current': values[-1],
                'avg': np.mean(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary

class HealthChecker:
    """System health monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_thresholds = {
            'max_response_time': 5.0,  # seconds
            'min_success_rate': 0.95,
            'max_prediction_error': 0.1,
            'min_correlation': 0.3
        }
    
    def check_health(self) -> Dict:
        """Perform comprehensive health check."""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # API health
        api_summary = self.metrics.get_api_summary(hours=1)
        if api_summary.get('status') != 'no_data':
            api_healthy = (
                api_summary['avg_response_time'] < self.health_thresholds['max_response_time'] and
                api_summary['success_rate'] > self.health_thresholds['min_success_rate']
            )
            health_status['checks']['api'] = {
                'status': 'healthy' if api_healthy else 'unhealthy',
                'avg_response_time': api_summary['avg_response_time'],
                'success_rate': api_summary['success_rate']
            }
            if not api_healthy:
                health_status['overall_status'] = 'degraded'
        
        # Model performance health
        pred_summary = self.metrics.get_prediction_summary(hours=1)
        if pred_summary.get('status') != 'no_data':
            model_healthy = (
                pred_summary['mae'] < self.health_thresholds['max_prediction_error'] and
                pred_summary['correlation'] > self.health_thresholds['min_correlation']
            )
            health_status['checks']['model_performance'] = {
                'status': 'healthy' if model_healthy else 'unhealthy',
                'mae': pred_summary['mae'],
                'correlation': pred_summary['correlation']
            }
            if not model_healthy:
                health_status['overall_status'] = 'degraded'
        
        # System metrics health
        system_summary = self.metrics.get_system_summary(hours=1)
        if system_summary.get('status') != 'no_data':
            # Check for common system metrics
            cpu_usage = system_summary.get('cpu_usage', {}).get('current', 0)
            memory_usage = system_summary.get('memory_usage', {}).get('current', 0)
            
            system_healthy = cpu_usage < 80 and memory_usage < 80  # 80% thresholds
            health_status['checks']['system'] = {
                'status': 'healthy' if system_healthy else 'unhealthy',
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage
            }
            if not system_healthy:
                health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def get_alerts(self) -> List[str]:
        """Get current system alerts."""
        alerts = []
        health = self.check_health()
        
        if health['overall_status'] != 'healthy':
            for check_name, check_result in health['checks'].items():
                if check_result['status'] != 'healthy':
                    alerts.append(f"{check_name.upper()}: {check_result}")
        
        return alerts
