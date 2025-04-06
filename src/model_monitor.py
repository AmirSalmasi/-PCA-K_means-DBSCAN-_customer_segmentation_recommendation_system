import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from .logging_config import logger
from .email_service import EmailService

class ModelMonitor:
    def __init__(self, config_path='config/monitor_config.json'):
        """Initialize model monitor with configuration."""
        self.config = self._load_config(config_path)
        self.email_service = EmailService()
        self.drift_threshold = self.config.get('drift_threshold', 0.05)
        self.alert_recipients = self.config.get('alert_recipients', [])
        
    def _load_config(self, config_path):
        """Load monitoring configuration from file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Monitor config file not found: {config_path}")
                return {}
            with open(config_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading monitor config: {str(e)}", exc_info=True)
            return {}
    
    def detect_data_drift(self, current_data, reference_data, features):
        """Detect data drift between current and reference data."""
        drift_metrics = {}
        
        for feature in features:
            # Calculate Kolmogorov-Smirnov test statistic
            ks_stat, p_value = stats.ks_2samp(
                current_data[feature],
                reference_data[feature]
            )
            
            # Calculate Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(
                current_data[feature],
                reference_data[feature]
            )
            
            drift_metrics[feature] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'wasserstein_distance': wasserstein_dist,
                'drift_detected': p_value < self.drift_threshold
            }
        
        return drift_metrics
    
    def monitor_model_performance(self, model_type, current_predictions, reference_predictions):
        """Monitor model performance degradation."""
        performance_metrics = {}
        
        # Calculate accuracy metrics
        accuracy_diff = np.mean(current_predictions == reference_predictions)
        performance_metrics['accuracy'] = accuracy_diff
        
        # Calculate distribution metrics
        current_dist = np.bincount(current_predictions)
        reference_dist = np.bincount(reference_predictions)
        
        # Normalize distributions
        current_dist = current_dist / np.sum(current_dist)
        reference_dist = reference_dist / np.sum(reference_dist)
        
        # Calculate distribution difference
        distribution_diff = np.sum(np.abs(current_dist - reference_dist))
        performance_metrics['distribution_difference'] = distribution_diff
        
        return performance_metrics
    
    def check_model_health(self, model_type, current_data, reference_data):
        """Check overall model health and send alerts if needed."""
        try:
            # Load models
            models_dir = Path("models")
            model = joblib.load(models_dir / f'{model_type.lower()}.joblib')
            scaler = joblib.load(models_dir / 'scaler.joblib')
            pca = joblib.load(models_dir / 'pca.joblib')
            
            # Preprocess data
            current_scaled = scaler.transform(current_data)
            current_pca = pca.transform(current_scaled)
            
            reference_scaled = scaler.transform(reference_data)
            reference_pca = pca.transform(reference_scaled)
            
            # Get predictions
            current_predictions = model.predict(current_pca)
            reference_predictions = model.predict(reference_pca)
            
            # Detect data drift
            drift_metrics = self.detect_data_drift(
                current_data,
                reference_data,
                current_data.columns
            )
            
            # Monitor performance
            performance_metrics = self.monitor_model_performance(
                model_type,
                current_predictions,
                reference_predictions
            )
            
            # Check for significant drift
            significant_drift = any(
                metric['drift_detected'] for metric in drift_metrics.values()
            )
            
            if significant_drift:
                logger.warning(f"Significant drift detected in {model_type} model")
                for recipient in self.alert_recipients:
                    self.email_service.send_model_drift_alert(
                        recipient,
                        model_type,
                        {
                            'drift_metrics': drift_metrics,
                            'performance_metrics': performance_metrics
                        }
                    )
            
            return {
                'drift_metrics': drift_metrics,
                'performance_metrics': performance_metrics,
                'significant_drift': significant_drift
            }
            
        except Exception as e:
            logger.error(f"Error checking model health: {str(e)}", exc_info=True)
            for recipient in self.alert_recipients:
                self.email_service.send_system_alert(
                    recipient,
                    'Model Health Check Error',
                    str(e)
                )
            raise
    
    def generate_monitoring_report(self, model_type, monitoring_results):
        """Generate a monitoring report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'drift_summary': {
                feature: metrics['drift_detected']
                for feature, metrics in monitoring_results['drift_metrics'].items()
            },
            'performance_summary': monitoring_results['performance_metrics'],
            'overall_status': 'Healthy' if not monitoring_results['significant_drift'] else 'Drift Detected'
        }
        
        return report 