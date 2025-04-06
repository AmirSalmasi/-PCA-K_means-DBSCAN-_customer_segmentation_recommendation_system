import json
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv
from .logging_config import logger

# Load environment variables
load_dotenv()

class ConfigManager:
    def __init__(self, config_dir='config'):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files."""
        config_files = {
            'app': 'app_config.json',
            'email': 'email_config.json',
            'monitor': 'monitor_config.json',
            'model': 'model_config.json',
            'api': 'api_config.json'
        }
        
        for config_name, filename in config_files.items():
            self.configs[config_name] = self._load_config(filename)
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a specific configuration file."""
        try:
            config_file = self.config_dir / filename
            if not config_file.exists():
                logger.warning(f"Config file not found: {filename}")
                return self._get_default_config(filename)
            
            with open(config_file) as f:
                config = json.load(f)
                
            # Override with environment variables if they exist
            if filename == 'api_config.json':
                config['host'] = os.getenv('API_HOST', config.get('host', '0.0.0.0'))
                config['port'] = int(os.getenv('API_PORT', config.get('port', 8000)))
                config['debug'] = os.getenv('API_DEBUG', 'false').lower() == 'true'
                config['security']['api_key'] = os.getenv('API_KEY', '')
                
            elif filename == 'email_config.json':
                config['smtp_server'] = os.getenv('SMTP_SERVER', config.get('smtp_server', ''))
                config['smtp_port'] = int(os.getenv('SMTP_PORT', config.get('smtp_port', 587)))
                config['sender_email'] = os.getenv('SMTP_USERNAME', config.get('sender_email', ''))
                config['sender_password'] = os.getenv('SMTP_PASSWORD', config.get('sender_password', ''))
                config['alert_recipients'] = [os.getenv('ALERT_EMAIL', '')]
                
            elif filename == 'monitor_config.json':
                config['drift_threshold'] = float(os.getenv('DRIFT_THRESHOLD', config.get('drift_threshold', 0.05)))
                config['monitoring_interval'] = int(os.getenv('MONITORING_INTERVAL', config.get('monitoring_interval', 24)))
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading config {filename}: {str(e)}", exc_info=True)
            return self._get_default_config(filename)
    
    def _get_default_config(self, filename: str) -> Dict[str, Any]:
        """Get default configuration for a specific file."""
        default_configs = {
            'app_config.json': {
                'debug': os.getenv('API_DEBUG', 'false').lower() == 'true',
                'log_level': 'INFO',
                'data_dir': os.getenv('DATA_DIR', 'data'),
                'model_dir': os.getenv('MODEL_DIR', 'models'),
                'log_dir': os.getenv('LOG_DIR', 'logs')
            },
            'email_config.json': {
                'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                'smtp_port': int(os.getenv('SMTP_PORT', 587)),
                'sender_email': os.getenv('SMTP_USERNAME', ''),
                'sender_password': os.getenv('SMTP_PASSWORD', ''),
                'alert_recipients': [os.getenv('ALERT_EMAIL', '')]
            },
            'monitor_config.json': {
                'drift_threshold': float(os.getenv('DRIFT_THRESHOLD', 0.05)),
                'monitoring_interval': int(os.getenv('MONITORING_INTERVAL', 24)),
                'alert_recipients': [os.getenv('ALERT_EMAIL', '')],
                'performance_thresholds': {
                    'accuracy': 0.8,
                    'distribution_difference': 0.1
                }
            },
            'model_config.json': {
                'kmeans': {
                    'n_clusters': 4,
                    'random_state': 42
                },
                'dbscan': {
                    'eps': 0.5,
                    'min_samples': 5
                },
                'pca': {
                    'n_components': 2
                }
            },
            'api_config.json': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', 8000)),
                'debug': os.getenv('API_DEBUG', 'false').lower() == 'true',
                'api_key_header': 'X-API-Key',
                'security': {
                    'api_key_validation': True,
                    'api_key': os.getenv('API_KEY', ''),
                    'ssl_enabled': False
                },
                'rate_limit': {
                    'requests': 100,
                    'period': 60  # seconds
                }
            }
        }
        
        return default_configs.get(filename, {})
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a specific configuration."""
        return self.configs.get(config_name, {})
    
    def update_config(self, config_name: str, updates: Dict[str, Any]):
        """Update a specific configuration."""
        if config_name not in self.configs:
            logger.warning(f"Config {config_name} not found")
            return False
        
        try:
            self.configs[config_name].update(updates)
            self._save_config(config_name)
            logger.info(f"Config {config_name} updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating config {config_name}: {str(e)}", exc_info=True)
            return False
    
    def _save_config(self, config_name: str):
        """Save a configuration to file."""
        try:
            config_file = self.config_dir / f"{config_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(self.configs[config_name], f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config {config_name}: {str(e)}", exc_info=True)
    
    def create_default_configs(self):
        """Create default configuration files if they don't exist."""
        for config_name in self.configs:
            config_file = self.config_dir / f"{config_name}_config.json"
            if not config_file.exists():
                self._save_config(config_name)
                logger.info(f"Created default config file: {config_file}") 