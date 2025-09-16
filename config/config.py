"""
Configuration management for plant species identification
"""

import json
import os
from typing import Dict, Any


class Config:
    """Configuration manager for the plant identification system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config_path = config_path
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "model": {
                "input_shape": [224, 224, 3],
                "num_classes": 100,
                "architecture": "custom"
            },
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "validation_split": 0.2,
                "learning_rate": 0.001,
                "optimizer": "adam"
            },
            "data": {
                "target_size": [224, 224],
                "augmentation": {
                    "rotation_range": 20,
                    "horizontal_flip": True,
                    "zoom_range": 0.1
                }
            },
            "callbacks": {
                "early_stopping": {
                    "monitor": "val_accuracy",
                    "patience": 10
                },
                "reduce_lr": {
                    "monitor": "val_loss",
                    "factor": 0.2,
                    "patience": 5
                }
            },
            "paths": {
                "data_dir": "data/training",
                "models_dir": "models",
                "logs_dir": "logs"
            }
        }
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        
        # Update config with values from file
        self._update_nested_dict(self.config, file_config)
    
    def save_to_file(self, config_path: str = None) -> None:
        """Save configuration to JSON file"""
        path = config_path or self.config_path
        if not path:
            raise ValueError("No config path specified")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value (e.g., "model.input_shape")
            default: Default value if key not found
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.config, indent=2)


# Global configuration instance
_global_config = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_path)
    
    return _global_config


def load_config(config_path: str) -> Config:
    """Load configuration from file"""
    global _global_config
    _global_config = Config(config_path)
    return _global_config