#!/usr/bin/env python3
"""Configuration loader for AI-Prepping."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for AI-Prepping."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config file. Defaults to configs/config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        self.config_path = config_path
        self._config = self._load_config()
        self._expand_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _expand_paths(self):
        """Expand home directory and make paths absolute."""
        paths = self._config.get('paths', {})
        for key, path in paths.items():
            if path and isinstance(path, str):
                expanded = os.path.expanduser(path)
                if not os.path.isabs(expanded):
                    expanded = os.path.abspath(expanded)
                paths[key] = expanded
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.llm.default')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        value = self.get(key)
        if value is None:
            raise KeyError(f"Configuration key not found: {key}")
        return value
    
    def ensure_directories(self):
        """Create all configured directories if they don't exist."""
        paths = self._config.get('paths', {})
        for key, path in paths.items():
            if path and key.endswith('_dir'):
                Path(path).mkdir(parents=True, exist_ok=True)
    
    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return Path(self.get('paths.models_dir', './models'))
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.get('paths.data_dir', './data'))
    
    @property
    def index_dir(self) -> Path:
        """Get index directory path."""
        return Path(self.get('paths.index_dir', './index'))
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return Path(self.get('paths.cache_dir', './cache'))
    
    @property
    def is_low_memory(self) -> bool:
        """Check if running in low memory mode."""
        if self.get('system.low_memory_mode'):
            return True
        
        # Auto-detect based on available memory
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            return available_gb < 20
        except ImportError:
            return False
    
    @property
    def llm_model(self) -> str:
        """Get LLM model name based on memory constraints."""
        if self.is_low_memory:
            return self.get('models.llm.small')
        return self.get('models.llm.default')


# Global config instance
_config = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get global configuration instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Models directory: {config.models_dir}")
    print(f"LLM model: {config.llm_model}")
    print(f"Low memory mode: {config.is_low_memory}")