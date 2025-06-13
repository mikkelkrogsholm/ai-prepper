"""Tests for config_loader module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from scripts.config_loader import Config, get_config


class TestConfig:
    """Test Config class functionality."""
    
    def test_load_config(self):
        """Test loading configuration from file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'models': {
                    'llm': {
                        'default': 'test-model',
                        'small': 'test-model-small'
                    }
                },
                'paths': {
                    'models_dir': './test_models',
                    'data_dir': './test_data'
                }
            }
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = Config(temp_path)
            assert config.get('models.llm.default') == 'test-model'
            assert config.get('models.llm.small') == 'test-model-small'
        finally:
            temp_path.unlink()
    
    def test_get_with_default(self):
        """Test getting config value with default."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test': 'value'}, f)
            temp_path = Path(f.name)
        
        try:
            config = Config(temp_path)
            assert config.get('test') == 'value'
            assert config.get('nonexistent', 'default') == 'default'
        finally:
            temp_path.unlink()
    
    def test_getitem(self):
        """Test bracket notation access."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test': {'nested': 'value'}}, f)
            temp_path = Path(f.name)
        
        try:
            config = Config(temp_path)
            assert config['test.nested'] == 'value'
            
            with pytest.raises(KeyError):
                _ = config['nonexistent']
        finally:
            temp_path.unlink()
    
    def test_path_expansion(self):
        """Test home directory expansion in paths."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'paths': {
                    'history_file': '~/.test_history',
                    'models_dir': './models'
                }
            }
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = Config(temp_path)
            history_path = config.get('paths.history_file')
            assert not history_path.startswith('~')
            assert Path(history_path).is_absolute()
            
            models_path = config.get('paths.models_dir')
            assert Path(models_path).is_absolute()
        finally:
            temp_path.unlink()
    
    def test_property_accessors(self):
        """Test property accessors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'paths': {
                    'models_dir': './test_models',
                    'data_dir': './test_data',
                    'index_dir': './test_index',
                    'cache_dir': './test_cache'
                }
            }
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = Config(temp_path)
            assert config.models_dir == Path('./test_models').absolute()
            assert config.data_dir == Path('./test_data').absolute()
            assert config.index_dir == Path('./test_index').absolute()
            assert config.cache_dir == Path('./test_cache').absolute()
        finally:
            temp_path.unlink()
    
    def test_is_low_memory(self):
        """Test low memory detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'system': {
                    'low_memory_mode': False
                },
                'models': {
                    'llm': {
                        'default': 'large-model',
                        'small': 'small-model'
                    }
                }
            }
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            config = Config(temp_path)
            
            # Test explicit setting
            assert not config.is_low_memory
            
            # Test with low_memory_mode = True
            config._config['system']['low_memory_mode'] = True
            assert config.is_low_memory
            assert config.llm_model == 'small-model'
        finally:
            temp_path.unlink()
    
    def test_ensure_directories(self, tmp_path):
        """Test directory creation."""
        config_data = {
            'paths': {
                'models_dir': str(tmp_path / 'models'),
                'data_dir': str(tmp_path / 'data'),
                'index_dir': str(tmp_path / 'index'),
                'some_file': str(tmp_path / 'file.txt')  # Should not create
            }
        }
        
        config_file = tmp_path / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = Config(config_file)
        config.ensure_directories()
        
        # Check directories were created
        assert (tmp_path / 'models').exists()
        assert (tmp_path / 'data').exists()
        assert (tmp_path / 'index').exists()
        
        # Check file was not created
        assert not (tmp_path / 'file.txt').exists()


class TestGetConfig:
    """Test get_config function."""
    
    def test_singleton(self):
        """Test that get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_with_custom_path(self):
        """Test get_config with custom path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test': 'custom'}, f)
            temp_path = Path(f.name)
        
        try:
            # Clear global config
            import scripts.config_loader
            scripts.config_loader._config = None
            
            config = get_config(temp_path)
            assert config.get('test') == 'custom'
        finally:
            temp_path.unlink()
            scripts.config_loader._config = None