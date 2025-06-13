"""Pytest configuration and fixtures."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def temp_config():
    """Create a temporary configuration file."""
    config_data = {
        'models': {
            'llm': {
                'default': 'test-llm-model',
                'small': 'test-llm-small',
                'max_tokens': 512,
                'temperature': 0.7
            },
            'embeddings': {
                'model': 'test-embeddings',
                'dimension': 384,
                'max_length': 256
            }
        },
        'paths': {
            'models_dir': './test_models',
            'data_dir': './test_data',
            'index_dir': './test_index',
            'cache_dir': './test_cache'
        },
        'wikipedia': {
            'chunk_size': 128,
            'chunk_overlap': 32
        },
        'retrieval': {
            'top_k': 3,
            'score_threshold': 0.5
        },
        'downloads': {
            'chunk_size': 1024,
            'timeout': 30
        },
        'system': {
            'low_memory_mode': False
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink()


@pytest.fixture
def mock_config(temp_config):
    """Create a mock configuration object."""
    from scripts.config_loader import Config
    return Config(temp_config)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    Machine learning (ML) is a subset of artificial intelligence that provides 
    systems the ability to automatically learn and improve from experience 
    without being explicitly programmed. Machine learning focuses on the 
    development of computer programs that can access data and use it to 
    learn for themselves.
    
    Deep learning is part of a broader family of machine learning methods 
    based on artificial neural networks with representation learning. 
    Learning can be supervised, semi-supervised or unsupervised.
    """


@pytest.fixture
def mock_faiss_index():
    """Create a mock FAISS index."""
    mock_index = Mock()
    mock_index.ntotal = 1000
    mock_index.search.return_value = (
        [[0.9, 0.85, 0.8]],  # distances
        [[0, 5, 10]]  # indices
    )
    return mock_index


@pytest.fixture
def sample_chunks():
    """Create sample text chunks."""
    from scripts.chunk_utils import TextChunk
    
    chunks = [
        TextChunk(
            text="Artificial intelligence is intelligence demonstrated by machines.",
            start_idx=0,
            end_idx=65,
            metadata={'title': 'Artificial Intelligence', 'source': 'wikipedia'}
        ),
        TextChunk(
            text="Machine learning is a subset of artificial intelligence.",
            start_idx=200,
            end_idx=256,
            metadata={'title': 'Machine Learning', 'source': 'wikipedia'}
        ),
        TextChunk(
            text="Deep learning uses artificial neural networks.",
            start_idx=400,
            end_idx=447,
            metadata={'title': 'Deep Learning', 'source': 'wikipedia'}
        )
    ]
    
    return chunks


@pytest.fixture(autouse=True)
def cleanup_global_config():
    """Clean up global config after each test."""
    yield
    # Reset global config
    import scripts.config_loader
    scripts.config_loader._config = None


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )