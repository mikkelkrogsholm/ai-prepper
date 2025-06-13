# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Prepper is an offline RAG (Retrieval-Augmented Generation) chatbot designed for Apple Silicon Macs. It combines Wikipedia knowledge with a local LLM to answer questions without internet connectivity.

## Common Commands

### Setup and Development
```bash
# Complete setup (creates venv, downloads models/data, builds index)
make all

# Individual setup steps
make deps       # Install dependencies in virtual environment
make models     # Download LLM and embedding models (~15GB)
make wiki       # Download and extract Wikipedia (~20GB compressed)
make index      # Build FAISS search index

# Running the application
make run        # Interactive CLI chatbot
make web        # Web UI at http://localhost:8000
make question Q="What is photosynthesis?"  # Single question

# Development
make test       # Run full test suite
make test-quick # Run unit tests only (skip slow integration tests)
make check      # Verify system status
make lint       # Run code linters
make format     # Auto-format code
```

### Testing Specific Components
```bash
# Test individual modules
source venv/bin/activate
python -m pytest tests/test_chunk_utils.py -v
python -m pytest tests/test_config_loader.py::TestConfig::test_load_config -v

# Test scripts directly
python scripts/download_models.py --help
python scripts/build_index.py --verify
python scripts/chat.py --question "test query" --show-context
```

## Architecture Overview

The system follows a RAG pipeline:

1. **Data Pipeline** (`scripts/download_wikipedia.py` → `scripts/chunk_utils.py` → `scripts/build_index.py`):
   - Downloads Wikipedia XML dump, extracts articles to plaintext
   - Chunks text using token-aware sliding windows (512 tokens, 128 overlap)
   - Creates embeddings and builds FAISS index for similarity search

2. **Retrieval System** (`scripts/chat.py:retrieve_context()`):
   - Encodes queries using same embedding model
   - Searches FAISS index for top-k similar chunks
   - Filters by relevance threshold (0.7 default)

3. **Generation System** (`scripts/chat.py:generate_response()`):
   - Formats retrieved chunks as context
   - Uses MLX-optimized Mistral-7B for response generation
   - Supports graceful degradation to smaller models

## Key Design Patterns

### Configuration-Driven Architecture
All major settings are in `configs/config.yaml`. The `Config` class provides:
- Dot notation access: `config.get('models.llm.default')`
- Automatic path expansion and memory detection
- Model selection based on available RAM

### Memory-Aware Design
The system automatically detects available memory:
- <20GB RAM: Uses 4-bit quantized model
- ≥20GB RAM: Uses 8-bit quantized model
- Configurable via `system.low_memory_mode`

### MLX Optimization
Uses Apple's MLX framework for Metal GPU acceleration:
- Primary path for both LLM and embeddings
- Falls back to sentence-transformers for embeddings if MLX fails
- ~3-5x faster than CPU inference

### Error Handling Strategy
- Resume support for interrupted downloads
- Graceful fallbacks for model loading
- Clear error messages pointing to solutions (e.g., "Run 'make all' first")

## Data Flow

```
User Query → Query Embedding → FAISS Search → Retrieve Chunks
                                                      ↓
Response ← LLM Generation ← Prompt with Context ← Format Context
```

## Important Implementation Details

### Text Chunking (`chunk_utils.py`)
- Uses tiktoken for accurate token counting
- Implements overlapping windows to preserve context
- Handles large sentences that exceed chunk size

### Index Building (`build_index.py`)
- Normalizes embeddings for cosine similarity
- Stores chunks separately from index for flexibility
- Saves metadata for filtering and display

### Chat Interface (`chat.py`)
- Maintains conversation history in `~/.ai-prepping_history`
- Supports both interactive and single-question modes
- Shows source attribution with relevance scores

## Testing Approach

- Unit tests for utilities (chunking, config)
- Integration tests with small Wikipedia stubs
- Mocked external dependencies (downloads, models)
- GitHub Actions on macOS ARM64 runners

## Working with Models

The project uses MLX-community models from Hugging Face:
- LLM: `mlx-community/Mistral-7B-v0.1-8bit` (default)
- Embeddings: `mlx-community/bge-large-en-v1.5`

To add new models:
1. Add to `configs/config.yaml` under `models.llm.alternatives`
2. Test with `make model-<model-name>`
3. Update checksums if implementing verification