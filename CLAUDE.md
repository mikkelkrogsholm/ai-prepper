# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-Prepper is a Docker-based offline RAG (Retrieval-Augmented Generation) system designed for complete offline operation. It combines Wikipedia knowledge with local LLMs using Ollama, ChromaDB for vector storage, and includes both a custom Streamlit UI and Flowise for visual RAG workflows.

## Architecture Overview

The system uses Docker Compose to orchestrate multiple services:

1. **Ollama** - LLM server providing model inference
2. **ChromaDB** - Vector database for semantic search
3. **Flowise** - Complete chat interface and visual workflow builder
4. **Processor** - Python service for data ingestion

### Service Communication
```
User → Flowise UI → Ollama (LLM)
                  ↘
                    ChromaDB (Vector Search)
                  ↗
Wikipedia Data → Processor → ChromaDB
```

## Common Commands

### Docker Management
```bash
# Start/stop services
make up         # Start all services
make down       # Stop all services
make status     # Check service health
make logs       # View service logs

# Data management
make pull-models    # Download Ollama models
make download-wiki  # Download Wikipedia dump
make process        # Process Wikipedia into ChromaDB

# Development
make shell      # Shell into processor container
make test       # Run tests
make clean-all  # Remove everything including data
```

## Key Files and Their Purposes

### Docker Configuration
- `docker-compose.yml` - Service definitions and networking
- `Dockerfile` - Processor service image
- `.env.example` - Environment configuration template

### Core Scripts
- `scripts/orchestrator.py` - Service initialization and health checks
- `scripts/process_wikipedia.py` - Wikipedia ingestion into ChromaDB
- `scripts/download_wikipedia.py` - Wikipedia dump downloader
- `scripts/chunk_utils.py` - Text chunking utilities

### Flowise Configuration
- Flows are stored in `data/flowise/`
- Access at http://localhost:3000
- Default credentials: admin/admin

## Configuration

All configuration is done via environment variables in `.env`:

```env
# Models
OLLAMA_MODELS=llama3.2,nomic-embed-text
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text

# Data paths (can be external drive)
OLLAMA_DATA_PATH=./data/ollama
CHROMA_DATA_PATH=./data/chroma
WIKIPEDIA_DATA_PATH=./data/wikipedia

# Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=128
MAX_ARTICLES=10000
```

## Development Patterns

### Adding New Features

1. **New Ollama Model**: Update `OLLAMA_MODELS` in `.env`
2. **New UI Features**: Modify `app/app.py`
3. **New Data Sources**: Create processor in `scripts/`
4. **New Services**: Add to `docker-compose.yml`

### Service Health Checks

All services include health checks:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:PORT/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### Data Persistence

All data is stored in Docker volumes mapped to local directories:
- Models: `./data/ollama`
- Vectors: `./data/chroma`
- Wikipedia: `./data/wikipedia`
- Workflows: `./data/flowise`

## Testing Approach

```bash
# Test service connectivity
make test

# Test chat interface
curl http://localhost:3000

# Test Ollama
curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"Hello"}'

# Test ChromaDB
curl http://localhost:8000/api/v1/heartbeat
```

## Common Issues and Solutions

### Services won't start
- Check Docker is running: `docker info`
- Check port conflicts: `lsof -i :11434`
- View logs: `make logs`

### Out of memory
- Reduce model size in `.env`
- Use `phi3` instead of `llama3.2`
- Increase Docker memory allocation

### Slow performance
- Use GPU acceleration (remove from docker-compose.yml if no GPU)
- Reduce `CHUNK_SIZE` for faster search
- Limit `MAX_ARTICLES` for smaller dataset

## External Drive Setup

For complete portability:

1. Set `EXTERNAL_DRIVE_PATH` in `.env`
2. All data paths will use this as base
3. Can move entire `data/` directory to external drive
4. Services will work from any machine with Docker

## Important Notes

- **No internet required** after initial setup
- All models cached locally in Docker volumes
- ChromaDB persists vectors between restarts
- Flowise workflows saved automatically
- Wikipedia data processed once, reused forever

## Working with the Codebase

When making changes:
1. Test locally first: `make shell` then run scripts
2. Update both Dockerfiles if adding dependencies
3. Consider memory/disk usage for offline scenarios
4. Maintain compatibility with external drive setup
5. Keep services independent and loosely coupled

The goal is a system that can be set up once and run forever offline.