# AI-Prepping Makefile
# Docker-based offline RAG system

# Docker settings
COMPOSE := docker-compose
DOCKER := docker

# Data directories
DATA_DIR := data
OLLAMA_DIR := $(DATA_DIR)/ollama
CHROMA_DIR := $(DATA_DIR)/chroma
FLOWISE_DIR := $(DATA_DIR)/flowise
WIKIPEDIA_DIR := $(DATA_DIR)/wikipedia

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: all help up down start stop restart logs build clean clean-all setup-dirs pull-models download-wiki process status shell

# Help target
help:
	@echo "AI-Prepper Docker System"
	@echo "========================"
	@echo ""
	@echo "Quick start:"
	@echo "  make setup      - Initial setup (create dirs, copy env)"
	@echo "  make up         - Start all services"
	@echo "  make status     - Check service status"
	@echo ""
	@echo "Docker commands:"
	@echo "  make build      - Build Docker images"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View service logs"
	@echo "  make status     - Check service status"
	@echo ""
	@echo "Data management:"
	@echo "  make pull-models    - Pull Ollama models"
	@echo "  make download-wiki  - Download Wikipedia data"
	@echo "  make process        - Process Wikipedia into ChromaDB"
	@echo ""
	@echo "Access points:"
	@echo "  - Ollama API:    http://localhost:11434"
	@echo "  - ChromaDB API:  http://localhost:8000"
	@echo "  - Flowise UI:    http://localhost:3000"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove containers and networks"
	@echo "  make clean-all  - Remove everything including data"
	@echo "  make shell      - Shell into processor container"

# Initial setup
setup: setup-dirs
	@echo "Setting up AI-Prepper..."
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "✓ Created .env file - please review and adjust settings"; \
	else \
		echo "✓ .env file already exists"; \
	fi
	@echo ""
	@echo "Next steps:"
	@echo "1. Review and edit .env file"
	@echo "2. Run 'make up' to start services"
	@echo "3. Run 'make pull-models' to download models"
	@echo "4. Run 'make download-wiki' to get Wikipedia data"

# Create data directories
setup-dirs:
	@echo "Creating data directories..."
	@mkdir -p $(OLLAMA_DIR) $(CHROMA_DIR) $(FLOWISE_DIR) $(WIKIPEDIA_DIR)
	@echo "✓ Data directories created"

# Build Docker images
build:
	@echo "Building Docker images..."
	$(COMPOSE) build

# Start all services
up: setup-dirs
	@echo "Starting AI-Prepper services..."
	$(COMPOSE) up -d
	@echo ""
	@echo "✓ Services starting..."
	@echo "Run 'make logs' to view logs"
	@echo "Run 'make status' to check status"

# Stop all services
down:
	@echo "Stopping AI-Prepper services..."
	$(COMPOSE) down

# Start services (if stopped)
start:
	$(COMPOSE) start

# Stop services (without removing)
stop:
	$(COMPOSE) stop

# Restart all services
restart:
	@echo "Restarting AI-Prepper services..."
	$(COMPOSE) restart

# View logs
logs:
	$(COMPOSE) logs -f

# Check service status
status:
	@echo "AI-Prepper Service Status"
	@echo "========================"
	@$(COMPOSE) ps
	@echo ""
	@echo "Checking service health..."
	@curl -s http://localhost:11434/api/version > /dev/null 2>&1 && echo "✓ Ollama is running" || echo "✗ Ollama is not accessible"
	@curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1 && echo "✓ ChromaDB is running" || echo "✗ ChromaDB is not accessible"
	@curl -s http://localhost:3000 > /dev/null 2>&1 && echo "✓ Flowise is running" || echo "✗ Flowise is not accessible"

# Pull Ollama models
pull-models:
	@echo "Pulling Ollama models..."
	@echo "This will pull models defined in .env OLLAMA_MODELS"
	$(COMPOSE) run --rm processor python -m scripts.orchestrator --wait-for-services

# Download Wikipedia
download-wiki:
	@echo "Downloading Wikipedia data..."
	$(COMPOSE) run --rm processor python -m scripts.download_wikipedia

# Process Wikipedia into ChromaDB
process:
	@echo "Processing Wikipedia data into ChromaDB..."
	$(COMPOSE) run --rm processor python -m scripts.process_wikipedia

# Clean up containers and networks
clean:
	@echo "Cleaning up containers and networks..."
	$(COMPOSE) down -v
	@echo "✓ Containers and networks removed"

# Clean everything including data
clean-all: clean
	@echo "Removing all data..."
	@read -p "This will delete all downloaded models and data. Continue? [y/N] " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf $(DATA_DIR); \
		echo "✓ All data removed"; \
	else \
		echo "Cancelled"; \
	fi

# Shell into processor container
shell:
	$(COMPOSE) run --rm processor /bin/bash

# Quick test
test:
	@echo "Testing AI-Prepper setup..."
	$(COMPOSE) run --rm processor python -c "import ollama; import chromadb; print('✓ Dependencies OK')"

# Development mode - start with logs
dev: up
	$(COMPOSE) logs -f

# External drive setup
external-setup:
	@echo "Setting up for external drive..."
	@echo "Edit .env and uncomment EXTERNAL_DRIVE_PATH section"
	@echo "Then run 'make setup' again"