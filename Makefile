# AI-Prepping Makefile
# Build offline RAG chatbot for Apple Silicon Macs

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Directories
MODELS_DIR := models
DATA_DIR := data
INDEX_DIR := index
CACHE_DIR := cache

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: all help venv deps models wiki index clean clean-all test run web check

# Help target
help:
	@echo "AI-Prepping Build System"
	@echo "========================"
	@echo ""
	@echo "Setup targets:"
	@echo "  make all        - Complete setup (download models, wiki, build index)"
	@echo "  make venv       - Create Python virtual environment"
	@echo "  make deps       - Install Python dependencies"
	@echo "  make models     - Download ML models"
	@echo "  make wiki       - Download and extract Wikipedia"
	@echo "  make index      - Build FAISS search index"
	@echo ""
	@echo "Usage targets:"
	@echo "  make run        - Start interactive chatbot"
	@echo "  make web        - Start web UI server"
	@echo "  make test       - Run test suite"
	@echo "  make check      - Verify all components are ready"
	@echo ""
	@echo "Maintenance targets:"
	@echo "  make clean      - Remove temporary files"
	@echo "  make clean-all  - Remove all downloaded data"
	@echo ""

# Virtual environment
$(VENV)/bin/activate:
	@echo "Creating virtual environment..."
	@python3 -m venv $(VENV)
	@echo "✓ Virtual environment created"

# Shortcut for venv
venv: $(VENV)/bin/activate

# Dependencies
deps: $(VENV)/bin/activate requirements.txt
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

# Create directories
$(MODELS_DIR) $(DATA_DIR) $(INDEX_DIR) $(CACHE_DIR):
	@mkdir -p $@

# Download models
models: deps $(MODELS_DIR)
	@echo "Downloading ML models..."
	@$(PYTHON) scripts/download_models.py --model all
	@echo "✓ Models downloaded"

# Download Wikipedia
wiki: deps $(DATA_DIR)
	@echo "Downloading Wikipedia dump..."
	@$(PYTHON) scripts/download_wikipedia.py --action all
	@echo "✓ Wikipedia downloaded and extracted"

# Build search index
index: deps $(INDEX_DIR) wiki
	@echo "Building FAISS index..."
	@$(PYTHON) scripts/build_index.py
	@echo "✓ Search index built"

# Complete setup
all: deps models wiki index
	@echo ""
	@echo "✓ AI-Prepping setup complete!"
	@echo ""
	@echo "Run 'make run' to start the chatbot"

# Run chatbot
run: deps
	@if [ ! -f "$(INDEX_DIR)/wikipedia.index" ]; then \
		echo "Error: Index not found. Run 'make all' first."; \
		exit 1; \
	fi
	@$(PYTHON) scripts/chat.py

# Run single question
question: deps
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make question Q=\"your question here\""; \
		exit 1; \
	fi
	@$(PYTHON) scripts/chat.py --question "$(Q)"

# Start web UI
web: deps
	@if [ ! -f "$(INDEX_DIR)/wikipedia.index" ]; then \
		echo "Error: Index not found. Run 'make all' first."; \
		exit 1; \
	fi
	@echo "Starting web UI at http://localhost:8000"
	@cd web && ../$(PYTHON) -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Run tests
test: deps
	@echo "Running test suite..."
	@$(PYTHON) -m pytest tests/ -v

# Quick test
test-quick: deps
	@$(PYTHON) -m pytest tests/ -v -m "not slow"

# Check system status
check: deps
	@echo "Checking AI-Prepping status..."
	@echo ""
	@echo -n "Python version: "
	@$(PYTHON) --version
	@echo ""
	@echo -n "Virtual environment: "
	@if [ -d "$(VENV)" ]; then \
		echo "✓ Active"; \
	else \
		echo "✗ Not found"; \
	fi
	@echo -n "Models: "
	@if [ -d "$(MODELS_DIR)" ] && [ "$$(ls -A $(MODELS_DIR) 2>/dev/null)" ]; then \
		echo "✓ Downloaded"; \
	else \
		echo "✗ Not found"; \
	fi
	@echo -n "Wikipedia data: "
	@if [ -f "$(DATA_DIR)/wikipedia_articles.txt" ]; then \
		echo "✓ Downloaded"; \
	else \
		echo "✗ Not found"; \
	fi
	@echo -n "Search index: "
	@if [ -f "$(INDEX_DIR)/wikipedia.index" ]; then \
		echo "✓ Built"; \
	else \
		echo "✗ Not found"; \
	fi
	@echo ""
	@$(PYTHON) scripts/download_wikipedia.py --action verify || true
	@$(PYTHON) scripts/build_index.py --verify || true

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name ".DS_Store" -delete
	@rm -rf $(CACHE_DIR)
	@echo "✓ Cleaned"

# Clean all downloaded data
clean-all: clean
	@echo "Removing all downloaded data..."
	@echo "This will delete:"
	@echo "  - $(MODELS_DIR)/"
	@echo "  - $(DATA_DIR)/"
	@echo "  - $(INDEX_DIR)/"
	@echo ""
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(MODELS_DIR) $(DATA_DIR) $(INDEX_DIR); \
		echo "✓ All data removed"; \
	else \
		echo "Cancelled"; \
	fi

# Development helpers
lint: deps
	@echo "Running linters..."
	@$(PYTHON) -m ruff check scripts/ tests/
	@$(PYTHON) -m mypy scripts/ --ignore-missing-imports

format: deps
	@echo "Formatting code..."
	@$(PYTHON) -m black scripts/ tests/
	@$(PYTHON) -m isort scripts/ tests/

# Install development dependencies
dev-deps: deps
	@$(PIP) install black isort mypy ruff

# Show disk usage
disk-usage:
	@echo "Disk usage:"
	@du -sh $(MODELS_DIR) 2>/dev/null || echo "  Models: not downloaded"
	@du -sh $(DATA_DIR) 2>/dev/null || echo "  Data: not downloaded"
	@du -sh $(INDEX_DIR) 2>/dev/null || echo "  Index: not built"
	@echo ""
	@echo -n "Total: "
	@du -sh . 2>/dev/null

# Download specific model
model-%: deps
	@$(PYTHON) scripts/download_models.py --llm-name mlx-community/$*

# Run with specific model
run-%: deps
	@$(PYTHON) scripts/chat.py --model $*

# Memory check
memory-check: deps
	@echo "Checking system memory..."
	@$(PYTHON) -c "import psutil; m=psutil.virtual_memory(); print(f'Available: {m.available/(1024**3):.1f}GB / {m.total/(1024**3):.1f}GB ({m.percent}% used)')"

# Create shell script wrapper (for non-Python users)
shell-wrapper:
	@echo "#!/bin/bash" > ai-prepper.sh
	@echo "source venv/bin/activate 2>/dev/null || true" >> ai-prepper.sh
	@echo "python scripts/chat.py \$$@" >> ai-prepper.sh
	@chmod +x ai-prepper.sh
	@echo "✓ Created ai-prepper.sh wrapper script"