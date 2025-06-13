# AI-Prepping: Offline RAG Chatbot for Emergencies

An offline-first chatbot that combines a state-of-the-art LLM with Wikipedia-based retrieval, designed for Apple Silicon Macs. Perfect for situations with limited or no internet connectivity.

## Features

- 🔌 **100% Offline**: Works completely without internet once assets are downloaded
- 🍎 **Apple Silicon Optimized**: Uses MLX framework for efficient M-series GPU acceleration
- 📚 **Wikipedia-powered**: Full English Wikipedia as knowledge base
- 🔍 **RAG Architecture**: Retrieval-Augmented Generation for accurate, grounded responses
- 💾 **Efficient Storage**: ~50GB total disk usage with full Wikipedia
- 🎯 **Memory-aware**: Graceful degradation for 16GB RAM systems

## Requirements

- Mac with Apple Silicon (M1/M2/M3)
- macOS 14.0 or newer
- 16-32GB RAM
- ~50GB free disk space
- Python 3.10+

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mikkelkrogsholm/ai-prepper.git
cd ai-prepper

# Download all assets and build index (one-time setup)
# This automatically creates a virtual environment and installs dependencies
make all

# Start chatting!
make run
```

The Makefile automatically handles virtual environment creation and activation, so you don't need to worry about it!

## Usage

### Command Line Interface

```bash
# Interactive mode
make run

# Single question mode
make question Q="What is photosynthesis?"

# Use smaller model (for 16GB RAM)
make run-small

# Or if you prefer using Python directly (after make deps):
source venv/bin/activate
python scripts/chat.py --question "What is photosynthesis?"
```

### Web Interface (Optional)

```bash
# Start the web server
make web

# Open http://localhost:8000 in your browser
```

## Project Structure

```
ai-prepping/
├── scripts/               # Core functionality
│   ├── chat.py           # CLI chat interface
│   ├── download_models.py # Model downloader
│   ├── download_wikipedia.py # Wikipedia downloader
│   └── build_index.py    # FAISS index builder
├── configs/              # Configuration files
├── web/                  # Optional web UI
├── models/               # Downloaded model files
├── data/                 # Wikipedia corpus
└── index/                # FAISS search index
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model selection (7B vs 8B)
- Memory limits
- Retrieval parameters
- Paths and cache locations

## Makefile Targets

```bash
make all        # Complete setup (creates venv, installs deps, downloads everything)
make deps       # Install dependencies in virtual environment
make models     # Download LLM and embedding models
make wiki       # Download and extract Wikipedia
make index      # Build FAISS search index
make run        # Start interactive chatbot
make test       # Run test suite
make clean      # Remove temporary files
make clean-all  # Remove all downloaded data
make web        # Start web UI server
make check      # Check system status
```

All commands automatically use the virtual environment, so you don't need to activate it manually.

## Models and Data

This project uses:
- **LLM**: Mistral-7B-v0.1 (quantized for Apple Silicon)
- **Embeddings**: BGE-large-en-v1.5
- **Corpus**: English Wikipedia (latest dump)
- **Index**: FAISS with 1024-dimensional embeddings

## License

This project is MIT licensed. See LICENSE file for details.

### Third-party Licenses
- Models: Check individual model cards for licenses
- Wikipedia: CC BY-SA 3.0
- Dependencies: See requirements.txt for package licenses

## Troubleshooting

### Out of Memory Errors
- Use `--model small` flag to load the 7B model
- Reduce context window in config.yaml
- Close other applications

### Slow Performance
- Ensure you're on Apple Silicon (not Intel Mac)
- Check Activity Monitor for memory pressure
- Consider upgrading to a Mac with more RAM

### Download Issues
- Downloads resume automatically if interrupted
- Use `make clean` and retry if corruption detected
- Check disk space (need ~100GB during setup)

## Contributing

Pull requests welcome! Please:
- Follow existing code style
- Add tests for new features
- Update documentation as needed

## Emergency Preparedness Tips

While this tool provides offline information access, remember:
- Keep your device charged
- Download updates periodically when connected
- Test the system before you need it
- Consider solar charging options
- Have backup power banks ready