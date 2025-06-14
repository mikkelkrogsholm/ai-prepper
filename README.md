# 🤖 AI-Prepper

A fully offline RAG (Retrieval-Augmented Generation) system that combines Wikipedia knowledge with local LLMs. Perfect for emergency preparedness, external drive deployment, or completely air-gapped environments.

## 🌟 Features

- **100% Offline**: Works without any internet connection after initial setup
- **Docker-based**: All services run in containers - no local dependencies
- **Portable**: Can run entirely from an external drive
- **No-Code Interface**: Flowise provides drag-and-drop workflow creation
- **Modern Stack**: Ollama (LLMs), ChromaDB (vectors), Flowise (UI & Workflows)
- **Wikipedia Knowledge**: Process and query the entire Wikipedia offline

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- 50GB+ free disk space
- 8GB+ RAM (16GB recommended)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ai-prepper.git
cd ai-prepper

# 2. Configure environment
cp .env.example .env
# Edit .env if needed (especially for external drive setup)

# 3. Cache Docker images for offline use
make cache-images

# 4. Initial setup
make setup

# 5. Start all services
make up

# 6. Pull AI models (requires internet, ~5GB)
make pull-models

# 7. Download Wikipedia (requires internet, ~20GB)
make download-wiki

# 8. Process Wikipedia into vector database
make process
```

## 🖥️ Access Points

Once running, access these services:

- **Flowise UI**: http://localhost:3000 - Complete chat interface and workflow builder
- **Ollama API**: http://localhost:11434 - LLM server (used by Flowise)
- **ChromaDB API**: http://localhost:8000 - Vector database (used by Flowise)

## 💽 Offline Docker Images

Run `make cache-images` while online to pull all Docker containers and store them
as tar files under `offline_images/`. On a new machine without internet access,
restore them with `make load-images` before running `make up`.

## 📁 Project Structure

```
ai-prepper/
├── docker-compose.yml    # Service definitions
├── .env.example         # Configuration template
├── Makefile            # Common commands
├── scripts/            # Processing scripts
└── data/               # Persistent storage
    ├── ollama/         # AI models
    ├── chroma/         # Vector database
    ├── flowise/        # Workflows and chat data
    └── wikipedia/      # Source data
```

## 💾 External Drive Setup

To run from an external drive:

1. Edit `.env`:
```env
EXTERNAL_DRIVE_PATH=/Volumes/YourDrive/ai-prepper
OLLAMA_DATA_PATH=${EXTERNAL_DRIVE_PATH}/ollama
CHROMA_DATA_PATH=${EXTERNAL_DRIVE_PATH}/chroma
FLOWISE_DATA_PATH=${EXTERNAL_DRIVE_PATH}/flowise
WIKIPEDIA_DATA_PATH=${EXTERNAL_DRIVE_PATH}/wikipedia
```

2. Run setup:
```bash
make setup
make up
```

## 🔧 Common Commands

```bash
make help          # Show all commands
make status        # Check service health
make logs          # View service logs
make shell         # Access processor container
make down          # Stop all services
make clean         # Remove containers
make clean-all     # Remove everything (including data!)
```

## 🎯 Using Flowise

### First Time Setup
1. Open http://localhost:3000
2. Create a new flow:
   - Add **Ollama** node (connect to `http://ollama:11434`)
   - Add **ChromaDB** node (connect to `http://chromadb:8000`)
   - Add **Conversational Retrieval QA Chain** node
   - Connect: ChromaDB → Chain → Ollama
3. Save and deploy the flow
4. Use the built-in chat interface

### Chat Interface
- Click the chat bubble icon in Flowise
- Ask questions about Wikipedia content
- Get AI responses with source citations
- Export chat history as needed

### API Access
```python
# Use Flowise API
import requests

# Get your flow's API endpoint from Flowise
api_url = "http://localhost:3000/api/v1/prediction/{your-flow-id}"

response = requests.post(api_url, json={
    "question": "What is artificial intelligence?"
})

print(response.json())
```

## 🧰 Troubleshooting

### Services won't start
```bash
# Check Docker is running
docker info

# Check service status
make status

# View logs
make logs
```

### Out of disk space
- Use external drive setup (see above)
- Reduce MAX_ARTICLES in .env
- Use smaller models (edit OLLAMA_MODELS in .env)

### Performance issues
- Allocate more RAM to Docker Desktop
- Use smaller models (phi3 instead of llama3.2)
- Reduce chunk size in .env

### Flowise Issues
- Default login: admin/admin (change in .env)
- If flows disappear, check `data/flowise` volume
- For API access, enable API key in Flowise settings

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Ollama for local LLM serving
- ChromaDB for vector storage
- Flowise for no-code AI workflows
- Wikipedia for knowledge base

---

Built with ❤️ for offline resilience