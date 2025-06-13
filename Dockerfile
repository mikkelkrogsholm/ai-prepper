FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for Docker environment
RUN pip install --no-cache-dir \
    chromadb>=0.4.0 \
    ollama>=0.1.0 \
    python-dotenv>=1.0.0

# Copy application code
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY configs/ ./configs/

# Create data directories
RUN mkdir -p /app/data/wikipedia /app/data/chroma

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "scripts.orchestrator"]