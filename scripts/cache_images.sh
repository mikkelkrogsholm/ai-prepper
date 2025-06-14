#!/bin/bash
# Pull and save required Docker images for offline use
set -e

DIR="${1:-offline_images}"
mkdir -p "$DIR"

IMAGES=(
  "ollama/ollama:latest"
  "chromadb/chroma:latest"
  "flowiseai/flowise:latest"
  "ai-prepper/processor:latest"
)

echo "Pulling Docker images..."
docker-compose pull ollama chromadb flowise
# build processor image
docker-compose build processor

for img in "${IMAGES[@]}"; do
  fname=$(echo "$img" | tr '/:' '_').tar
  echo "Saving $img to $DIR/$fname"
  docker save "$img" -o "$DIR/$fname"
done

echo "All images saved to $DIR"

