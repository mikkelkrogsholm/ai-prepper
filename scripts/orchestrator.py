#!/usr/bin/env python3
"""Orchestrator for Docker-based AI-Prepper setup."""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import requests
import chromadb
from rich.console import Console
from dotenv import load_dotenv

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config

console = Console()

class Orchestrator:
    """Manages the AI-Prepper pipeline in Docker environment."""
    
    def __init__(self):
        """Initialize orchestrator."""
        load_dotenv()
        self.config = get_config()
        
        # Service URLs from environment
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        self.chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        self.chroma_port = os.getenv('CHROMA_PORT', '8000')
        
        self.ollama_url = f"http://{self.ollama_host}:{self.ollama_port}"
        self.chroma_url = f"http://{self.chroma_host}:{self.chroma_port}"
    
    def wait_for_service(self, name: str, url: str, endpoint: str, max_retries: int = 30):
        """Wait for a service to be ready.
        
        Args:
            name: Service name for display
            url: Base URL of the service
            endpoint: Health check endpoint
            max_retries: Maximum number of retries
        """
        console.print(f"[cyan]Waiting for {name} to be ready...[/cyan]")
        
        for i in range(max_retries):
            try:
                response = requests.get(f"{url}{endpoint}", timeout=2)
                if response.status_code == 200:
                    console.print(f"[green]✓ {name} is ready![/green]")
                    return True
            except:
                pass
            
            time.sleep(2)
            if i % 5 == 0:
                console.print(f"  Still waiting for {name}... ({i}/{max_retries})")
        
        console.print(f"[red]✗ {name} failed to start[/red]")
        return False
    
    def wait_for_all_services(self):
        """Wait for all required services to be ready."""
        services = [
            ("Ollama", self.ollama_url, "/api/version"),
            ("ChromaDB", self.chroma_url, "/api/v1/heartbeat")
        ]
        
        for name, url, endpoint in services:
            if not self.wait_for_service(name, url, endpoint):
                return False
        
        return True
    
    def ensure_ollama_models(self):
        """Ensure required Ollama models are available."""
        console.print("\n[bold]Checking Ollama models...[/bold]")
        
        # Get list of required models
        models = os.getenv('OLLAMA_MODELS', 'llama3.2,nomic-embed-text').split(',')
        
        for model in models:
            model = model.strip()
            console.print(f"[cyan]Checking model: {model}[/cyan]")
            
            # Check if model exists
            try:
                response = requests.get(f"{self.ollama_url}/api/show", 
                                      json={"name": model}, timeout=5)
                if response.status_code == 200:
                    console.print(f"[green]✓ Model {model} is available[/green]")
                    continue
            except:
                pass
            
            # Pull model if not exists
            console.print(f"[yellow]Pulling model {model}...[/yellow]")
            try:
                response = requests.post(f"{self.ollama_url}/api/pull", 
                                       json={"name": model}, stream=True)
                
                for line in response.iter_lines():
                    if line:
                        console.print(f"  {line.decode('utf-8')}")
                
                console.print(f"[green]✓ Model {model} pulled successfully[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to pull {model}: {e}[/red]")
                return False
        
        return True
    
    def setup_chromadb(self):
        """Initialize ChromaDB collections."""
        console.print("\n[bold]Setting up ChromaDB...[/bold]")
        
        try:
            # Connect to ChromaDB
            client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port
            )
            
            # Create or get collection
            collection_name = os.getenv('CHROMA_COLLECTION', 'wikipedia')
            
            # Check if collection exists
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            
            if collection_name in collection_names:
                console.print(f"[green]✓ Collection '{collection_name}' already exists[/green]")
                collection = client.get_collection(collection_name)
                console.print(f"  Contains {collection.count()} documents")
            else:
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                console.print(f"[green]✓ Created collection '{collection_name}'[/green]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]✗ ChromaDB setup failed: {e}[/red]")
            return False
    
    def check_wikipedia_data(self):
        """Check if Wikipedia data exists."""
        data_path = Path(os.getenv('WIKIPEDIA_DATA_PATH', './data/wikipedia'))
        
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Check for extracted articles
        articles_file = data_path / "wikipedia_articles.txt"
        if articles_file.exists():
            size_mb = articles_file.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓ Wikipedia data found ({size_mb:.1f} MB)[/green]")
            return True
        else:
            console.print("[yellow]! Wikipedia data not found[/yellow]")
            console.print("\nTo download Wikipedia data:")
            console.print("  docker-compose run processor python -m scripts.download_wikipedia")
            return False
    
    def run_pipeline(self):
        """Run the complete setup pipeline."""
        console.print("[bold cyan]AI-Prepper Docker Orchestrator[/bold cyan]\n")
        
        # Wait for services
        if not self.wait_for_all_services():
            return False
        
        # Ensure models
        if not self.ensure_ollama_models():
            return False
        
        # Setup ChromaDB
        if not self.setup_chromadb():
            return False
        
        # Check Wikipedia data
        self.check_wikipedia_data()
        
        console.print("\n[bold green]✓ AI-Prepper is ready![/bold green]")
        console.print("\nAccess points:")
        console.print(f"  - Ollama API: http://localhost:11434")
        console.print(f"  - ChromaDB API: http://localhost:8000")
        console.print(f"  - Flowise UI: http://localhost:3000")
        console.print(f"  - Custom Web UI: http://localhost:8080")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Prepper Docker Orchestrator"
    )
    parser.add_argument(
        '--wait-for-services',
        action='store_true',
        help='Wait for all services to be ready'
    )
    parser.add_argument(
        '--skip-models',
        action='store_true',
        help='Skip model downloads'
    )
    
    args = parser.parse_args()
    
    orchestrator = Orchestrator()
    
    if args.wait_for_services:
        success = orchestrator.run_pipeline()
        sys.exit(0 if success else 1)
    else:
        # Just check status
        orchestrator.check_wikipedia_data()
        sys.exit(0)


if __name__ == "__main__":
    main()