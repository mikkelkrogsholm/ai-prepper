#!/usr/bin/env python3
"""Process Wikipedia data into ChromaDB."""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

import chromadb
from chromadb.utils import embedding_functions
import ollama
from tqdm import tqdm
from rich.console import Console

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.chunk_utils import chunk_text, TextChunk

console = Console()


class WikipediaProcessor:
    """Process Wikipedia articles into ChromaDB."""
    
    def __init__(self):
        """Initialize processor."""
        # Service connections
        self.chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        self.chroma_port = os.getenv('CHROMA_PORT', '8000')
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        
        # ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=self.chroma_host,
            port=self.chroma_port
        )
        
        # Ollama client
        self.ollama_client = ollama.Client(
            host=f"http://{self.ollama_host}:{self.ollama_port}"
        )
        
        # Processing settings
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '512'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '128'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '100'))
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
        
    def create_collection(self, name: str = "wikipedia"):
        """Create or get ChromaDB collection."""
        console.print(f"[cyan]Setting up collection '{name}'...[/cyan]")
        
        # Check if collection exists
        collections = self.chroma_client.list_collections()
        collection_names = [c.name for c in collections]
        
        if name in collection_names:
            # Delete existing collection
            console.print(f"[yellow]Deleting existing collection '{name}'...[/yellow]")
            self.chroma_client.delete_collection(name)
        
        # Create new collection with Ollama embeddings
        embedding_fn = embedding_functions.OllamaEmbeddingFunction(
            api_base=f"http://{self.ollama_host}:{self.ollama_port}",
            model_name=self.embedding_model
        )
        
        collection = self.chroma_client.create_collection(
            name=name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        console.print(f"[green]✓ Created collection '{name}'[/green]")
        return collection
    
    def load_wikipedia_data(self, data_path: str) -> List[Dict]:
        """Load Wikipedia articles from file."""
        articles_file = Path(data_path) / "wikipedia_articles.txt"
        
        if not articles_file.exists():
            raise FileNotFoundError(f"Wikipedia data not found at {articles_file}")
        
        console.print(f"[cyan]Loading Wikipedia data from {articles_file}...[/cyan]")
        
        articles = []
        current_article = None
        
        with open(articles_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("=== ") and line.endswith(" ==="):
                    # Save previous article
                    if current_article and current_article['content']:
                        articles.append(current_article)
                    
                    # Start new article
                    title = line[4:-4].strip()
                    current_article = {
                        'title': title,
                        'content': ''
                    }
                elif current_article:
                    current_article['content'] += line + '\n'
            
            # Save last article
            if current_article and current_article['content']:
                articles.append(current_article)
        
        console.print(f"[green]✓ Loaded {len(articles)} articles[/green]")
        return articles
    
    def process_articles(self, articles: List[Dict], collection):
        """Process articles into chunks and add to ChromaDB."""
        console.print(f"[cyan]Processing {len(articles)} articles...[/cyan]")
        
        all_chunks = []
        
        # Process each article
        for article in tqdm(articles, desc="Chunking articles"):
            # Chunk the article
            chunks = chunk_text(
                article['content'],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f"{article['title']}_{i}",
                    'text': chunk.text,
                    'metadata': {
                        'title': article['title'],
                        'chunk_index': i,
                        'start_char': chunk.start_char,
                        'end_char': chunk.end_char
                    }
                })
        
        console.print(f"[green]✓ Created {len(all_chunks)} chunks[/green]")
        
        # Add to ChromaDB in batches
        console.print(f"[cyan]Adding chunks to ChromaDB...[/cyan]")
        
        for i in tqdm(range(0, len(all_chunks), self.batch_size), desc="Indexing"):
            batch = all_chunks[i:i + self.batch_size]
            
            collection.add(
                ids=[chunk['id'] for chunk in batch],
                documents=[chunk['text'] for chunk in batch],
                metadatas=[chunk['metadata'] for chunk in batch]
            )
        
        console.print(f"[green]✓ Indexed {len(all_chunks)} chunks[/green]")
    
    def verify_collection(self, collection_name: str = "wikipedia"):
        """Verify the collection was created successfully."""
        console.print(f"\n[bold]Verifying collection '{collection_name}'...[/bold]")
        
        collection = self.chroma_client.get_collection(collection_name)
        count = collection.count()
        
        console.print(f"[green]✓ Collection contains {count:,} documents[/green]")
        
        # Test query
        test_query = "What is artificial intelligence?"
        results = collection.query(
            query_texts=[test_query],
            n_results=3
        )
        
        if results['documents'][0]:
            console.print(f"\n[green]✓ Test query successful![/green]")
            console.print(f"Query: '{test_query}'")
            console.print(f"Found {len(results['documents'][0])} results")
        else:
            console.print(f"[yellow]! Test query returned no results[/yellow]")
    
    def run(self):
        """Run the complete processing pipeline."""
        try:
            # Create collection
            collection = self.create_collection()
            
            # Load Wikipedia data
            data_path = os.getenv('WIKIPEDIA_DATA_PATH', './data/wikipedia')
            articles = self.load_wikipedia_data(data_path)
            
            # Limit articles if specified
            max_articles = int(os.getenv('MAX_ARTICLES', '0'))
            if max_articles > 0:
                articles = articles[:max_articles]
                console.print(f"[yellow]Limited to {max_articles} articles[/yellow]")
            
            # Process articles
            self.process_articles(articles, collection)
            
            # Verify
            self.verify_collection()
            
            console.print("\n[bold green]✓ Wikipedia processing complete![/bold green]")
            
        except Exception as e:
            console.print(f"[red]✗ Processing failed: {e}[/red]")
            raise


def main():
    """Main entry point."""
    processor = WikipediaProcessor()
    processor.run()


if __name__ == "__main__":
    main()