#!/usr/bin/env python3
"""Build FAISS index for Wikipedia articles."""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

import faiss
from tqdm import tqdm
from rich.console import Console
import mlx.core as mx
import mlx.nn as nn

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config
from scripts.chunk_utils import TextChunker, TextChunk


console = Console()


class EmbeddingModel:
    """Wrapper for embedding model using MLX."""
    
    def __init__(self, model_name: str, config: Optional[Any] = None):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the model
            config: Configuration object
        """
        self.model_name = model_name
        self.config = config or get_config()
        self.dimension = self.config.get('models.embeddings.dimension', 1024)
        self.max_length = self.config.get('models.embeddings.max_length', 512)
        
        # Try to load with MLX first
        try:
            from mlx_lm import load
            self.model, self.tokenizer = load(model_name)
            self.framework = "mlx"
            console.print(f"[green]Loaded embeddings with MLX[/green]")
        except Exception as e:
            # Fallback to sentence-transformers
            console.print(f"[yellow]MLX load failed, trying sentence-transformers[/yellow]")
            self._load_sentence_transformers()
    
    def _load_sentence_transformers(self):
        """Load model using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            model_path = self.config.models_dir / self.model_name.replace('/', '_')
            
            if model_path.exists():
                self.model = SentenceTransformer(str(model_path))
            else:
                self.model = SentenceTransformer(self.model_name)
            
            self.framework = "sentence-transformers"
            self.tokenizer = None
            console.print(f"[green]Loaded embeddings with sentence-transformers[/green]")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        if self.framework == "mlx":
            return self._encode_mlx(texts, batch_size)
        else:
            return self._encode_sentence_transformers(texts, batch_size)
    
    def _encode_mlx(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using MLX model."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            
            # Convert to MLX arrays
            input_ids = mx.array(inputs['input_ids'])
            attention_mask = mx.array(inputs['attention_mask'])
            
            # Get embeddings (using last hidden state mean pooling)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            # Mean pooling
            mask_expanded = attention_mask.expand_dims(-1)
            sum_embeddings = (hidden_states * mask_expanded).sum(axis=1)
            sum_mask = mask_expanded.sum(axis=1)
            embeddings_batch = sum_embeddings / sum_mask
            
            # Convert to numpy
            embeddings.append(np.array(embeddings_batch))
        
        return np.vstack(embeddings)
    
    def _encode_sentence_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using sentence-transformers model."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )


class IndexBuilder:
    """Builds FAISS index from Wikipedia chunks."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize index builder."""
        self.config = config or get_config()
        self.index_dir = Path(self.config.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path(self.config.data_dir)
        self.dimension = self.config.get('models.embeddings.dimension', 1024)
        
        # Paths
        self.index_path = self.index_dir / "wikipedia.index"
        self.metadata_path = self.index_dir / "wikipedia_metadata.pkl"
        self.chunks_path = self.index_dir / "wikipedia_chunks.pkl"
    
    def load_wikipedia_articles(self) -> List[Tuple[str, dict]]:
        """Load Wikipedia articles from extracted file.
        
        Returns:
            List of (text, metadata) tuples
        """
        articles_path = self.data_dir / "wikipedia_articles.txt"
        if not articles_path.exists():
            raise FileNotFoundError(
                f"Wikipedia articles not found at {articles_path}. "
                "Run download_wikipedia.py first."
            )
        
        articles = []
        current_title = None
        current_text = []
        
        console.print("[cyan]Loading Wikipedia articles...[/cyan]")
        
        with open(articles_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("=== ") and line.endswith(" ===\n"):
                    # Save previous article
                    if current_title and current_text:
                        articles.append((
                            '\n'.join(current_text),
                            {'title': current_title, 'source': 'wikipedia'}
                        ))
                    
                    # Start new article
                    current_title = line[4:-5]  # Remove === markers
                    current_text = []
                else:
                    current_text.append(line.strip())
        
        # Save last article
        if current_title and current_text:
            articles.append((
                '\n'.join(current_text),
                {'title': current_title, 'source': 'wikipedia'}
            ))
        
        console.print(f"[green]Loaded {len(articles):,} articles[/green]")
        return articles
    
    def chunk_articles(self, articles: List[Tuple[str, dict]]) -> List[TextChunk]:
        """Chunk articles into smaller pieces.
        
        Args:
            articles: List of (text, metadata) tuples
            
        Returns:
            List of text chunks
        """
        console.print("[cyan]Chunking articles...[/cyan]")
        
        chunker = TextChunker(
            chunk_size=self.config.get('wikipedia.chunk_size', 512),
            chunk_overlap=self.config.get('wikipedia.chunk_overlap', 128)
        )
        
        all_chunks = []
        
        for text, metadata in tqdm(articles, desc="Chunking"):
            chunks = chunker.chunk_text(text, metadata)
            all_chunks.extend(chunks)
        
        console.print(f"[green]Created {len(all_chunks):,} chunks[/green]")
        return all_chunks
    
    def create_embeddings(self, chunks: List[TextChunk]) -> np.ndarray:
        """Create embeddings for all chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        console.print("[cyan]Creating embeddings...[/cyan]")
        
        # Load embedding model
        model_name = self.config.get('models.embeddings.model')
        model = EmbeddingModel(model_name, self.config)
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Create embeddings in batches
        batch_size = 32
        embeddings = model.encode(texts, batch_size=batch_size)
        
        console.print(f"[green]Created embeddings with shape: {embeddings.shape}[/green]")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        console.print("[cyan]Building FAISS index...[/cyan]")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Add vectors
        index.add(embeddings)
        
        console.print(f"[green]Built index with {index.ntotal:,} vectors[/green]")
        return index
    
    def save_index(self, index: faiss.Index, chunks: List[TextChunk], metadata: dict):
        """Save index and associated data.
        
        Args:
            index: FAISS index
            chunks: List of text chunks
            metadata: Index metadata
        """
        console.print("[cyan]Saving index and metadata...[/cyan]")
        
        # Save FAISS index
        faiss.write_index(index, str(self.index_path))
        
        # Save chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save metadata
        metadata.update({
            'dimension': self.dimension,
            'total_chunks': len(chunks),
            'index_type': 'IndexFlatIP'
        })
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Calculate sizes
        index_size_mb = self.index_path.stat().st_size / (1024 * 1024)
        chunks_size_mb = self.chunks_path.stat().st_size / (1024 * 1024)
        
        console.print(f"[green]✓ Index saved:[/green]")
        console.print(f"  Index size: {index_size_mb:.1f} MB")
        console.print(f"  Chunks size: {chunks_size_mb:.1f} MB")
    
    def build(self, force: bool = False) -> bool:
        """Build the complete index.
        
        Args:
            force: Force rebuild even if index exists
            
        Returns:
            True if successful
        """
        # Check if index already exists
        if self.index_path.exists() and not force:
            console.print("[green]✓ Index already exists[/green]")
            return True
        
        try:
            # Load articles
            articles = self.load_wikipedia_articles()
            
            # Chunk articles
            chunks = self.chunk_articles(articles)
            
            # Create embeddings
            embeddings = self.create_embeddings(chunks)
            
            # Build FAISS index
            index = self.build_faiss_index(embeddings)
            
            # Save everything
            metadata = {
                'article_count': len(articles),
                'chunk_count': len(chunks),
                'embedding_model': self.config.get('models.embeddings.model')
            }
            self.save_index(index, chunks, metadata)
            
            console.print("\n[green]✓ Index built successfully![/green]")
            return True
            
        except Exception as e:
            console.print(f"\n[red]✗ Index building failed: {e}[/red]")
            return False
    
    def verify_index(self) -> bool:
        """Verify that the index is valid and ready to use.
        
        Returns:
            True if index is valid
        """
        required_files = [
            self.index_path,
            self.metadata_path,
            self.chunks_path
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                console.print(f"[red]Missing file: {file_path}[/red]")
                return False
        
        try:
            # Load and check index
            index = faiss.read_index(str(self.index_path))
            console.print(f"[green]✓ Index loaded: {index.ntotal:,} vectors[/green]")
            
            # Load and check metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            console.print(f"[green]✓ Metadata loaded[/green]")
            
            # Load and check chunks
            with open(self.chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            console.print(f"[green]✓ Chunks loaded: {len(chunks):,} chunks[/green]")
            
            # Verify consistency
            if index.ntotal != len(chunks):
                console.print(f"[red]Mismatch: {index.ntotal} vectors vs {len(chunks)} chunks[/red]")
                return False
            
            return True
            
        except Exception as e:
            console.print(f"[red]Verification failed: {e}[/red]")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index for Wikipedia",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Only verify existing index'
    )
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = IndexBuilder()
    
    if args.verify:
        # Just verify
        if builder.verify_index():
            console.print("\n[green]✓ Index verification passed![/green]")
            return 0
        else:
            console.print("\n[red]✗ Index verification failed![/red]")
            return 1
    else:
        # Build index
        if builder.build(force=args.force):
            return 0
        else:
            return 1


if __name__ == "__main__":
    sys.exit(main())