#!/usr/bin/env python3
"""Interactive chat interface for AI-Prepping."""

import os
import sys
import argparse
import pickle
import readline
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import numpy as np
import faiss
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.syntax import Syntax
import mlx.core as mx
from mlx_lm import load, generate

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config
from scripts.build_index import EmbeddingModel
from scripts.chunk_utils import TextChunk


console = Console()


class RAGChatbot:
    """Retrieval-Augmented Generation chatbot."""
    
    def __init__(self, config: Optional[Any] = None, model_size: str = "default"):
        """Initialize chatbot.
        
        Args:
            config: Configuration object
            model_size: Model size to use ('default' or 'small')
        """
        self.config = config or get_config()
        self.model_size = model_size
        
        # Set up history file
        history_file = os.path.expanduser(
            self.config.get('paths.history_file', '~/.ai-prepping_history')
        )
        self.history_file = Path(history_file)
        self._setup_history()
        
        # Load components
        console.print("[cyan]Loading AI-Prepping components...[/cyan]")
        self._load_index()
        self._load_embeddings()
        self._load_llm()
        
        console.print("[green]✓ AI-Prepping ready![/green]")
    
    def _setup_history(self):
        """Set up command history."""
        # Create history file if it doesn't exist
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.touch(exist_ok=True)
        
        # Configure readline
        readline.set_history_length(
            self.config.get('ui.cli.max_history', 1000)
        )
        try:
            readline.read_history_file(self.history_file)
        except:
            pass
    
    def _save_history(self):
        """Save command history."""
        try:
            readline.write_history_file(self.history_file)
        except:
            pass
    
    def _load_index(self):
        """Load FAISS index and chunks."""
        index_dir = Path(self.config.index_dir)
        
        # Load FAISS index
        index_path = index_dir / "wikipedia.index"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. Run 'make index' first."
            )
        
        self.index = faiss.read_index(str(index_path))
        console.print(f"  Loaded index: {self.index.ntotal:,} vectors")
        
        # Load chunks
        chunks_path = index_dir / "wikipedia_chunks.pkl"
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load metadata
        metadata_path = index_dir / "wikipedia_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.index_metadata = pickle.load(f)
    
    def _load_embeddings(self):
        """Load embedding model."""
        model_name = self.config.get('models.embeddings.model')
        self.embeddings = EmbeddingModel(model_name, self.config)
        console.print(f"  Loaded embeddings: {model_name}")
    
    def _load_llm(self):
        """Load language model."""
        if self.model_size == "small" or self.config.is_low_memory:
            model_name = self.config.get('models.llm.small')
            console.print("  Using small model (low memory mode)")
        else:
            model_name = self.config.get('models.llm.default')
        
        self.model, self.tokenizer = load(model_name)
        console.print(f"  Loaded LLM: {model_name}")
        
        # Model settings
        self.max_tokens = self.config.get('models.llm.max_tokens', 2048)
        self.temperature = self.config.get('models.llm.temperature', 0.7)
        self.top_p = self.config.get('models.llm.top_p', 0.95)
    
    def retrieve_context(self, query: str) -> List[TextChunk]:
        """Retrieve relevant chunks for query.
        
        Args:
            query: User query
            
        Returns:
            List of relevant text chunks
        """
        # Encode query
        query_embedding = self.embeddings.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = self.config.get('retrieval.top_k', 5)
        distances, indices = self.index.search(query_embedding, k)
        
        # Get chunks
        retrieved_chunks = []
        score_threshold = self.config.get('retrieval.score_threshold', 0.7)
        
        for idx, score in zip(indices[0], distances[0]):
            if score >= score_threshold:
                chunk = self.chunks[idx]
                chunk.metadata['score'] = float(score)
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def format_context(self, chunks: List[TextChunk]) -> str:
        """Format retrieved chunks as context.
        
        Args:
            chunks: Retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            title = chunk.metadata.get('title', 'Unknown')
            score = chunk.metadata.get('score', 0.0)
            context_parts.append(
                f"[Source {i+1}: {title} (relevance: {score:.2f})]\n{chunk.text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM with context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        # Build prompt
        if context:
            prompt = f"""You are a helpful AI assistant with access to Wikipedia knowledge. 
Use the following context to answer the user's question. If the context doesn't contain 
relevant information, say so and provide the best answer you can based on your general knowledge.

Context:
{context}

User Question: {query}

Assistant Response:"""
        else:
            prompt = f"""You are a helpful AI assistant. Please answer the following question:

User Question: {query}

Assistant Response:"""
        
        # Generate response
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        return response
    
    def answer_question(self, query: str) -> Tuple[str, List[TextChunk]]:
        """Answer a question using RAG.
        
        Args:
            query: User question
            
        Returns:
            Tuple of (answer, retrieved_chunks)
        """
        # Retrieve relevant chunks
        chunks = self.retrieve_context(query)
        
        # Format context
        context = self.format_context(chunks)
        
        # Generate response
        response = self.generate_response(query, context)
        
        return response, chunks
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        console.print(Panel.fit(
            "[bold cyan]AI-Prepping Offline Chatbot[/bold cyan]\n"
            "Type your questions below. Use 'exit' or Ctrl+C to quit.\n"
            "Use 'help' for more commands.",
            border_style="cyan"
        ))
        
        prompt_text = self.config.get('ui.cli.prompt', '🤖 AI-Prepper> ')
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask(prompt_text)
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                # Answer question
                console.print("[dim]Searching knowledge base...[/dim]")
                response, chunks = self.answer_question(user_input)
                
                # Display response
                console.print("\n[bold green]Response:[/bold green]")
                console.print(Markdown(response))
                
                # Show sources if available
                if chunks:
                    console.print("\n[dim]Sources:[/dim]")
                    for i, chunk in enumerate(chunks):
                        title = chunk.metadata.get('title', 'Unknown')
                        score = chunk.metadata.get('score', 0.0)
                        console.print(f"  [{i+1}] {title} (relevance: {score:.2f})")
                
                console.print()  # Empty line
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
                continue
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
        
        # Save history on exit
        self._save_history()
        console.print("\n[cyan]Goodbye! Stay prepared! 👋[/cyan]")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
[bold]Available Commands:[/bold]
  exit/quit/q  - Exit the chatbot
  help         - Show this help message
  stats        - Show system statistics
  
[bold]Tips:[/bold]
  • Ask clear, specific questions for best results
  • The bot has access to Wikipedia knowledge
  • Responses are based on retrieved context
  • Works completely offline once set up
        """
        console.print(Panel(help_text, title="Help", border_style="blue"))
    
    def _show_stats(self):
        """Show system statistics."""
        stats = f"""
[bold]System Statistics:[/bold]
  Index vectors: {self.index.ntotal:,}
  Total chunks: {len(self.chunks):,}
  Model: {self.config.llm_model}
  Embeddings: {self.config.get('models.embeddings.model')}
  Memory mode: {'Low' if self.config.is_low_memory else 'Normal'}
        """
        console.print(Panel(stats, title="Statistics", border_style="green"))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Prepping offline chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--question', '-q',
        help='Single question to answer (non-interactive mode)'
    )
    parser.add_argument(
        '--model',
        choices=['default', 'small'],
        default='default',
        help='Model size to use'
    )
    parser.add_argument(
        '--show-context',
        action='store_true',
        help='Show retrieved context chunks'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    args = parser.parse_args()
    
    # Configure console
    if args.no_color:
        console._color_system = None
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbot(model_size=args.model)
        
        if args.question:
            # Single question mode
            response, chunks = chatbot.answer_question(args.question)
            
            # Print response
            console.print("\n[bold green]Response:[/bold green]")
            console.print(response)
            
            # Show context if requested
            if args.show_context and chunks:
                console.print("\n[bold]Retrieved Context:[/bold]")
                for i, chunk in enumerate(chunks):
                    console.print(f"\n[dim]--- Chunk {i+1} ---[/dim]")
                    console.print(chunk.text[:300] + "...")
        else:
            # Interactive mode
            chatbot.interactive_mode()
            
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Run 'make all' to download and prepare all data first.[/yellow]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        return 0
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())