#!/usr/bin/env python3
"""Check and validate Ollama models for AI-Prepping."""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

from rich.console import Console
from rich.table import Table

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config


console = Console()


class OllamaChecker:
    """Checks for required Ollama models."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize checker with configuration."""
        self.config = config or get_config()
        
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed and running.
        
        Returns:
            True if Ollama is accessible
        """
        try:
            # Check if ollama command exists
            result = subprocess.run(['which', 'ollama'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                console.print("[red]✗ Ollama is not installed[/red]")
                console.print("\nPlease install Ollama:")
                console.print("  brew install ollama")
                console.print("  ollama serve  # Start the server")
                return False
            
            # Check if server is running
            import requests
            try:
                response = requests.get('http://localhost:11434/api/version', timeout=2)
                if response.status_code == 200:
                    version = response.json().get('version', 'unknown')
                    console.print(f"[green]✓ Ollama server is running (version: {version})[/green]")
                    return True
            except:
                pass
            
            console.print("[yellow]! Ollama is installed but server is not running[/yellow]")
            console.print("\nPlease start Ollama server:")
            console.print("  ollama serve")
            return False
            
        except Exception as e:
            console.print(f"[red]Error checking Ollama: {e}[/red]")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List installed Ollama models.
        
        Returns:
            List of model info dicts
        """
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return []
            
            models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        name = parts[0].split(':')[0]  # Remove tag
                        size = ' '.join(parts[2:4])  # Size with unit
                        models.append({
                            'name': name,
                            'full_name': parts[0],
                            'size': size
                        })
            
            return models
        except Exception as e:
            console.print(f"[red]Error listing models: {e}[/red]")
            return []
    
    def check_model(self, model_name: str) -> bool:
        """Check if a specific model is installed.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is installed
        """
        models = self.list_models()
        model_names = [m['name'] for m in models]
        return model_name in model_names
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful
        """
        console.print(f"[cyan]Pulling model: {model_name}[/cyan]")
        console.print("This may take several minutes depending on model size...")
        
        try:
            # Run ollama pull with real-time output
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    console.print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                console.print(f"[green]✓ Model {model_name} pulled successfully[/green]")
                return True
            else:
                console.print(f"[red]✗ Failed to pull model {model_name}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error pulling model: {e}[/red]")
            return False
    
    def check_required_models(self) -> Dict[str, bool]:
        """Check all required models.
        
        Returns:
            Dict mapping model names to installation status
        """
        required_models = []
        
        # LLM models
        llm_model = self.config.get('models.llm.default')
        if llm_model:
            required_models.append(llm_model)
        
        # Embedding model
        embed_model = self.config.get('models.embeddings.model')
        if embed_model:
            required_models.append(embed_model)
        
        # Check each model
        status = {}
        for model in required_models:
            status[model] = self.check_model(model)
        
        return status
    
    def ensure_models(self, auto_pull: bool = False) -> bool:
        """Ensure all required models are available.
        
        Args:
            auto_pull: Automatically pull missing models
            
        Returns:
            True if all models are available
        """
        # First check Ollama
        if not self.check_ollama_installed():
            return False
        
        # Check required models
        console.print("\n[bold]Checking required models:[/bold]")
        status = self.check_required_models()
        
        # Display status table
        table = Table(title="Model Status")
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Action", style="yellow")
        
        missing_models = []
        for model, installed in status.items():
            if installed:
                table.add_row(model, "✓ Installed", "-")
            else:
                table.add_row(model, "✗ Not found", "Need to pull")
                missing_models.append(model)
        
        console.print(table)
        
        # Handle missing models
        if missing_models:
            console.print(f"\n[yellow]Missing {len(missing_models)} model(s)[/yellow]")
            
            if auto_pull:
                console.print("\n[cyan]Auto-pulling missing models...[/cyan]")
                for model in missing_models:
                    if not self.pull_model(model):
                        return False
            else:
                console.print("\nTo download missing models, run:")
                for model in missing_models:
                    console.print(f"  ollama pull {model}")
                console.print("\nOr run: make models")
                return False
        else:
            console.print("\n[green]✓ All required models are installed![/green]")
        
        return True
    
    def list_alternatives(self):
        """List alternative models that can be used."""
        console.print("\n[bold]Alternative models you can use:[/bold]")
        
        alternatives = self.config.get('models.llm.alternatives', [])
        
        table = Table(title="Recommended LLM Alternatives")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Description", style="white")
        
        # Model descriptions
        descriptions = {
            "llama3.2": "3B, Latest Llama, very efficient",
            "mistral": "7B, Fast and capable",
            "qwen2.5": "7B, Great multilingual support",
            "phi3": "3.8B, Microsoft's efficient model",
            "gemma2": "9B, Google's powerful model"
        }
        
        sizes = {
            "llama3.2": "~2GB",
            "mistral": "~4GB",
            "qwen2.5": "~4GB",
            "phi3": "~2.3GB",
            "gemma2": "~5GB"
        }
        
        for model in alternatives:
            desc = descriptions.get(model, "Alternative LLM")
            size = sizes.get(model, "Varies")
            table.add_row(model, size, desc)
        
        console.print(table)
        
        console.print("\n[bold]Embedding models:[/bold]")
        embed_table = Table()
        embed_table.add_column("Model", style="cyan")
        embed_table.add_column("Size", style="yellow")
        embed_table.add_column("Dimensions", style="white")
        
        embed_table.add_row("nomic-embed-text", "~274MB", "768")
        embed_table.add_row("mxbai-embed-large", "~670MB", "1024")
        embed_table.add_row("all-minilm", "~22MB", "384")
        
        console.print(embed_table)
        
        console.print("\nTo use a different model:")
        console.print("1. Pull it: ollama pull <model-name>")
        console.print("2. Update configs/config.yaml")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check and manage Ollama models for AI-Prepping",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--pull', 
        action='store_true',
        help='Automatically pull missing models'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List installed models'
    )
    parser.add_argument(
        '--alternatives',
        action='store_true',
        help='Show alternative model options'
    )
    parser.add_argument(
        '--model',
        help='Check or pull a specific model'
    )
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = OllamaChecker()
    
    # Handle different modes
    if args.list:
        if not checker.check_ollama_installed():
            return 1
            
        models = checker.list_models()
        if models:
            console.print("\n[bold]Installed Ollama models:[/bold]")
            table = Table()
            table.add_column("Model", style="cyan")
            table.add_column("Full Name", style="yellow")
            table.add_column("Size", style="green")
            
            for model in models:
                table.add_row(model['name'], model['full_name'], model['size'])
            
            console.print(table)
        else:
            console.print("[yellow]No models installed yet[/yellow]")
        return 0
    
    elif args.alternatives:
        checker.list_alternatives()
        return 0
    
    elif args.model:
        if not checker.check_ollama_installed():
            return 1
            
        if checker.check_model(args.model):
            console.print(f"[green]✓ Model {args.model} is installed[/green]")
            return 0
        else:
            console.print(f"[yellow]Model {args.model} is not installed[/yellow]")
            if args.pull:
                if checker.pull_model(args.model):
                    return 0
                else:
                    return 1
            else:
                console.print(f"\nTo install: ollama pull {args.model}")
                return 1
    
    else:
        # Default: ensure all required models
        if checker.ensure_models(auto_pull=args.pull):
            return 0
        else:
            return 1


if __name__ == "__main__":
    sys.exit(main())