#!/usr/bin/env python3
"""Download and validate ML models for AI-Prepping."""

import os
import sys
import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import mlx_lm

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config


console = Console()


class ModelDownloader:
    """Handles downloading and validation of ML models."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize downloader with configuration."""
        self.config = config or get_config()
        self.models_dir = Path(self.config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """Download a file with progress bar and resume support.
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            desc: Description for progress bar
            
        Returns:
            True if successful, False otherwise
        """
        headers = {}
        mode = 'wb'
        resume_pos = 0
        
        # Check if partial download exists
        if dest_path.exists():
            resume_pos = dest_path.stat().st_size
            headers['Range'] = f'bytes={resume_pos}-'
            mode = 'ab'
        
        try:
            response = requests.get(url, headers=headers, stream=True, 
                                  timeout=self.config.get('downloads.timeout', 300))
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos
            
            # Download with progress bar
            with open(dest_path, mode) as f:
                with tqdm(total=total_size, initial=resume_pos, 
                         unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(
                        chunk_size=self.config.get('downloads.chunk_size', 8192)):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return True
            
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            return False
    
    def verify_checksum(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file checksum.
        
        Args:
            file_path: Path to file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            True if checksum matches
        """
        if not expected_hash or expected_hash == "sha256:expected_hash_here":
            console.print("[yellow]Warning: Checksum verification skipped (no hash provided)[/yellow]")
            return True
            
        console.print("Verifying checksum...")
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        calculated = f"sha256:{sha256_hash.hexdigest()}"
        if calculated == expected_hash:
            console.print("[green]✓ Checksum verified[/green]")
            return True
        else:
            console.print(f"[red]✗ Checksum mismatch![/red]")
            console.print(f"  Expected: {expected_hash}")
            console.print(f"  Got:      {calculated}")
            return False
    
    def download_llm(self, model_name: Optional[str] = None) -> bool:
        """Download LLM model using mlx_lm.
        
        Args:
            model_name: Model name to download, or None for default
            
        Returns:
            True if successful
        """
        if model_name is None:
            model_name = self.config.llm_model
        
        model_path = self.models_dir / model_name.replace('/', '_')
        
        # Check if already exists
        if model_path.exists() and any(model_path.iterdir()):
            console.print(f"[green]✓ LLM model already exists: {model_name}[/green]")
            return True
        
        console.print(f"[cyan]Downloading LLM: {model_name}[/cyan]")
        
        try:
            # Download using mlx_lm
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"Downloading {model_name}...", total=None)
                
                # This will download and cache the model
                from mlx_lm import load
                model, tokenizer = load(model_name)
                
                # Save model info
                info_path = model_path / "model_info.json"
                model_path.mkdir(parents=True, exist_ok=True)
                with open(info_path, 'w') as f:
                    json.dump({
                        "name": model_name,
                        "type": "llm",
                        "framework": "mlx"
                    }, f, indent=2)
            
            console.print(f"[green]✓ LLM downloaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to download LLM: {e}[/red]")
            return False
    
    def download_embeddings(self) -> bool:
        """Download embeddings model.
        
        Returns:
            True if successful
        """
        model_name = self.config.get('models.embeddings.model')
        model_path = self.models_dir / model_name.replace('/', '_')
        
        # Check if already exists
        if model_path.exists() and any(model_path.iterdir()):
            console.print(f"[green]✓ Embeddings model already exists: {model_name}[/green]")
            return True
        
        console.print(f"[cyan]Downloading embeddings: {model_name}[/cyan]")
        
        try:
            # Download using mlx_lm
            from mlx_lm import load
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"Downloading {model_name}...", total=None)
                
                # Load the model (this downloads it)
                model, tokenizer = load(model_name)
                
                # Save model info
                info_path = model_path / "model_info.json"
                model_path.mkdir(parents=True, exist_ok=True)
                with open(info_path, 'w') as f:
                    json.dump({
                        "name": model_name,
                        "type": "embeddings",
                        "framework": "mlx",
                        "dimension": self.config.get('models.embeddings.dimension')
                    }, f, indent=2)
            
            console.print(f"[green]✓ Embeddings downloaded successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to download embeddings: {e}[/red]")
            # Try alternative download method for embeddings
            console.print("[yellow]Trying alternative download method...[/yellow]")
            return self._download_embeddings_alternative(model_name, model_path)
    
    def _download_embeddings_alternative(self, model_name: str, model_path: Path) -> bool:
        """Alternative method to download embeddings using sentence-transformers format."""
        try:
            # Import at runtime to avoid dependency if not needed
            from sentence_transformers import SentenceTransformer
            
            console.print(f"[cyan]Downloading via sentence-transformers...[/cyan]")
            model = SentenceTransformer(model_name)
            
            # Save the model
            model_path.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            
            # Save model info
            info_path = model_path / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump({
                    "name": model_name,
                    "type": "embeddings",
                    "framework": "sentence-transformers",
                    "dimension": self.config.get('models.embeddings.dimension')
                }, f, indent=2)
            
            console.print(f"[green]✓ Embeddings downloaded successfully (sentence-transformers)[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Alternative download also failed: {e}[/red]")
            return False
    
    def download_all(self) -> bool:
        """Download all required models.
        
        Returns:
            True if all downloads successful
        """
        success = True
        
        # Download LLM
        if not self.download_llm():
            success = False
        
        # Download alternative LLM if configured
        if self.config.get('system.low_memory_mode'):
            small_model = self.config.get('models.llm.small')
            if small_model and not self.download_llm(small_model):
                success = False
        
        # Download embeddings
        if not self.download_embeddings():
            success = False
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download ML models for AI-Prepping",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model', 
        choices=['llm', 'embeddings', 'all'],
        default='all',
        help='Which model(s) to download'
    )
    parser.add_argument(
        '--llm-name',
        help='Specific LLM to download (overrides config)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if models exist'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader()
    
    # Clear existing models if forced
    if args.force:
        console.print("[yellow]Force mode: clearing existing models...[/yellow]")
        import shutil
        if downloader.models_dir.exists():
            shutil.rmtree(downloader.models_dir)
        downloader.models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download requested models
    success = True
    
    if args.model in ['llm', 'all']:
        if not downloader.download_llm(args.llm_name):
            success = False
    
    if args.model in ['embeddings', 'all']:
        if not downloader.download_embeddings():
            success = False
    
    if success:
        console.print("\n[green]✓ All models downloaded successfully![/green]")
        return 0
    else:
        console.print("\n[red]✗ Some downloads failed. Please check errors above.[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())