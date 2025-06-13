#!/usr/bin/env python3
"""Download and extract Wikipedia dump for AI-Prepping."""

import os
import sys
import argparse
import bz2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Iterator, Tuple
import re

import requests
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from scripts.config_loader import get_config


console = Console()


class WikipediaDownloader:
    """Downloads and processes Wikipedia dumps."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize downloader with configuration."""
        self.config = config or get_config()
        self.data_dir = Path(self.config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.dump_url = self.config.get('wikipedia.dump_url')
        self.language = self.config.get('wikipedia.language', 'en')
        self.max_articles = self.config.get('wikipedia.max_articles')
        
        # Paths
        self.dump_path = self.data_dir / "wikipedia_dump.xml.bz2"
        self.extracted_path = self.data_dir / "wikipedia_articles.txt"
        self.metadata_path = self.data_dir / "wikipedia_metadata.json"
    
    def download_dump(self, force: bool = False) -> bool:
        """Download Wikipedia dump file.
        
        Args:
            force: Force re-download even if file exists
            
        Returns:
            True if successful
        """
        if self.dump_path.exists() and not force:
            size_mb = self.dump_path.stat().st_size / (1024 * 1024)
            console.print(f"[green]✓ Wikipedia dump already exists ({size_mb:.1f} MB)[/green]")
            return True
        
        console.print(f"[cyan]Downloading Wikipedia dump from:[/cyan]")
        console.print(f"  {self.dump_url}")
        
        headers = {}
        mode = 'wb'
        resume_pos = 0
        
        # Check if partial download exists
        if self.dump_path.exists() and not force:
            resume_pos = self.dump_path.stat().st_size
            headers['Range'] = f'bytes={resume_pos}-'
            mode = 'ab'
            console.print(f"[yellow]Resuming from {resume_pos / (1024*1024):.1f} MB[/yellow]")
        
        try:
            response = requests.get(
                self.dump_url, 
                headers=headers, 
                stream=True,
                timeout=self.config.get('downloads.timeout', 300)
            )
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos
            
            # Download with progress
            with open(self.dump_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=resume_pos,
                    unit='B',
                    unit_scale=True,
                    desc="Downloading Wikipedia"
                ) as pbar:
                    for chunk in response.iter_content(
                        chunk_size=self.config.get('downloads.chunk_size', 8192)
                    ):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            console.print("[green]✓ Download completed successfully[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            return False
    
    def extract_articles(self, force: bool = False) -> bool:
        """Extract articles from Wikipedia dump.
        
        Args:
            force: Force re-extraction even if output exists
            
        Returns:
            True if successful
        """
        if self.extracted_path.exists() and not force:
            console.print("[green]✓ Extracted articles already exist[/green]")
            return True
        
        if not self.dump_path.exists():
            console.print("[red]Error: Wikipedia dump not found. Run download first.[/red]")
            return False
        
        console.print("[cyan]Extracting articles from Wikipedia dump...[/cyan]")
        
        try:
            article_count = 0
            total_size = 0
            
            with open(self.extracted_path, 'w', encoding='utf-8') as output:
                # Process the compressed XML
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed} articles"),
                ) as progress:
                    
                    task = progress.add_task(
                        "Extracting articles", 
                        total=self.max_articles or float('inf')
                    )
                    
                    for title, text in self._parse_wikipedia_dump():
                        if text and len(text) > 100:  # Skip very short articles
                            # Clean text
                            cleaned_text = self._clean_wiki_text(text)
                            
                            # Write to output
                            output.write(f"=== {title} ===\n")
                            output.write(cleaned_text)
                            output.write("\n\n")
                            
                            article_count += 1
                            total_size += len(cleaned_text)
                            progress.update(task, completed=article_count)
                            
                            # Check limit
                            if self.max_articles and article_count >= self.max_articles:
                                break
            
            # Save metadata
            import json
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    'article_count': article_count,
                    'total_size': total_size,
                    'language': self.language,
                    'extraction_complete': True
                }, f, indent=2)
            
            console.print(f"[green]✓ Extracted {article_count:,} articles[/green]")
            console.print(f"[green]  Total size: {total_size / (1024*1024):.1f} MB[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Extraction failed: {e}[/red]")
            return False
    
    def _parse_wikipedia_dump(self) -> Iterator[Tuple[str, str]]:
        """Parse Wikipedia XML dump and yield (title, text) pairs."""
        with bz2.BZ2File(self.dump_path, 'r') as f:
            # Use iterative parsing to handle large files
            context = ET.iterparse(f, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            title = None
            text = None
            
            for event, elem in context:
                if event == 'end':
                    if elem.tag.endswith('title'):
                        title = elem.text
                    elif elem.tag.endswith('text'):
                        text = elem.text
                        
                        # Yield when we have both title and text
                        if title and text:
                            # Skip special pages
                            if not any(title.startswith(prefix) for prefix in 
                                     ['Wikipedia:', 'Template:', 'Category:', 'File:']):
                                yield (title, text)
                        
                        title = None
                        text = None
                    
                    # Clear element to save memory
                    elem.clear()
                    root.clear()
    
    def _clean_wiki_text(self, text: str) -> str:
        """Clean Wikipedia markup from text.
        
        Args:
            text: Raw Wikipedia text
            
        Returns:
            Cleaned plain text
        """
        # Remove wiki markup
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # [[links]]
        text = re.sub(r'\{\{[^}]+\}\}', '', text)  # {{templates}}
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags
        text = re.sub(r'&[a-z]+;', ' ', text)  # HTML entities
        text = re.sub(r'https?://[^\s]+', '', text)  # URLs
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
        text = re.sub(r'={2,}([^=]+)={2,}', r'\n\1\n', text)  # Section headers
        
        # Clean up references
        text = re.sub(r'\[[\d\s,]+\]', '', text)  # [1], [2,3], etc.
        text = re.sub(r'^[*#:;]+', '', text, flags=re.MULTILINE)  # List markers
        
        # Remove remaining markup
        text = re.sub(r"'''?", '', text)  # Bold/italic
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        
        return text.strip()
    
    def verify_data(self) -> bool:
        """Verify that Wikipedia data is ready for use.
        
        Returns:
            True if data is valid and ready
        """
        if not self.extracted_path.exists():
            console.print("[red]Extracted articles not found[/red]")
            return False
        
        if not self.metadata_path.exists():
            console.print("[red]Metadata file not found[/red]")
            return False
        
        # Check file sizes
        article_size = self.extracted_path.stat().st_size / (1024 * 1024)
        if article_size < 10:  # Less than 10 MB seems wrong
            console.print(f"[red]Extracted articles too small: {article_size:.1f} MB[/red]")
            return False
        
        # Load and check metadata
        import json
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        article_count = metadata.get('article_count', 0)
        if article_count < 100:  # Less than 100 articles seems wrong
            console.print(f"[red]Too few articles: {article_count}[/red]")
            return False
        
        console.print(f"[green]✓ Wikipedia data verified:[/green]")
        console.print(f"  Articles: {article_count:,}")
        console.print(f"  Size: {article_size:.1f} MB")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and process Wikipedia for AI-Prepping",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--action',
        choices=['download', 'extract', 'verify', 'all'],
        default='all',
        help='What action to perform'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download/re-extraction'
    )
    parser.add_argument(
        '--max-articles',
        type=int,
        help='Maximum number of articles to extract'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    config = get_config()
    if args.max_articles:
        config._config['wikipedia']['max_articles'] = args.max_articles
    
    downloader = WikipediaDownloader(config)
    
    # Perform requested actions
    success = True
    
    if args.action in ['download', 'all']:
        if not downloader.download_dump(force=args.force):
            success = False
    
    if args.action in ['extract', 'all'] and success:
        if not downloader.extract_articles(force=args.force):
            success = False
    
    if args.action in ['verify', 'all'] and success:
        if not downloader.verify_data():
            success = False
    
    if success:
        console.print("\n[green]✓ Wikipedia processing completed successfully![/green]")
        return 0
    else:
        console.print("\n[red]✗ Wikipedia processing failed. Check errors above.[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())