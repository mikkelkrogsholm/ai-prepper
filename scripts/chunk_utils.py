#!/usr/bin/env python3
"""Text chunking utilities for AI-Prepping."""

import re
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass
import tiktoken


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    text: str
    start_idx: int
    end_idx: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """Handles text chunking with token awareness."""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 tokenizer_name: str = "cl100k_base"):
        """Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            tokenizer_name: Name of tiktoken tokenizer to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
        # Sentence splitter pattern
        self.sentence_pattern = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentences
            r'(?<=\n)\s*(?=\n)|'         # Paragraph breaks
            r'(?<=:)\s+(?=[A-Z])'        # After colons
        )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        """Chunk text into overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        chunk_start_idx = 0
        
        for sent_idx, (sentence, sent_start, sent_end) in enumerate(sentences):
            sent_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sent_tokens > self.chunk_size:
                # Flush current chunk if any
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_idx=chunk_start_idx,
                        end_idx=sentences[sent_idx-1][2] if sent_idx > 0 else sent_start,
                        metadata=metadata
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large sentence
                sub_chunks = self._split_large_sentence(
                    sentence, sent_start, sent_end, metadata
                )
                chunks.extend(sub_chunks)
                chunk_start_idx = sent_end
                continue
            
            # Check if adding this sentence exceeds chunk size
            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_idx=chunk_start_idx,
                    end_idx=sentences[sent_idx-1][2],
                    metadata=metadata
                ))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                # Add sentences from the end for overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    sent = current_chunk[i]
                    sent_tok = self.count_tokens(sent)
                    if overlap_tokens + sent_tok <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tok
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
                chunk_start_idx = sent_start
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sent_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(TextChunk(
                text=chunk_text,
                start_idx=chunk_start_idx,
                end_idx=len(text),
                metadata=metadata
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with position tracking.
        
        Args:
            text: Text to split
            
        Returns:
            List of (sentence, start_idx, end_idx) tuples
        """
        sentences = []
        last_end = 0
        
        for match in self.sentence_pattern.finditer(text):
            sent_end = match.start()
            if sent_end > last_end:
                sentence = text[last_end:sent_end].strip()
                if sentence:
                    sentences.append((sentence, last_end, sent_end))
                last_end = match.end()
        
        # Add final sentence
        if last_end < len(text):
            sentence = text[last_end:].strip()
            if sentence:
                sentences.append((sentence, last_end, len(text)))
        
        # If no sentences found, treat whole text as one sentence
        if not sentences and text.strip():
            sentences.append((text.strip(), 0, len(text)))
        
        return sentences
    
    def _split_large_sentence(self, 
                            sentence: str, 
                            start_idx: int,
                            end_idx: int,
                            metadata: Optional[dict]) -> List[TextChunk]:
        """Split a large sentence that exceeds chunk size.
        
        Args:
            sentence: Sentence to split
            start_idx: Start index in original text
            end_idx: End index in original text
            metadata: Optional metadata
            
        Returns:
            List of chunks
        """
        chunks = []
        tokens = self.tokenizer.encode(sentence)
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Calculate approximate character positions
            chars_per_token = len(sentence) / len(tokens)
            chunk_start = start_idx + int(i * chars_per_token)
            chunk_end = start_idx + int((i + len(chunk_tokens)) * chars_per_token)
            
            chunks.append(TextChunk(
                text=chunk_text,
                start_idx=chunk_start,
                end_idx=min(chunk_end, end_idx),
                metadata=metadata
            ))
        
        return chunks
    
    def chunk_documents(self, 
                       documents: List[Tuple[str, dict]],
                       show_progress: bool = True) -> List[TextChunk]:
        """Chunk multiple documents.
        
        Args:
            documents: List of (text, metadata) tuples
            show_progress: Whether to show progress bar
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        
        if show_progress:
            from tqdm import tqdm
            documents = tqdm(documents, desc="Chunking documents")
        
        for doc_text, doc_metadata in documents:
            chunks = self.chunk_text(doc_text, doc_metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def sliding_window_chunks(self, 
                            text: str,
                            window_size: int = 512,
                            step_size: int = 256) -> Iterator[TextChunk]:
        """Generate chunks using sliding window (token-based).
        
        Args:
            text: Text to chunk
            window_size: Window size in tokens
            step_size: Step size in tokens
            
        Yields:
            Text chunks
        """
        tokens = self.tokenizer.encode(text)
        
        for i in range(0, len(tokens), step_size):
            window_tokens = tokens[i:i + window_size]
            if len(window_tokens) < window_size // 2:  # Skip too-small final chunks
                break
                
            chunk_text = self.tokenizer.decode(window_tokens)
            
            # Approximate character positions
            chars_per_token = len(text) / len(tokens) if tokens else 1
            start_idx = int(i * chars_per_token)
            end_idx = int((i + len(window_tokens)) * chars_per_token)
            
            yield TextChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=min(end_idx, len(text)),
                metadata={'window_index': i // step_size}
            )


def create_chunker(config: dict) -> TextChunker:
    """Create a text chunker from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TextChunker instance
    """
    return TextChunker(
        chunk_size=config.get('wikipedia.chunk_size', 512),
        chunk_overlap=config.get('wikipedia.chunk_overlap', 128)
    )


if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    test_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    
    Colloquially, the term "artificial intelligence" is often used to describe 
    machines (or computers) that mimic "cognitive" functions that humans 
    associate with the human mind, such as "learning" and "problem solving".
    """
    
    chunks = chunker.chunk_text(test_text, metadata={'source': 'test'})
    
    print(f"Original text tokens: {chunker.count_tokens(test_text)}")
    print(f"Number of chunks: {len(chunks)}")
    print("\nChunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Tokens: {chunker.count_tokens(chunk.text)}")
        print(f"  Text: {chunk.text[:100]}...")
        print(f"  Position: {chunk.start_idx}-{chunk.end_idx}")