"""Tests for chunk_utils module."""

import pytest
from scripts.chunk_utils import TextChunker, TextChunk


class TestTextChunker:
    """Test TextChunker functionality."""
    
    def test_init(self):
        """Test chunker initialization."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20
    
    def test_count_tokens(self):
        """Test token counting."""
        chunker = TextChunker()
        text = "This is a simple test sentence."
        token_count = chunker.count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []
        
        chunks = chunker.chunk_text("   ")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_idx == 0
        assert chunks[0].end_idx == len(text)
    
    def test_chunk_long_text(self):
        """Test chunking long text with overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        # Create a long text with multiple sentences
        sentences = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence.",
            "This is the fourth sentence.",
            "This is the fifth sentence."
        ]
        text = " ".join(sentences)
        
        chunks = chunker.chunk_text(text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check that chunks have proper indices
        for chunk in chunks:
            assert chunk.start_idx >= 0
            assert chunk.end_idx <= len(text)
            assert chunk.start_idx < chunk.end_idx
    
    def test_chunk_with_metadata(self):
        """Test chunking with metadata."""
        chunker = TextChunker()
        text = "Test text with metadata."
        metadata = {"source": "test", "id": 123}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0].metadata == metadata
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        chunker = TextChunker()
        
        text = "First sentence. Second sentence! Third sentence? Fourth."
        sentences = chunker._split_sentences(text)
        
        assert len(sentences) == 4
        assert sentences[0][0] == "First sentence."
        assert sentences[1][0] == "Second sentence!"
        assert sentences[2][0] == "Third sentence?"
        assert sentences[3][0] == "Fourth."
    
    def test_sliding_window(self):
        """Test sliding window chunking."""
        chunker = TextChunker()
        text = "This is a longer text that will be chunked using sliding window approach."
        
        chunks = list(chunker.sliding_window_chunks(text, window_size=30, step_size=15))
        
        assert len(chunks) > 1
        # Check overlap exists
        for i in range(len(chunks) - 1):
            # Some tokens should appear in consecutive chunks
            assert chunker.count_tokens(chunks[i].text) > 15  # Due to overlap


class TestTextChunk:
    """Test TextChunk dataclass."""
    
    def test_init(self):
        """Test TextChunk initialization."""
        chunk = TextChunk(
            text="Sample text",
            start_idx=0,
            end_idx=11
        )
        
        assert chunk.text == "Sample text"
        assert chunk.start_idx == 0
        assert chunk.end_idx == 11
        assert chunk.metadata == {}
    
    def test_init_with_metadata(self):
        """Test TextChunk with metadata."""
        metadata = {"source": "wikipedia", "score": 0.95}
        chunk = TextChunk(
            text="Sample text",
            start_idx=0,
            end_idx=11,
            metadata=metadata
        )
        
        assert chunk.metadata == metadata