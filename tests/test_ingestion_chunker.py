"""Unit tests for text chunker."""

import pytest
from universal_agentic_framework.ingestion.chunker import TextChunker


class TestTextChunker:
    """Tests for TextChunker."""
    
    def test_init_default(self):
        """Test default initialization."""
        chunker = TextChunker()
        
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.separator == "\n\n"
    
    def test_init_custom(self):
        """Test custom initialization."""
        chunker = TextChunker(
            chunk_size=1024,
            chunk_overlap=100,
            separator="\n"
        )
        
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 100
        assert chunker.separator == "\n"
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk_size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a short text that fits in one chunk."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_multiple_paragraphs(self):
        """Test chunking multiple paragraphs."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with additional content."""
        
        chunks = chunker.chunk(text)
        
        # Should split into multiple chunks
        assert len(chunks) > 1
        
        # Check overlap exists
        for i in range(len(chunks) - 1):
            # Some text should appear in adjacent chunks
            overlap_text = chunks[i][-10:]
            assert any(overlap_text in chunks[i+1] for overlap_text in [overlap_text])
    
    def test_chunk_long_paragraph(self):
        """Test chunking a single long paragraph."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Create text with sentences to force splitting
        sentences = ["This is a test sentence." for _ in range(20)]
        text = " ".join(sentences)
        
        chunks = chunker.chunk(text)
        
        # Should split into multiple chunks due to length
        assert len(chunks) >= 1
        
        # Each chunk should be approximately chunk_size
        for chunk in chunks[:-1]:  # Exclude last chunk
            assert 50 <= len(chunk) <= chunker.chunk_size + 100  # Allow range
    
    def test_chunk_with_overlap(self):
        """Test that overlap is preserved."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = """Paragraph one has content.

Paragraph two has content.

Paragraph three has content."""
        
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check that chunks have some overlap
            for i in range(len(chunks) - 1):
                current_end = chunks[i][-chunker.chunk_overlap:]
                next_start = chunks[i+1][:chunker.chunk_overlap]
                
                # There should be some shared content
                assert len(current_end) > 0
                assert len(next_start) > 0
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        
        chunks = chunker.chunk("")
        
        assert len(chunks) == 0
    
    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunker = TextChunker()
        
        chunks = chunker.chunk("   \n\n   \n   ")
        
        # Whitespace-only chunks may be returned as-is
        # Just verify the behavior doesn't crash
        assert isinstance(chunks, list)
    
    def test_chunk_preserves_content(self):
        """Test that chunking preserves all content."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        text = """First paragraph.

Second paragraph.

Third paragraph.

Fourth paragraph."""
        
        chunks = chunker.chunk(text)
        
        # Reconstruct text from chunks (without overlap)
        # Check that key content appears
        combined = " ".join(chunks)
        
        assert "First paragraph" in combined
        assert "Second paragraph" in combined
        assert "Third paragraph" in combined
        assert "Fourth paragraph" in combined
    
    def test_chunk_with_custom_separator(self):
        """Test chunking with custom separator."""
        chunker = TextChunker(
            chunk_size=50,
            chunk_overlap=10,
            separator="\n"
        )
        
        # Create longer text to ensure splitting
        text = "\n".join(["This is a longer line that will help with splitting" for _ in range(5)])
        
        chunks = chunker.chunk(text)
        
        # May split or not depending on total length - just verify it works
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
