"""Text chunking with language awareness."""

from typing import List
import re


class TextChunker:
    """Chunks text into overlapping segments for embedding."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = "\n\n"
    ):
        """Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
            separator: Primary separator for splitting (paragraphs by default)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
    
    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) == 0:
            return []
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by separator first (paragraphs)
        paragraphs = text.split(self.separator)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            para_length = len(paragraph)
            
            # If single paragraph exceeds chunk size, split it
            if para_length > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = self._split_into_sentences(paragraph)
                temp_chunk = []
                temp_length = 0
                
                for sentence in sentences:
                    if temp_length + len(sentence) > self.chunk_size:
                        if temp_chunk:
                            chunks.append(" ".join(temp_chunk))
                        temp_chunk = [sentence]
                        temp_length = len(sentence)
                    else:
                        temp_chunk.append(sentence)
                        temp_length += len(sentence)
                
                if temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                
                continue
            
            # Add paragraph to current chunk
            if current_length + para_length + len(self.separator) <= self.chunk_size:
                current_chunk.append(paragraph)
                current_length += para_length + len(self.separator)
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(self.separator.join(current_chunk))
                
                # Add overlap from previous chunk
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = chunks[-1][-self.chunk_overlap:]
                    current_chunk = [overlap_text, paragraph]
                    current_length = len(overlap_text) + para_length + len(self.separator)
                else:
                    current_chunk = [paragraph]
                    current_length = para_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter (can be improved with language-specific rules)
        sentence_endings = r'[.!?]+[\s]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
