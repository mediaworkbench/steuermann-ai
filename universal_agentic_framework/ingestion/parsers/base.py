"""Document parser interface and implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class DocumentParser(ABC):
    """Base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Parse document and return text content.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text content
        """
        pass
    
    @abstractmethod
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from document.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary of metadata (title, author, etc.)
        """
        pass
    
    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if parser supports file extension.
        
        Args:
            extension: File extension (including dot, e.g., '.pdf')
            
        Returns:
            True if supported
        """
        return False
