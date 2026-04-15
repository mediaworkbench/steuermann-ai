"""PDF document parser using pypdf."""

from pathlib import Path
from typing import Dict, Any

from pypdf import PdfReader

from universal_agentic_framework.ingestion.parsers.base import DocumentParser


class PDFParser(DocumentParser):
    """Parser for PDF documents."""
    
    def parse(self, file_path: Path) -> str:
        """Extract text from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        reader = PdfReader(str(file_path))
        
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with title, author, subject, etc.
        """
        reader = PdfReader(str(file_path))
        metadata = {}
        
        if reader.metadata:
            metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
                "producer": reader.metadata.get("/Producer", ""),
                "pages": len(reader.pages),
            }
        
        # Clean empty values
        metadata = {k: v for k, v in metadata.items() if v}
        metadata["file_name"] = file_path.name
        metadata["file_extension"] = file_path.suffix
        
        return metadata
    
    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if extension is .pdf."""
        return extension.lower() == ".pdf"
