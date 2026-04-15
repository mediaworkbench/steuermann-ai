"""Markdown document parser."""

from pathlib import Path
from typing import Dict, Any
import markdown

from universal_agentic_framework.ingestion.parsers.base import DocumentParser


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""
    
    def parse(self, file_path: Path) -> str:
        """Extract text from Markdown.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Markdown content as plain text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Return raw markdown (could convert to HTML if needed)
        return content
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract Markdown metadata.
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Basic file metadata
        """
        metadata = {
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
        }
        
        # Try to extract title from first heading
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# '):
                        metadata["title"] = line[2:].strip()
                        break
        except Exception:
            pass
        
        return metadata
    
    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if extension is .md or .markdown."""
        return extension.lower() in [".md", ".markdown"]
