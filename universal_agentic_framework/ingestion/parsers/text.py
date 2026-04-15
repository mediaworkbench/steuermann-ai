"""Plain text (.txt) document parser."""

from pathlib import Path
from typing import Dict, Any

from universal_agentic_framework.ingestion.parsers.base import DocumentParser


class TextParser(DocumentParser):
    """Parser for plain text documents (.txt)."""

    def parse(self, file_path: Path) -> str:
        """Read text content from a .txt file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            File contents as a string
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from a text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Basic file metadata
        """
        metadata: Dict[str, Any] = {
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
        }

        # Try to use the first non-empty line as a title
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    title = line.strip()
                    if title:
                        metadata["title"] = title[:200]
                        break
        except Exception:
            pass

        return metadata

    @classmethod
    def supports_extension(cls, extension: str) -> bool:
        """Check if extension is .txt."""
        return extension.lower() == ".txt"
