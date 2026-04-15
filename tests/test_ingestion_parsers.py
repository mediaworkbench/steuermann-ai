"""Unit tests for document parsers."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from io import BytesIO

from universal_agentic_framework.ingestion.parsers import (
    PDFParser,
    DOCXParser,
    MarkdownParser,
)


class TestPDFParser:
    """Tests for PDFParser."""
    
    def test_supports_extension(self):
        """Test file extension support check."""
        parser = PDFParser()
        
        assert parser.supports_extension(".pdf")
        assert parser.supports_extension(".PDF")
        assert not parser.supports_extension(".docx")
        assert not parser.supports_extension(".md")
    
    @patch("universal_agentic_framework.ingestion.parsers.pdf.PdfReader")
    def test_parse_simple_pdf(self, mock_reader):
        """Test parsing a simple PDF."""
        # Mock PDF with 2 pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_pdf = Mock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_reader.return_value = mock_pdf
        
        parser = PDFParser()
        
        with patch("builtins.open", mock_open(read_data=b"fake pdf")):
            text = parser.parse(Path("test.pdf"))
        
        assert "Page 1 content" in text
        assert "Page 2 content" in text
    
    @patch("universal_agentic_framework.ingestion.parsers.pdf.PdfReader")
    def test_get_metadata(self, mock_reader):
        """Test metadata extraction."""
        mock_pdf = Mock()
        mock_pdf.metadata = {
            "/Title": "Test Document",
            "/Author": "Test Author",
            "/Subject": "Test Subject",
            "/Creator": "Test Creator",
            "/Producer": "Test Producer",
        }
        mock_pdf.pages = [Mock()]
        mock_reader.return_value = mock_pdf
        
        parser = PDFParser()
        
        with patch("builtins.open", mock_open(read_data=b"fake pdf")):
            metadata = parser.get_metadata(Path("test.pdf"))
        
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"
        assert "page_count" in metadata or "pages" in metadata


class TestDOCXParser:
    """Tests for DOCXParser."""
    
    def test_supports_extension(self):
        """Test file extension support check."""
        parser = DOCXParser()
        
        assert parser.supports_extension(".docx")
        assert parser.supports_extension(".DOCX")
        assert not parser.supports_extension(".pdf")
        assert not parser.supports_extension(".md")
    
    @patch("universal_agentic_framework.ingestion.parsers.docx.Document")
    def test_parse_simple_docx(self, mock_document):
        """Test parsing a simple DOCX."""
        # Mock paragraphs
        mock_para1 = Mock()
        mock_para1.text = "First paragraph"
        
        mock_para2 = Mock()
        mock_para2.text = "Second paragraph"
        
        mock_doc = Mock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_document.return_value = mock_doc
        
        parser = DOCXParser()
        text = parser.parse(Path("test.docx"))
        
        assert "First paragraph" in text
        assert "Second paragraph" in text
    
    @patch("universal_agentic_framework.ingestion.parsers.docx.Document")
    def test_get_metadata(self, mock_document):
        """Test metadata extraction."""
        mock_props = Mock()
        mock_props.title = "Test Document"
        mock_props.author = "Test Author"
        mock_props.subject = "Test Subject"
        mock_props.keywords = "test, keywords"
        mock_props.created = None
        mock_props.modified = None
        mock_props.last_modified_by = "Test User"
        mock_props.comments = ""
        
        mock_doc = Mock()
        mock_doc.core_properties = mock_props
        mock_doc.paragraphs = [Mock(), Mock()]  # Mock paragraphs list
        mock_document.return_value = mock_doc
        
        parser = DOCXParser()
        metadata = parser.get_metadata(Path("test.docx"))
        
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"


class TestMarkdownParser:
    """Tests for MarkdownParser."""
    
    def test_supports_extension(self):
        """Test file extension support check."""
        parser = MarkdownParser()
        
        assert parser.supports_extension(".md")
        assert parser.supports_extension(".MD")
        assert parser.supports_extension(".markdown")
        assert not parser.supports_extension(".pdf")
        assert not parser.supports_extension(".docx")
    
    def test_parse_simple_markdown(self):
        """Test parsing simple markdown."""
        markdown_content = """# Title

This is a paragraph.

## Subtitle

Another paragraph."""
        
        parser = MarkdownParser()
        
        with patch("builtins.open", mock_open(read_data=markdown_content)):
            text = parser.parse(Path("test.md"))
        
        assert "Title" in text
        assert "This is a paragraph" in text
        assert "Subtitle" in text
    
    def test_get_metadata(self):
        """Test metadata extraction."""
        markdown_content = """# Document Title

Content here."""
        
        parser = MarkdownParser()
        
        with patch("builtins.open", mock_open(read_data=markdown_content)):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = Mock(
                    st_ctime=1705320000.0,
                    st_mtime=1705320000.0
                )
                
                metadata = parser.get_metadata(Path("test.md"))
        
        assert metadata["title"] == "Document Title"
        assert metadata["file_name"] == "test.md"
        assert metadata["file_extension"] == ".md"
    
    def test_get_metadata_no_title(self):
        """Test metadata when no heading exists."""
        markdown_content = """Just content, no heading."""
        
        parser = MarkdownParser()
        
        with patch("builtins.open", mock_open(read_data=markdown_content)):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = Mock(
                    st_ctime=1705320000.0,
                    st_mtime=1705320000.0
                )
                
                metadata = parser.get_metadata(Path("test.md"))
        
        # If no title found, it may not be in metadata or be None
        assert metadata["file_name"] == "test.md"
        assert metadata["file_extension"] == ".md"
