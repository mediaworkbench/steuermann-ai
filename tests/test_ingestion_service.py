"""Integration tests for ingestion service."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import numpy as np

from universal_agentic_framework.ingestion import IngestionService, IngestionConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client."""
    with patch("universal_agentic_framework.ingestion.service.QdrantClient") as mock:
        client = Mock()
        client.get_collection.side_effect = Exception("Collection doesn't exist")
        client.create_collection.return_value = None
        client.upsert.return_value = None
        mock.return_value = client
        yield client


@pytest.fixture
def mock_embedder():
    """Mock embedding provider built by ingestion service."""
    with patch("universal_agentic_framework.ingestion.service.build_embedding_provider") as mock_builder:
        embedder = Mock()

        # Return fake embeddings as numpy arrays to mimic provider output.
        def encode_side_effect(texts, **kwargs):
            return np.array([[0.1] * 384 for _ in range(len(texts))])

        embedder.encode.side_effect = encode_side_effect
        mock_builder.return_value = embedder
        yield embedder


@pytest.fixture
def sample_config(temp_dir):
    """Create sample ingestion configuration."""
    return IngestionConfig(
        source_path=temp_dir,
        file_patterns=["*.txt", "*.md"],
        collection_name="test-collection",
        collection_description="Test collection",
        chunk_size=100,
        chunk_overlap=20,
        target_language="en",
        language_threshold=0.7,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dimension=384,
        qdrant_host="localhost",
        qdrant_port=6333,
        metadata={"test": "metadata"}
    )


class TestIngestionService:
    """Integration tests for IngestionService."""
    
    def test_init_creates_collection(self, sample_config, mock_qdrant, mock_embedder):
        """Test that initialization creates collection if it doesn't exist."""
        service = IngestionService(sample_config)
        
        # Should attempt to get collection
        mock_qdrant.get_collection.assert_called_once_with("test-collection")
        
        # Should create collection when get fails
        mock_qdrant.create_collection.assert_called_once()
    
    def test_init_uses_existing_collection(self, sample_config, mock_qdrant, mock_embedder):
        """Test that initialization uses existing collection."""
        # Mock existing collection
        mock_qdrant.get_collection.side_effect = None
        mock_qdrant.get_collection.return_value = Mock()
        
        service = IngestionService(sample_config)
        
        # Should not create collection
        mock_qdrant.create_collection.assert_not_called()
    
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_file_success(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir
    ):
        """Test successful file ingestion."""
        # Create test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nThis is test content.")
        
        # Mock parser
        mock_parse.return_value = "Test content for chunking"
        mock_get_metadata.return_value = {"title": "Test"}
        
        # Mock language detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)
        
        assert result["status"] == "success"
        assert result["file"] == str(test_file)
        assert result["chunks"] > 0
        assert "timings_ms" in result
        assert "parse" in result["timings_ms"]
        assert "upsert_calls" in result
        
        # Should have called upsert
        mock_qdrant.upsert.assert_called_once()

    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_file_skips_unchanged_by_hash(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir,
    ):
        """Incremental ingestion skips already-ingested files with same hash."""
        test_file = temp_dir / "unchanged.md"
        test_file.write_text("# Test\n\nThis is test content.")

        mock_parse.return_value = "Test content for chunking"
        mock_get_metadata.return_value = {"title": "Test"}

        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]

        # First count call checks same file_path + file_hash.
        mock_qdrant.count.return_value = Mock(count=1)

        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)

        assert result["status"] == "skipped"
        assert "unchanged" in result["reason"].lower()
        mock_qdrant.upsert.assert_not_called()

    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_file_replaces_changed_hash(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir,
    ):
        """Incremental ingestion replaces existing chunks when file hash changed."""
        test_file = temp_dir / "changed.md"
        test_file.write_text("# Test\n\nThis is changed content.")

        mock_parse.return_value = "Test content for chunking"
        mock_get_metadata.return_value = {"title": "Test"}

        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]

        # 1) same hash count => 0 (not unchanged)
        # 2) any file_path count => 2 (changed)
        mock_qdrant.count.side_effect = [Mock(count=0), Mock(count=2)]

        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)

        assert result["status"] == "success"
        mock_qdrant.delete.assert_called_once()
        mock_qdrant.upsert.assert_called_once()

    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    @patch("universal_agentic_framework.ingestion.service.build_embedding_provider")
    def test_ingest_file_supports_list_embeddings(
        self,
        mock_provider_factory,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        temp_dir,
    ):
        """Regression: remote providers may return Python lists without .tolist()."""
        test_file = temp_dir / "list_embeddings.md"
        test_file.write_text("# Test\n\nThis is test content.")

        mock_parse.return_value = "Test content for chunking"
        mock_get_metadata.return_value = {"title": "Test"}

        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]

        mock_provider = Mock()
        mock_provider.encode.return_value = [[0.1] * sample_config.embedding_dimension]
        mock_provider_factory.return_value = mock_provider

        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)

        assert result["status"] == "success"
        mock_qdrant.upsert.assert_called_once()
    
    def test_ingest_file_unsupported_type(
        self,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir
    ):
        """Test ingestion of unsupported file type."""
        # Create unsupported file
        test_file = temp_dir / "test.xyz"
        test_file.write_text("content")
        
        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)
        
        assert result["status"] == "skipped"
        assert "Unsupported file type" in result["reason"]
    
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_file_wrong_language(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir
    ):
        """Test ingestion with wrong language."""
        # Create test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nGerman content")
        
        # Mock parser
        mock_parse.return_value = "German content"
        mock_get_metadata.return_value = {"title": "Test"}
        
        # Mock language detection (German instead of English)
        mock_lang = Mock()
        mock_lang.lang = "de"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)
        
        # As of 2026-01-22: all languages accepted and tagged with detected language
        assert result["status"] == "success"
        assert result["chunks"] > 0
    
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_file_validate_only(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir
    ):
        """Test validate-only mode."""
        # Create test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nThis is longer content for validation")
        
        # Mock parser
        mock_parse.return_value = "This is longer content for validation"
        mock_get_metadata.return_value = {"title": "Test"}
        
        # Mock language detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        service = IngestionService(sample_config)
        result = service.ingest_file(test_file, validate_only=True)
        
        assert result["status"] == "valid"
        assert "metadata" in result
        
        # Should NOT have called upsert
        mock_qdrant.upsert.assert_not_called()
    
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_directory(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir
    ):
        """Test directory ingestion."""
        # Create test files
        (temp_dir / "file1.md").write_text("# File 1\n\nContent 1")
        (temp_dir / "file2.md").write_text("# File 2\n\nContent 2")
        (temp_dir / "file3.txt").write_text("File 3 content")
        (temp_dir / "ignored.pdf").write_text("PDF content")
        
        # Mock parser to return longer text
        mock_parse.return_value = "This is longer content for validation purposes"
        mock_get_metadata.return_value = {"title": "Test"}
        
        # Mock language detection
        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]
        
        service = IngestionService(sample_config)
        stats = service.ingest_directory()
        
        # Should process 3 files (.md and .txt supported; .pdf not)
        assert stats["processed"] == 3
        assert stats["errors"] == 0
        assert len(stats["files"]) == 3
        assert "timings_ms" in stats
        assert "total" in stats["timings_ms"]

    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.get_metadata")
    @patch("universal_agentic_framework.ingestion.validator.detect_langs")
    def test_ingest_file_uses_upsert_batches(
        self,
        mock_detect_langs,
        mock_get_metadata,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir,
    ):
        """Large chunk sets should be written in configured upsert batches."""
        sample_config.upsert_batch_size = 2

        test_file = temp_dir / "batch.md"
        test_file.write_text("# Test\n\nBatch content")

        mock_parse.return_value = "Chunk source"
        mock_get_metadata.return_value = {"title": "Batch"}

        mock_lang = Mock()
        mock_lang.lang = "en"
        mock_lang.prob = 0.95
        mock_detect_langs.return_value = [mock_lang]

        service = IngestionService(sample_config)
        service.chunker.chunk = Mock(return_value=["c1", "c2", "c3", "c4", "c5"])

        result = service.ingest_file(test_file)

        assert result["status"] == "success"
        assert result["upsert_calls"] == 3
        assert mock_qdrant.upsert.call_count == 3
    
    def test_clear_collection(self, sample_config, mock_qdrant, mock_embedder):
        """Test collection clearing."""
        service = IngestionService(sample_config)
        service.clear_collection()
        
        # Should delete and recreate collection
        mock_qdrant.delete_collection.assert_called_once_with("test-collection")
        assert mock_qdrant.create_collection.call_count >= 2  # Once in init, once in clear
    
    @patch("universal_agentic_framework.ingestion.parsers.markdown.MarkdownParser.parse")
    def test_ingest_file_parsing_error(
        self,
        mock_parse,
        sample_config,
        mock_qdrant,
        mock_embedder,
        temp_dir
    ):
        """Test handling of parsing errors."""
        # Create test file
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test\n\nContent")
        
        # Mock parser to raise error
        mock_parse.side_effect = Exception("Parsing failed")
        
        service = IngestionService(sample_config)
        result = service.ingest_file(test_file)
        
        assert result["status"] == "error"
        assert "Parsing failed" in result["error"]
    
    def test_find_files(self, sample_config, mock_qdrant, mock_embedder, temp_dir):
        """Test file discovery."""
        # Create test structure
        (temp_dir / "file1.txt").write_text("content")
        (temp_dir / "file2.md").write_text("content")
        (temp_dir / "ignored.pdf").write_text("content")
        
        service = IngestionService(sample_config)
        files = service._find_files()
        
        # Should find .txt and .md files (not .pdf)
        # Pattern matching may not be recursive depending on glob behavior
        assert len(files) >= 2
        assert all(f.suffix in [".txt", ".md"] for f in files)


class TestIngestionConfig:
    """Tests for IngestionConfig."""
    
    def test_init_with_defaults(self, temp_dir):
        """Test configuration with defaults."""
        config = IngestionConfig(
            source_path=temp_dir,
            file_patterns=["*.pdf"],
            collection_name="test",
            collection_description="Test"
        )
        
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
        assert config.target_language == "en"
        assert config.language_threshold == 0.8
        assert config.embedding_model == "text-embedding-granite-embedding-278m-multilingual"
        assert config.embedding_dimension == 768
        assert config.metadata == {}
    
    def test_init_with_custom_values(self, temp_dir):
        """Test configuration with custom values."""
        config = IngestionConfig(
            source_path=temp_dir,
            file_patterns=["*.pdf"],
            collection_name="custom",
            collection_description="Custom",
            chunk_size=1024,
            chunk_overlap=100,
            target_language="de",
            language_threshold=0.9,
            embedding_model="custom-model",
            embedding_dimension=768,
            qdrant_host="custom-host",
            qdrant_port=7777,
            metadata={"custom": "metadata"}
        )
        
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 100
        assert config.target_language == "de"
        assert config.language_threshold == 0.9
        assert config.embedding_model == "custom-model"
        assert config.embedding_dimension == 768
        assert config.qdrant_host == "custom-host"
        assert config.qdrant_port == 7777
        assert config.metadata == {"custom": "metadata"}
