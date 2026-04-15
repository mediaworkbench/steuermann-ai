"""Ingestion service for processing documents into knowledge base."""

from pathlib import Path
from typing import List, Dict, Any, Optional, cast
from dataclasses import dataclass, field
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient, models
import time
from qdrant_client.models import Distance, VectorParams, PointStruct

from universal_agentic_framework.ingestion.parsers import PDFParser, DOCXParser, MarkdownParser, TextParser
from universal_agentic_framework.ingestion.chunker import TextChunker
from universal_agentic_framework.ingestion.validator import LanguageValidator
from universal_agentic_framework.embeddings import build_embedding_provider, EmbeddingProvider
from universal_agentic_framework.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""
    
    # Source configuration
    source_path: Path
    file_patterns: List[str]
    
    # Collection configuration
    collection_name: str
    collection_description: str
    
    # Processing configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    file_concurrency: int = 1
    upsert_batch_size: int = 128
    enable_incremental: bool = True
    enable_phase_timing: bool = True
    target_language: str = "en"
    language_threshold: float = 0.8
    
    # Embedding configuration
    embedding_model: str = "text-embedding-granite-embedding-278m-multilingual"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32
    embedding_provider_type: str = "remote"
    embedding_remote_endpoint: Optional[str] = None
    
    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.file_concurrency = max(1, int(self.file_concurrency or 1))
        self.upsert_batch_size = max(1, int(self.upsert_batch_size or 1))
        self.embedding_batch_size = max(1, int(self.embedding_batch_size or 1))


class IngestionService:
    """Orchestrates document ingestion into Qdrant."""
    
    def __init__(self, config: IngestionConfig):
        """Initialize ingestion service.
        
        Args:
            config: Ingestion configuration
        """
        self.config = config
        
        # Initialize components
        self.parsers = {
            ".pdf": PDFParser(),
            ".docx": DOCXParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".txt": TextParser(),
        }
        
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        self.validator = LanguageValidator(
            target_language=config.target_language,
            confidence_threshold=config.language_threshold
        )
        
        # Initialize embedding provider
        logger.info(
            "Loading embedding provider",
            model=config.embedding_model,
            provider_type=config.embedding_provider_type
        )
        self.embedder: EmbeddingProvider = build_embedding_provider(
            model_name=config.embedding_model,
            dimension=config.embedding_dimension,
            provider_type=config.embedding_provider_type,
            remote_endpoint=config.embedding_remote_endpoint,
        )
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            check_compatibility=False,
        )
        
        # Ensure collection exists
        self._ensure_collection()

    @staticmethod
    def _normalize_vector(embedding: Any) -> List[float]:
        """Normalize embedding vector to plain Python list[float]."""
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        if isinstance(embedding, list):
            return embedding
        return list(embedding)

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    @staticmethod
    def _readable_bool(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _batched(items: List[Any], batch_size: int):
        for idx in range(0, len(items), batch_size):
            yield items[idx:idx + batch_size]

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with file_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _safe_count(self, ingestion_filter: models.Filter) -> int:
        """Count matching points while tolerating client/API variations."""
        try:
            count_result = self.qdrant.count(
                collection_name=self.config.collection_name,
                count_filter=ingestion_filter,
                # Incremental ingestion decisions require precise counts.
                exact=True,
            )
        except Exception:
            return 0

        count_value = getattr(count_result, "count", 0)
        if isinstance(count_value, bool):
            return int(count_value)
        if not isinstance(count_value, (int, float)):
            return 0
        return int(count_value)

    def _get_file_ingestion_state(self, file_path: Path, file_hash: str) -> str:
        """Return one of: new, changed, unchanged."""
        file_path_str = str(file_path)

        same_hash_count = self._safe_count(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_path_str),
                    ),
                    models.FieldCondition(
                        key="file_hash",
                        match=models.MatchValue(value=file_hash),
                    ),
                ]
            )
        )
        if same_hash_count > 0:
            return "unchanged"

        any_file_count = self._safe_count(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_path_str),
                    )
                ]
            )
        )
        if any_file_count > 0:
            return "changed"

        return "new"

    def _encode_chunks(self, chunks: List[str]) -> List[Any]:
        """Encode chunks in batches to reduce peak memory and smooth remote calls."""
        all_embeddings: List[Any] = []
        encode = cast(Any, self.embedder.encode)
        for chunk_batch in self._batched(chunks, self.config.embedding_batch_size):
            try:
                batch_embeddings = encode(
                    chunk_batch,
                    show_progress_bar=False,
                    batch_size=self.config.embedding_batch_size,
                )
            except TypeError:
                batch_embeddings = encode(
                    chunk_batch,
                    show_progress_bar=False,
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _upsert_points_batched(self, points: List[PointStruct]) -> int:
        upsert_calls = 0
        for point_batch in self._batched(points, self.config.upsert_batch_size):
            self.qdrant.upsert(
                collection_name=self.config.collection_name,
                points=point_batch,
            )
            upsert_calls += 1
        return upsert_calls
    

    def _connect_with_retry(self, max_retries: int = 10, initial_delay: float = 1.0):
        """Connect to Qdrant with retry/backoff to handle startup races."""
        delay = initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                self.qdrant.get_collections()
                logger.info("Connected to Qdrant", attempt=attempt + 1)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max_retries - 1:
                    logger.warning(
                        "Qdrant connection failed, retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=delay,
                        error=str(exc),
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
        raise RuntimeError(f"Failed to connect to Qdrant after {max_retries} attempts: {last_error}")
    def _ensure_collection(self, max_retries: int = 5, initial_delay: float = 1.0):
        """Create collection if it doesn't exist, with retry/backoff."""
        delay = initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                self.qdrant.get_collection(self.config.collection_name)
                logger.info("Collection exists", collection=self.config.collection_name)
                return
            except Exception:
                try:
                    logger.info("Creating collection", collection=self.config.collection_name)
                    self.qdrant.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config=VectorParams(
                            size=self.config.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Collection creation failed, retrying",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(exc),
                        )
                        time.sleep(delay)
                        delay = min(delay * 2, 30)
        raise RuntimeError(f"Failed to ensure collection after {max_retries} attempts: {last_error}")
    
    def ingest_directory(
        self,
        validate_only: bool = False
    ) -> Dict[str, Any]:
        """Ingest all matching documents from source directory.
        
        Args:
            validate_only: If True, only validate without ingesting
            
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "timings_ms": {
                "total": 0,
                "parse": 0,
                "chunk": 0,
                "embed": 0,
                "upsert": 0,
                "hash": 0,
                "lookup": 0,
            },
            "files": []
        }
        ingest_start = self._now()
        
        # Find matching files
        files = self._find_files()
        logger.info("Found files to process", count=len(files))
        
        def process_one(current_file: Path) -> Dict[str, Any]:
            return self.ingest_file(current_file, validate_only=validate_only)

        if self.config.file_concurrency == 1 or len(files) <= 1:
            file_results: List[Dict[str, Any]] = []
            for file_path in files:
                try:
                    file_results.append(process_one(file_path))
                except Exception as e:
                    logger.error("File processing failed", file=str(file_path), error=str(e))
                    file_results.append(
                        {
                            "file": str(file_path),
                            "status": "error",
                            "error": str(e),
                        }
                    )
        else:
            logger.info("Using concurrent ingestion", workers=self.config.file_concurrency)
            file_results = []
            with ThreadPoolExecutor(max_workers=self.config.file_concurrency) as executor:
                future_to_path = {
                    executor.submit(process_one, file_path): file_path for file_path in files
                }
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    try:
                        file_results.append(future.result())
                    except Exception as e:
                        logger.error("File processing failed", file=str(file_path), error=str(e))
                        file_results.append(
                            {
                                "file": str(file_path),
                                "status": "error",
                                "error": str(e),
                            }
                        )

        for result in file_results:
            if result["status"] == "success":
                stats["processed"] += 1
                stats["total_chunks"] += result.get("chunks", 0)
            elif result["status"] == "skipped":
                stats["skipped"] += 1
            else:
                stats["errors"] += 1

            if "timings_ms" in result:
                for phase, value in result["timings_ms"].items():
                    stats["timings_ms"][phase] = stats["timings_ms"].get(phase, 0) + int(value)

            stats["files"].append(result)

        stats["timings_ms"]["total"] = self._elapsed_ms(ingest_start)
        
        logger.info(
            "Ingestion complete",
            processed=stats["processed"],
            skipped=stats["skipped"],
            errors=stats["errors"],
            total_chunks=stats["total_chunks"],
            timing_total_ms=stats["timings_ms"]["total"],
            timing_parse_ms=stats["timings_ms"].get("parse", 0),
            timing_chunk_ms=stats["timings_ms"].get("chunk", 0),
            timing_embed_ms=stats["timings_ms"].get("embed", 0),
            timing_upsert_ms=stats["timings_ms"].get("upsert", 0),
            timing_hash_ms=stats["timings_ms"].get("hash", 0),
            timing_lookup_ms=stats["timings_ms"].get("lookup", 0),
        )
        
        return stats
    
    def ingest_file(
        self,
        file_path: Path,
        validate_only: bool = False
    ) -> Dict[str, Any]:
        """Ingest a single document file.
        
        Args:
            file_path: Path to document
            validate_only: If True, only validate without ingesting
            
        Returns:
            Dictionary with ingestion result
        """
        logger.info("Processing file", file=str(file_path))
        total_start = self._now()
        timings_ms = {
            "parse": 0,
            "chunk": 0,
            "embed": 0,
            "upsert": 0,
            "hash": 0,
            "lookup": 0,
            "total": 0,
        }
        
        # Get parser for file type
        parser = self.parsers.get(file_path.suffix.lower())
        if not parser:
            logger.warning("Unsupported file type", file=str(file_path), extension=file_path.suffix)
            timings_ms["total"] = self._elapsed_ms(total_start)
            return {
                "file": str(file_path),
                "status": "skipped",
                "reason": f"Unsupported file type: {file_path.suffix}",
                "timings_ms": timings_ms,
            }

        file_hash = ""
        if not validate_only and self.config.enable_incremental:
            hash_start = self._now()
            try:
                file_hash = self._compute_file_hash(file_path)
            finally:
                timings_ms["hash"] = self._elapsed_ms(hash_start)

            lookup_start = self._now()
            file_state = self._get_file_ingestion_state(file_path, file_hash)
            timings_ms["lookup"] = self._elapsed_ms(lookup_start)

            if file_state == "unchanged":
                timings_ms["total"] = self._elapsed_ms(total_start)
                logger.info("Skipping unchanged file", file=str(file_path), file_hash=file_hash)
                return {
                    "file": str(file_path),
                    "status": "skipped",
                    "reason": "File unchanged; already ingested",
                    "file_hash": file_hash,
                    "timings_ms": timings_ms,
                }

            if file_state == "changed":
                logger.info("File changed; replacing existing chunks", file=str(file_path))
                self.delete_file_from_collection(file_path)
        
        # Parse document
        parse_start = self._now()
        try:
            text = parser.parse(file_path)
            metadata = parser.get_metadata(file_path)
        except Exception as e:
            logger.error("Parsing failed", file=str(file_path), error=str(e))
            timings_ms["parse"] = self._elapsed_ms(parse_start)
            timings_ms["total"] = self._elapsed_ms(total_start)
            return {
                "file": str(file_path),
                "status": "error",
                "error": f"Parsing failed: {str(e)}",
                "timings_ms": timings_ms,
            }
        timings_ms["parse"] = self._elapsed_ms(parse_start)
        
        # Validate language and detect actual language
        should_accept, reason, detected_language, confidence = self.validator.should_accept(text)
        
        logger.info(
            "Language detection result",
            file=str(file_path),
            detected=detected_language,
            confidence=f"{confidence:.2%}",
            reason=reason
        )
        
        if validate_only:
            timings_ms["total"] = self._elapsed_ms(total_start)
            return {
                "file": str(file_path),
                "status": "valid",
                "reason": reason,
                "detected_language": detected_language,
                "language_confidence": confidence,
                "metadata": metadata,
                "timings_ms": timings_ms,
            }
        
        # Chunk text
        chunk_start = self._now()
        chunks = self.chunker.chunk(text)
        timings_ms["chunk"] = self._elapsed_ms(chunk_start)
        logger.info("Document chunked", file=str(file_path), chunks=len(chunks))
        
        # Embed chunks
        embed_start = self._now()
        embeddings = self._encode_chunks(chunks)
        timings_ms["embed"] = self._elapsed_ms(embed_start)
        
        # Prepare points for Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            
            # Combine file metadata with config metadata, add detected language
            full_metadata = {
                **self.config.metadata,
                **metadata,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "text": chunk,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "target_language": self.config.target_language,
                "detected_language": detected_language,
                "language_confidence": confidence,
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=self._normalize_vector(embedding),
                payload=full_metadata
            ))
        
        # Upsert to Qdrant
        upsert_start = self._now()
        upsert_calls = self._upsert_points_batched(points)
        timings_ms["upsert"] = self._elapsed_ms(upsert_start)
        
        logger.info(
            "File ingested successfully",
            file=str(file_path),
            chunks=len(chunks),
            upsert_calls=upsert_calls,
            timing_parse_ms=timings_ms["parse"],
            timing_chunk_ms=timings_ms["chunk"],
            timing_embed_ms=timings_ms["embed"],
            timing_upsert_ms=timings_ms["upsert"],
            timing_hash_ms=timings_ms["hash"],
            timing_lookup_ms=timings_ms["lookup"],
        )
        timings_ms["total"] = self._elapsed_ms(total_start)
        
        return {
            "file": str(file_path),
            "status": "success",
            "chunks": len(chunks),
            "metadata": metadata,
            "file_hash": file_hash,
            "upsert_calls": upsert_calls,
            "timings_ms": timings_ms,
        }
    
    def _find_files(self) -> List[Path]:
        """Find all files matching configured patterns recursively.
        
        Returns:
            List of file paths
        """
        files = []
        
        # If source is a file, just return it
        if self.config.source_path.is_file():
            return [self.config.source_path]
        
        # If source is a directory, search recursively
        if not self.config.source_path.is_dir():
            logger.warning("Source path does not exist", path=str(self.config.source_path))
            return []
        
        for pattern in self.config.file_patterns:
            # Ensure pattern supports recursive search with **
            if not pattern.startswith("**/"):
                pattern = f"**/{pattern}"
            
            try:
                matches = list(self.config.source_path.glob(pattern))
                files.extend([f for f in matches if f.is_file()])
                logger.debug("Pattern matched files", pattern=pattern, count=len(matches))
            except Exception as e:
                logger.warning("Glob pattern failed", pattern=pattern, error=str(e))
        
        # Remove duplicates and sort
        files = sorted(list(set(files)))
        logger.info("Files found in directory", total=len(files), patterns=self.config.file_patterns)
        
        return files
    
    def delete_file_from_collection(self, file_path: Path) -> Dict[str, Any]:
        """Delete all chunks from a file from the collection.
        
        Args:
            file_path: Path to the file whose chunks should be deleted
            
        Returns:
            Dictionary with deletion result
        """
        file_path_str = str(file_path)
        logger.info("Deleting file chunks from collection", file=file_path_str)
        
        try:
            # Delete all points with this file_path in payload
            self.qdrant.delete(
                collection_name=self.config.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchValue(value=file_path_str)
                            )
                        ]
                    )
                )
            )
            
            logger.info("File chunks deleted successfully", file=file_path_str)
            return {
                "file": file_path_str,
                "status": "success",
                "action": "deleted"
            }
        except Exception as e:
            logger.error("Failed to delete file chunks", file=file_path_str, error=str(e))
            return {
                "file": file_path_str,
                "status": "error",
                "error": str(e)
            }
    
    def clear_collection(self):
        """Delete and recreate collection (for reindexing)."""
        logger.warning("Clearing collection", collection=self.config.collection_name)

        try:
            self.qdrant.delete_collection(self.config.collection_name)
            logger.info("Collection deleted", collection=self.config.collection_name)
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "404" in message:
                logger.info("Collection not found during clear; creating fresh", collection=self.config.collection_name)
            else:
                logger.error(
                    "Failed to clear collection",
                    collection=self.config.collection_name,
                    error=str(exc),
                )
                raise
        
        self._ensure_collection()
        logger.info("Collection cleared", collection=self.config.collection_name)

