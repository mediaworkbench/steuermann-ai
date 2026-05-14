"""Command-line interface for document ingestion."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
import threading
from typing import Dict, Any
import yaml

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from universal_agentic_framework.ingestion import IngestionService, IngestionConfig
from universal_agentic_framework.monitoring.logging import get_logger, configure_logging

logger = get_logger(__name__)


@dataclass(frozen=True)
class RuntimeIngestionDefaults:
    source_path: str
    collection_name: str
    language: str
    language_threshold: float
    embedding_batch_size: int
    upsert_batch_size: int
    file_concurrency: int
    incremental_mode: bool
    phase_timing: bool
    reingest_timeout_seconds: int
    embedding_provider_type: str
    embedding_remote_endpoint: str | None
    embedding_model: str
    embedding_dimension: int
    qdrant_host: str
    qdrant_port: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_env_placeholder(value: Any, default: str | None = None) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        env_name = value[1:]
        return os.getenv(env_name, default if default is not None else value)
    if value in (None, ""):
        return default
    return value


def resolve_runtime_ingestion_defaults() -> RuntimeIngestionDefaults:
    from universal_agentic_framework.config import load_core_config

    core_config = load_core_config()
    ingestion_config = core_config.ingestion
    rag_config = core_config.rag

    source_path = _resolve_env_placeholder(ingestion_config.source_path)
    if not source_path:
        raise ValueError("core.ingestion.source_path must be set in the active profile")

    collection_name = rag_config.collection_name if rag_config and rag_config.collection_name else None
    if not collection_name:
        raise ValueError("core.rag.collection_name must be set in the active profile")

    return RuntimeIngestionDefaults(
        source_path=str(source_path),
        collection_name=str(collection_name),
        language=ingestion_config.language,
        language_threshold=float(ingestion_config.language_threshold),
        embedding_batch_size=int(ingestion_config.embedding_batch_size),
        upsert_batch_size=int(ingestion_config.upsert_batch_size),
        file_concurrency=int(ingestion_config.file_concurrency),
        incremental_mode=bool(ingestion_config.incremental_mode),
        phase_timing=bool(ingestion_config.phase_timing),
        reingest_timeout_seconds=int(ingestion_config.reingest_timeout_seconds),
        embedding_provider_type=core_config.memory.embeddings.provider,
        embedding_remote_endpoint=_resolve_env_placeholder(
            core_config.memory.embeddings.remote_endpoint,
            os.getenv("EMBEDDING_SERVER"),
        ),
        embedding_model=core_config.memory.embeddings.model,
        embedding_dimension=int(core_config.memory.embeddings.dimension),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
    )


def _build_runtime_ingestion_config(
    *,
    source_override: str | None,
    collection_override: str | None,
    language_override: str | None,
    collection_description: str,
) -> IngestionConfig:
    defaults = resolve_runtime_ingestion_defaults()
    return IngestionConfig(
        source_path=Path(source_override or defaults.source_path),
        file_patterns=["**/*.pdf", "**/*.docx", "**/*.md", "**/*.markdown", "**/*.txt"],
        collection_name=collection_override or defaults.collection_name,
        collection_description=collection_description,
        target_language=language_override or defaults.language,
        language_threshold=defaults.language_threshold,
        file_concurrency=defaults.file_concurrency,
        upsert_batch_size=defaults.upsert_batch_size,
        enable_incremental=defaults.incremental_mode,
        enable_phase_timing=defaults.phase_timing,
        embedding_model=defaults.embedding_model,
        embedding_dimension=defaults.embedding_dimension,
        embedding_batch_size=defaults.embedding_batch_size,
        embedding_provider_type=defaults.embedding_provider_type,
        embedding_remote_endpoint=defaults.embedding_remote_endpoint,
        qdrant_host=defaults.qdrant_host,
        qdrant_port=defaults.qdrant_port,
    )


class DocumentEventHandler(FileSystemEventHandler):
    """Watch for new documents and ingest them automatically."""
    
    def __init__(self, service: IngestionService):
        self.service = service
        self.processing = set()
    
    def on_created(self, event):
        """Handle new file creation."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if file type is supported (keep in sync with file_patterns)
        if file_path.suffix.lower() not in [".pdf", ".docx", ".md", ".markdown", ".txt"]:
            return
        
        # Avoid duplicate processing
        if str(file_path) in self.processing:
            return
        
        self.processing.add(str(file_path))
        
        try:
            logger.info("New file detected", file=str(file_path))
            
            # Wait a moment to ensure file is fully written
            time.sleep(1)
            
            result = self.service.ingest_file(file_path)
            
            if result["status"] == "success":
                logger.info(
                    "Auto-ingestion successful",
                    file=str(file_path),
                    chunks=result["chunks"]
                )
            else:
                logger.warning(
                    "Auto-ingestion skipped or failed",
                    file=str(file_path),
                    status=result["status"],
                    reason=result.get("reason") or result.get("error")
                )
        
        except Exception as e:
            logger.error("Auto-ingestion error", file=str(file_path), error=str(e))
        
        finally:
            self.processing.discard(str(file_path))
    
    def on_deleted(self, event):
        """Handle file deletion."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if file type is supported (keep in sync with file_patterns)
        if file_path.suffix.lower() not in [".pdf", ".docx", ".md", ".markdown", ".txt"]:
            return
        
        # Avoid duplicate processing
        if str(file_path) in self.processing:
            return
        
        self.processing.add(str(file_path))
        
        try:
            logger.info("File deleted", file=str(file_path))
            result = self.service.delete_file_from_collection(file_path)
            
            if result["status"] == "success":
                logger.info(
                    "Auto-deletion successful",
                    file=str(file_path)
                )
            else:
                logger.warning(
                    "Auto-deletion failed",
                    file=str(file_path),
                    error=result.get("error")
                )
        
        except Exception as e:
            logger.error("Auto-deletion error", file=str(file_path), error=str(e))
        
        finally:
            self.processing.discard(str(file_path))    

def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load ingestion configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file cannot be read or YAML is malformed
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file {config_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load configuration file {config_path}: {e}") from e
    
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must be a YAML object, got {type(config).__name__}")
    
    # Extract ingestion config
    ingestion_config = config.get("ingestion", {})
    
    return ingestion_config


def create_service_from_config(config_dict: Dict[str, Any]) -> IngestionService:
    """Create ingestion service from config dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configured IngestionService
    """
    source_config = config_dict.get("source", {})
    processing_config = config_dict.get("processing", {})
    embedding_config = config_dict.get("embedding", {})
    performance_config = config_dict.get("performance", {})
    
    # Get first collection (simplified - in real usage might process multiple)
    collections = config_dict.get("collections", [])
    if not collections:
        raise ValueError("No collections defined in configuration")
    
    collection = collections[0]
    
    # Get embedding config
    embedding_config = config_dict.get("embedding", {})
    embedding_model = embedding_config.get("model", "text-embedding-granite-embedding-278m-multilingual")
    embedding_dimension = embedding_config.get("dimension", 768)
    embedding_provider_type = embedding_config.get("provider", "local")
    embedding_remote_endpoint = embedding_config.get("remote_endpoint", None)
    
    config = IngestionConfig(
        source_path=Path(source_config.get("path", "/data/ingest")),
        file_patterns=collection.get("file_patterns", ["**/*.pdf", "**/*.docx", "**/*.md"]),
        collection_name=collection.get("name", "default"),
        collection_description=collection.get("description", ""),
        chunk_size=processing_config.get("chunk_size", 512),
        chunk_overlap=processing_config.get("chunk_overlap", 50),
        file_concurrency=performance_config.get("file_concurrency", 1),
        upsert_batch_size=performance_config.get("upsert_batch_size", 128),
        enable_incremental=performance_config.get("enable_incremental", True),
        enable_phase_timing=performance_config.get("enable_phase_timing", True),
        target_language=processing_config.get("language", "en"),
        language_threshold=config_dict.get("validation", {}).get("reject_threshold", 0.8),
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        embedding_batch_size=embedding_config.get("batch_size", 32),
        embedding_provider_type=embedding_provider_type,
        embedding_remote_endpoint=embedding_remote_endpoint,
        qdrant_host=source_config.get("qdrant_host", "localhost"),
        qdrant_port=source_config.get("qdrant_port", 6333),
        metadata=collection.get("metadata", {})
    )
    
    return IngestionService(config)


def cmd_ingest(args):
    """Execute manual ingestion."""
    configure_logging()
    
    logger.info("Starting ingestion", source=args.source, config=args.config)
    
    if args.config:
        # Load from config file
        config_dict = load_config_from_yaml(Path(args.config))
        
        # Override source if provided
        if args.source:
            config_dict.setdefault("source", {})["path"] = args.source
        
        service = create_service_from_config(config_dict)
    else:
        config = _build_runtime_ingestion_config(
            source_override=args.source,
            collection_override=args.collection,
            language_override=args.language,
            collection_description="Manual ingestion collection",
        )
        service = IngestionService(config)
    
    # Run ingestion
    stats = service.ingest_directory(validate_only=args.validate_only)
    
    # Log results
    logger.info(
        "Ingestion complete",
        processed=stats['processed'],
        skipped=stats['skipped'],
        errors=stats['errors'],
        total_chunks=stats['total_chunks'],
        total_time_ms=stats.get("timings_ms", {}).get("total", 0)
    )
    
    if args.verbose:
        for file_result in stats["files"]:
            status_level = "info" if file_result["status"] == "success" else "warning"
            log_fn = getattr(logger, status_level)
            log_fn(
                f"{file_result['file']} - {file_result['status']}",
                file=file_result['file'],
                status=file_result['status'],
                chunks=file_result.get('chunks'),
                reason=file_result.get('reason') or file_result.get('error')
            )
    
    return 0 if stats["errors"] == 0 else 1


def cmd_watch(args):
    """Watch directory for new files and auto-ingest."""
    configure_logging()
    
    logger.info("Starting watch mode", source=args.source, config=args.config)
    
    if args.config:
        config_dict = load_config_from_yaml(Path(args.config))
        if args.source:
            config_dict.setdefault("source", {})["path"] = args.source
        service = create_service_from_config(config_dict)
    else:
        config = _build_runtime_ingestion_config(
            source_override=args.source,
            collection_override=args.collection,
            language_override=args.language,
            collection_description="Watch mode collection",
        )
        service = IngestionService(config)
    
    # Create observer
    event_handler = DocumentEventHandler(service)
    observer = Observer()
    observer.schedule(
        event_handler,
        str(service.config.source_path),
        recursive=True
    )
    
    # Track ingested files to avoid re-processing
    ingested_files = set()
    
    def periodic_check():
        """Periodically check for files that may have been missed by watchdog."""
        while observer.is_alive():
            try:
                time.sleep(30)  # Check every 30 seconds
                current_files = set(str(f) for f in service._find_files())
                new_files = current_files - ingested_files
                
                if new_files:
                    logger.info("Periodic check found new files", count=len(new_files))
                    for file_path in new_files:
                        try:
                            result = service.ingest_file(Path(file_path))
                            if result["status"] == "success":
                                ingested_files.add(file_path)
                                logger.info("Periodic check ingested file", file=file_path)
                        except Exception as e:
                            logger.debug("Periodic check ingest failed", file=file_path, error=str(e))
                
                ingested_files.update(current_files)
            except Exception as e:
                logger.debug("Periodic check failed", error=str(e))
    
    check_thread = threading.Thread(target=periodic_check, daemon=True)
    check_thread.start()
    
    logger.info(
        "Watch mode started",
        source_path=str(service.config.source_path),
        collection=service.config.collection_name,
        language=service.config.target_language
    )
    
    # Initial scan and ingest of existing documents
    logger.info("Scanning for existing documents")
    try:
        stats = service.ingest_directory()
        if stats["processed"] > 0:
            logger.info(
                "Initial ingestion complete",
                processed=stats['processed'],
                total_chunks=stats['total_chunks'],
                skipped=stats.get('skipped', 0),
                errors=stats.get('errors', 0)
            )
        else:
            logger.info("No new documents found in initial scan")
    except Exception as e:
        logger.warning("Initial scan failed", error=str(e))
        print(f"⚠️  Initial scan failed: {e}")
    
    print("\n👁️  Now watching for new documents...\n")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watch mode")
        observer.stop()
        print("\n\nWatch mode stopped.")
    
    observer.join()
    return 0


def cmd_validate(args):
    """Validate documents without ingesting."""
    configure_logging()
    
    logger.info("Starting validation", source=args.source)
    
    # Run with validate_only flag
    args.validate_only = True
    return cmd_ingest(args)


def cmd_reindex(args):
    """Clear and reindex collection."""
    configure_logging()
    
    logger.warning("Starting reindex", collection=args.collection)
    
    if args.config:
        config_dict = load_config_from_yaml(Path(args.config))
        if args.source:
            config_dict.setdefault("source", {})["path"] = args.source
        service = create_service_from_config(config_dict)
    else:
        config = _build_runtime_ingestion_config(
            source_override=args.source,
            collection_override=args.collection,
            language_override=args.language,
            collection_description="Reindex collection",
        )
        service = IngestionService(config)
    
    # Confirm before clearing
    if not args.yes:
        response = input(f"⚠️  This will DELETE collection '{service.config.collection_name}'. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Reindex cancelled.")
            return 0
    
    # Clear collection
    service.clear_collection()
    print(f"✓ Collection '{service.config.collection_name}' cleared.")
    
    # Re-ingest
    print(f"\nRe-ingesting documents from {service.config.source_path}...")
    stats = service.ingest_directory()
    
    print("\n" + "=" * 60)
    print("REINDEX COMPLETE")
    print("=" * 60)
    print(f"Files processed:  {stats['processed']}")
    print(f"Total chunks:     {stats['total_chunks']}")
    print("=" * 60)
    
    return 0


def add_ingest_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Register ingestion subcommands under an argparse subparser group."""
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--source", help="Source directory or file (defaults to core.ingestion.source_path)")
    ingest_parser.add_argument("--config", help="Path to configuration YAML file")
    ingest_parser.add_argument("--collection", help="Collection name override")
    ingest_parser.add_argument("--language", help="Target language override")
    ingest_parser.add_argument("--validate-only", action="store_true", help="Only validate, don't ingest")
    ingest_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ingest_parser.set_defaults(func=cmd_ingest)

    watch_parser = subparsers.add_parser("watch", help="Watch directory for new files")
    watch_parser.add_argument("--source", help="Source directory to watch (defaults to core.ingestion.source_path)")
    watch_parser.add_argument("--config", help="Path to configuration YAML file")
    watch_parser.add_argument("--collection", help="Collection name override")
    watch_parser.add_argument("--language", help="Target language override")
    watch_parser.set_defaults(func=cmd_watch)

    validate_parser = subparsers.add_parser("validate", help="Validate documents only")
    validate_parser.add_argument("--source", help="Source directory or file (defaults to core.ingestion.source_path)")
    validate_parser.add_argument("--config", help="Path to configuration YAML file")
    validate_parser.add_argument("--language", help="Target language override")
    validate_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    validate_parser.set_defaults(func=cmd_validate)

    reindex_parser = subparsers.add_parser("reindex", help="Clear and reindex collection")
    reindex_parser.add_argument("--source", help="Source directory (defaults to core.ingestion.source_path)")
    reindex_parser.add_argument("--config", help="Path to configuration YAML file")
    reindex_parser.add_argument("--collection", help="Collection name override")
    reindex_parser.add_argument("--language", help="Target language override")
    reindex_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    reindex_parser.set_defaults(func=cmd_reindex)


def create_parser() -> argparse.ArgumentParser:
    """Create standalone ingestion parser."""
    parser = argparse.ArgumentParser(description="Document ingestion CLI for Steuermann")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    add_ingest_subcommands(subparsers)
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Run ingestion parser with optional argv."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        logger.error("Command failed", error=str(e), exc_info=True)
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
    return run_cli()


if __name__ == "__main__":
    sys.exit(main())
