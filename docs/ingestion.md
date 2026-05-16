# Ingestion Pipeline Guide

Complete guide to ingesting domain knowledge into Steuermann.

> **CLI entrypoint:** All ingestion commands run through the unified `steuermann` CLI: `poetry run steuermann ingest <subcommand>`. There is no separate `ingest` script.

---

## Overview

The ingestion pipeline processes domain-specific documents into vector embeddings stored in Qdrant for retrieval-augmented generation (RAG). It supports multiple document formats, language validation, and automatic embedding generation.

**Pipeline Flow:**
```
Documents → Parser → Chunker → Validator → Embedder → Qdrant
```

**Key Features:**
- Multi-format support (PDF, DOCX, Markdown (.md, .markdown), Text (.txt))
- Recursive subdirectory discovery
- Language detection with metadata tagging (all languages accepted)
- Configurable chunking with overlap
- Automatic embedding generation
- Collection management
- Watch mode with periodic fallback check (every 30 seconds)
- Auto-deletion of chunks when source files are removed
- Metadata preservation including detected language

---

## Quick Start

### 1. Basic Usage

```bash
# Install dependencies (if not already installed)
poetry install

# Ingest documents from a directory
poetry run steuermann ingest ingest \
  --source /path/to/documents \
  --collection my-knowledge \
  --language en

# Watch directory for new files
poetry run steuermann ingest watch \
  --source /path/to/documents \
  --collection my-knowledge \
  --language en
```

### 2. Standard Configuration (via `core.yaml`)

The preferred way to configure ingestion is through the active profile overlay. All ingestion parameters come from the `ingestion:` and `rag:` keys in `config/profiles/<profile_id>/core.yaml`:

```yaml
# config/profiles/starter/core.yaml
ingestion:
  source_path: $RAG_DATA_PATH       # or set --source on the CLI
  language: "en"
  language_threshold: 0.8
  embedding_batch_size: 32
  upsert_batch_size: 128
  file_concurrency: 1
  incremental_mode: true

rag:
  collection_name: "my-app"         # collection used for ingest AND retrieval
```

With this in place, running `poetry run steuermann ingest ingest` with no flags uses the profile config automatically.

### 3. Per-Run Override File (`--config`)

You can pass `--config <path>` to override the profile config for a single run. This uses a separate YAML schema (not `core.yaml` format):

```yaml
# my-override.yaml  (any filename, passed via --config)
ingestion:
  source:
    path: "/mnt/knowledge-sources"
    qdrant_host: "localhost"
    qdrant_port: 6333

  processing:
    chunk_size: 512
    chunk_overlap: 50
    language: "en"

  performance:
    file_concurrency: 2
    upsert_batch_size: 128
    enable_incremental: true
    enable_phase_timing: true

  embedding:
    model: "text-embedding-granite-embedding-278m-multilingual"
    provider: "remote"
    remote_endpoint: "${EMBEDDING_SERVER}"
    batch_size: 32
    dimension: 768

  collections:
    - name: "my-app-procedures"
      description: "Standard operating procedures"
      file_patterns:
        - "procedures_*.pdf"
        - "sop_*.docx"
      metadata:
        category: "procedures"
        priority: "high"

  validation:
    reject_threshold: 0.8
```

Then run:

```bash
poetry run steuermann ingest ingest --config my-override.yaml
```

> **Note:** `--config` bypasses `core.yaml` entirely and requires all embedding/collection fields to be explicit. For normal operation, prefer the `core.yaml` approach.

---

## Docker-Based Ingestion

The framework includes a containerized ingestion service that watches a host folder for new documents and automatically ingests them.

### Setup

1. **Create host folder for RAG documents:**
```bash
# Default path used by this project
sudo mkdir -p /data/rag-data
```

On macOS with Docker Desktop:
- Go to **Settings → Resources → File Sharing**
- Add `/data`
- Apply and restart Docker

2. **Start ingestion container:**
```bash
# Copy .env.example to .env and adjust if needed
cp .env.example .env

# Start all services including ingestion
docker compose --env-file .env up -d

# View ingestion logs
docker compose logs -f ingestion
```

3. **Add documents:**
```bash
# Copy documents to /data/rag-data (any supported format)
cp /path/to/my-docs/*.pdf /data/rag-data/
cp /path/to/my-notes/*.txt /data/rag-data/

# Ingestion container auto-detects new files and ingests them
# Check logs or Qdrant UI to confirm
```

### Configuration

Environment variables (in `.env`):
- `RAG_DATA_PATH`: Host path to mount (default: `/data/rag-data`)
- `WORKSPACES_PATH`: Host path for workspace documents (default: `/data/workspaces`)
- `QDRANT_HOST`: Qdrant host (default: `qdrant`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)

Profile-owned ingestion settings (in `config/profiles/<profile_id>/core.yaml`):
- `rag.collection_name`: Canonical Qdrant collection used by both ingestion and retrieval
- `ingestion.language`: Target language code for corpus metadata (default: `de`)
- `ingestion.language_threshold`: Language confidence threshold (default: `0.8`, range: `0.0-1.0`)
- `ingestion.embedding_batch_size`: Chunk embedding batch size (default: `32`)
- `ingestion.upsert_batch_size`: Qdrant upsert batch size (default: `128`)
- `ingestion.file_concurrency`: Parallel file ingestion workers (default: `1`)
- `ingestion.incremental_mode`: Skip unchanged files by hash and replace changed ones (`true`/`false`, default: `true`)
- `ingestion.phase_timing`: Include phase timing metrics in ingestion results (`true`/`false`, default: `true`)

**RAG retrieval alignment:**
- `rag.collection_name` is the single collection identifier. Keep ingestion and retrieval pointed at that same profile-owned value.

**Embedding provider alignment:**
- Ingestion and runtime retrieval should use the same embedding model family and dimensions.
- Set `EMBEDDING_SERVER` and ensure it is reachable from containers (embedding mode is remote-only).
- If embedding dimensions change, recreate the affected Qdrant collections before re-ingesting.

**Language Handling:**
- All languages are accepted regardless of detection result
- Each chunk is tagged with:
  - `detected_language`: Actual detected language (e.g., "en", "de", "unknown")
  - `language_confidence`: Detection confidence (0.0-1.0)
  - `target_language`: Expected corpus language from config
- Lower `core.ingestion.language_threshold` for mixed-language or technical documents

**Watch Mode Features:**
- Initial sweep ingests all existing files on startup
- Watchdog monitors for new/deleted files in real-time
- Periodic fallback check (every 30 seconds) catches missed files
- File deletion automatically removes associated chunks from Qdrant
- Recursive subdirectory support - files in any nested folder are discovered

**Performance and Incremental Features:**
- Per-file phase timing metrics are captured (`parse`, `chunk`, `embed`, `upsert`, `hash`, `lookup`, `total`)
- Directory-level timing summary aggregates phase timings across files
- Incremental hashing stores `file_hash` in payload and skips unchanged documents
- Changed files are detected via hash mismatch and old chunks are replaced automatically
- Embedding and upsert operations run in configurable batches to improve throughput and reduce peak memory

Startup behavior (containerized watch mode):
- On launch, the ingestion service performs an initial scan of the mounted source path and ingests any existing documents before starting the watchdog.
- After the initial sweep, new files dropped into the source path are ingested automatically.

To change collection or language defaults, edit the active profile's `rag` / `ingestion` section and restart the ingestion service.

---

## Supported Document Formats

### PDF Documents

**Extensions:** `.pdf`

**Metadata extracted:**
- Title
- Author
- Subject
- Creator
- Producer
- Creation date
- Page count

**Example:**
```bash
poetry run steuermann ingest ingest \
  --source /docs/manuals.pdf \
  --collection manuals
```

### Word Documents

**Extensions:** `.docx`

**Metadata extracted:**
- Title
- Author
- Subject
- Keywords
- Created date
- Modified date
- Last modified by

**Example:**
```bash
poetry run steuermann ingest ingest \
  --source /docs/reports \
  --collection reports
```

### Markdown Files

**Extensions:** `.md`, `.markdown`

**Metadata extracted:**
- Title (from first heading)
- File name
- Creation date
- Modified date

**Example:**
```bash
poetry run steuermann ingest ingest \
  --source /docs/wiki/*.md \
  --collection wiki
```

---

### Text Files

**Extensions:** `.txt`

**Metadata extracted:**
- Title (first non-empty line)
- File name
- Creation date
- Modified date

**Example:**
```bash
poetry run steuermann ingest ingest \
  --source /docs/notes/*.txt \
  --collection notes
```

---

## CLI Commands

### `ingest` - Manual Ingestion

Ingest documents from a directory or file.

**Syntax:**
```bash
poetry run steuermann ingest ingest [OPTIONS]
```

**Options:**
- `--source PATH` - Source directory or file (required)
- `--config PATH` - Configuration YAML file
- `--collection NAME` - Collection name
- `--language CODE` - Target language (en, de, fr, etc.)
- `--validate-only` - Only validate, don't ingest
- `--verbose, -v` - Show detailed output

**Examples:**

```bash
# Basic ingestion
poetry run steuermann ingest ingest \
  --source /data/documents \
  --collection docs \
  --language en

# With per-run override file (bypasses core.yaml; must include all embedding/collection fields)
poetry run steuermann ingest ingest \
  --config my-override.yaml

# Validate only (dry run)
poetry run steuermann ingest ingest \
  --source /data/documents \
  --language en \
  --validate-only \
  --verbose
```

### `watch` - Auto-Ingestion

Watch a directory for new files and automatically ingest them.

**Syntax:**
```bash
poetry run steuermann ingest watch [OPTIONS]
```

**Options:**
- `--source PATH` - Source directory to watch (required)
- `--config PATH` - Configuration YAML file
- `--collection NAME` - Collection name
- `--language CODE` - Target language

**Example:**

```bash
# Watch directory
poetry run steuermann ingest watch \
  --source /data/incoming \
  --collection live-docs \
  --language en

# Press Ctrl+C to stop
```

**Use cases:**
- Development: Auto-ingest as you add documents
- Production: Monitor shared folder for new content
- Testing: Quickly iterate on document sets

### `validate` - Document Validation

Validate documents without ingesting (language check, parsing test).

**Syntax:**
```bash
poetry run steuermann ingest validate [OPTIONS]
```

**Options:**
- `--source PATH` - Source directory or file (required)
- `--config PATH` - Configuration YAML file
- `--language CODE` - Target language
- `--verbose, -v` - Show detailed results

**Example:**

```bash
# Validate documents
poetry run steuermann ingest validate \
  --source /data/new-docs \
  --language de \
  --verbose
```

**Output:**
```
✓ document1.pdf - Accepted (de, confidence: 0.95)
✓ document2.pdf - Accepted (en, confidence: 0.87)
✗ document3.pdf - Error (Parsing failed: Corrupted file)
```

### `reindex` - Collection Reindex

Clear and rebuild a collection from source.

**Syntax:**
```bash
poetry run steuermann ingest reindex [OPTIONS]
```

**Options:**
- `--source PATH` - Source directory (required)
- `--config PATH` - Configuration YAML file
- `--collection NAME` - Collection name
- `--language CODE` - Target language
- `--yes, -y` - Skip confirmation prompt

**Example:**

```bash
# Reindex collection (with confirmation)
poetry run steuermann ingest reindex \
  --source /data/documents \
  --collection docs \
  --language en

# Skip confirmation
poetry run steuermann ingest reindex \
  --source /data/documents \
  --collection docs \
  --yes
```

**⚠️ Warning:** This deletes all existing vectors in the collection.

---

## Configuration Reference

### Complete Configuration Schema

```yaml
ingestion:
  # Source configuration
  source:
    type: "filesystem"              # or "s3" (future)
    path: "/mnt/knowledge-sources"
    watch: true                     # Enable watch mode
    qdrant_host: "localhost"
    qdrant_port: 6333
  
  # Processing configuration
  processing:
    chunk_size: 512                 # Characters per chunk
    chunk_overlap: 50               # Overlap between chunks
    language: "en"                  # Target language
  
  # Embedding configuration
  embedding:
    model: "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: 32                  # Embeddings per batch
  
  # Collection definitions
  collections:
    - name: "app-procedures"
      description: "Standard operating procedures"
      file_patterns:
        - "procedures_*.pdf"
        - "sop_*.docx"
      metadata:
        category: "procedures"
        priority: "high"
    
    - name: "app-guidelines"
      description: "Guidelines and policies"
      file_patterns:
        - "guidelines_*.md"
        - "policies_*.pdf"
      metadata:
        category: "guidelines"
        source: "official"
  
  # Validation configuration
  validation:
    language_detection: true        # Detect and tag language (all languages accepted)
    reject_threshold: 0.8           # Not used for rejection as of 2026-01-22
```

### Chunking Configuration

**chunk_size:**
- Default: 512
- Recommended: 256-1024
- Smaller chunks: More precise retrieval, higher storage
- Larger chunks: More context, less precise

**chunk_overlap:**
- Default: 50
- Recommended: 10-20% of chunk_size
- Prevents splitting sentences/paragraphs

**Example:**
```yaml
processing:
  chunk_size: 768        # Larger chunks for technical docs
  chunk_overlap: 100     # ~13% overlap
  language: "en"
```

### Embedding Models

**Available models:**

1. **paraphrase-multilingual-MiniLM-L12-v2** (Default)
   - Dimensions: 384
   - Languages: 50+
   - Speed: Fast
   - Use case: General multilingual

2. **all-MiniLM-L6-v2**
   - Dimensions: 384
   - Languages: English only
   - Speed: Very fast
   - Use case: English-only apps

3. **all-mpnet-base-v2**
   - Dimensions: 768
   - Languages: English only
   - Speed: Slower
   - Use case: High quality English

4. **distiluse-base-multilingual-cased-v2**
   - Dimensions: 512
   - Languages: 15+
   - Speed: Medium
   - Use case: Multilingual with quality

**Changing models:**
```yaml
embedding:
  model: "all-mpnet-base-v2"
  # Note: Must match dimension in memory.embeddings.dimension
```

**⚠️ Important:** Changing models requires reindexing all collections.

### Language Validation

**Note:** Language validation has been changed to **accept all languages** and tag chunks with detected language metadata.

**reject_threshold / core.ingestion.language_threshold:**
- Default: 0.8
- Range: 0.0-1.0
- **Not used for rejection** - only for logging/metadata purposes
- Documents are accepted regardless of detected language

**Chunk Metadata:**
Each ingested chunk includes language metadata:
```json
{
  "text": "chunk content...",
  "file_path": "/data/ingest/document.pdf",
  "target_language": "de",          // Expected language from config
  "detected_language": "en",        // Actually detected language
  "language_confidence": 0.87,      // Detection confidence (0.0-1.0)
  "chunk_index": 0,
  "chunk_count": 5
}
```

**Use cases:**
- Filter by `detected_language` at query time if needed
- Monitor language distribution in your corpus
- Accept mixed-language or technical documents without rejection
- Track confidence for quality metrics

**Example configuration:**
```yaml
processing:
  language: "de"  # Target language (for reference only)

validation:
  language_detection: true  # Still performs detection for tagging
  reject_threshold: 0.8     # Not used for rejection
```

**Profile setting:**
```yaml
ingestion:
  language_threshold: 0.8  # Can be set but doesn't affect acceptance
```

---

## Programmatic Usage

### Basic Example

```python
from pathlib import Path
from universal_agentic_framework.ingestion import IngestionService, IngestionConfig

# Create configuration
config = IngestionConfig(
    source_path=Path("/data/documents"),
    file_patterns=["**/*.pdf", "**/*.docx"],
    collection_name="my-knowledge",
    collection_description="Domain knowledge base",
    target_language="en",
    chunk_size=512,
    chunk_overlap=50,
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    embedding_dimension=384,
    qdrant_host="localhost",
    qdrant_port=6333,
    metadata={"source": "production", "version": "1.0"}
)

# Create service
service = IngestionService(config)

# Ingest directory
stats = service.ingest_directory()

print(f"Processed: {stats['processed']}")
print(f"Total chunks: {stats['total_chunks']}")
```

### Advanced Example

```python
from pathlib import Path
from universal_agentic_framework.ingestion import (
    IngestionService,
    IngestionConfig,
    LanguageValidator
)

# Custom validation
validator = LanguageValidator(
    target_language="de",
    confidence_threshold=0.85
)

# Check document before processing
with open("document.pdf", "rb") as f:
    text = extract_text(f)  # Your extraction logic

is_valid, detected_lang, confidence = validator.validate(text)

if not is_valid:
    print(f"Validation failed: {detected_lang} (confidence: {confidence})")
else:
    # Process document
    config = IngestionConfig(
        source_path=Path("document.pdf"),
        file_patterns=["*.pdf"],
        collection_name="german-docs",
        target_language="de",
    )
    
    service = IngestionService(config)
    result = service.ingest_file(Path("document.pdf"))
    
    print(f"Ingested: {result['chunks']} chunks")
```

---

## Collection Management

### Listing Collections

```bash
# Using Qdrant API
curl http://localhost:6333/collections

# Using Python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collections = client.get_collections()

for collection in collections.collections:
    print(f"{collection.name}: {collection.vectors_count} vectors")
```

### Inspecting Collection

```bash
# Get collection info
curl http://localhost:6333/collections/my-knowledge

# Response:
{
  "result": {
    "status": "green",
    "vectors_count": 1523,
    "indexed_vectors_count": 1523,
    "points_count": 1523
  }
}
```

### Deleting Collection

```bash
# Via CLI (safe - requires confirmation)
poetry run steuermann ingest reindex \
  --source /data/documents \
  --collection my-knowledge

# Via API (immediate)
curl -X DELETE http://localhost:6333/collections/my-knowledge

# Via Python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
client.delete_collection("my-knowledge")
```

---

## Monitoring & Logging

### Structured Logging

Ingestion uses structured logging (JSON format):

```json
{
  "event": "Processing file",
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info",
  "file": "/data/docs/manual.pdf"
}
```

**Log levels:**
- `INFO`: Normal operations
- `WARNING`: Skipped files, validation failures
- `ERROR`: Parsing errors, Qdrant issues

### Progress Tracking

```python
from universal_agentic_framework.ingestion import IngestionService

service = IngestionService(config)

# Ingest with progress
stats = service.ingest_directory()

# stats contains:
{
    "processed": 42,      # Successfully processed
    "skipped": 3,         # Validation failed
    "errors": 1,          # Parsing/embedding errors
    "total_chunks": 2150, # Total chunks created
    "files": [            # Per-file results
        {
            "file": "/data/docs/manual.pdf",
            "status": "success",
            "chunks": 45,
            "metadata": {...}
        }
    ]
}
```

### Prometheus Metrics

Ingestion operations are tracked via Prometheus (if monitoring enabled):

**Metrics:**
- `ingestion_files_total{status="success|skipped|error"}`
- `ingestion_chunks_total`
- `ingestion_duration_seconds`
- `embedding_batch_duration_seconds`

**Query example:**
```promql
# Files ingested per hour
rate(ingestion_files_total{status="success"}[1h]) * 3600
```

---

## Troubleshooting

### Issue: Documents not being ingested

**Symptom:**
```
File not found or ingestion skipped
```

**Cause:** Document may be in unsupported format or watchdog missed the file.

**Solutions:**

1. **Check file format:**
   - Supported: .pdf, .docx, .md, .markdown, .txt
   - Verify file extension matches

2. **Use periodic check:**
   - Ingestion service runs 30-second fallback scan
   - Wait up to 30 seconds after file creation

3. **Manual ingestion:**
```bash
docker compose restart ingestion
```

### Issue: Chunks missing expected language

**Symptom:**
```
Chunks have detected_language != target_language
```

**Cause:** Document contains mixed languages or detection failed.

**Solutions:**

1. **Check metadata:**
```python
from universal_agentic_framework.ingestion import LanguageValidator

validator = LanguageValidator("de")
is_valid, detected, confidence = validator.validate(text)
print(f"Detected: {detected}, Confidence: {confidence}")
```

3. **Disable validation:**
```yaml
validation:
  language_detection: false
```

### Issue: Parsing failed for PDF

**Symptom:**
```
Error: Parsing failed: EOF marker not found
```

**Cause:** Corrupted or encrypted PDF.

**Solutions:**

1. **Verify PDF:**
```bash
pdfinfo document.pdf
# or
pdftk document.pdf dump_data
```

2. **Re-save PDF:**
Open in PDF viewer and "Save As" to rebuild structure.

3. **Check encryption:**
```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")
if reader.is_encrypted:
    print("PDF is encrypted")
```

### Issue: Out of memory during embedding

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Cause:** Too many documents or large embedding batches.

**Solutions:**

1. **Reduce batch size:**
```yaml
embedding:
  batch_size: 16  # Reduce from 32
```

2. **Process in smaller batches:**
```bash
# Split directory and ingest separately
poetry run steuermann ingest ingest --source /data/docs/batch1
poetry run steuermann ingest ingest --source /data/docs/batch2
```

3. **Use smaller embedding model:**
```yaml
embedding:
  model: "all-MiniLM-L6-v2"  # 384d instead of 768d
```

### Issue: Qdrant connection refused

**Symptom:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Cause:** Qdrant not running or wrong host/port.

**Solutions:**

1. **Check Qdrant is running:**
```bash
docker ps | grep qdrant
# or
curl http://localhost:6333/collections
```

2. **Start Qdrant:**
```bash
docker compose up -d qdrant
```

3. **Check configuration:**
```yaml
source:
  qdrant_host: "localhost"  # or "qdrant" in Docker
  qdrant_port: 6333
```

### Issue: Watch mode not detecting files

**Symptom:**
Files added to directory but not auto-ingested.

**Cause:** File permissions, network mounts, or unsupported file system.

**Solutions:**

1. **Check file permissions:**
```bash
ls -la /data/incoming/
# Files should be readable
```

2. **Test manually:**
```bash
# Stop watch mode, try manual ingest
poetry run steuermann ingest ingest --source /data/incoming/new_file.pdf
```

3. **Use polling (slower but more compatible):**
Edit watch implementation to use polling instead of OS events.

---

## Best Practices

### 1. Organize Source Documents

```
/data/knowledge-sources/
├── procedures/
│   ├── sop_001.pdf
│   ├── sop_002.pdf
├── guidelines/
│   ├── guideline_a.md
│   ├── guideline_b.md
└── reference/
    ├── manual.pdf
    └── glossary.docx
```

### 2. Use Descriptive Collection Names

**Good:**
- `medical-ai-de-procedures`
- `financial-ai-en-regulations`
- `legal-ai-de-contracts`

**Bad:**
- `collection1`
- `docs`
- `data`

### 3. Add Rich Metadata

```yaml
collections:
  - name: "medical-procedures"
    metadata:
      source: "hospital-X"
      validated_by: "Dr. Smith"
      validation_date: "2024-01-15"
      version: "2.1"
      compliance: "HIPAA"
```

### 4. Version Your Collections

When updating knowledge:

```bash
# Create new versioned collection
poetry run steuermann ingest ingest \
  --source /data/docs-v2 \
  --collection my-knowledge-v2

# Test with new collection
# If good, delete old collection
curl -X DELETE http://localhost:6333/collections/my-knowledge-v1

# Rename new collection (or update app config)
```

### 5. Validate Before Production

```bash
# Always validate first
poetry run steuermann ingest validate \
  --source /data/new-docs \
  --verbose

# Review output, fix issues
# Then ingest
poetry run steuermann ingest ingest --source /data/new-docs
```

### 6. Monitor Collection Size

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost")
info = client.get_collection("my-knowledge")

print(f"Vectors: {info.vectors_count}")
print(f"Points: {info.points_count}")

# Estimate size
approx_size_mb = (info.vectors_count * 384 * 4) / (1024 * 1024)
print(f"Approx size: {approx_size_mb:.2f} MB")
```

### 7. Backup Collections

```bash
# Qdrant snapshot API
curl -X POST 'http://localhost:6333/collections/my-knowledge/snapshots'

# Download snapshot
curl 'http://localhost:6333/collections/my-knowledge/snapshots/snapshot-2024-01-15.snapshot' \
  --output backup.snapshot

# Restore
curl -X PUT 'http://localhost:6333/collections/my-knowledge/snapshots/upload' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @backup.snapshot
```

---

## Performance Optimization

### Chunking Strategy

**For technical documentation:**
```yaml
processing:
  chunk_size: 768      # Larger for context
  chunk_overlap: 100
```

**For conversational content:**
```yaml
processing:
  chunk_size: 384      # Smaller for precision
  chunk_overlap: 50
```

### Embedding Optimization

**CPU-only (no GPU):**
```python
# Use smallest model
embedding:
  model: "all-MiniLM-L6-v2"
  batch_size: 16
```

**With GPU:**
```python
# Use larger model
embedding:
  model: "all-mpnet-base-v2"
  batch_size: 64
```

### Batch Processing

For large document sets:

```bash
# Process in parallel (multiple terminals)
poetry run steuermann ingest ingest --source /data/batch1 --collection kb &
poetry run steuermann ingest ingest --source /data/batch2 --collection kb &
poetry run steuermann ingest ingest --source /data/batch3 --collection kb &
wait
```

---

## See Also

- [Configuration Reference](configuration.md) - Full config schema
- [Memory Architecture](technical_architecture.md#6-memory-architecture) - How ingested data is used
- [Profile Setup Guide](profile_creation.md) - Setting up domain knowledge
- [Qdrant Documentation](https://qdrant.tech/documentation/) - Vector store details
