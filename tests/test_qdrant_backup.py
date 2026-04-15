"""Tests for Qdrant backup and restore functionality."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import tempfile
import shutil

from universal_agentic_framework.caching.backup import (
    QdrantBackupManager,
    create_backup_manager
)


@pytest.fixture
def temp_backup_dir():
    """Create temporary backup directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def backup_manager(temp_backup_dir):
    """Create backup manager with temp directory."""
    with patch("universal_agentic_framework.caching.backup.QdrantClient"):
        manager = QdrantBackupManager(
            host="localhost",
            port=6333,
            backup_dir=temp_backup_dir,
            timeout=30
        )
        return manager


@pytest.mark.asyncio
async def test_backup_manager_initialization(temp_backup_dir):
    """Test backup manager initialization."""
    with patch("universal_agentic_framework.caching.backup.QdrantClient"):
        manager = QdrantBackupManager(
            host="localhost",
            port=6333,
            backup_dir=temp_backup_dir
        )
        
        assert manager.host == "localhost"
        assert manager.port == 6333
        assert manager.base_url == "http://localhost:6333"
        assert manager.backup_dir == Path(temp_backup_dir)
        assert manager.backup_dir.exists()


@pytest.mark.asyncio
async def test_create_snapshot(backup_manager):
    """Test snapshot creation."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"name": "test-snapshot-123.snapshot"}
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_ctx
        
        snapshot_name = await backup_manager.create_snapshot("test_collection")
        
        assert snapshot_name == "test-snapshot-123.snapshot"
        mock_ctx.__aenter__.return_value.post.assert_called_once()


@pytest.mark.asyncio
async def test_create_snapshot_failure(backup_manager):
    """Test snapshot creation failure handling."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.post = AsyncMock(
            side_effect=Exception("Connection error")
        )
        mock_client.return_value = mock_ctx
        
        with pytest.raises(Exception, match="Connection error"):
            await backup_manager.create_snapshot("test_collection")


@pytest.mark.asyncio
async def test_list_snapshots(backup_manager):
    """Test listing snapshots."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": [
            {"name": "snapshot1.snapshot", "size": 1024},
            {"name": "snapshot2.snapshot", "size": 2048}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_ctx
        
        snapshots = await backup_manager.list_snapshots("test_collection")
        
        assert len(snapshots) == 2
        assert snapshots[0]["name"] == "snapshot1.snapshot"
        assert snapshots[1]["size"] == 2048


@pytest.mark.asyncio
async def test_list_snapshots_empty(backup_manager):
    """Test listing snapshots when none exist."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": []}
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_ctx
        
        snapshots = await backup_manager.list_snapshots("test_collection")
        
        assert len(snapshots) == 0


@pytest.mark.asyncio
async def test_download_snapshot(backup_manager):
    """Test snapshot download."""
    snapshot_data = b"test snapshot data"
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    async def mock_iter_bytes(chunk_size):
        yield snapshot_data
    
    mock_response.aiter_bytes = mock_iter_bytes
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aenter__.return_value.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.return_value = mock_ctx
        
        # Download snapshot
        destination = await backup_manager.download_snapshot(
            "test_collection",
            "test.snapshot"
        )
        
        assert destination.exists()
        assert destination.read_bytes() == snapshot_data
        assert destination.parent.name == "test_collection"


@pytest.mark.asyncio
async def test_download_snapshot_custom_destination(backup_manager, temp_backup_dir):
    """Test snapshot download with custom destination."""
    snapshot_data = b"custom destination data"
    custom_dest = Path(temp_backup_dir) / "custom" / "snapshot.snapshot"
    custom_dest.parent.mkdir(parents=True, exist_ok=True)
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    async def mock_iter_bytes(chunk_size):
        yield snapshot_data
    
    mock_response.aiter_bytes = mock_iter_bytes
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__.return_value = mock_response
        mock_ctx.__aenter__.return_value.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.return_value = mock_ctx
        
        destination = await backup_manager.download_snapshot(
            "test_collection",
            "test.snapshot",
            destination=custom_dest
        )
        
        assert destination == custom_dest
        assert destination.exists()


@pytest.mark.asyncio
async def test_delete_snapshot(backup_manager):
    """Test snapshot deletion."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.delete = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_ctx
        
        result = await backup_manager.delete_snapshot("test_collection", "test.snapshot")
        
        assert result is True


@pytest.mark.asyncio
async def test_delete_snapshot_failure(backup_manager):
    """Test snapshot deletion failure."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.delete = AsyncMock(
            side_effect=Exception("Not found")
        )
        mock_client.return_value = mock_ctx
        
        result = await backup_manager.delete_snapshot("test_collection", "test.snapshot")
        
        assert result is False


@pytest.mark.asyncio
async def test_restore_snapshot(backup_manager, temp_backup_dir):
    """Test snapshot restore."""
    # Create fake snapshot file
    snapshot_path = Path(temp_backup_dir) / "test.snapshot"
    snapshot_path.write_bytes(b"snapshot data")
    
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.put = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_ctx
        
        result = await backup_manager.restore_snapshot(
            "restored_collection",
            snapshot_path,
            priority="snapshot"
        )
        
        assert result is True


@pytest.mark.asyncio
async def test_restore_snapshot_file_not_found(backup_manager):
    """Test restore with missing snapshot file."""
    nonexistent = Path("/fake/path/snapshot.snapshot")
    
    with pytest.raises(FileNotFoundError):
        await backup_manager.restore_snapshot("test_collection", nonexistent)


@pytest.mark.asyncio
async def test_backup_collection_no_download(backup_manager):
    """Test complete backup workflow without download."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "result": {"name": "backup-123.snapshot"}
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_ctx
        
        result = await backup_manager.backup_collection(
            "test_collection",
            download=False
        )
        
        assert result["snapshot_name"] == "backup-123.snapshot"
        assert result["collection_name"] == "test_collection"
        assert result["downloaded"] is False
        assert "timestamp" in result


@pytest.mark.asyncio
async def test_backup_collection_with_download(backup_manager):
    """Test complete backup workflow with download."""
    # Mock snapshot creation
    mock_create_response = MagicMock()
    mock_create_response.json.return_value = {
        "result": {"name": "backup-456.snapshot"}
    }
    mock_create_response.raise_for_status = MagicMock()
    
    # Mock snapshot download
    snapshot_data = b"downloaded backup data"
    mock_download_response = MagicMock()
    mock_download_response.raise_for_status = MagicMock()
    
    async def mock_iter_bytes(chunk_size):
        yield snapshot_data
    
    mock_download_response.aiter_bytes = mock_iter_bytes
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        
        # Setup post for create
        mock_ctx.__aenter__.return_value.post = AsyncMock(return_value=mock_create_response)
        
        # Setup stream for download
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__.return_value = mock_download_response
        mock_ctx.__aenter__.return_value.stream = MagicMock(return_value=mock_stream_ctx)
        
        mock_client.return_value = mock_ctx
        
        result = await backup_manager.backup_collection(
            "test_collection",
            download=True,
            cleanup_server=False
        )
        
        assert result["snapshot_name"] == "backup-456.snapshot"
        assert result["downloaded"] is True
        assert "local_path" in result
        assert "size" in result
        assert Path(result["local_path"]).exists()


@pytest.mark.asyncio
async def test_backup_collection_with_cleanup(backup_manager):
    """Test backup with server cleanup."""
    # Mock all required calls
    mock_create_response = MagicMock()
    mock_create_response.json.return_value = {
        "result": {"name": "backup-789.snapshot"}
    }
    mock_create_response.raise_for_status = MagicMock()
    
    snapshot_data = b"data"
    mock_download_response = MagicMock()
    mock_download_response.raise_for_status = MagicMock()
    
    async def mock_iter_bytes(chunk_size):
        yield snapshot_data
    
    mock_download_response.aiter_bytes = mock_iter_bytes
    
    mock_delete_response = MagicMock()
    mock_delete_response.raise_for_status = MagicMock()
    
    with patch("httpx.AsyncClient") as mock_client:
        mock_ctx = MagicMock()
        mock_ctx.__aenter__.return_value.post = AsyncMock(return_value=mock_create_response)
        mock_ctx.__aenter__.return_value.delete = AsyncMock(return_value=mock_delete_response)
        
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__.return_value = mock_download_response
        mock_ctx.__aenter__.return_value.stream = MagicMock(return_value=mock_stream_ctx)
        
        mock_client.return_value = mock_ctx
        
        result = await backup_manager.backup_collection(
            "test_collection",
            download=True,
            cleanup_server=True
        )
        
        assert result["server_deleted"] is True


def test_list_local_backups_empty(backup_manager):
    """Test listing local backups when directory is empty."""
    backups = backup_manager.list_local_backups()
    assert len(backups) == 0


def test_list_local_backups(backup_manager, temp_backup_dir):
    """Test listing local backup files."""
    # Create fake backup files
    collection_dir = Path(temp_backup_dir) / "test_collection"
    collection_dir.mkdir(parents=True)
    
    (collection_dir / "backup1.snapshot").write_bytes(b"data1")
    (collection_dir / "backup2.snapshot").write_bytes(b"data2")
    
    backups = backup_manager.list_local_backups("test_collection")
    
    assert len(backups) == 2
    assert all("path" in b for b in backups)
    assert all("size" in b for b in backups)
    assert all("modified" in b for b in backups)


def test_list_local_backups_all_collections(backup_manager, temp_backup_dir):
    """Test listing backups across all collections."""
    # Create multiple collections
    for i in range(3):
        collection_dir = Path(temp_backup_dir) / f"collection{i}"
        collection_dir.mkdir(parents=True)
        (collection_dir / f"backup{i}.snapshot").write_bytes(b"data")
    
    backups = backup_manager.list_local_backups()
    
    assert len(backups) == 3


def test_cleanup_old_backups(backup_manager, temp_backup_dir):
    """Test cleaning up old local backups."""
    # Create multiple backup files
    collection_dir = Path(temp_backup_dir) / "test_collection"
    collection_dir.mkdir(parents=True)
    
    import time
    for i in range(10):
        backup_file = collection_dir / f"backup{i}.snapshot"
        backup_file.write_bytes(b"data")
        # Small delay to ensure different timestamps
        time.sleep(0.01)
    
    # Keep only 3 most recent
    deleted = backup_manager.cleanup_old_backups("test_collection", keep_count=3)
    
    assert deleted == 7
    remaining = backup_manager.list_local_backups("test_collection")
    assert len(remaining) == 3


def test_cleanup_old_backups_no_deletion(backup_manager, temp_backup_dir):
    """Test cleanup when fewer backups than keep_count."""
    collection_dir = Path(temp_backup_dir) / "test_collection"
    collection_dir.mkdir(parents=True)
    
    (collection_dir / "backup1.snapshot").write_bytes(b"data")
    (collection_dir / "backup2.snapshot").write_bytes(b"data")
    
    deleted = backup_manager.cleanup_old_backups("test_collection", keep_count=5)
    
    assert deleted == 0


def test_create_backup_manager():
    """Test factory function."""
    with patch("universal_agentic_framework.caching.backup.QdrantClient"):
        manager = create_backup_manager(
            host="test-host",
            port=9999,
            backup_dir="/tmp/backups",
            timeout=120
        )
        
        assert manager.host == "test-host"
        assert manager.port == 9999
        assert manager.timeout == 120
