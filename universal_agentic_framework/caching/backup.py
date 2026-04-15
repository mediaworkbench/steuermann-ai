"""Backup and restore functionality for Qdrant vector database collections.

Provides snapshot creation, download, and restore capabilities for disaster recovery
and data migration scenarios.
"""

import logging
import os
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import SnapshotDescription
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class QdrantBackupManager:
    """Manages Qdrant collection snapshots and restores.
    
    Provides high-level interface for:
    - Creating collection snapshots
    - Listing available snapshots
    - Downloading snapshots to local storage
    - Restoring collections from snapshots
    - Managing snapshot lifecycle
    
    Snapshots are stored in Qdrant's snapshot directory and can be
    downloaded for off-site backup or collection migration.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        backup_dir: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize backup manager.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            backup_dir: Local directory for downloaded snapshots (default: ./backups)
            timeout: HTTP request timeout in seconds
        """
        if not HAS_QDRANT:
            raise ImportError("qdrant-client required. Install: pip install qdrant-client")
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        
        # Setup backup directory
        self.backup_dir = Path(backup_dir or "./backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized QdrantBackupManager "
            f"(host={host}:{port}, backup_dir={self.backup_dir})"
        )
    
    async def create_snapshot(self, collection_name: str) -> str:
        """Create a snapshot of a collection.
        
        Args:
            collection_name: Name of collection to snapshot
            
        Returns:
            Snapshot name/filename
            
        Raises:
            Exception: If snapshot creation fails
        """
        try:
            # Use Qdrant REST API for snapshot creation
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection_name}/snapshots"
                )
                response.raise_for_status()
                
                result = response.json()
                snapshot_name = result.get("result", {}).get("name")
                
                if not snapshot_name:
                    raise ValueError(f"No snapshot name in response: {result}")
                
                logger.info(
                    f"Created snapshot for collection '{collection_name}': {snapshot_name}"
                )
                
                return snapshot_name
                
        except Exception as e:
            logger.error(
                f"Failed to create snapshot for collection '{collection_name}': {e}"
            )
            raise
    
    async def list_snapshots(self, collection_name: str) -> List[Dict[str, Any]]:
        """List all snapshots for a collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            List of snapshot metadata dictionaries with keys:
            - name: Snapshot filename
            - size: Size in bytes
            - creation_time: ISO 8601 timestamp (if available)
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/collections/{collection_name}/snapshots"
                )
                response.raise_for_status()
                
                result = response.json()
                snapshots = result.get("result", [])
                
                logger.info(
                    f"Found {len(snapshots)} snapshots for collection '{collection_name}'"
                )
                
                return snapshots
                
        except Exception as e:
            logger.error(
                f"Failed to list snapshots for collection '{collection_name}': {e}"
            )
            return []
    
    async def download_snapshot(
        self,
        collection_name: str,
        snapshot_name: str,
        destination: Optional[Path] = None
    ) -> Path:
        """Download a snapshot to local storage.
        
        Args:
            collection_name: Name of collection
            snapshot_name: Name of snapshot to download
            destination: Custom destination path (default: backup_dir/collection/snapshot)
            
        Returns:
            Path to downloaded snapshot file
            
        Raises:
            Exception: If download fails
        """
        # Determine destination path
        if destination is None:
            collection_dir = self.backup_dir / collection_name
            collection_dir.mkdir(parents=True, exist_ok=True)
            destination = collection_dir / snapshot_name
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Stream download to handle large files
                async with client.stream(
                    "GET",
                    f"{self.base_url}/collections/{collection_name}/snapshots/{snapshot_name}"
                ) as response:
                    response.raise_for_status()
                    
                    # Write to file
                    with open(destination, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
                
                file_size = destination.stat().st_size
                logger.info(
                    f"Downloaded snapshot '{snapshot_name}' "
                    f"for collection '{collection_name}' "
                    f"to {destination} ({file_size:,} bytes)"
                )
                
                return destination
                
        except Exception as e:
            logger.error(
                f"Failed to download snapshot '{snapshot_name}' "
                f"for collection '{collection_name}': {e}"
            )
            # Clean up partial download
            if destination.exists():
                destination.unlink()
            raise
    
    async def delete_snapshot(self, collection_name: str, snapshot_name: str) -> bool:
        """Delete a snapshot from Qdrant server.
        
        Args:
            collection_name: Name of collection
            snapshot_name: Name of snapshot to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/collections/{collection_name}/snapshots/{snapshot_name}"
                )
                response.raise_for_status()
                
                logger.info(
                    f"Deleted snapshot '{snapshot_name}' "
                    f"for collection '{collection_name}'"
                )
                
                return True
                
        except Exception as e:
            logger.error(
                f"Failed to delete snapshot '{snapshot_name}' "
                f"for collection '{collection_name}': {e}"
            )
            return False
    
    async def restore_snapshot(
        self,
        collection_name: str,
        snapshot_path: Path,
        priority: str = "snapshot"
    ) -> bool:
        """Restore a collection from a snapshot file.
        
        Args:
            collection_name: Name of collection to restore (can be new name)
            snapshot_path: Path to local snapshot file
            priority: Conflict resolution strategy:
                - "snapshot": Snapshot data takes priority (replaces existing)
                - "replica": Existing data takes priority (keeps existing)
                
        Returns:
            True if restore successful
            
        Raises:
            Exception: If restore fails
        """
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
        
        try:
            # Upload snapshot file to Qdrant
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(snapshot_path, "rb") as f:
                    files = {"snapshot": (snapshot_path.name, f, "application/octet-stream")}
                    
                    response = await client.put(
                        f"{self.base_url}/collections/{collection_name}/snapshots/upload",
                        files=files,
                        params={"priority": priority}
                    )
                    response.raise_for_status()
                
                logger.info(
                    f"Restored collection '{collection_name}' "
                    f"from snapshot {snapshot_path} (priority={priority})"
                )
                
                return True
                
        except Exception as e:
            logger.error(
                f"Failed to restore collection '{collection_name}' "
                f"from snapshot {snapshot_path}: {e}"
            )
            raise
    
    async def backup_collection(
        self,
        collection_name: str,
        download: bool = True,
        cleanup_server: bool = False
    ) -> Dict[str, Any]:
        """Complete backup workflow for a collection.
        
        Creates snapshot, optionally downloads it locally, and optionally
        removes snapshot from server to free space.
        
        Args:
            collection_name: Name of collection to backup
            download: Whether to download snapshot locally
            cleanup_server: Whether to delete snapshot from server after download
            
        Returns:
            Dictionary with backup metadata:
            - snapshot_name: Name of created snapshot
            - local_path: Path to downloaded file (if download=True)
            - size: Size in bytes (if downloaded)
            - timestamp: Backup timestamp
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # Create snapshot
            snapshot_name = await self.create_snapshot(collection_name)
            
            result = {
                "snapshot_name": snapshot_name,
                "collection_name": collection_name,
                "timestamp": timestamp,
                "downloaded": False,
            }
            
            # Download if requested
            if download:
                local_path = await self.download_snapshot(collection_name, snapshot_name)
                result["local_path"] = str(local_path)
                result["size"] = local_path.stat().st_size
                result["downloaded"] = True
                
                # Cleanup server snapshot if requested
                if cleanup_server:
                    await self.delete_snapshot(collection_name, snapshot_name)
                    result["server_deleted"] = True
            
            logger.info(
                f"Backup completed for collection '{collection_name}': {result}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backup failed for collection '{collection_name}': {e}")
            raise
    
    def list_local_backups(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List locally stored backup files.
        
        Args:
            collection_name: Filter by collection name (default: all)
            
        Returns:
            List of backup file metadata
        """
        backups = []
        
        search_dir = self.backup_dir / collection_name if collection_name else self.backup_dir
        
        if not search_dir.exists():
            return backups
        
        # Find all snapshot files
        for filepath in search_dir.rglob("*.snapshot"):
            stat = filepath.stat()
            backups.append({
                "path": str(filepath),
                "collection": filepath.parent.name if collection_name is None else collection_name,
                "filename": filepath.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        return sorted(backups, key=lambda x: x["modified"], reverse=True)
    
    def cleanup_old_backups(
        self,
        collection_name: str,
        keep_count: int = 5
    ) -> int:
        """Remove old local backup files, keeping only the most recent.
        
        Args:
            collection_name: Collection to clean up
            keep_count: Number of most recent backups to keep
            
        Returns:
            Number of backups deleted
        """
        backups = self.list_local_backups(collection_name)
        
        if len(backups) <= keep_count:
            return 0
        
        # Delete oldest backups
        to_delete = backups[keep_count:]
        deleted = 0
        
        for backup in to_delete:
            try:
                Path(backup["path"]).unlink()
                deleted += 1
                logger.info(f"Deleted old backup: {backup['path']}")
            except Exception as e:
                logger.error(f"Failed to delete backup {backup['path']}: {e}")
        
        return deleted


def create_backup_manager(
    host: str = "localhost",
    port: int = 6333,
    backup_dir: Optional[str] = None,
    timeout: int = 60
) -> QdrantBackupManager:
    """Factory function to create QdrantBackupManager instance.
    
    Args:
        host: Qdrant server host
        port: Qdrant server port
        backup_dir: Local backup directory
        timeout: HTTP timeout in seconds
        
    Returns:
        Configured QdrantBackupManager instance
    """
    return QdrantBackupManager(
        host=host,
        port=port,
        backup_dir=backup_dir,
        timeout=timeout
    )
