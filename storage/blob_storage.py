"""
MediSuite Agent — Azure Blob Storage Client
Wrapper for uploading, downloading, and listing blobs.
"""

import logging
from pathlib import Path

from azure.storage.blob import BlobServiceClient, ContentSettings

from config import settings

logger = logging.getLogger(__name__)


class BlobStorageClient:
    """Thin wrapper around Azure Blob Storage."""

    def __init__(self) -> None:
        if not settings.validate_blob_storage():
            logger.warning(
                "Azure Blob Storage not configured — "
                "blob operations will be logged but not executed."
            )
            self._available = False
            return

        self._service_client = BlobServiceClient.from_connection_string(
            settings.storage_connection_string
        )
        self._container_client = self._service_client.get_container_client(
            settings.blob_container_name
        )
        # Create container if it doesn't exist
        try:
            self._container_client.get_container_properties()
        except Exception:
            self._container_client.create_container()
            logger.info("Created blob container '%s'.", settings.blob_container_name)

        self._available = True

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    def upload_file(
        self,
        file_path: str | Path,
        blob_name: str | None = None,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload a local file to blob storage.

        Args:
            file_path: Path to the local file.
            blob_name: Target blob name; defaults to the filename.
            content_type: MIME type of the file.

        Returns:
            The blob URL.
        """
        file_path = Path(file_path)
        blob_name = blob_name or file_path.name

        if not self._available:
            logger.info("[BLOB-MOCK] upload_file(%s → %s)", file_path, blob_name)
            return f"https://mock.blob.core.windows.net/{settings.blob_container_name}/{blob_name}"

        blob_client = self._container_client.get_blob_client(blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )

        url = blob_client.url
        logger.info("Uploaded '%s' → %s", file_path.name, url)
        return url

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def download_file(self, blob_name: str, dest_path: str | Path) -> Path:
        """Download a blob to a local file.

        Args:
            blob_name: Name of the blob.
            dest_path: Destination local path.

        Returns:
            Path to the downloaded file.
        """
        dest_path = Path(dest_path)

        if not self._available:
            logger.info("[BLOB-MOCK] download_file(%s → %s)", blob_name, dest_path)
            return dest_path

        blob_client = self._container_client.get_blob_client(blob_name)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)

        logger.info("Downloaded '%s' → %s", blob_name, dest_path)
        return dest_path

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    def list_blobs(self, prefix: str | None = None) -> list[str]:
        """List blob names in the container.

        Args:
            prefix: Optional prefix filter.

        Returns:
            List of blob names.
        """
        if not self._available:
            logger.info("[BLOB-MOCK] list_blobs(prefix=%s)", prefix)
            return []

        blobs = self._container_client.list_blobs(name_starts_with=prefix)
        names = [b.name for b in blobs]
        logger.info("Listed %d blobs (prefix=%s)", len(names), prefix)
        return names
