"""
MediSuite Agent — Configuration Module
Loads environment variables and exposes a Settings dataclass.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)


@dataclass
class Settings:
    """Centralised settings loaded from environment variables."""

    # Azure AI Foundry
    project_endpoint: str = field(
        default_factory=lambda: os.getenv("PROJECT_ENDPOINT", "")
    )
    model_deployment_name: str = field(
        default_factory=lambda: os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4o")
    )

    # Azure Blob Storage
    storage_connection_string: str = field(
        default_factory=lambda: os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    )
    blob_container_name: str = field(
        default_factory=lambda: os.getenv("AZURE_BLOB_CONTAINER_NAME", "medisuite-claims")
    )

    # Azure Cognitive Search
    search_endpoint: str = field(
        default_factory=lambda: os.getenv("AZURE_SEARCH_ENDPOINT", "")
    )
    search_api_key: str = field(
        default_factory=lambda: os.getenv("AZURE_SEARCH_API_KEY", "")
    )
    search_index_name: str = field(
        default_factory=lambda: os.getenv("AZURE_SEARCH_INDEX_NAME", "medisuite-knowledge-base")
    )

    # Local paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent
    )
    output_dir: Path = field(default=None)  # type: ignore[assignment]
    data_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = self.project_root / "output"
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_azure_ai(self) -> bool:
        """Return True if Azure AI Foundry settings are configured."""
        return bool(self.project_endpoint)

    def validate_blob_storage(self) -> bool:
        """Return True if Blob Storage settings are configured."""
        return bool(self.storage_connection_string)

    def validate_search(self) -> bool:
        """Return True if Cognitive Search settings are configured."""
        return bool(self.search_endpoint and self.search_api_key)


# Singleton instance
settings = Settings()
