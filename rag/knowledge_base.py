"""
MediSuite Agent — RAG Knowledge Base
Wraps Azure Cognitive Search for document indexing and semantic retrieval.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
)

from config import settings

logger = logging.getLogger(__name__)


class RAGKnowledgeBase:
    """Azure Cognitive Search–backed RAG Knowledge Base."""

    # Supported document types for categorisation
    DOC_TYPES = [
        "patient_data",
        "clinical_notes",
        "icd10_code",
        "cpt4_code",
        "claim",
        "validation_result",
        "triage_decision",
        "document_metadata",
    ]

    def __init__(self) -> None:
        if not settings.validate_search():
            logger.warning(
                "Azure Cognitive Search not configured — "
                "RAG operations will be logged but not executed."
            )
            self._available = False
            return

        credential = AzureKeyCredential(settings.search_api_key)
        self._index_client = SearchIndexClient(
            endpoint=settings.search_endpoint, credential=credential
        )
        self._search_client = SearchClient(
            endpoint=settings.search_endpoint,
            index_name=settings.search_index_name,
            credential=credential,
        )
        self._available = True
        self._ensure_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def _ensure_index(self) -> None:
        """Create the search index if it does not already exist."""
        existing = [idx.name for idx in self._index_client.list_indexes()]
        if settings.search_index_name in existing:
            logger.info("Search index '%s' already exists.", settings.search_index_name)
            return

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(
                name="doc_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(
                name="timestamp",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
            ),
        ]
        index = SearchIndex(name=settings.search_index_name, fields=fields)
        self._index_client.create_index(index)
        logger.info("Created search index '%s'.", settings.search_index_name)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert_document(
        self,
        doc_type: str,
        data: dict[str, Any],
        doc_id: str | None = None,
    ) -> str:
        """Upsert a document into the knowledge base.

        Args:
            doc_type: One of DOC_TYPES.
            data: The data to store (serialised to JSON in the content field).
            doc_id: Optional explicit ID; auto-generated if omitted.

        Returns:
            The document ID.
        """
        doc_id = doc_id or str(uuid.uuid4())
        document = {
            "id": doc_id,
            "doc_type": doc_type,
            "content": json.dumps(data, default=str),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": json.dumps(
                {"doc_type": doc_type, "keys": list(data.keys())}, default=str
            ),
        }

        if not self._available:
            logger.info(
                "[RAG-MOCK] insert_document(type=%s, id=%s): %s",
                doc_type,
                doc_id,
                json.dumps(data, default=str)[:200],
            )
            return doc_id

        self._search_client.upload_documents(documents=[document])
        logger.info("Inserted document id=%s type=%s", doc_id, doc_type)
        return doc_id

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------
    def retrieve_documents(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the knowledge base and return top-k results.

        Args:
            query: Free-text search query.
            doc_type: Optional filter by document type.
            top_k: Number of results to return.

        Returns:
            List of matching documents (parsed content + metadata).
        """
        if not self._available:
            logger.info(
                "[RAG-MOCK] retrieve_documents(query='%s', type=%s, top_k=%d)",
                query[:80],
                doc_type,
                top_k,
            )
            return []

        filter_expr = f"doc_type eq '{doc_type}'" if doc_type else None
        results = self._search_client.search(
            search_text=query,
            filter=filter_expr,
            top=top_k,
        )

        documents: list[dict[str, Any]] = []
        for result in results:
            try:
                content = json.loads(result["content"])
            except (json.JSONDecodeError, KeyError):
                content = result.get("content", "")
            documents.append(
                {
                    "id": result["id"],
                    "doc_type": result.get("doc_type", ""),
                    "content": content,
                    "timestamp": result.get("timestamp", ""),
                    "score": result.get("@search.score", 0),
                }
            )
        logger.info("Retrieved %d documents for query='%s'", len(documents), query[:80])
        return documents
