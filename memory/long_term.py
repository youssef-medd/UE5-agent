from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "ue5_agent_memory"


class LongTermMemory:
    """ChromaDB-backed vector store for persistent agent knowledge."""

    def __init__(self, persist_dir: str = ".chromadb") -> None:
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = self._client.get_or_create_collection(_COLLECTION_NAME)
            logger.debug("ChromaDB initialized at %s", self._persist_dir)
        except ImportError:
            raise RuntimeError(
                "chromadb is not installed. Run: pip install chromadb"
            )

    def store(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self._ensure_client()
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def query(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        self._ensure_client()
        results = self._collection.query(query_texts=[text], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return [{"id": i, "text": d, "metadata": m} for i, d, m in zip(ids, docs, metas)]

    def delete(self, doc_id: str) -> None:
        self._ensure_client()
        self._collection.delete(ids=[doc_id])
