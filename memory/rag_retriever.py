from __future__ import annotations

import logging

from memory.long_term import LongTermMemory

logger = logging.getLogger(__name__)

_UE5_COLLECTION = "ue5_knowledge"


class RAGRetriever:
    """Retrieves relevant UE5 API docs and patterns at runtime for agent context."""

    def __init__(self, persist_dir: str = ".chromadb") -> None:
        self._mem = LongTermMemory(persist_dir)
        self._persist_dir = persist_dir

    def _ensure_collection(self) -> None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection = client.get_or_create_collection(_UE5_COLLECTION)
        except ImportError:
            raise RuntimeError("chromadb is not installed. Run: pip install chromadb")

    def retrieve(self, query: str, n_results: int = 3) -> str:
        try:
            self._ensure_collection()
            results = self._collection.query(query_texts=[query], n_results=n_results)
            docs = results.get("documents", [[]])[0]
            if not docs:
                return ""
            return "\n\n---\n\n".join(docs)
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return ""

    def index_document(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self._ensure_collection()
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )
        logger.debug("Indexed document: %s", doc_id)
