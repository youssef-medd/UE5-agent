#!/usr/bin/env python3
"""Index all UE5 knowledge documents into ChromaDB for RAG retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.rag_retriever import RAGRetriever

_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

_DOCS = [
    ("ue5_python_api", "ue5_python_api.md"),
    ("blueprint_patterns", "blueprint_patterns.md"),
    ("unreal_classes", "unreal_classes.md"),
]


def main() -> None:
    retriever = RAGRetriever()
    print("Indexing UE5 knowledge base...")

    for doc_id, filename in _DOCS:
        path = _KNOWLEDGE_DIR / filename
        if not path.exists():
            print(f"  SKIP: {filename} not found")
            continue
        text = path.read_text(encoding="utf-8")
        retriever.index_document(doc_id, text, metadata={"source": filename})
        print(f"  OK: {filename} ({len(text)} chars)")

    jsonl_path = _KNOWLEDGE_DIR / "task_examples.jsonl"
    if jsonl_path.exists():
        import json
        for i, line in enumerate(jsonl_path.read_text().splitlines()):
            if line.strip():
                obj = json.loads(line)
                retriever.index_document(
                    f"example_{i}",
                    obj.get("prompt", ""),
                    metadata={"type": "example"},
                )
        print(f"  OK: task_examples.jsonl ({i + 1} examples)")

    print("Done. Knowledge base is ready.")


if __name__ == "__main__":
    main()
