"""
ChromaService — Semantic indexing and search of notebook version summaries.
ChromaDB stores ONLY lightweight text (prompt + summary + keywords).
It never stores full notebook content or .ipynb files.

Collection: notebook_versions
Document: "{prompt}\n{summary}\n{keywords}"
Metadata: {version_id, session_id, version_number, operation_type, summary}
"""
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ".chroma_version_store"
)

COLLECTION_NAME = "notebook_versions"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        try:
            import chromadb
            _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": "Semantic index of notebook version summaries for natural language search",
                    "hnsw:space": "cosine"
                }
            )
            logger.info(f"[ChromaDB] Collection '{COLLECTION_NAME}' ready at {CHROMA_DB_PATH}")
        except Exception as e:
            logger.warning(f"[ChromaDB] Init failed: {e}")
            _collection = None
    return _collection


class ChromaService:
    """
    Manages ChromaDB collections for semantic version search.
    Uses ChromaDB's built-in sentence-transformer embeddings (no OPENAI_API_KEY needed).
    """

    def index_version(
        self,
        version_id: str,
        session_id: str,
        version_number: int,
        prompt: str,
        summary: str,
        cells_modified: List[str],
        operation_type: str,
    ) -> bool:
        """
        Indexes a notebook version in ChromaDB.
        Document = prompt + summary + cells_modified keywords (not full code).
        """
        collection = _get_collection()
        if not collection:
            return False

        try:
            keywords = ", ".join(cells_modified) if cells_modified else "all cells"
            document = f"{prompt}\n{summary}\nModified: {keywords}\nOperation: {operation_type}"

            metadata = {
                "version_id": version_id,
                "session_id": session_id,
                "version_number": version_number,
                "operation_type": operation_type,
                "summary": summary[:500],
                "cells_modified": ",".join(cells_modified) if cells_modified else "",
            }

            collection.upsert(
                ids=[version_id],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"[ChromaDB] Indexed version {version_number} for session '{session_id}'")
            return True
        except Exception as e:
            logger.warning(f"[ChromaDB] Failed to index version: {e}")
            return False

    def semantic_search(
        self,
        session_id: str,
        query: str,
        n_results: int = 5
    ) -> List[dict]:
        """
        Natural language search over version summaries.
        Examples:
          "restore the version before tournament selection was added"
          "find where elitism was introduced"
          "show me when mutation probability was fixed"
        Returns ranked list of matching versions.
        """
        collection = _get_collection()
        if not collection:
            return []

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, collection.count())),
                where={"session_id": session_id} if session_id else None
            )
            items = []
            if results and results.get("metadatas"):
                for meta, doc, dist in zip(
                    results["metadatas"][0],
                    results["documents"][0],
                    results["distances"][0]
                ):
                    items.append({
                        "version_id": meta.get("version_id"),
                        "version_number": meta.get("version_number"),
                        "summary": meta.get("summary"),
                        "operation_type": meta.get("operation_type"),
                        "cells_modified": meta.get("cells_modified", "").split(","),
                        "relevance_score": round(1 - dist, 4),
                    })
            return items
        except Exception as e:
            logger.warning(f"[ChromaDB] Semantic search failed: {e}")
            return []

    def delete_session_versions(self, session_id: str) -> None:
        """Removes all ChromaDB entries for a session (called on /generate reset)."""
        collection = _get_collection()
        if not collection:
            return
        try:
            collection.delete(where={"session_id": session_id})
            logger.info(f"[ChromaDB] Deleted all versions for session '{session_id}'")
        except Exception as e:
            logger.warning(f"[ChromaDB] Failed to delete session versions: {e}")
