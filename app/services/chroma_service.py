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


class HashingEmbeddingFunction:
    """A lightweight, zero-dependency embedding function that uses word-hashing."""
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            vector = [0.0] * 128
            words = text.lower().replace("\n", " ").replace("\t", " ").split()
            for word in words:
                idx = abs(hash(word)) % 128
                vector[idx] += 1.0
            
            norm = sum(x**2 for x in vector) ** 0.5
            if norm > 0:
                vector = [x / norm for x in vector]
            embeddings.append(vector)
        return embeddings


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


def _get_client():
    global _client
    if _client is None:
        _get_collection()
    return _client


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

        # Also delete active cells collection
        client = _get_client()
        if client:
            try:
                collection_name = f"active_cells_{session_id}"
                client.delete_collection(name=collection_name)
                logger.info(f"[ChromaDB] Deleted active cells collection for session '{session_id}'")
            except Exception as e:
                logger.debug(f"[ChromaDB] Failed to delete active cells collection: {e}")

    def index_active_cells(self, session_id: str, cells: dict) -> None:
        """Indexes the individual cells of the active notebook in ChromaDB for fast retrieval."""
        client = _get_client()
        if not client:
            return
        
        try:
            collection_name = f"active_cells_{session_id}"
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=HashingEmbeddingFunction(),
                metadata={
                    "description": "Active notebook cells for semantic query matching",
                    "hnsw:space": "cosine"
                }
            )
            
            # Clear existing cells for this session first
            try:
                results = collection.get()
                if results and results.get("ids"):
                    collection.delete(ids=results["ids"])
            except Exception as e:
                logger.debug(f"[ChromaDB] Clear active cells failed: {e}")

            ids = []
            documents = []
            metadatas = []
            
            if isinstance(cells, dict):
                for cell_name, code in cells.items():
                    if code and not code.startswith("# ERROR"):
                        ids.append(f"{session_id}_{cell_name}")
                        documents.append(f"Cell: {cell_name}\nCode:\n{code}")
                        metadatas.append({"session_id": session_id, "cell_name": cell_name})
            
            if ids:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                logger.info(f"[ChromaDB] Indexed {len(ids)} active cells for session '{session_id}'")
        except Exception as e:
            logger.warning(f"[ChromaDB] index_active_cells failed: {e}")

    def get_relevant_cells(self, session_id: str, query: str, n_results: int = 3) -> dict:
        """Queries the active cells collection and returns the top matching cells."""
        client = _get_client()
        if not client:
            return {}
            
        try:
            collection_name = f"active_cells_{session_id}"
            collection = client.get_collection(
                name=collection_name,
                embedding_function=HashingEmbeddingFunction()
            )
            count = collection.count()
            if count == 0:
                return {}
                
            results = collection.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )
            relevant_cells = {}
            if results and results.get("metadatas") and len(results["metadatas"]) > 0:
                for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
                    cell_name = meta.get("cell_name")
                    code = doc.split("Code:\n", 1)[1] if "Code:\n" in doc else doc
                    relevant_cells[cell_name] = code
            return relevant_cells
        except Exception as e:
            logger.warning(f"[ChromaDB] get_relevant_cells failed: {e}")
            return {}
