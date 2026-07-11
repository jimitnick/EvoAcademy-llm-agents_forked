import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Path for the local ChromaDB persistent store
CHROMA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ".chroma_version_store"
)

# Collection names
COLLECTION_VERSION_HISTORY = "notebook_version_history"
COLLECTION_USER_INTENTS = "user_intents"

_chroma_client = None
_version_collection = None
_intent_collection = None


def _get_chroma_client():
    """Lazy-initializes and returns the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            logger.info(f"ChromaDB client initialized at: {CHROMA_DB_PATH}")
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")
            _chroma_client = None
    return _chroma_client


def _get_version_collection():
    """Returns the ChromaDB collection for notebook versions."""
    global _version_collection
    if _version_collection is None:
        client = _get_chroma_client()
        if client:
            try:
                _version_collection = client.get_or_create_collection(
                    name=COLLECTION_VERSION_HISTORY,
                    metadata={
                        "description": "Stores notebook DEAP cell versions with semantic search capability",
                        "hnsw:space": "cosine"
                    }
                )
                logger.info(f"ChromaDB collection '{COLLECTION_VERSION_HISTORY}' ready.")
            except Exception as e:
                logger.warning(f"Failed to create ChromaDB version collection: {e}")
    return _version_collection


def _get_intent_collection():
    """Returns the ChromaDB collection for user intents."""
    global _intent_collection
    if _intent_collection is None:
        client = _get_chroma_client()
        if client:
            try:
                _intent_collection = client.get_or_create_collection(
                    name=COLLECTION_USER_INTENTS,
                    metadata={
                        "description": "Stores user modification intents for semantic retrieval",
                        "hnsw:space": "cosine"
                    }
                )
                logger.info(f"ChromaDB collection '{COLLECTION_USER_INTENTS}' ready.")
            except Exception as e:
                logger.warning(f"Failed to create ChromaDB intent collection: {e}")
    return _intent_collection


def store_version(
    doc_id: str,
    session_id: str,
    version_number: int,
    user_intent: str,
    cells: Dict[str, str],
    compiled_script: str,
    status: str,
    created_at: str,
    error_message: Optional[str] = None
) -> bool:
    """
    Stores a notebook version in ChromaDB.
    The document content is the compiled_script (or cells JSON) for semantic search.
    All key fields are stored as metadata for filtering.
    """
    collection = _get_version_collection()
    if not collection:
        return False

    try:
        # Use compiled script as the searchable document (or cells JSON if no script)
        document_content = compiled_script if compiled_script else json.dumps(cells)
        # Truncate very large scripts to ChromaDB's document size limits
        document_content = document_content[:10000]

        metadata = {
            "session_id": session_id,
            "version_number": version_number,
            "user_intent": user_intent[:500],  # ChromaDB metadata must be str/int/float
            "status": status,
            "created_at": created_at,
            "cell_names": ",".join(cells.keys()),
            "error_message": (error_message or "")[:500]
        }

        collection.upsert(
            ids=[doc_id],
            documents=[document_content],
            metadatas=[metadata]
        )
        logger.info(f"[ChromaDB] Stored version {version_number} for session '{session_id}' (status={status})")
        return True
    except Exception as e:
        logger.warning(f"[ChromaDB] Failed to store version: {e}")
        return False


def store_user_intent(
    doc_id: str,
    session_id: str,
    version_number: int,
    user_intent: str,
    cells_modified: List[str],
    created_at: str
) -> bool:
    """
    Stores the user's modification intent in a separate ChromaDB collection
    for semantic retrieval (e.g. 'find versions where crossover was changed').
    """
    collection = _get_intent_collection()
    if not collection:
        return False

    try:
        metadata = {
            "session_id": session_id,
            "version_number": version_number,
            "cells_modified": ",".join(cells_modified),
            "created_at": created_at
        }
        collection.upsert(
            ids=[doc_id],
            documents=[user_intent],
            metadatas=[metadata]
        )
        logger.info(f"[ChromaDB] Stored intent for session '{session_id}' v{version_number}: '{user_intent[:60]}'")
        return True
    except Exception as e:
        logger.warning(f"[ChromaDB] Failed to store intent: {e}")
        return False


def semantic_search_versions(session_id: str, query: str, n_results: int = 5) -> List[dict]:
    """
    Perform semantic search over notebook versions for a given session.
    Returns the most relevant versions matching the query.
    """
    collection = _get_version_collection()
    if not collection:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"session_id": session_id}
        )
        items = []
        if results and results.get("metadatas"):
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0]
            ):
                items.append({
                    "version_number": meta.get("version_number"),
                    "user_intent": meta.get("user_intent"),
                    "status": meta.get("status"),
                    "created_at": meta.get("created_at"),
                    "relevance_score": round(1 - dist, 4),
                    "snippet": doc[:200]
                })
        return items
    except Exception as e:
        logger.warning(f"[ChromaDB] Semantic search failed: {e}")
        return []


def get_all_versions_from_chroma(session_id: str) -> List[dict]:
    """Returns all versions for a session from ChromaDB."""
    collection = _get_version_collection()
    if not collection:
        return []

    try:
        results = collection.get(where={"session_id": session_id})
        items = []
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                items.append(meta)
        return sorted(items, key=lambda x: x.get("version_number", 0))
    except Exception as e:
        logger.warning(f"[ChromaDB] Failed to get all versions: {e}")
        return []
