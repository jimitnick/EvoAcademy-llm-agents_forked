import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ChromaMemoryClient:
    """
    A persistent local memory client backed by our local ChromaDB client and HashingEmbeddingFunction.
    Requires no external API keys or cloud connections.
    """
    def __init__(self):
        logger.info("Initializing ChromaMemoryClient (persistent local vector store)")
        self.collection_name = "student_memories"

    def _get_collection(self):
        from app.services.chroma_service import _get_client, HashingEmbeddingFunction
        client = _get_client()
        if not client:
            return None
        try:
            return client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=HashingEmbeddingFunction(),
                metadata={"description": "Long-term student preferences and profile", "hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"[ChromaMemory] Failed to get collection: {e}")
            return None

    def add(self, text: str, user_id: str, metadata: dict = None):
        collection = self._get_collection()
        if not collection:
            logger.warning("[ChromaMemory] Add failed: no collection")
            return {"status": "error", "error": "No database collection"}
        
        try:
            import uuid
            doc_id = f"mem_{uuid.uuid4().hex[:12]}"
            meta = {**(metadata or {}), "user_id": user_id, "text": text}
            collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta]
            )
            logger.info(f"[ChromaMemory] Stored local persistent memory: {text[:80]}")
            return {"status": "success"}
        except Exception as e:
            logger.warning(f"[ChromaMemory] Failed to add memory: {e}")
            return {"status": "error", "error": str(e)}

    def get_all(self, user_id: str):
        collection = self._get_collection()
        if not collection:
            return []
        
        try:
            results = collection.get(where={"user_id": user_id})
            memories = []
            if results and results.get("metadatas"):
                for meta in results["metadatas"]:
                    memories.append({
                        "text": meta.get("text", ""),
                        "metadata": meta
                    })
            return memories
        except Exception as e:
            logger.warning(f"[ChromaMemory] Failed to fetch memories: {e}")
            return []


def _build_mem0_client():
    # Use our local persistent ChromaMemoryClient by default
    return ChromaMemoryClient()


mem0_client = _build_mem0_client()
