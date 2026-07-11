import os
import logging
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class FallbackMemory:
    """In-process fallback used only when all remote/local options fail."""
    def __init__(self):
        self.store = {}
        logger.info("Initializing Fallback In-Memory Storage (data will NOT persist)")

    def add(self, text: str, user_id: str, metadata: dict = None):
        if user_id not in self.store:
            self.store[user_id] = []
        self.store[user_id].append({"text": text, "metadata": metadata})
        logger.info(f"[Memory] Added: {text}")
        return {"status": "success"}

    def get_all(self, user_id: str):
        return self.store.get(user_id, [])


class CloudMemoryClient:
    """
    Thin wrapper around Mem0 MemoryClient (cloud mode).
    Lazily initializes to avoid blocking startup.
    """
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = None
        self._fallback = None

    def _get_client(self):
        """Lazily initialize the client."""
        if self._client is None and self._fallback is None:
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self._api_key)
                logger.info("Initialized Mem0 Cloud client (MemoryClient).")
            except Exception as e:
                logger.warning(f"Mem0 Cloud init failed: {e}. Falling back to in-memory.")
                self._fallback = FallbackMemory()
        return self._client if self._client is not None else self._fallback

    def add(self, text: str, user_id: str, metadata: dict = None):
        try:
            client = self._get_client()
            result = client.add(text, user_id=user_id, metadata=metadata or {})
            logger.info(f"[Mem0 Cloud] Added memory for user '{user_id}': {text[:80]}")
            return result
        except Exception as e:
            logger.warning(f"[Mem0 Cloud] Failed to add memory: {e}")
            return {"status": "error", "error": str(e)}

    def get_all(self, user_id: str):
        try:
            client = self._get_client()
            # If fallback is active, call its get_all
            if isinstance(client, FallbackMemory):
                return client.get_all(user_id)

            response = client.get_all(filters={"user_id": user_id})
            if isinstance(response, dict):
                items = response.get("results", [])
            elif isinstance(response, list):
                items = response
            else:
                items = []
            return [{"text": m.get("memory", str(m)), "metadata": m.get("metadata", {})} for m in items]
        except Exception as e:
            logger.warning(f"[Mem0 Cloud] Failed to fetch memories: {e}")
            return []


def _build_mem0_client():
    # --- Option 1: Mem0 Cloud (MemoryClient) using MEM0_API_KEY ---
    # Return wrapper instantly without calling the blocking network request in constructor
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if mem0_api_key:
        logger.info("Mem0 Cloud client wrapper prepared (will lazy-load on first use).")
        return CloudMemoryClient(mem0_api_key)

    # --- Option 2: Local Mem0 with ChromaDB (requires OPENAI_API_KEY for embeddings) ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from mem0 import Memory
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {"path": "./.mem0_chromadb"},
                }
            }
            client = Memory.from_config(config)
            logger.info("Initialized Mem0 with local ChromaDB vector store.")
            return client
        except Exception as e:
            logger.warning(f"Local Mem0 + ChromaDB init failed: {e}. Falling back to in-memory store.")

    # --- Option 3: Pure in-memory fallback (no persistence) ---
    logger.warning(
        "No MEM0_API_KEY or OPENAI_API_KEY found. Using in-memory fallback — memories will NOT persist between sessions. "
        "To fix: set MEM0_API_KEY in your .env file."
    )
    return FallbackMemory()


mem0_client = _build_mem0_client()
