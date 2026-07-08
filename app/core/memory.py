import os
import logging
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Fallback local in-memory store if Chroma DB is not available
class FallbackMemory:
    def __init__(self):
        self.store = {}
        logger.info("Initializing Fallback In-Memory Storage")

    def add(self, text: str, user_id: str, metadata: dict = None):
        if user_id not in self.store:
            self.store[user_id] = []
        self.store[user_id].append({"text": text, "metadata": metadata})
        logger.info(f"[Memory] Added: {text}")
        return {"status": "success"}

    def get_all(self, user_id: str):
        return self.store.get(user_id, [])

mem0_client = None
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if api_key:
    try:
        # chroma vector database configurations
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {"path": "./.mem0_chromadb"}
            }
        }
        # OpenRouter fallback routes
        if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            config["llm"] = {
                "provider": "openai",
                "config": {
                    "model": "meta-llama/llama-3-70b-instruct",
                    "api_key": os.getenv("OPENROUTER_API_KEY"),
                    "openai_api_base": "https://openrouter.ai/api/v1"
                }
            }
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "api_key": os.getenv("OPENROUTER_API_KEY"),
                    "openai_api_base": "https://openrouter.ai/api/v1"
                }
            }
        mem0_client = Memory.from_config(config)
        logger.info("Initialized Mem0 vector database")
    except Exception as e:
        logger.warning(f"Mem0 init failure, using in-memory store: {e}")
        mem0_client = FallbackMemory()
else:
    mem0_client = FallbackMemory()
