"""
MemoryService — Wraps Mem0 for long-term user preferences across sessions.
Mem0 ONLY stores user preferences, NOT notebook versions or history.

Examples of what gets stored:
  - Preferred EA library (DEAP, PyGAD, pymoo)
  - Preferred mutation/crossover operators
  - Whether to include markdown explanations
  - Preferred visualization library (matplotlib, plotly)
  - Verbosity preferences
  - Coding conventions

These preferences are injected into LLM prompts to personalize notebook generation.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MemoryService:
    """
    Wraps Mem0 MemoryClient for user preference management.
    Falls back gracefully if MEM0_API_KEY is not configured or network fails.
    """

    def __init__(self):
        self._api_key = os.getenv("MEM0_API_KEY")
        self._client = None
        self._initialized = False

    def _get_client(self):
        """Lazily initialize the Mem0 client with error handling."""
        if not self._initialized:
            self._initialized = True
            if not self._api_key:
                logger.warning("[Mem0] MEM0_API_KEY not set. Preferences will not persist.")
                return None
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self._api_key)
                logger.info("[Mem0] MemoryClient initialized successfully.")
            except Exception as e:
                logger.warning(f"[Mem0] Init failed: {e}. Preferences will not persist.")
        return self._client

    def get_preferences(self, session_id: str) -> str:
        """
        Retrieves all stored preferences for a user (keyed by session_id).
        Returns a formatted string ready to be injected into an LLM prompt.
        """
        client = self._get_client()
        if not client:
            return ""
        try:
            response = client.get_all(filters={"user_id": session_id})
            if isinstance(response, dict):
                items = response.get("results", [])
            else:
                items = response if isinstance(response, list) else []

            if not items:
                return ""

            lines = [m.get("memory", str(m)) for m in items if m.get("memory")]
            preference_text = "\n".join(f"- {line}" for line in lines)
            return f"Student preferences (apply automatically):\n{preference_text}"
        except Exception as e:
            logger.warning(f"[Mem0] Failed to retrieve preferences: {e}")
            return ""

    def update_preferences(self, session_id: str, prompt: str, summary: str) -> None:
        """
        Sends the prompt and summary to Mem0 so it can extract and store
        any new preferences expressed by the user.
        Mem0 automatically deduplicates and merges similar preferences.
        """
        client = self._get_client()
        if not client:
            return
        try:
            interaction = (
                f"User asked: {prompt}\n"
                f"Result: {summary}"
            )
            client.add(interaction, user_id=session_id)
            logger.info(f"[Mem0] Updated preferences for session '{session_id}'")
        except Exception as e:
            logger.warning(f"[Mem0] Failed to update preferences: {e}")

    def add_explicit_preference(self, session_id: str, preference: str) -> None:
        """Directly store a user preference statement."""
        client = self._get_client()
        if not client:
            return
        try:
            client.add(preference, user_id=session_id)
        except Exception as e:
            logger.warning(f"[Mem0] Failed to add explicit preference: {e}")
