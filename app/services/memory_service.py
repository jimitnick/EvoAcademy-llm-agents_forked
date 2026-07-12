"""
MemoryService — Wraps ChromaMemoryClient for long-term user preferences across sessions.
It stores student preferences and profile statement vector embeddings in the local database.
"""
import logging
from app.core.memory import mem0_client

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Wraps ChromaMemoryClient for user preference management.
    Processes user queries and saves resulting preferences as local vector embeddings.
    """

    def get_preferences(self, session_id: str) -> str:
        """
        Retrieves all stored preferences for a user (keyed by session_id).
        Returns a formatted string ready to be injected into an LLM prompt.
        """
        try:
            items = mem0_client.get_all(session_id)
            if not items:
                return ""

            lines = [m.get("text", str(m)) for m in items if m.get("text")]
            preference_text = "\n".join(f"- {line}" for line in lines)
            return f"Student preferences (apply automatically):\n{preference_text}"
        except Exception as e:
            logger.warning(f"[MemoryService] Failed to retrieve preferences: {e}")
            return ""

    def update_preferences(self, session_id: str, prompt: str, summary: str) -> None:
        """
        Sends the prompt and summary to ChromaMemoryClient so it can store
        any new preferences expressed by the user.
        """
        try:
            interaction = (
                f"User asked: {prompt}\n"
                f"Result: {summary}"
            )
            mem0_client.add(interaction, user_id=session_id)
            logger.info(f"[MemoryService] Updated preferences for session '{session_id}'")
        except Exception as e:
            logger.warning(f"[MemoryService] Failed to update preferences: {e}")

    def add_explicit_preference(self, session_id: str, preference: str) -> None:
        """Directly store a user preference statement."""
        try:
            mem0_client.add(preference, user_id=session_id)
        except Exception as e:
            logger.warning(f"[MemoryService] Failed to add explicit preference: {e}")
