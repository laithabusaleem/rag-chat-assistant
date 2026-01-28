from typing import Dict, Any, List

from .base import LLMClient


class DummyLLMClient(LLMClient):
    """
    Offline LLM client for local development and demos.

    This does NOT call any external API. It simply echoes the last user
    message with a small prefix and a note that retrieval was used.
    """

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        last_user = user_messages[-1] if user_messages else ""
        return f"[dummy-rag-model] Answer based on retrieved docs about: {last_user}"

