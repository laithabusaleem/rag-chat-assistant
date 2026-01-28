from abc import ABC, abstractmethod
from typing import Dict, Any, List


class LLMClient(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Send a list of chat messages and return the assistant's reply."""
        raise NotImplementedError

