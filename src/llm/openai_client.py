import os
from typing import Dict, Any, List

from openai import OpenAI

from .base import LLMClient


class OpenAILLMClient(LLMClient):
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before running the RAG assistant with provider=openai."
            )

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            temperature=kwargs.get("temperature", self._temperature),
            max_completion_tokens=kwargs.get("max_tokens", self._max_tokens),
        )
        choice = response.choices[0]
        return choice.message.content or ""

