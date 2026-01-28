from dataclasses import dataclass
from typing import List, Dict

import yaml
import numpy as np

from llm.base import LLMClient
from llm.openai_client import OpenAILLMClient
from llm.dummy_client import DummyLLMClient
from retrieval.document_store import InMemoryVectorStore, DocumentChunk
from retrieval.ingest import RetrievalConfig, build_vector_store


@dataclass
class ModelConfig:
    provider: str
    name: str
    temperature: float
    max_tokens: int


@dataclass
class RAGConfig:
    model: ModelConfig
    retrieval: RetrievalConfig


def load_config(path: str) -> RAGConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    retrieval_raw = raw.get("retrieval", {})
    data_raw = raw.get("data", {})

    model = ModelConfig(
        provider=model_raw.get("provider", "openai"),
        name=model_raw.get("name", "gpt-4.1-mini"),
        temperature=float(model_raw.get("temperature", 0.2)),
        max_tokens=int(model_raw.get("max_tokens", 512)),
    )

    retrieval = RetrievalConfig(
        embedding_model=retrieval_raw.get("embedding_model", "all-MiniLM-L6-v2"),
        chunk_size=int(retrieval_raw.get("chunk_size", 400)),
        chunk_overlap=int(retrieval_raw.get("chunk_overlap", 50)),
        docs_path=data_raw.get("docs_path", "data/docs"),
        top_k=int(retrieval_raw.get("top_k", 4)),
    )

    return RAGConfig(model=model, retrieval=retrieval)


def build_llm_client(cfg: ModelConfig) -> LLMClient:
    if cfg.provider == "openai":
        return OpenAILLMClient(
            model_name=cfg.name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    if cfg.provider == "dummy":
        return DummyLLMClient()
    raise ValueError(f"Unsupported provider: {cfg.provider}")


def format_context(chunks_with_scores: List[tuple[DocumentChunk, float]]) -> str:
    lines: List[str] = []
    for chunk, score in chunks_with_scores:
        lines.append(f"[source: {chunk.source} | score: {score:.3f}]\n{chunk.text}\n")
    return "\n---\n".join(lines)


def build_rag_prompt(question: str, context: str) -> List[Dict[str, str]]:
    system = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions strictly "
            "based on the provided context. If the answer cannot be found "
            "in the context, say you do not know and do not hallucinate."
        ),
    }
    user = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    }
    return [system, user]


class RAGChatSession:
    """High-level RAG pipeline: retrieve relevant chunks and answer."""

    def __init__(self, llm_client: LLMClient, store: InMemoryVectorStore, top_k: int) -> None:
        self._llm_client = llm_client
        self._store = store
        self._top_k = top_k

    def answer(self, question: str, query_embedding: np.ndarray) -> str:
        results = self._store.search(query_embedding, top_k=self._top_k)
        if not results:
            messages = build_rag_prompt(
                question,
                context="No documents are available.",
            )
            return self._llm_client.chat(messages)

        context = format_context(results)
        messages = build_rag_prompt(question, context)
        return self._llm_client.chat(messages)

