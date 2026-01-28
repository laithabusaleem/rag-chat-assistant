import numpy as np

from src.llm.base import LLMClient
from src.retrieval.document_store import InMemoryVectorStore, DocumentChunk
from src.pipelines.rag_chat import RAGChatSession


class DummyLLM(LLMClient):
    def chat(self, messages, **kwargs) -> str:
        # Return the last user message content for easy assertions.
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        return user_messages[-1] if user_messages else ""


def test_rag_chat_session_includes_context_in_prompt():
    store = InMemoryVectorStore()

    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    chunks = [
        DocumentChunk(id=0, text="Important RAG knowledge.", source="doc1.txt"),
    ]
    store.add(embeddings, chunks)

    session = RAGChatSession(DummyLLM(), store, top_k=1)

    question = "What do the docs say?"
    query_embedding = np.array([1.0, 0.0], dtype=np.float32)

    reply = session.answer(question, query_embedding)

    # The dummy LLM echoes the user message content, which contains context + question.
    assert "Important RAG knowledge." in reply
    assert "What do the docs say?" in reply

