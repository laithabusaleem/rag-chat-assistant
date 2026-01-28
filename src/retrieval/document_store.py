from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class DocumentChunk:
    id: int
    text: str
    source: str


class InMemoryVectorStore:
    """
    Simple in-memory vector store using NumPy and cosine similarity.
    Not meant for huge datasets, but perfect for a portfolio RAG demo.
    """

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._chunks: List[DocumentChunk] = []

    @property
    def chunks(self) -> List[DocumentChunk]:
        return self._chunks

    def add(self, embeddings: np.ndarray, chunks: List[DocumentChunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks size mismatch.")

        if self._embeddings is None:
            self._embeddings = embeddings.astype(np.float32)
            self._chunks = list(chunks)
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings.astype(np.float32)])
            self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if self._embeddings is None or not self._chunks:
            return []

        # Normalize embeddings for cosine similarity.
        doc_norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        query_norm = np.linalg.norm(query_embedding) + 1e-8
        normalized_docs = self._embeddings / doc_norms
        normalized_query = query_embedding / query_norm

        scores = normalized_docs @ normalized_query
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            results.append((self._chunks[int(idx)], float(scores[int(idx)])))
        return results

