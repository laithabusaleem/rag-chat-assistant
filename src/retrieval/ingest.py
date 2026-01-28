from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .document_store import DocumentChunk, InMemoryVectorStore


@dataclass
class RetrievalConfig:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    docs_path: str
    top_k: int


def load_text_files(docs_path: str) -> List[tuple[str, str]]:
    """Return list of (path, text) for all .txt files under docs_path."""
    root = Path(docs_path)
    if not root.exists():
        return []

    items: List[tuple[str, str]] = []
    for path in root.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        items.append((str(path), text))
    return items


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple character-based chunking with overlap."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def build_vector_store(cfg: RetrievalConfig) -> InMemoryVectorStore:
    docs = load_text_files(cfg.docs_path)
    store = InMemoryVectorStore()

    if not docs:
        return store

    model = SentenceTransformer(cfg.embedding_model)

    all_chunks: List[DocumentChunk] = []
    chunk_texts: List[str] = []
    chunk_id = 0

    for path, text in docs:
        for chunk in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap):
            normalized = chunk.strip()
            if not normalized:
                continue
            all_chunks.append(DocumentChunk(id=chunk_id, text=normalized, source=path))
            chunk_texts.append(normalized)
            chunk_id += 1

    if not all_chunks:
        return store

    embeddings = np.asarray(model.encode(chunk_texts, convert_to_numpy=True))
    store.add(embeddings, all_chunks)
    return store

