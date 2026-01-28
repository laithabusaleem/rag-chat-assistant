import numpy as np

from src.retrieval.document_store import InMemoryVectorStore, DocumentChunk


def test_in_memory_vector_store_returns_top_results():
    store = InMemoryVectorStore()

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    chunks = [
        DocumentChunk(id=0, text="about cats", source="doc1.txt"),
        DocumentChunk(id=1, text="about dogs", source="doc2.txt"),
    ]

    store.add(embeddings, chunks)

    query_embedding = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query_embedding, top_k=1)

    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.text == "about cats"
    assert score > 0.9

