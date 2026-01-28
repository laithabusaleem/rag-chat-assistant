import os
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.rag_chat import (  # type: ignore  # noqa: E402
    load_config,
    build_llm_client,
    RAGChatSession,
)
from retrieval.ingest import RetrievalConfig, build_vector_store  # type: ignore  # noqa: E402


def main() -> None:
    console = Console()

    config_path = PROJECT_ROOT / "config" / "rag_chat.yaml"
    if not config_path.exists():
        console.print(f"[red]Config file not found:[/red] {config_path}")
        raise SystemExit(1)

    cfg = load_config(str(config_path))

    if cfg.model.provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        console.print(
            "[red]OPENAI_API_KEY is not set.[/red] "
            "Set it in your environment or switch the config provider to 'dummy'."
        )
        raise SystemExit(1)

    console.print("[bold green]Building vector store from documents...[/bold green]")
    store = build_vector_store(cfg.retrieval)

    if not store.chunks:
        console.print(
            "[yellow]No documents found or all documents are empty.[/yellow] "
            "Add .txt files under the configured docs path and run again."
        )
        raise SystemExit(1)

    console.print(
        f"[green]Indexed {len(store.chunks)} chunks "
        f"from documents under '{cfg.retrieval.docs_path}'.[/green]\n"
    )

    # Embedding model for user queries.
    query_model = SentenceTransformer(cfg.retrieval.embedding_model)

    llm_client = build_llm_client(cfg.model)
    session = RAGChatSession(llm_client, store, top_k=cfg.retrieval.top_k)

    console.print("[bold cyan]RAG Chat Assistant[/bold cyan]")
    console.print("Ask questions about your documents. Type 'exit' or 'quit' to end.\n")

    while True:
        question = Prompt.ask("[bold cyan]You[/bold cyan]")
        if question.strip().lower() in {"exit", "quit"}:
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        query_embedding = np.asarray(
            query_model.encode([question], convert_to_numpy=True)[0]
        )
        answer = session.answer(question, query_embedding)
        console.print(f"[bold magenta]Assistant[/bold magenta]: {answer}\n")


if __name__ == "__main__":
    main()

