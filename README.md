## RAG Chat Assistant

This repository contains a **Retrieval-Augmented Generation (RAG) chat assistant**:

- **Document-aware**: answer questions based on your own `.txt` documents.
- **Config-driven**: YAML config for models, retrieval, and runtime behavior.
- **Separation of concerns**: ingestion, retrieval, and chat pipeline are clearly separated.
- **Tested**: includes unit tests for the retriever and RAG pipeline.
- **Offline-friendly**: supports a dummy LLM for local development without API calls.

---

## Features

- Load plain-text documents from a folder and split them into chunks.
- Embed chunks and store them in an in-memory vector store.
- Retrieve the most relevant chunks for a user question.
- Build a combined RAG prompt and call an LLM (OpenAI by default).
- Optional dummy mode for development and testing.

---

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

---

## Configuration

- Main config: `config/rag_chat.yaml`

It controls:

- **Model**: provider (`openai` or `dummy`), name, temperature, max tokens.
- **Retrieval**: embedding model name, chunk size, chunk overlap, top-k.
- **Data**: path to the documents folder.

---

## Usage

### 1. Prepare your documents

Place one or more `.txt` files in the `data/docs/` directory (you can create it if it does not exist).

Example:

- `data/docs/introduction.txt`
- `data/docs/design_notes.txt`

### 2. With a real OpenAI key

Set your API key (PowerShell example):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Ensure the config uses `provider: openai`:

```yaml
# config/rag_chat.yaml
model:
  provider: openai
```

Then run:

```bash
python examples/run_rag_cli.py
```

You can now ask questions about your documents.

### 3. Offline / dummy mode

Switch the config to `provider: dummy` to use the dummy LLM (no API calls):

```yaml
model:
  provider: dummy
```

Run the same CLI command:

```bash
python examples/run_rag_cli.py
```

---

## Running tests

```bash
pytest
```

