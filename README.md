# AIMemory

A powerful, modular, multi-store memory layer for conversational agents and LLM applications. 

AIMemory combines vector retrieval, graph knowledge bases, and highly-durable structured storage to give your AI agents a robust, scalable long-term memory.

## Features

- **Multi-Store Architecture**: Coordinated state across SQLite (append-only events), DuckDB (canonical relational state), FAISS (vector similarity), and KuzuDB (graph relationships).
- **Advanced Ingestion**: End-to-end pipeline with preprocessing, PII stripping, SHA-256 deduplication, and semantic chunking.
- **Rich Retrieval**: Context assembly from multiple sources, reranking, and trace scoring.
- **Coordinated Pruning**: Safely `forget()` memories with guaranteed cleanup across all storage backends without violating relational integrity.
- **Snapshotting**: Take portable, verified SQLite backups of your system's event history instantly.
- **Hallucination Firewall**: Isolate "hypothesized" or "imagined" memories to separate tables so they don't pollute your agent's ground-truth recall.

---

## Installation

This project uses Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) (or standard `pip`).

```powershell
# 1. Clone the repository
git clone https://github.com/your-org/aimemory.git
cd aimemory

# 2. Setup your virtual environment
uv venv
.venv\Scripts\activate

# 3. Install dependencies
uv pip install -r pyproject.toml # or standard pip install
```

---

## Quickstart (Plug & Play)

The easiest way to integrate AIMemory into your codebase is by using the `MemoryClient` (available via the `Memory` facade in `main.py`).

```python
import asyncio
from main import Memory

async def run():
    # Initialize the high-level memory facade
    memory = Memory()

    # 1. Ingest information into the memory pipeline
    print("Ingesting memory...")
    await memory.save(
        document_id="doc_001",
        content="My favorite programming language is Python and I love pizza."
    )

    # 2. Retrieve relevant context based on semantic similarity
    print("Retrieving context...")
    results = memory.retrieve(
        query="What foods do I like?",
        top_k=5,
        score_threshold=0.2
    )
    
    # Results include the content, metadata, and relevance score
    for r in results:
        print(f"[{r['score']:.2f}] {r['metadata']['content']}")

    # 3. Forget a memory across all stores (DuckDB, FAISS, Graph, SQLite)
    memory_id = results[0]["metadata"]["memory_id"]
    await memory.forget(memory_id=memory_id)
    
    # 4. Take a verifiable snapshot of the entire event history
    from pathlib import Path
    await memory.snapshot(output_path=Path("backups/memory_snapshot.sqlite"))

if __name__ == "__main__":
    asyncio.run(run())
```

### Try the Interactive CLI App

We include a built-in CLI app that wires up AIMemory directly to an LLM so you can talk to an agent that actually remembers!

```powershell
# First, configure OpenRouter or your LLM API keys:
$env:OPENROUTER_API_KEY="sk-or-..."

# Run the app
python app.py
```
*(If no OpenRouter key is provided, the CLI will gracefully degrade into a retrieval-only mode).*

---

## Configuration

By default, runtime databases and indexes are stored in `data/`. You can override this globally:

```powershell
$env:MEMORYOS_DATA_DIR="custom-data"
```

For LLM-powered context generation in `app.py`, configure:

```powershell
$env:OPENROUTER_API_KEY="..."
$env:OPENROUTER_MODEL="openai/gpt-4o-mini"
```

---

## Project Layout

- `agents/`: SDK-facing components. Contains `MemoryClient`, which is the primary unified interface to the ecosystem.
- `main.py`: A lightweight, backwards-compatible wrapper around `MemoryClient` for immediate plug-and-play.
- `app.py`: Interactive CLI chatbot demo.
- `core/runtime.py`: The composition root. Handles dependency injection, configuration, and instantiating storage/retrieval services.
- `ingestion/`: Text preprocessing, SHA deduplication, routing, and chunking boundaries.
- `retrieval/`: Similarity search, vector reranking, and `ContextBuilder` logic.
- `storage/`: Highly durable persistence backends (`duckdb_store.py`, `faiss_store.py`, `sqlite_log.py`) and the `orchestrator.py`.
- `vector/`: Embedding generation (`sentence-transformers`) and model management.
- `graph/`: KuzuDB ontology and schema definitions.
- `tests/`: End-to-end and unit regression test suites.

## Contributing

```powershell
# Run the test suite
pytest -q
```
