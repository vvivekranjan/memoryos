# AIMemory

A modular memory layer for conversational agents with:

- ingestion and chunking
- vector retrieval with FAISS
- canonical memory state in DuckDB
- append-only event history in SQLite
- optional graph relationships through Kuzu

## Project Layout

- `app.py`: interactive CLI demo wired to OpenRouter
- `main.py`: lightweight `Memory` facade for save/retrieve flows
- `core/runtime.py`: composition root for storage and retrieval services
- `agents/`: SDK-facing helpers such as `MemoryClient`
- `ingestion/`: preprocessing, deduplication, routing, and chunking
- `retrieval/`: retrieval coordination, reranking, and context assembly
- `storage/`: persistence backends and orchestration
- `vector/`: embedding and index management
- `tests/`: regression coverage for core utility and retrieval flows

## Runtime Design

The repository now uses a centralized runtime builder in `core/runtime.py`.

That gives us a cleaner path to scale because:

- storage paths are configured in one place
- entrypoints no longer duplicate backend wiring
- `MemoryClient` can accept injected dependencies for tests and future service containers
- adding alternative stores or hosted backends can happen behind one composition layer

By default, runtime data is stored under `data/`. You can override that with:

```powershell
$env:MEMORYOS_DATA_DIR="custom-data"
```

## Local Usage

Install dependencies, then run:

```powershell
pytest -q
python app.py
```

To enable LLM responses in the CLI, configure:

```text
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Without OpenRouter configured, the CLI still works in retrieval-only mode.

## Next Scale-Up Steps

- add a real configuration object for model, index, and storage tuning
- implement coordinated delete/forget semantics across DuckDB, FAISS, and graph storage
- add service-level integration tests for ingestion and retrieval with fixture-backed data directories
- split transport/demo code from the core library if this becomes an API service
