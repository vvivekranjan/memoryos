# AIMemoryOS

A powerful, modular, multi-store memory layer for conversational agents and LLM applications. 

AIMemoryOS combines vector retrieval, graph knowledge bases, and highly-durable structured storage to give your AI agents a robust, scalable long-term memory that you can plug and play directly into your own applications.

## Features

- **Multi-Store Architecture**: Coordinated state across SQLite (append-only events), DuckDB (canonical relational state), FAISS (vector similarity), and KuzuDB (graph relationships).
- **Advanced Ingestion**: End-to-end pipeline with preprocessing, PII stripping, SHA-256 deduplication, and semantic chunking.
- **Rich Retrieval**: Context assembly from multiple sources, reranking, and trace scoring.
- **Coordinated Pruning**: Safely `forget()` memories with guaranteed cleanup across all storage backends without violating relational integrity.
- **Snapshotting**: Take portable, verified SQLite backups of your system's event history instantly.
- **Hallucination Firewall**: Isolate "hypothesized" or "imagined" memories to separate tables so they don't pollute your agent's ground-truth recall.

---

## Installation

AIMemoryOS relies on powerful machine learning models under the hood. Make sure you have at least 2GB of free disk space before installing, as it will download heavy dependencies like PyTorch, FAISS, and HuggingFace Transformers.

Install AIMemoryOS via pip directly into your project:

```bash
pip install aimemoryos
```

After installation, you must download the default SpaCy NLP model used for entity extraction:

```bash
python -m spacy download en_core_web_sm
```

---

## Quickstart (Plug & Play)

The easiest way to integrate AIMemoryOS into your codebase is by using the `Memory` facade. Simply import it into your application and start saving/retrieving knowledge.

```python
import asyncio
from pathlib import Path
from aimemoryos import Memory

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
    if results:
        memory_id = results[0]["metadata"]["memory_id"]
        await memory.forget(memory_id=memory_id)
    
    # 4. Take a verifiable snapshot of the entire event history
    await memory.snapshot(output_path=Path("backups/memory_snapshot.sqlite"))

if __name__ == "__main__":
    asyncio.run(run())
```

---

## Configuration

By default, runtime databases and indexes are stored in the `.aimemoryos/` folder at the root of the installed package. You can easily override this globally if you want your data saved elsewhere:

```bash
export AIMEMORYOS_DATA_DIR="/path/to/custom/data"
```

---

## Architecture / Project Layout

For those interested in extending the SDK or contributing, the folder architecture is maintained as follows:

- `agents/`: SDK-facing components. Contains `MemoryClient`, which is the primary unified interface to the ecosystem.
- `main.py`: A lightweight, backwards-compatible wrapper around `MemoryClient` for immediate plug-and-play.
- `core/runtime.py`: The composition root. Handles dependency injection, configuration, and instantiating storage/retrieval services.
- `ingestion/`: Text preprocessing, SHA deduplication, routing, and chunking boundaries.
- `retrieval/`: Similarity search, vector reranking, and `ContextBuilder` logic.
- `storage/`: Highly durable persistence backends (`duckdb_store.py`, `faiss_store.py`, `sqlite_log.py`) and the `orchestrator.py`.
- `vector/`: Embedding generation (`sentence-transformers`) and model management.
- `graph/`: KuzuDB ontology and schema definitions.

## Contributing

We welcome contributions! To set up for local development:

```bash
git clone https://github.com/your-org/aimemoryos.git
cd aimemoryos
pip install -e .
pytest -q
```
