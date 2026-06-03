from __future__ import annotations

from typing import Any

from graph.ontology import KuzuDBStore


async def bfs_traversal(store: KuzuDBStore, seeds: list[str], max_hops: int = 2) -> list[dict[str, Any]]:
    """Async BFS traversal over the graph store."""

    return await store.bfs_traversal_async(seeds=seeds, max_hops=max_hops)
