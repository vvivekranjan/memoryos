from __future__ import annotations

import logging
from typing import Optional

from aimemoryos.core.exceptions import FirewallViolationError, HallucinationFirewallError
from aimemoryos.memory.models import BaseMemory, ProvenanceEnum
from aimemoryos.storage.duckdb_store import DuckDBStore

logger = logging.getLogger(__name__)

_FIREWALL_ISOLATED: frozenset[str] = frozenset({
    ProvenanceEnum.IMAGINED.value,
    ProvenanceEnum.HYPOTHESISED.value,
})

class HallucinationFirewall:
    """
    Storage-layer routing enforcement.

    Intercepts memories with IMAGINED or HYPOTHESISED provenance
    and routes them to an isolated table.
    """

    def __init__(self, duckdb_store: DuckDBStore):
        self.duckdb = duckdb_store

    def enforce_ingestion_routing(self, memory: BaseMemory) -> bool:
        """
        Enforce routing at the ingestion layer.
        Returns True if the memory was routed to the isolated store,
        False if it should proceed through the standard pipeline.
        """
        provenance = getattr(memory, "provenance", None)
        if provenance is None:
            return False

        prov_value = (
            provenance.value if hasattr(provenance, "value") else str(provenance)
        )

        if prov_value in _FIREWALL_ISOLATED:
            logger.info(
                "HallucinationFirewall: Routing memory %s to isolated store due to provenance %s",
                memory.memory_id,
                prov_value,
            )
            # Route to isolated table
            self.duckdb.insert_isolated_memory(memory)
            return True

        return False

    def enforce_retrieval_guard(self, memory: BaseMemory, agent_clearance: list[str] | None = None) -> None:
        """
        Enforce retrieval guard at the storage layer.
        Raises HallucinationFirewallError if the agent doesn't have clearance
        to access isolated memories.
        """
        provenance = getattr(memory, "provenance", None)
        if provenance is None:
            return

        prov_value = (
            provenance.value if hasattr(provenance, "value") else str(provenance)
        )

        if prov_value in _FIREWALL_ISOLATED:
            agent_clearance = agent_clearance or []
            if prov_value not in agent_clearance:
                logger.warning(
                    "HallucinationFirewall: Blocked unauthorized retrieval of memory %s (provenance=%s)",
                    memory.memory_id,
                    prov_value,
                )
                raise HallucinationFirewallError(
                    f"Blocked attempt to retrieve memory {memory.memory_id} with provenance {prov_value}. "
                    f"Agent clearance required."
                )
