import pytest
import asyncio
from uuid import uuid4
from datetime import datetime, timezone
import os
import shutil

from storage.duckdb_store import DuckDBStore
from storage.orchestrator import StorageOrchestrator
from memory.episodic import EpisodicMemory
from memory.models import SpeakerRoleEnum, ModalityEnum, LifecycleStateEnum, ProvenanceEnum

@pytest.fixture
def test_db_path(tmp_path):
    path = tmp_path / "test_fw.duckdb"
    yield path

def test_hallucination_firewall_routing(test_db_path):
    duck = DuckDBStore(test_db_path)
    duck.initialise()
    
    mem_id = uuid4()
    sess_id = uuid4()
    
    mem = EpisodicMemory(
        memory_id=mem_id,
        agent_id="test_agent",
        content="This is a test memory",
        sha256="0" * 64,
        modality=ModalityEnum.TEXT,
        lifecycle_state=LifecycleStateEnum.ACTIVE,
        created_at=datetime.now(timezone.utc),
        last_accessed_at=datetime.now(timezone.utc),
        access_count=0,
        decay_anchor=datetime.now(timezone.utc),
        decay_multiplier=1.0,
        importance_score=0.5,
        salience_score=0.5,
        provenance=ProvenanceEnum.HYPOTHESISED,
        provenance_confidence=0.9,
        session_id=sess_id,
        turn_index=1,
        speaker_role=SpeakerRoleEnum.USER,
        is_system_message=False
    )
    
    class MockLog:
        def log_ingestion(self, *args, **kwargs): pass
        def log_retrieval(self, *args, **kwargs): pass
        def log_feedback(self, *args, **kwargs): pass
        def log_lifecycle_transition(self, *args, **kwargs): pass

    class MockFaiss:
        def add_embedding(self, *args, **kwargs): pass

    orch = StorageOrchestrator(
        duckdb_store=duck,
        faiss_store=MockFaiss(),
        sqlite_log=MockLog()
    )
    
    # Run ingestion
    res = asyncio.run(orch.ingest_memory(mem, [0.1, 0.2, 0.3]))
    assert res.success is True
    assert res.error == "Routed to isolated store"
    
    # Verify DuckDB isolated_memories contains the memory
    with duck._connect() as conn:
        row = conn.execute("SELECT memory_id FROM isolated_memories WHERE memory_id = ?", [str(mem_id)]).fetchone()
        assert row is not None
        assert row[0] == str(mem_id)
        
        # Verify it is NOT in standard memories
        row_standard = conn.execute("SELECT memory_id FROM memories WHERE memory_id = ?", [str(mem_id)]).fetchone()
        assert row_standard is None

