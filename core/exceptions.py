class MemoryOSError(Exception):
    """Base exception for all MemoryOS errors."""
    pass

class HallucinationFirewallError(MemoryOSError):
    """Raised when an attempt to access a HYPOTHESISED memory is blocked."""
    pass

class CrossAgentAccessError(MemoryOSError):
    """Raised when an agent tries to access another agent's memory without shared scope."""
    pass

class DuplicateMemoryError(MemoryOSError):
    """Raised when attempting to ingest a memory that already exists."""
    pass

class MemoryNotFoundError(MemoryOSError):
    """Raised when attempting to retrieve a memory that doesn't exist"""
    pass

class InvalidLifecycleTransitionError(MemoryOSError):
    """Raised when there is invalid lifecycle transition"""
    pass

class StorageError(MemoryOSError):
    """Raised when a storage backend encounters an error."""
    pass

class IngestionError(MemoryOSError):
    """Raised when the ingestion pipeline fails."""
    pass

class RetrievalEngineError(MemoryOSError):
    """Base retrieval engine error."""
    pass

class EmptyQueryError(MemoryOSError):
    """Raised when query is empty."""
    pass

class RetrievalValidationError(MemoryOSError):
    """Raised when retrieval parameters are invalid."""
    pass

class EventLogCorruptionError(MemoryOSError):
    """Raised when checksum mismatch detected."""
    pass

class SequenceGapError(MemoryOSError):
    """Raised when sequence gap detected."""
    pass

class VectorStoreError(MemoryOSError):
    """Base vector store error."""
    pass

class VectorDimensionError(MemoryOSError):
    """Raised when embedding dimension mismatch occurs."""
    pass

class IndexNotInitialisedError(MemoryOSError):
    """Raised when index missing."""
    pass

class StorageOrchestrationError(MemoryOSError):
    """Base orchestration error."""
    pass

class ReplayRebuildError(MemoryOSError):
    """Raised when replay reconstruction fails."""
    pass

class InvariantViolationError(MemoryOSError):
    """Raised when system invariants are violated."""
    pass

class ChunkingError(MemoryOSError):
    """Base chunking error."""
    pass

class EmptyChunkError(MemoryOSError):
    """Raised when chunk generation fails."""
    pass

class DeduplicationError(MemoryOSError):
    """Base deduplication error."""
    pass

class InvalidContentError(MemoryOSError):
    """Raised when content invalid."""
    pass

class FirewallViolationError(MemoryOSError):
    """Raised when firewall breach occur"""
    pass
