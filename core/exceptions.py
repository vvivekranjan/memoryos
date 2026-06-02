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

class StorageError(MemoryOSError):
    """Raised when a storage backend encounters an error."""
    pass

class IngestionError(MemoryOSError):
    """Raised when the ingestion pipeline fails."""
    pass
