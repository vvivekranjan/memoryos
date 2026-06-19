import asyncio
from pathlib import Path
from memoryos.agents.memory_client import MemoryClient

async def main():
    print("Testing forget and snapshot...")
    client = MemoryClient()
    
    # 1. Ingest
    res = await client.ingest(content="My favorite color is green and I have 73 apples.", agent_id="test_agent")
    mem_ids = res.memory_ids
    print(f"Ingested memories: {mem_ids}")
    
    if mem_ids:
        mem_id = mem_ids[0]
        
        # 2. Forget
        print(f"Forgetting {mem_id}...")
        await client.forget(memory_id=str(mem_id))
        print("Forget successful!")
        
    # 3. Snapshot
    out_path = Path("data/snapshot_test.sqlite")
    if out_path.exists():
        out_path.unlink()
        
    print(f"Taking snapshot to {out_path}...")
    await client.snapshot(output_path=out_path)
    print("Snapshot successful!")
    print(f"Snapshot exists: {out_path.exists()}")

if __name__ == "__main__":
    asyncio.run(main())
