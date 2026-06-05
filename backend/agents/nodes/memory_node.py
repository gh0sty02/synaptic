import asyncio
from agents.graph import SynapticState
from memory.manager import MemoryManager
from langfuse import observe

memory_manager: MemoryManager = None


@observe(name="memory_node")
async def memory_node(state: SynapticState) -> dict:
    return await memory_manager.load(
        session_id=state["session_id"], query=state["query"]
    )
