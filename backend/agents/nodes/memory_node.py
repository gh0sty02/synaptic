import asyncio
from ..graph import SynapticState
from ...memory.manager import MemoryManager

memory_manager: MemoryManager = None


async def memory_node(state: SynapticState) -> dict:
    """
    Reads Short and long term memory in parallel
    returns state slices that langgraph merges back
    """
    result = await memory_manager.load(
        session_id=state["session_id"], query=state["query"]
    )

    return result
