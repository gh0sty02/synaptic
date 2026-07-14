from langfuse import observe

from agents.graph import SynapticState
from memory.manager import MemoryManager

memory_manager: MemoryManager = None


@observe(name="writer_node")
async def writer_node(state: SynapticState) -> dict:
    human_msg = state["query"]
    ai_msg = state.get("final_answer", "")
    if ai_msg:
        await memory_manager.append_turn(
            session_id=state["session_id"], human_msg=human_msg, ai_msg=ai_msg
        )
    return {}
