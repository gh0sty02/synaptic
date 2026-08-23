import asyncio
import logging

from langfuse import observe

from agents.state import SynapticState
from memory.manager import MemoryManager

logger = logging.getLogger(__name__)

memory_manager: MemoryManager = None


@observe(name="writer_node")
async def writer_node(state: SynapticState) -> dict:
    human_msg = state["query"]
    ai_msg = state.get("final_answer", "")
    if ai_msg:
        session_id = state["session_id"]
        try:
            await asyncio.wait_for(
                memory_manager.append_turn(
                    session_id=session_id, human_msg=human_msg, ai_msg=ai_msg
                ),
                timeout=10,
            )
        except TimeoutError:
            logger.error(
                "writer_node: append_turn timed out after 10s for session=%s",
                session_id,
            )
            raise
    return {}
