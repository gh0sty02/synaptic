from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import asyncio


class MemoryManager:
    """
    Single entry point for memory operations
    Instanate once at app startup and inject wherever needed
    """

    def __init__(
        self, short_term: ShortTermMemory, long_term: LongTermMemory, llm: ChatOpenAI
    ):
        self.stm = short_term
        self.ltm = long_term
        self._llm = llm

    async def load(
        self,
        session_id: str,
        query: str,
    ) -> dict:

        stm, ltm = await asyncio.gather(self.stm.load(session_id), self.ltm.load(query))

        return {"short_term_memory": stm, "long_term_memory": ltm}

    async def append_turn(self, session_id: str, human_msg: str, ai_msg: str) -> None:
        await self.stm.append(session_id, "human", human_msg)
        await self.stm.append(session_id, "ai", ai_msg)

    async def archive_session(self, session_id: str) -> None:
        turns = await self.stm.get_all_for_archival(session_id)
        if not turns:
            return

        turn_count = len(turns) // 2

        history_text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)

        prompt = (
            "Summarise the following conversation in 2-3 sentences, "
            "capturing the main topics, decisions, and any unresolved questions.\n\n"
            f"{history_text}"
        )
        summary = (await self._llm.ainvoke(prompt)).content
        await self.ltm.save_summary(session_id, summary, turn_count)
        await self.stm.delete(session_id)
