import asyncio
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import SecretStr
import os
from dotenv import load_dotenv
from agents.graph import SynapticState
from memory.manager import MemoryManager
from .rag_agent import retriever, format_memory, extract_citations
from constants import BEHAVIORAL_GUARDRAILS

load_dotenv()

LLM_MODEL = os.environ["LLM_MODEL"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]

ORCHESTRATOR_PROMPT = """
You are a synthesis assistant. The user's query requires both factual knowledge and conversation history.
You have been given:
1. Relevant documents retrieved from the knowledge base
2. Relevant conversation history from prior sessions

Combine both sources to produce a single, coherent, grounded answer.
If conversation history contradicts the documents, prefer the documents but acknowledge the discrepancy.
""" + BEHAVIORAL_GUARDRAILS

memory_manager: Optional[MemoryManager] = None


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('title', 'Unknown')}]\n{d.page_content}"
        for d in docs
    )


def _format_long_term(long_term: list[dict[str, Any]]) -> str:
    return "\n\n".join(m.get("summary", "") for m in long_term if m.get("summary"))


class Orchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=SecretStr(LLM_API_KEY),
            temperature=1.0,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def merge(
        self,
        query: str,
        docs_context: str,
        short_term: str,
        long_term: str,
    ) -> str:
        context = f"Retrieved documents:\n{docs_context}"
        if short_term:
            context += f"\n\nRecent conversation:\n{short_term}"
        if long_term:
            context += f"\n\nOlder session summaries:\n{long_term}"

        response = await self.llm.ainvoke([
            SystemMessage(content=ORCHESTRATOR_PROMPT),
            HumanMessage(content=f"{context}\n\nUser question: {query}"),
        ])
        return response.content


_orchestrator = Orchestrator()


async def orchestrator_node(state: SynapticState) -> dict[str, Any]:
    """
    For multi-intent queries: loads memory and retrieves chunks in parallel,
    then calls the LLM to merge both into a single coherent final answer.
    """
    assert memory_manager is not None, "memory_manager not initialised"

    memory_result, docs = await asyncio.gather(
        memory_manager.load(session_id=state["session_id"], query=state["query"]),
        retriever.ainvoke(state["query"]),
    )

    short_term = format_memory(memory_result.get("short_term_memory", []))
    long_term = _format_long_term(memory_result.get("long_term_memory", []))
    docs_context = _format_docs(docs)

    answer = await _orchestrator.merge(
        query=state["query"],
        docs_context=docs_context,
        short_term=short_term,
        long_term=long_term,
    )

    return {
        **memory_result,
        "retrieved_chunks": [{"content": d.page_content, **d.metadata} for d in docs],
        "citations": extract_citations(docs),
        "final_answer": answer,
    }
