import asyncio
from typing import Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from llm import utility_llm
from agents.graph import SynapticState
from memory.manager import MemoryManager
from . import rag_agent as _rag_agent
from .rag_agent import format_memory, extract_citations
from constants import BEHAVIORAL_GUARDRAILS
from langfuse import observe

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
    async def merge(
        self,
        query: str,
        docs_context: str,
        short_term: str,
        long_term: str,
        callbacks: list | None = None,
    ) -> str:
        context = f"Retrieved documents:\n{docs_context}"
        if short_term:
            context += f"\n\nRecent conversation:\n{short_term}"
        if long_term:
            context += f"\n\nOlder session summaries:\n{long_term}"

        response = await utility_llm.ainvoke(
            [
                SystemMessage(content=ORCHESTRATOR_PROMPT),
                HumanMessage(content=f"{context}\n\nUser question: {query}"),
            ],
            config={"callbacks": callbacks},
        )
        return response.content


_orchestrator = Orchestrator()


@observe(name="orchestrator_node")
async def orchestrator_node(state: SynapticState) -> dict[str, Any]:
    """
    For multi-intent queries: loads memory and retrieves chunks in parallel,
    then calls the LLM to merge both into a single coherent final answer.
    """
    assert memory_manager is not None, "memory_manager not initialised"

    memory_result, docs = await asyncio.gather(
        memory_manager.load(session_id=state["session_id"], query=state["query"]),
        _rag_agent.retriever.ainvoke(state["query"]),
    )

    short_term = format_memory(memory_result.get("short_term_memory", []))
    long_term = _format_long_term(memory_result.get("long_term_memory", []))
    docs_context = _format_docs(docs)
    callbacks = state.get("callbacks", [])

    answer = await _orchestrator.merge(
        query=state["query"],
        docs_context=docs_context,
        short_term=short_term,
        long_term=long_term,
        callbacks=callbacks,
    )

    return {
        **memory_result,
        "retrieved_chunks": [{"content": d.page_content, **d.metadata} for d in docs],
        "citations": extract_citations(docs),
        "final_answer": answer,
    }
