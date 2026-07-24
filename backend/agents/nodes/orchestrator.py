import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, observe

from agents.state import SynapticState
from constants import BEHAVIORAL_GUARDRAILS
from context.engineer import assemble
from llm import utility_llm
from memory.manager import MemoryManager

from . import rag_agent as _rag_agent
from .rag_agent import extract_citations

ORCHESTRATOR_PROMPT = """
You are a synthesis assistant. The user's query requires both factual knowledge and conversation history.
You have been given:
1. Relevant documents retrieved from the knowledge base
2. Relevant conversation history from prior sessions

Combine both sources to produce a single, coherent, grounded answer.
If conversation history contradicts the documents, prefer the documents but acknowledge the discrepancy.
""" + BEHAVIORAL_GUARDRAILS

memory_manager: MemoryManager | None = None


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

    langfuse = get_client()

    memory_result, docs = await asyncio.gather(
        memory_manager.load(session_id=state["session_id"], query=state["query"]),
        _rag_agent.retriever.ainvoke(state["query"]),
    )

    bundle = assemble(
        "orchestrator_node",
        short_term_memory=memory_result.get("short_term_memory", []),
        long_term_memory=memory_result.get("long_term_memory", []),
        retrieved_chunks=docs,
    )

    langfuse.update_current_span(
        metadata={
            "budget_decision": bundle.decision,
            "token_count": bundle.token_count,
            "budget_exceeded": state.get("budget_exceeded"),
        }
    )

    callbacks = state.get("callbacks", [])

    answer = await _orchestrator.merge(
        query=state["query"],
        docs_context=bundle.chunks_context,
        short_term=bundle.short_term_context,
        long_term=bundle.long_term_context,
        callbacks=callbacks,
    )

    return {
        **memory_result,
        "retrieved_chunks": [{"content": d.page_content, **d.metadata} for d in docs],
        "citations": extract_citations(docs),
        "final_answer": answer,
        "token_counts": bundle.token_count,
        "total_tokens": bundle.total_tokens,
        "budget_exceeded": bundle.budget_exceeded,
        "decision": bundle.decision,
    }
