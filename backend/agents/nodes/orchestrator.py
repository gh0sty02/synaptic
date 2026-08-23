from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import get_client, observe

from agents.state import SynapticState
from constants import BEHAVIORAL_GUARDRAILS
from context.engineer import assemble
from llm import main_llm
from memory.manager import MemoryManager
from retrieval.query_condensation import condense_query

from . import rag_agent as _rag_agent
from .rag_agent import extract_citations

ORCHESTRATOR_PROMPT = """
You are a synthesis assistant. The user's query requires both factual knowledge and conversation history.
You have been given:
1. Relevant documents retrieved from the knowledge base
2. Relevant conversation history from prior sessions

Combine both sources to produce a single, coherent, grounded answer.
If conversation history contradicts the documents, prefer the documents but acknowledge the discrepancy.
Only cite a document with [Source: <title>] when that document actually supports the claim. If you must
answer from conversation history or general knowledge instead of a retrieved document, say so explicitly
rather than attaching a citation to it.
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
        context = (
            f"Retrieved documents:\n{docs_context}"
            if docs_context.strip()
            else "Retrieved documents: none found for this question."
        )
        if short_term:
            context += f"\n\nRecent conversation:\n{short_term}"
        if long_term:
            context += f"\n\nOlder session summaries:\n{long_term}"

        response = await main_llm.ainvoke(
            [
                SystemMessage(content=ORCHESTRATOR_PROMPT),
                HumanMessage(content=f"{context}\n\nUser question: {query}"),
            ],
            config={"callbacks": callbacks, "tags": ["final_answer"]},
        )
        return response.content


_orchestrator = Orchestrator()


@observe(name="orchestrator_node")
async def orchestrator_node(state: SynapticState) -> dict[str, Any]:
    """
    For multi-intent queries: loads memory, condenses the query against it to
    resolve referential follow-ups, retrieves chunks with the condensed query,
    then calls the LLM to merge memory and retrieval into one final answer.
    """
    assert memory_manager is not None, "memory_manager not initialised"

    langfuse = get_client()

    memory_result = await memory_manager.load(
        session_id=state["session_id"], query=state["query"]
    )

    pre_bundle = assemble(
        "orchestrator_node",
        short_term_memory=memory_result.get("short_term_memory", []),
        long_term_memory=memory_result.get("long_term_memory", []),
    )
    retrieval_query = await condense_query(state["query"], pre_bundle.memory_context)

    docs = await _rag_agent.retriever.ainvoke(retrieval_query)

    bundle = assemble(
        "orchestrator_node",
        short_term_memory=memory_result.get("short_term_memory", []),
        long_term_memory=memory_result.get("long_term_memory", []),
        retrieved_chunks=docs,
    )

    langfuse.update_current_span(
        metadata={
            "retrieval_query": retrieval_query,
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
