import operator
import os
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class SynapticState(TypedDict):
    # Core
    session_id: str
    query: str
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Routing — set by triage_node, read by route_query
    intent: str  # 'rag' | 'memory' | 'multi' | 'blocked'
    active_agents: list[str]

    # Metadata filters extracted by triage from natural language
    metadata_filters: dict[str, Any]

    # Context — populated at API layer before graph.ainvoke()
    system_prompt: str
    short_term_memory: list[
        dict[str, Any]
    ]  # last N turns from Redis (session:{id}:turns)
    long_term_memory: list[dict[str, Any]]  # retrieved session summaries from pgvector
    retrieved_chunks: list[dict[str, Any]]  # RAG output per turn
    tool_results: list[dict[str, Any]]  # tool call outputs
    # Query reformulation — written by rag_agent; equals query when memory_context is empty
    condensed_query: str
    use_hyde: bool

    # Token budget (phase 3 — initialised here, acted on later)
    token_counts: dict[str, int]
    total_tokens: int
    budget_exceeded: bool

    # Output
    final_answer: str
    citations: list[dict[str, Any]]
    agent_scratchpad: str

    # Observability
    trace_id: str
    latency_ms: dict[str, int]
    langfuse_callbacks: list[dict[str, Any]]  # captures all LLM calls for Langfuse


def route_query(state: SynapticState) -> str:
    """
    Conditional edge function — reads intent set by triage_node.
    Returns a node name that matches the keys in add_conditional_edges.
    """
    intent = state.get("intent", "rag")
    if intent == "memory":
        return "memory_agent"
    if intent == "multi":
        return "orchestrator"
    return "rag_agent"


# Node imports after SynapticState is defined to avoid circular import failure
from .nodes.memory_node import memory_node  # noqa: E402
from .nodes.orchestrator import orchestrator_node  # noqa: E402
from .nodes.rag_agent import rag_agent  # noqa: E402
from .nodes.triage import triage_node  # noqa: E402
from .nodes.writer_node import writer_node  # noqa: E402

# ── Graph definition ──────────────────────────────────────────────────────────

graph = StateGraph(SynapticState)

graph.add_node("triage", triage_node)
graph.add_node("rag_agent", rag_agent)
graph.add_node("memory_agent", memory_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_node("writer_node", writer_node)

graph.set_entry_point("triage")

graph.add_conditional_edges(
    "triage",
    route_query,
    {
        "rag_agent": "rag_agent",
        "memory_agent": "memory_agent",
        "orchestrator": "orchestrator",
    },
)

# All agent paths converge on writer_node → END
graph.add_edge("rag_agent", "writer_node")
graph.add_edge("memory_agent", "rag_agent")
graph.add_edge("orchestrator", "writer_node")
graph.add_edge("writer_node", END)

# ── Compile ───────────────────────────────────────────────────────────────────
# graph_builder is compiled in main.py lifespan with AsyncRedisSaver
graph_builder = graph
