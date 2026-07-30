import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


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
