import os

from langgraph.graph import END, StateGraph

from .nodes.memory_node import memory_node
from .nodes.orchestrator import orchestrator_node
from .nodes.rag_agent import rag_agent
from .nodes.triage import triage_node
from .nodes.writer_node import writer_node
from .state import SynapticState

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


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
