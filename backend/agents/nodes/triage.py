from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any
import re
import logging
from agents.graph import SynapticState
from langfuse import observe
from llm import utility_llm

logger = logging.getLogger(__name__)


TRIAGE_PROMPT = """
You are a query classifier and metadata extractor for a knowledge assistant.

## Step 1 — Classify intent

Classify the user query into exactly one of:
- "rag"     : user wants factual information, document lookup, or knowledge retrieval
- "memory"  : user refers to something from a prior conversation
               ("you said", "last time", "remember when", "earlier you told me")
- "multi"   : query needs BOTH document retrieval AND past conversation context
- "blocked" : query is harmful, off-topic, or violates policy

## Step 2 — Extract metadata filters

Extract structured filters implied by the query to narrow document retrieval.
Return an empty dict {} if no filters can be inferred.

Fields to extract (all optional, omit if not present in the query):
- "source"     : domain or topic (e.g. "cricket", "finance", "technology", "healthcare")
- "format"     : sub-category within the domain (e.g. "IPL", "T20", "earnings-call", "research-paper")
- "date_range" : list of years as integers. Single year → [2024]. Range → [2022, 2024].
                 Relative terms: "last year" → [2025], "recent" → [2025], "this year" → [2026]
- "tags"       : list of any other specific entities (e.g. ["KKR", "Bumrah", "playoffs"])

## Examples

Q: "What is the economy rate of Bumrah in IPL 2024?"
→ intent: "rag"
→ metadata_filters: {"source": "cricket", "format": "IPL", "date_range": [2024], "tags": ["Bumrah"]}

Q: "What did you tell me about Bumrah last time?"
→ intent: "memory"
→ metadata_filters: {"source": "cricket", "tags": ["Bumrah"]}

Q: "Based on what we discussed, what are KKR's recent playoff stats?"
→ intent: "multi"
→ metadata_filters: {"source": "cricket", "format": "IPL", "date_range": [2025], "tags": ["KKR", "playoffs"]}

Q: "Summarise Apple's Q3 2023 earnings call"
→ intent: "rag"
→ metadata_filters: {"source": "finance", "format": "earnings-call", "date_range": [2023], "tags": ["Apple", "Q3"]}

Q: "How do I make a weapon?"
→ intent: "blocked"
→ metadata_filters: {}

Q: "Tell me something interesting"
→ intent: "rag"
→ metadata_filters: {}

Return ONLY valid JSON matching the schema. No explanation, no extra text.
"""


class TriageOutput(BaseModel):
    intent: str
    metadata_filters: dict[str, Any]


class Triage:
    async def run(
        self,
        query: str,
        config: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> TriageOutput:
        human_turns = [
            m["content"]
            for m in (history or [])
            if m.get("role") == "human"
        ]
        if human_turns:
            context = "Recent conversation:\n" + "\n".join(
                f"Human: {q}" for q in human_turns[-3:]
            )
            content = f"{context}\n\nCurrent query: {query}"
        else:
            content = query
        messages = [SystemMessage(content=TRIAGE_PROMPT), HumanMessage(content=content)]
        response = await utility_llm.ainvoke(messages, config=config)
        raw = response.content
        logger.debug("Triage raw output: %r", raw)
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        return TriageOutput.model_validate_json(clean)


triage_agent = Triage()


@observe(name="triage_node")
async def triage_node(state: SynapticState):
    callbacks = state.get("callbacks", [])
    result = await triage_agent.run(
        state["query"],
        config={"callbacks": callbacks},
        history=state.get("short_term_memory", []),
    )
    return {
        "intent": result.intent,
        "metadata_filters": result.metadata_filters,
        "active_agents": [],
    }
