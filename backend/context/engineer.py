from dataclasses import dataclass
from typing import Any

import tiktoken
from langchain_core.documents import Document
from optimiser import Optimiser

ENCODING = tiktoken.get_encoding("cl100k_base")

# Unsloth Studio Gemma 4 E4B deployment.
N_CTX = 60_928

# Tokens reserved for the model's own generated answer. Enforced via
# llm.py's main_llm(max_tokens=RESERVED_OUTPUT_TOKENS).
RESERVED_OUTPUT_TOKENS = 2_048

# Buffer against tiktoken cl100k_base being only an approximation of
# Gemma's real tokenizer (see ARCHITECTURE.md §3). Not enforced at a
# specific call site — it's headroom baked into TOTAL_INPUT_BUDGET below.
SAFETY_MARGIN_TOKENS = 2_048

TOTAL_INPUT_BUDGET = N_CTX - RESERVED_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS

AGENT_BUDGET = {
    "rag_agent": {
        "total": TOTAL_INPUT_BUDGET,
        "system_prompt": 1_500,
        "query_and_prompt_scaffolding": 1_100,
        "short_term_memory": 9_000,
        "long_term_memory": 6_800,
        "retrieved_chunks": 36_000,
        "spare": 2_432,
    },
    "orchestrator_node": {
        "total": TOTAL_INPUT_BUDGET,
        "system_prompt": 1_500,
        "query_and_prompt_scaffolding": 1_100,
        "short_term_memory": 7_900,
        "long_term_memory": 11_300,
        "retrieved_chunks": 31_600,
        "spare": 3_432,
    },
}


optimiser = Optimiser()


@dataclass
class ContextBundle:
    memory_context: str
    short_term_context: str
    long_term_context: str
    chunks_context: str
    token_count: dict[str, int]
    total_tokens: int
    budget_exceeded: bool
    decision: list[dict[str, Any]]


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _format_short_term(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in turns:
        role = "AI" if message["role"] == "ai" else "Human"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def _format_long_term(summaries: list[dict[str, Any]]) -> str:
    return "\n\n".join(s.get("summary", "") for s in summaries if s.get("summary"))


def _format_chunks(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def fit_short_term(turns: list[dict[str, Any]], cap: int) -> tuple[str, int, bool]:
    """Drop oldest human/assistant pairs first until the formatted text fits cap."""
    truncated = False
    while turns:
        text = _format_short_term(turns)
        tokens = count_tokens(text)
        if tokens <= cap:
            return text, tokens, truncated
        turns = turns[2:]
        truncated = True
    return "", 0, truncated


def fit_long_term(summaries: list[dict[str, Any]], cap: int) -> tuple[str, int, bool]:
    """Summaries arrive sorted best-similarity-first (LongTermMemory.load); drop the
    lowest-ranked (tail) first."""
    truncated = False
    while summaries:
        text = _format_long_term(summaries)
        tokens = count_tokens(text)
        if tokens <= cap:
            return text, tokens, truncated
        summaries = summaries[:-1]
        truncated = True
    return "", 0, truncated


def fit_chunks(docs: list[Document], cap: int) -> tuple[str, int, bool]:
    """Sort the Docs according to the best-relevance-first (ChunksRetriever rerank) using the relevance score; drop the lowest-ranked (tail) first."""

    docs = sorted(docs, key=lambda d: d.metadata.get("score", 0.0), reverse=True)
    truncated = False
    original_count = len(docs)
    overflow_threshold = original_count * 0.5
    while docs:
        text = _format_chunks(docs)
        tokens = count_tokens(text)
        if tokens <= cap:
            return text, tokens, truncated
        if len(docs) <= overflow_threshold:
            compresssed = optimiser.compress(
                [d.page_content for d in docs], target_tokens=cap
            )
            return compresssed, count_tokens(compresssed), True
        docs = docs[:-1]
        truncated = True
    return "", 0, truncated


def assemble(
    agent_name: str,
    short_term_memory: list[dict[str, Any]] | None = None,
    long_term_memory: list[dict[str, Any]] | None = None,
    retrieved_chunks: list[Document] | None = None,
) -> ContextBundle:
    budget = AGENT_BUDGET[agent_name]

    short_term_text, st_tokens, st_truncated = fit_short_term(
        short_term_memory or [], budget["short_term_memory"]
    )
    long_term_text, lt_tokens, lt_truncated = fit_long_term(
        long_term_memory or [], budget["long_term_memory"]
    )
    chunks_text, chunks_tokens, chunks_truncated = fit_chunks(
        retrieved_chunks or [], budget["retrieved_chunks"]
    )

    token_count = {
        "short_term_memory": st_tokens,
        "long_term_memory": lt_tokens,
        "retrieved_chunks": chunks_tokens,
    }

    decision: list[dict[str, Any]] = []
    if st_truncated:
        decision.append(
            {"field": "short_term_memory", "action": "dropped_oldest_pairs"}
        )
    if lt_truncated:
        decision.append(
            {"field": "long_term_memory", "action": "dropped_lowest_relevance"}
        )
    if chunks_truncated:
        decision.append(
            {"field": "retrieved_chunks", "action": "dropped_lowest_relevance"}
        )

    memory_context = "\n\n".join(t for t in (short_term_text, long_term_text) if t)

    return ContextBundle(
        memory_context=memory_context,
        short_term_context=short_term_text,
        long_term_context=long_term_text,
        chunks_context=chunks_text,
        token_count=token_count,
        total_tokens=sum(token_count.values()),
        budget_exceeded=bool(decision),
        decision=decision,
    )
