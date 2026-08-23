# Synaptic — Agent Instructions

This file is the primary source of procedural knowledge for agents working on this repository. Read it before touching code.

## Conversation Mode

Default to explaining and discussing — derivations, tradeoffs, "why" behind existing code, concept clarification. Do **not** implement, edit, or refactor code unless the user explicitly asks for the change. If a clarifying question surfaces something that looks like it should change, say so and ask before editing rather than editing and then summarizing.

## Living Engineering Records

[`decisions.md`](decisions.md) and [`flow.md`](flow.md) are mandatory living engineering records.
Read both before planning or implementing any phase, feature, migration, architectural change, or cross-module refactor.

`decisions.md` records meaningful choices, alternatives, rationale, accepted tradeoffs, consequences, evidence, and status.
Use it as the chronological working log.
Continue to create an ADR for an architectural decision that is difficult to reverse or affects multiple phases.

`flow.md` records implemented execution at file and function level, graph branches, state movement, failure paths, Mermaid diagrams, and the exact path affected by active work.
Treat present code as authoritative when phase prose, commit wording, and runtime behavior disagree.

### Mandatory Update Triggers

Every agent must update these records at the following points:

1. When a phase is created, add proposed decisions and the planned execution path without presenting them as implemented.
2. Before phase implementation begins, replace the active-change section with the exact planned files, functions, modules, state fields, and boundaries.
3. At each meaningful implementation milestone, reconcile the decisions and active path with the code that now exists.
4. Before continuing after implementation diverges from the plan, record the divergence, its reason, and its tradeoff.
5. At phase completion, trace the code again, record actual rather than planned flow, finalize decision statuses, and prompt the user to review both files.

Agents must proactively tell the user when a record update or review is due.
Do not wait for the user to remember this process.
Do not add an unimplemented phase to the implemented phase ledger.

### Mental-Model Comprehension Gate

Phase work is always a substantial change.
A feature, migration, architectural change, protocol change, storage change, or cross-module refactor is also substantial.

Before modifying implementation code for a substantial change:

1. Explain the proposed mental model, including execution order, file and function ownership, state movement, dependencies, failure behavior, and accepted tradeoffs.
2. Show a Mermaid diagram when control flow, data flow, state transitions, or three or more interacting components are involved.
3. Quiz the user with three to five questions that test the key concepts rather than trivia.
4. If an answer is incomplete or incorrect, explain the missed concept and re-quiz it.
5. Continue the teaching loop until the key answers are correct.
6. Ask the user to confirm readiness after the answers are correct.
7. Begin implementation only after correct answers and explicit readiness confirmation.

This is a hard gate and a teaching loop.
Design approval alone does not satisfy it.

Small isolated bug fixes and documentation-only edits are exempt unless they change architecture, a cross-module boundary, or phase scope.
When uncertain, treat the change as substantial and ask before editing code.

---

## Skill Routing

Before any task, load every applicable skill. Skill invocation is not optional — general reasoning is a fallback only when no relevant skill exists.

| Task                                                  | Skills to load                                  |
| ----------------------------------------------------- | ----------------------------------------------- |
| Add or modify a LangGraph node                        | `ai-engineer`, `tdd`                            |
| Modify the RAG pipeline (chain, retriever, reranker)  | `ai-engineer`, `tdd`                            |
| Add or change a FastAPI endpoint                      | `api-design`, `tdd`                             |
| Modify memory system (Redis, pgvector, MemoryManager) | `ai-engineer`, `tdd`                            |
| Debug a broken node, failing retrieval, or bad intent | `diagnose`                                      |
| Improve or analyse architecture                       | `architecture`, `improve-codebase-architecture` |
| Plan a refactor                                       | `request-refactor-plan`                         |
| Review a branch or PR                                 | `review`                                        |
| Run a QA session and file issues                      | `qa`                                            |
| Triage GitHub issues                                  | `triage`                                        |
| Write a PRD from conversation context                 | `to-prd`                                        |
| Write Python code (any file)                          | `python-expert`                                 |
| Understand an unfamiliar module                       | `zoom-out`                                      |
| End a session for handoff                             | `handoff`                                       |
| Any software design decision                          | `software-architecture`                         |

Combinations are the norm: adding a new node requires `ai-engineer` + `tdd` + `python-expert` simultaneously.

---

## Architecture

The canonical reference is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The summary below is orientation for agents — always defer to `docs/ARCHITECTURE.md` for authoritative detail.

### Request Flow

```
POST /v1/chat/completions
        │
        ▼
  main.py — assembles SynapticState
        │  (short_term_memory, long_term_memory, system_prompt pre-loaded)
        ▼
  LangGraph StateGraph (graph.py)
        │
        ▼
  triage_node
        │  sets: intent ('rag' | 'memory' | 'multi' | 'blocked')
        │  sets: metadata_filters (structured filters from NL query)
        ├──► rag_agent       (intent == 'rag')
        ├──► memory_agent    (intent == 'memory') → rag_agent
        └──► orchestrator    (intent == 'multi')
        │
        ▼
  writer_node → END
        │
        ▼
  StreamingResponse (OpenAI-compatible SSE, chat.completion.chunk)
```

### LangGraph State Contract

All nodes read and write `SynapticState` (defined in [backend/agents/graph.py](backend/agents/graph.py)). **This is the API between nodes — not function signatures.**

Key fields every node author must understand:

| Field               | Type             | Set by                      | Read by                     |
| ------------------- | ---------------- | --------------------------- | --------------------------- |
| `query`             | `str`            | `main.py`                   | all nodes                   |
| `intent`            | `str`            | `triage_node`               | `route_query` edge          |
| `metadata_filters`  | `dict`           | `triage_node`               | `rag_agent`                 |
| `short_term_memory` | `list[dict]`     | `main.py` (pre-load)        | `rag_agent`, `memory_agent` |
| `long_term_memory`  | `list[dict]`     | `main.py` (pre-load)        | `orchestrator`              |
| `retrieved_chunks`  | `list[dict]`     | `rag_agent`                 | `writer_node`, citations    |
| `final_answer`      | `str`            | `rag_agent` / `writer_node` | `main.py` → SSE             |
| `citations`         | `list[dict]`     | `rag_agent`                 | `writer_node`               |
| `trace_id`          | `str`            | `main.py`                   | Langfuse                    |
| `latency_ms`        | `dict[str, int]` | each node                   | `metrics` endpoint          |

Valid `intent` values: `"rag"`, `"memory"`, `"multi"`, `"blocked"`.

### Agent Roster

| Node                | File                           | Role                                                                          |
| ------------------- | ------------------------------ | ----------------------------------------------------------------------------- |
| `triage_node`       | `agents/nodes/triage.py`       | Classifies intent, extracts `metadata_filters`                                |
| `rag_agent`         | `agents/nodes/rag_agent.py`    | Invokes `RagChain`, populates `retrieved_chunks`, `citations`, `final_answer` |
| `memory_node`       | `agents/nodes/memory_node.py`  | Handles explicit memory read/write; chains into `rag_agent`                   |
| `orchestrator_node` | `agents/nodes/orchestrator.py` | Merges multi-intent results via `MemoryManager`                               |
| `writer_node`       | `agents/nodes/writer_node.py`  | Final answer formatting; may compress context                                 |

All agents use the **same model**: `unsloth/gemma-4-E4B-it` via Unsloth Studio's OpenAI-compatible REST API. Agent differentiation is entirely through system prompt and tool access — not separate model instances.

### Tech Stack

| Layer               | Technology                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| Backend             | FastAPI (Python 3.11+), async, SSE via `StreamingResponse`                                                     |
| Agent orchestration | LangGraph 0.2+ — stateful graph, typed state, AsyncRedisSaver checkpointing                                    |
| LLM                 | Gemma 4 E4B (Q5_K_M) via Unsloth Studio — `ChatOpenAI` client, env: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` |
| Embeddings          | `nomic-embed-text` (HuggingFace, local) — 768-dim, `HuggingFaceEmbeddings`                                     |
| Vector DB           | pgvector on PostgreSQL 16 — HNSW index, cosine similarity                                                      |
| Short-term memory   | Redis 7 — `session:{id}:turns` list, TTL 24 h, max 10 turns                                                    |
| Long-term memory    | pgvector `memory_summaries` table — embedded session summaries                                                 |
| Sparse retrieval    | `rank-bm25` — in-memory, rebuilt on startup                                                                    |
| Reranker            | `bge-reranker-base` (sentence-transformers cross-encoder, CPU)                                                 |
| Observability       | Langfuse (self-hosted Docker) — LangChain `CallbackHandler`                                                    |
| Evals               | RAGAS — offline eval suite on held-out test sets                                                               |

---

## Engineering Standards

These standards are derived from the existing codebase. Every pull request is reviewed against them.

### 1. Node Authorship Rules

Every LangGraph node **must**:

- Be an `async` function with signature `async def node_name(state: SynapticState) -> dict`.
- Return only the state fields it sets — not a full `SynapticState`.
- Be decorated with `@observe(name="node_name")` from `langfuse` for tracing.
- Forward `callbacks` from state when invoking chains: `config={"callbacks": state.get("callbacks", [])}`.

```python
from langfuse import observe
from agents.graph import SynapticState

@observe(name="my_node")
async def my_node(state: SynapticState) -> dict:
    callbacks = state.get("callbacks", [])
    result = await some_chain.ainvoke(..., config={"callbacks": callbacks})
    return {"final_answer": result["answer"]}
```

Never return the entire state dict from a node.

### 2. Dependency Injection at Startup

Module-level singletons (LLM, embeddings, MemoryManager, RagChain) are initialised once in the FastAPI `lifespan` context and injected into node modules via module-level attributes:

```python
# main.py lifespan
rag_agent_module.init(_embeddings)
orch_module.memory_manager = manager
mem_module.memory_manager = manager
writer_module.memory_manager = manager
```

Do **not** instantiate `ChatOpenAI`, `HuggingFaceEmbeddings`, or `MemoryManager` inside node files. Read env vars at module level only for the `Triage` class which has its own `_create_llm()` pattern — this is intentional.

### 3. Structured Outputs via Pydantic

Use Pydantic `BaseModel` for all LLM output schemas. Parse with `.model_validate_json()`. Strip markdown fences before parsing:

````python
import re
from pydantic import BaseModel

class MyOutput(BaseModel):
    field: str

clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
result = MyOutput.model_validate_json(clean)
````

### 4. Python Standards

Follow the `python-expert` skill. Priority order: **Correctness → Type Safety → Performance → Style**.

- All function signatures carry type hints including return types.
- Use `Optional[T]` / `T | None` for nullable returns.
- No mutable default arguments.
- Specific exception types in `except` clauses — no bare `except`.
- Functions > 50 lines → split. Files > 200 lines → split.
- Early return over nested conditions.

### 5. API Conventions

The backend exposes an **OpenAI-compatible** interface. The existing `/v1/chat/completions` endpoint streams `chat.completion.chunk` SSE events and terminates with `data: [DONE]\n\n`. New endpoints must follow conventions from the `api-design` skill:

- Plural, kebab-case resource URLs under `/v1/`.
- Correct HTTP status codes (201 for creates, 204 for deletes).
- Pydantic `BaseModel` for all request bodies.
- Error responses: `{"error": {"code": "...", "message": "..."}}` — never expose stack traces.

### 6. Observability Requirements

Every node records its latency in `latency_ms`. LLM calls must flow through the Langfuse `CallbackHandler` passed via `config={"callbacks": ...}`. Do not add `print` statements to production paths — use `logging.getLogger(__name__)`.

### 7. Memory System Rules

`MemoryManager` is the single interface to memory — never call `ShortTermMemory` or `LongTermMemory` directly from nodes. The manager's key methods:

- `load(session_id, query)` — parallel fetch of STM + LTM
- `append_turn(session_id, human_msg, ai_msg)` — called after each turn
- `archive_session(session_id)` — summarises and clears; only runs if ≥ 5 turns

Redis key structure: `session:{session_id}:turns` (list of JSON turn dicts).

### 8. Test-Driven Development

Use the `tdd` skill for all new features and bug fixes. Key rules:

- One test → one implementation → repeat (vertical slices, not horizontal).
- Tests verify behaviour through public interfaces, not implementation details.
- No mocking of internal collaborators — prefer integration-style tests against real code paths.
- Write the regression test **before** the fix, only if a correct seam exists.

---

## Data Schemas (Quick Reference)

Full schemas in [docs/ARCHITECTURE.md §5](docs/ARCHITECTURE.md).

### pgvector — `document_chunks`

Columns: `id`, `content`, `embedding` (768-dim), `source`, `doc_id`, `title`, `tags TEXT[]`, `quality` (`'HQ'|'LQ_EDIT'|'LQ_CLOSE'`), `score`, `metadata JSONB`.
Index: HNSW cosine on `embedding`, GIN on `tags`.

### pgvector — `memory_summaries`

Columns: `id`, `session_id`, `summary`, `embedding` (768-dim), `turn_count`, `created_at`, `relevance_score` (pruned < 0.1).

### Redis Keys

- `session:{session_id}:turns` — list of JSON turn objects, TTL 24 h, max 10 turns
- `cache:{sha256(system_prompt+query_embedding_hex)}` — prompt cache, TTL 1 h
- `checkpoint:{thread_id}:{checkpoint_id}` — managed by LangGraph AsyncRedisSaver

---

## ADRs

Architectural decisions are recorded in [docs/adr/](docs/adr/). Before proposing a structural change, read the existing ADRs. When an architectural decision is made or rejected during work, record it using the `architecture-decision-records` skill and [docs/adr/template.md](docs/adr/template.md).

Current ADRs:

- [0001 — Query condensation / contextual RAG](docs/adr/0001-query-condensation-contextual-rag.md)

---

## Domain Vocabulary

Use these terms exactly. Do not drift to generic alternatives.

| Term                | Meaning                                                                         |
| ------------------- | ------------------------------------------------------------------------------- |
| `SynapticState`     | The typed state dict threaded through the LangGraph graph                       |
| `intent`            | The triage classification: `rag`, `memory`, `multi`, `blocked`                  |
| `metadata_filters`  | Structured filters extracted by triage from natural language                    |
| `triage_node`       | The first node; classifies intent and extracts metadata filters                 |
| `rag_agent`         | The node that invokes `RagChain` and populates retrieved chunks                 |
| `memory_node`       | The node that handles explicit memory read/write requests                       |
| `orchestrator_node` | The node that merges results for multi-intent queries                           |
| `writer_node`       | The final node; formats the answer before the graph ends                        |
| `MemoryManager`     | The single interface to both short-term (Redis) and long-term (pgvector) memory |
| `RagChain`          | The LCEL chain in `chain/rag_chain.py` — hybrid search + rerank + generate      |
| `retrieved_chunks`  | The list of `{content, source, score, metadata}` dicts from RAG                 |
| `citations`         | The `[{title, relevance_score}]` list derived from retrieved chunks             |
| `session_id`        | The UUID that keys Redis turns and LangGraph checkpoints                        |
| `trace_id`          | The Langfuse trace ID derived from `session_id`                                 |

---

## Environment Variables

Required at runtime (set in `.env` or Docker Compose):

| Variable                    | Used for                                                    |
| --------------------------- | ----------------------------------------------------------- |
| `LLM_MODEL`                 | Model name passed to `ChatOpenAI`                           |
| `LLM_BASE_URL`              | Unsloth Studio OpenAI-compatible base URL                   |
| `LLM_API_KEY`               | API key for LLM endpoint                                    |
| `REDIS_URL`                 | Redis connection string (default: `redis://localhost:6379`) |
| `POSTGRES_URL` / `CONN_STR` | pgvector connection string                                  |
| `LANGFUSE_HOST`             | Langfuse self-hosted URL                                    |
| `LANGFUSE_PUBLIC_KEY`       | Langfuse auth                                               |
| `LANGFUSE_SECRET_KEY`       | Langfuse auth                                               |

Swap models without code changes: set `LLM_MODEL` to a larger variant (e.g. `unsloth/gemma-4-26b-A4B-it`).

---

## Project Phase Tracker

The build is structured in phases documented in [docs/](docs/):

| Phase | Doc                                              | Focus                                                |
| ----- | ------------------------------------------------ | ---------------------------------------------------- |
| 1     | [Phase 1](docs/phase-1/README.md)                | FastAPI + pgvector + ingestion + basic RAG           |
| 1.2   | [Phase 1.2](docs/phase-1.2/README.md)            | Hybrid search, reranker, HyDE                        |
| 2     | [Phase 2](docs/phase-2/README.md)                | LangGraph graph, memory system, triage               |
| 3     | [Phase 3](docs/phase-3/README.md)                | Context engineer, token budget, compression          |
| 3.2   | [Phase 3.2](docs/phase-3.2/README.md)            | Listable, persistent conversation history per client |
| 4     | [Phase 4](docs/phase-4/README.md)                | Guardrail classifier, advanced routing               |
| 5     | [Phase 5](docs/phase-5/README.md)                | Langfuse, RAGAS evals, streaming polish              |
