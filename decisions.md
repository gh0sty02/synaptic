# Synaptic Engineering Decisions

This file is the living log of meaningful engineering decisions that affect Synaptic's code, runtime behavior, architecture, dependencies, or delivery phases.
It explains why a change was made, which alternatives were considered, which tradeoffs were accepted, and what future work must preserve or revisit.
It complements architecture decision records in `docs/adr/`.
Use this file for the chronological working record and use an ADR when a decision is architectural, difficult to reverse, or important across multiple phases.

## Maintenance Rules

Update this file whenever any of the following happens:

- A phase is created and introduces a proposed technical direction.
- A phase begins implementation and a proposed decision becomes active.
- A meaningful implementation choice is accepted, rejected, or superseded.
- The implementation diverges from the phase plan.
- A new library, protocol, storage model, API shape, orchestration pattern, or cross-module boundary is chosen.
- A phase milestone or phase completion changes the consequences of an earlier decision.

Do not record routine edits that have no meaningful alternative or tradeoff.
Do not invent historical rationale.
Label rationale as `Verified` when it is stated in code comments, an ADR, a phase record, or a commit.
Label rationale as `Inferred` when it follows from the implementation but no source states it directly.
Proposed decisions remain `Proposed` until implementation begins or the user accepts them.

## Decision Statuses

- `Proposed` means the choice is under discussion and must not be described as implemented.
- `Accepted` means the choice is present in implemented code or explicitly approved for active implementation.
- `Superseded` means a later decision replaced the choice.
- `Rejected` means the choice was considered and deliberately not selected.

## Entry Template

```markdown
## D-NNN: Short decision title

- Date: YYYY-MM-DD
- Status: Proposed | Accepted | Superseded | Rejected
- Phase: Phase identifier or cross-phase
- Scope: Files, functions, modules, or interfaces affected
- Evidence: Commit, code comment, phase record, ADR, test, or user decision
- Rationale confidence: Verified | Inferred

### Context

State the problem and the constraint that forced a choice.

### Decision

State the chosen approach precisely.

### Alternatives Considered

- Name the meaningful alternative and why it was not selected.

### Accepted Tradeoffs

- State what became harder, slower, more expensive, or more constrained.

### Consequences

- State what future implementation must preserve, measure, or revisit.
```

## Backfill Boundary

This backfill is based on code present on `feat/guardrails` at commit `ea21d35`, reachable branch history, phase records, code comments, and the current worktree inspected on 2026-08-12.
Only phases with implemented code are represented.
Partially implemented phases contribute only the decisions supported by existing code.
Uncommitted backend edits present during the backfill are not treated as accepted historical decisions.

## D-001: Own the document and chunk schema instead of using LangChain's PGVector store

- Date: Backfilled through 2026-07-24
- Status: Accepted
- Phase: Phase 1 and Phase 3.1
- Scope: `backend/db/schema.sql`, `backend/ingestion/`, and `backend/retrieval/chunks_retriever.py`
- Evidence: The `ChunksRetriever` class comment and the implemented ingestion and retrieval code
- Rationale confidence: Verified

### Context

Synaptic needs document-level metadata, chunk-level embeddings, deduplication, resumable ingestion, bulk loading, and explicit control over HNSW maintenance.

### Decision

Use owned `documents` and `chunks` tables and query them directly with `psycopg` from a custom LangChain `BaseRetriever`.

### Alternatives Considered

- LangChain's PGVector store was not selected because it does not directly provide the required document/chunk split, ingestion status, content-hash deduplication, or index lifecycle control.

### Accepted Tradeoffs

- Synaptic must maintain SQL, vector registration, sync and async retrieval paths, metadata filtering, and any future MMR behavior itself.

### Consequences

- Retrieval and ingestion changes must preserve the document-to-chunk foreign-key boundary and resumable ingestion semantics.
- Schema changes that alter this boundary require an ADR in addition to an entry here.

## D-002: Use staged retrieval quality controls

- Date: Backfilled from commits beginning 2026-05-18
- Status: Accepted
- Phase: Phase 1.2
- Scope: `backend/retrieval/chunks_retriever.py` and `backend/retrieval/reranker.py`
- Evidence: Retrieval cutoff, over-fetch, cross-encoder reranking, and empty-result behavior in the current code
- Rationale confidence: Verified

### Context

Nearest-neighbor similarity alone can return weak candidates and does not order passages by answer relevance as accurately as a pairwise model.

### Decision

Filter dense candidates by cosine-distance cutoff, over-fetch candidates, rerank them with `cross-encoder/ms-marco-MiniLM-L-6-v2`, apply a reranker score floor, and return only the configured top results.

### Alternatives Considered

- Returning raw pgvector nearest neighbors was rejected because it provides weaker precision.
- Running the cross-encoder across the full corpus was rejected because its CPU cost is unsuitable for first-stage retrieval.

### Accepted Tradeoffs

- The reranker adds model memory, startup or first-query latency, and CPU work.
- Thresholds can hide relevant evidence if they are set too aggressively.

### Consequences

- Retrieval evaluation must measure both recall before reranking and answer quality after reranking.
- Threshold changes are meaningful decisions and must be recorded with evaluation evidence.

## D-003: Use `SynapticState` as the contract between LangGraph nodes

- Date: Backfilled from Phase 2 commits through 2026-05-28
- Status: Accepted
- Phase: Phase 2
- Scope: `backend/agents/state.py`, `backend/agents/graph.py`, and `backend/agents/nodes/`
- Evidence: The implemented `StateGraph`, typed state, conditional edges, and node return shapes
- Rationale confidence: Verified

### Context

Triage, retrieval, memory, orchestration, writing, and guardrails need to exchange data without tightly coupling node function signatures.

### Decision

Thread one typed `SynapticState` through LangGraph and require nodes to return only the fields they set.

### Alternatives Considered

- Direct node-to-node calls with custom parameters were not selected because they would couple orchestration to each function signature.
- Passing an untyped dictionary without a declared contract was not selected because state ownership would be harder to inspect and validate.

### Accepted Tradeoffs

- The state can accumulate fields and stale values if ownership is not documented.
- Runtime behavior still depends on consistent field names across independently implemented nodes.

### Consequences

- Every new node must document the state fields it reads and writes in `flow.md`.
- State contract changes require updates to all affected nodes and flow branches.

## D-004: Put short-term and long-term memory behind `MemoryManager`

- Date: Backfilled from commits through 2026-05-28
- Status: Accepted
- Phase: Phase 2
- Scope: `backend/memory/manager.py`, `backend/memory/short_term.py`, `backend/memory/long_term.py`, and memory-consuming nodes
- Evidence: Implemented `MemoryManager.load`, `append_turn`, `archive_session`, and startup injection
- Rationale confidence: Verified

### Context

Conversation turns and cross-session summaries have different storage, retention, and retrieval behavior, but nodes need one memory interface.

### Decision

Use Redis for bounded short-term turns, pgvector for embedded long-term summaries, and `MemoryManager` as the node-facing interface.

### Alternatives Considered

- Calling Redis and pgvector adapters directly from each node was rejected because it would duplicate orchestration and leak storage details.
- Keeping all memory only in Redis was rejected because it would not support semantic retrieval across archived sessions.

### Accepted Tradeoffs

- A request depends on two storage systems and must tolerate their different latency and failure modes.
- Long-term memory is a summary, not verbatim conversation history.

### Consequences

- Phase 2 memory must not be described as listable persistent conversation history.
- Node code should use `MemoryManager`; direct adapter access is a boundary exception that must be recorded.

## D-005: Initialize shared model and storage dependencies once at startup

- Date: Backfilled from commits `191bca6` through `5399ca7`
- Status: Accepted
- Phase: Phase 2 and cross-phase runtime
- Scope: `backend/llm.py`, `backend/main.py`, and node module initialization
- Evidence: Shared LLM singleton commits and FastAPI lifespan dependency injection
- Rationale confidence: Verified

### Context

LLM clients, embeddings, database pools, memory adapters, and the compiled graph are expensive or stateful resources.

### Decision

Create shared LLM clients at module scope, initialize embeddings and storage resources in the FastAPI lifespan, inject shared collaborators into node modules, and compile the graph once with its Redis checkpointer.

### Alternatives Considered

- Constructing models and managers inside each node invocation was rejected because it would repeat initialization and fragment resource ownership.
- Hiding dependency creation inside every module was rejected because startup and shutdown would become difficult to coordinate.

### Accepted Tradeoffs

- Module-level injection requires startup to complete before nodes are usable.
- Tests need deliberate seams for replacing initialized collaborators.

### Consequences

- New heavyweight dependencies must be owned by the lifespan or an existing shared module.
- Node code must not instantiate model clients or memory managers per request.

## D-006: Apply per-agent context budgets before generation

- Date: Backfilled from commits `e0470af` and `95c18f9`
- Status: Accepted
- Phase: Phase 3
- Scope: `backend/context/budgets.py`, `backend/context/engineer.py`, and `backend/context/optimiser.py`
- Evidence: Implemented budget tables, context assembly, token counts, truncation, and compression fallback
- Rationale confidence: Verified

### Context

Short-term memory, long-term memory, and retrieved chunks compete for a finite model context window.

### Decision

Allocate context by agent and source, fit each source to its budget, preserve higher-value chunks first, and compress oversized chunk context through the optimiser.

### Alternatives Considered

- Concatenating every available context item was rejected because it can exceed the model window and obscure relevant evidence.
- Applying one undifferentiated global truncation was rejected because it would not protect the relative roles of memory and retrieved evidence.

### Accepted Tradeoffs

- Budgeting can discard useful context.
- Compression adds latency and can alter wording.
- The current fallback compressor uses an LLM and needs evaluation before its thresholds can be considered stable.

### Consequences

- Nodes must expose token counts, budget decisions, and truncation state.
- Budget or compression changes require quality evaluation and a decision update.

## D-007: Condense follow-up questions and make HyDE optional

- Date: Backfilled through commit `9d86b3f`
- Status: Accepted
- Phase: Phase 3
- Scope: `backend/chain/rag_chain.py`, `backend/retrieval/hyde.py`, and ADR 0001
- Evidence: ADR 0001, the condensation chain, and the optional HyDE stage
- Rationale confidence: Verified

### Context

Follow-up questions can depend on conversation history, while retrieval works best with a standalone query.
Some queries also retrieve better when embedded as answer-like text rather than question-like text.

### Decision

Condense a follow-up question using only human turns from memory, then optionally replace the retrieval query with a HyDE passage while preserving the original question for answer generation.

### Alternatives Considered

- Embedding ambiguous follow-up text directly was rejected because it loses referents from prior turns.
- Always enabling HyDE was rejected because it adds an LLM call and can introduce vocabulary that harms retrieval.

### Accepted Tradeoffs

- Condensation and HyDE add latency and another source of model error.
- HyDE text is intentionally not required to be factually correct because it is used only for retrieval.

### Consequences

- Evaluation must compare direct retrieval, condensed retrieval, and HyDE retrieval separately.
- The answer prompt must continue to receive the user's original question.

## D-008: Run dense and sparse retrieval concurrently, then rerank their union

- Date: Backfilled from commit `72688b7` and current implementation
- Status: Accepted
- Phase: Phase 3
- Scope: `backend/retrieval/hybrid.py`, `backend/retrieval/chunks_retriever.py`, and `backend/retrieval/bm25_retriever.py`
- Evidence: `HybridRetriever._aget_relevant_documents` and `merge_candidates`
- Rationale confidence: Verified for implementation and inferred for the selection rationale

### Context

Dense retrieval captures semantic similarity while BM25 captures exact tokens, identifiers, and rare terms.

### Decision

Invoke both retrievers concurrently, deduplicate candidates by exact page content, and use the cross-encoder to produce the final ranking.

### Alternatives Considered

- Dense-only retrieval was not retained because it misses some lexical matches.
- Sparse-only retrieval was not selected because it misses semantic matches.
- Score fusion is not the current implementation even when historical commit wording suggests it.

### Accepted Tradeoffs

- The in-memory BM25 corpus consumes memory and is rebuilt from PostgreSQL.
- Exact-content deduplication does not merge or normalize near-duplicate chunks.
- Cross-retriever scores are discarded before the shared reranker.

### Consequences

- `flow.md` must describe the implemented union-and-rerank path, not label it RRF unless code actually implements RRF.
- A future move to PostgreSQL full-text search or score fusion is a new decision.

## D-009: Build an answer-grounded, multi-source ingestion pipeline

- Date: Backfilled from commits `839e453` through `7347a70`
- Status: Accepted
- Phase: Phase 3.1
- Scope: `backend/ingestion/`, `backend/db/schema.sql`, and dataset evaluation inputs
- Evidence: Kaggle and StackExchange adapters, shared document builder, schema, and ingestion commits
- Rationale confidence: Verified

### Context

Retrieval quality depends on answer-bearing chunks, stable source metadata, deduplication, and an evaluation holdout that is not ingested into the searchable corpus.

### Decision

Normalize multiple Stack Overflow sources through adapters into a shared document builder, persist answer-grounded documents and chunks, and exclude held-out evaluation identifiers from ingestion.

### Alternatives Considered

- Maintaining separate ingestion pipelines for every source was rejected because normalization and quality rules would drift.
- Treating question titles alone as retrieval documents was rejected because citations need answer-bearing evidence.

### Accepted Tradeoffs

- Adapters must absorb source-specific encoding, field, and batching differences.
- Full-corpus ingestion and index rebuild remain operationally expensive.

### Consequences

- New sources must implement the shared adapter boundary and preserve holdout isolation.
- Dataset changes require both retrieval and evaluation review.

## D-010: Correlate requests and model calls with Langfuse

- Date: Backfilled from commit `60ac653` and later tracing changes
- Status: Accepted
- Phase: Implemented Phase 5 observability slice
- Scope: `backend/main.py`, observed nodes, and LLM invocation configs
- Evidence: Langfuse `CallbackHandler`, `@observe`, deterministic trace IDs, and node span metadata
- Rationale confidence: Verified

### Context

Multi-node requests contain multiple model calls, retrieval stages, budget decisions, and failure points that need one trace identity.

### Decision

Derive one Langfuse trace ID from `session_id`, pass a callback handler in graph configuration, observe nodes, and attach guardrail and budget metadata to spans.

### Alternatives Considered

- Independent traces for each model call were rejected because they would fragment one user request.
- Production `print` statements were rejected in favor of structured logging and tracing.

### Accepted Tradeoffs

- Callback propagation must remain consistent across graph and LCEL boundaries.
- Observability adds configuration and an external service dependency.

### Consequences

- Every new node and LLM call must preserve trace propagation.
- Missing or inconsistent callback state must be treated as an observability gap, not silently assumed to work.

## D-011: Make the input guardrail the graph entry point

- Date: 2026-08-06
- Status: Accepted on `feat/guardrails`
- Phase: Phase 4
- Scope: `backend/agents/graph.py`, `backend/agents/nodes/guardrail_node.py`, `backend/guardrails/classifier.py`, and `backend/guardrails/heuristics.py`
- Evidence: Commits `76183d2` and `ea21d35`
- Rationale confidence: Verified

### Context

Unsafe, off-topic, or prompt-injection input must be rejected before triage, retrieval, memory orchestration, or answer generation runs.

### Decision

Run `guardrail_node` first, use narrow regex heuristics for known injection forms, fall back to an LLM classifier for nuanced classification, and route blocked input to a static `blocked_node`.

### Alternatives Considered

- Guarding only inside `triage_node` was rejected because it mixes safety with intent routing and permits other work before enforcement.
- Regex-only classification was rejected as the primary defense because narrow patterns cannot cover semantic attacks.
- LLM-only classification was not selected because obvious patterns can be caught without model latency or cost.

### Accepted Tradeoffs

- Safe input without a heuristic match pays for an additional classifier call.
- Regex coverage is intentionally narrow and depends on evaluation-driven additions.
- The classifier can produce parsing or upstream failures.

### Consequences

- New graph entry paths must not bypass `guardrail_node`.
- Heuristic expansion must be based on labeled false negatives rather than speculative keyword growth.

## D-012: Fail closed when the input classifier fails

- Date: 2026-08-06
- Status: Accepted on `feat/guardrails`
- Phase: Phase 4
- Scope: `backend/agents/nodes/guardrail_node.py` and `backend/agents/nodes/blocked_node.py`
- Evidence: The fail-close code comment and `classifier_error` refusal path
- Rationale confidence: Verified

### Context

A classifier exception must not silently turn safety enforcement off.

### Decision

Convert any input-classifier exception into a blocked `classifier_error` verdict and return a static refusal without another model call.

### Alternatives Considered

- Failing open was rejected because classifier outages would allow every request through.
- Asking the same failing model to generate a refusal was rejected because it would repeat the unavailable or unsafe dependency.

### Accepted Tradeoffs

- Benign traffic is unavailable when the classifier fails.
- Broad exception handling is deliberate at this safety boundary, with the original failure logged.

### Consequences

- Classifier availability affects overall chat availability and must be monitored.
- The static refusal map must cover any new blocked category.

## D-013: Check accumulated streamed output in configurable windows

- Date: 2026-08-06
- Status: Accepted on `feat/guardrails`
- Phase: Phase 4
- Scope: `backend/main.py`, `backend/guardrails/classifier.py`, and `backend/evals/guardrail_runner.py`
- Evidence: Commit `144dc24` and the current SSE generator
- Rationale confidence: Verified for implementation and inferred for the latency tradeoff

### Context

Input safety does not prevent generated output from leaking prompt content or producing harmful text.
Classifying every token would make streaming prohibitively expensive.

### Decision

Accumulate generated text, run the output classifier after each configured character window, stop on a blocked verdict, emit a retraction marker, and terminate with `finish_reason="content_filter"`.

### Alternatives Considered

- Token-by-token classification was rejected because it multiplies classifier calls and latency.
- Buffering the entire answer before any SSE output was not selected because it removes token streaming.

### Accepted Tradeoffs

- Each chunk is yielded before the window check, so text already sent cannot actually be retracted.
- Responses shorter than one full window and the final partial window are not checked by the current generator.
- Window size trades classifier cost and latency against the amount of unchecked text.

### Consequences

- `flow.md` must show the yield-before-check order exactly.
- Any future change to buffer-before-yield, check the final remainder, or change the window size is a meaningful safety decision.
- Guardrail evaluation must track false positives and false negatives separately.

## D-014: Use OpenAI-compatible SSE as the frontend/backend boundary

- Date: 2026-08-06
- Status: Accepted on `feat/guardrails`
- Phase: Phase 1 streaming capability completed on the current branch
- Scope: `frontend/components/ui/chat-container.tsx` and `backend/main.py`
- Evidence: Commit `16205a1`, OpenAI SDK consumption, and `chat.completion.chunk` generation
- Rationale confidence: Verified for implementation and inferred for interoperability

### Context

The frontend needs incremental answer delivery and the backend aims to expose an OpenAI-compatible chat interface.

### Decision

Send `POST /v1/chat/completions`, return `StreamingResponse` events shaped as `chat.completion.chunk`, and terminate with `data: [DONE]`.

### Alternatives Considered

- A project-specific SSE schema was not selected because it would require a custom client protocol.
- Waiting for a complete JSON response was rejected because it removes incremental rendering.

### Accepted Tradeoffs

- The backend must reproduce OpenAI chunk ordering and completion semantics accurately.
- Static graph outputs need explicit bridging because they do not naturally emit chat-model token events.

### Consequences

- API changes must preserve OpenAI SDK compatibility unless this decision is superseded.
- New answer paths must be tested for both streamed tokens and static final answers.

## D-015: Maintain living decisions and execution flow with a comprehension gate

- Date: 2026-08-12
- Status: Accepted
- Phase: Cross-phase engineering process
- Scope: `decisions.md`, `flow.md`, `AGENTS.md`, and `CLAUDE.md`
- Evidence: Explicit user decision in the implementation session
- Rationale confidence: Verified

### Context

Phase plans and code changes are difficult to own when rationale, actual call paths, and the currently modified segment are not kept together.

### Decision

Maintain this decision log and a function-level `flow.md` throughout phase creation and implementation.
Before a phase, feature, migration, architectural change, or cross-module refactor changes code, present the mental model and enforce a quiz gate.
Teach missed concepts and re-quiz until the key answers are correct, then obtain explicit readiness confirmation before implementation.

### Alternatives Considered

- Relying only on commit messages was rejected because commits rarely preserve alternatives and tradeoffs.
- Relying only on ADRs was rejected because many meaningful working decisions do not justify a full ADR.
- A non-blocking quiz was rejected because it would not guarantee comprehension before a large change begins.

### Accepted Tradeoffs

- Large changes require an additional review loop before coding.
- Agents must spend time keeping the records aligned with implementation milestones.

### Consequences

- Phase creation, implementation start, meaningful milestones, plan divergence, and phase completion are mandatory update triggers.
- Agents must proactively prompt for the mental-model gate and record review.
- Small isolated fixes and documentation-only edits are exempt unless they change architecture or cross module boundaries.

## D-016: Condense the retrieval query in `orchestrator_node` before hybrid retrieval

- Date: 2026-08-18
- Status: Accepted
- Phase: Phase 3 follow-up (extends D-007)
- Scope: `backend/retrieval/query_condensation.py` (new), `backend/chain/rag_chain.py`, `backend/agents/nodes/orchestrator.py`
- Evidence: Langfuse trace `a69fecb582164783ac2ef7e6d59455d0`, user report, and this implementation session
- Rationale confidence: Verified

### Context

D-007 added query condensation only inside `RagChain`, used by the `rag` intent path. The `multi` intent path (`orchestrator_node`) ran `MemoryManager.load` and hybrid retrieval concurrently via `asyncio.gather`, so retrieval always received the raw, possibly referential follow-up text (e.g. "how to do the same in react") with no access to the resolved topic from prior turns. The trace showed this producing weak-relevance chunks on unrelated subtopics, which the answer LLM then included wholesale instead of a focused answer.

### Decision

Extract the condensation chain out of `RagChain` into a shared `retrieval/query_condensation.py::condense_query(question, memory_context)`. In `orchestrator_node`, load memory first, assemble a memory-only context bundle, condense the query against it, then retrieve with the condensed query. Memory load and retrieval are now sequential in this path instead of concurrent. The original `state["query"]` (not the condensed query) continues to be passed to `Orchestrator.merge` for final answer generation, preserving the D-007 rule that the answer prompt receives the user's actual question. `ORCHESTRATOR_PROMPT` also gained an explicit line forbidding citation of a document that does not actually support the claim being made, since the trace showed the model citing an unrelated retrieved-turn source for content it generalized from memory.

### Alternatives Considered

- Keeping memory load and retrieval concurrent and condensing after the fact was rejected — retrieval needs the condensed query as input, so this is a genuine sequential dependency, not just an implementation convenience.
- Duplicating a second condensation chain/prompt inside `orchestrator.py` was rejected in favor of sharing `RagChain`'s existing chain, to avoid prompt drift between the two intent paths.
- Changing `RERANK_SCORE_CUTOFF` (`backend/retrieval/reranker.py`) to filter out the weak-relevance chunks at the reranker stage was considered and rejected for this change — D-002 requires evaluation evidence before threshold changes, which does not yet exist. Left as an explicit follow-up.

### Accepted Tradeoffs

- `orchestrator_node` loses the concurrency between `MemoryManager.load` and hybrid retrieval; every `multi`-intent turn now pays for a serialized memory load, a `utility_llm` condensation call, and then retrieval, adding latency (same tradeoff class already accepted in D-007).
- `context.engineer.assemble` is now called twice in `orchestrator_node` (once memory-only, once with retrieved chunks) to get a pre-retrieval `memory_context` for condensation; this duplicates short/long-term token counting work but involves no extra LLM or I/O call.

### Consequences

- `flow.md` must be updated to show `orchestrator_node`'s memory-load-then-retrieve order instead of the concurrent `asyncio.gather` it previously documented.
- `RERANK_SCORE_CUTOFF` remains an open follow-up: raising it requires an eval run per D-002 before any change is recorded.
- Any future third intent path that retrieves using conversation-dependent queries must also call `condense_query` rather than retrieving on raw `state["query"]`.

## D-017: Treat a structurally mismatched retrieved source as insufficient context, not just a different technology

- Date: 2026-08-18
- Status: Accepted
- Phase: Phase 3 follow-up (extends D-002, D-011)
- Scope: `backend/constants.py` (`BEHAVIORAL_GUARDRAILS`)
- Evidence: Live retrieval/rerank run against production DB in this session ("how to do to the same in java" -> condensed to "How do I reverse a list in Java?"), corpus query confirming no Java array/List/Collections reversal document exists, and user confirmation
- Rationale confidence: Verified

### Context

D-016 fixed `orchestrator_node` retrieving on an unresolved query. With that fixed and independently verified (condensed query confirmed correct: "How do I reverse a list in Java?"), the same class of bad answer still occurred: the reranker's top result was "Reverse Singly Linked List Java," a `LinkedList`-specific pointer-manipulation algorithm, when the user meant reversing a built-in `List`/array. Direct corpus inspection (132,981 documents) found no document answering Java array/List/Collections reversal at all — this is a genuine corpus gap, not a retrieval bug. `BEHAVIORAL_GUARDRAILS` line 7 only instructed the model to ignore a source from "a different technology or context," which both retrieved documents technically satisfy (same language: Java) despite answering a different data structure's algorithm.

A live check of `CrossEncoder.predict()` output also showed relevance scores of 5.9-7.6 for this query, versus 0.13-0.62 seen for an unrelated query in D-016's investigation — confirming the reranker returns unbounded raw logits, not a normalized [0,1] probability. This reinforces D-002's requirement for evaluation evidence before setting `RERANK_SCORE_CUTOFF`; a fixed threshold picked without calibration would be meaningless across queries.

### Decision

Extend `BEHAVIORAL_GUARDRAILS`' existing "different technology or context" rule to explicitly cover a different underlying data structure, algorithm, or API within the same technology (for example a linked list when the user means an array or built-in list type). The model should treat such a source as insufficient context and say so, rather than answering with the mismatched structure. This is a single-file prompt-text change shared by `SYSTEM_PROMPT` and `ORCHESTRATOR_PROMPT` and does not touch retrieval, reranking, or the graph.

### Alternatives Considered

- Setting `RERANK_SCORE_CUTOFF` to filter out these scores was rejected again here for the same reason as D-016: the score is an uncalibrated raw logit, and this session's evidence (5.9-7.6 here vs 0.13-0.62 in D-016) shows a fixed cutoff cannot be chosen without an eval run establishing what the scale means across queries.
- Ingesting or authoring a corpus document specifically answering Java array/List reversal was rejected as out of scope for a guardrail fix; it is a Phase 3.1 ingestion-pipeline decision, not a prompt or retrieval-code change.

### Accepted Tradeoffs

- The guardrail change is judgment delegated to the answer LLM at generation time; it does not prevent a structurally mismatched document from being retrieved or reranked highly, only from being used to answer with confidence. A genuinely borderline case (e.g. an array-backed vs. linked implementation of the same abstract interface) could still be judged either way by the model.
- Relies on prompt-following rather than a deterministic code check, consistent with how the existing "different technology" rule and D-011's guardrail heuristics already operate at this layer.

### Consequences

- Retrieval and eval work on `RERANK_SCORE_CUTOFF` must establish a query-normalized or percentile-based threshold, not a single fixed logit value, given the score-scale variance observed here and in D-016.
- Any future ingestion work that fills this Java array/List reversal corpus gap should be recorded as its own ingestion decision, not folded into this one.

### Addendum (2026-08-19): empirically insufficient, superseded in direction by D-019

Direct replay of `Orchestrator.merge` against the exact repro chunk, with the shipped guardrail text, still produced the `LinkedList` answer with a citation. Retested with a stronger worded version (explicit worked example matching this exact case) at both temperature 1.0 and 0.2 — same failure both times. The guardrail line as implemented does not reliably prevent the failure mode it was written for; it remains in the prompt (harmless, may help on less contested cases) but must not be treated as a fix. Investigated per-document and batched LLM-classifier gates as a runtime alternative (see session transcript); found unreliable at batch sizes beyond ~3 documents and, per web/prior-art research (Corrective RAG, LangChain `EmbeddingsFilter`/`LLMChainFilter` docs), not how production RAG systems solve this regardless — real fix is calibrating `RERANK_SCORE_CUTOFF` with actual evaluation evidence, tracked as D-019.

## D-018: Replace in-memory BM25 with Postgres tsvector for sparse retrieval

- Date: 2026-08-19
- Status: Accepted
- Phase: Phase 3.5
- Scope: `backend/retrieval/fulltext_retriever.py` (new, replaces `bm25_retriever.py`), `backend/retrieval/tokenize.py` (new), `backend/retrieval/hybrid.py`, `backend/db/schema.sql`, `backend/ingestion/stackoverflow_loader.py`, `backend/scripts/backfill_pretokenized.py`
- Evidence: Code present on branch `feat/replace-bm25-with-postgres-tsvector`, and live-DB verification in this session (`chunks.content_pretokenized`/`content_tsv` columns present, 0 of 132,974 rows NULL — migration and backfill already applied and running, not merely written)
- Rationale confidence: Verified for what changed; Inferred for why `to_tsvector('simple', ...)` was chosen over an English-stemming config (no comment states it, but it is the coherent explanation: the custom `tokenize()` already does snake_case/camelCase splitting and stopword removal, and English stemming would mangle exact code-identifier matches that `simple` preserves)

### Context

D-008 accepted an in-memory, rebuilt-from-Postgres BM25 corpus (`BM25Retriever`) for sparse retrieval, with an explicitly recorded tradeoff: "The in-memory BM25 corpus consumes memory and is rebuilt from PostgreSQL" and "A future move to PostgreSQL full-text search or score fusion is a new decision." This is that decision.

### Decision

Move sparse scoring into Postgres itself: a generated `chunks.content_tsv tsvector` column (`GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content_pretokenized, ''))) STORED`) backed by a GIN index, scored at query time with `ts_rank_cd` against a `websearch_to_tsquery('simple', ...)`. A shared `retrieval/tokenize.py::tokenize()` (regex tokenizer, snake_case/camelCase splitting, small stopword list) is used both at ingestion time to populate `content_pretokenized` and at query time in `FullTextRetriever` before building the tsquery, so index-side and query-side tokenization stay identical by construction rather than by convention. `HybridRetriever.sparse` now holds a `FullTextRetriever` instead of `BM25Retriever`; `bm25_retriever.py` and the `rank-bm25` dependency are removed.

### Alternatives Considered

- Keeping in-memory BM25 was rejected per D-008's own recorded consequence — it doesn't scale and requires a rebuild step.
- Using Postgres's `english` text-search config directly on raw `content` (skipping the custom tokenizer and `content_pretokenized` column entirely) was rejected in favor of `simple` + a custom tokenizer, so code identifiers keep exact matchability instead of being run through English stemming.

### Accepted Tradeoffs

- `content_pretokenized` is a second stored copy of tokenized text per chunk (on top of `content` and the embedding) — extra storage, and every ingestion path must populate it or `content_tsv` silently generates from an empty string for those rows (the column is nullable with no ingestion-time enforcement).
- GIN index adds insert overhead per chunk row, traded against removing the BM25 in-memory rebuild cost.
- `FULLTEXT_MIN_SCORE` (default `0.0`) and `FULLTEXT_TOP_K` (default `20`) are new tunable knobs introduced by this retriever, currently unevaluated defaults — same caution as D-002/D-019 applies before changing them.

### Consequences

- `flow.md` section 5 (RAG path sequence diagram) referenced `BM25Retriever` by name; must be updated to `FullTextRetriever` + GIN-index lookup instead of in-memory corpus scoring.
- Any new ingestion path added later must populate `content_pretokenized`, not just `content`, or that path's chunks are invisible to sparse retrieval.
- Phase 3.5 moves from "external active work, absent from the implemented phase ledger" (per the prior `flow.md` Active Change entry) to implemented and verified as of this entry.

## D-019: Calibrate `RERANK_SCORE_CUTOFF` with `LLMContextPrecisionWithoutReference`/`LLMContextRecall` sweep

- Date: 2026-08-19
- Status: Proposed (sweep in progress — 2 of 5 planned cutoff values run as of this entry)
- Phase: Phase 3 follow-up (fulfills the evaluation requirement D-002 and D-017's addendum both deferred to)
- Scope: `backend/evals/ragas_runner.py`, `backend/retrieval/reranker.py` (`RERANK_SCORE_CUTOFF` value only, once chosen)
- Evidence: `experiments/ragas_run20260819_081122.json` (cutoff 0.0), `experiments/ragas_run20260819_085247.json` (cutoff 1.5); remaining cutoffs (3.0, 4.5, 6.0) not yet run
- Rationale confidence: Verified for the two data points collected so far; the eventual cutoff choice is not yet decided

### Context

D-002 required evaluation evidence before changing `RERANK_SCORE_CUTOFF`; D-016 and D-017's investigation (see D-017 addendum) confirmed the default `0.0` lets structurally-mismatched documents through and that a prompt-only guardrail can't reliably compensate. Web/prior-art research into how production RAG systems handle this (Corrective RAG, LangChain's `EmbeddingsFilter`) converged on: use the already-computed reranker score with a properly calibrated threshold, not additional LLM calls at request time.

### Decision

Sweep `RERANK_SCORE_CUTOFF` across `{0.0, 1.5, 3.0, 4.5, 6.0}` (range chosen from observed cross-encoder scores in this session, which ranged 0.13-7.67 across different queries), running `evals/ragas_runner.py --metrics context` (added in this session — restricts the RAGAS metric set to `LLMContextPrecisionWithoutReference` + `LLMContextRecall`, the two metrics that measure what this cutoff actually affects, since running the full 5-metric suite per sweep point was taking 2-3 hours per run) at `--sample-size 50` per cutoff. `ragas_runner.py` was also fixed to run its pipeline stage concurrently (`--pipeline-concurrency`, was fully sequential) and to record `rerank_score_cutoff`/`metrics` in each report for traceability. Final cutoff choice will be confirmed with a `--metrics all` run at the winning value once picked.

### Alternatives Considered (so far)

- Per-document/batched LLM relevance classifiers at request time — rejected, see D-017 addendum.
- A single global "safe-looking" cutoff picked from the 2-3 anecdotal score readings gathered earlier in this session — rejected; those readings already showed the score scale varies by query (0.13-0.62 vs 5.9-7.6), so a threshold needs to be chosen from a real sweep across many questions, not 1-2 examples.

### Accepted Tradeoffs (so far)

- `--metrics context` intentionally does not check whether the cutoff change affects `Faithfulness`/`AnswerRelevancy`/`AnswerCorrectness` during the sweep itself — accepted because those metrics judge generation, not retrieval, and are confirmed by the winning cutoff's final `--metrics all` run instead of on every sweep point.
- Early data (0.0 vs 1.5) is counter-intuitive: raising the cutoff to 1.5 *decreased* both `llm_context_precision_without_reference` (0.608 → 0.530) and `context_recall` (0.265 → 0.227), rather than trading recall for precision as naively expected. Noted here rather than acted on — only 2 of 5 points exist; this must not be read as a conclusion until the sweep completes.

### Consequences

- Do not treat this entry as resolved until all 5 sweep points are run and a cutoff is chosen; update Status to Accepted with the final value and full evidence at that point, per the counter-intuitive early trend noted above.
- `flow.md` Active Change must track this as in-progress work, including which retrieval-path files must stay unedited for the remainder of the sweep so all 5 points stay comparable.
