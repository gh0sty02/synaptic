# Synaptic

A local-first, multi-agent RAG assistant with memory engineering and observability, built on FastAPI, LangGraph, pgvector, and Redis.

Also a hands-on learning project: an excuse to get industry-depth, hands-on experience with the tools real AI engineering teams use — LangChain, LangGraph, hybrid retrieval, context engineering, evals, and tracing — while shipping something a user could actually query.

---

## What It Does

Synaptic answers questions grounded in an answer-bearing Stack Overflow knowledge base (~134k documents built from Kaggle + StackExchange Data Explorer exports — every chunk is `title + question + accepted/best answer`, not just a bare question) while maintaining memory across sessions. Each query is condensed against conversation history, optionally expanded via HyDE, retrieved with hybrid dense (pgvector) + sparse (Postgres full-text search) retrieval, reranked with a cross-encoder, and answered by an LLM through a LangGraph multi-agent pipeline before streaming back to the UI. An input/output guardrail classifier sits ahead of the graph.

Everything runs against a local, OpenAI-compatible model endpoint — no data leaves the machine.

---

## Architecture

```
User Query (Next.js UI)
    │
    ▼
Guardrail (input heuristic + LLM classifier, fail-closed)
    │
    ▼
Triage Agent  ──► RAG Agent ──► query condensation → HyDE (optional) → hybrid search (dense + Postgres full-text) → rerank → generate
                └──► Memory Agent ──► Redis (short-term) + pgvector (long-term summaries)
                └──► Orchestrator (multi-intent: condenses query against history, then retrieves)
    │
    ▼
Writer Node
    │
    ▼
Context Engineer + Optimiser (per-agent token budgets, LLMLingua compression on overflow)
    │
    ▼
Streamed response (OpenAI-compatible SSE, windowed output guardrail check)
    │
    ▼
Langfuse (per-node trace)
```

All agents call the same model through one OpenAI-compatible endpoint (`LLM_MODEL` / `LLM_BASE_URL`), differentiated by system prompt and tool access rather than by separate model deployments — swapping models is a config change, not a code change. Embeddings run locally via `nomic-embed-text-v1.5` (sentence-transformers).

---

## Tech Stack

| Layer               | Technology                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| Backend             | FastAPI (Python 3.11+), async SSE streaming, OpenAI-compatible `/v1/chat/completions` |
| Agent orchestration | LangGraph (stateful graph, Redis checkpointing via `AsyncRedisSaver`)          |
| LLM                 | Any OpenAI-compatible endpoint (local model server) — one model, prompt-differentiated agents |
| Embeddings          | nomic-embed-text-v1.5 via `langchain-huggingface` (local)                      |
| Vector DB           | pgvector on PostgreSQL (HNSW index, cosine similarity)                         |
| Dense retrieval     | pgvector cosine search (`retrieval/chunks_retriever.py`)                       |
| Sparse retrieval    | Postgres full-text search — generated `tsvector`/GIN column on `chunks`, scored with `ts_rank_cd` (`retrieval/fulltext_retriever.py`). Replaced an in-memory, rebuilt-on-startup BM25 index. |
| Hybrid fusion       | Dense + sparse candidate union, cross-encoder reranked (`retrieval/hybrid.py`) |
| Query expansion     | HyDE — hypothetical document embedding (`retrieval/hyde.py`), gated by `use_hyde` |
| Reranker            | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)                   |
| Context engineering | Per-agent token budgets + LLMLingua compression on overflow (`context/engineer.py`, `context/optimiser.py`) |
| Short-term memory   | Redis (sliding window, ~10 turns, 24h TTL)                                     |
| Long-term memory    | pgvector (session summaries, linear relevance decay)                           |
| Evals               | RAGAS — faithfulness, answer relevancy, context precision/recall, answer correctness (reference-based) |
| Observability       | Langfuse (LangChain `CallbackHandler`, per-node `@observe` spans)              |
| Frontend            | Next.js + TypeScript + shadcn/radix (SSE streaming via the OpenAI SDK)         |

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- An OpenAI-compatible local LLM server (e.g. Unsloth Studio, Ollama, LM Studio, vLLM) reachable at `LLM_BASE_URL`

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/gh0sty02/synaptic.git
cd synaptic
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

```env
# LLM Provider (OpenAI-compatible)
LLM_MODEL=your-model-name
LLM_BASE_URL=http://localhost:8000
LLM_API_KEY=your-api-key

# Database
CONN_STR=postgresql://user:password@localhost:5432/dbname

# Vector Store
CHUNK_COLLECTION_NAME=your-collection-name
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_VERSION=your-embedding-version
REBUILD_INDEX_THRESHOLD=1000

# Retrieval
RETRIEVAL_CANDIDATE_K=100
RETRIEVAL_TOP_K=10
RETRIEVAL_SCORE_CUTOFF=0.35
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Observability (Langfuse)
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3. Start infrastructure

```bash
docker-compose up -d
```

Starts: PostgreSQL + pgvector, Redis.

### 4. Run database migrations

```bash
psql "$CONN_STR" -f backend/db/schema.sql
```

### 5. Ingest data

Place the source CSVs under `dataset/kaggle/` and `dataset/stackexchange/`, then:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

Builds the combined Kaggle + StackExchange corpus (~134,000 answer-grounded documents), embeds, and bulk-loads into pgvector.

### 6. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 7. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## API

| Method   | Endpoint                | Description                                       |
| -------- | ------------------------ | -------------------------------------------------- |
| `POST`   | `/v1/chat/completions`   | Stream a response (OpenAI-compatible SSE), guardrail-checked on input and output |
| `POST`   | `/ingest`                | Trigger dataset ingestion                          |
| `GET`    | `/health`                | Service health check                               |
| `GET`    | `/sessions/{id}`         | Get session history — **currently broken**, calls the loader with no session ID; see [Roadmap](#roadmap) 3.2 |
| `DELETE` | `/sessions/{id}`         | End session (archive → summarise long-term memory) |
| `GET`    | `/metrics`               | Query count, error count, average latency          |

### POST /v1/chat/completions

```json
{
  "session_id": "optional-uuid",
  "model": "your-model-name",
  "messages": [{ "role": "user", "content": "What is the difference between useEffect and useLayoutEffect?" }],
  "use_hyde": false
}
```

Streams standard OpenAI chat-completion chunks.

---

## Project Structure

```
synaptic/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── main.py                   # FastAPI app, OpenAI-compatible SSE endpoint
│   ├── requirements.txt
│   ├── agents/                   # LangGraph graph + agent nodes (guardrail, triage, rag_agent,
│   │                              # memory_node, orchestrator, writer_node)
│   ├── chain/                    # RAG chain (condensation → HyDE → retrieve → generate)
│   ├── context/                  # Per-agent token budgets + LLMLingua compression
│   ├── guardrails/                # Input heuristics + LLM classifier, output streaming check
│   ├── memory/                   # Short-term (Redis) + long-term (pgvector)
│   ├── retrieval/                # Dense, Postgres full-text sparse, hybrid fusion, cross-encoder reranker, HyDE
│   ├── ingestion/                # Kaggle + StackExchange adapters, shared dataset builder
│   ├── evals/                    # RAGAS eval runner, guardrail eval runner
│   └── db/                       # schema.sql
├── frontend/
│   ├── app/                      # Next.js app router (chat, auth, profile)
│   └── components/               # Chat UI, app shell/sidebar, auth/profile (shadcn/radix)
├── dataset/                      # Source CSVs (not committed)
└── experiments/                  # Retrieval experiment results
```

---

## Roadmap

| Phase | Focus                                                              | Status                    |
| ----- | ------------------------------------------------------------------- | -------------------------- |
| 1     | RAG pipeline end-to-end (ingestion → retrieval → streaming)        | Done (Next.js UI, `/v1/chat/completions` instead of the originally specced `/chat`) |
| 1.2   | Retrieval quality — over-fetch + cross-encoder reranking           | Done                        |
| 2     | LangGraph multi-agent graph + memory engineering                   | Done                        |
| 3     | Context engineering, hybrid search, HyDE, RAGAS evals              | Core done — Langfuse budget-decision logging shipped; reference-based RAGAS metrics (`AnswerCorrectness`) shipped in `ragas_runner.py`'s default metric set |
| 3.1   | Answer-grounded Stack Overflow dataset (Kaggle + StackExchange)    | Mostly done — `ragas_runner.py` reference metrics remain |
| 3.2   | Conversation history — listable, resumable past sessions           | Not started on the backend. Frontend UI shell exists (sidebar, session list) but reads mock data from `frontend/lib/mock.ts`, and the one real backend piece (`GET /sessions/{id}`) is currently broken. Real design in progress — needs a client-persisted session ID (currently regenerated on every page load) and a decision on session ownership given no auth exists yet. |
| 3.5   | Sparse retrieval: Postgres full-text search, replacing in-memory BM25 | Done — generated `tsvector`/GIN column on `chunks`, shared tokenizer for index and query time |
| 4     | Guardrail classifier — input injection/jailbreak/off-topic + output streaming check | Done — input heuristic + LLM classifier ahead of triage (fail-closed on classifier error), windowed output check during streaming |
| 4.5   | Metadata filters, tool use, KV/prompt cache, API fixes              | Unblocked now that Phase 4 has shipped, not started            |
| 5     | Full Langfuse observability, memory depth, UI polish (citations, latency) | Partially done — chat UI shell (sidebar, citations, warning banner, theme toggle) built, citations component currently reads mock data |
| 6     | OAuth authentication (Google + GitHub)                              | Not started on the backend. Frontend has a sign-in page (Google/GitHub buttons, email/password form) that validates client-side and redirects — no backend call, since no auth endpoint exists |
| 7     | Tests, RAGAS eval harness, CI                                       | Not started — no test files in the repo             |
| 8     | Multimodal image input                                              | Not started                 |
| 9     | Infra and VPS deployment                                            | Not started                 |

---

## Eval Results

| Configuration            | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Answer Correctness |
| ------------------------ | ------------------ | --------------- | ------------ | ----------------- | ------------------- |
| Dense-only                | 0.542               | 0.171            | 0.658         | 0.585              | 0.258                |
| + Hybrid search (BM25)    | 0.617               | 0.288            | 0.747         | 0.555              | 0.270                |
| + HyDE                    | —                   | —                | —             | —                  | —                    |

Dense-only vs hybrid: same 20 held-out questions, same judge model. `n=20`,
single run each, not yet averaged over repeats — directionally reliable (recall gain exceeds
measured run-to-run noise) but not a tight confidence interval. Provisional, still in
development. HyDE not yet measured. The hybrid row predates the BM25 → Postgres full-text
migration (Roadmap 3.5) — not yet rerun against the new sparse retriever.

_The RAGAS runner exists (`backend/evals/ragas_runner.py`, `--dense-only` flag for
retriever-ablation runs, `--metrics context` to run only the two retrieval-relevant metrics,
`--workers`/`--pipeline-concurrency` to parallelize a run)._

### Reranker cutoff calibration

Swept `RERANK_SCORE_CUTOFF` (the cross-encoder reranker's score floor, default `0.0`, unset) against `n=50`
held-out questions:

| Cutoff | Context Precision | Context Recall | Questions with zero retrieved docs |
| ------ | ------------------ | --------------- | ----------------------------------- |
| 0.0    | 0.608               | 0.265            | 18%                                  |
| 1.5    | 0.530               | 0.227            | 28%                                  |
| 3.0    | 0.406               | 0.167            | 48%                                  |
| 4.5    | —                   | —                | 80%                                  |
| 6.0    | —                   | —                | 94%                                  |

Raising the cutoff doesn't trade recall for precision as naively expected — both drop, because the
cutoff deletes retrieval outright for a growing share of queries rather than selectively rejecting
worse documents (confirmed via a zero-doc-count check across all 5 values; the last two weren't run
through the full RAGAS metrics once that was clear). Conclusion: cutoff stays at its default. The
cross-encoder score is an unbounded raw logit whose useful range varies by query — not a threshold
that can be picked from a handful of examples.

Also found and fixed in the same pass: the multi-intent path (`orchestrator_node`) was retrieving on
the raw, unresolved follow-up question ("how do I do the same in java") instead of resolving it
against conversation history first, unlike the single-intent RAG path. And a citation guardrail
("don't cite a source describing a different data structure/algorithm than the question implies")
that tested as unreliable against one configured model and reliable against another — model-dependent,
not something to assume holds. Both shipped in [PR #6](https://github.com/gh0sty02/synaptic/pull/6);
full writeup with evidence lands in `decisions.md` once that PR merges (the engineering-log entries
for this work exist locally but aren't in `main`'s copy of the file yet).

---

## Swapping Models

All agents read the model from `LLM_MODEL`/`LLM_BASE_URL` at startup — pointing these at a different OpenAI-compatible endpoint (a larger local model, or a hosted API) requires no code changes.
