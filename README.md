# Synaptic

A local-first, multi-agent RAG assistant with memory engineering and observability, built on FastAPI, LangGraph, pgvector, and Redis.

Also a hands-on learning project: an excuse to get industry-depth, hands-on experience with the tools real AI engineering teams use — LangChain, LangGraph, hybrid retrieval, context engineering, evals, and tracing — while shipping something a user could actually query.

---

## What It Does

Synaptic answers questions grounded in an answer-bearing Stack Overflow knowledge base (~134k documents built from Kaggle + StackExchange Data Explorer exports — every chunk is `title + question + accepted/best answer`, not just a bare question) while maintaining memory across sessions. Each query is condensed against conversation history, optionally expanded via HyDE, retrieved with hybrid dense + BM25 search, reranked with a cross-encoder, and answered by an LLM through a LangGraph multi-agent pipeline before streaming back to the UI.

Everything runs against a local, OpenAI-compatible model endpoint — no data leaves the machine.

---

## Architecture

```
User Query (Next.js UI)
    │
    ▼
Triage Agent  ──► RAG Agent ──► query condensation → HyDE (optional) → hybrid search (dense + BM25/RRF) → rerank → generate
                └──► Memory Agent ──► Redis (short-term) + pgvector (long-term summaries)
                └──► Orchestrator (multi-intent only)
    │
    ▼
Writer Node
    │
    ▼
Context Engineer + Optimiser (per-agent token budgets, LLMLingua compression on overflow)
    │
    ▼
Streamed response (OpenAI-compatible SSE)
    │
    ▼
Langfuse (per-node trace)
```

A guardrail classifier (input-side injection/jailbreak/off-topic detection + output-side streaming check) sits ahead of `triage` as the next planned entry point — see [Roadmap](#roadmap).

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
| Sparse retrieval    | rank-bm25, in-memory, rebuilt on app startup (`retrieval/bm25_retriever.py`)   |
| Hybrid fusion       | Reciprocal Rank Fusion, dense + BM25 (`retrieval/hybrid.py`)                   |
| Query expansion     | HyDE — hypothetical document embedding (`retrieval/hyde.py`), gated by `use_hyde` |
| Reranker            | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)                   |
| Context engineering | Per-agent token budgets + LLMLingua compression on overflow (`context/engineer.py`, `context/optimiser.py`) |
| Short-term memory   | Redis (sliding window, ~10 turns, 24h TTL)                                     |
| Long-term memory    | pgvector (session summaries, linear relevance decay)                           |
| Evals               | RAGAS — faithfulness, answer relevancy, context precision (reference-based recall/correctness in progress) |
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
| `POST`   | `/v1/chat/completions`   | Stream a response (OpenAI-compatible SSE)          |
| `POST`   | `/ingest`                | Trigger dataset ingestion                          |
| `GET`    | `/health`                | Service health check                               |
| `GET`    | `/sessions/{id}`         | Get session history                                |
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
│   ├── agents/                   # LangGraph graph + agent nodes (triage, rag_agent,
│   │                              # memory_node, orchestrator, writer_node)
│   ├── chain/                    # RAG chain (condensation → HyDE → retrieve → generate)
│   ├── context/                  # Per-agent token budgets + LLMLingua compression
│   ├── memory/                   # Short-term (Redis) + long-term (pgvector)
│   ├── retrieval/                # Dense, BM25, hybrid (RRF), reranker, HyDE
│   ├── ingestion/                # Kaggle + StackExchange adapters, shared dataset builder
│   ├── evals/                    # RAGAS eval runner
│   └── db/                       # schema.sql
├── frontend/
│   ├── app/                      # Next.js app router
│   └── components/               # Chat UI (shadcn/radix)
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
| 3     | Context engineering, hybrid search (BM25 + RRF), HyDE, RAGAS evals | Core done — Langfuse budget-decision logging and reference-based RAGAS metrics remain |
| 3.1   | Answer-grounded Stack Overflow dataset (Kaggle + StackExchange)    | Mostly done — `ragas_runner.py` reference metrics remain |
| 3.2   | Conversation history — listable, resumable past sessions           | Not started                 |
| 4     | Guardrail classifier — input injection/jailbreak/off-topic + output streaming check | Planned, implementation-ready |
| 4.5   | Metadata filters, tool use, KV/prompt cache, API fixes              | Deferred until Phase 4 ships |
| 5     | Full Langfuse observability, memory depth, UI polish (citations, latency) | Partially done         |
| 6     | OAuth authentication (Google + GitHub)                              | Not started                 |
| 7     | Tests, RAGAS eval harness, CI                                       | Not started                 |
| 8     | Multimodal image input                                              | Not started                 |
| 9     | Infra and VPS deployment                                            | Not started                 |

---

## Eval Results

| Configuration                 | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
| ----------------------------- | ----------------- | --------------- | ------------ | ---------------- |
| Dense-only baseline           | —                  | —               | —            | —                 |
| + Hybrid search (BM25 + RRF)  | —                  | —               | —            | —                 |
| + HyDE                        | —                  | —               | —            | —                 |

_The RAGAS runner exists (`backend/evals/ragas_runner.py`); results are populated once the reference-based metrics (Phase 3.1) and a committed baseline (Phase 7) land._

---

## Swapping Models

All agents read the model from `LLM_MODEL`/`LLM_BASE_URL` at startup — pointing these at a different OpenAI-compatible endpoint (a larger local model, or a hosted API) requires no code changes.
