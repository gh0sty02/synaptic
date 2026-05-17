# Synaptic

A multi-agent AI assistant with RAG, memory engineering, and full observability. Built on FastAPI, LangGraph, Unsloth Studio (Gemma 4), pgvector, and Redis.

---

## What It Does

Synaptic answers questions grounded in two indexed knowledge bases — Stack Overflow (software engineering) and cricket match data — while maintaining memory across sessions. Each query passes through a guardrail classifier, a triage agent that routes intent, specialist agents (RAG or memory), and a context optimiser before a streamed response reaches the UI.

---

## Architecture

```
User Query (React UI)
    │
    ▼
Guardrail Classifier  ──blocked──► rejection message
    │ passed
    ▼
Triage Agent  ──► RAG Agent ──► hybrid search → reranker → generate
                └──► Memory Agent ──► Redis + pgvector summaries
                └──► Orchestrator (multi-intent only)
    │
    ▼
Context Optimiser (token budget + compression)
    │
    ▼
Streamed response + citations
    │
    ▼
Langfuse (full trace per request)
```

All agents use **Gemma 4 E4B** served by Unsloth Studio via an OpenAI-compatible API. Embeddings run locally via `nomic-ai/nomic-embed-text-v1.5` (sentence-transformers).

---

## Tech Stack

| Layer               | Technology                                                                |
| ------------------- | ------------------------------------------------------------------------- |
| Backend             | FastAPI (Python 3.11+), async SSE streaming                               |
| Agent orchestration | LangGraph 0.2+ (stateful graph, Redis checkpointing)                      |
| LLM                 | Unsloth Studio — Gemma 4 E4B (Q5_K_M), OpenAI-compatible API              |
| Embeddings          | nomic-embed-text-v1.5 via sentence-transformers (local)                   |
| Vector DB           | pgvector on PostgreSQL 16 (HNSW index, cosine similarity)                 |
| Short-term memory   | Redis 7 (sliding window, 10 turns, 24h TTL)                               |
| Long-term memory    | pgvector (session summaries with relevance decay)                         |
| KV cache            | Redis (prompt cache keyed on query hash)                                  |
| Sparse retrieval    | rank-bm25 (in-memory, rebuilt on startup)                                 |
| Reranker            | cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-transformers)              |
| Context compression | LangChain summarise chain (map-reduce)                                    |
| Observability       | Langfuse (self-hosted, LangChain CallbackHandler)                         |
| Evals               | RAGAS (context_precision, context_recall, faithfulness, answer_relevancy) |
| Frontend            | React + TypeScript (SSE streaming, citation cards, memory indicator)      |

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose
- [Unsloth Studio](https://unsloth.ai/studio) running with Gemma 4 E4B model loaded and API enabled

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

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

```env
# Unsloth Studio — OpenAI-compatible API
LLM_BASE_URL=http://localhost:2242/v1
LLM_API_KEY=your-unsloth-api-key
LLM_MODEL=unsloth/gemma-4-E4B-it

# Database
POSTGRES_URL=postgresql://synaptic:synaptic_local@localhost:5432/synaptic

# Redis
REDIS_URL=redis://localhost:6379

# Langfuse (get keys from http://localhost:3000 after docker-compose up)
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

### 3. Start infrastructure

```bash
docker-compose up -d
```

Starts: PostgreSQL + pgvector, Redis, Langfuse.

### 4. Run database migrations

```bash
psql $POSTGRES_URL -f backend/migrations/init.sql
```

### 5. Ingest data

```bash
# Stack Overflow (~45,000 chunks, ~15–25 min on CPU)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "stackoverflow", "file_path": "data/stackoverflow_60k.csv"}'

# Cricket dataset
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "cricket", "file_path": "data/cricket.csv"}'
```

### 6. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 7. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## API

| Method   | Endpoint         | Description                             |
| -------- | ---------------- | --------------------------------------- |
| `POST`   | `/chat`          | Stream a response (SSE)                 |
| `POST`   | `/ingest`        | Trigger dataset ingestion               |
| `GET`    | `/health`        | Service health check                    |
| `GET`    | `/sessions/{id}` | Get session history                     |
| `DELETE` | `/sessions/{id}` | End session (summarise → store → clear) |
| `GET`    | `/metrics`       | Cache hit rate, avg latency, doc count  |

### POST /chat

```json
{
  "session_id": "optional-uuid",
  "query": "What is the difference between useEffect and useLayoutEffect?",
  "stream": true
}
```

**SSE response:**

```
data: {"type": "token", "content": "The "}
data: {"type": "citation", "doc_id": "abc123", "title": "useEffect vs useLayoutEffect", "source": "stackoverflow"}
data: {"type": "meta", "agent": "rag_agent", "latency_ms": 1840, "cache_hit": false}
data: {"type": "done"}
```

---

## Project Structure

```
synaptic/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── main.py                   # FastAPI app, SSE streaming
│   ├── requirements.txt
│   ├── agents/                   # LangGraph graph + agent nodes
│   ├── chains/                   # RAG, summary, compression chains
│   ├── context/                  # Token budgeting + context assembly
│   ├── memory/                   # Short-term (Redis) + long-term (pgvector)
│   ├── retrieval/                # Dense, BM25, hybrid, reranker, HyDE
│   ├── guardrails/               # Input classifier
│   ├── cache/                    # Redis KV prompt cache
│   ├── tools/                    # metadata_filter_tool, document_loader_tool
│   ├── ingestion/                # SO + cricket data loaders
│   └── evals/                    # RAGAS eval runner
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── components/           # ChatWindow, CitationCard, MemoryIndicator
│       └── hooks/useStream.ts
├── data/                         # CSV datasets (not committed)
├── experiments/                  # Retrieval experiment results
├── scripts/
│   └── pull_models.sh
└── docs/
    ├── ARCHITECTURE.md
    ├── DATASET.md
    └── phase-1 through phase-5 build guides
```

---

## Build Phases

| Phase | Focus                                                             | Status      |
| ----- | ----------------------------------------------------------------- | ----------- |
| 1     | RAG pipeline end-to-end (ingestion → retrieval → streaming)       | In progress |
| 2     | LangGraph multi-agent graph + memory engineering                  | Planned     |
| 3     | Hybrid search, reranking, HyDE, context optimisation, RAGAS evals | Planned     |
| 4     | Guardrails, tool use, cricket dataset, KV cache                   | Planned     |
| 5     | Full Langfuse observability, memory depth, UI polish              | Planned     |

---

## Eval Results

| Configuration                 | Context Precision | Context Recall | Faithfulness | Answer Relevancy |
| ----------------------------- | ----------------- | -------------- | ------------ | ---------------- |
| Phase 1 baseline (dense only) | —                 | —              | —            | —                |
| + Hybrid search               | —                 | —              | —            | —                |
| + Reranking                   | —                 | —              | —            | —                |
| Final (all optimisations)     | —                 | —              | —            | —                |

_Results populated after Phase 3._

---

## Swapping to a Larger Model

Change one env var:

```env
LLM_MODEL=unsloth/gemma-4-26B-A4B-it
```

No code changes required — all agents read from `LLM_MODEL` at startup.
