import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import asyncpg
import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from openai import APIConnectionError
from pydantic import BaseModel

import agents.nodes.memory_node as mem_module
import agents.nodes.orchestrator as orch_module
import agents.nodes.rag_agent as rag_agent_module
import agents.nodes.writer_node as writer_module
from agents.graph import REDIS_URL, graph_builder
from agents.state import SynapticState
from constants import SYSTEM_PROMPT
from ingestion.stackoverflow_data_builder import (
    DATA_PATH,
    EVAL_IDS_PATH,
    KAGGLE_DIR,
    STACKEXCHANGE_DIR,
    SODatasetBuilder,
)
from ingestion.stackoverflow_loader import CONN_STR, EMBEDDING_MODEL, IngestionPipeline
from llm import utility_llm
from memory.long_term import LongTermMemory
from memory.manager import MemoryManager
from memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)
app_graph = None
memory_manager: MemoryManager | None = None
_metrics: dict[str, float] = {
    "total_queries": 0,
    "total_errors": 0,
    "_latency_sum_ms": 0.0,
}
langfuse_client = get_client()
trace = langfuse_client.create_trace_id()
LANGFUSE_TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
UPSTREAM_CONNECTION_ERROR_MESSAGE = (
    "I couldn't reach the configured language model service. "
    "Please check that the LLM server is running and that LLM_BASE_URL is reachable."
)


def make_langfuse_trace_id(session_id: str) -> str:
    normalized = session_id.replace("-", "").lower()
    if LANGFUSE_TRACE_ID_RE.fullmatch(normalized):
        return normalized

    return uuid.uuid5(uuid.NAMESPACE_URL, f"synaptic-session:{session_id}").hex


def chat_completion_chunk(
    completion_id: str,
    created_at: int,
    model: str,
    delta: dict,
    finish_reason: str | None = None,
) -> str:
    return f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_at, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"


def stream_done() -> str:
    return "data: [DONE]\n\n"


def is_upstream_connection_error(exc: Exception) -> bool:
    while exc is not None:
        if isinstance(exc, APIConnectionError):
            return True
        exc = exc.__cause__ or exc.__context__

    return False


@asynccontextmanager
async def lifespan(application: FastAPI):
    global app_graph, memory_manager

    redis_client = aioredis.from_url(REDIS_URL)
    db_url = CONN_STR.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )
    db_pool = await asyncpg.create_pool(db_url)

    _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    rag_agent_module.init(_embeddings)

    async def embed_fn(text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _embeddings.embed_query, text)

    manager = MemoryManager(
        ShortTermMemory(redis_client), LongTermMemory(db_pool, embed_fn), utility_llm
    )
    memory_manager = manager
    orch_module.memory_manager = manager
    mem_module.memory_manager = manager
    writer_module.memory_manager = manager

    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        await checkpointer.asetup()
        app_graph = graph_builder.compile(checkpointer=checkpointer)
        yield

    await redis_client.aclose()
    await db_pool.close()
    get_client().flush()


app = FastAPI(title="Synaptic", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    session_id: str | None = None
    use_hyde: bool = False


class IngestRequest(BaseModel):
    source: Literal["stackoverflow"] = "stackoverflow"
    limit: int = 45000


def extract_query_from_messages(messages: list[ChatMessage]) -> str:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return messages[i].content

    from fastapi import HTTPException

    raise HTTPException(status_code=422, detail="No user message found in messages")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    await mem_module.memory_manager.archive_session(session_id)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    turns = await mem_module.memory_manager.stm.load()
    return {"session_id": session_id, "turns": turns, "turn_count": len(turns)}


@app.get("/metrics")
def metrics():
    total = _metrics["total_queries"]
    avg_latency = (_metrics["_latency_sum_ms"] / total) if total else 0.0
    return {
        "total_queries": total,
        "total_errors": _metrics["total_errors"],
        "average_latency_ms": round(avg_latency, 2),
    }


@app.post("/ingest")
async def ingest(request: IngestRequest):
    def _run() -> int:
        docs = SODatasetBuilder(DATA_PATH, EVAL_IDS_PATH).build_from_sources(
            kaggle_dir=KAGGLE_DIR, stackexchange_dir=STACKEXCHANGE_DIR
        )
        if request.limit:
            docs = docs[: request.limit]
        IngestionPipeline(CONN_STR).run(docs)
        return len(docs)

    count = await asyncio.to_thread(_run)
    return {"status": "ok", "rows_ingested": count}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    query = extract_query_from_messages(request.messages)
    session_id = request.session_id or str(uuid.uuid4())
    langfuse_trace_id = make_langfuse_trace_id(session_id)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_at = int(time.time())

    langfuse_handler = CallbackHandler(trace_context={"trace_id": langfuse_trace_id})
    mem_result = await memory_manager.load(session_id, query)

    initial_state: SynapticState = {
        "session_id": session_id,
        "query": query,
        "messages": [],
        "intent": "",
        "active_agents": [],
        "metadata_filters": {},
        "system_prompt": SYSTEM_PROMPT,
        "short_term_memory": mem_result["short_term_memory"],
        "long_term_memory": mem_result["long_term_memory"],
        "retrieved_chunks": [],
        "tool_results": [],
        "token_counts": {},
        "total_tokens": 0,
        "budget_exceeded": False,
        "final_answer": "",
        "citations": [],
        "agent_scratchpad": "",
        "trace_id": langfuse_trace_id,
        "latency_ms": {},
        "langfuse_callbacks": [],
        "condensed_query": "",
        "use_hyde": request.use_hyde,
    }
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [langfuse_handler],
    }

    async def generate():
        start = time.monotonic()
        try:
            # Role announcement — required by OpenAI SDK before any content
            yield chat_completion_chunk(
                completion_id,
                created_at,
                request.model,
                {"role": "assistant", "content": ""},
            )

            result = await app_graph.ainvoke(initial_state, config=config)
            elapsed_ms = (time.monotonic() - start) * 1000
            _metrics["total_queries"] += 1
            _metrics["_latency_sum_ms"] += elapsed_ms
            answer = result.get("final_answer", "")

            if answer:
                yield chat_completion_chunk(
                    completion_id,
                    created_at,
                    request.model,
                    {"content": answer},
                )

            yield chat_completion_chunk(
                completion_id,
                created_at,
                request.model,
                {},
                finish_reason="stop",
            )
            yield stream_done()

        except Exception as exc:
            _metrics["total_errors"] += 1
            if is_upstream_connection_error(exc):
                logger.warning("LLM upstream connection failed", exc_info=True)
                yield chat_completion_chunk(
                    completion_id,
                    created_at,
                    request.model,
                    {"content": UPSTREAM_CONNECTION_ERROR_MESSAGE},
                )
            else:
                logger.error("Unhandled error in stream", exc_info=True)
                yield chat_completion_chunk(
                    completion_id,
                    created_at,
                    request.model,
                    {
                        "content": "Sorry, something went wrong while generating the response."
                    },
                )

            yield chat_completion_chunk(
                completion_id,
                created_at,
                request.model,
                {},
                finish_reason="stop",
            )
            yield stream_done()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
