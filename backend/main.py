from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, SecretStr
from typing import Literal, Optional
import time
import uuid
import json
import os
import asyncio
import logging
import re

import redis.asyncio as aioredis
import asyncpg
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from openai import APIConnectionError

from agents.graph import graph_builder, SynapticState, REDIS_URL
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.manager import MemoryManager
from ingestion.stackoverflow_loader import EMBEDDING_MODEL, CONN_STR
from constants import SYSTEM_PROMPT
import agents.nodes.orchestrator as orch_module
import agents.nodes.memory_node as mem_module
import agents.nodes.writer_node as writer_module
import agents.nodes.rag_agent as rag_agent_module

LLM_MODEL = os.environ["LLM_MODEL"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]

logger = logging.getLogger(__name__)
app_graph = None
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
    finish_reason: Optional[str] = None,
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
    global app_graph

    redis_client = aioredis.from_url(REDIS_URL)
    db_url = CONN_STR.replace("postgresql+psycopg2://", "postgresql://").replace("postgresql+psycopg://", "postgresql://")
    db_pool = await asyncpg.create_pool(db_url)

    _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    rag_agent_module.init(_embeddings)

    async def embed_fn(text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _embeddings.embed_query, text)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=SecretStr(LLM_API_KEY),
        temperature=1.0,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    manager = MemoryManager(ShortTermMemory(redis_client), LongTermMemory(db_pool, embed_fn), llm)
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
    session_id: Optional[str] = None


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
    await mem_module.memory_manager.delete_session(session_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    query = extract_query_from_messages(request.messages)
    session_id = request.session_id or str(uuid.uuid4())
    langfuse_trace_id = make_langfuse_trace_id(session_id)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_at = int(time.time())

    initial_state: SynapticState = {
        "session_id": session_id,
        "query": query,
        "messages": [],
        "intent": "",
        "active_agents": [],
        "metadata_filters": {},
        "system_prompt": SYSTEM_PROMPT,
        "short_term_memory": [],
        "long_term_memory": [],
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
    }

    langfuse_handler = CallbackHandler(trace_context={"trace_id": langfuse_trace_id})
    config = {"configurable": {"thread_id": session_id}, "callbacks": [langfuse_handler]}

    async def generate():
        try:
            # Role announcement — required by OpenAI SDK before any content
            yield chat_completion_chunk(
                completion_id,
                created_at,
                request.model,
                {"role": "assistant", "content": ""},
            )

            result = await app_graph.ainvoke(initial_state, config=config)
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
                    {"content": "Sorry, something went wrong while generating the response."},
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
