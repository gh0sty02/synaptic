from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import time

import uuid
import json

from chain.rag_chain import RagChain
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()


app = FastAPI(title="Synaptic")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True


def extract_query_from_messages(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content

    from fastapi import HTTPException

    raise HTTPException(status_code=422, detail="No user message found in messages")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = RagChain().build()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    query = extract_query_from_messages(request.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_at = int(time.time())

    async def generate():
        # Role announcement — required by OpenAI SDK before any content
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_at, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

        async for chunk in rag_chain.astream(query, config={"callbacks": [langfuse_handler]}):
            print(chunk)
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_at, 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"

        # Stop chunk
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_at, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

        # OpenAI SDK stop sentinel
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# @app.post("/chat")
# async def chat(request: Request):
#     body = await request.json()

#     session_id = body.get("id", str(uuid.uuid4()))

#     messages = body.get("messages", [])

#     query = extract_user_message(messages)

#     async def generate():
#         # handler = CallbackHandler(
#         #     trace_name="rag-chat",
#         #     user_id=session_id,
#         # )

#         async for chunk in rag_chain.astream(
#             query,
#             # config={"callbacks": [handler]},
#         ):
#             escaped = json.dumps(chunk)

#             # AI SDK text stream protocol
#             yield f"0:{escaped}\n"

#         yield 'd:{"finishReason":"stop"}\n'

#     return StreamingResponse(
#         generate(),
#         media_type="text/plain",
#     )
