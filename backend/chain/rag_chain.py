from typing import Any, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings

from constants import SYSTEM_PROMPT
from context.engineer import AGENT_BUDGET, fit_chunks
from ingestion.stackoverflow_loader import CONN_STR
from llm import main_llm
from retrieval.fulltext_retriever import FullTextRetriever
from retrieval.chunks_retriever import ChunksRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import generate_hypothetical_answer
from retrieval.query_condensation import condense_query


class _RagPipelineState(TypedDict):
    question: str
    memory_context: str
    use_hyde: NotRequired[bool]
    retrieval_question: NotRequired[str]
    condensed_query: NotRequired[str]
    source_documents: NotRequired[list[Document]]
    answer: NotRequired[str]
    context: NotRequired[str]
    chunks_tokens: NotRequired[int]
    chunks_truncated: NotRequired[bool]


class RagChain:
    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self.embeddings = embeddings
        dense = ChunksRetriever(
            embeddings=self.embeddings,
            conn_str=CONN_STR,
            rerank=False,
        )
        sparse = FullTextRetriever(conn_str=CONN_STR)
        self.retriever = HybridRetriever(dense=dense, sparse=sparse)
        self.prompt = self._build_prompt()

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Previous Conversation:\n{memory_context}\n\nCurrent Question: {question}\n\nContext:\n{context}",
                ),
            ]
        )

    async def _hyde(self, inputs: _RagPipelineState) -> _RagPipelineState:
        if not inputs.get("use_hyde", False):
            return inputs
        hypothetical = await generate_hypothetical_answer(
            inputs.get("retrieval_question", inputs["question"])
        )

        return {**inputs, "retrieval_question": hypothetical}

    def build(self) -> Any:
        _NO_CONTEXT_REPLY = "I couldn't find relevant information for your question."
        _llm_chain = (self.prompt | main_llm | StrOutputParser()).with_config(
            {"tags": ["final_answer"]}
        )

        async def _condense(inputs: _RagPipelineState) -> _RagPipelineState:
            condensed = await condense_query(
                inputs["question"], inputs.get("memory_context", "")
            )
            return {
                **inputs,
                "retrieval_question": condensed,
                "condensed_query": condensed,
            }

        async def _retrieve(inputs: _RagPipelineState) -> _RagPipelineState:
            docs = await self.retriever.ainvoke(
                inputs.get("retrieval_question", inputs["question"])
            )
            return {**inputs, "source_documents": docs}

        async def _answer(inputs: _RagPipelineState) -> _RagPipelineState:
            docs = inputs.get("source_documents", [])
            if not docs:
                return {
                    **inputs,
                    "context": "",
                    "answer": _NO_CONTEXT_REPLY,
                    "chunks_tokens": 0,
                    "chunks_truncated": False,
                }

            context, chunks_tokens, chunks_truncated = fit_chunks(
                docs, AGENT_BUDGET["rag_agent"]["retrieved_chunks"]
            )
            answer = await _llm_chain.ainvoke(
                {
                    "question": inputs["question"],
                    "memory_context": inputs["memory_context"],
                    "context": context,
                }
            )
            return {
                **inputs,
                "answer": answer,
                "context": context,
                "chunks_tokens": chunks_tokens,
                "chunks_truncated": chunks_truncated,
            }

        return (
            RunnableLambda(_condense)
            | RunnableLambda(self._hyde)
            | RunnableLambda(_retrieve)
            | RunnableLambda(_answer)
        )
