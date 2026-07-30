from typing import Any, NotRequired, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings

from constants import SYSTEM_PROMPT
from context.engineer import AGENT_BUDGET, fit_chunks
from ingestion.stackoverflow_loader import CONN_STR
from llm import main_llm, utility_llm
from retrieval.bm25_retriever import BM25Retriever
from retrieval.chunks_retriever import ChunksRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.hyde import generate_hypothetical_answer


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


_CONDENSATION_SYSTEM_PROMPT = (
    "Rewrite the follow-up question as a standalone question by replacing any "
    "pronouns or vague references with the specific topic from the conversation history.\n\n"
    "Example:\n"
    "History: Human: How do I sort a list in Python?\n"
    "Follow-up: How do I do the same in Ruby?\n"
    "Rewritten: How do I sort a list in Ruby?\n\n"
    "Output only the rewritten question, nothing else."
)


class RagChain:
    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self.embeddings = embeddings
        dense = ChunksRetriever(
            embeddings=self.embeddings,
            conn_str=CONN_STR,
            rerank=False,
        )
        sparse = BM25Retriever(conn_str=CONN_STR)
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

    def _build_condensation_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _CONDENSATION_SYSTEM_PROMPT),
                (
                    "human",
                    "Conversation History:\n{memory_context}\n\nFollow-up Question: {question}",
                ),
            ]
        )
        return prompt | utility_llm | StrOutputParser()

    def build(self) -> Any:
        _NO_CONTEXT_REPLY = "I couldn't find relevant information for your question."
        _condensation_chain = self._build_condensation_chain()
        _llm_chain = self.prompt | main_llm | StrOutputParser()

        async def _condense(inputs: _RagPipelineState) -> _RagPipelineState:
            if not inputs.get("memory_context", "").strip():
                return {
                    **inputs,
                    "retrieval_question": inputs["question"],
                    "condensed_query": inputs["question"],
                }

            human_history = "\n".join(
                line
                for line in inputs["memory_context"].splitlines()
                if line.startswith("Human:")
            )
            condensed = (
                await _condensation_chain.ainvoke(
                    {
                        "memory_context": human_history,
                        "question": inputs["question"],
                    }
                )
            ).strip() or inputs["question"]

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
