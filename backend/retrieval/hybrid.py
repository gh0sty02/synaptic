import asyncio
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from retrieval.bm25_retriever import BM25Retriever
from retrieval.chunks_retriever import RETRIEVAL_TOP_K, ChunksRetriever
from retrieval.reranker import rerank

load_dotenv()


def merge_candidates(
    dense_docs: list[Document], sparse_docs: list[Document]
) -> list[Document]:
    merged: dict[str, Document] = {}
    for doc in dense_docs + sparse_docs:
        merged.setdefault(doc.page_content, doc)
    return list(merged.values())


class HybridRetriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense: ChunksRetriever
    sparse: BM25Retriever

    top_n: int = RETRIEVAL_TOP_K

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        dense_docs, sparse_docs = await asyncio.gather(
            self.dense.ainvoke(query), self.sparse.ainvoke(query)
        )

        fused = merge_candidates(dense_docs, sparse_docs)

        return rerank(query, fused, self.top_n)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        raise NotImplementedError(
            "Hybrid Retriever is async only - Ragchain only calls ainvoke."
        )
