import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from retrieval.bm25_retriever import BM25Retriever
from retrieval.chunks_retriever import RETRIEVAL_TOP_K, ChunksRetriever
from retrieval.reranker import rerank

load_dotenv()

RRF_K = int(os.environ.get("RRF_k", "60"))


def reciprocal_rank_fusion(
    result_lists: list[list[Document]], *, k: int = RRF_K
) -> list[Document]:
    ranked: dict[str, tuple[float, Document]] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            prev_score, kept_doc = ranked.get(doc.page_content, (0.0, doc))
            ranked[doc.page_content] = (prev_score + 1.0 / (k + rank), kept_doc)

    fused = sorted(ranked.values(), key=lambda entry: entry[0], reverse=True)
    return [doc for _, doc in fused]


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

        fused = reciprocal_rank_fusion([dense_docs, sparse_docs])

        return rerank(query, fused, self.top_n)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        raise NotImplementedError(
            "Hybrid Retriever is async only - Ragchain only calls ainvoke."
        )
