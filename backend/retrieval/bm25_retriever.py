import os
from typing import ClassVar, Any

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

load_dotenv()

BM25_TOP_K = int(os.environ.get("BM25_TOP_K", 20))


class BM25Retriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn_str: str
    k: int = BM25_TOP_K

    _bm25: ClassVar[BM25Okapi | None] = None
    _corpus: ClassVar[list[tuple[str, str]] | None] = None

    def _pg_dsn(self) -> str:
        return self.conn_str.replace("postgresql+psycopg://", "postgresql://", 1)

    def _load_index(self) -> None:
        if BM25Retriever._bm25 is not None:
            return

        with psycopg.connect(self._pg_dsn()) as conn:
            rows = conn.execute(
                "SELECT c.content, d.title FROM chunks c JOIN documents d ON c.document_id = d.id"
            ).fetchall()

            BM25Retriever._corpus = rows
            BM25Retriever._bm25 = BM25Okapi(
                [content.lower().split() for content, _ in rows]
            )

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        self._load_index()
        assert BM25Retriever._bm25 is not None and BM25Retriever._corpus is not None

        scores = BM25Retriever._bm25.get_scores(query.lower().split())

        sorted_scores = sorted(
            zip(scores, BM25Retriever._corpus, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )[: self.k]

        return [
            Document(
                page_content=content,
                metadata={"title": title, "bm25_score": float(score)},
            )
            for score, (content, title) in sorted_scores
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
