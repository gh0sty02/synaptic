import os
from typing import ClassVar, Any
import re

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

load_dotenv()

BM25_TOP_K = int(os.environ.get("BM25_TOP_K", 20))
BM25_MIN_SCORE = float(os.environ.get("BM25_MIN_SCORE", 20))
import re

_STOPWORDS = frozenset(
    {
        "the",
        "is",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "and",
        "or",
        "on",
        "with",
        "this",
        "that",
        "it",
        "be",
        "as",
        "are",
        "was",
        "were",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for part in raw.split("_"):  # snake_case -> sub-parts
            if not part:
                continue
            for sub in _CAMEL_BOUNDARY_RE.sub(" ", part).split() or [
                part
            ]:  # camelCase -> sub-parts
                lowered = sub.lower()
                if lowered and lowered not in _STOPWORDS:
                    tokens.append(lowered)
    return tokens


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

            # corpus is just a list of (content, title)
            BM25Retriever._corpus = rows
            BM25Retriever._bm25 = BM25Okapi([_tokenize(content) for content, _ in rows])

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        self._load_index()
        assert BM25Retriever._bm25 is not None and BM25Retriever._corpus is not None

        scores = BM25Retriever._bm25.get_scores(_tokenize(query))

        sorted_scores = sorted(
            (
                (score, doc)
                for score, doc in zip(scores, BM25Retriever._corpus, strict=True)
                if score > BM25_MIN_SCORE
            ),
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
