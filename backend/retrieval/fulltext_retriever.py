import os
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from retrieval.tokenize import tokenize

load_dotenv()

FULLTEXT_TOP_K = int(os.environ.get("FULLTEXT_TOP_K", 20))
FULLTEXT_MIN_SCORE = float(
    os.environ.get("FULLTEXT_MIN_SCORE", "0.0")
)  # re-tuned in Commit 6


class FullTextRetriever(BaseRetriever):
    """
    Sparse retrieval via Postgres tsvector + GIN, scored with ts_rank_cd

    Async only - mirrors HybridRetriever's contract with the retriever it replaces.
    No in-process index: every query hits the GIN index
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn_str: str
    k: int = FULLTEXT_TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError(
            "Full Text Retriever is async only - Ragchain only calls ainvoke"
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        tokenized_query = " ".join(tokenize(query))

        async with await psycopg.AsyncConnection.connect(self.conn_str) as conn:
            curr = await conn.execute(
                """
                SELECT c.content, d.title, 
                    ts_rank_cd(c.content_tsv, query) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id,
                    websearch_to_tsquery('simple', %s) query
                WHERE c.content_tsv @@ query
                    AND ts_rank_cd(c.content_tsv, query) > %s
                    
                ORDER BY score DESC
                LIMIT %s
                """,
                (tokenized_query, FULLTEXT_MIN_SCORE, self.k),
            )

            rows = await curr.fetchall()

            return [
                Document(
                    page_content=content,
                    metadata={"title": title, "fulltext_score": float(score)},
                )
                for content, title, score in rows
            ]
