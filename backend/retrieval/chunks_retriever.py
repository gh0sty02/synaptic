from typing import Any, ClassVar
import numpy as np
import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from pydantic import ConfigDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from dotenv import load_dotenv
import os
from sentence_transformers import CrossEncoder

load_dotenv()

RETRIEVAL_SCORE_CUTOFF = float(os.environ.get("RETRIEVAL_SCORE_CUTOFF", "0.35"))
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "3"))
_DISTANCE_CUTOFF = 1 - RETRIEVAL_SCORE_CUTOFF  # <=> returns distance, not similarity


class ChunksRetriever(BaseRetriever):
    """Retrieves chunks via pgvector cosine similarity against our custom schema.

    We own the schema instead of using LangChain's PGVector store. Tradeoffs:

    Gains:
    - documents/chunks split with FK — lets us JOIN document metadata (title, tags,
      quality) without duplicating it across every chunk row
    - content_hash deduplication and status tracking (pending/indexed/failed) enable
      resumable ingestion; LangChain's store has no built-in dedup or retry state
    - COPY-based bulk insert + optional HNSW index drop/rebuild for throughput
    - Full control over index params (m, ef_construction) and vacuum strategy

    Costs:
    - Must implement BaseRetriever manually (sync + async paths shown below)
    - No built-in MMR, similarity-score threshold, or metadata filter — those would
      need to be added to the SQL query by hand
    - register_vector() must be called per connection instead of being handled by
      the store abstraction
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: HuggingFaceEmbeddings
    encoder: ClassVar[CrossEncoder] = CrossEncoder(
        model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    conn_str: str
    k: int = 100

    def _pg_dsn(self) -> str:
        return self.conn_str.replace("postgresql+psycopg://", "postgresql://", 1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        vec = np.array(self.embeddings.embed_query(query))
        with psycopg.connect(self._pg_dsn()) as conn:
            register_vector(conn)
            rows = conn.execute(
                """
                SELECT c.content, d.title
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE (c.embedding <=> %s) < %s
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (vec, _DISTANCE_CUTOFF, vec, self.k),
            ).fetchall()

            print(rows)

            re_ranked = self.encoder.predict([(query, row[0]) for row in rows])

            scored = sorted(zip(re_ranked, rows), key=lambda x: x[0], reverse=True)

            top = [(score, r) for score, r in scored[:RETRIEVAL_TOP_K]]
        return [
            Document(
                page_content=r[0],
                metadata={"title": r[1], "relevance_score": float(score)},
            )
            for score, r in top
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Any
    ) -> list[Document]:
        vec = np.array(self.embeddings.embed_query(query))
        async with await psycopg.AsyncConnection.connect(self._pg_dsn()) as conn:
            await register_vector_async(conn)
            cur = await conn.execute(
                """
                SELECT c.content, d.title
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE (c.embedding <=> %s) < %s
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (vec, _DISTANCE_CUTOFF, vec, self.k),
            )
            rows = await cur.fetchall()

            re_ranked = self.encoder.predict([(query, row[0]) for row in rows])

            scored = sorted(zip(re_ranked, rows), key=lambda x: x[0], reverse=True)

            top = [(score, r) for score, r in scored[:RETRIEVAL_TOP_K]]
        return [
            Document(
                page_content=r[0],
                metadata={"title": r[1], "relevance_score": float(score)},
            )
            for score, r in top
        ]
