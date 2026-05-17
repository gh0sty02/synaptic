from typing import Any
import numpy as np
import psycopg
from pgvector.psycopg import register_vector, register_vector_async
from pydantic import ConfigDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


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
    conn_str: str
    k: int = 5

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
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (vec, self.k),
            ).fetchall()
        return [Document(page_content=r[0], metadata={"title": r[1]}) for r in rows]

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
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (vec, self.k),
            )
            rows = await cur.fetchall()
        return [Document(page_content=r[0], metadata={"title": r[1]}) for r in rows]
