import asyncpg
from typing import Any


class LongTermMemory:
    TOP_K = 3

    def __init__(self, pool: asyncpg.Pool, embed_fn: Any):

        self._pool = pool
        self._embed = embed_fn

    async def load(self, query: str) -> list[dict[str, Any]]:
        """
        Embed the currenct query, cosine search session_summaries filtered by user_id, return top k most sementaically similar past sessins
        """

        query_vec = await self._embed(query)

        # convert the python list into string as pg understands strings and not python lists
        vec_str = f"[{','.join(str(v) for v in query_vec)}]"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, summary,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM memory_summaries
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_str,
                self.TOP_K,
            )

        return [
            {
                "session_id": r["session_id"],
                "summary": r["summary"],
                "similarity": r["similarity"],
            }
            for r in rows
        ]

    async def save_summary(
        self, session_id: str, summary: str, turn_count: int
    ) -> None:

        embedding = await self._embed(summary)
        vector_str = f"[{','.join(str(v) for v in embedding)}]"

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_summaries (session_id, summary, embedding, turn_count)
                VALUES ($1, $2, $3::vector, $4)
                ON CONFLICT DO NOTHING
                """,
                session_id,
                summary,
                vector_str,
                turn_count,
            )
