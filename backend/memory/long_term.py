from typing import Any

import asyncpg


class LongTermMemory:
    TOP_K = 3
    DECAY_DAYS = 30.0

    def __init__(self, pool: asyncpg.Pool, embed_fn: Any):

        self._pool = pool
        self._embed = embed_fn

    async def load(self, query: str) -> list[dict[str, Any]]:
        """
        Embed the current query, cosine search session_summaries, return top k most semantically similar past sessions
        """

        query_vec = await self._embed(query)

        # convert the python list into string as pg understands strings and not python lists
        vec_str = f"[{','.join(str(v) for v in query_vec)}]"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, summary,
                    1 - (embedding <=> $1::vector) AS cosine_similarity,
                    EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0 AS days_old
                FROM memory_summaries
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_str,
                self.TOP_K * 3,
            )

        results = []
        for r in rows:
            days_old = float(r["days_old"])
            decay = min(1.0, max(0.0, 1.0 - days_old / self.DECAY_DAYS))
            results.append(
                {
                    "session_id": r["session_id"],
                    "summary": r["summary"],
                    "similarity": float(r["cosine_similarity"]) * decay,
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[: self.TOP_K]

    async def delete_by_session(self, session_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM memory_summaries WHERE session_id = $1",
                session_id,
            )

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
