import json
import tiktoken
import redis.asyncio as aioredis
from datetime import datetime, timezone
from typing import Any, cast

ENCODING = tiktoken.get_encoding("cl100k_base")


class ShortTermMemory:
    MAX_TURNS = 10       # pairs → up to 20 list entries
    TTL_SECONDS = 86400  # 24-hour inactivity window (ARCHITECTURE.md §5.3)
    TOKEN_BUDGET = 2000  # soft limit; oldest pairs dropped first if exceeded

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._r = redis_client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:turns"

    @staticmethod
    def _meta_key(session_id: str) -> str:
        return f"session:{session_id}:meta"

    async def append(
        self, session_id: str, role: str, content: str, agent: str = ""
    ) -> None:
        """Push one turn and reset inactivity TTL."""
        key = self._key(session_id)
        entry = json.dumps({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent":     agent,
        })
        async with self._r.pipeline() as pipe:
            pipe.rpush(key, entry)
            pipe.expire(key, self.TTL_SECONDS)
            await pipe.execute()

    async def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return last MAX_TURNS pairs. Drops oldest pairs if over TOKEN_BUDGET."""
        key = self._key(session_id)
        raw = cast(list[str], await self._r.lrange(key, -(self.MAX_TURNS * 2), -1))  # type: ignore[misc]
        turns: list[dict[str, Any]] = [json.loads(r) for r in raw]
        return self._truncate_to_budget(turns)

    def _truncate_to_budget(self, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop oldest human+assistant pairs until under TOKEN_BUDGET."""
        while turns:
            total = sum(len(ENCODING.encode(str(t["content"]))) for t in turns)
            if total <= self.TOKEN_BUDGET:
                break
            turns = turns[2:]
        return turns

    async def get_all_for_archival(self, session_id: str) -> list[dict[str, Any]]:
        """Fetch full history before archiving to long-term memory."""
        raw = cast(list[str], await self._r.lrange(self._key(session_id), 0, -1))  # type: ignore[misc]
        return [json.loads(r) for r in raw]

    async def delete(self, session_id: str) -> None:
        """Remove session from Redis after successful archival."""
        await self._r.delete(self._key(session_id), self._meta_key(session_id))
