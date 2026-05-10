import json
import logging
import os
import warnings
import time
import torch
import psycopg
from bs4 import XMLParsedAsHTMLWarning
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from stackoverflow_data_builder import (
    DocMeta,
    DATA_PATH,
    EVAL_IDS_PATH,
    SODatasetBuilder,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CONN_STR = "postgresql://synaptic:synaptic_local@localhost:5433/synaptic"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_VERSION = "1.5"
REBUILD_INDEX_THRESHOLD = 10_000


def _meta(doc: Document) -> DocMeta:
    return doc.metadata  # type: ignore[return-value]


class IngestionPipeline:
    """Embeds Documents and persists them into the documents + chunks tables."""

    _EXPECTED_DOCUMENTS_COLS = {
        "id",
        "source",
        "external_id",
        "title",
        "body",
        "content_hash",
        "tags",
        "quality",
        "created_at",
        "status",
    }
    _EXPECTED_CHUNKS_COLS = {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "content_hash",
        "embedding",
        "embedding_model",
        "embedding_version",
    }

    def __init__(
        self,
        conn_str: str,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_version: str = EMBEDDING_VERSION,
        rebuild_index_threshold: int = REBUILD_INDEX_THRESHOLD,
        chunk_size: int = 5000,
    ) -> None:
        self.conn_str = conn_str
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self.rebuild_index_threshold = rebuild_index_threshold
        self.chunk_size = chunk_size

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, docs: list[Document]) -> None:
        new_docs, resume_hash_to_id, skip_count = self._filter_and_categorize(docs)
        resume_docs = [d for d in docs if _meta(d)["content_hash"] in resume_hash_to_id]
        pending_docs = new_docs + resume_docs

        if not pending_docs:
            log.info("All %d docs already indexed — nothing to do", skip_count)
            return

        model = self._load_model()
        new_hash_to_id = self._insert_pending_documents(new_docs)
        hash_to_id = {**new_hash_to_id, **resume_hash_to_id}
        doc_ids = list(hash_to_id.values())

        if resume_docs:
            self._cleanup_partial_chunks(list(resume_hash_to_id.values()))

        texts, vectors, metadatas = self._embed(pending_docs, model)

        try:
            self._insert_chunks(texts, vectors, metadatas, hash_to_id)
            self._mark_indexed(doc_ids)
        except Exception as exc:
            log.error("Ingestion failed: %s", exc)
            self._mark_failed(doc_ids, str(exc))
            raise

    # ── Pre-filter ────────────────────────────────────────────────────────────

    def _filter_and_categorize(
        self, docs: list[Document]
    ) -> tuple[list[Document], dict[str, int], int]:
        all_hashes = [_meta(d)["content_hash"] for d in docs]

        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                row = cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'documents')"
                ).fetchone()
                if not row or not row[0]:
                    log.info(
                        "documents table not yet created — all %d docs are new",
                        len(docs),
                    )
                    return docs, {}, 0

                self._verify_schema(cur)

                existing: dict[str, tuple[int, str]] = {
                    r[0]: (r[1], r[2])
                    for r in cur.execute(
                        "SELECT content_hash, id, status FROM documents WHERE content_hash = ANY(%s)",
                        (all_hashes,),
                    )
                }

        new_docs = [d for d in docs if _meta(d)["content_hash"] not in existing]
        resume_hash_to_id = {
            h: id_ for h, (id_, status) in existing.items() if status != "indexed"
        }
        skip_count = sum(1 for _, status in existing.values() if status == "indexed")

        log.info(
            "%d new | %d resuming (pending/failed) | %d already indexed",
            len(new_docs),
            len(resume_hash_to_id),
            skip_count,
        )
        return new_docs, resume_hash_to_id, skip_count

    # ── DB writes ─────────────────────────────────────────────────────────────

    def _insert_pending_documents(self, docs: list[Document]) -> dict[str, int]:
        if not docs:
            return {}

        metadatas: list[DocMeta] = [_meta(d) for d in docs]
        self._validate_metadata(metadatas)

        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO documents
                        (source, external_id, title, body, content_hash, tags, quality, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    """,
                    [
                        (
                            m["source"],
                            m["doc_id"],
                            m["title"],
                            m["body"],
                            m["content_hash"],
                            m["tags"],
                            m["quality"],
                            m["created_at"],
                        )
                        for m in metadatas
                    ],
                )
                conn.commit()

                rows = cur.execute(
                    "SELECT id, content_hash FROM documents WHERE content_hash = ANY(%s)",
                    ([m["content_hash"] for m in metadatas],),
                ).fetchall()

        log.info("Inserted %d new documents (pending)", len(docs))
        return {row[1]: row[0] for row in rows}

    def _cleanup_partial_chunks(self, doc_ids: list[int]) -> None:
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE document_id = ANY(%s)", (doc_ids,)
                )
                deleted = cur.rowcount
                conn.commit()
        if deleted:
            log.info("Cleaned up %d orphaned chunks from previous failed run", deleted)

    def _mark_indexed(self, doc_ids: list[int]) -> None:
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = 'indexed', ingestion_error = NULL WHERE id = ANY(%s)",
                    (doc_ids,),
                )
                conn.commit()
        log.info("Marked %d documents as indexed", len(doc_ids))

    def _mark_failed(self, doc_ids: list[int], error: str) -> None:
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE documents SET status = 'failed', ingestion_error = %s WHERE id = ANY(%s)",
                    (error[:2000], doc_ids),
                )
                conn.commit()
        log.info("Marked %d documents as failed: %s", len(doc_ids), error[:120])

    # ── Embed ─────────────────────────────────────────────────────────────────

    def _load_model(self) -> HuggingFaceEmbeddings:

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        device = self._select_device()
        log.info("Loading embedding model %s on %s...", self.embedding_model, device)

        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"trust_remote_code": True, "device": device},
            encode_kwargs={"batch_size": 8},
        )

    @staticmethod
    def _select_device() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as e:
            log.warning("CUDA probe failed, falling back to CPU: %s", e)
            return "cpu"

    def _embed(
        self, docs: list[Document], model: HuggingFaceEmbeddings
    ) -> tuple[list[str], list[list[float]], list[DocMeta]]:
        texts = [doc.page_content for doc in docs]
        metadatas: list[DocMeta] = [_meta(d) for d in docs]
        batch_size: int = model.encode_kwargs.get("batch_size", 32)
        vectors: list[list[float]] = []

        log.info("Embedding %d docs in batches of %d...", len(docs), batch_size)
        t = time.time()
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            vectors.extend(model.embed_documents(texts[i : i + batch_size]))
        elapsed = time.time() - t
        log.info("Done in %.1fs (%.1f docs/s)", elapsed, len(docs) / elapsed)

        return texts, vectors, metadatas

    # ── Insert chunks ─────────────────────────────────────────────────────────

    def _insert_chunks(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[DocMeta],
        hash_to_id: dict[str, int],
    ) -> None:
        n = len(texts)
        rebuild = n >= self.rebuild_index_threshold

        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                if rebuild:
                    log.info("%d rows — dropping HNSW index for bulk load", n)
                    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding;")
                    conn.commit()

                total, t = 0, time.time()
                n_batches = (n + self.chunk_size - 1) // self.chunk_size

                for batch_idx, start in enumerate(range(0, n, self.chunk_size)):
                    sl = slice(start, start + self.chunk_size)
                    with cur.copy(
                        "COPY chunks (document_id, chunk_index, content, content_hash, "
                        "embedding, embedding_model, embedding_version) FROM STDIN"
                    ) as copy:
                        for text, vector, meta in zip(
                            texts[sl], vectors[sl], metadatas[sl]
                        ):
                            copy.write_row(
                                (
                                    hash_to_id[meta["content_hash"]],
                                    0,
                                    text,
                                    meta["content_hash"],
                                    str(vector),
                                    self.embedding_model,
                                    self.embedding_version,
                                )
                            )
                    conn.commit()
                    total += len(texts[sl])
                    log.info(
                        "Batch %d/%d committed — %d/%d rows",
                        batch_idx + 1,
                        n_batches,
                        total,
                        n,
                    )

                elapsed = time.time() - t
                log.info(
                    "Inserted %d chunks in %.1fs (%.0f rows/s)",
                    total,
                    elapsed,
                    total / elapsed,
                )

                if rebuild:
                    log.info("Rebuilding HNSW index...")
                    t = time.time()
                    cur.execute("""
                        CREATE INDEX idx_chunks_embedding
                        ON chunks
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                    """)
                    conn.commit()
                    log.info("Index built in %.1fs", time.time() - t)
                else:
                    log.info("Incremental insert — HNSW index updated in place")

    # ── Guards ────────────────────────────────────────────────────────────────

    def _verify_schema(self, cur: psycopg.Cursor) -> None:
        for table, expected in [
            ("documents", self._EXPECTED_DOCUMENTS_COLS),
            ("chunks", self._EXPECTED_CHUNKS_COLS),
        ]:
            actual = {
                row[0]
                for row in cur.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                    (table,),
                )
            }
            missing = expected - actual
            if missing:
                raise RuntimeError(
                    f"Table '{table}' schema mismatch — missing columns: {missing}. "
                    "Run latest schema.sql or pin the schema version."
                )

    @staticmethod
    def _validate_metadata(metadatas: list[DocMeta]) -> None:
        for i, meta in enumerate(metadatas):
            try:
                json.dumps(meta)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Non-JSON-serializable metadata at index {i} "
                    f"(doc_id={meta.get('doc_id')}): {exc}"
                ) from exc


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    docs = SODatasetBuilder(DATA_PATH, EVAL_IDS_PATH).build()
    IngestionPipeline(CONN_STR).run(docs)


if __name__ == "__main__":
    main()
