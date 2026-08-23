import json
import logging
import math
import os
import time
import warnings

import psycopg
from bs4 import XMLParsedAsHTMLWarning
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


from ingestion.stackoverflow_data_builder import (
    DATA_PATH,
    EVAL_IDS_PATH,
    KAGGLE_DIR,
    STACKEXCHANGE_DIR,
    DocumentMetadata,
    SODatasetBuilder,
)
from retrieval.tokenize import tokenize

load_dotenv(override=True)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────

CONN_STR = os.environ["CONN_STR"]
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBEDDING_VERSION = os.environ.get("EMBEDDING_VERSION", "1.5")
REBUILD_INDEX_THRESHOLD = int(os.environ.get("REBUILD_INDEX_THRESHOLD", "10000"))


def _meta(doc: Document) -> DocumentMetadata:
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
        "score",
    }
    _EXPECTED_CHUNKS_COLS = {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "content_hash",
        "content_pretokenized",
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
        new_docs, resumable_hash_to_doc_id, skip_count = self._filter_and_categorize(
            docs
        )
        resume_docs = [
            d for d in docs if _meta(d)["content_hash"] in resumable_hash_to_doc_id
        ]
        pending_docs = new_docs + resume_docs

        if not pending_docs:
            log.info("All %d docs already indexed — nothing to do", skip_count)
            return

        model = self._load_model()
        newly_inserted_hash_to_doc_id_mapping = self._insert_pending_documents(new_docs)
        content_hash_to_doc_id = {
            **newly_inserted_hash_to_doc_id_mapping,
            **resumable_hash_to_doc_id,
        }

        if resume_docs:
            self._cleanup_partial_chunks(list(resumable_hash_to_doc_id.values()))

        texts, vectors, metadatas = self._embed(pending_docs, model)
        texts, vectors, metadatas, invalid_doc_ids = self._drop_invalid_vectors(
            texts, vectors, metadatas, content_hash_to_doc_id
        )
        if invalid_doc_ids:
            self._mark_failed(invalid_doc_ids, "embedding produced NaN/Inf")

        doc_ids = [content_hash_to_doc_id[m["content_hash"]] for m in metadatas]

        try:
            self._insert_chunks(texts, vectors, metadatas, content_hash_to_doc_id)
            self._mark_indexed(doc_ids)
        except Exception as exc:
            log.error("Ingestion failed: %s", exc)
            self._mark_failed(doc_ids, str(exc))
            raise

    # ── Pre-filter ────────────────────────────────────────────────────────────

    def _filter_and_categorize(
        self, docs: list[Document]
    ) -> tuple[list[Document], dict[str, int], int]:
        all_content_hashes = [_meta(d)["content_hash"] for d in docs]

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

                db_docs_by_content_hash: dict[str, tuple[int, str]] = {
                    r[0]: (r[1], r[2])
                    for r in cur.execute(
                        "SELECT content_hash, id, status FROM documents WHERE content_hash = ANY(%s)",
                        (all_content_hashes,),
                    )
                }

        new_docs = [
            d for d in docs if _meta(d)["content_hash"] not in db_docs_by_content_hash
        ]
        resumable_hash_to_doc_id = {
            h: id_
            for h, (id_, status) in db_docs_by_content_hash.items()
            if status != "indexed"
        }
        skip_count = sum(
            1 for _, status in db_docs_by_content_hash.values() if status == "indexed"
        )

        log.info(
            "%d new | %d resuming (pending/failed) | %d already indexed",
            len(new_docs),
            len(resumable_hash_to_doc_id),
            skip_count,
        )
        return new_docs, resumable_hash_to_doc_id, skip_count

    # ── DB writes ─────────────────────────────────────────────────────────────

    def _insert_pending_documents(self, docs: list[Document]) -> dict[str, int]:
        if not docs:
            return {}

        metadatas: list[DocumentMetadata] = [_meta(d) for d in docs]
        self._validate_metadata(metadatas)

        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO documents
                        (source, external_id, title, body, content_hash, tags, quality, score, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            m["score"],
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
        log.info("Loading embedding model %s on %s...", self.embedding_model, "cuda")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"trust_remote_code": True, "device": "cuda"},
            encode_kwargs={"batch_size": 16, "normalize_embeddings": True},
        )
        embeddings._client = embeddings._client.half()  # type: ignore[union-attr]
        return embeddings

    def _embed(
        self, docs: list[Document], model: HuggingFaceEmbeddings
    ) -> tuple[list[str], list[list[float]], list[DocumentMetadata]]:
        texts = [doc.page_content for doc in docs]
        metadatas: list[DocumentMetadata] = [_meta(d) for d in docs]
        batch_size: int = model.encode_kwargs.get("batch_size", 32)
        vectors: list[list[float]] = []

        log.info("Embedding %d docs in batches of %d...", len(docs), batch_size)
        embed_start_time = time.time()
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            vectors.extend(model.embed_documents(texts[i : i + batch_size]))
        elapsed = time.time() - embed_start_time
        log.info("Done in %.1fs (%.1f docs/s)", elapsed, len(docs) / elapsed)

        return texts, vectors, metadatas

    def _drop_invalid_vectors(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[DocumentMetadata],
        content_hash_to_doc_id: dict[str, int],
    ) -> tuple[list[str], list[list[float]], list[DocumentMetadata], list[int]]:
        """Drops any (text, vector, metadata) whose vector contains NaN/Inf.

        pgvector rejects NaN/Inf outright, and COPY fails the entire batch on the
        first bad row — without this, one degenerate embedding (seen with fp16
        inference on an unusual input) aborts insertion for every row in that
        chunk_size batch, discarding the whole embedding pass. Filtering here
        means only the genuinely bad document(s) are lost, marked 'failed' with
        a clear reason, instead of every pending document.
        """
        clean_texts, clean_vectors, clean_metadatas, invalid_doc_ids = [], [], [], []

        for text, vector, meta in zip(texts, vectors, metadatas, strict=True):
            if any(math.isnan(x) or math.isinf(x) for x in vector):
                log.warning(
                    "Dropping doc_id=%s (title=%r) — embedding contains NaN/Inf",
                    meta["doc_id"],
                    meta["title"][:80],
                )
                invalid_doc_ids.append(content_hash_to_doc_id[meta["content_hash"]])
                continue
            clean_texts.append(text)
            clean_vectors.append(vector)
            clean_metadatas.append(meta)

        if invalid_doc_ids:
            log.warning(
                "Dropped %d documents with invalid embeddings", len(invalid_doc_ids)
            )

        return clean_texts, clean_vectors, clean_metadatas, invalid_doc_ids

    # ── Insert chunks ─────────────────────────────────────────────────────────

    def _insert_chunks(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[DocumentMetadata],
        content_hash_to_doc_id: dict[str, int],
    ) -> None:
        chunk_count = len(texts)
        rebuild = chunk_count >= self.rebuild_index_threshold

        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                if rebuild:
                    log.info("%d rows — dropping HNSW index for bulk load", chunk_count)
                    cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding;")
                    conn.commit()

                inserted_chunk_count, batch_insert_start_time = 0, time.time()

                for start in range(0, chunk_count, self.chunk_size):
                    batch_slice = slice(start, start + self.chunk_size)
                    with cur.copy(
                        "COPY chunks (document_id, chunk_index, content, content_hash, "
                        "content_pretokenized, embedding, embedding_model, embedding_version) FROM STDIN"
                    ) as copy:
                        for text, vector, meta in zip(
                            texts[batch_slice],
                            vectors[batch_slice],
                            metadatas[batch_slice],
                            strict=True,
                        ):
                            copy.write_row(
                                (
                                    content_hash_to_doc_id[meta["content_hash"]],
                                    0,
                                    text,
                                    meta["content_hash"],
                                    " ".join(tokenize(text)),
                                    str(vector),
                                    self.embedding_model,
                                    self.embedding_version,
                                )
                            )
                    conn.commit()
                    inserted_chunk_count += len(texts[batch_slice])

                elapsed = time.time() - batch_insert_start_time
                log.info(
                    "Inserted %d chunks in %.1fs (%.0f rows/s)",
                    inserted_chunk_count,
                    elapsed,
                    inserted_chunk_count / elapsed,
                )

                if rebuild:
                    log.info("Rebuilding HNSW index...")
                    index_rebuild_start_time = time.time()
                    cur.execute("""
                        CREATE INDEX idx_chunks_embedding
                        ON chunks
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 200)
                    """)
                    conn.commit()
                    log.info(
                        "Index built in %.1fs", time.time() - index_rebuild_start_time
                    )

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
    def _validate_metadata(metadatas: list[DocumentMetadata]) -> None:
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
    docs = SODatasetBuilder(DATA_PATH, EVAL_IDS_PATH).build_from_sources(
        kaggle_dir=KAGGLE_DIR, stackexchange_dir=STACKEXCHANGE_DIR
    )
    IngestionPipeline(CONN_STR).run(docs)


if __name__ == "__main__":
    main()
