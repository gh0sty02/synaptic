CREATE EXTENSION IF NOT EXISTS vector;

-- ── Ingestion Status ──────────────────────────────────────────────────────────

CREATE TYPE ingestion_status AS ENUM ('pending', 'indexed', 'failed');

-- ── Documents ─────────────────────────────────────────────────────────────────
-- Source-of-truth for original documents. Metadata stored once here,
-- not duplicated across chunks.

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL           PRIMARY KEY,
    source          TEXT                NOT NULL,
    external_id     TEXT                NOT NULL UNIQUE,
    title           TEXT,
    body            TEXT,
    content_hash    TEXT                NOT NULL UNIQUE,
    tags            TEXT[],
    quality         TEXT,
    created_at      TIMESTAMP,
    ingested_at     TIMESTAMP           DEFAULT NOW(),
    status          ingestion_status    NOT NULL DEFAULT 'pending',
    ingestion_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents (content_hash);

-- Partial indexes for retry worker — only scan rows that need attention
CREATE INDEX IF NOT EXISTS idx_documents_pending
    ON documents (id)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_documents_failed
    ON documents (id)
    WHERE status = 'failed';

-- ── Chunks ────────────────────────────────────────────────────────────────────
-- Semantic retrieval units. Currently 1-to-1 with documents.
-- chunk_index is ready for multi-chunk documents (answers, sections) later.

CREATE TABLE IF NOT EXISTS chunks (
    id                BIGSERIAL   PRIMARY KEY,
    document_id       BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index       INT         NOT NULL DEFAULT 0,
    content           TEXT        NOT NULL,
    content_hash      TEXT        NOT NULL UNIQUE,
    embedding         vector(768),
    embedding_model   TEXT        NOT NULL,
    embedding_version TEXT        NOT NULL,
    created_at        TIMESTAMP   DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks (document_id);

-- content_hash index is implicit from the UNIQUE constraint above

-- HNSW index is managed by the ingestion script (dropped before bulk load,
-- rebuilt after). Not defined here to avoid conflicts on re-runs.
-- CREATE INDEX idx_chunks_embedding ON chunks
--     USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 200);

-- ── Memory Summaries ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS memory_summaries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL,
    summary         TEXT NOT NULL,
    embedding       VECTOR(768),
    turn_count      INTEGER NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    relevance_score FLOAT DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS memory_summaries_embedding_idx
    ON memory_summaries
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS memory_summaries_session_idx
    ON memory_summaries (session_id, created_at);


ALTER TABLE documents ADD COLUMN score INTEGER;