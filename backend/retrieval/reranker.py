import os

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANK_SCORE_CUTOFF = float(os.environ.get("RERANK_SCORE_CUTOFF", "0.0"))

_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _encoder


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:

    if not docs:
        return []

    scores = _get_encoder().predict([(query, doc.page_content) for doc in docs])

    scores_sorted = sorted(
        zip(scores, docs, strict=True), key=lambda x: x[0], reverse=True
    )

    return [
        doc.model_copy(
            update={"metadata": {**doc.metadata, "relevance_score": float(score)}}
        )
        for score, doc in scores_sorted
        if score >= RERANK_SCORE_CUTOFF
    ][:top_k]
