import argparse
import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from chain.rag_chain import RagChain, _RagPipelineState
from ingestion.stackoverflow_data_builder import (
    DATA_PATH,
    EVAL_IDS_PATH,
    STACKEXCHANGE_DIR,
    SODatasetBuilder,
)
from ingestion.stackoverflow_loader import CONN_STR
from llm import judge_llm
from retrieval.chunks_retriever import ChunksRetriever
from retrieval.reranker import RERANK_SCORE_CUTOFF

REPORT_DIR = Path(__file__).resolve().parents[2] / "experiments"


async def _run_pipeline(
    rag_chain: Runnable[_RagPipelineState, _RagPipelineState],
    questions: list[dict[str, str]],
    concurrency: int,
) -> list[dict[str, str | list[str]]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one(q: dict[str, str]) -> dict[str, str | list[str]]:
        async with semaphore:
            result = await rag_chain.ainvoke(
                {"question": q["title"], "memory_context": ""}
            )
        return {
            "user_input": q["title"],
            "response": result.get("answer", ""),
            "retrieved_contexts": [
                doc.page_content for doc in result.get("source_documents", [])
            ],
            "reference": q["answer"],
        }

    return list(await asyncio.gather(*(_run_one(q) for q in questions)))


"""

get the git commit hash so the eval report can be correlated with the changes during which the eval ran. each eval result will be different as new changes are added and may or may not improve the eval output
"""


def _get_commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


_METRIC_SETS = {
    "all": [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
        AnswerCorrectness(),
    ],
    # Retrieval-quality only. Faithfulness/AnswerRelevancy/AnswerCorrectness judge the
    # generated answer, not what the reranker cutoff actually affects (which chunks make
    # it into context) -- irrelevant to a cutoff sweep and the most expensive metrics
    # (multi-step claim decomposition), so skipping them cuts sweep time substantially.
    "context": [
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ],
}


async def main(
    sample_size: int, dense_only: bool, metrics: str, workers: int, pipeline_concurrency: int
) -> None:
    builder = SODatasetBuilder(DATA_PATH, EVAL_IDS_PATH)
    questions = builder.eval_holdout_questions(STACKEXCHANGE_DIR)[:sample_size]

    embeddings = HuggingFaceEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    rag_chain_builder = RagChain(embeddings)
    if dense_only:
        rag_chain_builder.retriever = ChunksRetriever(
            embeddings=embeddings, conn_str=CONN_STR, rerank=True
        )
    rag_chain = rag_chain_builder.build()

    rows = await _run_pipeline(
        rag_chain=rag_chain, questions=questions, concurrency=pipeline_concurrency
    )

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset,
        metrics=_METRIC_SETS[metrics],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=embeddings,
        run_config=RunConfig(max_workers=workers, timeout=300),
    )

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = (
        REPORT_DIR / f"ragas_run{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "commit": _get_commit_hash(),
                "sample_size": sample_size,
                "hybrid_search_enabled": "false" if dense_only else "true",
                "rerank_score_cutoff": RERANK_SCORE_CUTOFF,
                "metrics": metrics,
                "scores": result.to_pandas().mean(numeric_only=True).to_dict(),
            },
            indent=2,
        )
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of held-out questions to evaluate (default 50; full set is 1000)",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Bypass hybrid retrieval, use ChunksRetriever directly (dense-vs-hybrid comparison)",
    )
    parser.add_argument(
        "--metrics",
        choices=list(_METRIC_SETS),
        default="all",
        help="'all' (default, original 5 metrics) or 'context' (LLMContextPrecisionWithoutReference "
        "+ LLMContextRecall only -- the two metrics a RERANK_SCORE_CUTOFF sweep actually needs, "
        "and far cheaper since it skips Faithfulness/AnswerCorrectness's multi-step claim checks)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="RAGAS RunConfig max_workers (concurrent judge_llm calls during metric scoring). "
        "Default 2 matches prior behavior; raise cautiously and watch for rate-limit errors.",
    )
    parser.add_argument(
        "--pipeline-concurrency",
        type=int,
        default=5,
        help="Concurrent RagChain.ainvoke calls when building the eval dataset (was fully "
        "sequential before). Bounded by a semaphore, not by RAGAS's own worker count.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            args.sample_size,
            args.dense_only,
            args.metrics,
            args.workers,
            args.pipeline_concurrency,
        )
    )
