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

REPORT_DIR = Path(__file__).resolve().parents[2] / "experiments"


async def _run_pipeline(
    rag_chain: Runnable[_RagPipelineState, _RagPipelineState],
    questions: list[dict[str, str]],
) -> list[dict[str, str | list[str]]]:
    rows: list[dict[str, str | list[str]]] = []

    for q in questions:
        result = await rag_chain.ainvoke({"question": q["title"], "memory_context": ""})

        rows.append(
            {
                "user_input": q["title"],
                "response": result.get("answer", ""),
                "retrieved_contexts": [
                    doc.page_content for doc in result.get("source_documents", [])
                ],
                "reference": q["answer"],
            }
        )

    return rows


"""

get the git commit hash so the eval report can be correlated with the changes during which the eval ran. each eval result will be different as new changes are added and may or may not improve the eval output
"""


def _get_commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


async def main(sample_size: int, dense_only: bool) -> None:
    builder = SODatasetBuilder(DATA_PATH, EVAL_IDS_PATH)
    questions = builder.eval_holdout_questions(STACKEXCHANGE_DIR)[:sample_size]

    embeddings = HuggingFaceEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    rag_chain_builder = RagChain(embeddings)
    if dense_only:
        rag_chain_builder.retriever = ChunksRetriever(
            embeddings=embeddings, conn_str=CONN_STR, rerank=True
        )
    rag_chain = rag_chain_builder.build()

    rows = await _run_pipeline(rag_chain=rag_chain, questions=questions)

    dataset = Dataset.from_list(rows)

    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithoutReference(),
            LLMContextRecall(),
            AnswerCorrectness(),
        ],
        llm=LangchainLLMWrapper(judge_llm),
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=300),
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
    args = parser.parse_args()
    asyncio.run(main(args.sample_size, args.dense_only))
