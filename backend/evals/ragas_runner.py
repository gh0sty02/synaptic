import argparse
import asyncio
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    Faithfulness,
)

from langchain_core.runnables import Runnable

from chain.rag_chain import RagChain, _RagPipelineState
from ingestion.stackoverflow_data_builder import (
    DATA_PATH,
    EVAL_IDS_PATH,
    SODatasetBuilder,
)

REPORT_DIR = Path(__file__).resolve().parents[2] / "experiments"


async def _run_pipeline(
    rag_chain: Runnable[_RagPipelineState, _RagPipelineState],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    rows = []

    for q in questions:
        result = await rag_chain.ainvoke({"question": q["title"], "memory_context": ""})

        rows.append(
            {
                "user_input": q["title"],
                "response": result.get("answer", ""),
                "retrieved_contexts": [
                    doc.page_content for doc in result.get("source_documents", [])
                ],
            }
        )

    return rows
