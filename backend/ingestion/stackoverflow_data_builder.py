import hashlib
import logging
import random
import warnings
from pathlib import Path
from typing import TypedDict
from itertools import chain

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_core.documents import Document
from tqdm import tqdm

from ingestion.kaggle_so_adapter import load_kaggle_rows
from ingestion.stackexchange_adapter import load_stackexchange_rows

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).resolve().parents[2] / "dataset" / "train.csv"
EVAL_IDS_PATH = Path(__file__).resolve().parents[2] / "dataset" / "eval_ids.txt"
KAGGLE_DIR = Path(__file__).resolve().parents[2] / "dataset" / "kaggle"
STACKEXCHANGE_DIR = Path(__file__).resolve().parents[2] / "dataset" / "stackexchange"


# ── Types ─────────────────────────────────────────────────────────────────────


class DocumentMetadata(TypedDict):
    source: str
    doc_id: str
    content_hash: str
    title: str
    body: str
    tags: list[str]
    quality: str | None
    score: int
    created_at: str


class SODatasetBuilder:
    """Loads, filters, and normalizes the Stack Overflow dataset into Documents."""

    QUALITY_KEEP = {"HQ", "LQ_EDIT"}
    EVAL_SAMPLE_N = 1000
    EVAL_RANDOM_STATE = 42

    def __init__(self, data_path: Path, eval_ids_path: Path) -> None:
        self.data_path = data_path
        self.eval_ids_path = eval_ids_path

    def build(self) -> list[Document]:
        train_df = self._load_and_filter()
        return self._build_documents(train_df)

    def sample_eval_holdout(self, stackexchange_dir: Path) -> set[str]:
        """Samples eval holdout IDs from the StackExchange batch only and writes them to
        eval_ids_path. Kaggle rows are never eligible — eval measures whether retrieval/
        generation reflects current technical content, and Kaggle's 2008-2016 coverage
        would test against stale question style, defeating that goal.

        Must be called (and eval_ids_path must exist) BEFORE build_from_sources(), so the
        holdout rows are excluded from ingestion — see _read_eval_ids/build_from_sources.
        This is a deliberate one-off call, not something build_from_sources triggers
        automatically: unlike the old build()/_load_and_filter(), which re-samples and
        overwrites eval_ids_path as a side effect on every call, sampling here only ever
        happens when this method is explicitly invoked.
        """
        rows = list(load_stackexchange_rows(stackexchange_dir))
        rng = random.Random(self.EVAL_RANDOM_STATE)
        holdout_ids = {row["doc_id"] for row in rng.sample(rows, k=self.EVAL_SAMPLE_N)}

        self.eval_ids_path.write_text("\n".join(sorted(holdout_ids)) + "\n")
        log.info("Saved %d eval IDs → %s", len(holdout_ids), self.eval_ids_path)
        return holdout_ids

    def _read_eval_ids(self) -> set[str]:
        if not self.eval_ids_path.exists():
            raise FileNotFoundError(
                f"{self.eval_ids_path} does not exist — call sample_eval_holdout() first "
                "so eval questions are excluded from ingestion, not the other way around."
            )
        return {
            line.strip()
            for line in self.eval_ids_path.read_text().splitlines()
            if line.strip()
        }

    def build_from_sources(
        self, kaggle_dir: Path, stackexchange_dir: Path
    ) -> list[Document]:
        eval_ids = self._read_eval_ids()
        docs: list[Document] = []

        for row in chain(
            load_kaggle_rows(kaggle_dir), load_stackexchange_rows(stackexchange_dir)
        ):
            if row["doc_id"] in eval_ids:
                continue

            content = f"{row["title"]}\n\n{row['question_body']}\n\nAnswer:\n{row['answer_body']}"

            metadata: DocumentMetadata = {
                "source": "stackoverflow",
                "doc_id": row["doc_id"],
                "content_hash": self._content_hash(content),
                "title": row["title"],
                "body": row["question_body"],
                "tags": row["tags"],
                "quality": None,
                "score": row["score"],
                "created_at": row["created_at"],
            }

            docs.append(Document(page_content=content, metadata=metadata))

        return docs

    def _load_and_filter(self) -> pd.DataFrame:
        log.info("Loading dataset from %s", self.data_path)
        df = pd.read_csv(self.data_path)
        log.info("Loaded %d rows", len(df))

        quality_filtered_df = df[df["Y"].isin(self.QUALITY_KEEP)]
        log.info(
            "After quality filter: %d rows\n%s",
            len(quality_filtered_df),
            quality_filtered_df["Y"].value_counts().to_string(),
        )

        high_quality_df = quality_filtered_df[quality_filtered_df["Y"] == "HQ"]
        eval_holdout_df = high_quality_df.sample(
            n=self.EVAL_SAMPLE_N, random_state=self.EVAL_RANDOM_STATE
        )
        eval_holdout_df["Id"].to_csv(self.eval_ids_path, index=False, header=False)
        log.info("Saved %d eval IDs → %s", len(eval_holdout_df), self.eval_ids_path)

        train_df = quality_filtered_df[
            ~quality_filtered_df["Id"].isin(eval_holdout_df["Id"])
        ]
        log.info("Train set: %d rows", len(train_df))
        return train_df

    def _build_documents(self, df: pd.DataFrame) -> list[Document]:
        log.info("Building documents...")
        docs: list[Document] = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing HTML"):
            plain_text_body = self._strip_html(str(row["Body"]))
            content = f"{row['Title']}\n\n{plain_text_body}"
            doc_metadata: DocumentMetadata = {
                "source": "stackoverflow",
                "doc_id": str(row["Id"]),
                "content_hash": self._content_hash(content),
                "title": str(row["Title"]),
                "body": plain_text_body,
                "tags": str(row["Tags"]).split(),
                "quality": str(row["Y"]),
                "created_at": str(row["CreationDate"]),
            }
            docs.append(Document(page_content=content, metadata=doc_metadata))
        log.info("Built %d documents", len(docs))
        return docs

    def eval_holdout_questions(self, stackexchange_dir: Path) -> list[dict[str, str]]:
        """Returns the held-out eval questions with ground truth, sourced only from the
        StackExchange batch (never Kaggle — see sample_eval_holdout). Each row's accepted
        answer is included as `answer`, enabling reference-based ragas metrics.
        """
        eval_ids = self._read_eval_ids()
        return [
            {"id": row["doc_id"], "title": row["title"], "answer": row["answer_body"]}
            for row in load_stackexchange_rows(stackexchange_dir)
            if row["doc_id"] in eval_ids
        ]

    @staticmethod
    def _strip_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for code_block in soup.find_all("code"):
            code_block.string = f"\n```\n{code_block.get_text()}\n```\n"
        return soup.get_text(separator="\n").strip()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
