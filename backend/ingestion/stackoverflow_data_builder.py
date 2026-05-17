import hashlib
import logging
import warnings
from typing import TypedDict
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm
from langchain_core.documents import Document

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


# ── Types ─────────────────────────────────────────────────────────────────────


class DocumentMetadata(TypedDict):
    source: str
    doc_id: str
    content_hash: str
    title: str
    body: str
    tags: list[str]
    quality: str
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

    @staticmethod
    def _strip_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for code_block in soup.find_all("code"):
            code_block.string = f"\n```\n{code_block.get_text()}\n```\n"
        return soup.get_text(separator="\n").strip()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


# print ls

print("he")
