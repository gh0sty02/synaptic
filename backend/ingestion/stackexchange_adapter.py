from __future__ import annotations
import csv
import glob
import sys
import re
from datetime import datetime
from collections.abc import Iterator


from pathlib import Path
from typing import TypedDict

csv.field_size_limit(sys.maxsize)

MIN_ANSWER_SCORE = 1
_TAG_RE = re.compile(r"<([^>]+)>")


class StackExchangeRow(TypedDict):
    doc_id: str
    title: str
    question_body: str
    answer_body: str
    tags: list[str]
    score: int
    created_at: str


def _parse_tags(raw: str) -> list[str]:
    return _TAG_RE.findall(raw)


def _normalize_date(raw: str) -> str:
    dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def load_stackexchange_rows(stackexchange_dir: Path) -> Iterator[StackExchangeRow]:
    paths = sorted(glob.glob(str(stackexchange_dir / "QueryResults*.csv")))

    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    answer_score = int(row["AnswerScore"])

                except ValueError:
                    continue

                if answer_score < MIN_ANSWER_SCORE:
                    continue

                yield StackExchangeRow(
                    doc_id=row["QuestionId"],
                    title=row["Title"],
                    question_body=row["QuestionBody"],
                    answer_body=row["AnswerBody"],
                    tags=_parse_tags(row["Tags"]),
                    score=int(row["QuestionScore"]),
                    created_at=_normalize_date(row["CreationDate"]),
                )
