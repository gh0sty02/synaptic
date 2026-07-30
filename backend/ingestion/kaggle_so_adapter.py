from pathlib import Path
from collections.abc import Iterator
from typing import TypedDict, Any
import csv
import sys

csv.field_size_limit(sys.maxsize)


MIN_QUESTION_SCORE = 5
MIN_ANSWER_SCORE = 5


class KaggleRow(TypedDict):
    doc_id: str
    title: str
    question_body: str
    answer_body: str
    tags: list[str]
    score: int
    created_at: str


def _filtered_question_ids(questions_path: Path) -> dict[str, dict[str, Any]]:
    kept: dict[str, Any] = {}

    with open(questions_path, newline="", encoding="latin1") as f:
        for row in csv.DictReader(f):
            try:
                score = int(row["Score"])
            except ValueError:
                continue

            if score >= MIN_QUESTION_SCORE:
                kept[row["Id"]] = row

    return kept


def _best_answers(
    answer_path: Path, question_ids: set[str]
) -> dict[str, dict[str, Any]]:
    best: dict[str, Any] = {}

    with open(answer_path, newline="", encoding="latin1") as f:
        for row in csv.DictReader(f):
            # check if the id of the question (saved as parent_id) exists in filtered question ids
            parent_id = row["ParentId"]
            if parent_id not in question_ids:
                continue

            try:
                score = int(row["Score"])
            except ValueError:
                continue

            if score < MIN_ANSWER_SCORE:
                continue

            # get if a answer for this question already exists in the array
            current_best = best.get(parent_id)

            # check if we do not have a existing ans or if the score of existing answer is less than we we have right now, if yes reassign the new ans as it has high score
            if current_best is None or score > int(current_best["Score"]):
                best[parent_id] = row

    return best


def _tags_by_questions(tags_path: Path, question_ids: set[str]) -> dict[str, list[str]]:
    tags: dict[str, list[str]] = {}

    with open(tags_path, newline="", encoding="latin1") as f:
        for row in csv.DictReader(f):
            # only keep tags that are related to filtered question ids
            if row["Id"] in question_ids:
                tags.setdefault(row["Id"], []).append(row["Tag"])

    return tags


def load_kaggle_rows(kaggle_dir: Path) -> Iterator[KaggleRow]:
    questions = _filtered_question_ids(kaggle_dir / "Questions.csv")

    best_answers = _best_answers(kaggle_dir / "Answers.csv", set(questions))

    # only keep question that also cleared the answers-score floor
    surviving_ids = set(questions) & set(best_answers)

    tags = _tags_by_questions(kaggle_dir / "Tags.csv", surviving_ids)

    for qid in surviving_ids:
        q, a = questions[qid], best_answers[qid]

        yield (
            KaggleRow(
                doc_id=qid,
                title=q["Title"],
                question_body=q["Body"],
                answer_body=a["Body"],
                tags=tags.get(qid, []),
                score=int(q["Score"]),
                created_at=q["CreationDate"],
            )
        )
