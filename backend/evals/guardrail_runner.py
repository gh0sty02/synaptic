import argparse
import asyncio
import json
from pathlib import Path

from guardrails.classifier import guardrail_classifier

EVAL_SET_PATH = Path(__file__).resolve().parents[2] / "dataset" / "guardrail_eval.jsonl"


async def main(eval_set_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in eval_set_path.read_text().splitlines()
        if line.strip()
    ]

    false_positives = 0
    false_negatives = 0
    benign_total = sum(1 for r in rows if not r["should_block"])
    attack_total = sum(1 for r in rows if r["should_block"])

    for row in rows:
        verdict = await guardrail_classifier.check_input(row["query"], config={})
        if row["should_block"] and not verdict.blocked:
            false_negatives += 1
            print(f"FALSE NEGATIVE: {row['query']!r} (expected {row['category']})")
        elif not row["should_block"] and verdict.blocked:
            false_positives += 1
            print(
                f"FALSE POSITIVE: {row['query']!r} -> {verdict.category} ({verdict.reason})"
            )

    fp_rate = false_positives / benign_total if benign_total else 0.0
    fn_rate = false_negatives / attack_total if attack_total else 0.0

    print(f"\nFalse positive rate: {fp_rate:.1%} ({false_positives}/{benign_total})")
    print(f"False negative rate: {fn_rate:.1%} ({false_negatives}/{attack_total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_PATH)
    args = parser.parse_args()
    asyncio.run(main(args.eval_set))
