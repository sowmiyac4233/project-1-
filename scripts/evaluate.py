"""
Runs every question in an eval file (produced by scripts/prepare_hotpotqa.py)
through the full self-corrective RAG pipeline, compares each generated
answer against the known correct answer, and prints a summary report.

Usage:
    python scripts/evaluate.py --eval-file data/hotpotqa_eval.json --limit 20
    python scripts/evaluate.py --limit 15 --offset 15

--limit caps how many questions to run (each one makes several LLM calls,
so start with a smaller number like 10-20 to stay inside free-tier limits).

--offset skips this many questions from the start before taking --limit of
them, so you can test a different batch (e.g. --offset 15 --limit 15 runs
questions 16-30). Each batch saves to its own file:
data/eval_results_offset<N>.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import run  # noqa: E402
from src.config import settings  # noqa: E402


_STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "with", "by", "as",
    "it", "its", "this", "that", "he", "she", "they", "his", "her", "their",
}


def normalize(text: str) -> str:
    """Loose normalization so 'Yes.' matches 'yes' and 'Chief of Protocol'
    matches 'She was Chief of Protocol.'"""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _stem(word: str) -> str:
    """Very lightweight suffix stripping so 'seat' matches 'seated', 'direct'
    matches 'directed', etc."""
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def key_tokens(text: str) -> set:
    """The meaningful words/numbers in a short answer, with stopwords
    dropped and light stemming applied."""
    return {_stem(w) for w in normalize(text).split() if w not in _STOPWORDS}


def is_correct(generated: str, expected: str) -> bool:
    """Correct if every key token of the expected answer appears somewhere
    in the generated answer, regardless of order or exact phrasing."""
    expected_tokens = key_tokens(expected)
    if not expected_tokens:
        return False
    generated_tokens = key_tokens(generated)
    return expected_tokens.issubset(generated_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/hotpotqa_eval.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many questions from the start before taking --limit of them "
        "(e.g. --offset 15 --limit 15 runs questions 16-30 instead of 1-15).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to pause between questions, to stay under free-tier rate limits",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause and wait for you to press Enter before running each question, "
        "instead of running through all of them automatically.",
    )
    args = parser.parse_args()

    settings.validate()

    with open(args.eval_file, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    eval_set = eval_set[args.offset :]
    if args.limit:
        eval_set = eval_set[: args.limit]

    results = []
    correct = 0
    refused = 0
    wrong = 0

    print(f"Running {len(eval_set)} question(s) through the pipeline...\n")

    for i, item in enumerate(eval_set, start=1):
        question = item["question"]
        expected = item["answer"]

        if args.interactive:
            input(f"\nPress Enter to run question {i}/{len(eval_set)}: \"{question}\" ... ")

        try:
            state = run(question)
            generated = state.get("generation", "")
        except Exception as exc:  # noqa: BLE001 -- keep evaluating even if one question errors
            generated = f"[ERROR: {exc}]"
            state = {"retries": None, "hallucination_ok": None, "answers_question": None}

        correct_flag = is_correct(generated, expected)
        refused_flag = "wasn't able to find and verify" in generated or "don't have enough information" in generated.lower()

        if correct_flag:
            correct += 1
            verdict = "CORRECT"
        elif refused_flag:
            refused += 1
            verdict = "REFUSED (no hallucination)"
        else:
            wrong += 1
            verdict = "WRONG / UNVERIFIED"

        results.append(
            {
                "question": question,
                "expected": expected,
                "generated": generated,
                "verdict": verdict,
                "retries": state.get("retries"),
            }
        )

        print(f"[{i}/{len(eval_set)}] {verdict}")
        print(f"  Q: {question}")
        print(f"  Expected: {expected}")
        print(f"  Got: {generated[:200]}{'...' if len(generated) > 200 else ''}")
        print()

        time.sleep(args.sleep)

    total = len(eval_set)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total questions:     {total}")
    print(f"Correct:             {correct}  ({100 * correct / total:.1f}%)")
    print(f"Refused (honest):    {refused}  ({100 * refused / total:.1f}%)")
    print(f"Wrong/unverified:    {wrong}  ({100 * wrong / total:.1f}%)")
    print()
    print(
        "Note: the script only recognizes refusals that use its fixed phrases.\n"
        "Answers like 'the information does not say...' can be counted as\n"
        "wrong, so read the saved results file and re-grade by hand."
    )

    out_path = Path(f"data/eval_results_offset{args.offset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": total,
                "correct": correct,
                "refused": refused,
                "wrong": wrong,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()