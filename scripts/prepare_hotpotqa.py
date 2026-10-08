"""
Converts a HotpotQA JSON file into two things:

1. data/sample_docs/*.txt -- one text file per unique Wikipedia paragraph
   found in the "context" field of the dataset, so they can be ingested by
   our normal pipeline (python app.py ingest --source data/sample_docs).

2. data/hotpotqa_eval.json -- a flat list of {question, answer} pairs
   pulled from the same file, so you can test the RAG pipeline against
   real questions with known correct answers instead of guessing.

HotpotQA record shape (relevant fields only):
{
  "_id": "...",
  "question": "...",
  "answer": "...",
  "context": [
      ["Wikipedia Article Title", ["sentence 1", "sentence 2", ...]],
      ...
  ],
  ...
}

Usage:
    python scripts/prepare_hotpotqa.py --input path/to/hotpot_dev_distractor_v1.json --limit 200

--limit caps how many QUESTIONS worth of context to pull in (each question
brings in ~10 paragraphs). HotpotQA files can be large and contain
thousands of questions -- start small (e.g. 50-200) so ingestion and
testing stay fast, then increase later if you want a bigger corpus.
"""
import argparse
import json
import re
from pathlib import Path


def safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", title).strip("_")
    return (name or "untitled")[:120] + ".txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a HotpotQA JSON file")
    parser.add_argument(
        "--docs-out", default="data/sample_docs", help="Where to write the .txt documents"
    )
    parser.add_argument(
        "--eval-out", default="data/hotpotqa_eval.json", help="Where to write the eval question set"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Max number of questions to pull context/eval from"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    docs_out = Path(args.docs_out)
    eval_out = Path(args.eval_out)
    docs_out.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_hotpotqa] loading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit:
        data = data[: args.limit]

    seen_titles = {}
    eval_pairs = []

    for record in data:
        question = record.get("question", "").strip()
        answer = record.get("answer", "").strip()
        if question and answer:
            eval_pairs.append({"question": question, "answer": answer})

        for title, sentences in record.get("context", []):
            if title in seen_titles:
                continue
            paragraph = " ".join(s.strip() for s in sentences).strip()
            if not paragraph:
                continue
            seen_titles[title] = paragraph

    # Write one .txt file per unique Wikipedia paragraph.
    for title, paragraph in seen_titles.items():
        out_path = docs_out / safe_filename(title)
        out_path.write_text(f"{title}\n\n{paragraph}\n", encoding="utf-8")

    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, indent=2)

    print(f"[prepare_hotpotqa] wrote {len(seen_titles)} document(s) to {docs_out}")
    print(f"[prepare_hotpotqa] wrote {len(eval_pairs)} eval question(s) to {eval_out}")
    print(
        "\nNext steps:\n"
        f"  python app.py ingest --source {docs_out}\n"
        '  python app.py ask "<one of the questions from hotpotqa_eval.json>" --trace'
    )


if __name__ == "__main__":
    main()