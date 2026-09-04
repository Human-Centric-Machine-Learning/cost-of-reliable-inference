#!/usr/bin/env python3
"""
Build compact per-model correctness / token datasets for GSM8K, GPQA and AIME.

Source
------
The Hugging Face dataset ``Human-Centric-Machine-Learning/strategic-ttc-data``,
whose records use the same format as the original ``final_runs/`` files
(one JSON object per question, with ``qid``, ``correct`` and ``num_tokens``
lists among other fields). You can also point ``--source`` at a local copy of
that data (e.g. an existing ``final_runs/`` directory) to run fully offline.

Output
------
    <out>/
    ├── GSM8K/<model>.jsonl
    ├── GPQA/<model>.jsonl
    └── AIME/<model>.jsonl

One JSONL file per model; one line per question, of exactly the form:

    {"qid": ..., "correctness": [1, 0, 1, ...], "num_tokens": [123, 98, 145, ...]}

``correctness[i]`` is the binary correctness of sample ``i`` and ``num_tokens[i]``
is the number of generated (output) tokens for that same sample, so the two
lists are aligned one-to-one. Nothing else (prompts, responses, answers,
rewards, explanations, …) is carried over.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ID = "Human-Centric-Machine-Learning/strategic-ttc-data"
BENCHMARKS = {"GSM8K": "gsm8k", "GPQA": "gpqa", "AIME": "aime"}


def model_name_from_filename(stem: str) -> str:
    """`Llama-3-8B--temp-0.6--samples-128--max-512` -> `Llama-3-8B`."""
    return stem.split("--temp")[0]


def token_count(t) -> int:
    """
    Generated-token count for one sample.

    Standard models store an int; reasoning models store ``[think, total]``
    where ``total`` is the full completion length (think is a subset of it),
    so the total is the number of generated tokens.
    """
    if isinstance(t, (list, tuple)):
        if not t:
            raise ValueError("empty token-count sequence")
        t = t[-1]
    if isinstance(t, bool):
        raise ValueError("boolean is not a token count")
    value = int(t)
    if value < 0 or value != float(t):
        raise ValueError(f"invalid token count {t!r}")
    return value


def correctness_value(value: object) -> int:
    """Parse a binary value without truthiness coercion."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    raise ValueError(f"correctness is not binary: {value!r}")


def resolve_source(source: str | None, cache_dir: str | None) -> Path:
    """Return a local directory containing the raw JSONL records."""
    if source:
        p = Path(source).expanduser()
        if not p.exists():
            sys.exit(f"--source path does not exist: {p}")
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required to download the dataset.\n"
            "  pip install huggingface_hub\n"
            "or pass --source <local dir> pointing at a copy of the data."
        )
    print(f"Downloading {REPO_ID} from the Hugging Face Hub ...")
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", cache_dir=cache_dir)
    return Path(path)


def benchmark_of(jsonl_path: Path, root: Path) -> str | None:
    """Which known benchmark a file belongs to, inferred from its path parts."""
    parts = set(jsonl_path.relative_to(root).parts)
    hit = BENCHMARKS.keys() & parts
    return next(iter(hit)) if hit else None


def convert(root: Path, out_dir: Path) -> None:
    jsonl_files = sorted(root.rglob("*.jsonl"))
    if not jsonl_files:
        sys.exit(
            f"No .jsonl files found under {root}. If the Hub repo stores the data "
            f"differently, download it and pass --source pointing at the JSONL files."
        )

    # (benchmark, model) -> {qid: (correctness_list, num_tokens_list)}, deduped by qid
    collected: dict[tuple[str, str], dict] = defaultdict(dict)
    skipped_files = 0
    dropped_lines = 0
    contamination_dropped = 0
    invalid_rows = 0

    for fp in jsonl_files:
        bench = benchmark_of(fp, root)
        if bench is None:
            skipped_files += 1
            continue
        prefix = BENCHMARKS[bench]
        model = model_name_from_filename(fp.stem)
        bucket = collected[(bench, model)]

        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    dropped_lines += 1
                    continue

                qid = obj.get("qid")
                correct = obj.get("correct")
                num_tokens = obj.get("num_tokens")
                if qid is None or not correct or not num_tokens:
                    dropped_lines += 1
                    continue
                # Reject records whose qid does not match the benchmark folder.
                if not str(qid).lower().startswith(prefix):
                    contamination_dropped += 1
                    continue
                if qid in bucket:
                    continue  # keep first occurrence, like the original loader

                if len(correct) != len(num_tokens):
                    invalid_rows += 1
                    continue
                try:
                    correctness = [correctness_value(c) for c in correct]
                    tokens = [token_count(t) for t in num_tokens]
                except (TypeError, ValueError):
                    invalid_rows += 1
                    continue

                bucket[qid] = (correctness, tokens)

    # write one JSONL per (benchmark, model)
    total_lines = 0
    for (bench, model), bucket in sorted(collected.items()):
        bench_dir = out_dir / bench
        bench_dir.mkdir(parents=True, exist_ok=True)
        out_path = bench_dir / f"{model}.jsonl"
        tmp = out_path.with_suffix(".jsonl.partial")
        with tmp.open("w", encoding="utf-8") as f:
            for qid, (correctness, tokens) in bucket.items():
                f.write(json.dumps(
                    {"qid": qid, "correctness": correctness, "num_tokens": tokens}
                ) + "\n")
        tmp.replace(out_path)
        total_lines += len(bucket)
        print(f"  {bench}/{model}.jsonl  ({len(bucket)} questions)")

    print(
        f"\nDone. Wrote {len(collected)} model files "
        f"({total_lines} question-lines) under {out_dir}/"
    )
    if skipped_files:
        print(f"  ({skipped_files} source files ignored: not in {sorted(BENCHMARKS)})")
    if dropped_lines:
        print(f"  ({dropped_lines} records dropped: unparseable or missing qid/correct/num_tokens)")
    if contamination_dropped:
        print(f"  ({contamination_dropped} records dropped: qid prefix did not match the benchmark folder)")
    if invalid_rows:
        print(f"  ({invalid_rows} invalid records dropped: non-binary correctness, "
              "bad token counts, or misaligned lists)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help="Local directory of raw JSONL records (e.g. a final_runs/ copy). "
                         "If omitted, the dataset is downloaded from the Hugging Face Hub.")
    ap.add_argument("--out", default="datasets",
                    help="Output directory (default: ./datasets)")
    ap.add_argument("--cache-dir", default=None,
                    help="Hugging Face cache directory for the download (optional)")
    args = ap.parse_args()

    root = resolve_source(args.source, args.cache_dir)
    convert(root, Path(args.out).expanduser())


if __name__ == "__main__":
    main()
