#!/usr/bin/env python3
"""Extract reward-model scores for Best-of-N provider variants.

Source
------
The same Hugging Face dataset as build_datasets.py,
``Human-Centric-Machine-Learning/strategic-ttc-data``. Each record carries a
``rewards`` list aligned with ``correct`` and ``num_tokens`` -- one score per
generation, from the reward model named in ``reward_model_name`` (ArmoRM-
Llama3-8B-v0.1). build_datasets.py stores only correctness and token counts;
this script stores the reward column separately.

Output
------
    <out>/
    |-- GSM8K/<model>.jsonl
    |-- GPQA/<model>.jsonl
    +-- AIME/<model>.jsonl

One line per question: {"qid": ..., "rewards": [0.056, 0.045, ...]}, aligned
one-to-one with the correctness/token lists in datasets/.

Run it after build_datasets.py. Every reward file, new or already present,
is checked against its dataset counterpart.
"""
from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path

REPO_ID = "Human-Centric-Machine-Learning/strategic-ttc-data"
BENCHMARKS = ("GSM8K", "GPQA", "AIME")


def model_name_from_filename(stem: str) -> str:
    return stem.split("--temp")[0]


def source_files(include_reasoning: bool) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    return [
        f for f in files
        if f.endswith(".jsonl")
        and f.split("/")[0] in BENCHMARKS
        and (include_reasoning or "reason-R1-D" not in f)
    ]


def resolve(remote: str, source: Path | None, cache_dir: str | None) -> Path:
    """The local file for a remote path: from --source if given, else the Hub."""
    if source is not None:
        local = source / remote
        if not local.is_file():
            raise FileNotFoundError(f"--source has no {remote}")
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, remote, repo_type="dataset", cache_dir=cache_dir))


def extract(remote: str, out_dir: Path, cache_dir: str | None,
            source: Path | None = None) -> tuple[str, int, Path]:
    bench, filename = remote.split("/")
    model = model_name_from_filename(Path(filename).stem)
    target = out_dir / bench / f"{model}.jsonl"
    if target.is_file():
        return f"{bench}/{model}", -1, target

    local = resolve(remote, source, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    seen: set[str] = set()
    # Written to a sibling and renamed, so a failure never leaves a partial
    # file that a later run would mistake for a finished one.
    tmp = target.with_suffix(".jsonl.partial")
    with open(local, encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # build_datasets.py drops these too
            qid, rewards = obj.get("qid"), obj.get("rewards")
            if qid is None or not rewards or qid in seen:
                continue  # keep the first occurrence, like build_datasets.py
            values = [float(r) for r in rewards]
            if not all(math.isfinite(r) for r in values):
                raise ValueError(f"{remote}/{qid}: non-finite reward")
            seen.add(qid)
            dst.write(json.dumps({"qid": qid, "rewards": values}) + "\n")
            n += 1
    tmp.replace(target)
    return f"{bench}/{model}", n, target


def _rows(path: Path, field: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            obj = json.loads(line)
            qid = str(obj["qid"])
            if qid in rows:
                raise ValueError(f"{path}:{lineno}: duplicate qid {qid!r}")
            values = obj[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{path}:{lineno}: {field} must be a nonempty list")
            if field == "rewards":
                try:
                    numeric = [float(value) for value in values]
                except (TypeError, ValueError):
                    raise ValueError(f"{path}:{lineno}: rewards are not numeric") from None
                if not all(math.isfinite(value) for value in numeric):
                    raise ValueError(f"{path}:{lineno}: rewards must be finite")
            rows[qid] = len(values)
    return rows


def verify_rewards(dataset: Path, rewards: Path) -> None:
    """Require identical qids and one reward per recorded generation."""
    expected = _rows(dataset, "correctness")
    actual = _rows(rewards, "rewards")
    if set(expected) != set(actual):
        raise ValueError(
            f"{rewards}: qids do not match {dataset} "
            f"(missing={len(set(expected) - set(actual))}, "
            f"extra={len(set(actual) - set(expected))})"
        )
    bad = [qid for qid in expected if expected[qid] != actual[qid]]
    if bad:
        qid = bad[0]
        raise ValueError(
            f"{rewards}: {qid!r} has {actual[qid]} rewards for "
            f"{expected[qid]} generations"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="rewards", type=Path)
    ap.add_argument("--source", default=None, type=Path,
                    help="Local directory laid out as <BENCHMARK>/<file>.jsonl, as an "
                         "offline alternative to downloading from the Hub "
                         "(mirrors build_datasets.py's --source)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--data-root", default="datasets", type=Path,
                    help="Dataset tree to validate reward alignment against")
    ap.add_argument("--include-reasoning", action="store_true")
    args = ap.parse_args()

    remotes = source_files(args.include_reasoning) if args.source is None else [
        f"{b}/{f.name}" for b in BENCHMARKS
        for f in sorted((args.source / b).glob("*.jsonl"))
        if args.include_reasoning or "reason-R1-D" not in f.name
    ]
    print(f"{len(remotes)} source files to process", flush=True)
    failures: list[str] = []
    for i, remote in enumerate(remotes, 1):
        try:
            name, n, target = extract(remote, args.out, args.cache_dir, args.source)
            verify_rewards(args.data_root / f"{name}.jsonl", target)
        except Exception as exc:  # keep going; report at the end
            print(f"[{i}/{len(remotes)}] FAILED {remote}: {type(exc).__name__}: {exc}",
                  flush=True)
            failures.append(remote)
            continue
        print(f"[{i}/{len(remotes)}] {name}: "
              + ("already present" if n < 0 else f"{n} questions"), flush=True)
    if failures:
        raise SystemExit(f"failed to build or verify {len(failures)} reward files")
    print("done", flush=True)


if __name__ == "__main__":
    main()
