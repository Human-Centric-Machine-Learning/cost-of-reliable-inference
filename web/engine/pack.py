"""Compact binary benchmarks for the page, and a loader equivalent to cri.data.

A pack is one compressed npz per benchmark holding, for every model with reward
scores, the correctness bits, the token counts and the within-question rank of
each generation's reward. Ranks replace the raw scores: Best-of-N only ever
asks which drawn generation scores highest and whether two scores tie, and
dense ranks preserve both exactly. A per-question flag records whether the
model's reward row lined up with its generations, so ``load_pack`` drops the
same questions ``cri.data.load_benchmark`` would drop for the same models.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cri.data import BenchmarkData, DataValidationError, ModelRecords, _read_rewards, load_benchmark

PACK_VERSION = 1


def pack_benchmark(
    data_root: str | Path,
    rewards_root: str | Path,
    benchmark: str,
    models: Sequence[str],
    out: str | Path,
) -> Path:
    """Write ``<out>`` from the JSONL files; returns the path."""
    models = list(models)
    data = load_benchmark(data_root, benchmark, models)  # every question, validated
    arrays: dict[str, np.ndarray] = {
        "version": np.array(PACK_VERSION),
        "benchmark": np.array(benchmark),
        "qids": np.array(data.qids),
        "models": np.array(models),
    }
    n_samples = data.samples_per_question
    for m in models:
        rec = data.records[m]
        rewards = _read_rewards(Path(rewards_root) / benchmark / f"{m}.jsonl")
        rank = np.zeros(rec.shape, dtype=np.uint8)
        ok = np.zeros(len(data.qids), dtype=bool)
        for i, qid in enumerate(data.qids):
            row = rewards.get(qid)
            if row is None or len(row) != n_samples:
                continue
            ok[i] = True
            rank[i] = np.unique(np.asarray(row, dtype=np.float64), return_inverse=True)[1]
        if rec.num_tokens.max() >= 2**16:
            raise DataValidationError(f"{benchmark}/{m}: token count does not fit 16 bits")
        arrays[f"{m}__correctness"] = rec.correctness.astype(np.uint8)
        arrays[f"{m}__tokens"] = rec.num_tokens.astype(np.uint16)
        arrays[f"{m}__rank"] = rank
        arrays[f"{m}__reward_ok"] = ok
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        np.savez_compressed(f, **arrays)
    return out


def pack_models(path: str | Path) -> list[str]:
    with np.load(path) as z:
        return z["models"].tolist()


def load_pack(
    path: str | Path, models: Sequence[str], *, with_rewards: bool
) -> BenchmarkData:
    """The pack's equivalent of ``load_benchmark(root, benchmark, models, rewards_root)``."""
    models = list(models)
    if len(models) < 2:
        raise DataValidationError(f"need at least 2 models, got {models}")
    if len(set(models)) != len(models):
        raise DataValidationError(f"duplicate model in roster: {models}")
    with np.load(path) as z:
        if int(z["version"]) != PACK_VERSION:
            raise DataValidationError(f"{path}: pack version {int(z['version'])}, expected {PACK_VERSION}")
        benchmark = str(z["benchmark"])
        qids = z["qids"].tolist()
        available = set(z["models"].tolist())
        missing = [m for m in models if m not in available]
        if missing:
            raise DataValidationError(f"{benchmark}: pack has no data for {missing}")
        keep = np.ones(len(qids), dtype=bool)
        if with_rewards:
            for m in models:
                keep &= z[f"{m}__reward_ok"]
        dropped = tuple(q for q, k in zip(qids, keep) if not k)
        if with_rewards and int(keep.sum()) < 2:
            raise DataValidationError(f"{benchmark}: only {int(keep.sum())} questions survive reward alignment")
        records = {
            m: ModelRecords(
                model=m,
                correctness=z[f"{m}__correctness"][keep],
                num_tokens=z[f"{m}__tokens"][keep].astype(np.int32),
                rewards=z[f"{m}__rank"][keep].astype(np.float64) if with_rewards else None,
            )
            for m in models
        }
    return BenchmarkData(
        benchmark=benchmark,
        qids=tuple(q for q, k in zip(qids, keep) if k),
        records=records,
        dropped_for_rewards=dropped,
    )
