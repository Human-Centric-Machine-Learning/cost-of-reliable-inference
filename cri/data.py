"""Load and validate recorded generations from ``datasets/`` and ``rewards/``.

datasets/<BENCH>/<model>.jsonl holds one line per question,
{"qid", "correctness": [...], "num_tokens": [...]}, and
rewards/<BENCH>/<model>.jsonl holds {"qid", "rewards": [...]}; entry k of
each list is the same recorded generation. Rewards are loaded only when a
roster has Best-of-N providers.

The loader returns aligned (n_questions, n_samples) matrices per model in
sorted-qid order and refuses misaligned or non-binary lists, negative token
counts, duplicate qids, and models whose qid sets or sample counts differ. A
question whose reward row does not align is dropped for every model and
reported.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DataValidationError(ValueError):
    """A dataset file violates an invariant the shared query stream needs."""


@dataclass(frozen=True)
class ModelRecords:
    """One model's recorded generations on one benchmark, as aligned matrices.

    Each matrix has one row per question, in the benchmark's canonical
    (sorted-qid) order, and one column per recorded generation, so
    ``correctness[i, k]``, ``num_tokens[i, k]`` and ``rewards[i, k]`` all
    describe generation ``k`` of question ``i``. Values are raw: a
    correctness bit, an output token count not yet priced, and the reward
    model's score. ``rewards`` is ``None`` unless the benchmark was loaded
    with a rewards root; only Best-of-N providers need it.
    """

    model: str
    correctness: np.ndarray  # (n_questions, n_samples) uint8
    num_tokens: np.ndarray  # (n_questions, n_samples) int32
    rewards: np.ndarray | None = None  # (n_questions, n_samples) float64

    @property
    def shape(self) -> tuple[int, int]:
        return self.correctness.shape


@dataclass(frozen=True)
class BenchmarkData:
    """Every loaded model's records for one benchmark, mutually aligned.

    ``qids`` is the canonical question order and ``records`` maps model name
    to that model's matrices. Alignment is guaranteed by the loader: every
    model has a row for every qid, in the same order, with the same number
    of generations per question. Row ``i`` therefore means the same question
    for every model, which is what lets the stream give each provider a
    potential outcome for the same round.
    """

    benchmark: str
    qids: tuple[str, ...]
    records: dict[str, ModelRecords]
    #: Questions excluded because some model's reward row did not line up with
    #: its generations. Reported rather than silently absorbed.
    dropped_for_rewards: tuple[str, ...] = ()

    @property
    def has_rewards(self) -> bool:
        return all(r.rewards is not None for r in self.records.values())

    @property
    def models(self) -> list[str]:
        return list(self.records)

    @property
    def n_questions(self) -> int:
        return len(self.qids)

    @property
    def samples_per_question(self) -> int:
        """Recorded generations per question: the pool repetitions draw from.

        Not a horizon. An episode serves each question at most once, so the
        horizon is ``n_questions``.
        """
        return next(iter(self.records.values())).shape[1]


def list_models(root: str | Path, benchmark: str) -> list[str]:
    """Model names available for a benchmark, from the JSONL filenames."""
    d = Path(root) / benchmark
    if not d.is_dir():
        raise DataValidationError(f"no such benchmark directory: {d}")
    return sorted(f.stem for f in d.glob("*.jsonl"))


def _read_model(
    path: Path, model: str
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Read and validate one file. Returns qids and the two aligned matrices."""
    qids: list[str] = []
    corr: list[list[int]] = []
    toks: list[list[int]] = []
    n_samples: int | None = None

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{path}:{lineno}: unparseable JSON: {exc}"
                ) from None

            where = f"{path}:{lineno}"
            try:
                qid, c, t = obj["qid"], obj["correctness"], obj["num_tokens"]
            except KeyError as exc:
                raise DataValidationError(f"{where}: missing field {exc}") from None

            if len(c) != len(t):
                raise DataValidationError(
                    f"{where}: correctness/num_tokens misaligned ({len(c)} vs {len(t)})"
                )
            if not c:
                raise DataValidationError(f"{where}: empty generation lists")
            if n_samples is None:
                n_samples = len(c)
            elif len(c) != n_samples:
                raise DataValidationError(
                    f"{where}: {len(c)} samples, but {model} uses {n_samples} elsewhere; "
                    "the shared stream needs a constant sample count"
                )
            if any(v not in (0, 1) for v in c):
                raise DataValidationError(f"{where}: correctness is not binary")
            if any(v < 0 for v in t):
                raise DataValidationError(f"{where}: negative token count")

            qids.append(qid)
            corr.append([int(v) for v in c])
            toks.append([int(v) for v in t])

    if not qids:
        raise DataValidationError(f"{path}: no records")
    if len(set(qids)) != len(qids):
        raise DataValidationError(f"{path}: duplicate qid")
    return qids, corr, toks


def _read_rewards(path: Path) -> dict[str, list[float]]:
    """Read one rewards file into {qid: scores}; alignment is checked by the caller."""
    out: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"{path}:{lineno}: unparseable JSON: {exc}"
                ) from None
            where = f"{path}:{lineno}"
            try:
                qid, raw_rewards = obj["qid"], obj["rewards"]
            except KeyError as exc:
                raise DataValidationError(f"{where}: missing field {exc}") from None
            if qid in out:
                raise DataValidationError(f"{where}: duplicate qid {qid!r}")
            if not isinstance(raw_rewards, list) or not raw_rewards:
                raise DataValidationError(f"{where}: rewards must be a nonempty list")
            try:
                values = [float(r) for r in raw_rewards]
            except (TypeError, ValueError):
                raise DataValidationError(f"{where}: rewards are not numeric") from None
            if not all(np.isfinite(values)):
                raise DataValidationError(f"{where}: rewards must be finite")
            out[str(qid)] = values
    if not out:
        raise DataValidationError(f"{path}: no reward records")
    return out


def load_benchmark(
    root: str | Path,
    benchmark: str,
    models: Sequence[str] | Iterable[str],
    rewards_root: str | Path | None = None,
) -> BenchmarkData:
    """Load one benchmark for a roster of models, with rewards if a root is given.

    A question whose reward row does not align with its generations, for any
    model, is dropped for every model and listed in ``dropped_for_rewards``.
    """
    models = list(models)
    if len(models) < 2:
        raise DataValidationError(f"need at least 2 models, got {models}")
    if len(set(models)) != len(models):
        raise DataValidationError(f"duplicate model in roster: {models}")

    root = Path(root)
    raw: dict[str, tuple[list[str], list[list[int]], list[list[int]]]] = {}
    for m in models:
        path = root / benchmark / f"{m}.jsonl"
        if not path.is_file():
            raise DataValidationError(f"missing dataset file: {path}")
        raw[m] = _read_model(path, m)

    canonical = sorted(raw[models[0]][0])
    full_set = set(canonical)  # cross-model check uses the set before reward drops
    n_samples = len(raw[models[0]][1][0])

    rewards: dict[str, dict[str, list[float]]] = {}
    dropped: set[str] = set()
    if rewards_root is not None:
        for m in models:
            path = Path(rewards_root) / benchmark / f"{m}.jsonl"
            if not path.is_file():
                raise DataValidationError(
                    f"missing rewards file: {path}; run build_rewards.py first"
                )
            rewards[m] = _read_rewards(path)
            for qid in canonical:
                row = rewards[m].get(qid)
                if row is None or len(row) != n_samples:
                    dropped.add(qid)
        canonical = [q for q in canonical if q not in dropped]
        if len(canonical) < 2:
            raise DataValidationError(
                f"{benchmark}: only {len(canonical)} questions survive reward alignment"
            )

    records: dict[str, ModelRecords] = {}
    for m in models:
        qids, corr, toks = raw[m]
        if set(qids) != full_set:
            missing = sorted(full_set - set(qids))[:3]
            extra = sorted(set(qids) - full_set)[:3]
            raise DataValidationError(
                f"{benchmark}/{m}: qid set differs from {models[0]} "
                f"(missing e.g. {missing}, extra e.g. {extra})"
            )
        if len(corr[0]) != n_samples:
            raise DataValidationError(
                f"{benchmark}/{m}: {len(corr[0])} samples per question, but "
                f"{models[0]} has {n_samples}; the shared stream needs them equal"
            )
        order = {q: i for i, q in enumerate(qids)}
        idx = [order[q] for q in canonical]
        records[m] = ModelRecords(
            model=m,
            correctness=np.array([corr[i] for i in idx], dtype=np.uint8),
            num_tokens=np.array([toks[i] for i in idx], dtype=np.int32),
            rewards=(
                np.array([rewards[m][q] for q in canonical], dtype=np.float64)
                if rewards
                else None
            ),
        )

    return BenchmarkData(
        benchmark=benchmark,
        qids=tuple(canonical),
        records=records,
        dropped_for_rewards=tuple(sorted(dropped)),
    )
