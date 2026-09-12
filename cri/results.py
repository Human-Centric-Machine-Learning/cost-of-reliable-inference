"""On-disk layout of a run, and its readers and writers.

    outputs/runs/<run_id>/
        manifest.json      resolved config, seeds, per-environment ground truth and constants
        episodes.jsonl     one record per (environment, policy, repetition)
        checkpoints/       cumulative metrics at log-spaced rounds, every repetition
        rounds/            full round log for the audit repetition (all, with full_log);
                           the forced-exploration variant adds <slug>__exploration.csv.gz
        provider_states/   per-provider main-round state, with full_log

The resolved config is stored rather than the source TOML, so a sweep point
is reproducible from the run alone.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class ResultsError(ValueError):
    """A malformed or missing run directory."""


@dataclass(frozen=True)
class RunPaths:
    """A run is a directory; a run_id is its name."""

    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def episodes(self) -> Path:
        return self.root / "episodes.jsonl"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def rounds(self) -> Path:
        return self.root / "rounds"

    @property
    def provider_states(self) -> Path:
        return self.root / "provider_states"

    def slug(self, environment: str, policy: str, rep: int) -> str:
        return f"{environment}__{policy}__rep{rep}"


def create_run(
    output_dir: str | Path, name: str, *, timestamp: str | None = None
) -> RunPaths:
    """Create outputs/runs/<name>-<utc timestamp>/ and its subdirectories."""
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir) / "runs" / f"{name}-{stamp}"
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "rounds").mkdir(parents=True, exist_ok=True)
    (root / "provider_states").mkdir(parents=True, exist_ok=True)
    return RunPaths(root)


def checkpoint_rounds(
    t_max: int, per_decade: int = 20, landmarks: Sequence[int] = ()
) -> list[int]:
    """Log-spaced rounds, plus any landmarks inside the horizon and t_max itself."""
    if t_max < 1:
        raise ResultsError(f"t_max must be >= 1, got {t_max}")
    decades = np.log10(t_max)
    n = max(int(np.ceil(decades * per_decade)), 1)
    grid = np.unique(np.round(np.logspace(0, decades, n + 1)).astype(int))
    marks = [int(v) for v in landmarks if 1 <= v <= t_max]
    return sorted(set(grid.tolist()) | set(marks) | {t_max})


def write_manifest(paths: RunPaths, manifest: Mapping[str, object]) -> None:
    paths.manifest.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def read_manifest(paths: RunPaths) -> dict:
    if not paths.manifest.is_file():
        raise ResultsError(f"no manifest at {paths.manifest}")
    return json.loads(paths.manifest.read_text(encoding="utf-8"))


def append_episode(paths: RunPaths, record: Mapping[str, object]) -> None:
    """One JSON object per finished episode, appended as the run proceeds."""
    with paths.episodes.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_jsonable) + "\n")


def read_episodes(paths: RunPaths) -> list[dict]:
    if not paths.episodes.is_file():
        return []
    with paths.episodes.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_csv(path: Path, columns: Mapping[str, Iterable], *, compress: bool) -> None:
    names = list(columns)
    rows = zip(*(np.asarray(columns[k]).tolist() for k in names))
    opener = (
        (lambda: gzip.open(path, "wt", newline="", encoding="utf-8"))
        if compress
        else (lambda: path.open("w", newline="", encoding="utf-8"))
    )
    with opener() as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(rows)


def _read_csv(path: Path, *, compress: bool) -> dict[str, np.ndarray]:
    opener = (
        (lambda: gzip.open(path, "rt", newline="", encoding="utf-8"))
        if compress
        else (lambda: path.open(newline="", encoding="utf-8"))
    )
    with opener() as f:
        reader = csv.reader(f)
        names = next(reader)
        cols: list[list[str]] = [[] for _ in names]
        for row in reader:
            if len(row) != len(names):
                raise ResultsError(
                    f"{path}: row has {len(row)} fields, expected {len(names)}"
                )
            for i, value in enumerate(row):
                cols[i].append(value)
    # Use int64 for integer columns, float64 for other numeric columns, and str otherwise.
    out: dict[str, np.ndarray] = {}
    for name, values in zip(names, cols):
        arr = np.array(values)
        try:
            out[name] = (
                arr.astype(np.int64)
                if all(v.lstrip("-").isdigit() for v in values)
                else arr.astype(np.float64)
            )
        except ValueError:
            out[name] = arr
    return out


def write_checkpoints(
    paths: RunPaths,
    environment: str,
    policy: str,
    rep: int,
    columns: Mapping[str, Iterable],
) -> Path:
    """Cumulative metrics at the checkpoint rounds."""
    path = paths.checkpoints / f"{paths.slug(environment, policy, rep)}.csv"
    _write_csv(path, columns, compress=False)
    return path


def read_checkpoints(paths: RunPaths, environment: str, policy: str, rep: int):
    return _read_csv(
        paths.checkpoints / f"{paths.slug(environment, policy, rep)}.csv",
        compress=False,
    )


def write_round_log(
    paths: RunPaths,
    environment: str,
    policy: str,
    rep: int,
    columns: Mapping[str, Iterable],
) -> Path:
    """The full per-round log."""
    path = paths.rounds / f"{paths.slug(environment, policy, rep)}.csv.gz"
    _write_csv(path, columns, compress=True)
    return path


def read_round_log(paths: RunPaths, environment: str, policy: str, rep: int):
    return _read_csv(
        paths.rounds / f"{paths.slug(environment, policy, rep)}.csv.gz", compress=True
    )


def write_exploration_log(
    paths: RunPaths,
    environment: str,
    policy: str,
    rep: int,
    columns: Mapping[str, Iterable],
) -> Path:
    """The forced-exploration services, next to the round log they belong to."""
    path = paths.rounds / f"{paths.slug(environment, policy, rep)}__exploration.csv.gz"
    _write_csv(path, columns, compress=True)
    return path


def read_exploration_log(paths: RunPaths, environment: str, policy: str, rep: int):
    return _read_csv(
        paths.rounds / f"{paths.slug(environment, policy, rep)}__exploration.csv.gz",
        compress=True,
    )


def write_provider_states(
    paths: RunPaths,
    environment: str,
    policy: str,
    rep: int,
    roster: Sequence[str],
    columns: Mapping[str, np.ndarray],
) -> Path:
    """Per-provider vectors, one column per field::model."""
    if not columns:
        raise ResultsError("provider-state log is empty")
    width = len(roster)
    flattened: dict[str, Iterable] = {}
    n_rows: int | None = None
    for field, values in columns.items():
        arr = np.asarray(values)
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ResultsError(
                f"provider-state {field!r} has shape {arr.shape}, expected (*, {width})"
            )
        n_rows = arr.shape[0] if n_rows is None else n_rows
        if arr.shape[0] != n_rows:
            raise ResultsError("provider-state fields have different row counts")
        for j, model in enumerate(roster):
            flattened[f"{field}::{model}"] = arr[:, j]
    path = paths.provider_states / f"{paths.slug(environment, policy, rep)}.csv.gz"
    _write_csv(path, flattened, compress=True)
    return path


def read_provider_states(paths: RunPaths, environment: str, policy: str, rep: int):
    return _read_csv(
        paths.provider_states / f"{paths.slug(environment, policy, rep)}.csv.gz",
        compress=True,
    )
