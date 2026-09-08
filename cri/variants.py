"""Best-of-N provider variants built from the recorded generations.

``base@N`` serves N of the base model's recorded generations for a question,
keeps the one with the highest reward score, and pays for all N; N = 1 is the
base model. Expected quality is computed exactly: for each reward level, the
probability that it is the maximum of a uniform N-subset times the level's
mean correctness (a tied maximum is picked uniformly, which is what
numpy.argmax does on a randomly ordered draw). E[cost] = N * E[tokens].
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import comb

import numpy as np

from cri.data import BenchmarkData, ModelRecords
from cri.pricing import MODEL_PRICE_PER_1M, unit_cost

VALID_N = (1, 2, 4, 8, 16, 32, 64)  # powers of two dividing the 128 recorded generations

SEPARATOR = "@"


class VariantError(ValueError):
    """A malformed variant name or an unsupported N."""


@dataclass(frozen=True)
class Variant:
    """A provider: a base model served Best-of-N."""

    base: str
    n: int = 1

    def __post_init__(self) -> None:
        if self.n not in VALID_N:
            raise VariantError(f"N must be one of {VALID_N}, got {self.n!r}")

    @property
    def name(self) -> str:
        return self.base if self.n == 1 else f"{self.base}{SEPARATOR}{self.n}"


def parse_variant(name: str) -> Variant:
    """Parse a provider name, using N = 1 when no suffix is present."""
    if SEPARATOR not in name:
        return Variant(name, 1)
    base, _, suffix = name.partition(SEPARATOR)
    try:
        n = int(suffix)
    except ValueError:
        raise VariantError(f"cannot read an N from {name!r}") from None
    return Variant(base, n)


def price_rate(name: str) -> float:
    """Advertised rate, USD per 1M tokens times N; not the price of a query."""
    v = parse_variant(name)
    try:
        return v.n * MODEL_PRICE_PER_1M[v.base]
    except KeyError:
        raise VariantError(
            f"no list price for {v.base!r}; known models: {sorted(MODEL_PRICE_PER_1M)}"
        ) from None


def expected_price(records: ModelRecords, n: int) -> float:
    """Return the query list price before margin deflation."""
    return n * float(records.num_tokens.mean()) * MODEL_PRICE_PER_1M[records.model]


def expected_quality(records: ModelRecords, n: int) -> float:
    """Exact expected correctness of the Best-of-N pick."""
    return expected_qualities(records, (n,))[n]


def expected_qualities(records: ModelRecords, ns: Iterable[int]) -> dict[int, float]:
    """Several Best-of-N qualities with one reward sort."""
    if records.rewards is None:
        raise VariantError(f"{records.model}: rewards are required for Best-of-N")
    requested = tuple(dict.fromkeys(int(n) for n in ns))
    k = records.shape[1]
    if any(n < 1 or n > k for n in requested):
        raise VariantError(f"N must lie in 1..{k}, got {requested}")
    out = {n: 0.0 for n in requested}
    if 1 in out:
        out[1] = float(records.correctness.mean())
    active = tuple(n for n in requested if n != 1)
    if not active:
        return out

    # choose[n][p] = C(p, n) / C(k, n): the probability that all N drawn
    # generations lie among the p lowest-reward ones of a question.
    choose = {
        n: np.array([comb(position, n) for position in range(k + 1)], dtype=float)
        / comb(k, n)
        for n in active
    }
    order = np.argsort(records.rewards, axis=1, kind="stable")
    sorted_rewards = np.take_along_axis(records.rewards, order, axis=1)
    sorted_correct = np.take_along_axis(records.correctness, order, axis=1)
    for rewards, correctness in zip(sorted_rewards, sorted_correct):
        # Split the sorted generations into runs of equal reward, [start, end).
        # The best drawn generation falls in a run with probability
        # choose[end] - choose[start], and is then a uniform member of it, so
        # its expected correctness is the run's mean correctness.
        starts = np.r_[0, np.flatnonzero(rewards[1:] != rewards[:-1]) + 1]
        ends = np.r_[starts[1:], k]
        means = np.add.reduceat(correctness.astype(float), starts) / (ends - starts)
        for n in active:
            out[n] += float(means @ (choose[n][ends] - choose[n][starts]))
    for n in active:
        out[n] /= records.shape[0]
    return out


def expected_cost(records: ModelRecords, n: int, margin: float) -> float:
    """E[cost]: all N generations are paid for."""
    return n * float(records.num_tokens.mean()) * unit_cost(records.model, margin)


def max_cost(records: ModelRecords, n: int, margin: float) -> float:
    """Largest cost this variant can realize."""
    return n * float(records.num_tokens.max()) * unit_cost(records.model, margin)


@dataclass(frozen=True)
class ProviderProfile:
    """One candidate provider's evaluation-only characteristics."""

    variant: Variant
    quality: float
    cost: float
    worst_cost: float

    @property
    def name(self) -> str:
        return self.variant.name


def landscape(
    data: BenchmarkData,
    models: Sequence[str] | None = None,
    ns: Iterable[int] = VALID_N,
    margin: float = 0.25,
) -> list[ProviderProfile]:
    """Every (model, N) candidate's quality and cost, cheapest first."""
    models, ns = list(models or data.models), tuple(ns)
    out = []
    for model in models:
        qualities = expected_qualities(data.records[model], ns)
        out.extend(
            ProviderProfile(
                variant=Variant(model, n),
                quality=qualities[n],
                cost=expected_cost(data.records[model], n, margin),
                worst_cost=max_cost(data.records[model], n, margin),
            )
            for n in ns
        )
    return sorted(out, key=lambda p: p.cost)


def pareto_frontier(profiles: Sequence[ProviderProfile]) -> list[ProviderProfile]:
    """Candidates no other candidate beats on both cost and quality."""
    best = -1.0
    front = []
    for p in sorted(profiles, key=lambda p: p.cost):
        if p.quality > best:
            front.append(p)
            best = p.quality
    return front
