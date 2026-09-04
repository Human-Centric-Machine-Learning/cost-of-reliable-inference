"""The shared query stream: one pass over the benchmark's questions.

An episode is a seeded permutation of the questions, one per round, so the
horizon is the question count. Every provider has a preassigned potential
outcome for every question: N distinct recorded generations (N = 1 for a base
model), the correctness of the reward model's pick, and the tokens of all N.
The draw is seeded by (environment, repetition), never by the policy, so
compared policies see the same outcomes. The ``independent_costs`` ablation
draws the token subset separately from the answer subset.

``reveal`` is the mechanism-facing accessor, counted and allowed once per
round; ``potential_outcome`` is the evaluation-facing one and is free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cri.data import BenchmarkData
from cri.pricing import unit_cost
from cri.variants import Variant, parse_variant


class StreamError(ValueError):
    """Malformed stream construction, or an illegal access pattern."""


@dataclass(frozen=True)
class Instance:
    """One query: the benchmark question served at this round."""

    qid: str


@dataclass(frozen=True)
class Outcome:
    """What one provider would produce on one question, on this repetition."""

    qid: str
    sample_index: int
    correctness: int
    num_tokens: int
    cost: float  # USD x 1e-6


class QueryStream:
    """A shared, seeded permutation of the questions, one pass per episode."""

    def __init__(
        self,
        data: BenchmarkData,
        roster: Sequence[str],
        margin: float,
        order_rng: np.random.Generator,
        draw_rng: np.random.Generator,
        repetition: int = 0,
        independent_costs: bool = False,
    ) -> None:
        roster = list(roster)
        if len(roster) < 2:
            raise StreamError(f"need at least 2 providers, got {roster}")
        if repetition < 0:
            raise StreamError(f"repetition must be >= 0, got {repetition}")

        self.data = data
        self.roster = roster
        self.variants: dict[str, Variant] = {m: parse_variant(m) for m in roster}
        self.margin = float(margin)
        self.repetition = int(repetition)
        self.independent_costs = bool(independent_costs)
        self.n_samples = data.samples_per_question

        missing = [v.base for v in self.variants.values() if v.base not in data.records]
        if missing:
            raise StreamError(f"base models not loaded for {data.benchmark}: {missing}")
        if any(v.n > 1 for v in self.variants.values()) and not data.has_rewards:
            raise StreamError(
                "Best-of-N providers need reward scores; load the benchmark with "
                "rewards_root=... (see build_rewards.py)"
            )
        for v in self.variants.values():
            if v.n > self.n_samples:
                raise StreamError(
                    f"{v.name}: N={v.n} exceeds the {self.n_samples} recorded generations"
                )

        self._order: np.ndarray = order_rng.permutation(data.n_questions)

        # Preassigned outcomes, one per (provider, question).
        self._corr: dict[str, np.ndarray] = {}
        self._tok: dict[str, np.ndarray] = {}
        self._pick: dict[str, np.ndarray] = {}
        self._unit = {m: unit_cost(m, margin) for m in roster}
        rows = np.arange(data.n_questions)
        pool = np.tile(np.arange(self.n_samples), (data.n_questions, 1))
        answer_draws: dict[str, np.ndarray] = {}
        for name, v in self.variants.items():
            rec = data.records[v.base]
            drawn = draw_rng.permuted(pool, axis=1)[:, : v.n]  # N distinct per question
            answer_draws[name] = drawn
            if v.n == 1:
                kept = drawn[:, 0]
            else:
                # Reward model's pick; the drawn order is random, so a tied
                # maximum is chosen uniformly, as variants.expected_quality assumes.
                kept = drawn[rows, np.argmax(rec.rewards[rows[:, None], drawn], axis=1)]
            self._pick[name] = kept
            self._corr[name] = rec.correctness[rows, kept]

        # Cost subsets are drawn after all answer subsets, so the answers are
        # the same with and without independent_costs.
        for name, v in self.variants.items():
            rec = data.records[v.base]
            cost_drawn = (
                draw_rng.permuted(pool, axis=1)[:, : v.n]
                if self.independent_costs
                else answer_draws[name]
            )
            self._tok[name] = rec.num_tokens[rows[:, None], cost_drawn].sum(axis=1)

        self.reveal_count = 0
        self._revealed: set[int] = set()

    @property
    def disjoint_folds(self) -> dict[str, int]:
        """How many repetitions could draw disjoint generations per provider (diagnostic)."""
        return {m: self.n_samples // v.n for m, v in self.variants.items()}

    def __len__(self) -> int:
        return len(self._order)

    def _question(self, t: int) -> int:
        if not 0 <= t < len(self._order):
            raise IndexError(
                f"round {t} outside a stream of {len(self._order)} questions"
            )
        return int(self._order[t])

    def instance(self, t: int) -> Instance:
        """The question served at round ``t``; reveals nothing."""
        return Instance(qid=self.data.qids[self._question(t)])

    def _outcome(self, t: int, provider: str) -> Outcome:
        q = self._question(t)
        try:
            kept = int(self._pick[provider][q])
        except KeyError:
            raise StreamError(f"{provider!r} is not in this stream's roster") from None
        tokens = int(self._tok[provider][q])
        return Outcome(
            qid=self.data.qids[q],
            sample_index=kept,
            correctness=int(self._corr[provider][q]),
            num_tokens=tokens,
            cost=tokens * self._unit[provider],
        )

    def reveal(self, t: int, provider: str) -> Outcome:
        """The winner's outcome for round t; counted, at most once per round."""
        if t in self._revealed:
            raise StreamError(f"round {t} already revealed; one outcome per round")
        outcome = self._outcome(t, provider)
        self._revealed.add(t)
        self.reveal_count += 1
        return outcome

    def potential_outcome(self, t: int, provider: str) -> Outcome:
        """What ``provider`` would produce at round t; for evaluation and oracles only."""
        return self._outcome(t, provider)

    def max_cost(self) -> float:
        """Largest cost any roster provider realizes on this stream."""
        return max(float(self._tok[m].max()) * self._unit[m] for m in self.roster)


def build_stream(
    data: BenchmarkData,
    roster: Sequence[str],
    margin: float,
    order_rng: np.random.Generator,
    draw_rng: np.random.Generator,
    repetition: int = 0,
    independent_costs: bool = False,
) -> QueryStream:
    """Build one episode's question order and generation draws."""
    return QueryStream(
        data, roster, margin, order_rng, draw_rng, repetition, independent_costs
    )
