"""Forced exploration: optional experimental variants of the round protocol.

The mechanism selects and pays one provider per round. A forced-exploration
variant makes the platform observe a second provider now and then, so that
competitors' cost radii keep shrinking and the critical payment can approach
the second-lowest qualified cost. Every variant is described by one
``ForcedExploration`` object, built by ``make_exploration`` from a mode name
and a few parameters, and passed to ``simulate_episode`` as ``exploration``;
``None`` is the mechanism as published. Three modes exist:

``runner_up``
    After the winner has been priced, revealed and updated, with probability
    p_t the round's runner-up is served as well: the provider in the winner's
    candidate set whose score is the lowest after the winner's, i.e. the score
    that sets the critical payment. It is paid the bid it held at the start of
    the round. Logged as an additional service in ``ExplorationLog``.
``uniform``
    Like ``runner_up``, but the second provider is drawn uniformly from the
    candidate set without the winner and paid C_max.
``count``
    Replaces the round instead of adding a service. When some eligible
    provider has been observed fewer than g(t) = ceil(scale * (t / n)^alpha)
    times (alpha = 1/2 by default, so g(t) = ceil(scale * sqrt(t / n))),
    one of the under-sampled eligible providers is drawn uniformly,
    served and paid C_max, and no mechanism winner is selected; otherwise the
    round is the mechanism's. The choice depends on eligibility and counts
    only. Such rounds carry phase ``"explore"`` in the round log.

In every mode the explored provider observes its own generation cost and
revises its bid as after any selection, and the platform records its
answer's correctness and counts the selection. The winner of an ordinary
round is selected, priced and updated exactly as without exploration.

The additive modes use a schedule for p_t: ``sqrt`` is p_t = min{1, sqrt(n / t)},
``constant`` is p_t = p. The simulator draws two round-indexed uniforms per
round from a seed child of its own, so the exploration draws depend only on
the episode seed, never on the policy or on what a round reveals.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Protocol, runtime_checkable

from cri.mechanism import Selection
from cri.platform import PlatformState


class ExplorationError(ValueError):
    """An invalid exploration setting or an ill-defined exploration step."""


# --------------------------------------------------------------------------
# Schedules: when an additive mode explores
# --------------------------------------------------------------------------


@runtime_checkable
class ExplorationSchedule(Protocol):
    def probability(self, t: int, n: int) -> float:
        """The probability of exploring at main round t with n providers."""


@dataclass(frozen=True)
class ConstantExploration:
    """p_t = p at every main round."""

    p: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ExplorationError(f"p must lie in [0, 1], got {self.p!r}")

    def probability(self, t: int, n: int) -> float:
        return self.p


@dataclass(frozen=True)
class SqrtExploration:
    """p_t = min{1, sqrt(n / t)}: every round while t <= n, then decaying like 1/sqrt(t)."""

    def probability(self, t: int, n: int) -> float:
        if t < 1 or n < 1:
            raise ExplorationError(f"t and n must be >= 1, got t={t}, n={n}")
        return min(1.0, math.sqrt(n / t))


SCHEDULES = {"sqrt": SqrtExploration, "constant": ConstantExploration}


def make_schedule(name: str = "sqrt", **params: float) -> ExplorationSchedule:
    """A schedule by name; parameters a schedule does not take (``p`` for ``sqrt``) are ignored."""
    try:
        cls = SCHEDULES[name]
    except KeyError:
        raise ExplorationError(
            f"unknown exploration schedule {name!r}; known: {sorted(SCHEDULES)}"
        ) from None
    taken = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in params.items() if k in taken})


def probability_at(schedule: ExplorationSchedule, t: int, n: int) -> float:
    """``schedule.probability(t, n)`` checked to lie in [0, 1]."""
    p = float(schedule.probability(t, n))
    if not 0.0 <= p <= 1.0:
        raise ExplorationError(
            f"{type(schedule).__name__} gave p_t={p!r} at t={t}, outside [0, 1]"
        )
    return p


# --------------------------------------------------------------------------
# Rules: whom an additive mode explores, and what it pays
# --------------------------------------------------------------------------


def runner_up_set(selection: Selection) -> list[str]:
    """The candidates other than the winner with the lowest score, in roster order.

    For the mechanism this is the second-lowest score in A_t, whose holder
    sets the critical payment; ties are broken by the caller.
    """
    if selection.halted or selection.winner is None:
        raise ExplorationError("a halted round has no runner-up")
    others = [j for j in selection.eligible if j != selection.winner]
    if not others:
        raise ExplorationError("the candidate set holds no provider besides the winner")
    best = min(selection.scores[j] for j in others)
    return [j for j in others if selection.scores[j] == best]


def pick_runner_up(selection: Selection, tie_u: float) -> str:
    """The runner-up, with exact ties broken uniformly by ``tie_u`` in [0, 1)."""
    if not 0.0 <= tie_u < 1.0:
        raise ExplorationError(f"tie_u must lie in [0, 1), got {tie_u!r}")
    tied = runner_up_set(selection)
    return tied[min(int(tie_u * len(tied)), len(tied) - 1)]


@runtime_checkable
class ExplorationRule(Protocol):
    def choose(self, selection: Selection, u: float) -> tuple[str, int]:
        """The provider to explore and the size of the set it was drawn from; ``u`` is uniform in [0, 1)."""

    def payment(self, state: PlatformState, provider: str) -> float:
        """What the explored provider is paid, from the pre-round state."""


@dataclass(frozen=True)
class RunnerUpExploration:
    """Explore the runner-up by score, paid the bid it holds at the start of the round."""

    def choose(self, selection: Selection, u: float) -> tuple[str, int]:
        return pick_runner_up(selection, u), len(runner_up_set(selection))

    def payment(self, state: PlatformState, provider: str) -> float:
        bid = state.providers[provider].bid
        assert bid is not None
        return float(bid)


@dataclass(frozen=True)
class UniformExploration:
    """Explore a provider drawn uniformly from the candidate set without the winner, paid C_max."""

    def choose(self, selection: Selection, u: float) -> tuple[str, int]:
        if selection.halted or selection.winner is None:
            raise ExplorationError("a halted round has nothing to explore")
        if not 0.0 <= u < 1.0:
            raise ExplorationError(f"u must lie in [0, 1), got {u!r}")
        candidates = [j for j in selection.eligible if j != selection.winner]
        if not candidates:
            raise ExplorationError("the candidate set holds no provider besides the winner")
        return candidates[min(int(u * len(candidates)), len(candidates) - 1)], len(candidates)

    def payment(self, state: PlatformState, provider: str) -> float:
        return float(state.radii.c_max)


RULES = {"runner_up": RunnerUpExploration, "uniform": UniformExploration}


# --------------------------------------------------------------------------
# The count target of the round-replacing mode
# --------------------------------------------------------------------------


@runtime_checkable
class CountExploration(Protocol):
    def target(self, t: int, n: int) -> int:
        """The observation count every eligible provider must have reached before main round t."""


@dataclass(frozen=True)
class MinimumCountExploration:
    """Forced rounds for eligible providers below g(t) = ceil(scale * (t / n)^alpha), paid C_max.

    ``alpha`` in (0, 1) keeps the forced rounds sublinear in the horizon;
    alpha = 1/2 is the square-root target g(t) = ceil(scale * sqrt(t / n)).
    """

    scale: float = 1.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not self.scale > 0:
            raise ExplorationError(f"scale must be positive, got {self.scale!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ExplorationError(f"alpha must lie in (0, 1), got {self.alpha!r}")

    def target(self, t: int, n: int) -> int:
        if t < 1 or n < 1:
            raise ExplorationError(f"t and n must be >= 1, got t={t}, n={n}")
        return math.ceil(self.scale * (t / n) ** self.alpha)


def under_sampled(
    eligible: Sequence[str], counts: Mapping[str, int], target: int
) -> list[str]:
    """The eligible providers observed fewer than ``target`` times, in roster order."""
    return [i for i in eligible if counts[i] < target]


def pick_under_sampled(candidates: Sequence[str], u: float) -> str:
    """A uniform draw from the under-sampled set with ``u`` in [0, 1)."""
    if not candidates:
        raise ExplorationError("no under-sampled eligible provider to explore")
    if not 0.0 <= u < 1.0:
        raise ExplorationError(f"u must lie in [0, 1), got {u!r}")
    return candidates[min(int(u * len(candidates)), len(candidates) - 1)]


# --------------------------------------------------------------------------
# The single interface: one object per variant
# --------------------------------------------------------------------------

MODES = ("runner_up", "uniform", "count")


@dataclass(frozen=True)
class ForcedExploration:
    """One forced-exploration variant, ready for ``simulate_episode``.

    The additive modes carry a ``schedule`` (when a second service happens)
    and a ``rule`` (whom it goes to and what it pays); the ``count`` mode
    carries the ``count`` target and replaces the round instead.
    """

    mode: str
    schedule: ExplorationSchedule | None = None
    rule: ExplorationRule | None = None
    count: CountExploration | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ExplorationError(f"unknown exploration mode {self.mode!r}; known: {MODES}")
        if self.mode == "count":
            if self.count is None or self.schedule is not None or self.rule is not None:
                raise ExplorationError("the count mode takes a count target and nothing else")
        elif self.schedule is None or self.rule is None or self.count is not None:
            raise ExplorationError(f"the {self.mode} mode takes a schedule and a rule, no count target")

    @property
    def replaces_round(self) -> bool:
        """Whether an exploration replaces the mechanism's round (count) or adds a service to it."""
        return self.count is not None


def make_exploration(
    mode: str = "none", *, schedule: str = "sqrt", p: float = 0.10,
    scale: float = 1.0, alpha: float = 0.5,
) -> ForcedExploration | None:
    """The variant named by ``mode``, or ``None`` for the mechanism as published.

    ``schedule`` and ``p`` (constant schedule only) apply to the additive
    modes ``runner_up`` and ``uniform``; ``scale`` and ``alpha`` apply to
    ``count``, whose target is g(t) = ceil(scale * (t / n)^alpha).
    """
    if mode == "none":
        return None
    if mode == "count":
        return ForcedExploration("count", count=MinimumCountExploration(scale, alpha))
    if mode not in RULES:
        raise ExplorationError(f"unknown exploration mode {mode!r}; known: none, {', '.join(MODES)}")
    return ForcedExploration(mode, schedule=make_schedule(schedule, p=p), rule=RULES[mode]())


def describe(exploration: ForcedExploration | None) -> dict[str, object]:
    """A JSON-safe description of a variant, for manifests and records."""
    if exploration is None:
        return {"mode": "none"}
    out: dict[str, object] = {"mode": exploration.mode}
    if exploration.schedule is not None:
        name = next((k for k, v in SCHEDULES.items() if isinstance(exploration.schedule, v)), None)
        out["schedule"] = name if name is not None else type(exploration.schedule).__name__
        if isinstance(exploration.schedule, ConstantExploration):
            out["p"] = exploration.schedule.p
    if exploration.count is not None:
        out.update(asdict(exploration.count) if isinstance(exploration.count, MinimumCountExploration)
                   else {"count": type(exploration.count).__name__})
    return out
