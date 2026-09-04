"""Platform state: what the platform observes and derives.

Per provider it keeps the selection count, the correctness sum and the
standing bid; it never sees a cost. Derived pre-round quantities:

    x_i(t)   = b_i(t) - rho(m_i(t))                score
    q_UCB(t) = min{1, q_hat_i(t) + beta(m_i(t))}
    A_t      = {i : q_UCB_i(t) >= Theta}           eligible set

record_selection runs at the end of a round, so every read at the start of
the next one uses m_i(t) = #{s < t : I_s = i}.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from cri.confidence import Radii


class PlatformError(ValueError):
    """An illegal platform update, or a read that the protocol forbids."""


@dataclass(frozen=True)
class PlatformProvider:
    model: str
    m: int = 0
    correct_sum: int = 0
    bid: float | None = None

    @property
    def q_hat(self) -> float:
        if self.m < 1:
            raise PlatformError(f"{self.model}: no answer evaluated yet")
        return self.correct_sum / self.m


@dataclass(frozen=True)
class PlatformState:
    providers: Mapping[str, PlatformProvider]
    radii: Radii
    theta: float

    def __post_init__(self) -> None:
        if not 0.0 < self.theta < 1.0:
            raise PlatformError(f"Theta must lie in (0, 1), got {self.theta!r}")
        if len(self.providers) != self.radii.n:
            raise PlatformError(
                f"radii were built for n={self.radii.n} but the roster has "
                f"{len(self.providers)} providers"
            )

    @property
    def roster(self) -> list[str]:
        return list(self.providers)

    def _provider(self, model: str) -> PlatformProvider:
        try:
            return self.providers[model]
        except KeyError:
            raise PlatformError(f"{model!r} is not in this roster") from None

    def _ready(self, model: str) -> PlatformProvider:
        p = self._provider(model)
        if p.m < 1 or p.bid is None:
            raise PlatformError(
                f"{model}: no standing bid yet -- initialization must select every "
                "provider once before any round is scored"
            )
        return p

    def q_hat(self, model: str) -> float:
        return self._provider(model).q_hat

    def q_ucb(self, model: str) -> float:
        p = self._ready(model)
        return self.radii.quality_ucb(p.q_hat, p.m)

    def score(self, model: str) -> float:
        p = self._ready(model)
        return p.bid - self.radii.rho(p.m)

    def scores(self) -> dict[str, float]:
        return {m: self.score(m) for m in self.providers}

    def eligible(self) -> list[str]:
        return [m for m in self.providers if self.q_ucb(m) >= self.theta]

    def selection_counts(self) -> dict[str, int]:
        return {m: p.m for m, p in self.providers.items()}


def init_platform(roster: Sequence[str], radii: Radii, theta: float) -> PlatformState:
    roster = list(roster)
    if len(set(roster)) != len(roster):
        raise PlatformError(f"duplicate model in roster: {roster}")
    providers = {m: PlatformProvider(model=m) for m in roster}
    return PlatformState(
        providers=MappingProxyType(providers), radii=radii, theta=theta
    )


def record_selection(
    state: PlatformState, model: str, correctness: int, new_bid: float
) -> PlatformState:
    """End of a round: record the evaluated answer and the winner's new bid."""
    p = state._provider(model)
    if correctness not in (0, 1):
        raise PlatformError(f"correctness must be 0 or 1, got {correctness!r}")
    if not 0.0 <= new_bid <= state.radii.c_max:
        raise PlatformError(
            f"{model}: bid {new_bid!r} outside [0, {state.radii.c_max}]"
        )
    updated = dict(state.providers)
    updated[model] = replace(
        p, m=p.m + 1, correct_sum=p.correct_sum + int(correctness), bid=float(new_bid)
    )
    return replace(state, providers=MappingProxyType(updated))
