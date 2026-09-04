"""Provider private state: own cost history, estimate and standing bid.

A provider knows only its own costs, kept as the sufficient statistics
(m, cost_sum); nothing about rounds, competitors or payments. Its bid is set
on its first selection and revised only after a round in which it was
selected, so observe_cost is the only mutator and is called only
for the winner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from cri.estimators import EstimatorConfig


class ProviderError(ValueError):
    """An illegal provider update, or a cost outside [0, C_max]."""


@dataclass(frozen=True)
class ProviderState:
    model: str
    m: int = 0
    cost_sum: float = 0.0
    estimate: float | None = None
    bid: float | None = None

    @property
    def c_hat(self) -> float:
        """Empirical mean cost over this provider's own selections."""
        if self.m < 1:
            raise ProviderError(f"{self.model}: no cost observed yet")
        return self.cost_sum / self.m

    @property
    def has_bid(self) -> bool:
        return self.bid is not None


def bid_from_estimate(estimate: float, state: ProviderState) -> float:
    """Cost-estimate bidding: bid the current estimate."""
    return estimate


#: (estimate, state) -> standing bid. A rule that shades the estimate within
#: [-d_-, d_+] is approximate bidding.
BidRule = Callable[[float, ProviderState], float]


def initial_state(model: str) -> ProviderState:
    """No history and no bid: a bid is read only after the first selection."""
    return ProviderState(model=model)


def init_providers(roster: Sequence[str]) -> dict[str, ProviderState]:
    roster = list(roster)
    if len(set(roster)) != len(roster):
        raise ProviderError(f"duplicate model in roster: {roster}")
    return {m: initial_state(m) for m in roster}


def observe_cost(
    state: ProviderState,
    cost: float,
    *,
    estimator: EstimatorConfig,
    c_max: float,
    bid_rule: BidRule = bid_from_estimate,
) -> ProviderState:
    """One selection: observe the cost, re-estimate, re-bid."""
    if not 0.0 <= cost <= c_max:
        raise ProviderError(
            f"{state.model}: cost {cost!r} outside [0, {c_max}] -- C_max is misconfigured "
            "or the cost process is not the one the mechanism was told about"
        )
    m = state.m + 1
    cost_sum = state.cost_sum + float(cost)
    estimate = estimator.estimate(m, cost_sum / m, c_max)
    if not 0.0 <= estimate <= c_max:
        raise ProviderError(
            f"{state.model}: estimate {estimate!r} outside [0, {c_max}]"
        )
    bid = bid_rule(estimate, state)
    if not 0.0 <= bid <= c_max:
        raise ProviderError(f"{state.model}: bid {bid!r} outside [0, {c_max}]")
    return replace(state, m=m, cost_sum=cost_sum, estimate=estimate, bid=bid)
