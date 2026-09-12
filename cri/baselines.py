"""Baselines and ablations that share the mechanism's policy interface.

A policy maps ``(platform_state, tie_draw)`` to a ``Selection``. Oracle
policies receive ground truth explicitly; it is never added to platform state.

The eligible-set baselines keep the mechanism's quality filter A_t and
replace its score-based selection with a uniform draw or with listed prices.
``invoice_lcb_eligible`` is a diagnostic only: in the recorded data invoices
are proportional to generation cost, so it is left out of the main comparison.

Critical payments apply only when the winner minimizes the score over its
candidate set. Other policies run unpaid; the raw critical payment is still
logged for them. ``experiment.py`` resolves the arms that change the radii,
the estimator, the query stream or the round protocol (the forced-exploration
variants) instead of the selection rule.
"""

from __future__ import annotations

from collections.abc import Sequence

from cri.mechanism import Selection, select
from cri.variants import price_rate
from cri.platform import PlatformState
from cri.simulate import PaymentRule, Policy, critical_payment_rule
from cri.metrics import GroundTruth


class BaselineError(ValueError):
    """Unknown policy or payment rule."""


def _pick(candidates: Sequence[str], tie_u: float) -> str:
    """Uniform choice with the round's draw; same indexing as mechanism.select."""
    return candidates[min(int(tie_u * len(candidates)), len(candidates) - 1)]


def _selection(
    state: PlatformState, candidates: Sequence[str], winner: str
) -> Selection:
    return Selection(
        winner=winner, eligible=tuple(candidates), scores=state.scores(), tied=(winner,)
    )


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def oracle_cheapest_qualified(truth: GroundTruth) -> Policy:
    """Always i*, with the true Q as candidate set."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        return _selection(state, truth.qualified, truth.istar)

    return policy


def oracle_quality_random(truth: GroundTruth) -> Policy:
    """Uniform over the true qualified set."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        candidates = list(truth.qualified)
        return _selection(state, candidates, _pick(candidates, tie_u))

    return policy


def uniform_random() -> Policy:
    """Uniform over the whole roster."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        roster = state.roster
        return _selection(state, roster, _pick(roster, tie_u))

    return policy


def quality_greedy() -> Policy:
    """Highest quality UCB among the eligible, cost ignored."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        sel = select(state, tie_u)
        if sel.halted:
            return sel
        best = max(state.q_ucb(m) for m in sel.eligible)
        top = [m for m in sel.eligible if state.q_ucb(m) == best]
        return _selection(state, sel.eligible, _pick(top, tie_u))

    return policy


def cheapest_price() -> Policy:
    """Lowest advertised per-token rate (times N), no quality filter, no bids needed."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        roster = state.roster
        cheapest = min(price_rate(m) for m in roster)
        top = [m for m in roster if price_rate(m) == cheapest]
        return _selection(state, roster, _pick(top, tie_u))

    return policy


def cheapest_bid() -> Policy:
    """Lowest standing bid over all providers: no quality filter, no radius."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        roster = state.roster
        bids = {m: state.providers[m].bid for m in roster}
        best = min(bids.values())
        top = [m for m in roster if bids[m] == best]
        return _selection(state, roster, _pick(top, tie_u))

    return policy


def uniform_eligible() -> Policy:
    """Uniform over A_t: the mechanism's quality filter with no cost information."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        sel = select(state, tie_u)
        if sel.halted:
            return sel
        return _selection(state, sel.eligible, _pick(list(sel.eligible), tie_u))

    return policy


def cheapest_rate_eligible() -> Policy:
    """Lowest advertised rate in A_t: public prices only, once quality is learned."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        sel = select(state, tie_u)
        if sel.halted:
            return sel
        cheapest = min(price_rate(m) for m in sel.eligible)
        top = [m for m in sel.eligible if price_rate(m) == cheapest]
        return _selection(state, sel.eligible, _pick(top, tie_u))

    return policy


def invoice_lcb_eligible() -> Policy:
    """argmin_{i in A_t} invoice_hat_i - rho_c(m_i): a diagnostic.

    It uses the invoices it has paid (tokens times listed rate) with the
    mechanism's cost radius. In the recorded data generation cost is that same
    invoice deflated by the margin, so this is not a main baseline.
    """

    def policy(state: PlatformState, tie_u: float) -> Selection:
        sel = select(state, tie_u)
        if sel.halted:
            return sel
        lcb = {
            m: state.invoice_hat(m) - state.radii.rho_c(state.providers[m].m)
            for m in sel.eligible
        }
        best = min(lcb.values())
        top = [m for m in sel.eligible if lcb[m] == best]
        return _selection(state, sel.eligible, _pick(top, tie_u))

    return policy


# --------------------------------------------------------------------------
# Ablations: one mechanism ingredient removed at a time
# --------------------------------------------------------------------------


def greedy_cheapest_qualified() -> Policy:
    """argmin_{i in A_t} b_i(t): the mechanism without the -rho(m_i) term."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        sel = select(state, tie_u)
        if sel.halted:
            return sel
        bids = {m: state.providers[m].bid for m in sel.eligible}
        best = min(bids.values())
        top = [m for m in sel.eligible if bids[m] == best]
        return _selection(state, sel.eligible, _pick(top, tie_u))

    return policy


def no_quality_filter() -> Policy:
    """The mechanism's score rule over the whole roster: A_t = N."""

    def policy(state: PlatformState, tie_u: float) -> Selection:
        roster = state.roster
        scores = state.scores()
        best = min(scores[m] for m in roster)
        top = [m for m in roster if scores[m] == best]
        return _selection(state, roster, _pick(top, tie_u))

    return policy


def pay_your_bid(state: PlatformState, selection: Selection) -> float:
    """First price: the winner is paid its own bid."""
    assert selection.winner is not None
    bid = state.providers[selection.winner].bid
    assert bid is not None
    return float(bid)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

#: Policies that take the ground truth as an argument.
ORACLES = {
    "oracle_cheapest_qualified": oracle_cheapest_qualified,
    "oracle_quality_random": oracle_quality_random,
}

#: Policies that see only the public state.
PLAIN = {
    "mechanism": lambda: select,
    "uniform_random": uniform_random,
    "quality_greedy": quality_greedy,
    "cheapest_bid": cheapest_bid,
    "uniform_eligible": uniform_eligible,
    "cheapest_rate_eligible": cheapest_rate_eligible,
    "invoice_lcb_eligible": invoice_lcb_eligible,
    "greedy_cheapest_qualified": greedy_cheapest_qualified,
    "no_quality_filter": no_quality_filter,
    "cheapest_price": cheapest_price,
}

POLICY_NAMES = frozenset(PLAIN) | frozenset(ORACLES)
EXPERIMENT_ABLATIONS = frozenset(
    {"pay_your_bid", "gamma_zero", "biased_beliefs", "independent_cost_stream"}
)
#: Forced-exploration variants of the round protocol (exploration.py), one arm per
#: mode, named "<mode>_exploration"; resolved by ``experiment.py`` like the
#: ablations, not part of the paper's comparison.
EXPERIMENT_VARIANTS = frozenset({"runner_up_exploration", "uniform_exploration", "count_exploration"})

#: Policies whose winner is the score-argmin of its candidate set; only these
#: carry the critical payment (the threshold lemma).
PAYMENT_APPLIES = frozenset({"mechanism", "no_quality_filter"})

PAYMENT_RULES = {
    "critical": critical_payment_rule,
    "pay_your_bid": pay_your_bid,
}


def make_policy(name: str, truth: GroundTruth | None = None) -> Policy:
    """Policy by name; oracles require ``truth``."""
    if name in PLAIN:
        return PLAIN[name]()
    if name in ORACLES:
        if truth is None:
            raise BaselineError(f"{name} needs an oracle; pass truth")
        return ORACLES[name](truth)
    raise BaselineError(f"unknown policy {name!r}; known: " f"{sorted(POLICY_NAMES)}")


def make_payment_rule(name: str | None = "critical") -> PaymentRule | None:
    """Payment rule by name; ``None`` means the arm is unpaid."""
    if name is None:
        return None
    try:
        return PAYMENT_RULES[name]
    except KeyError:
        raise BaselineError(
            f"unknown payment rule {name!r}; known: {sorted(PAYMENT_RULES)}"
        ) from None
