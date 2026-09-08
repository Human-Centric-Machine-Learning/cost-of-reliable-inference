"""Empirical metrics and theoretical quantities derived from round logs.

Ground truth is available only to evaluation and oracle policies. The module
computes regret, payments, provider payoffs, selection statistics,
confidence-event diagnostics, belief slack, elimination counts, and both
branches of the paper's bounds. Empirical means are rebuilt from the log,
keeping ground truth out of the simulator.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cri.confidence import Lambda, Radii
from cri.data import BenchmarkData
from cri.simulate import EpisodeLog
from cri.pricing import invoice as public_invoice
from cri.variants import expected_cost, expected_price, expected_quality, parse_variant


class MetricsError(ValueError):
    """An environment that violates a standing assumption of the paper."""


# --------------------------------------------------------------------------
# Evaluation-only ground truth
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """The paper's unknown parameters; used by metrics and oracle policies only."""

    benchmark: str
    roster: tuple[str, ...]
    theta: float
    q: dict[str, float]
    c: dict[str, float]
    qualified: tuple[str, ...]
    istar: str
    c_star: float
    c_second: float
    delta_gap: float
    eps: dict[str, float]  # Theta - q_j, for unqualified j
    delta_j: dict[str, float]  # c_j - c_i*, for qualified j != i*
    price: dict[str, float] = field(default_factory=dict)  # expected list price per query

    @property
    def n(self) -> int:
        return len(self.roster)

    @property
    def cheapest_overall(self) -> str:
        return min(self.roster, key=lambda m: self.c[m])

    @property
    def trap(self) -> bool:
        """True when the cheapest provider overall is unqualified."""
        return self.cheapest_overall not in self.qualified


def ground_truth(
    data: BenchmarkData, roster: Sequence[str], theta: float, margin: float
) -> GroundTruth:
    """Compute q_i, c_i, and the derived quantities for one roster and Theta."""
    roster = list(roster)
    variants = {m: parse_variant(m) for m in roster}
    q = {
        m: (
            expected_quality(data.records[v.base], v.n)
            if v.n > 1
            else float(data.records[v.base].correctness.mean())
        )
        for m, v in variants.items()
    }
    c = {
        m: expected_cost(data.records[v.base], v.n, margin) for m, v in variants.items()
    }
    price = {m: expected_price(data.records[v.base], v.n) for m, v in variants.items()}

    qualified = [m for m in roster if q[m] >= theta]
    if len(qualified) < 2:
        raise MetricsError(
            f"{data.benchmark} at Theta={theta}: |Q|={len(qualified)}; the paper "
            "assumes at least two qualified providers and the mechanism halts otherwise"
        )
    istar = min(qualified, key=lambda m: c[m])
    contenders = [m for m in qualified if m != istar]
    c_second = min(c[m] for m in contenders)
    if sum(1 for m in qualified if math.isclose(c[m], c[istar], rel_tol=1e-12)) > 1:
        raise MetricsError(
            f"{data.benchmark} at Theta={theta}: the cheapest qualified provider is "
            "not unique, which the paper assumes"
        )
    return GroundTruth(
        benchmark=data.benchmark,
        roster=tuple(roster),
        theta=theta,
        q=q,
        c=c,
        qualified=tuple(qualified),
        istar=istar,
        c_star=c[istar],
        c_second=c_second,
        delta_gap=c_second - c[istar],
        eps={m: theta - q[m] for m in roster if q[m] < theta},
        delta_j={m: c[m] - c[istar] for m in contenders},
        price=price,
    )


# --------------------------------------------------------------------------
# Theoretical reference quantities
# --------------------------------------------------------------------------


def M_count(eps: float, n: int, delta: float) -> int:
    """Selection bound for an unqualified provider with quality gap eps."""
    if eps <= 0:
        raise MetricsError(f"quality gap must be positive, got {eps!r}")
    return math.ceil(4 / eps**2 * math.log(57 * n / (delta * eps**4)))


def L_count(gap: float, n: int, delta: float, c_max: float, gamma: float = 0.0) -> int:
    """Selection bound for a qualified provider with cost gap Delta_j."""
    if gap <= 0:
        raise MetricsError(f"cost gap must be positive, got {gap!r}")
    g = 1.0 + gamma
    return math.ceil(
        4
        * g**2
        * c_max**2
        / gap**2
        * math.log(57 * n * g**4 * c_max**4 / (delta * gap**4))
    )


@dataclass(frozen=True)
class TheoryConstants:
    """B_id and the horizon n + 2 B_id that the recovery corollary needs."""

    sum_M: int
    sum_L: int
    B_id: int
    need: int  # n + 2 B_id
    stream: int | None = None

    @property
    def reachable(self) -> bool | None:
        return None if self.stream is None else self.need < self.stream


def theory_constants(
    truth: GroundTruth, radii: Radii, stream: int | None = None
) -> TheoryConstants:
    sum_M = sum(M_count(e, radii.n, radii.delta) for e in truth.eps.values())
    sum_L = sum(
        L_count(g, radii.n, radii.delta, radii.c_max, radii.gamma)
        for g in truth.delta_j.values()
    )
    b = sum_M + sum_L
    return TheoryConstants(
        sum_M=sum_M, sum_L=sum_L, B_id=b, need=radii.n + 2 * b, stream=stream
    )


@dataclass(frozen=True)
class Bounds:
    """Both branches of each theorem's min, at horizon T."""

    quality_anytime: float
    quality_gap: float
    generation_anytime: float
    generation_gap: float
    payment_anytime: float

    @property
    def quality(self) -> float:
        return min(self.quality_anytime, self.quality_gap)

    @property
    def generation(self) -> float:
        return min(self.generation_anytime, self.generation_gap)


def bounds(T: int, truth: GroundTruth, radii: Radii) -> Bounds:
    """Evaluate the regret and payment bounds at horizon T."""
    n = radii.n
    if T < n:
        raise MetricsError(f"T={T} is below n={n}")
    root = math.sqrt(2 * n * (T - n) * Lambda(max(T, 1), n, radii.delta))
    widened = (1.0 + radii.gamma) * radii.c_max

    q_gap = sum((M_count(e, n, radii.delta) + 1) * e for e in truth.eps.values())
    g_gap = sum(
        (M_count(truth.eps[m], n, radii.delta) + 1)
        * max(truth.c[m] - truth.c_star, 0.0)
        for m in truth.eps
    ) + sum(
        (L_count(g, n, radii.delta, radii.c_max, radii.gamma) + 1) * g
        for m, g in truth.delta_j.items()
    )
    return Bounds(
        quality_anytime=n + 2 * root,
        quality_gap=q_gap,
        generation_anytime=(n - 1) * radii.c_max + 2 * widened * root,
        generation_gap=g_gap,
        payment_anytime=n * (radii.c_max - truth.c_second) + widened * root,
    )


# --------------------------------------------------------------------------
# Empirical metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeMetrics:
    """Episode-level metrics; payment fields are None for arms without a payment rule."""

    rounds: int
    quality_regret: float
    realized_accuracy: float
    generation_regret: float
    total_cost: float
    total_invoice: float  # public invoices, tokens x advertised rate: what a list-price platform pays
    unqualified_rounds: int  # rounds served by a provider below Theta
    excess_payment: float | None
    excess_payment_init: float | None
    excess_payment_post: float | None
    total_payment: float | None
    init_payment: float | None
    provider_payoff: dict[str, float] | None  # sum(pi - C) per provider
    min_provider_payoff: float | None
    critical_below_bid_rounds: int | None  # main rounds with the critical payment below the bid
    critical_shortfall: float | None
    selection_counts: dict[str, int]
    istar_share: float
    non_istar_rounds: int
    last_non_istar_round: int | None
    identifies_istar: bool | None
    argmax_is_istar: bool
    good_event_holds: bool | None
    good_event_violations: int | None
    max_slack_ratio: float | None


def _by_provider(log: EpisodeLog) -> dict[str, dict[str, np.ndarray]]:
    """Per-provider views of the log, in the order that provider was selected."""
    col = log.columns()
    out: dict[str, dict[str, np.ndarray]] = {}
    for model in dict.fromkeys(col["winner"].tolist()):
        mask = col["winner"] == model
        out[model] = {
            k: col[k][mask] for k in ("cost", "correctness", "bid_winner", "t")
        }
    return out


def good_event_report(
    log: EpisodeLog, truth: GroundTruth, radii: Radii
) -> tuple[bool, int]:
    """Whether the good event held at every selection count of every provider."""
    violations = 0
    for model, view in _by_provider(log).items():
        m = np.arange(1, len(view["cost"]) + 1)
        beta = radii.beta_table(len(m))[1:]
        q_hat = np.cumsum(view["correctness"]) / m
        c_hat = np.cumsum(view["cost"]) / m
        violations += int(np.sum(np.abs(q_hat - truth.q[model]) > beta))
        violations += int(np.sum(np.abs(c_hat - truth.c[model]) > radii.c_max * beta))
    return violations == 0, violations


def slack_report(log: EpisodeLog, radii: Radii) -> dict[str, float]:
    """Largest realized |e_i - c_hat_i| / rho_c(m) per provider (belief slack).

    The bid at a provider's (k+1)-st win is the estimate formed after its k-th.
    """
    out: dict[str, float] = {}
    for model, view in _by_provider(log).items():
        costs, bids = view["cost"], view["bid_winner"]
        if len(costs) < 2:
            out[model] = 0.0
            continue
        k = np.arange(1, len(costs))
        c_hat = np.cumsum(costs)[:-1] / k
        rho_c = radii.c_max * radii.beta_table(len(k))[1:]
        out[model] = float(np.max(np.abs(bids[1:] - c_hat) / rho_c))
    return out


def window(log: EpisodeLog, start: int = 1, end: int | None = None) -> EpisodeLog:
    """Rounds in [start, end]; diagnostics that need the full prefix become None."""
    col = log.columns()
    end = int(col["t"].max()) if end is None else end
    keep = (col["t"] >= start) & (col["t"] <= end)
    out = EpisodeLog(source_start=max(log.source_start, start))
    for key, values in log.columns().items():
        getattr(out, key).extend(np.asarray(values)[keep].tolist())
    return out


def cumulative_series(
    log: EpisodeLog, truth: GroundTruth, radii: Radii | None = None
) -> dict[str, np.ndarray]:
    """Cumulative metric paths, for checkpoints and figures."""
    col = log.columns()
    q = np.array([truth.q[m] for m in col["winner"]])
    c = np.array([truth.c[m] for m in col["winner"]])
    elapsed = np.arange(1, len(col["t"]) + 1, dtype=float)
    payment = col["payment"].astype(float)  # all NaN for an arm without a rule
    invoice = np.array(
        [
            public_invoice(model, tokens)
            for model, tokens in zip(col["winner"], col["num_tokens"])
        ],
        dtype=float,
    )
    unqualified = np.array(
        [model not in truth.qualified for model in col["winner"]], dtype=int
    )
    out = {
        "t": col["t"],
        "quality_regret": np.cumsum(np.maximum(truth.theta - q, 0.0)),
        "generation_regret": np.cumsum(np.maximum(c - truth.c_star, 0.0)),
        "total_cost": np.cumsum(col["cost"]),
        "total_invoice": np.cumsum(invoice),
        "unqualified_rounds": np.cumsum(unqualified),
        "excess_payment": np.cumsum(np.maximum(payment - truth.c_second, 0.0)),
        "total_payment": np.cumsum(payment),
        "provider_payoff": np.cumsum(payment - col["cost"]),
        "accuracy": np.cumsum(col["correctness"]) / elapsed,
        "istar_share": np.cumsum(col["winner"] == truth.istar) / elapsed,
    }
    if radii is not None:
        names = (
            "quality_anytime",
            "quality_gap",
            "generation_anytime",
            "generation_gap",
            "payment_anytime",
        )
        paths = {name: np.full(len(elapsed), np.nan) for name in names}
        for idx, t in enumerate(col["t"].astype(int)):
            if t < radii.n:
                continue
            value = bounds(t, truth, radii)
            for name in names:
                paths[name][idx] = getattr(value, name)
        out.update(paths)
    return out


def evaluate(log: EpisodeLog, truth: GroundTruth, radii: Radii) -> EpisodeMetrics:
    """All episode-level metrics from the log and the ground truth."""
    col = log.columns()
    if len(col["t"]) == 0:
        raise MetricsError("empty round log")
    is_init = col["phase"] == "init"
    q = np.array([truth.q[m] for m in col["winner"]])
    c = np.array([truth.c[m] for m in col["winner"]])
    payment = col["payment"].astype(float)
    paid = not np.isnan(payment).any()
    if paid:
        excess = np.maximum(payment - truth.c_second, 0.0)
        payoff = {
            m: float((payment - col["cost"])[col["winner"] == m].sum())
            for m in truth.roster
        }

    # Raw critical payment against the winner's bid, main rounds only; None if the
    # column is absent (hand-built logs).
    if len(col["critical_payment"]) == len(col["t"]):
        shortfall = np.where(
            is_init,
            0.0,
            np.maximum(
                col["bid_winner"].astype(float) - col["critical_payment"].astype(float),
                0.0,
            ),
        )
        below_rounds, below_total = int(np.sum(shortfall > 1e-9)), float(
            np.nansum(shortfall)
        )
    else:
        below_rounds, below_total = None, None

    counts = {m: int((col["winner"] == m).sum()) for m in truth.roster}
    non_istar = col["winner"] != truth.istar
    is_complete_prefix = log.source_start == 1 and int(col["t"][0]) == 1
    theory = theory_constants(truth, radii)
    if is_complete_prefix:
        holds, violations = good_event_report(log, truth, radii)
        slack = max(slack_report(log, radii).values(), default=0.0)
    else:
        holds, violations, slack = None, None, None
    other_counts = [v for m, v in counts.items() if m != truth.istar]

    return EpisodeMetrics(
        rounds=len(col["t"]),
        quality_regret=float(np.sum(np.maximum(truth.theta - q, 0.0))),
        realized_accuracy=float(col["correctness"].mean()),
        generation_regret=float(np.sum(np.maximum(c - truth.c_star, 0.0))),
        total_cost=float(col["cost"].sum()),
        total_invoice=float(sum(public_invoice(m, k) for m, k in zip(col["winner"], col["num_tokens"]))),
        unqualified_rounds=int(sum(v for m, v in counts.items() if m not in truth.qualified)),
        excess_payment=float(excess.sum()) if paid else None,
        excess_payment_init=float(excess[is_init].sum()) if paid else None,
        excess_payment_post=float(excess[~is_init].sum()) if paid else None,
        total_payment=float(payment.sum()) if paid else None,
        init_payment=float(payment[is_init].sum()) if paid else None,
        provider_payoff=payoff if paid else None,
        min_provider_payoff=min(payoff.values()) if paid else None,
        critical_below_bid_rounds=below_rounds,
        critical_shortfall=below_total,
        selection_counts=counts,
        istar_share=float((~non_istar).mean()),
        non_istar_rounds=int(non_istar.sum()),
        last_non_istar_round=int(col["t"][non_istar][-1]) if non_istar.any() else None,
        identifies_istar=(
            all(  # the recovery corollary
                (v > theory.B_id + 1) == (m == truth.istar) for m, v in counts.items()
            )
            if is_complete_prefix
            else None
        ),
        argmax_is_istar=counts[truth.istar] > max(other_counts, default=-1),
        good_event_holds=holds,
        good_event_violations=violations,
        max_slack_ratio=slack,
    )
