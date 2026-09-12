"""Empirical metrics and theoretical quantities derived from round logs.

Ground truth is available only to evaluation and oracle policies. The module
computes regret, payments, provider payoffs, selection statistics,
confidence-event diagnostics, belief slack, elimination counts, and both
branches of the paper's bounds. Empirical means are rebuilt from the log,
keeping ground truth out of the simulator.

Under a forced-exploration variant the round log holds one row per round:
the normal selections, plus the forced rounds of the count-based variant. An
additive variant's extra services live in the episode's ``ExplorationLog``;
``exploration_events`` returns either kind in that layout, and
``exploration_metrics`` totals them. The diagnostics that rebuild a
provider's estimates (``good_event_report``, ``slack_report``, and through
them ``evaluate``) take the additional-service log as well, since the
platform's counts and the providers' bids include those observations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cri.confidence import Lambda, Radii
from cri.data import BenchmarkData
from cri.exploration import ForcedExploration
from cri.simulate import EpisodeLog, EpisodeResult, ExplorationLog, forced_rounds
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


def forced_cap(T: int, n: int, exploration: ForcedExploration | None = None) -> int:
    """G_T = g(T) - 1, the most forced rounds one provider can be served by round T.

    A forced round for j at round t needs m_j(t) < g(t) <= g(T), and successive
    services of j have distinct pre-round counts. Zero without exploration; the
    additive variants have no derived bound and are refused.
    """
    if exploration is None:
        return 0
    if not exploration.replaces_round or exploration.count is None:
        raise MetricsError(
            f"no regret bound is derived for the {exploration.mode!r} exploration mode"
        )
    return exploration.count.target(T, n) - 1


def selection_caps(
    T: int, truth: GroundTruth, radii: Radii, exploration: ForcedExploration | None = None
) -> dict[str, int]:
    """Cap on each provider j != i*'s post-initialization services by round T, forced rounds included.

    M(eps_j) for an unqualified provider (forced rounds are drawn from A_t) and
    max{L(Delta_j), G_T} for a qualified competitor (ordinary selections within
    L(Delta_j), forced rounds within G_T); M and L without exploration.
    """
    n = radii.n
    G = forced_cap(T, n, exploration)
    caps = {m: M_count(e, n, radii.delta) for m, e in truth.eps.items()}
    caps.update(
        {
            m: max(L_count(g, n, radii.delta, radii.c_max, radii.gamma), G)
            for m, g in truth.delta_j.items()
        }
    )
    return caps


def B_id_at(
    T: int, truth: GroundTruth, radii: Radii, exploration: ForcedExploration | None = None
) -> int:
    """B_id(T): the identification bound at horizon T, the sum of the selection caps."""
    return sum(selection_caps(T, truth, radii, exploration).values())


@dataclass(frozen=True)
class TheoryConstants:
    """B_id, its finite-horizon version and the identification horizon of the recovery corollary.

    ``sum_M``, ``sum_L`` and ``B_id`` are the lemma constants, which do not
    depend on the horizon or on exploration. ``horizon`` is T_id, the smallest
    T from which T > n + 2 B_id(T) holds at every later round, so that the
    recovery corollary applies from T_id on; ``need`` is the round before it,
    n + 2 B_id(T_id), which is n + 2 B_id without exploration.
    """

    sum_M: int
    sum_L: int
    B_id: int
    need: int  # n + 2 B_id(T_id) = T_id - 1
    stream: int | None = None
    horizon: int = 0  # T_id; set by theory_constants
    B_id_horizon: int = 0  # B_id(T_id); equals B_id without exploration

    @property
    def reachable(self) -> bool | None:
        return None if self.stream is None else self.need < self.stream


def identification_horizon(
    truth: GroundTruth, radii: Radii, exploration: ForcedExploration | None = None
) -> int:
    """T_id: the smallest T from which T > n + 2 B_id(T) holds at every later round.

    Without exploration T_id = n + 2 B_id + 1. With the count-based variant
    B_id(T) steps up by at most |Q| - 1 whenever g(T) does, so the first
    solution is found by fixed-point iteration from below and the rounds after
    it are scanned: the margin T - n - 2 B_id(T) gains one per round and loses
    at most 2(|Q| - 1) per step of g, whose steps grow further apart (g is
    concave), so once the margin reaches w = 2(|Q| - 1) + 1 and the last two
    steps were more than w rounds apart no later step can break the inequality.
    """
    n = radii.n
    plain = sum(M_count(e, n, radii.delta) for e in truth.eps.values()) + sum(
        L_count(g, n, radii.delta, radii.c_max, radii.gamma) for g in truth.delta_j.values()
    )
    if exploration is None:
        return n + 2 * plain + 1
    count = exploration.count
    assert count is not None

    def B(t: int) -> int:
        return B_id_at(t, truth, radii, exploration)

    t = n + 2 * plain + 1  # B_id(t) >= B_id, so no smaller round can qualify
    while (t_next := n + 2 * B(t) + 1) > t:
        t = t_next
    w = 2 * len(truth.delta_j) + 1
    horizon, u = t, t
    last_step, spacing, g_prev = None, 0, count.target(t, n)
    while True:
        g_u = count.target(u, n)
        if g_u != g_prev:
            spacing = u - last_step if last_step is not None else 0
            last_step, g_prev = u, g_u
        if not u > n + 2 * B(u):
            horizon = u + 1
        if spacing > w + 1 and u - n - 2 * B(u) >= w:
            return horizon
        u += 1


def theory_constants(
    truth: GroundTruth,
    radii: Radii,
    stream: int | None = None,
    exploration: ForcedExploration | None = None,
) -> TheoryConstants:
    sum_M = sum(M_count(e, radii.n, radii.delta) for e in truth.eps.values())
    sum_L = sum(
        L_count(g, radii.n, radii.delta, radii.c_max, radii.gamma)
        for g in truth.delta_j.values()
    )
    b = sum_M + sum_L
    horizon = identification_horizon(truth, radii, exploration)
    return TheoryConstants(
        sum_M=sum_M,
        sum_L=sum_L,
        B_id=b,
        need=horizon - 1,
        stream=stream,
        horizon=horizon,
        B_id_horizon=B_id_at(horizon, truth, radii, exploration),
    )


@dataclass(frozen=True)
class Bounds:
    """Both branches of each theorem's min, at horizon T.

    ``payment_anytime`` bounds the excess payment over initialization and the
    ordinary rounds; ``payment_anytime_forced`` adds the forced rounds, which
    pay C_max. ``forced_cap`` is G_T; both extras vanish without exploration.
    """

    quality_anytime: float
    quality_gap: float
    generation_anytime: float
    generation_gap: float
    payment_anytime: float
    payment_anytime_forced: float = float("nan")
    forced_cap: int = 0

    @property
    def quality(self) -> float:
        return min(self.quality_anytime, self.quality_gap)

    @property
    def generation(self) -> float:
        return min(self.generation_anytime, self.generation_gap)


def bounds(
    T: int, truth: GroundTruth, radii: Radii, exploration: ForcedExploration | None = None
) -> Bounds:
    """Evaluate the regret and payment bounds at horizon T.

    With the count-based forced exploration the quality bound is unchanged,
    the generation bound's anytime branch gains (n - 1) G_T C_max for the forced
    rounds and its gap branch uses the caps max{L(Delta_j), G_T}, and the
    excess payment including the forced rounds gains n G_T (C_max - c^(2)).
    """
    n = radii.n
    if T < n:
        raise MetricsError(f"T={T} is below n={n}")
    root = math.sqrt(2 * n * (T - n) * Lambda(max(T, 1), n, radii.delta))
    widened = (1.0 + radii.gamma) * radii.c_max
    G = forced_cap(T, n, exploration)
    caps = selection_caps(T, truth, radii, exploration)

    q_gap = sum((M_count(e, n, radii.delta) + 1) * e for e in truth.eps.values())
    g_gap = sum(
        (caps[m] + 1) * max(truth.c[m] - truth.c_star, 0.0) for m in truth.eps
    ) + sum((caps[m] + 1) * g for m, g in truth.delta_j.items())
    payment = n * (radii.c_max - truth.c_second) + widened * root
    return Bounds(
        quality_anytime=n + 2 * root,
        quality_gap=q_gap,
        generation_anytime=(n - 1) * radii.c_max + 2 * widened * root + (n - 1) * G * radii.c_max,
        generation_gap=g_gap,
        payment_anytime=payment,
        payment_anytime_forced=payment + n * G * (radii.c_max - truth.c_second),
        forced_cap=G,
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


def _by_provider(
    log: EpisodeLog, exploration: ExplorationLog | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Per-provider views of the log, in the order that provider was selected.

    With ``exploration``, a provider's exploration services are merged into its
    sequence by round; a round serves a provider at most once, so the order is
    unambiguous.
    """
    col = log.columns()
    keys = ("cost", "correctness", "bid_winner", "t")
    if exploration is not None and len(exploration):
        x = exploration.columns()
        merged = {
            "t": np.concatenate([col["t"], x["t"]]),
            "winner": np.concatenate([col["winner"], x["explored"]]),
            "cost": np.concatenate([col["cost"], x["cost"]]),
            "correctness": np.concatenate([col["correctness"], x["correctness"]]),
            "bid_winner": np.concatenate([col["bid_winner"], x["bid_explored"]]),
        }
        order = np.argsort(merged["t"], kind="stable")
        col = {k: v[order] for k, v in merged.items()}
    out: dict[str, dict[str, np.ndarray]] = {}
    for model in dict.fromkeys(col["winner"].tolist()):
        mask = col["winner"] == model
        out[model] = {k: col[k][mask] for k in keys}
    return out


def good_event_report(
    log: EpisodeLog,
    truth: GroundTruth,
    radii: Radii,
    exploration: ExplorationLog | None = None,
) -> tuple[bool, int]:
    """Whether the good event held at every selection count of every provider."""
    violations = 0
    for model, view in _by_provider(log, exploration).items():
        m = np.arange(1, len(view["cost"]) + 1)
        beta = radii.beta_table(len(m))[1:]
        q_hat = np.cumsum(view["correctness"]) / m
        c_hat = np.cumsum(view["cost"]) / m
        violations += int(np.sum(np.abs(q_hat - truth.q[model]) > beta))
        violations += int(np.sum(np.abs(c_hat - truth.c[model]) > radii.c_max * beta))
    return violations == 0, violations


def slack_report(
    log: EpisodeLog, radii: Radii, exploration: ExplorationLog | None = None
) -> dict[str, float]:
    """Largest realized |e_i - c_hat_i| / rho_c(m) per provider (belief slack).

    The bid at a provider's (k+1)-st win is the estimate formed after its k-th.
    """
    out: dict[str, float] = {}
    for model, view in _by_provider(log, exploration).items():
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
    log: EpisodeLog,
    truth: GroundTruth,
    radii: Radii | None = None,
    exploration: ForcedExploration | None = None,
) -> dict[str, np.ndarray]:
    """Cumulative metric paths, for checkpoints and figures.

    With ``radii`` the bound paths are included, evaluated for the forced
    exploration the episode ran with (``exploration``), the paper's bounds when None.
    """
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
            "payment_anytime_forced",
        )
        paths = {name: np.full(len(elapsed), np.nan) for name in names}
        for idx, t in enumerate(col["t"].astype(int)):
            if t < radii.n:
                continue
            value = bounds(t, truth, radii, exploration)
            for name in names:
                paths[name][idx] = getattr(value, name)
        out.update(paths)
    return out


def evaluate(
    log: EpisodeLog,
    truth: GroundTruth,
    radii: Radii,
    exploration: ExplorationLog | None = None,
    variant: ForcedExploration | None = None,
) -> EpisodeMetrics:
    """All episode-level metrics from the log and the ground truth.

    Every field counts the rounds of the log, which under the count-based
    variant include its forced rounds. ``exploration``, an additive variant's
    services, enters the good-event and slack diagnostics alone, since those
    rebuild the estimates the platform and providers actually held;
    ``exploration_metrics`` totals the services themselves. ``variant`` is the
    forced exploration the episode ran with; the recovery test
    ``identifies_istar`` then uses B_id(T) instead of B_id.
    """
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
    b_id = B_id_at(len(col["t"]), truth, radii, variant)
    if is_complete_prefix:
        holds, violations = good_event_report(log, truth, radii, exploration)
        slack = max(slack_report(log, radii, exploration).values(), default=0.0)
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
                (v > b_id + 1) == (m == truth.istar) for m, v in counts.items()
            )
            if is_complete_prefix
            else None
        ),
        argmax_is_istar=counts[truth.istar] > max(other_counts, default=-1),
        good_event_holds=holds,
        good_event_violations=violations,
        max_slack_ratio=slack,
    )


# --------------------------------------------------------------------------
# Forced-exploration variants
# --------------------------------------------------------------------------


def exploration_events(result: EpisodeResult) -> ExplorationLog:
    """Every exploration event of an episode, in the additional-service layout.

    The additional services of an additive variant, or the forced rounds of
    the count-based one, which replace mechanism rounds and therefore also sit
    in the round log; empty for the mechanism as published.
    """
    if result.exploration is not None and len(result.exploration):
        return result.exploration
    return forced_rounds(result.log)


@dataclass(frozen=True)
class ExplorationMetrics:
    """Totals of the exploration events of one episode."""

    rounds: int  # rounds in which a runner-up was explored
    total_cost: float
    total_payment: float
    counts: dict[str, int]
    payoff: dict[str, float]  # sum(payment - cost) per provider
    istar_rounds: int
    unqualified_rounds: int
    excess_payment: float  # sum of max(payment - c_second, 0)


def exploration_metrics(
    exploration: ExplorationLog, truth: GroundTruth
) -> ExplorationMetrics:
    x = exploration.columns()
    explored = x["explored"] if len(exploration) else np.empty(0, dtype=str)
    cost = x["cost"].astype(float)
    payment = x["payment"].astype(float)
    counts = {m: int((explored == m).sum()) for m in truth.roster}
    payoff = {m: float((payment - cost)[explored == m].sum()) for m in truth.roster}
    return ExplorationMetrics(
        rounds=len(exploration),
        total_cost=float(cost.sum()),
        total_payment=float(payment.sum()),
        counts=counts,
        payoff=payoff,
        istar_rounds=counts.get(truth.istar, 0),
        unqualified_rounds=int(sum(v for m, v in counts.items() if m not in truth.qualified)),
        excess_payment=float(np.maximum(payment - truth.c_second, 0.0).sum()),
    )
