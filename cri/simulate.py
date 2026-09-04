"""Run one episode and record its rounds.

Initialization selects each provider once. Later rounds compute selection and
payment from pre-round state, reveal only the winner's outcome, update the
winner's bid and quality record, and then log the result. A missing payment
rule records ``NaN`` payments. ``full_log`` adds per-provider state for each
main round.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from cri.confidence import Radii
from cri.data import BenchmarkData
from cri.estimators import EstimatorConfig
from cri.mechanism import Selection, critical_payment, select
from cri.platform import PlatformState, init_platform, record_selection
from cri.providers import ProviderState, init_providers, observe_cost
from cri.stream import QueryStream, build_stream

#: (platform_state, tie_draw) -> Selection
Policy = Callable[[PlatformState, float], Selection]
#: (platform_state, selection) -> payment, computed before the reveal
PaymentRule = Callable[[PlatformState, Selection], float]


class SimulationError(ValueError):
    """An episode was configured in a way the protocol does not allow."""


def critical_payment_rule(state: PlatformState, selection: Selection) -> float:
    """Compute the critical payment over the selection's candidate set."""
    assert selection.winner is not None
    return critical_payment(state, selection.winner, selection.eligible)


@dataclass
class EpisodeLog:
    """Columnar round log."""

    source_start: int = 1  # first round of a sliced log; not a column

    t: list[int] = field(default_factory=list)
    phase: list[str] = field(default_factory=list)
    winner: list[str] = field(default_factory=list)
    qid: list[str] = field(default_factory=list)
    sample_index: list[int] = field(default_factory=list)
    correctness: list[int] = field(default_factory=list)
    num_tokens: list[int] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    payment: list[float] = field(default_factory=list)  # NaN: no payment rule
    critical_payment: list[float] = field(default_factory=list)  # raw critical payment
    m_winner: list[int] = field(default_factory=list)
    bid_winner: list[float] = field(default_factory=list)
    score_winner: list[float] = field(default_factory=list)
    runner_up_score: list[float] = field(default_factory=list)
    n_eligible: list[int] = field(default_factory=list)
    tie_size: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def append(self, **row: object) -> None:
        for key, value in row.items():
            getattr(self, key).append(value)

    def columns(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in vars(self).items() if isinstance(v, list)}


@dataclass(frozen=True)
class EpisodeResult:
    """One finished episode."""

    log: EpisodeLog
    platform: PlatformState
    providers: dict[str, ProviderState]
    stream: QueryStream
    termination: str  # "t_max" | "halted"
    rounds: int
    init_order: tuple[str, ...]
    full_log: dict[str, np.ndarray] | None = None


def simulate_episode(
    data: BenchmarkData,
    roster: Sequence[str],
    *,
    theta: float,
    radii: Radii,
    estimator: EstimatorConfig,
    margin: float,
    t_max: int,
    seed: int,
    draw_seed: int | None = None,
    repetition: int = 0,
    policy: Policy = select,
    payment_rule: PaymentRule | None = critical_payment_rule,
    full_log: bool = False,
    independent_costs: bool = False,
) -> EpisodeResult:
    """Run one episode; see the module docstring for the round protocol.

    ``seed`` drives the question order, the initialization order and the
    round-indexed tie draws; ``draw_seed`` (default: ``seed``) combined with
    ``repetition`` drives the generation draws. Neither should depend on the
    policy. ``payment_rule=None`` runs the arm unpaid.
    """
    roster = list(roster)
    n = len(roster)
    if radii.n != n:
        raise SimulationError(f"radii were built for n={radii.n}, roster has {n}")
    if t_max < n:
        raise SimulationError(f"t_max={t_max} is below the {n} initialization rounds")
    if t_max > data.n_questions:
        raise SimulationError(
            f"t_max={t_max} exceeds the {data.n_questions} questions in "
            f"{data.benchmark}; an episode serves each question at most once"
        )

    seed_stream, seed_init, seed_tie = np.random.SeedSequence(seed).spawn(3)
    stream = build_stream(
        data,
        roster,
        margin,
        np.random.default_rng(seed_stream),
        np.random.default_rng(
            np.random.SeedSequence(
                [seed if draw_seed is None else draw_seed, repetition]
            )
        ),
        repetition,
        independent_costs,
    )
    init_order = tuple(np.random.default_rng(seed_init).permutation(roster).tolist())
    tie_draws = np.random.default_rng(seed_tie).random(t_max)

    providers = init_providers(roster)
    platform = init_platform(roster, radii, theta)
    log = EpisodeLog()
    wide: dict[str, list] = {k: [] for k in ("m", "bid", "score", "q_ucb", "eligible")}

    # ---- initialization: rounds 1..n --------------------------------------
    for t, model in enumerate(init_order, start=1):
        idx = t - 1
        outcome = stream.reveal(idx, model)
        pre = platform.providers[model]
        providers[model] = observe_cost(
            providers[model], outcome.cost, estimator=estimator, c_max=radii.c_max
        )
        platform = record_selection(
            platform, model, outcome.correctness, providers[model].bid
        )
        log.append(
            t=t,
            phase="init",
            winner=model,
            qid=outcome.qid,
            sample_index=outcome.sample_index,
            correctness=outcome.correctness,
            num_tokens=outcome.num_tokens,
            cost=outcome.cost,
            payment=radii.c_max if payment_rule is not None else np.nan,
            critical_payment=np.nan,
            m_winner=pre.m,
            bid_winner=np.nan,
            score_winner=np.nan,
            runner_up_score=np.nan,
            n_eligible=n,
            tie_size=0,
        )

    # ---- main loop: rounds n+1..t_max -------------------------------------
    termination = "t_max"
    for t in range(n + 1, t_max + 1):
        idx = t - 1
        selection = policy(platform, float(tie_draws[idx]))
        if selection.halted:
            termination = "halted"
            break

        winner = selection.winner
        assert winner is not None
        # Priced before the reveal; the raw critical payment is logged for every arm.
        raw = critical_payment(platform, winner, selection.eligible)
        payment = (
            payment_rule(platform, selection) if payment_rule is not None else np.nan
        )
        pre = platform.providers[winner]
        others = [j for j in selection.eligible if j != winner]
        runner_up = min(selection.scores[j] for j in others) if others else np.nan

        if full_log:
            wide["m"].append([platform.providers[k].m for k in roster])
            wide["bid"].append([platform.providers[k].bid for k in roster])
            wide["score"].append([selection.scores[k] for k in roster])
            wide["q_ucb"].append([platform.q_ucb(k) for k in roster])
            wide["eligible"].append([k in selection.eligible for k in roster])

        outcome = stream.reveal(idx, winner)
        providers[winner] = observe_cost(
            providers[winner], outcome.cost, estimator=estimator, c_max=radii.c_max
        )
        platform = record_selection(
            platform, winner, outcome.correctness, providers[winner].bid
        )
        log.append(
            t=t,
            phase="main",
            winner=winner,
            qid=outcome.qid,
            sample_index=outcome.sample_index,
            correctness=outcome.correctness,
            num_tokens=outcome.num_tokens,
            cost=outcome.cost,
            payment=payment,
            critical_payment=raw,
            m_winner=pre.m,
            bid_winner=pre.bid,
            score_winner=selection.scores[winner],
            runner_up_score=runner_up,
            n_eligible=selection.n_eligible,
            tie_size=len(selection.tied),
        )

    return EpisodeResult(
        log=log,
        platform=platform,
        providers=providers,
        stream=stream,
        termination=termination,
        rounds=len(log),
        init_order=init_order,
        full_log={k: np.asarray(v) for k, v in wide.items()} if full_log else None,
    )
