"""Run one episode and record its rounds.

Initialization selects each provider once. Later rounds compute selection and
payment from pre-round state, reveal only the winner's outcome, update the
winner's bid and quality record, and then log the result. A missing payment
rule records ``NaN`` payments. ``full_log`` adds per-provider state for each
main round. By default an episode is at most one pass over the benchmark;
``resample=True`` lets a longer horizon continue through further seeded
passes with fresh draws from the recorded generations (see stream.py), leaving
the first pass unchanged. ``exploration`` switches on a forced-exploration
variant (see exploration.py). An additive variant serves a second provider
after a main round's normal selection, with probability p_t, and logs it in a
separate ``ExplorationLog``; the round log and the winner's round are
unaffected. The count-based variant instead replaces a main round in which
some eligible provider is below its count target by a forced round for one
of the under-sampled eligible providers, drawn uniformly, paid C_max, with no
mechanism winner; such rounds carry phase ``"explore"`` in the round log and
are otherwise logged like any selection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from cri.confidence import Radii
from cri.data import BenchmarkData
from cri.estimators import EstimatorConfig
from cri.exploration import (
    ForcedExploration,
    pick_under_sampled,
    probability_at,
    under_sampled,
)
from cri.mechanism import Selection, critical_payment, select
from cri.platform import PlatformState, init_platform, record_selection
from cri.pricing import invoice as public_invoice
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
    """Columnar round log.

    ``phase`` is ``"init"``, ``"main"`` or, under the count-based exploration
    variant, ``"explore"`` for a forced round: the winner is then the explored
    provider, the payment C_max, and the critical payment and runner-up score
    NaN; ``tie_size`` holds the size of the under-sampled set it was drawn from.
    """

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


@dataclass
class ExplorationLog:
    """Columnar log of the forced-exploration services, one row per explored round.

    Pre-round quantities (``m_explored``, ``bid_explored``, ``score_explored``)
    are those the round's selection was computed from. ``payment`` is what the
    rule paid (the bid under the runner-up rule, C_max under the uniform rule)
    and ``tie_size`` the size of the set the provider was drawn from (the
    runner-ups tied on the score, or the candidate set without the winner).
    """

    t: list[int] = field(default_factory=list)
    explored: list[str] = field(default_factory=list)
    qid: list[str] = field(default_factory=list)
    sample_index: list[int] = field(default_factory=list)
    correctness: list[int] = field(default_factory=list)
    num_tokens: list[int] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    payment: list[float] = field(default_factory=list)
    m_explored: list[int] = field(default_factory=list)
    bid_explored: list[float] = field(default_factory=list)
    score_explored: list[float] = field(default_factory=list)
    tie_size: list[int] = field(default_factory=list)  # size of the set drawn from

    def __len__(self) -> int:
        return len(self.t)

    def append(self, **row: object) -> None:
        for key, value in row.items():
            getattr(self, key).append(value)

    def columns(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in vars(self).items() if isinstance(v, list)}


def forced_rounds(log: EpisodeLog) -> ExplorationLog:
    """The forced rounds of the count-based variant, in the layout of the additional-service log."""
    col = log.columns()
    out = ExplorationLog()
    for i in np.flatnonzero(col["phase"] == "explore") if len(log) else []:
        out.append(
            t=int(col["t"][i]), explored=str(col["winner"][i]), qid=str(col["qid"][i]),
            sample_index=int(col["sample_index"][i]), correctness=int(col["correctness"][i]),
            num_tokens=int(col["num_tokens"][i]), cost=float(col["cost"][i]), payment=float(col["payment"][i]),
            m_explored=int(col["m_winner"][i]), bid_explored=float(col["bid_winner"][i]),
            score_explored=float(col["score_winner"][i]), tie_size=int(col["tie_size"][i]),
        )
    return out


@dataclass(frozen=True)
class EpisodeResult:
    """One finished episode.

    ``exploration`` is the log of additional services: None for the mechanism
    as published, and empty when a variant added none (the count-based one
    never does; its forced rounds are in ``log``, see ``forced_rounds``).
    """

    log: EpisodeLog
    platform: PlatformState
    providers: dict[str, ProviderState]
    stream: QueryStream
    termination: str  # "t_max" | "halted"
    rounds: int
    init_order: tuple[str, ...]
    full_log: dict[str, np.ndarray] | None = None
    exploration: ExplorationLog | None = None


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
    resample: bool = False,
    exploration: ForcedExploration | None = None,
) -> EpisodeResult:
    """Run one episode; see the module docstring for the round protocol.

    ``seed`` drives the question order, the initialization order, the
    round-indexed tie draws and, when ``exploration`` is set, the
    round-indexed exploration draws; ``draw_seed`` (default: ``seed``)
    combined with ``repetition`` drives the generation draws. None of these
    should depend on the policy. ``payment_rule=None`` runs the arm unpaid.
    ``resample=True`` allows ``t_max`` beyond the question count by
    continuing through fresh passes. ``exploration`` runs a forced-exploration
    variant (``exploration.make_exploration``); ``None`` is the protocol as
    published.
    """
    roster = list(roster)
    n = len(roster)
    if radii.n != n:
        raise SimulationError(f"radii were built for n={radii.n}, roster has {n}")
    if t_max < n:
        raise SimulationError(f"t_max={t_max} is below the {n} initialization rounds")
    if t_max > data.n_questions and not resample:
        raise SimulationError(
            f"t_max={t_max} exceeds the {data.n_questions} questions in "
            f"{data.benchmark}; an episode serves each question at most once "
            "unless resample=True"
        )
    passes = -(-t_max // data.n_questions)  # ceil; 1 for any one-pass horizon

    seed_root = np.random.SeedSequence(seed)
    seed_stream, seed_init, seed_tie = seed_root.spawn(3)
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
        passes,
    )
    init_order = tuple(np.random.default_rng(seed_init).permutation(roster).tolist())
    tie_draws = np.random.default_rng(seed_tie).random(t_max)
    # A fourth child of the seed, spawned after the three above so those streams are
    # unchanged. Column 0 decides whether an additive round explores, column 1 is
    # the rule's draw (runner-up tie-break, uniform choice, or the forced round's).
    explore_draws = (
        np.random.default_rng(seed_root.spawn(1)[0]).random((t_max, 2))
        if exploration is not None
        else None
    )

    providers = init_providers(roster)
    platform = init_platform(roster, radii, theta)
    log = EpisodeLog()
    exploration_log = ExplorationLog() if exploration is not None else None
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
            platform, model, outcome.correctness, providers[model].bid,
            public_invoice(model, outcome.num_tokens),
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

        # ---- count-based forced exploration: the round goes to an under-sampled
        # eligible provider instead of the winner. A_t and the halt rule are the
        # mechanism's; the choice reads eligibility and pre-round counts only.
        if exploration is not None and exploration.replaces_round:
            assert explore_draws is not None and exploration.count is not None
            under = under_sampled(
                selection.eligible, platform.selection_counts(), exploration.count.target(t, n)
            )
            if under:
                explored = pick_under_sampled(under, float(explore_draws[idx, 1]))
                pre_x = platform.providers[explored]
                outcome_x = stream.reveal(idx, explored)
                providers[explored] = observe_cost(
                    providers[explored], outcome_x.cost, estimator=estimator, c_max=radii.c_max
                )
                platform = record_selection(
                    platform, explored, outcome_x.correctness, providers[explored].bid,
                    public_invoice(explored, outcome_x.num_tokens),
                )
                log.append(
                    t=t,
                    phase="explore",
                    winner=explored,
                    qid=outcome_x.qid,
                    sample_index=outcome_x.sample_index,
                    correctness=outcome_x.correctness,
                    num_tokens=outcome_x.num_tokens,
                    cost=outcome_x.cost,
                    payment=radii.c_max if payment_rule is not None else np.nan,
                    critical_payment=np.nan,
                    m_winner=pre_x.m,
                    bid_winner=pre_x.bid,
                    score_winner=selection.scores[explored],
                    runner_up_score=np.nan,
                    n_eligible=selection.n_eligible,
                    tie_size=len(under),
                )
                continue

        outcome = stream.reveal(idx, winner)
        providers[winner] = observe_cost(
            providers[winner], outcome.cost, estimator=estimator, c_max=radii.c_max
        )
        platform = record_selection(
            platform, winner, outcome.correctness, providers[winner].bid,
            public_invoice(winner, outcome.num_tokens),
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

        # ---- additive forced exploration: a second service after the winner's
        # round is final. The rule reads the pre-round selection and state, so
        # nothing revealed in this round influences whom it explores or the pay.
        if exploration is not None and not exploration.replaces_round:
            assert explore_draws is not None and exploration_log is not None
            assert exploration.schedule is not None and exploration.rule is not None
            u_explore, u_rule = explore_draws[idx]
            if u_explore < probability_at(exploration.schedule, t, n):
                explored, drawn_from = exploration.rule.choose(selection, float(u_rule))
                pre_x = platform.providers[explored]
                assert pre_x.bid is not None and explored != winner
                paid_x = exploration.rule.payment(platform, explored)
                outcome_x = stream.reveal_exploration(idx, explored)
                providers[explored] = observe_cost(
                    providers[explored], outcome_x.cost, estimator=estimator,
                    c_max=radii.c_max,
                )
                platform = record_selection(
                    platform, explored, outcome_x.correctness, providers[explored].bid,
                    public_invoice(explored, outcome_x.num_tokens),
                )
                exploration_log.append(
                    t=t,
                    explored=explored,
                    qid=outcome_x.qid,
                    sample_index=outcome_x.sample_index,
                    correctness=outcome_x.correctness,
                    num_tokens=outcome_x.num_tokens,
                    cost=outcome_x.cost,
                    payment=paid_x,
                    m_explored=pre_x.m,
                    bid_explored=pre_x.bid,
                    score_explored=selection.scores[explored],
                    tie_size=drawn_from,
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
        exploration=exploration_log,
    )
