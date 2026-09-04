"""JSON API for the companion page.

Public methods use JSON-safe dictionaries, with ``None`` for missing numbers.
They share the experiment runner's seeding scheme, so shipped environments are
reproducible across the CLI and browser.

    Engine.from_packs({"GSM8K": "GSM8K.npz", ...})   the page's data
    Engine.from_jsonl("datasets", "rewards")           the repository's data

    engine.catalog("GSM8K")          every base x N candidate's quality and token counts
    engine.describe(request)         ground truth, radii and theory numbers, no simulation
    engine.run(request, progress)    the arms x repetitions grid
    engine.sweep(request, thetas)    the same grid at several thresholds

Requests use the keys in ``DEFAULTS`` and require ``benchmark``, ``roster``,
and ``theta``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cri.baselines import (
    EXPERIMENT_ABLATIONS,
    PAYMENT_APPLIES,
    POLICY_NAMES,
    make_payment_rule,
    make_policy,
)
from cri.confidence import Radii
from cri.data import BenchmarkData, load_benchmark
from cri.estimators import EstimatorConfig
from cri.experiment import draw_seed, episode_seed, resolve_arm
from cri.metrics import (
    GroundTruth,
    MetricsError,
    cumulative_series,
    evaluate,
    ground_truth,
    theory_constants,
)
from cri.pricing import MODEL_PRICE_PER_1M, base_model, c_max_floor, pinned_c_max
from cri.results import checkpoint_rounds
from cri.simulate import simulate_episode
from cri.variants import VALID_N, expected_qualities, parse_variant

from engine.pack import load_pack, pack_models

ENGINE_VERSION = "1"

MAIN_ARMS = [
    "mechanism",
    "uniform_random",
    "cheapest_bid",
    "quality_greedy",
    "oracle_cheapest_qualified",
]
ABLATION_ARMS = [
    "no_quality_filter",
    "greedy_cheapest_qualified",
    "pay_your_bid",
    "gamma_zero",
    "biased_beliefs",
    "independent_cost_stream",
]
OTHER_ARMS = ["cheapest_price", "oracle_quality_random"]
ALL_ARMS = MAIN_ARMS + ABLATION_ARMS + OTHER_ARMS

DEFAULTS: dict[str, object] = {
    "benchmark": None,
    "roster": None,
    "theta": None,
    "delta": 0.05,
    "gamma": 0.0,
    "margin": 0.25,
    "token_cap": 512,
    "c_max": None,  # None: pinned to the price scale
    "estimator": {"kind": "empirical_mean", "k": 10.0, "prior_mean": 0.0, "bias": 0.25},
    "arms": list(MAIN_ARMS),
    "repetitions": 3,
    "t_max": None,  # None: one pass over the benchmark
    "master_seed": 20260826,
    "environment_id": None,  # None: derived from benchmark and roster
    "per_decade": 20,  # checkpoint density of the returned series
}

#: Episode metrics summarised as mean / sd / min / max per arm.
SUMMARY_KEYS = (
    "realized_accuracy",
    "total_cost",
    "quality_regret",
    "generation_regret",
    "istar_share",
    "non_istar_rounds",
    "total_payment",
    "excess_payment",
    "min_provider_payoff",
    "critical_below_bid_rounds",
    "max_slack_ratio",
)
RATE_KEYS = ("good_event_holds", "identifies_istar", "argmax_is_istar", "collapsed")
SERIES_KEYS = (
    "quality_regret",
    "generation_regret",
    "excess_payment",
    "total_payment",
    "provider_payoff",
    "accuracy",
    "istar_share",
)
BOUND_KEYS = (
    "quality_anytime",
    "quality_gap",
    "generation_anytime",
    "generation_gap",
    "payment_anytime",
)

Progress = Callable[[int, int, str], None]
Cancelled = Callable[[], bool]


class EngineError(ValueError):
    """A request the engine cannot serve; the message is meant for the page."""


class EngineCancelled(Exception):
    """The caller's cancel flag was set between episodes."""


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------


def _clean(value, ndigits: int | None = None):
    """numpy -> Python, NaN -> None, optional rounding of floats."""
    if isinstance(value, dict):
        return {str(k): _clean(v, ndigits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist(), ndigits)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits) if ndigits is not None else f
    return value


def _stats(values: Sequence[float | None]) -> dict[str, float | None] | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    a = np.array(xs)
    return {"mean": float(a.mean()), "sd": float(a.std()), "min": float(a.min()), "max": float(a.max())}


def _nanmean_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Mean over repetitions, ignoring the rounds a halted episode never reached."""
    a = np.array(rows, dtype=float)
    empty = np.isnan(a).all(axis=0)
    out = np.full(a.shape[1], np.nan)
    if not empty.all():
        out[~empty] = np.nanmean(a[:, ~empty], axis=0)
    return out


def _collapsed(metrics, truth: GroundTruth, t_max: int) -> tuple[bool, str]:
    """Did the most-selected provider take more than half the rounds while unqualified?"""
    top = max(metrics.selection_counts, key=metrics.selection_counts.get)
    return (
        metrics.selection_counts[top] > t_max / 2 and top not in truth.qualified,
        top,
    )


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


class Engine:
    def __init__(self, source: Callable[[str, Sequence[str], bool], BenchmarkData], models: dict[str, list[str]]):
        self._source = source
        self._models = models  # benchmark -> models with reward scores
        self._cache: dict[tuple, BenchmarkData] = {}

    @classmethod
    def from_packs(cls, packs: dict[str, str | Path]) -> "Engine":
        paths = {k: Path(v) for k, v in packs.items()}

        def source(benchmark: str, models: Sequence[str], with_rewards: bool) -> BenchmarkData:
            if benchmark not in paths:
                raise EngineError(f"no data for benchmark {benchmark!r}; available: {sorted(paths)}")
            return load_pack(paths[benchmark], models, with_rewards=with_rewards)

        engine = cls(source, {b: pack_models(p) for b, p in paths.items()})
        engine._packs = paths
        return engine

    @classmethod
    def from_jsonl(cls, data_root: str | Path, rewards_root: str | Path) -> "Engine":
        data_root, rewards_root = Path(data_root), Path(rewards_root)
        models = {
            d.name: sorted(f.stem for f in d.glob("*.jsonl") if (rewards_root / d.name / f.name).is_file())
            for d in data_root.iterdir()
            if d.is_dir()
        }

        def source(benchmark: str, roster_models: Sequence[str], with_rewards: bool) -> BenchmarkData:
            if benchmark not in models:
                raise EngineError(f"no data for benchmark {benchmark!r}; available: {sorted(models)}")
            return load_benchmark(data_root, benchmark, roster_models, rewards_root=rewards_root if with_rewards else None)

        return cls(source, models)

    def add_pack(self, benchmark: str, path: str | Path) -> None:
        """Register a pack after construction (the page fetches benchmarks on demand)."""
        if not hasattr(self, "_packs"):
            raise EngineError("add_pack needs an engine built from packs")
        self._packs[benchmark] = Path(path)
        self._models[benchmark] = pack_models(path)

    @property
    def benchmarks(self) -> list[str]:
        return sorted(self._models)

    def data(self, benchmark: str, models: Sequence[str], with_rewards: bool) -> BenchmarkData:
        key = (benchmark, tuple(models), with_rewards)
        if key not in self._cache:
            self._cache[key] = self._source(benchmark, models, with_rewards)
        return self._cache[key]

    # ---- catalog -----------------------------------------------------------

    def catalog(self, benchmark: str) -> dict:
        """Every base x N candidate: quality, mean and max tokens, list price.

        Cost at margin m is mean_tokens * price_per_1M / (1 + m) and the pinned
        C_max is token_cap * max rate; the page derives both. Qualities here come from the questions every model has reward
        scores for; ``describe`` gives the exact values for a roster.
        """
        models = self._models.get(benchmark)
        if not models:
            raise EngineError(f"no data for benchmark {benchmark!r}; available: {self.benchmarks}")
        plain = self.data(benchmark, models, False)
        rich = self.data(benchmark, models, True)
        candidates = []
        for m in models:
            rec = rich.records[m]
            qualities = expected_qualities(rec, VALID_N)
            for n in VALID_N:
                candidates.append(
                    {
                        "name": m if n == 1 else f"{m}@{n}",
                        "base": m,
                        "n": n,
                        "quality": qualities[n],
                        "mean_tokens": n * float(plain.records[m].num_tokens.mean()),
                        "max_tokens": n * float(plain.records[m].num_tokens.max()),
                        "price_per_1M": MODEL_PRICE_PER_1M[m],
                        "rate": n * MODEL_PRICE_PER_1M[m],
                    }
                )
        return _clean(
            {
                "benchmark": benchmark,
                "questions": plain.n_questions,
                "questions_with_rewards": rich.n_questions,
                "samples_per_question": plain.samples_per_question,
                "models": list(models),
                "valid_n": list(VALID_N),
                "candidates": candidates,
            },
            6,
        )

    # ---- request resolution --------------------------------------------------

    def _resolve(self, request: dict) -> SimpleNamespace:
        unknown = set(request) - set(DEFAULTS)
        if unknown:
            raise EngineError(f"unknown request fields {sorted(unknown)}")
        r = {**DEFAULTS, **{k: v for k, v in request.items() if v is not None or k in ("c_max", "t_max", "environment_id")}}
        r["estimator"] = {**DEFAULTS["estimator"], **(request.get("estimator") or {})}
        for key in ("benchmark", "roster", "theta"):
            if r[key] is None:
                raise EngineError(f"request needs {key!r}")
        roster = [str(m) for m in r["roster"]]
        if len(roster) < 2:
            raise EngineError("a roster needs at least two providers")
        if len(set(roster)) != len(roster):
            raise EngineError("a provider appears twice in the roster")
        benchmark = str(r["benchmark"])
        known = set(self._models.get(benchmark, []))
        if not known:
            raise EngineError(f"no data for benchmark {benchmark!r}; available: {self.benchmarks}")
        for m in roster:
            v = parse_variant(m)
            if v.base not in known:
                raise EngineError(f"{m!r}: no data for base model {v.base!r} on {benchmark}")
        bases = list(dict.fromkeys(base_model(m) for m in roster))
        needs_rewards = any(parse_variant(m).n > 1 for m in roster)
        data = self.data(benchmark, bases, needs_rewards)

        t_max = int(r["t_max"]) if r["t_max"] is not None else data.n_questions
        if t_max > data.n_questions:
            raise EngineError(f"horizon {t_max} exceeds the {data.n_questions} questions; an episode serves each question at most once")
        if t_max < len(roster):
            raise EngineError(f"horizon {t_max} is below the {len(roster)} initialization rounds")
        if not 0.0 < float(r["delta"]) < 1.0:
            raise EngineError("delta must lie strictly between 0 and 1")
        if float(r["gamma"]) < 0:
            raise EngineError("gamma must be non-negative")
        if float(r["margin"]) <= -1:
            raise EngineError("margin must exceed -1")
        if int(r["token_cap"]) < 1:
            raise EngineError("token cap must be at least 1")
        if int(r["repetitions"]) < 1:
            raise EngineError("repetitions must be at least 1")
        if not r["arms"]:
            raise EngineError("select at least one arm")
        unknown_arms = [a for a in r["arms"] if a not in POLICY_NAMES and a not in EXPERIMENT_ABLATIONS]
        if unknown_arms:
            raise EngineError(f"unknown arms {unknown_arms}; known: {ALL_ARMS}")
        if len(set(r["arms"])) != len(r["arms"]):
            raise EngineError("an arm appears twice")

        margin = float(r["margin"])
        try:
            truth = ground_truth(data, roster, float(r["theta"]), margin)
        except MetricsError as exc:
            raise EngineError(str(exc)) from None
        pinned = pinned_c_max(roster, int(r["token_cap"]))
        floor = c_max_floor(roster, int(r["token_cap"]), margin=margin)
        c_max = float(r["c_max"]) if r["c_max"] is not None else pinned
        if c_max < floor:
            raise EngineError(f"C_max = {c_max:.4g} is below the roster's validity floor {floor:.4g}")
        radii = Radii(n=len(roster), delta=float(r["delta"]), c_max=c_max, gamma=float(r["gamma"]))
        est = r["estimator"]
        if est.get("kind") not in EstimatorConfig.KINDS:
            raise EngineError(f"unknown estimator {est.get('kind')!r}; known: {list(EstimatorConfig.KINDS)}")
        estimator = EstimatorConfig(
            kind=est["kind"], k=float(est.get("k", 10.0)), prior_mean=float(est.get("prior_mean", 0.0)), bias=float(est.get("bias", 0.25))
        )
        if estimator.kind == "shrinkage" and not 0.0 <= estimator.prior_mean <= c_max:
            raise EngineError(f"the shrinkage prior mean must lie in [0, C_max = {c_max:.4g}]")
        env_id = str(r["environment_id"]) if r["environment_id"] else f"{benchmark}:{','.join(roster)}"
        r.update(roster=roster, benchmark=benchmark, t_max=t_max, environment_id=env_id, c_max=c_max)
        return SimpleNamespace(request=r, data=data, truth=truth, radii=radii, estimator=estimator, margin=margin, pinned=pinned, floor=floor)

    def _environment(self, s: SimpleNamespace) -> dict:
        truth, radii, data = s.truth, s.radii, s.data
        tc = theory_constants(truth, radii, stream=data.n_questions)
        ms = np.unique(np.logspace(0, 6, 121).astype(int))
        return {
            "benchmark": data.benchmark,
            "questions": data.n_questions,
            "samples_per_question": data.samples_per_question,
            "dropped_for_rewards": list(data.dropped_for_rewards),
            "roster": list(truth.roster),
            "theta": truth.theta,
            "n": radii.n,
            "delta": radii.delta,
            "gamma": radii.gamma,
            "c_max": radii.c_max,
            "pinned_c_max": s.pinned,
            "c_max_floor": s.floor,
            "margin": s.margin,
            "estimator": asdict(s.estimator),
            "required_gamma": s.estimator.required_gamma(radii.n, radii.delta),
            "q": truth.q,
            "c": truth.c,
            "price": truth.price,
            "qualified": list(truth.qualified),
            "istar": truth.istar,
            "c_star": truth.c_star,
            "c_second": truth.c_second,
            "delta_gap": truth.delta_gap,
            "eps": truth.eps,
            "delta_j": truth.delta_j,
            "cheapest_overall": truth.cheapest_overall,
            "trap": truth.trap,
            "theory": asdict(tc),
            "reachable": tc.reachable,
            "radius_table": {
                "m": ms.tolist(),
                "beta": [radii.beta(int(m)) for m in ms],
                "rho": [radii.rho(int(m)) for m in ms],
            },
        }

    def _arms(self, s: SimpleNamespace) -> dict[str, SimpleNamespace]:
        cfg = SimpleNamespace(estimator=s.estimator)  # resolve_arm reads only cfg.estimator
        out = {}
        for name in s.request["arms"]:
            arm = resolve_arm(name, cfg, s.truth, s.radii)
            out[name] = SimpleNamespace(
                resolved=arm,
                info={
                    "policy": arm.policy_name,
                    "payment": arm.payment_name,
                    "paid": arm.payment_name is not None,
                    "estimator": arm.estimator.kind,
                    "gamma": arm.radii.gamma,
                    "independent_costs": arm.independent_costs,
                    "group": "main" if name in MAIN_ARMS else "ablation" if name in ABLATION_ARMS else "other",
                },
            )
        return out

    def describe(self, request: dict) -> dict:
        """Resolve a request without simulating: environment, arms and horizon."""
        s = self._resolve(request)
        arms = self._arms(s)
        return _clean({"request": s.request, "environment": self._environment(s), "arms": {k: v.info for k, v in arms.items()}}, 6)

    # ---- episodes ------------------------------------------------------------

    def _episode(self, s: SimpleNamespace, arm: SimpleNamespace, rep: int, *, full_log: bool):
        r = s.request
        a = arm.resolved
        seed = episode_seed(int(r["master_seed"]), r["environment_id"], rep)
        result = simulate_episode(
            s.data,
            r["roster"],
            theta=s.truth.theta,
            radii=a.radii,
            estimator=a.estimator,
            margin=s.margin,
            t_max=r["t_max"],
            seed=seed,
            draw_seed=draw_seed(int(r["master_seed"]), r["environment_id"]),
            repetition=rep,
            policy=make_policy(a.policy_name, s.truth),
            payment_rule=make_payment_rule(a.payment_name),
            full_log=full_log,
            independent_costs=a.independent_costs,
        )
        return seed, result

    def run_job(self, request: dict) -> "RunJob":
        """A run as a job: call ``step()`` until it returns False, then ``result()``."""
        return RunJob(self, request)

    def sweep_job(self, request: dict, thetas: Sequence[float]) -> "SweepJob":
        return SweepJob(self, request, thetas)

    def run(self, request: dict, progress: Progress | None = None, cancelled: Cancelled | None = None) -> dict:
        """Simulate every (arm, repetition) and return metrics, summaries and series."""
        return _drive(self.run_job(request), progress, cancelled)

    def sweep(self, request: dict, thetas: Sequence[float], progress: Progress | None = None, cancelled: Cancelled | None = None) -> dict:
        """The arms x repetitions grid at each threshold; per-episode outcomes only."""
        return _drive(self.sweep_job(request, thetas), progress, cancelled)

    @staticmethod
    def _beliefs(ep, roster: Sequence[str], checkpoints: Sequence[int], n: int) -> dict:
        """Quality UCBs at the checkpoint rounds, and each provider's first ineligible round."""
        f = ep.full_log
        rows = len(f["q_ucb"])
        if rows == 0:  # halted before any main round
            return {"t": [], "q_ucb": [], "eligible": [], "first_ineligible": {m: None for m in roster}}
        main_t = np.arange(n + 1, n + 1 + rows)
        wanted = set(checkpoints) | {int(main_t[-1])}
        keep = [i for i, t in enumerate(main_t) if int(t) in wanted]
        first = {}
        for j, name in enumerate(roster):
            off = np.flatnonzero(~f["eligible"][:, j])
            first[name] = int(main_t[off[0]]) if len(off) else None
        return _clean(
            {
                "t": main_t[keep],
                "q_ucb": f["q_ucb"][keep],
                "eligible": f["eligible"][keep].astype(int),
                "first_ineligible": first,
            },
            4,
        )

    @staticmethod
    def _slack(ep, roster: Sequence[str], radii: Radii) -> dict:
        """Belief slack |bid - c_hat| / rho_c(m) at each provider's m-th selection."""
        col = ep.log.columns()
        out = {}
        for name in roster:
            mine = col["winner"] == name
            costs, bids = col["cost"][mine], col["bid_winner"][mine].astype(float)
            if len(costs) < 2:
                out[name] = {"m": [], "ratio": []}
                continue
            k = np.arange(1, len(costs))
            c_hat = np.cumsum(costs)[:-1] / k
            rho_c = radii.c_max * radii.beta_table(len(k))[1:]
            out[name] = {"m": k, "ratio": np.abs(bids[1:] - c_hat) / rho_c}
        return _clean(out, 4)


def _drive(job: "Job", progress: Progress | None, cancelled: Cancelled | None) -> dict:
    while job.step():
        if progress is not None:
            progress(job.done, job.total, job.label)
        if cancelled is not None and cancelled():
            raise EngineCancelled()
    if progress is not None:
        progress(job.done, job.total, job.label)
    return job.result()


# --------------------------------------------------------------------------
# Jobs: one episode per step, so a caller can report progress or cancel between episodes
# --------------------------------------------------------------------------


class Job:
    total: int
    done: int
    label: str

    def step(self) -> bool:
        """Run one episode; False once everything has run."""
        raise NotImplementedError

    def result(self) -> dict:
        raise NotImplementedError


class RunJob(Job):
    def __init__(self, engine: Engine, request: dict):
        self.engine = engine
        self.started = time.monotonic()
        self.s = engine._resolve(request)
        r = self.s.request
        self.arms = engine._arms(self.s)
        self.roster = r["roster"]
        self.n, self.t_max, self.reps = len(self.roster), r["t_max"], int(r["repetitions"])
        self.checkpoints = checkpoint_rounds(self.t_max, int(r["per_decade"]))
        self.cp_index = np.array(self.checkpoints) - 1
        self.index = {m: i for i, m in enumerate(self.roster)}
        self.queue = [(name, rep) for name in self.arms for rep in range(self.reps)]
        self.total, self.done, self.label = len(self.queue), 0, ""
        self.episodes: list[dict] = []
        self.per_rep: dict[str, list[dict]] = {name: [] for name in self.arms}
        self.paths: dict[str, dict[str, list]] = {name: {k: [] for k in SERIES_KEYS} for name in self.arms}
        self.bound_paths: dict[str, dict] = {}
        self.rounds: dict[str, dict] = {}
        self.beliefs: dict[str, dict] = {}
        self.slack: dict[str, dict] = {}

    def step(self) -> bool:
        if self.done >= self.total:
            return False
        name, rep = self.queue[self.done]
        s, arm = self.s, self.arms[name]
        seed, ep = self.engine._episode(s, arm, rep, full_log=rep == 0)
        metrics = evaluate(ep.log, s.truth, arm.resolved.radii)
        collapsed, top = _collapsed(metrics, s.truth, self.t_max)
        record = {
            "arm": name,
            "repetition": rep,
            "seed": seed,
            "rounds": ep.rounds,
            "termination": ep.termination,
            "collapsed": collapsed,
            "top_provider": top,
            **asdict(metrics),
        }
        self.per_rep[name].append(record)
        self.episodes.append(record)

        cs = cumulative_series(ep.log, s.truth, arm.resolved.radii if rep == 0 else None)
        valid = self.cp_index[self.cp_index < ep.rounds]
        for k in SERIES_KEYS:
            row = np.full(len(self.checkpoints), np.nan)
            row[: len(valid)] = cs[k][valid]
            self.paths[name][k].append(row)
        if rep == 0:
            self.bound_paths[name] = {}
            for k in BOUND_KEYS:
                row = np.full(len(self.checkpoints), np.nan)
                row[: len(valid)] = cs[k][valid]
                self.bound_paths[name][k] = row
            col = ep.log.columns()
            self.rounds[name] = _clean(
                {
                    "t": col["t"],
                    "init_rounds": self.n,
                    "winner": [self.index[w] for w in col["winner"]],
                    "correctness": col["correctness"],
                    "cost": col["cost"],
                    "payment": col["payment"].astype(float),
                    "critical_payment": col["critical_payment"].astype(float),
                    "bid": col["bid_winner"].astype(float),
                    "m_winner": col["m_winner"],
                    "n_eligible": col["n_eligible"],
                    "rho_m": [arm.resolved.radii.rho(int(m)) if m > 0 else None for m in col["m_winner"]],
                },
                4,
            )
            self.beliefs[name] = Engine._beliefs(ep, self.roster, self.checkpoints, self.n)
            self.slack[name] = Engine._slack(ep, self.roster, arm.resolved.radii)
        self.done += 1
        self.label = name
        return self.done < self.total

    def result(self) -> dict:
        if self.done < self.total:
            raise EngineError("the run has not finished")
        summary, series = {}, {}
        for name in self.arms:
            per_rep = self.per_rep[name]
            summary[name] = {k: _stats([e[k] for e in per_rep]) for k in SUMMARY_KEYS}
            for k in RATE_KEYS:
                flags = [e[k] for e in per_rep if e[k] is not None]
                summary[name][k] = float(np.mean(flags)) if flags else None
            counts = np.array([[e["selection_counts"][m] for m in self.roster] for e in per_rep], dtype=float)
            summary[name]["selection_share"] = (counts.mean(axis=0) / self.t_max).tolist()
            summary[name]["repetitions"] = self.reps
            series[name] = _clean(
                {"t": self.checkpoints, **{k: _nanmean_rows(self.paths[name][k]) for k in SERIES_KEYS}, **self.bound_paths[name]},
                4,
            )
        return _clean(
            {
                "engine": ENGINE_VERSION,
                "kind": "run",
                "request": self.s.request,
                "environment": self.engine._environment(self.s),
                "arms": {k: v.info for k, v in self.arms.items()},
                "checkpoints": self.checkpoints,
                "episodes": self.episodes,
                "summary": summary,
                "series": series,
                "rounds": self.rounds,
                "beliefs": self.beliefs,
                "slack": self.slack,
                "seconds": time.monotonic() - self.started,
            }
        )


class SweepJob(Job):
    KEYS = ("istar_share", "realized_accuracy", "quality_regret", "generation_regret", "total_cost")

    def __init__(self, engine: Engine, request: dict, thetas: Sequence[float]):
        self.engine = engine
        self.started = time.monotonic()
        base = dict(request)
        self.points: list[dict] = []
        self.skipped: list[dict] = []
        self.queue: list[tuple[int, str, int]] = []  # (point index, arm, repetition)
        self.resolved: list[SimpleNamespace] = []
        self.arms: list[dict] = []
        for theta in thetas:
            try:
                s = engine._resolve({**base, "theta": float(theta)})
            except EngineError as exc:
                self.skipped.append({"theta": float(theta), "reason": str(exc)})
                continue
            arms = engine._arms(s)
            idx = len(self.points)
            self.resolved.append(s)
            self.arms.append(arms)
            self.points.append(
                {
                    "theta": s.truth.theta,
                    "qualified": len(s.truth.qualified),
                    "istar": s.truth.istar,
                    "c_star": s.truth.c_star,
                    "trap": s.truth.trap,
                    "cheapest_overall": s.truth.cheapest_overall,
                    "arms": {name: {k: [] for k in (*self.KEYS, "collapsed", "top_provider")} for name in arms},
                }
            )
            self.queue.extend((idx, name, rep) for name in arms for rep in range(int(s.request["repetitions"])))
        if not self.points:
            raise EngineError("no admissible threshold in the sweep: " + "; ".join(p["reason"] for p in self.skipped[:3]))
        self.total, self.done, self.label = len(self.queue), 0, ""

    def step(self) -> bool:
        if self.done >= self.total:
            return False
        idx, name, rep = self.queue[self.done]
        s, arm = self.resolved[idx], self.arms[idx][name]
        _, ep = self.engine._episode(s, arm, rep, full_log=False)
        m = evaluate(ep.log, s.truth, arm.resolved.radii)
        collapsed, top = _collapsed(m, s.truth, s.request["t_max"])
        rows = self.points[idx]["arms"][name]
        for k in self.KEYS:
            rows[k].append(getattr(m, k))
        rows["collapsed"].append(collapsed)
        rows["top_provider"].append(top)
        self.done += 1
        self.label = f"theta {self.points[idx]['theta']}: {name}"
        return self.done < self.total

    def result(self) -> dict:
        if self.done < self.total:
            raise EngineError("the sweep has not finished")
        first = self.resolved[0]
        return _clean(
            {
                "engine": ENGINE_VERSION,
                "kind": "sweep",
                "request": {**first.request, "theta": None},
                "roster": first.request["roster"],
                "q": first.truth.q,
                "c": first.truth.c,
                "arms": {k: v.info for k, v in self.arms[0].items()},
                "points": self.points,
                "skipped": self.skipped,
                "seconds": time.monotonic() - self.started,
            },
            6,
        )
