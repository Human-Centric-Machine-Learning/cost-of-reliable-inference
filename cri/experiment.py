"""Run the environment x policy x repetition grid and store its results.

Seeds derive from (master_seed, environment, repetition) and never from the
policy, so within a repetition every policy faces the same question order,
initialization order, tie draws and preassigned generations.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from cri.baselines import (
    EXPERIMENT_ABLATIONS,
    PAYMENT_APPLIES,
    make_payment_rule,
    make_policy,
)
from cri.config import EnvironmentConfig, ExperimentConfig, episodes, validate_config
from cri.confidence import Radii
from cri.data import BenchmarkData, load_benchmark
from cri.estimators import EstimatorConfig
from cri.metrics import (
    GroundTruth,
    bounds,
    cumulative_series,
    evaluate,
    ground_truth,
    slack_report,
    theory_constants,
)
from cri.results import (
    RunPaths,
    append_episode,
    checkpoint_rounds,
    create_run,
    write_checkpoints,
    write_manifest,
    write_provider_states,
    write_round_log,
)
from cri.simulate import simulate_episode


class ExperimentError(ValueError):
    """A grid point that cannot be run as configured."""


def episode_seed(master_seed: int, environment_id: str, repetition: int) -> int:
    """Derive a stable seed from the master seed, environment, and repetition."""
    env_hash = zlib.crc32(environment_id.encode("utf-8"))
    return int(
        np.random.SeedSequence([master_seed, env_hash, repetition]).generate_state(1)[0]
    )


def draw_seed(master_seed: int, environment_id: str) -> int:
    """Seed for the generation draws; combined with the repetition in simulate_episode."""
    env_hash = zlib.crc32(environment_id.encode("utf-8"))
    return int(np.random.SeedSequence([master_seed, env_hash]).generate_state(1)[0])


def _radii(cfg: ExperimentConfig, env: EnvironmentConfig) -> Radii:
    roster = cfg.roster_for(env)
    return Radii(
        n=len(roster),
        delta=cfg.mechanism.delta,
        c_max=cfg.c_max_for(env),
        gamma=cfg.mechanism.gamma,
    )


@dataclass(frozen=True)
class ResolvedArm:
    """The policy, payment, radii, estimator, and stream settings for an arm."""

    policy_name: str
    payment_name: str | None
    radii: Radii
    estimator: EstimatorConfig
    independent_costs: bool = False


def resolve_arm(
    name: str, cfg: ExperimentConfig, truth: GroundTruth, radii: Radii
) -> ResolvedArm:
    if name not in EXPERIMENT_ABLATIONS:
        make_policy(name, truth)  # fails early on an unknown name
        payment = "critical" if name in PAYMENT_APPLIES else None
        return ResolvedArm(name, payment, radii, cfg.estimator)
    if name == "pay_your_bid":
        return ResolvedArm("mechanism", "pay_your_bid", radii, cfg.estimator)
    if name == "independent_cost_stream":
        return ResolvedArm("mechanism", "critical", radii, cfg.estimator, True)
    if name == "gamma_zero":
        return ResolvedArm(
            "mechanism", "critical", replace(radii, gamma=0.0), cfg.estimator
        )
    assert name == "biased_beliefs"
    estimator = EstimatorConfig("biased", bias=cfg.estimator.bias)
    return ResolvedArm("mechanism", "critical", radii, estimator)


def prepare_environment(
    cfg: ExperimentConfig, env: EnvironmentConfig, data: BenchmarkData | None = None
) -> tuple[BenchmarkData, GroundTruth, Radii]:
    """Load an environment's data, ground truth and radii; pass ``data`` to skip reloading."""
    roster = cfg.roster_for(env)
    bases = cfg.base_models_for(env)
    needs_rewards = any("@" in m for m in roster)
    if data is None:
        data = load_benchmark(
            cfg.data_root,
            env.benchmark,
            bases,
            rewards_root=cfg.rewards_root if needs_rewards else None,
        )
    elif data.benchmark != env.benchmark:
        raise ExperimentError(
            f"{env.id}: data is for {data.benchmark}, not {env.benchmark}"
        )
    elif set(bases) - set(data.records):
        raise ExperimentError(
            f"{env.id}: data lacks {sorted(set(bases) - set(data.records))}"
        )
    elif needs_rewards and not data.has_rewards:
        raise ExperimentError(f"{env.id}: Best-of-N providers need reward scores")
    if env.t_max > data.n_questions and not env.resample:
        raise ExperimentError(
            f"{env.id}: t_max={env.t_max} exceeds the {data.n_questions} questions "
            f"in {env.benchmark}; an episode serves each question at most once "
            "unless the environment sets resample = true"
        )
    truth = ground_truth(data, roster, env.theta, cfg.mechanism.margin)
    return data, truth, _radii(cfg, env)


def validate_experiment(cfg: ExperimentConfig) -> None:
    """Load and validate every environment and resolve every configured arm."""
    validate_config(cfg)
    for env in cfg.environments:
        _, truth, radii = prepare_environment(cfg, env)
        for name in cfg.policies:
            resolve_arm(name, cfg, truth, radii)


def run_experiment(cfg: ExperimentConfig, *, timestamp: str | None = None) -> RunPaths:
    """Run the environment x policy x repetition grid and write the results."""
    validate_config(cfg)
    paths = create_run(cfg.output_dir, cfg.name, timestamp=timestamp)
    prepared: dict[str, tuple[BenchmarkData, GroundTruth, Radii]] = {}
    manifest: dict[str, object] = {
        "run_id": paths.run_id,
        "name": cfg.name,
        "master_seed": cfg.master_seed,
        "repetitions": cfg.repetitions,
        "audit_rep": cfg.audit_rep,
        "policies": list(cfg.policies),
        "mechanism": asdict(cfg.mechanism),
        "estimator": asdict(cfg.estimator),
        "config": {
            "data_root": str(cfg.data_root),
            "rewards_root": str(cfg.rewards_root),
            "full_log": cfg.full_log,
            "rosters": cfg.rosters,
            "checkpoints": asdict(cfg.checkpoints),
        },
        "environments": {},
        "seeds": {},
    }

    for env, policy_name, rep in episodes(cfg):
        if env.id not in prepared:
            data, truth, radii = prepare_environment(cfg, env)
            prepared[env.id] = (data, truth, radii)
            tc = theory_constants(truth, radii, stream=env.t_max)
            manifest["environments"][env.id] = {
                **asdict(env),
                "roster_models": cfg.roster_for(env),
                "base_models": cfg.base_models_for(env),
                "c_max": radii.c_max,
                "stream": env.t_max,
                "passes": -(-env.t_max // data.n_questions),
                "questions": data.n_questions,
                "generations_per_question": data.samples_per_question,
                "draw_seed": draw_seed(cfg.master_seed, env.id),
                "q": truth.q,
                "c": truth.c,
                "qualified": list(truth.qualified),
                "istar": truth.istar,
                "c_star": truth.c_star,
                "c_second": truth.c_second,
                "delta_gap": truth.delta_gap,
                "trap": truth.trap,
                "theory": asdict(tc),
            }
        data, truth, radii = prepared[env.id]
        arm = resolve_arm(policy_name, cfg, truth, radii)

        seed = episode_seed(cfg.master_seed, env.id, rep)
        manifest["seeds"][f"{env.id}__rep{rep}"] = seed

        result = simulate_episode(
            data,
            cfg.roster_for(env),
            theta=env.theta,
            radii=arm.radii,
            estimator=arm.estimator,
            margin=cfg.mechanism.margin,
            t_max=env.t_max,
            seed=seed,
            draw_seed=draw_seed(cfg.master_seed, env.id),
            repetition=rep,
            policy=make_policy(arm.policy_name, truth),
            payment_rule=make_payment_rule(arm.payment_name),
            full_log=cfg.full_log,
            independent_costs=arm.independent_costs,
            resample=env.resample,
        )

        metrics = evaluate(result.log, truth, arm.radii)
        theory = theory_constants(truth, arm.radii, stream=env.t_max)
        bound = bounds(result.rounds, truth, arm.radii)
        append_episode(
            paths,
            {
                "environment": env.id,
                "policy": policy_name,
                "repetition": rep,
                "seed": seed,
                "termination": result.termination,
                "init_order": list(result.init_order),
                "arm": {
                    "selection": arm.policy_name,
                    "payment": arm.payment_name,
                    "radii": asdict(arm.radii),
                    "estimator": asdict(arm.estimator),
                    "independent_costs": arm.independent_costs,
                },
                **asdict(metrics),
                "theory": asdict(theory),
                "bounds": {
                    **asdict(bound),
                    "quality": bound.quality,
                    "generation": bound.generation,
                },
                "slack": slack_report(result.log, arm.radii),
            },
        )

        series = cumulative_series(result.log, truth, arm.radii)
        marks = [
            t
            for t in checkpoint_rounds(
                result.rounds, cfg.checkpoints.per_decade, cfg.checkpoints.landmarks
            )
            if t <= result.rounds
        ]
        idx = np.asarray(marks) - 1
        write_checkpoints(
            paths, env.id, policy_name, rep, {k: v[idx] for k, v in series.items()}
        )

        if cfg.full_log or rep == cfg.audit_rep:
            write_round_log(paths, env.id, policy_name, rep, result.log.columns())
        if cfg.full_log:
            assert result.full_log is not None
            write_provider_states(
                paths, env.id, policy_name, rep, cfg.roster_for(env), result.full_log
            )

    write_manifest(paths, manifest)
    return paths
