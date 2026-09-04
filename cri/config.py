"""Parse the TOML run configuration into frozen dataclasses.

Validation catches invalid parameters, unknown rosters or arms, short
horizons, an undersized C_max, and insufficient gamma before a run starts.
``experiment.py`` checks T_max against the loaded benchmark.

    load_config(path)     parse and validate a TOML file
    override(cfg, ...)    the same file with run-level fields changed
    expand_sweep(cfg)     one config per point of the optional [sweep] table
    episodes(cfg)         the (environment, policy, repetition) grid
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

from cri import pricing
from cri.estimators import EstimatorConfig
from cri.pricing import DEFAULT_TOKEN_CAP, base_model
from cri.variants import parse_variant


class ConfigError(ValueError):
    """An invalid experiment configuration."""


@dataclass(frozen=True)
class MechanismConfig:
    margin: float = 0.25
    token_cap: int = DEFAULT_TOKEN_CAP
    c_max: float | None = None  # None -> pinned to the price scale
    delta: float = 0.05
    gamma: float = 0.0

    def resolved_c_max(self, roster: list[str]) -> float:
        """The configured C_max, or the price-scale ceiling when unset."""
        return self.pinned_c_max(roster) if self.c_max is None else self.c_max

    def pinned_c_max(self, roster: list[str]) -> float:
        return pricing.pinned_c_max(roster, self.token_cap)

    def c_max_floor(self, roster: list[str]) -> float:
        """The largest cost the roster can realize; C_max may not be below it."""
        return pricing.c_max_floor(roster, self.token_cap, margin=self.margin)


@dataclass(frozen=True)
class EnvironmentConfig:
    """A (roster, benchmark, Theta, horizon) tuple: one environment to study.

    ``resample`` allows ``t_max`` beyond the question count: the stream then
    continues through fresh seeded passes over the benchmark.
    """

    id: str
    benchmark: str
    roster: str
    theta: float
    t_max: int
    resample: bool = False


@dataclass(frozen=True)
class CheckpointConfig:
    per_decade: int = 20
    landmarks: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: Path
    master_seed: int
    repetitions: int
    audit_rep: int
    full_log: bool
    data_root: Path
    rewards_root: Path
    rosters: dict[str, list[str]]
    environments: tuple[EnvironmentConfig, ...]
    mechanism: MechanismConfig
    estimator: EstimatorConfig
    policies: tuple[str, ...]
    checkpoints: CheckpointConfig
    sweep: dict[str, list] = field(default_factory=dict)

    def roster_for(self, env: EnvironmentConfig) -> list[str]:
        return list(self.rosters[env.roster])

    def base_models_for(self, env: EnvironmentConfig) -> list[str]:
        """Distinct base models a roster needs loaded (variants share a base)."""
        return list(dict.fromkeys(base_model(m) for m in self.roster_for(env)))

    def c_max_for(self, env: EnvironmentConfig) -> float:
        return self.mechanism.resolved_c_max(self.roster_for(env))


def _validate(cfg: ExperimentConfig) -> ExperimentConfig:
    from cri.baselines import EXPERIMENT_ABLATIONS, POLICY_NAMES

    m = cfg.mechanism
    if not cfg.name.strip():
        raise ConfigError("run name must not be empty")
    if cfg.repetitions < 1:
        raise ConfigError(f"repetitions must be >= 1, got {cfg.repetitions}")
    if not 0 <= cfg.audit_rep < cfg.repetitions:
        raise ConfigError(
            f"audit_rep {cfg.audit_rep} is outside 0..{cfg.repetitions - 1}"
        )
    if not 0.0 < m.delta < 1.0:
        raise ConfigError(f"delta must lie in (0, 1), got {m.delta}")
    if m.gamma < 0:
        raise ConfigError(f"gamma must be >= 0, got {m.gamma}")
    if m.margin <= -1:
        raise ConfigError(f"margin must be > -1, got {m.margin}")
    if m.token_cap < 1:
        raise ConfigError(f"token_cap must be >= 1, got {m.token_cap}")
    if not cfg.environments:
        raise ConfigError("no environments configured")
    if not cfg.policies:
        raise ConfigError("no policies configured")
    env_ids = [env.id for env in cfg.environments]
    if len(set(env_ids)) != len(env_ids):
        raise ConfigError("environment ids must be unique")
    if any(not env_id.strip() for env_id in env_ids):
        raise ConfigError("environment ids must not be empty")
    if len(set(cfg.policies)) != len(cfg.policies):
        raise ConfigError("policies/arms must be unique")
    unknown = set(cfg.policies) - POLICY_NAMES - EXPERIMENT_ABLATIONS
    if unknown:
        raise ConfigError(
            f"unknown policies/arms {sorted(unknown)}; known: "
            f"{sorted(POLICY_NAMES | EXPERIMENT_ABLATIONS)}"
        )
    if cfg.checkpoints.per_decade < 1:
        raise ConfigError(
            f"checkpoints.per_decade must be >= 1, got {cfg.checkpoints.per_decade}"
        )
    if any(v < 1 for v in cfg.checkpoints.landmarks):
        raise ConfigError("checkpoint landmarks must be positive integers")

    for env in cfg.environments:
        if env.roster not in cfg.rosters:
            raise ConfigError(
                f"environment {env.id!r} names roster {env.roster!r}; "
                f"known rosters: {sorted(cfg.rosters)}"
            )
        roster = cfg.roster_for(env)
        if len(roster) < 2:
            raise ConfigError(f"roster {env.roster!r} has fewer than 2 providers")
        if len(set(roster)) != len(roster):
            raise ConfigError(f"roster {env.roster!r} repeats a provider")
        for name in roster:
            parse_variant(name)  # rejects a malformed or unsupported N
        if not 0.0 < env.theta < 1.0:
            raise ConfigError(f"{env.id}: Theta must lie in (0, 1), got {env.theta}")
        if env.t_max < len(roster):
            raise ConfigError(f"{env.id}: t_max is below the {len(roster)} init rounds")

        c_max = cfg.c_max_for(env)
        floor = m.c_max_floor(roster)
        if c_max < floor - 1e-12:
            raise ConfigError(
                f"{env.id}: C_max={c_max} is below the validity floor {floor:.4f}; "
                "realized costs would leave [0, C_max] and the payment cap could "
                "sit below a provider's true cost"
            )

        # gamma must cover the estimator's slack, except for the deliberate violation.
        needed = cfg.estimator.required_gamma(len(roster), m.delta)
        if cfg.estimator.kind != "biased" and m.gamma < needed - 1e-12:
            raise ConfigError(
                f"{env.id}: estimator {cfg.estimator.kind!r} needs gamma >= {needed:.4f} "
                f"at n={len(roster)}, delta={m.delta}, but gamma={m.gamma}"
            )
    return cfg


def validate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Validate a programmatically constructed configuration."""
    return _validate(cfg)


def override(
    cfg: ExperimentConfig,
    *,
    name: str | None = None,
    repetitions: int | None = None,
    environments: Sequence[str] | None = None,
    policies: Sequence[str] | None = None,
    full_log: bool | None = None,
) -> ExperimentConfig:
    """Re-validate with run-level fields changed.

    ``environments`` and ``policies`` select a subset of what the file defines,
    keeping the file's order.
    """
    changes: dict[str, object] = {}
    if name is not None:
        changes["name"] = name
    if repetitions is not None:
        changes["repetitions"] = repetitions
        changes["audit_rep"] = min(cfg.audit_rep, repetitions - 1)
    if full_log is not None:
        changes["full_log"] = full_log
    if environments is not None:
        known = {env.id for env in cfg.environments}
        unknown = set(environments) - known
        if unknown:
            raise ConfigError(
                f"unknown environments {sorted(unknown)}; configured: {sorted(known)}"
            )
        changes["environments"] = tuple(
            env for env in cfg.environments if env.id in set(environments)
        )
    if policies is not None:
        changes["policies"] = tuple(policies)
    return _validate(replace(cfg, **changes))


def load_config(path: str | Path) -> ExperimentConfig:
    """Parse and validate a TOML configuration."""
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    run = raw.get("run", {})
    mech = raw.get("mechanism", {})
    prov = raw.get("provider", {})
    chk = raw.get("checkpoints", {})

    estimator = EstimatorConfig(
        kind=prov.get("estimator", "empirical_mean"),
        k=prov.get("shrinkage_k", 10.0),
        prior_mean=prov.get("prior_mean", 0.0),
        bias=prov.get("bias", 0.25),
    )
    cfg = ExperimentConfig(
        name=run.get("name", path.stem),
        output_dir=Path(run.get("output_dir", "outputs")),
        master_seed=int(run.get("master_seed", 0)),
        repetitions=int(run.get("repetitions", 1)),
        audit_rep=int(run.get("audit_rep", 0)),
        full_log=bool(run.get("full_log", False)),
        data_root=Path(raw.get("data", {}).get("root", "datasets")),
        rewards_root=Path(raw.get("data", {}).get("rewards_root", "rewards")),
        rosters={k: list(v) for k, v in raw.get("rosters", {}).items()},
        environments=tuple(
            EnvironmentConfig(
                id=e["id"],
                benchmark=e["benchmark"],
                roster=e["roster"],
                theta=float(e["Theta"]),
                t_max=int(e["T_max"]),
                resample=bool(e.get("resample", False)),
            )
            for e in raw.get("environment", [])
        ),
        mechanism=MechanismConfig(
            margin=float(mech.get("margin", 0.25)),
            token_cap=int(mech.get("token_cap", DEFAULT_TOKEN_CAP)),
            c_max=float(mech["C_max"]) if "C_max" in mech else None,
            delta=float(mech.get("delta", 0.05)),
            gamma=float(mech.get("gamma", 0.0)),
        ),
        estimator=estimator,
        policies=tuple(raw.get("policies", {}).get("run", ["mechanism"])),
        checkpoints=CheckpointConfig(
            per_decade=int(chk.get("per_decade", 20)),
            landmarks=tuple(int(v) for v in chk.get("landmarks", [])),
        ),
        sweep={k: list(v) for k, v in raw.get("sweep", {}).items()},
    )
    return validate_config(cfg)


SWEEPABLE = ("C_max", "Theta", "gamma", "delta", "margin")


def expand_sweep(cfg: ExperimentConfig) -> list[ExperimentConfig]:
    """One config per point of the [sweep] cross product; none yields [cfg]."""
    if not cfg.sweep:
        return [cfg]
    unknown = set(cfg.sweep) - set(SWEEPABLE)
    if unknown:
        raise ConfigError(
            f"cannot sweep {sorted(unknown)}; sweepable: {list(SWEEPABLE)}"
        )

    keys = sorted(cfg.sweep)
    out: list[ExperimentConfig] = []
    for combo in product(*(cfg.sweep[k] for k in keys)):
        point = dict(zip(keys, combo))
        mech = replace(
            cfg.mechanism,
            **{
                dest: float(point[src])
                for src, dest in (
                    ("C_max", "c_max"),
                    ("gamma", "gamma"),
                    ("delta", "delta"),
                    ("margin", "margin"),
                )
                if src in point
            },
        )
        envs = cfg.environments
        if "Theta" in point:
            envs = tuple(replace(e, theta=float(point["Theta"])) for e in envs)
        tag = "__".join(f"{k}={point[k]}" for k in keys)
        out.append(
            _validate(
                replace(
                    cfg,
                    mechanism=mech,
                    environments=envs,
                    sweep={},
                    name=f"{cfg.name}__{tag}",
                )
            )
        )
    return out


def episodes(cfg: ExperimentConfig) -> Iterator[tuple[EnvironmentConfig, str, int]]:
    """The (environment, policy, repetition) grid, in a stable order."""
    for env in cfg.environments:
        for policy in cfg.policies:
            for rep in range(cfg.repetitions):
                yield env, policy, rep
