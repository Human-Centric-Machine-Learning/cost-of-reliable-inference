#!/usr/bin/env python3
"""Build the static assets of the companion page into web/app/data/.

    python web/build.py [--repetitions N] [--skip-default]

Writes
    <BENCH>.npz       compact benchmarks (web/engine/pack.py)
    catalog.json      every base x N candidate per benchmark, for the live roster builder
    presets.json      the shipped environments and rosters from configs/default.toml
    engine.zip        the cri package and the engine, unpacked into Pyodide's file system
    default.json      a precomputed run of the default environment, shown before any run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from cri.config import load_config  # noqa: E402
from engine.api import DEFAULTS, MAIN_ARMS, ABLATION_ARMS, Engine  # noqa: E402
from engine.pack import pack_benchmark  # noqa: E402

OUT = ROOT / "web" / "app" / "data"
DEFAULT_ENVIRONMENT = "GSM8K-ladder"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repetitions", type=int, default=10, help="repetitions in the precomputed default run")
    ap.add_argument("--skip-default", action="store_true", help="do not precompute default.json")
    args = ap.parse_args()

    cfg = load_config(ROOT / "configs/default.toml")
    OUT.mkdir(parents=True, exist_ok=True)
    models = cfg.rosters["full"]

    packs = {}
    for benchmark in sorted({env.benchmark for env in cfg.environments}):
        out = OUT / f"{benchmark}.npz"
        pack_benchmark(cfg.data_root, cfg.rewards_root, benchmark, models, out)
        packs[benchmark] = out
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1e6:.1f} MB")

    engine = Engine.from_packs(packs)
    catalog = {b: engine.catalog(b) for b in packs}
    (OUT / "catalog.json").write_text(json.dumps(catalog, separators=(",", ":")))
    print("catalog.json")

    presets = {
        "rosters": cfg.rosters,
        "environments": [asdict(env) for env in cfg.environments],
        "mechanism": asdict(cfg.mechanism),
        "estimator": asdict(cfg.estimator),
        "policies": list(cfg.policies),
        "master_seed": cfg.master_seed,
        "main_arms": MAIN_ARMS,
        "ablation_arms": ABLATION_ARMS,
        "default_environment": DEFAULT_ENVIRONMENT,
        "request_defaults": {k: v for k, v in DEFAULTS.items() if k not in ("benchmark", "roster", "theta")},
    }
    (OUT / "presets.json").write_text(json.dumps(presets, indent=1))
    print("presets.json")

    with zipfile.ZipFile(OUT / "engine.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for src in sorted((ROOT / "cri").glob("*.py")) + sorted((ROOT / "web" / "engine").glob("*.py")):
            z.write(src, f"{src.parent.name}/{src.name}")
    print("engine.zip")

    if args.skip_default:
        return
    env = next(e for e in cfg.environments if e.id == DEFAULT_ENVIRONMENT)
    request = {
        "benchmark": env.benchmark,
        "roster": cfg.rosters[env.roster],
        "theta": env.theta,
        "delta": cfg.mechanism.delta,
        "gamma": cfg.mechanism.gamma,
        "margin": cfg.mechanism.margin,
        "token_cap": cfg.mechanism.token_cap,
        "c_max": cfg.mechanism.c_max,
        "estimator": asdict(cfg.estimator),
        "arms": list(cfg.policies),
        "repetitions": args.repetitions,
        "t_max": env.t_max,
        "master_seed": cfg.master_seed,
        "environment_id": env.id,
    }
    start = time.monotonic()
    result = engine.run(request, progress=lambda d, t, a: print(f"\r  default run {d}/{t} {a:28}", end=""))
    result["precomputed"] = True
    text = json.dumps(result, separators=(",", ":"))
    (OUT / "default.json").write_text(text)
    print(f"\ndefault.json  {len(text) / 1e6:.1f} MB  ({time.monotonic() - start:.0f}s)")


if __name__ == "__main__":
    main()
