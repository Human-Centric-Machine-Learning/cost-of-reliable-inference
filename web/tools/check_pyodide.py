#!/usr/bin/env python3
"""Verify that the engine gives the same results under Pyodide as under CPython.

    PYODIDE_MODULE=.../node_modules/pyodide/pyodide.mjs python web/tools/check_pyodide.py

Runs one request natively and through web/tools/pyodide_run.mjs (Node and the
pyodide npm package are required), then compares seeds, every winner of every
round, selection counts and the episode metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from engine.api import Engine  # noqa: E402

DATA = ROOT / "web" / "app" / "data"

REQUEST = {
    "benchmark": "GSM8K",
    "roster": [
        "Llama-3-8B", "Llama-3-8B@2", "Llama-3-8B@4",
        "Llama-3.2-1B", "Llama-3.2-1B@2", "Llama-3.2-1B@4",
        "Qwen2-1.5B", "Qwen2-1.5B@2", "Qwen2-1.5B@4",
        "Qwen2.5-7B", "Qwen2.5-7B@2", "Qwen2.5-7B@4",
    ],
    "theta": 0.805,
    "arms": ["mechanism", "greedy_cheapest_qualified", "cheapest_bid", "pay_your_bid", "biased_beliefs"],
    "repetitions": 2,
    "environment_id": "GSM8K-ladder",
}


def _close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=1e-9)
    args = ap.parse_args()
    if "PYODIDE_MODULE" not in os.environ:
        print("set PYODIDE_MODULE to node_modules/pyodide/pyodide.mjs", file=sys.stderr)
        return 2

    native = Engine.from_packs({"GSM8K": DATA / "GSM8K.npz"}).run(REQUEST)
    with tempfile.TemporaryDirectory() as tmp:
        req, out = Path(tmp) / "request.json", Path(tmp) / "pyodide.json"
        req.write_text(json.dumps(REQUEST))
        subprocess.run(["node", str(ROOT / "web/tools/pyodide_run.mjs"), str(DATA), str(req), str(out)], check=True)
        remote = json.loads(out.read_text())
    print("pyodide:", remote["backend"])
    print(f"native {native['seconds']:.1f}s, pyodide {remote['seconds']:.1f}s for {len(native['episodes'])} episodes")

    problems = 0
    worst = 0.0
    for a, b in zip(native["episodes"], remote["episodes"]):
        assert (a["arm"], a["repetition"]) == (b["arm"], b["repetition"])
        if a["seed"] != b["seed"] or a["selection_counts"] != b["selection_counts"] or a["rounds"] != b["rounds"]:
            problems += 1
            print(f"MISMATCH {a['arm']} rep {a['repetition']}: seeds/selection counts differ")
        for key in ("realized_accuracy", "total_cost", "quality_regret", "generation_regret", "istar_share", "total_payment", "excess_payment", "max_slack_ratio"):
            if not _close(a[key], b[key], args.tolerance):
                problems += 1
                print(f"MISMATCH {a['arm']} rep {a['repetition']} {key}: {a[key]} vs {b[key]}")
            elif a[key] is not None and b[key] is not None:
                worst = max(worst, abs(a[key] - b[key]) / max(abs(a[key]), 1e-12))
    for arm in native["rounds"]:
        if native["rounds"][arm]["winner"] != remote["rounds"][arm]["winner"]:
            problems += 1
            print(f"MISMATCH {arm}: the round-by-round winners differ")
    print(f"largest relative difference in a metric: {worst:.2e}")
    print("OK: pyodide reproduces the native run" if problems == 0 else f"{problems} mismatches")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
