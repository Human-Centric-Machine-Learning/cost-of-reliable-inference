# Learning the Cost of Reliable Inference — experiments

An offline simulation of the mechanism in *Learning the Cost of Reliable
Inference*. The platform learns which LLM providers meet a quality threshold
and which qualified provider is cheapest. Providers observe their own costs
and submit standing bids. Each round replays a recorded generation from
`datasets/`; Best-of-N providers also use reward-model scores from `rewards/`.
The library is in `cri/`, with the default experiment in
`configs/default.toml`.

## Run it

```bash
# 1. environment (Python 3.11+)
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 2. inputs: datasets/ and rewards/ are not committed; this downloads the
#    Hugging Face source and builds both (about 20 MB)
venv/bin/python build_datasets.py
venv/bin/python build_rewards.py

# 3. tests
venv/bin/python -m pytest tests -q

# 4. a quick run, then the full run (nine environments x eight arms x 50 repetitions)
venv/bin/python scripts/run_experiment.py configs/default.toml --name smoke --repetitions 3
venv/bin/python scripts/run_experiment.py configs/default.toml

# 5. tables and figures from a finished run
venv/bin/python scripts/make_report.py outputs/runs/<run_id>
```

Run-level fields can be overridden without editing the file:
`--name`, `--repetitions`, `--environments ID ...`, `--policies NAME ...`,
`--full-log`, `--dry-run`.

To explore interactively, open `analysis/analysis.ipynb` with the `venv`
kernel: one step per cell, from loading the data to comparing policies,
ablations and sweeps.

The interactive companion page, which runs the same library in the browser,
is in `web/`; see `web/README.md` to build and serve it.
