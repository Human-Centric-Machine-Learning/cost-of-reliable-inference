# Learning the Cost of Reliable Inference: experiments

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
#    Hugging Face source and builds both (about 15 MB)
venv/bin/python build_datasets.py
venv/bin/python build_rewards.py

# 3. tests
venv/bin/python -m pytest tests -q

# 4. the configured grid (four environments x five arms x 30 repetitions),
#    written to outputs/runs/<run_id>/
venv/bin/python -c "from cri.config import load_config; from cri.experiment import run_experiment; print(run_experiment(load_config('configs/default.toml')).root)"
```

`cri.config.override` changes run-level fields such as the repetitions, the
environments or the policies before a run; `cri.results` reads a finished run
back.

The paper's experiments are the notebooks in `analysis/`: `GSM8K-full`,
`GSM8K-ladder`, `GPQA-full`, `GPQA-ladder` and `GPQA-strong` are the same
notebook run on one environment each, with the benchmark, roster and forced
exploration set in the first cell and the outputs kept. Each writes every
figure, table and per-repetition result, plus a manifest with the seeds, to
`analysis/results/variants/count_scale2_alpha0.75/<benchmark>/<roster>/`
(`analysis/results/<benchmark>/<roster>/` without forced exploration). Open one
with the `venv` kernel; a full run takes roughly 35 minutes, and `QUICK = True`
gives a short check. `main_paper_experiments_v2.ipynb` builds the main text's
figures and table (`analysis/results/main_paper_v2/`) from the four
environments run without forced exploration (`EXPLORATION = "none"`).

The interactive companion page, which runs the same library in the browser,
is in `web/`; see `web/README.md` to build and serve it.
