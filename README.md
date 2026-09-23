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

# 3. the configured grid (five environments x five arms x 30 repetitions),
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
gives a short check. `main_paper_figure.py` draws the main text's figure, and
the same panels for the appendix, from the results the notebooks write;
`istar_payment.py` recomputes the payment received by the optimal provider over
the last pass, as reported in the main text's table.

The interactive companion page, which runs the same library in the browser,
is in `web/`; see `web/README.md` to build and serve it.
