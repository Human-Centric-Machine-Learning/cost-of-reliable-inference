"""Offline experiments for Learning the Cost of Reliable Inference.

Package layout:
    config.py       run configuration (single TOML layer -> dataclasses)
    data.py         load/validate datasets/<BENCH>/<model>.jsonl
    pricing.py      list prices, margin, unit costs, pinned C_max
    variants.py     Best-of-N providers: exact quality, price, cost, landscape
    stream.py       shared seeded query stream + potential outcomes
    confidence.py   Lambda, beta, rho_c, rho, quality UCB
    estimators.py   provider cost estimates e_i(h) and the gamma slack
    providers.py    provider private state (history, estimate, standing bid)
    platform.py     platform state (quality stats, scores, eligibility)
    mechanism.py    the paper's selection rule and critical payment
    baselines.py    baselines and ablations behind the same policy interface
    simulate.py     one episode: exact round order, termination, round log
    metrics.py      empirical metrics and the paper's theoretical quantities
    results.py      read/write round-level and episode-level records
    experiment.py   repeated episodes over benchmarks, policies, seeds
"""
