"""List prices per 1M output tokens (USD) and the margin that turns them into costs.

Cost of one generation, in units of 1e-6 USD:

    C = num_tokens * price_per_1M / (1 + margin)

C_max is pinned to the price scale, token_cap * max_i (N_i * price_i), so that
it is a public constant of the mechanism. It must stay above the validity
floor, the same quantity deflated by (1 + margin); below it realized costs
could leave [0, C_max].
"""

from __future__ import annotations

from collections.abc import Iterable

MODEL_PRICE_PER_1M: dict[str, float] = {
    "Llama-3-8B": 0.1455,
    "Llama-3.1-8B": 0.1245,
    "Llama-3.2-1B": 0.10,
    "Llama-3.2-3B": 0.08,
    "Qwen2-0.5B": 0.10,
    "Qwen2-1.5B": 0.10,
    "Qwen2-7B": 0.20,
    "Qwen2.5-3B": 0.065,
    "Qwen2.5-7B": 0.1465,
    "reason-R1-D-Llama-8B": 0.1250,
    "reason-R1-D-Qwen-1.5B": 0.1000,
    "reason-R1-D-Qwen-7B": 0.1750,
}

DEFAULT_TOKEN_CAP = 512


class PricingError(ValueError):
    """Unknown model, or a margin outside (-1, inf)."""


def base_model(name: str) -> str:
    """'Qwen2-7B@8' -> 'Qwen2-7B'. A variant carries its base model's price."""
    return name.partition("@")[0]


def _price(model: str) -> float:
    try:
        return MODEL_PRICE_PER_1M[base_model(model)]
    except KeyError:
        raise PricingError(
            f"no list price for {model!r}; known models: "
            f"{sorted(MODEL_PRICE_PER_1M)}"
        ) from None


def _check_margin(margin: float) -> float:
    if not margin > -1.0:
        raise PricingError(f"margin must be > -1, got {margin!r}")
    return float(margin)


def unit_cost(model: str, margin: float) -> float:
    """Cost per token: the list price deflated by the margin."""
    return _price(model) / (1.0 + _check_margin(margin))


def invoice(name: str, num_tokens: int) -> float:
    """The public invoice of a query: generated tokens times the advertised rate.

    ``num_tokens`` already sums a Best-of-N provider's N generations. This is
    what a platform paying list prices observes, and equals (1 + margin) times
    the provider's private cost.
    """
    return float(num_tokens) * _price(name)


def _multiplicity(name: str) -> int:
    """'Qwen2-7B@8' -> 8; a bare model name -> 1."""
    _, _, suffix = name.partition("@")
    if not suffix:
        return 1
    try:
        return int(suffix)
    except ValueError:
        raise PricingError(f"cannot read a Best-of-N count from {name!r}") from None


def pinned_c_max(models: Iterable[str], token_cap: int = DEFAULT_TOKEN_CAP) -> float:
    """Return token_cap * max_i(N_i * price_i) for the roster."""
    models = list(models)
    if not models:
        raise PricingError("cannot compute C_max for an empty roster")
    if token_cap <= 0:
        raise PricingError(f"token_cap must be positive, got {token_cap!r}")
    return token_cap * max(_multiplicity(m) * _price(m) for m in models)


def c_max_floor(
    models: Iterable[str], token_cap: int = DEFAULT_TOKEN_CAP, *, margin: float
) -> float:
    """Smallest admissible C_max: the largest cost any roster provider can realize."""
    return pinned_c_max(models, token_cap) / (1.0 + _check_margin(margin))
