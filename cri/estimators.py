"""Provider cost estimates e_i(h) and their slack gamma against the empirical mean.

All three rules depend on the history only through (m, c_hat):

    empirical_mean   e = c_hat                        gamma = 0; the primary configuration
    shrinkage        e = (m c_hat + k mu_0) / (m + k)  needs gamma >= gamma_min_shrinkage(k, n, delta),
                                                      recomputed for the run rather than hard-coded
    biased           e = c_hat + bias * C_max         unbounded slack, on purpose

The realized slack |e - c_hat| / rho_c(m) is measured from the round log by
metrics.slack_report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cri.confidence import gamma_min_shrinkage


class EstimatorError(ValueError):
    """Malformed estimator configuration or out-of-range inputs."""


def _clip(value: float, c_max: float) -> float:
    return min(max(value, 0.0), c_max)


def empirical_mean(m: int, c_hat: float, c_max: float) -> float:
    return _clip(c_hat, c_max)


def shrinkage(
    m: int, c_hat: float, c_max: float, *, k: float, prior_mean: float
) -> float:
    """Posterior mean under a conjugate prior with pseudo-count k."""
    if k <= 0:
        raise EstimatorError(f"pseudo-count k must be positive, got {k!r}")
    if not 0.0 <= prior_mean <= c_max:
        raise EstimatorError(f"prior_mean {prior_mean!r} outside [0, {c_max}]")
    return _clip((m * c_hat + k * prior_mean) / (m + k), c_max)


def biased(m: int, c_hat: float, c_max: float, *, bias: float) -> float:
    """A constant offset of bias * C_max: no finite gamma bounds its slack."""
    return _clip(c_hat + bias * c_max, c_max)


@dataclass(frozen=True)
class EstimatorConfig:
    """Which estimation rule a provider uses, with its parameters."""

    kind: str = "empirical_mean"
    k: float = 10.0
    prior_mean: float = 0.0
    bias: float = 0.25

    KINDS = ("empirical_mean", "shrinkage", "biased")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise EstimatorError(
                f"unknown estimator {self.kind!r}; expected one of {self.KINDS}"
            )
        if self.kind == "shrinkage" and self.k <= 0:
            raise EstimatorError(f"pseudo-count k must be positive, got {self.k!r}")

    def estimate(self, m: int, c_hat: float, c_max: float) -> float:
        if m < 1:
            raise EstimatorError(
                f"no estimate is defined before the first selection (m={m})"
            )
        if self.kind == "empirical_mean":
            return empirical_mean(m, c_hat, c_max)
        if self.kind == "shrinkage":
            return shrinkage(m, c_hat, c_max, k=self.k, prior_mean=self.prior_mean)
        return biased(m, c_hat, c_max, bias=self.bias)

    def required_gamma(self, n: int, delta: float) -> float:
        """Smallest gamma that bounds this rule's slack; inf for the biased rule."""
        if self.kind == "empirical_mean":
            return 0.0
        if self.kind == "shrinkage":
            return gamma_min_shrinkage(self.k, n, delta)
        return math.inf
