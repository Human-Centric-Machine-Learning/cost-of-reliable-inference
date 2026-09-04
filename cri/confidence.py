"""Anytime confidence radii.

    Lambda(m) = ln(2 pi^2 n m^2 / (3 delta))
    beta(m)   = sqrt(Lambda(m) / (2m))         quality radius
    rho_c(m)  = C_max * beta(m)                cost radius
    rho(m)    = (1 + gamma) * rho_c(m)         widened radius, used in scores and payments
    q_UCB     = min{1, q_hat + beta(m)}

Every radius depends on the round only through the pre-round selection count
m_i(t) = #{s < t : I_s = i}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ConfidenceError(ValueError):
    """Parameters outside the domain of the confidence sequence."""


def _check(m: int, n: int, delta: float) -> None:
    if m < 1:
        raise ConfidenceError(f"selection count must be >= 1, got {m!r}")
    if n < 1:
        raise ConfidenceError(f"number of providers must be >= 1, got {n!r}")
    if not 0.0 < delta < 1.0:
        raise ConfidenceError(f"delta must lie in (0, 1), got {delta!r}")


def Lambda(m: int, n: int, delta: float) -> float:
    _check(m, n, delta)
    return math.log(2.0 * math.pi**2 * n * m * m / (3.0 * delta))


def beta(m: int, n: int, delta: float) -> float:
    return math.sqrt(Lambda(m, n, delta) / (2.0 * m))


def rho_c(m: int, n: int, delta: float, c_max: float) -> float:
    if c_max <= 0:
        raise ConfidenceError(f"C_max must be positive, got {c_max!r}")
    return c_max * beta(m, n, delta)


def rho(m: int, n: int, delta: float, c_max: float, gamma: float) -> float:
    if gamma < 0:
        raise ConfidenceError(f"gamma must be >= 0, got {gamma!r}")
    return (1.0 + gamma) * rho_c(m, n, delta, c_max)


def quality_ucb(q_hat: float, m: int, n: int, delta: float) -> float:
    return min(1.0, q_hat + beta(m, n, delta))


def beta_table(m_max: int, n: int, delta: float) -> np.ndarray:
    """beta(1..m_max) as a 1-indexed array; ``table[0]`` is NaN so an off-by-one shows."""
    _check(1, n, delta)
    if m_max < 1:
        raise ConfidenceError(f"m_max must be >= 1, got {m_max!r}")
    m = np.arange(1, m_max + 1, dtype=np.float64)
    lam = np.log(2.0 * np.pi**2 * n * m * m / (3.0 * delta))
    return np.concatenate(([np.nan], np.sqrt(lam / (2.0 * m))))


def gamma_min_shrinkage(k: float, n: int, delta: float, m_max: int = 200_000) -> float:
    """Smallest gamma that bounds the shrinkage estimator's belief slack.

    Its slack is at most k C_max / (m + k) against an allowance of
    gamma C_max beta(m), so gamma >= max_m k sqrt(2m) / ((m + k) sqrt(Lambda(m))).
    """
    _check(1, n, delta)
    if k <= 0:
        raise ConfidenceError(f"pseudo-count must be positive, got {k!r}")
    m = np.arange(1, m_max + 1, dtype=np.float64)
    lam = np.log(2.0 * np.pi**2 * n * m * m / (3.0 * delta))
    return float(np.max(k * np.sqrt(2.0 * m) / ((m + k) * np.sqrt(lam))))


@dataclass(frozen=True)
class Radii:
    """The public constants (n, delta, C_max, gamma) with the radii as methods."""

    n: int
    delta: float
    c_max: float
    gamma: float = 0.0

    def __post_init__(self) -> None:
        _check(1, self.n, self.delta)
        if self.c_max <= 0:
            raise ConfidenceError(f"C_max must be positive, got {self.c_max!r}")
        if self.gamma < 0:
            raise ConfidenceError(f"gamma must be >= 0, got {self.gamma!r}")

    def Lambda(self, m: int) -> float:
        return Lambda(m, self.n, self.delta)

    def beta(self, m: int) -> float:
        return beta(m, self.n, self.delta)

    def rho_c(self, m: int) -> float:
        return rho_c(m, self.n, self.delta, self.c_max)

    def rho(self, m: int) -> float:
        return rho(m, self.n, self.delta, self.c_max, self.gamma)

    def quality_ucb(self, q_hat: float, m: int) -> float:
        return quality_ucb(q_hat, m, self.n, self.delta)

    def beta_table(self, m_max: int) -> np.ndarray:
        return beta_table(m_max, self.n, self.delta)
