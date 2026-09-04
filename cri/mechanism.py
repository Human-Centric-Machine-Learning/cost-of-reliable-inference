"""The paper's selection rule and critical payment.

    A_t = {i : q_UCB_i(t) >= Theta}; halt if |A_t| <= 1
    I_t in argmin_{i in A_t} x_i(t), exact ties uniform at random
    P_i(t) = min{ min_{j in A_t \\ {i}} x_j(t) + rho(m_i(t)), C_max }

P_i(t) does not depend on b_i(t), so i wins iff b_i(t) < P_i(t) (the threshold lemma).
The tie-break draw is passed in by the caller so that compared policies share
one round-indexed tie stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cri.platform import PlatformError, PlatformState


class MechanismError(ValueError):
    """An invalid or undefined mechanism operation."""


@dataclass(frozen=True)
class Selection:
    """The outcome of one selection step, before anything is revealed."""

    winner: str | None
    eligible: tuple[str, ...]
    scores: dict[str, float]
    tied: tuple[str, ...] = ()
    halted: bool = False

    @property
    def n_eligible(self) -> int:
        return len(self.eligible)


def _argmin_set(scores: dict[str, float], among: Sequence[str]) -> list[str]:
    best = min(scores[m] for m in among)
    return [m for m in among if scores[m] == best]


def select(state: PlatformState, tie_u: float) -> Selection:
    """The selection rule with the halt condition; ``tie_u`` is a uniform draw in [0, 1)."""
    if not 0.0 <= tie_u < 1.0:
        raise MechanismError(f"tie_u must lie in [0, 1), got {tie_u!r}")

    eligible = state.eligible()
    scores = state.scores()
    if len(eligible) <= 1:
        return Selection(
            winner=None, eligible=tuple(eligible), scores=scores, halted=True
        )

    tied = _argmin_set(scores, eligible)
    # tie_u < 1 keeps the index below len(tied); min guards float rounding.
    winner = tied[min(int(tie_u * len(tied)), len(tied) - 1)]
    return Selection(
        winner=winner, eligible=tuple(eligible), scores=scores, tied=tuple(tied)
    )


def critical_payment(
    state: PlatformState, model: str, eligible: Sequence[str]
) -> float:
    """The critical payment: the runner-up score plus the provider's own radius, capped at C_max."""
    eligible = list(eligible)
    if model not in eligible:
        raise MechanismError(f"{model!r} is not eligible this round")
    others = [j for j in eligible if j != model]
    if not others:
        raise MechanismError(
            "a round is played only when |A_t| >= 2; there is no competing score"
        )
    try:
        p = state.providers[model]
    except KeyError:
        raise PlatformError(f"{model!r} is not in this roster") from None

    runner_up = min(state.score(j) for j in others)
    return min(runner_up + state.radii.rho(p.m), state.radii.c_max)
