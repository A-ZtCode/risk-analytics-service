"""Christoffersen (1998) independence and conditional coverage tests.

Kupiec's POF test (`kupiec.py`) checks unconditional coverage: the
right number of breaches on average. It says nothing about *when*
those breaches occur. A model that produces exactly the right breach
count but with every breach clustered into one turbulent week is
clearly miscalibrated -- it failed to capture volatility clustering --
and Kupiec cannot detect that.

Christoffersen's independence test treats the breach indicator
sequence as a first-order Markov chain and asks whether the
probability of a breach tomorrow depends on whether there was a
breach today. Under H0 (independence), it should not. Combined with
Kupiec's unconditional coverage statistic, the conditional coverage
test LR_cc = LR_uc + LR_ind jointly checks "right frequency" and "no
clustering", chi-squared with 2 degrees of freedom under H0.

Reference: Christoffersen, P. (1998). "Evaluating Interval Forecasts."
International Economic Review, 39(4), 841-862.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from risk_service.validation.kupiec import _kupiec_lr_statistic


@dataclass(frozen=True)
class ChristoffersenResult:
    """Result of Christoffersen's independence and conditional coverage tests.

    Attributes
    ----------
    breaches : int
        Number of breaches within the transition sample (n01 + n11).
    observations : int
        Number of consecutive (t-1, t) pairs tested, i.e. one fewer
        than the number of (VaR, return) pairs, since the first
        observation has no predecessor to form a transition with.
    transition_counts : tuple[int, int, int, int]
        (n00, n01, n10, n11): counts of (no breach -> no breach),
        (no breach -> breach), (breach -> no breach), (breach -> breach).
    pi01 : float
        P(breach at t | no breach at t-1), MLE. NaN if the "no breach"
        state was never visited.
    pi11 : float
        P(breach at t | breach at t-1), MLE. NaN if the "breach" state
        was never visited.
    lr_independence : float
        LR statistic for the independence test, ~ chi-squared(1) under H0.
    p_value_independence : float
        p-value for the independence test. Small p rejects H0 (breaches
        cluster rather than occurring independently).
    lr_conditional_coverage : float
        LR_uc + LR_ind, ~ chi-squared(2) under H0. LR_uc here is
        computed on the transition sample (`observations`, `breaches`)
        for internal consistency with LR_ind, so it will differ
        slightly from a standalone `kupiec_pof_test` run on the same
        data (which uses all observations, not transitions).
    p_value_conditional_coverage : float
        p-value for the joint test.
    reject_independence : bool
        True if p_value_independence < significance.
    reject_conditional_coverage : bool
        True if p_value_conditional_coverage < significance.
    """

    breaches: int
    observations: int
    transition_counts: tuple[int, int, int, int]
    pi01: float
    pi11: float
    lr_independence: float
    p_value_independence: float
    lr_conditional_coverage: float
    p_value_conditional_coverage: float
    reject_independence: bool
    reject_conditional_coverage: bool


def christoffersen_test(
    var_estimates: pd.Series,
    realised_returns: pd.Series,
    confidence_level: float = 0.99,
    significance: float = 0.05,
) -> ChristoffersenResult:
    """Run Christoffersen's independence and conditional coverage tests.

    A breach occurs when the realised loss (-return) exceeds the VaR
    forecast for that period. Alignment against var_estimates is done
    by index; unmatched or NaN pairs are dropped, mirroring
    kupiec_pof_test.

    Parameters
    ----------
    var_estimates : pd.Series
        VaR forecasts (positive numbers), indexed by date.
    realised_returns : pd.Series
        Realised returns, indexed by date.
    confidence_level : float
        VaR confidence level used to produce the estimates.
    significance : float
        Test significance level for the reject_* decisions.

    Returns
    -------
    ChristoffersenResult

    Raises
    ------
    ValueError
        If confidence_level/significance are out of range, there are
        no overlapping (VaR, return) pairs, or fewer than 2 pairs
        (need at least one transition to test independence).
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
    if not 0.0 < significance < 1.0:
        raise ValueError(f"significance must be in (0, 1), got {significance}")

    aligned = pd.concat(
        [var_estimates.rename("var"), realised_returns.rename("ret")],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError("No overlapping non-NaN (VaR, return) pairs available")
    if len(aligned) < 2:
        raise ValueError("Need at least 2 observations to form a transition")

    losses = -aligned["ret"].to_numpy()
    breach = (losses > aligned["var"].to_numpy()).astype(int)

    prev, curr = breach[:-1], breach[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n_transitions = n00 + n01 + n10 + n11
    x = n01 + n11  # breaches within the transition sample

    lr_ind = _christoffersen_lr_independence(n00=n00, n01=n01, n10=n10, n11=n11)
    p_ind = float(stats.chi2.sf(lr_ind, df=1))

    p = 1.0 - confidence_level
    lr_uc = _kupiec_lr_statistic(x=x, n=n_transitions, p=p)
    lr_cc = lr_uc + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, df=2))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else float("nan")
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else float("nan")

    return ChristoffersenResult(
        breaches=x,
        observations=n_transitions,
        transition_counts=(n00, n01, n10, n11),
        pi01=pi01,
        pi11=pi11,
        lr_independence=lr_ind,
        p_value_independence=p_ind,
        lr_conditional_coverage=lr_cc,
        p_value_conditional_coverage=p_cc,
        reject_independence=p_ind < significance,
        reject_conditional_coverage=p_cc < significance,
    )


def _christoffersen_lr_independence(n00: int, n01: int, n10: int, n11: int) -> float:
    """Christoffersen independence LR statistic.

    H0 (independence): the breach probability tomorrow does not depend
    on whether there was a breach today, i.e. pi01 = pi11 = pi.
    H1 (first-order Markov): pi01 and pi11 estimated separately.

    LR_ind = -2 * log(L0 / L1), where
        L0 = (1-pi)^(n00+n10)  * pi^(n01+n11)
        L1 = (1-pi01)^n00 * pi01^n01 * (1-pi11)^n10 * pi11^n11

    Uses the 0 * log(0) = 0 convention for boundary cases (a state
    never visited, or zero breaches), matching
    kupiec._kupiec_lr_statistic.
    """
    n_total = n00 + n01 + n10 + n11
    if n_total <= 0:
        raise ValueError("no transitions available")

    n0 = n00 + n01  # times in the "no breach yesterday" state
    n1 = n10 + n11  # times in the "breach yesterday" state
    n_breach = n01 + n11

    pi = n_breach / n_total
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0

    def term(count: int, prob: float) -> float:
        return 0.0 if count == 0 else count * np.log(prob)

    log_l0 = term(n00 + n10, 1.0 - pi) + term(n_breach, pi)
    log_l1 = (
        term(n00, 1.0 - pi01)
        + term(n01, pi01)
        + term(n10, 1.0 - pi11)
        + term(n11, pi11)
    )

    return float(-2.0 * (log_l0 - log_l1))
