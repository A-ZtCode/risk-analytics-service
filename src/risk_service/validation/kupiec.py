"""Kupiec Proportion of Failures (POF) test for VaR backtesting.

Given a VaR model at confidence level c, the expected breach probability
per period is p = 1 - c. The Kupiec test asks: is the observed breach
rate statistically consistent with p, or does the model materially
over- or under-state risk?

Reference: Kupiec (1995), "Techniques for Verifying the Accuracy of Risk
Measurement Models".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class KupiecResult:
    """Result of a Kupiec POF test.

    Attributes
    ----------
    breaches : int
        Number of observed VaR breaches (realised loss > VaR).
    observations : int
        Number of valid (VaR, return) pairs tested.
    breach_rate : float
        breaches / observations.
    expected_rate : float
        1 - confidence_level (theoretical breach rate).
    lr_statistic : float
        Likelihood ratio test statistic, ~ chi-squared with 1 df under H0.
    p_value : float
        Two-sided p-value. Small p rejects H0 (model is miscalibrated).
    reject_null : bool
        True if p_value < significance level (default 0.05).
    """

    breaches: int
    observations: int
    breach_rate: float
    expected_rate: float
    lr_statistic: float
    p_value: float
    reject_null: bool


def kupiec_pof_test(
    var_estimates: pd.Series,
    realised_returns: pd.Series,
    confidence_level: float = 0.99,
    significance: float = 0.05,
) -> KupiecResult:
    """Run the Kupiec Proportion of Failures test.

    A breach occurs when the realised loss (-return) exceeds the VaR
    forecast for that period. Under H0 (model correctly calibrated),
    breaches are Bernoulli(p) with p = 1 - confidence_level, and the
    LR statistic follows chi-squared(1) asymptotically.

    Parameters
    ----------
    var_estimates : pd.Series
        VaR forecasts (positive numbers), indexed by date.
    realised_returns : pd.Series
        Realised returns, indexed by date. Alignment against
        var_estimates is done by index; unmatched or NaN pairs are
        dropped.
    confidence_level : float
        VaR confidence level used to produce the estimates.
    significance : float
        Test significance level for the reject_null decision.

    Returns
    -------
    KupiecResult
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
    if not 0.0 < significance < 1.0:
        raise ValueError(
            f"significance must be in (0, 1), got {significance}"
        )

    aligned = pd.concat(
        [var_estimates.rename("var"), realised_returns.rename("ret")],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError(
            "No overlapping non-NaN (VaR, return) pairs available"
        )

    n = len(aligned)
    losses = -aligned["ret"].to_numpy()
    breaches_mask = losses > aligned["var"].to_numpy()
    x = int(breaches_mask.sum())

    p = 1.0 - confidence_level
    breach_rate = x / n

    lr = _kupiec_lr_statistic(x=x, n=n, p=p)
    # chi-squared with 1 df, upper-tail p-value.
    p_value = float(stats.chi2.sf(lr, df=1))

    return KupiecResult(
        breaches=x,
        observations=n,
        breach_rate=breach_rate,
        expected_rate=p,
        lr_statistic=lr,
        p_value=p_value,
        reject_null=p_value < significance,
    )


def _kupiec_lr_statistic(x: int, n: int, p: float) -> float:
    """Kupiec unconditional coverage LR statistic.

    LR_uc = -2 * log( (p^x (1-p)^(n-x)) / (p_hat^x (1-p_hat)^(n-x)) )

    Handles the boundary cases x == 0 and x == n where the MLE
    p_hat lies on the boundary of the parameter space and the naive
    formula produces log(0).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= x <= n:
        raise ValueError(f"x must be in [0, n], got x={x}, n={n}")

    if x == 0:
        # p_hat = 0; second term is 0*log(0) = 0 by convention.
        return float(-2.0 * n * np.log(1.0 - p))
    if x == n:
        return float(-2.0 * n * np.log(p))

    p_hat = x / n
    log_l0 = x * np.log(p) + (n - x) * np.log(1.0 - p)
    log_l1 = x * np.log(p_hat) + (n - x) * np.log(1.0 - p_hat)
    return float(-2.0 * (log_l0 - log_l1))
