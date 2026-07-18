"""Historical simulation VaR and CVaR.

Non-parametric approach. Makes no distributional assumption; uses the
empirical distribution of past returns directly. Robust to fat tails
present in the historical window, blind to tail events not yet observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_var(
    returns: pd.Series | np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """Compute one-period historical VaR.

    Returns VaR as a positive number representing the loss magnitude.
    A 99% VaR of 0.023 means: with 99% confidence, one-period loss will
    not exceed 2.3% of portfolio value (i.e. 1% of the time it will).

    Parameters
    ----------
    returns : array-like
        Historical returns (arithmetic or log, be consistent throughout
        the pipeline). NaNs are dropped.
    confidence_level : float
        Confidence level in (0, 1). Typical: 0.95, 0.99.

    Returns
    -------
    float
        VaR as a positive number. Zero if the empirical quantile is
        non-negative (portfolio only gained in the sample; degenerate case).

    Raises
    ------
    ValueError
        If confidence_level is out of range or returns is empty after
        dropping NaNs.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    r = pd.Series(returns).dropna().to_numpy()
    if r.size == 0:
        raise ValueError("returns is empty after dropping NaNs")

    # (1 - confidence) lower-tail quantile of returns; negate for loss.
    quantile = np.quantile(r, 1.0 - confidence_level)
    return float(max(-quantile, 0.0))


def historical_cvar(
    returns: pd.Series | np.ndarray,
    confidence_level: float = 0.99,
) -> float:
    """Compute one-period historical CVaR (Expected Shortfall).

    CVaR is the expected loss conditional on the loss exceeding VaR.
    Coherent risk measure (subadditive), unlike VaR. Preferred by
    Basel III (FRTB) precisely because VaR ignores the shape of the tail
    beyond the threshold.

    Parameters
    ----------
    returns : array-like
        Historical returns.
    confidence_level : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        CVaR as a positive number. Always >= corresponding VaR by
        construction.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    r = pd.Series(returns).dropna().to_numpy()
    if r.size == 0:
        raise ValueError("returns is empty after dropping NaNs")

    threshold = np.quantile(r, 1.0 - confidence_level)
    tail_losses = r[r <= threshold]

    if tail_losses.size == 0:
        # No returns in tail (can happen with very small samples).
        return float(max(-threshold, 0.0))

    return float(max(-tail_losses.mean(), 0.0))


def rolling_historical_var(
    returns: pd.Series,
    window: int = 250,
    confidence_level: float = 0.99,
) -> pd.Series:
    """Compute rolling one-period historical VaR.

    At each date t, uses the prior `window` returns (strictly excluding
    t itself, to avoid lookahead) to estimate VaR for t. This yields
    a series of VaR estimates aligned with the return index, ready for
    breach counting against realised returns.

    Parameters
    ----------
    returns : pd.Series
        Return series indexed by date.
    window : int
        Rolling window length in observations. 250 approximates one
        trading year.
    confidence_level : float
        Confidence level in (0, 1).

    Returns
    -------
    pd.Series
        VaR estimates indexed by date. First `window` entries are NaN
        (insufficient history).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    # shift(1) is the lookahead guard: VaR for date t uses returns up to t-1.
    return (
        returns.shift(1)
        .rolling(window=window, min_periods=window)
        .apply(
            lambda w: historical_var(w, confidence_level=confidence_level),
            raw=True,
        )
    )
