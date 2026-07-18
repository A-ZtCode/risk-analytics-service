"""Parametric (variance-covariance) VaR and CVaR.

Assumes portfolio returns are normally distributed. Estimates the mean
and covariance of asset returns from the historical sample, optionally
applying Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004) to the covariance
matrix, then projects onto the portfolio via the weight vector and
derives VaR/CVaR from the normal closed form.

Fast and stable in the many-assets/limited-history regime where the
sample covariance matrix is noisy or ill-conditioned; unlike historical
simulation, it also extrapolates into the tail rather than being bound
to observed quantiles. The cost is the normality assumption itself:
real returns are fat-tailed, so parametric VaR tends to *understate*
tail risk relative to historical simulation. See historical.py for the
non-parametric alternative and the README for further discussion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class PortfolioMoments:
    """Estimated portfolio mean and volatility.

    Attributes
    ----------
    mean : float
        Estimated portfolio mean return per period (w' mu).
    volatility : float
        Estimated portfolio return standard deviation per period
        (sqrt(w' Sigma w)).
    shrinkage : float | None
        Ledoit-Wolf shrinkage intensity in [0, 1] applied to the sample
        covariance (0 = pure sample covariance, 1 = fully shrunk to the
        scaled-identity target). None if shrinkage was not used.
    """

    mean: float
    volatility: float
    shrinkage: float | None


def portfolio_moments(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    use_shrinkage: bool = True,
) -> PortfolioMoments:
    """Estimate portfolio mean and volatility from asset returns.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Historical returns, one column per asset, rows indexed by date.
        Rows with any NaN are dropped (assets must be jointly observed
        to estimate a covariance matrix).
    weights : array-like
        Portfolio weights, aligned to asset_returns columns by position.
        Not required to sum to 1 (e.g. leveraged or net-short books).
    use_shrinkage : bool
        If True (default), estimate the covariance matrix with
        Ledoit-Wolf shrinkage instead of the raw sample covariance.
        Recommended whenever the asset count is not small relative to
        the observation count, where the sample covariance is noisy or
        singular. With a single asset this degenerates to a (mildly)
        shrunk variance estimate.

    Returns
    -------
    PortfolioMoments

    Raises
    ------
    ValueError
        If weights length does not match the number of columns, or
        fewer than 2 jointly-observed rows remain after dropping NaNs.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != asset_returns.shape[1]:
        raise ValueError(
            f"weights length ({w.shape[0]}) must match number of assets "
            f"({asset_returns.shape[1]})"
        )

    r = asset_returns.dropna(how="any").to_numpy()
    if r.shape[0] < 2:
        raise ValueError(
            "need at least 2 jointly-observed rows to estimate covariance"
        )

    mu = r.mean(axis=0)

    shrinkage: float | None = None
    if use_shrinkage:
        lw = LedoitWolf().fit(r)
        cov = lw.covariance_
        shrinkage = float(lw.shrinkage_)
    else:
        cov = np.atleast_2d(np.cov(r, rowvar=False))

    portfolio_mean = float(w @ mu)
    # Clip float noise: w' Sigma w can go slightly negative when Sigma
    # is near-singular and unshrunk.
    portfolio_variance = max(float(w @ cov @ w), 0.0)

    return PortfolioMoments(
        mean=portfolio_mean,
        volatility=float(np.sqrt(portfolio_variance)),
        shrinkage=shrinkage,
    )


def parametric_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    confidence_level: float = 0.99,
    use_shrinkage: bool = True,
) -> float:
    """Compute one-period parametric (variance-covariance) VaR.

    Assumes portfolio returns are normally distributed:
    R_p ~ N(mu_p, sigma_p^2), with mu_p and sigma_p estimated from the
    historical asset return sample (optionally with Ledoit-Wolf
    covariance shrinkage; see portfolio_moments). VaR follows the
    closed form:

        VaR_c = sigma_p * Phi^-1(c) - mu_p

    Returns VaR as a positive number representing the loss magnitude,
    consistent with historical_var.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Historical per-asset returns, one column per asset.
    weights : array-like
        Portfolio weights, aligned to asset_returns columns by position.
    confidence_level : float
        Confidence level in (0, 1). Typical: 0.95, 0.99.
    use_shrinkage : bool
        Whether to apply Ledoit-Wolf shrinkage to the covariance matrix.

    Returns
    -------
    float
        VaR as a positive number.

    Raises
    ------
    ValueError
        If confidence_level is out of range, or via portfolio_moments.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    moments = portfolio_moments(asset_returns, weights, use_shrinkage=use_shrinkage)
    z = stats.norm.ppf(confidence_level)
    return float(max(moments.volatility * z - moments.mean, 0.0))


def parametric_cvar(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    confidence_level: float = 0.99,
    use_shrinkage: bool = True,
) -> float:
    """Compute one-period parametric (variance-covariance) CVaR.

    Closed-form Expected Shortfall under normality:

        CVaR_c = sigma_p * phi(Phi^-1(c)) / (1 - c) - mu_p

    Parameters
    ----------
    asset_returns, weights, confidence_level, use_shrinkage
        See parametric_var.

    Returns
    -------
    float
        CVaR as a positive number. Always >= corresponding VaR.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    moments = portfolio_moments(asset_returns, weights, use_shrinkage=use_shrinkage)
    z = stats.norm.ppf(confidence_level)
    es_factor = stats.norm.pdf(z) / (1.0 - confidence_level)
    return float(max(moments.volatility * es_factor - moments.mean, 0.0))


def rolling_parametric_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    window: int = 250,
    confidence_level: float = 0.99,
    use_shrinkage: bool = True,
) -> pd.Series:
    """Compute rolling one-period parametric VaR.

    Mirrors rolling_historical_var: at each date t, uses the prior
    `window` returns (strictly excluding t) to estimate portfolio mean
    and covariance, then forecasts VaR for t. No-lookahead by
    construction, ready for breach counting (e.g. kupiec_pof_test)
    against a realised portfolio return series (asset_returns @ weights).

    Parameters
    ----------
    asset_returns, weights, confidence_level, use_shrinkage
        See parametric_var.
    window : int
        Rolling window length in observations.

    Returns
    -------
    pd.Series
        VaR estimates indexed by date. First `window` entries are NaN.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    shifted = asset_returns.shift(1)
    n = len(asset_returns.index)
    values = np.full(n, np.nan)

    for i in range(window, n):
        chunk = shifted.iloc[i - window : i]
        values[i] = parametric_var(
            chunk,
            weights,
            confidence_level=confidence_level,
            use_shrinkage=use_shrinkage,
        )

    return pd.Series(values, index=asset_returns.index, name="parametric_var")
