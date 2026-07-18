"""Monte Carlo VaR and CVaR with Student-t innovations.

Simulates portfolio P&L by drawing from a Student-t distribution
calibrated to the historical sample: mean and volatility from
`parametric.portfolio_moments` (optionally Ledoit-Wolf-shrunk), and
degrees of freedom fit from the excess kurtosis of the historical
portfolio return series. VaR/CVaR are then the empirical quantile /
tail mean of the simulated sample -- the same estimator as
historical_var/historical_cvar, just applied to a large synthetic
sample instead of the (finite, possibly short) historical one.

This directly targets the gap documented in parametric.py and the
README: parametric VaR assumes normality and understates deep-tail
risk on fat-tailed data. Student-t innovations restore fat tails while
still letting the simulated sample size be made arbitrarily large,
unlike historical simulation, which is capped at the historical
window length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from risk_service.models.parametric import portfolio_moments

_MIN_DOF = 4.01  # Student-t kurtosis is undefined for dof <= 4.
_MAX_DOF = 200.0  # Effectively normal beyond this; keeps sampling well-behaved.


@dataclass(frozen=True)
class StudentTFit:
    """Fitted portfolio-level Student-t model.

    Attributes
    ----------
    mean : float
        Portfolio mean return per period (w' mu).
    volatility : float
        Portfolio return standard deviation per period, matching the
        *observed* covariance -- i.e. simulated draws are rescaled so
        their variance equals this, not the raw Student-t scale.
    dof : float
        Degrees of freedom of the fitted Student-t distribution.
        Larger dof means closer to normal; dof -> inf recovers the
        normal distribution exactly.
    shrinkage : float | None
        Ledoit-Wolf shrinkage intensity applied to the covariance
        matrix when estimating `volatility`. None if shrinkage was
        not used.
    """

    mean: float
    volatility: float
    dof: float
    shrinkage: float | None


def estimate_dof(portfolio_returns: np.ndarray) -> float:
    """Estimate Student-t degrees of freedom from excess kurtosis.

    A Student-t distribution has excess kurtosis 6 / (dof - 4) for
    dof > 4, so dof = 4 + 6 / excess_kurtosis. This is a simple
    method-of-moments fit, not MLE -- see the README for the
    trade-off (excess kurtosis is a high-variance statistic,
    especially with limited history).

    Parameters
    ----------
    portfolio_returns : np.ndarray
        1-D array of portfolio-level returns.

    Returns
    -------
    float
        Estimated degrees of freedom, clipped to [4.01, 200.0].
        Near-zero or negative sample excess kurtosis (thin-tailed or
        platykurtic data) maps to the upper bound: "close enough to
        normal that Student-t adds little."
    """
    excess_kurtosis = float(
        stats.kurtosis(portfolio_returns, fisher=True, bias=False)
    )
    if excess_kurtosis <= 1e-6:
        return _MAX_DOF
    dof = 4.0 + 6.0 / excess_kurtosis
    return float(np.clip(dof, _MIN_DOF, _MAX_DOF))


def fit_student_t(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    dof: float | None = None,
    use_shrinkage: bool = True,
) -> StudentTFit:
    """Fit a portfolio-level Student-t model to historical asset returns.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Historical per-asset returns, one column per asset.
    weights : array-like
        Portfolio weights, aligned to asset_returns columns by position.
    dof : float | None
        Degrees of freedom to use. If None (default), estimated via
        `estimate_dof` from the historical portfolio return series
        (asset_returns @ weights). Must be > 2 if provided (variance
        is undefined for dof <= 2).
    use_shrinkage : bool
        Whether to apply Ledoit-Wolf shrinkage when estimating
        portfolio volatility; see `parametric.portfolio_moments`.

    Returns
    -------
    StudentTFit
    """
    if dof is not None and dof <= 2.0:
        raise ValueError(f"dof must be > 2, got {dof}")

    moments = portfolio_moments(asset_returns, weights, use_shrinkage=use_shrinkage)

    if dof is None:
        w = np.asarray(weights, dtype=float)
        portfolio_returns = asset_returns.dropna(how="any").to_numpy() @ w
        dof = estimate_dof(portfolio_returns)

    return StudentTFit(
        mean=moments.mean,
        volatility=moments.volatility,
        dof=dof,
        shrinkage=moments.shrinkage,
    )


def simulate_portfolio_returns(
    fit: StudentTFit,
    n_simulations: int = 100_000,
    seed: int | None = None,
) -> np.ndarray:
    """Draw simulated portfolio returns from a fitted Student-t model.

    A standard Student-t with `dof` degrees of freedom has variance
    dof / (dof - 2) (for dof > 2). Draws are rescaled so the simulated
    sample's standard deviation matches `fit.volatility` exactly in
    expectation, not the raw unit-scale t-distribution.

    Parameters
    ----------
    fit : StudentTFit
        Output of `fit_student_t`.
    n_simulations : int
        Number of Monte Carlo draws.
    seed : int | None
        Seed for reproducibility. None uses fresh entropy.

    Returns
    -------
    np.ndarray
        Simulated portfolio returns, shape (n_simulations,).
    """
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

    rng = np.random.default_rng(seed)
    scale = fit.volatility * np.sqrt((fit.dof - 2.0) / fit.dof)
    draws = stats.t.rvs(df=fit.dof, size=n_simulations, random_state=rng)
    return fit.mean + scale * draws


def monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    confidence_level: float = 0.99,
    dof: float | None = None,
    use_shrinkage: bool = True,
    n_simulations: int = 100_000,
    seed: int | None = None,
) -> float:
    """Compute one-period Monte Carlo VaR with Student-t innovations.

    Fits a portfolio-level Student-t distribution to the historical
    sample (see `fit_student_t`), draws a large simulated sample, and
    returns the empirical VaR of that sample -- the same estimator as
    `historical_var`, applied to simulated rather than historical data.

    Returns VaR as a positive number, consistent with historical_var
    and parametric_var.

    Parameters
    ----------
    asset_returns, weights, use_shrinkage
        See `parametric_var`.
    confidence_level : float
        Confidence level in (0, 1).
    dof : float | None
        Degrees of freedom override; see `fit_student_t`.
    n_simulations : int
        Number of Monte Carlo draws. Larger reduces simulation noise
        at the far tail (99%+) at the cost of runtime.
    seed : int | None
        Seed for reproducibility.

    Returns
    -------
    float
        VaR as a positive number.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    fit = fit_student_t(asset_returns, weights, dof=dof, use_shrinkage=use_shrinkage)
    simulated = simulate_portfolio_returns(
        fit, n_simulations=n_simulations, seed=seed
    )
    quantile = np.quantile(simulated, 1.0 - confidence_level)
    return float(max(-quantile, 0.0))


def monte_carlo_cvar(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    confidence_level: float = 0.99,
    dof: float | None = None,
    use_shrinkage: bool = True,
    n_simulations: int = 100_000,
    seed: int | None = None,
) -> float:
    """Compute one-period Monte Carlo CVaR with Student-t innovations.

    See `monte_carlo_var`. CVaR is the mean of simulated returns at or
    below the VaR threshold, mirroring `historical_cvar`.

    Returns
    -------
    float
        CVaR as a positive number. Always >= corresponding VaR.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )

    fit = fit_student_t(asset_returns, weights, dof=dof, use_shrinkage=use_shrinkage)
    simulated = simulate_portfolio_returns(
        fit, n_simulations=n_simulations, seed=seed
    )
    threshold = np.quantile(simulated, 1.0 - confidence_level)
    tail = simulated[simulated <= threshold]
    if tail.size == 0:
        return float(max(-threshold, 0.0))
    return float(max(-tail.mean(), 0.0))


def rolling_monte_carlo_var(
    asset_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    window: int = 250,
    confidence_level: float = 0.99,
    dof: float | None = None,
    use_shrinkage: bool = True,
    n_simulations: int = 20_000,
    seed: int = 0,
) -> pd.Series:
    """Compute rolling one-period Monte Carlo VaR.

    Mirrors rolling_parametric_var / rolling_historical_var: at each
    date t, fits the Student-t model on the prior `window` returns
    (strictly excluding t) and simulates to forecast VaR for t.
    No-lookahead by construction.

    `n_simulations` defaults lower here than in `monte_carlo_var`
    (20,000 vs 100,000) since this runs once per date in the backtest.
    `seed` is fixed by default (rather than None) so a rolling
    backtest is reproducible run to run -- this estimator, unlike the
    closed-form parametric one, is itself stochastic.

    Parameters
    ----------
    asset_returns, weights, confidence_level, dof, use_shrinkage
        See `monte_carlo_var`.
    window : int
        Rolling window length in observations.
    n_simulations : int
        Monte Carlo draws per date.
    seed : int
        Base seed; offset by the date's position so each date's
        simulation is independent but reproducible.

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
        values[i] = monte_carlo_var(
            chunk,
            weights,
            confidence_level=confidence_level,
            dof=dof,
            use_shrinkage=use_shrinkage,
            n_simulations=n_simulations,
            seed=seed + i,
        )

    return pd.Series(values, index=asset_returns.index, name="monte_carlo_var")
