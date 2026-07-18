"""Tests for parametric (variance-covariance) VaR/CVaR.

Includes analytical-benchmark tests: for a large multivariate normal
sample, the parametric estimator should match the theoretical
closed-form VaR/CVaR within Monte Carlo error, since the model
assumption (normality) matches the data-generating process exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_service.models.parametric import (
    PortfolioMoments,
    parametric_cvar,
    parametric_var,
    portfolio_moments,
    rolling_parametric_var,
)


def _single_asset_frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"asset": values}, index=idx)


class TestPortfolioMoments:
    def test_single_asset_matches_sample_mean_std(self):
        rng = np.random.default_rng(seed=1)
        raw = rng.normal(loc=0.0005, scale=0.01, size=5000)
        df = _single_asset_frame(raw)

        moments = portfolio_moments(df, weights=[1.0], use_shrinkage=False)
        assert moments.mean == pytest.approx(raw.mean())
        # np.cov uses ddof=1 (unbiased sample variance) by default.
        assert moments.volatility == pytest.approx(raw.std(ddof=1))
        assert moments.shrinkage is None

    def test_shrinkage_reports_intensity_in_unit_interval(self):
        rng = np.random.default_rng(seed=2)
        raw = rng.normal(0.0, 0.01, size=(300, 5))
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        df = pd.DataFrame(raw, index=idx, columns=list("ABCDE"))

        moments = portfolio_moments(df, weights=[0.2] * 5, use_shrinkage=True)
        assert moments.shrinkage is not None
        assert 0.0 <= moments.shrinkage <= 1.0

    def test_weights_length_mismatch_raises(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01]))
        with pytest.raises(ValueError):
            portfolio_moments(df, weights=[0.5, 0.5])

    def test_insufficient_rows_raises(self):
        df = _single_asset_frame(np.array([0.01]))
        with pytest.raises(ValueError):
            portfolio_moments(df, weights=[1.0])

    def test_drops_rows_with_any_nan(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame(
            {
                "a": [0.01, np.nan, 0.02, 0.01, -0.01],
                "b": [0.02, 0.01, np.nan, 0.0, 0.01],
            },
            index=idx,
        )
        # Only rows 0, 3, 4 are jointly non-NaN.
        moments = portfolio_moments(df, weights=[0.5, 0.5], use_shrinkage=False)
        expected_mean = (
            df.dropna(how="any").to_numpy().mean(axis=0) @ np.array([0.5, 0.5])
        )
        assert moments.mean == pytest.approx(expected_mean)

    def test_result_dataclass_shape(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01, -0.01]))
        moments = portfolio_moments(df, weights=[1.0], use_shrinkage=False)
        assert isinstance(moments, PortfolioMoments)


class TestParametricVaR:
    def test_matches_normal_theoretical_var(self):
        # Analytical benchmark: for X ~ N(0, sigma^2), theoretical 99%
        # VaR is sigma * Phi^-1(0.99) = sigma * 2.326.
        rng = np.random.default_rng(seed=42)
        sigma = 0.01
        n = 20_000
        raw = rng.normal(loc=0.0, scale=sigma, size=n)
        df = _single_asset_frame(raw)

        var_99 = parametric_var(
            df, weights=[1.0], confidence_level=0.99, use_shrinkage=False
        )
        theoretical = sigma * stats.norm.ppf(0.99)

        assert var_99 == pytest.approx(theoretical, rel=0.05)

    def test_higher_confidence_higher_var(self):
        rng = np.random.default_rng(seed=1)
        raw = rng.normal(loc=0.0, scale=0.01, size=5000)
        df = _single_asset_frame(raw)

        v99 = parametric_var(df, [1.0], confidence_level=0.99, use_shrinkage=False)
        v95 = parametric_var(df, [1.0], confidence_level=0.95, use_shrinkage=False)
        v90 = parametric_var(df, [1.0], confidence_level=0.90, use_shrinkage=False)
        assert v99 > v95 > v90

    def test_multi_asset_matches_manual_projection(self):
        # Two independent assets with known variances; portfolio
        # variance under equal weights is a simple closed form.
        rng = np.random.default_rng(seed=7)
        n = 20_000
        sigma_a, sigma_b = 0.01, 0.02
        a = rng.normal(0.0, sigma_a, size=n)
        b = rng.normal(0.0, sigma_b, size=n)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"a": a, "b": b}, index=idx)

        weights = [0.5, 0.5]
        var = parametric_var(df, weights, confidence_level=0.99, use_shrinkage=False)

        portfolio_sigma = np.sqrt(0.5**2 * sigma_a**2 + 0.5**2 * sigma_b**2)
        theoretical = portfolio_sigma * stats.norm.ppf(0.99)
        assert var == pytest.approx(theoretical, rel=0.05)

    def test_rejects_invalid_confidence(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01]))
        with pytest.raises(ValueError):
            parametric_var(df, [1.0], confidence_level=1.0)
        with pytest.raises(ValueError):
            parametric_var(df, [1.0], confidence_level=0.0)


class TestParametricCVaR:
    def test_cvar_at_least_var(self):
        rng = np.random.default_rng(seed=8)
        for _ in range(10):
            raw = rng.normal(0.0, 0.01, size=2000)
            df = _single_asset_frame(raw)
            v = parametric_var(df, [1.0], confidence_level=0.95, use_shrinkage=False)
            c = parametric_cvar(df, [1.0], confidence_level=0.95, use_shrinkage=False)
            assert c >= v - 1e-12

    def test_matches_normal_theoretical_es(self):
        # For X ~ N(0, sigma^2), theoretical ES at level c is
        # sigma * phi(Phi^-1(c)) / (1 - c).
        rng = np.random.default_rng(seed=42)
        sigma = 0.01
        n = 20_000
        c = 0.99
        raw = rng.normal(loc=0.0, scale=sigma, size=n)
        df = _single_asset_frame(raw)

        cvar = parametric_cvar(df, [1.0], confidence_level=c, use_shrinkage=False)
        theoretical = sigma * stats.norm.pdf(stats.norm.ppf(c)) / (1.0 - c)

        assert cvar == pytest.approx(theoretical, rel=0.05)


class TestRollingParametricVaR:
    def test_no_lookahead(self):
        # VaR at date t must be computed using dates strictly < t.
        rng = np.random.default_rng(seed=3)
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        raw = rng.normal(0, 0.01, size=300)
        df = pd.DataFrame({"a": raw}, index=idx)

        var_series = rolling_parametric_var(
            df, weights=[1.0], window=250, confidence_level=0.99, use_shrinkage=False
        )
        var_before_crash = var_series.iloc[-1]

        crashed = df.copy()
        crashed.iloc[-1, 0] = -0.20
        var_after_crash = rolling_parametric_var(
            crashed,
            weights=[1.0],
            window=250,
            confidence_level=0.99,
            use_shrinkage=False,
        ).iloc[-1]

        assert var_before_crash == pytest.approx(var_after_crash)

    def test_first_window_entries_nan(self):
        rng = np.random.default_rng(seed=4)
        raw = rng.normal(0, 0.01, size=100)
        df = _single_asset_frame(raw)

        var_series = rolling_parametric_var(
            df, weights=[1.0], window=50, use_shrinkage=False
        )
        assert var_series.iloc[:50].isna().all()
        assert var_series.iloc[50:].notna().all()

    def test_rejects_short_window(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01]))
        with pytest.raises(ValueError):
            rolling_parametric_var(df, weights=[1.0], window=1)
