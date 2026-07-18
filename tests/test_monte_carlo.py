"""Tests for Monte Carlo VaR/CVaR with Student-t innovations.

Includes analytical-benchmark tests: with a fixed (not estimated)
degrees-of-freedom parameter and a large simulation count, the Monte
Carlo VaR/CVaR should match the closed-form Student-t quantile/ES
within Monte Carlo error -- the same "code that runs vs. code that is
correct" standard applied throughout this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_service.models.monte_carlo import (
    StudentTFit,
    estimate_dof,
    fit_student_t,
    monte_carlo_cvar,
    monte_carlo_var,
    rolling_monte_carlo_var,
    simulate_portfolio_returns,
)


def _single_asset_frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"asset": values}, index=idx)


class TestEstimateDof:
    def test_recovers_known_dof_from_large_sample(self):
        rng = np.random.default_rng(seed=1)
        true_dof = 6.0
        raw = rng.standard_t(df=true_dof, size=200_000)
        estimated = estimate_dof(raw)
        assert estimated == pytest.approx(true_dof, rel=0.3)

    def test_thin_tailed_data_clips_to_max(self):
        rng = np.random.default_rng(seed=2)
        # Uniform data has negative excess kurtosis.
        raw = rng.uniform(-1, 1, size=10_000)
        assert estimate_dof(raw) == 200.0

    def test_returns_within_bounds(self):
        rng = np.random.default_rng(seed=3)
        raw = rng.standard_t(df=3.0, size=5000)  # very fat-tailed
        dof = estimate_dof(raw)
        assert 4.01 <= dof <= 200.0


class TestFitStudentT:
    def test_moments_match_portfolio_moments(self):
        rng = np.random.default_rng(seed=4)
        raw = rng.normal(0.0005, 0.01, size=2000)
        df = _single_asset_frame(raw)

        fit = fit_student_t(df, [1.0], dof=6.0, use_shrinkage=False)
        assert isinstance(fit, StudentTFit)
        assert fit.mean == pytest.approx(raw.mean())
        assert fit.volatility == pytest.approx(raw.std(ddof=1))
        assert fit.dof == 6.0
        assert fit.shrinkage is None

    def test_dof_none_estimates_automatically(self):
        rng = np.random.default_rng(seed=5)
        raw = rng.standard_t(df=5.0, size=5000) * 0.01
        df = _single_asset_frame(raw)

        fit = fit_student_t(df, [1.0], dof=None, use_shrinkage=False)
        assert fit.dof != 6.0  # sanity: it was actually estimated
        assert 4.01 <= fit.dof <= 200.0

    def test_rejects_invalid_dof(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01]))
        with pytest.raises(ValueError):
            fit_student_t(df, [1.0], dof=2.0)
        with pytest.raises(ValueError):
            fit_student_t(df, [1.0], dof=1.0)


class TestSimulatePortfolioReturns:
    def test_simulated_moments_match_fit(self):
        fit = StudentTFit(mean=0.001, volatility=0.02, dof=6.0, shrinkage=None)
        sim = simulate_portfolio_returns(fit, n_simulations=500_000, seed=7)
        assert sim.mean() == pytest.approx(fit.mean, abs=2e-4)
        assert sim.std() == pytest.approx(fit.volatility, rel=0.02)

    def test_rejects_invalid_n_simulations(self):
        fit = StudentTFit(mean=0.0, volatility=0.01, dof=6.0, shrinkage=None)
        with pytest.raises(ValueError):
            simulate_portfolio_returns(fit, n_simulations=0)

    def test_reproducible_with_seed(self):
        fit = StudentTFit(mean=0.0, volatility=0.01, dof=6.0, shrinkage=None)
        a = simulate_portfolio_returns(fit, n_simulations=1000, seed=42)
        b = simulate_portfolio_returns(fit, n_simulations=1000, seed=42)
        np.testing.assert_array_equal(a, b)


class TestMonteCarloVaR:
    def test_matches_student_t_theoretical_var(self):
        # Analytical benchmark: with dof fixed (not estimated) and a
        # large simulation count, Monte Carlo VaR should match the
        # closed-form Student-t quantile.
        rng = np.random.default_rng(seed=42)
        sigma = 0.01
        dof = 6.0
        n = 20_000
        # Large normal sample so portfolio_moments recovers mean~0,
        # vol~sigma precisely; the *simulation* uses Student-t
        # regardless of the input sample's own shape.
        raw = rng.normal(loc=0.0, scale=sigma, size=n)
        df = _single_asset_frame(raw)

        var_99 = monte_carlo_var(
            df,
            [1.0],
            confidence_level=0.99,
            dof=dof,
            use_shrinkage=False,
            n_simulations=300_000,
            seed=123,
        )

        scale = sigma * np.sqrt((dof - 2.0) / dof)
        theoretical = -scale * stats.t.ppf(0.01, df=dof)

        assert var_99 == pytest.approx(theoretical, rel=0.05)

    def test_higher_confidence_higher_var(self):
        rng = np.random.default_rng(seed=1)
        raw = rng.normal(0.0, 0.01, size=2000)
        df = _single_asset_frame(raw)

        v99 = monte_carlo_var(
            df, [1.0], confidence_level=0.99, dof=6.0,
            use_shrinkage=False, n_simulations=50_000, seed=1,
        )
        v95 = monte_carlo_var(
            df, [1.0], confidence_level=0.95, dof=6.0,
            use_shrinkage=False, n_simulations=50_000, seed=1,
        )
        assert v99 > v95

    def test_fatter_tails_increase_var_at_fixed_volatility(self):
        # Same mean/vol, lower dof (fatter tails) -> higher 99% VaR.
        rng = np.random.default_rng(seed=9)
        raw = rng.normal(0.0, 0.01, size=2000)
        df = _single_asset_frame(raw)

        var_fat = monte_carlo_var(
            df, [1.0], confidence_level=0.99, dof=4.5,
            use_shrinkage=False, n_simulations=100_000, seed=1,
        )
        var_thin = monte_carlo_var(
            df, [1.0], confidence_level=0.99, dof=100.0,
            use_shrinkage=False, n_simulations=100_000, seed=1,
        )
        assert var_fat > var_thin

    def test_rejects_invalid_confidence(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01]))
        with pytest.raises(ValueError):
            monte_carlo_var(df, [1.0], confidence_level=1.0)
        with pytest.raises(ValueError):
            monte_carlo_var(df, [1.0], confidence_level=0.0)


class TestMonteCarloCVaR:
    def test_cvar_at_least_var(self):
        rng = np.random.default_rng(seed=11)
        raw = rng.normal(0.0, 0.01, size=2000)
        df = _single_asset_frame(raw)

        v = monte_carlo_var(
            df, [1.0], confidence_level=0.95, dof=6.0,
            use_shrinkage=False, n_simulations=50_000, seed=2,
        )
        c = monte_carlo_cvar(
            df, [1.0], confidence_level=0.95, dof=6.0,
            use_shrinkage=False, n_simulations=50_000, seed=2,
        )
        assert c >= v

    def test_matches_student_t_theoretical_es(self):
        rng = np.random.default_rng(seed=42)
        sigma = 0.01
        dof = 8.0
        c = 0.99
        n = 20_000
        raw = rng.normal(loc=0.0, scale=sigma, size=n)
        df = _single_asset_frame(raw)

        cvar = monte_carlo_cvar(
            df,
            [1.0],
            confidence_level=c,
            dof=dof,
            use_shrinkage=False,
            n_simulations=300_000,
            seed=123,
        )

        # Closed-form Student-t ES (McNeil, Frey & Embrechts, 2015):
        # ES_c = scale * (nu + t.ppf(alpha)^2) / (nu - 1) * t.pdf(t.ppf(alpha)) / alpha
        scale = sigma * np.sqrt((dof - 2.0) / dof)
        alpha = 1.0 - c
        t_q = stats.t.ppf(alpha, df=dof)
        theoretical = (
            scale
            * (stats.t.pdf(t_q, df=dof) / alpha)
            * ((dof + t_q**2) / (dof - 1.0))
        )

        assert cvar == pytest.approx(theoretical, rel=0.08)


class TestRollingMonteCarloVaR:
    def test_no_lookahead(self):
        rng = np.random.default_rng(seed=3)
        idx = pd.date_range("2020-01-01", periods=260, freq="B")
        raw = rng.normal(0, 0.01, size=260)
        df = pd.DataFrame({"a": raw}, index=idx)

        var_series = rolling_monte_carlo_var(
            df, weights=[1.0], window=250, confidence_level=0.99,
            use_shrinkage=False, n_simulations=5000, seed=0,
        )
        var_before_crash = var_series.iloc[-1]

        crashed = df.copy()
        crashed.iloc[-1, 0] = -0.20
        var_after_crash = rolling_monte_carlo_var(
            crashed, weights=[1.0], window=250, confidence_level=0.99,
            use_shrinkage=False, n_simulations=5000, seed=0,
        ).iloc[-1]

        assert var_before_crash == pytest.approx(var_after_crash)

    def test_first_window_entries_nan(self):
        rng = np.random.default_rng(seed=4)
        raw = rng.normal(0, 0.01, size=70)
        df = _single_asset_frame(raw)

        var_series = rolling_monte_carlo_var(
            df, weights=[1.0], window=50, use_shrinkage=False,
            n_simulations=2000, seed=0,
        )
        assert var_series.iloc[:50].isna().all()
        assert var_series.iloc[50:].notna().all()

    def test_rejects_short_window(self):
        df = _single_asset_frame(np.array([0.01, -0.02, 0.005, 0.01]))
        with pytest.raises(ValueError):
            rolling_monte_carlo_var(df, weights=[1.0], window=1)
