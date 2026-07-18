"""Tests for historical VaR/CVaR.

Includes analytical-benchmark tests: for a large sample from a known
distribution, the empirical VaR should match the theoretical quantile
within Monte Carlo error. That check is what distinguishes 'the code
runs' from 'the code is correct'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_service.models.historical import (
    historical_cvar,
    historical_var,
    rolling_historical_var,
)


class TestHistoricalVaR:
    def test_matches_normal_theoretical_quantile(self):
        # Analytical benchmark: for X ~ N(0, sigma^2), the theoretical
        # 99% VaR is -sigma * Phi^(-1)(0.01) = sigma * 2.326.
        rng = np.random.default_rng(seed=42)
        sigma = 0.01
        n = 100_000
        returns = rng.normal(loc=0.0, scale=sigma, size=n)

        var_99 = historical_var(returns, confidence_level=0.99)
        theoretical = -sigma * stats.norm.ppf(0.01)

        assert var_99 == pytest.approx(theoretical, rel=0.02)

    def test_var_positive_for_typical_returns(self):
        rng = np.random.default_rng(seed=0)
        returns = rng.normal(loc=0.0005, scale=0.01, size=1000)
        assert historical_var(returns, confidence_level=0.95) > 0

    def test_higher_confidence_higher_var(self):
        rng = np.random.default_rng(seed=1)
        returns = rng.normal(loc=0.0, scale=0.01, size=10_000)
        assert (
            historical_var(returns, 0.99)
            > historical_var(returns, 0.95)
            > historical_var(returns, 0.90)
        )

    def test_rejects_invalid_confidence(self):
        returns = np.array([0.01, -0.02, 0.005])
        with pytest.raises(ValueError):
            historical_var(returns, confidence_level=1.0)
        with pytest.raises(ValueError):
            historical_var(returns, confidence_level=0.0)

    def test_rejects_empty_returns(self):
        with pytest.raises(ValueError):
            historical_var(pd.Series([np.nan, np.nan]))

    def test_drops_nans(self):
        returns = pd.Series([0.01, np.nan, -0.02, 0.005, np.nan])
        # Should not raise; NaNs dropped silently.
        v = historical_var(returns, confidence_level=0.95)
        assert v >= 0.0


class TestHistoricalCVaR:
    def test_cvar_at_least_var(self):
        # CVaR >= VaR by construction. Property test on random samples.
        rng = np.random.default_rng(seed=7)
        for _ in range(20):
            returns = rng.standard_t(df=4, size=2000) * 0.01
            v = historical_var(returns, confidence_level=0.95)
            c = historical_cvar(returns, confidence_level=0.95)
            assert c >= v - 1e-12

    def test_matches_normal_theoretical_es(self):
        # For X ~ N(0, sigma^2), theoretical ES at level c is
        # sigma * phi(Phi^-1(1-c)) / (1 - c).
        rng = np.random.default_rng(seed=42)
        sigma = 0.01
        n = 200_000
        c = 0.99
        returns = rng.normal(loc=0.0, scale=sigma, size=n)

        cvar = historical_cvar(returns, confidence_level=c)
        alpha = 1.0 - c
        theoretical = sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha

        assert cvar == pytest.approx(theoretical, rel=0.03)


class TestRollingHistoricalVaR:
    def test_no_lookahead(self):
        # Verify: VaR at date t is computed using dates strictly < t.
        # Construct returns where the last date is a huge crash; VaR
        # on that date must not be influenced by it.
        rng = np.random.default_rng(seed=3)
        idx = pd.date_range("2020-01-01", periods=300, freq="B")
        returns = pd.Series(rng.normal(0, 0.01, size=300), index=idx)

        var_series = rolling_historical_var(
            returns, window=250, confidence_level=0.99
        )
        var_before_crash = var_series.iloc[-1]

        # Inject a crash on the final date, recompute.
        crashed = returns.copy()
        crashed.iloc[-1] = -0.20
        var_after_crash = rolling_historical_var(
            crashed, window=250, confidence_level=0.99
        ).iloc[-1]

        # Same VaR: the crash on date t does not affect VaR(t).
        assert var_before_crash == pytest.approx(var_after_crash)

    def test_first_window_entries_nan(self):
        rng = np.random.default_rng(seed=4)
        returns = pd.Series(rng.normal(0, 0.01, size=100))
        var_series = rolling_historical_var(returns, window=50)
        assert var_series.iloc[:50].isna().all()
        assert var_series.iloc[50:].notna().all()
