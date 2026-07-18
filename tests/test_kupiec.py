"""Tests for the Kupiec POF test.

The LR statistic has a closed form. We test it against manually
computed values, then also verify the calibration property: for a
model that IS correctly calibrated, the rejection rate over many
independent simulations should approximate the significance level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_service.validation.kupiec import (
    KupiecResult,
    _kupiec_lr_statistic,
    kupiec_pof_test,
)


class TestLRStatistic:
    def test_zero_breaches(self):
        # When x = 0: LR = -2 * n * log(1 - p)
        n, p = 250, 0.01
        expected = -2.0 * n * np.log(1.0 - p)
        assert _kupiec_lr_statistic(x=0, n=n, p=p) == pytest.approx(expected)

    def test_all_breaches(self):
        # When x = n: LR = -2 * n * log(p)
        n, p = 250, 0.01
        expected = -2.0 * n * np.log(p)
        assert _kupiec_lr_statistic(x=n, n=n, p=p) == pytest.approx(expected)

    def test_exact_calibration_gives_zero(self):
        # When x/n == p, p_hat == p, log-likelihoods equal, LR = 0.
        n, p = 1000, 0.01
        x = 10  # exactly p * n
        assert _kupiec_lr_statistic(x=x, n=n, p=p) == pytest.approx(0.0, abs=1e-10)

    def test_manual_calculation(self):
        # n=250, p=0.01, x=8 (overshooting expected 2.5).
        # p_hat = 8/250 = 0.032
        # log_L0 = 8*log(0.01) + 242*log(0.99)
        # log_L1 = 8*log(0.032) + 242*log(0.968)
        # LR = -2 * (log_L0 - log_L1)
        n, p, x = 250, 0.01, 8
        p_hat = x / n
        log_l0 = x * np.log(p) + (n - x) * np.log(1 - p)
        log_l1 = x * np.log(p_hat) + (n - x) * np.log(1 - p_hat)
        expected = -2.0 * (log_l0 - log_l1)
        assert _kupiec_lr_statistic(x=x, n=n, p=p) == pytest.approx(expected)


class TestKupiecPOF:
    def _make_series(
        self, var_values: list[float], returns: list[float]
    ) -> tuple[pd.Series, pd.Series]:
        idx = pd.date_range("2020-01-01", periods=len(var_values), freq="B")
        return pd.Series(var_values, index=idx), pd.Series(returns, index=idx)

    def test_counts_breaches_correctly(self):
        # Loss = -return. Breach when loss > VaR.
        # Returns: -0.03, -0.01, +0.02, -0.05
        # Losses:  +0.03, +0.01, -0.02, +0.05
        # VaR:     0.02,  0.02,  0.02,  0.02
        # Breaches: [T, F, F, T] -> 2
        var, ret = self._make_series(
            [0.02, 0.02, 0.02, 0.02], [-0.03, -0.01, 0.02, -0.05]
        )
        result = kupiec_pof_test(var, ret, confidence_level=0.99)
        assert result.breaches == 2
        assert result.observations == 4

    def test_calibrated_model_rarely_rejects(self):
        # Property test: simulate many independent Bernoulli(p) breach
        # sequences with p matching the model. Rejection rate should be
        # close to the significance level (5%).
        rng = np.random.default_rng(seed=12345)
        n = 500
        p = 0.01
        var_val = 0.02

        # Build fixed VaR = var_val. Generate returns: with probability
        # p we produce a breach (loss > var_val), else no breach.
        rejects = 0
        trials = 300
        for _ in range(trials):
            breaches = rng.random(size=n) < p
            # Loss = var_val * 1.5 on breach dates, else 0.
            losses = np.where(breaches, var_val * 1.5, 0.0)
            returns = -losses
            idx = pd.date_range("2020-01-01", periods=n, freq="B")
            result = kupiec_pof_test(
                pd.Series(var_val, index=idx),
                pd.Series(returns, index=idx),
                confidence_level=1.0 - p,
                significance=0.05,
            )
            if result.reject_null:
                rejects += 1

        rejection_rate = rejects / trials
        # 5% nominal; allow generous MC tolerance.
        assert 0.01 < rejection_rate < 0.12

    def test_severely_miscalibrated_model_rejects(self):
        # Model claims 99% VaR (expected 1% breach rate) but reality
        # produces 10% breaches. Should reject strongly.
        rng = np.random.default_rng(seed=99)
        n = 500
        var_val = 0.02
        breaches = rng.random(size=n) < 0.10
        losses = np.where(breaches, var_val * 1.5, 0.0)
        returns = -losses
        idx = pd.date_range("2020-01-01", periods=n, freq="B")

        result = kupiec_pof_test(
            pd.Series(var_val, index=idx),
            pd.Series(returns, index=idx),
            confidence_level=0.99,
        )
        assert result.reject_null
        assert result.p_value < 1e-6

    def test_result_dataclass_shape(self):
        var, ret = self._make_series(
            [0.02, 0.02, 0.02], [-0.01, -0.03, 0.01]
        )
        result = kupiec_pof_test(var, ret, confidence_level=0.95)
        assert isinstance(result, KupiecResult)
        assert result.observations == 3
        assert result.expected_rate == pytest.approx(0.05)

    def test_empty_alignment_raises(self):
        idx1 = pd.date_range("2020-01-01", periods=3, freq="B")
        idx2 = pd.date_range("2021-01-01", periods=3, freq="B")
        with pytest.raises(ValueError):
            kupiec_pof_test(
                pd.Series([0.02, 0.02, 0.02], index=idx1),
                pd.Series([-0.01, -0.01, -0.01], index=idx2),
            )
