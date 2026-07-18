"""Tests for the Christoffersen independence and conditional coverage tests.

Mirrors test_kupiec.py: manual calculations for the LR statistic, plus
the property that matters most here -- a model whose breaches cluster
should be flagged by the independence test even when it has exactly
the "right" breach count that a standalone Kupiec test would pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_service.validation.christoffersen import (
    ChristoffersenResult,
    _christoffersen_lr_independence,
    christoffersen_test,
)
from risk_service.validation.kupiec import kupiec_pof_test


def _make_series(
    var_values: list[float], returns: list[float]
) -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2020-01-01", periods=len(var_values), freq="B")
    return pd.Series(var_values, index=idx), pd.Series(returns, index=idx)


class TestLRIndependence:
    def test_no_breaches_gives_zero(self):
        # No breaches at all: independence can't be rejected, LR = 0.
        assert _christoffersen_lr_independence(n00=100, n01=0, n10=0, n11=0) == (
            pytest.approx(0.0, abs=1e-10)
        )

    def test_perfectly_iid_breach_pattern_gives_low_lr(self):
        # pi01 == pi11 exactly -> L0 == L1 -> LR = 0.
        # n01/n0 == n11/n1: pick n00=90, n01=10 (pi01=0.1),
        # n10=18, n11=2 (pi11=0.1).
        lr = _christoffersen_lr_independence(n00=90, n01=10, n10=18, n11=2)
        assert lr == pytest.approx(0.0, abs=1e-8)

    def test_manual_calculation(self):
        n00, n01, n10, n11 = 80, 5, 5, 10
        n_total = n00 + n01 + n10 + n11
        n0, n1 = n00 + n01, n10 + n11
        n_breach = n01 + n11
        pi = n_breach / n_total
        pi01 = n01 / n0
        pi11 = n11 / n1

        log_l0 = (n00 + n10) * np.log(1 - pi) + n_breach * np.log(pi)
        log_l1 = (
            n00 * np.log(1 - pi01)
            + n01 * np.log(pi01)
            + n10 * np.log(1 - pi11)
            + n11 * np.log(pi11)
        )
        expected = -2.0 * (log_l0 - log_l1)

        assert _christoffersen_lr_independence(
            n00=n00, n01=n01, n10=n10, n11=n11
        ) == pytest.approx(expected)

    def test_clustered_breaches_gives_high_lr(self):
        # All breaches immediately follow another breach: n01=0 (no
        # breach never transitions to breach), n11 large. Strong
        # violation of independence.
        lr = _christoffersen_lr_independence(n00=50, n01=0, n10=1, n11=9)
        assert lr > 5.0  # comfortably above the chi2(1) 95% critical value (3.84)

    def test_rejects_zero_transitions(self):
        with pytest.raises(ValueError):
            _christoffersen_lr_independence(n00=0, n01=0, n10=0, n11=0)


class TestChristoffersenTest:
    def test_transition_counts_correct(self):
        # Breach sequence from losses > VaR:
        # returns: -0.03, -0.01, 0.02, -0.05, -0.06, 0.01
        # losses:   0.03,  0.01, -0.02, 0.05,  0.06, -0.01
        # VaR:      0.02,  0.02,  0.02, 0.02,  0.02,  0.02
        # breach:   1,     0,     0,    1,     1,     0
        # transitions (prev->curr): 1->0, 0->0, 0->1, 1->1, 1->0
        # n00=1 (0->0), n01=1 (0->1), n10=2 (1->0), n11=1 (1->1)
        var, ret = _make_series(
            [0.02] * 6, [-0.03, -0.01, 0.02, -0.05, -0.06, 0.01]
        )
        result = christoffersen_test(var, ret, confidence_level=0.99)
        assert result.transition_counts == (1, 1, 2, 1)
        assert result.observations == 5
        assert result.breaches == 2  # n01 + n11

    def test_clustered_breaches_rejected_but_kupiec_passes(self):
        # Construct a sequence with exactly the expected breach count
        # (10 out of 500, matching p=0.02) but all breaches bunched
        # together in a single run. Kupiec (unconditional coverage)
        # should be satisfied; Christoffersen's independence test
        # should not be.
        n = 500
        var_val = 0.02
        losses = np.zeros(n)
        # 10 consecutive breaches in the middle of the sample.
        losses[200:210] = var_val * 1.5
        returns = -losses
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        var_series = pd.Series(var_val, index=idx)
        ret_series = pd.Series(returns, index=idx)

        kupiec_result = kupiec_pof_test(
            var_series, ret_series, confidence_level=0.98
        )
        assert not kupiec_result.reject_null  # right breach count on average

        cc_result = christoffersen_test(
            var_series, ret_series, confidence_level=0.98
        )
        assert cc_result.reject_independence
        assert cc_result.reject_conditional_coverage

    def test_iid_calibrated_model_rarely_rejects_independence(self):
        # Property test: simulate iid Bernoulli(p) breach sequences.
        # Independence-test rejection rate should approximate the
        # significance level.
        rng = np.random.default_rng(seed=2024)
        n = 500
        p = 0.02
        var_val = 0.02

        rejects = 0
        trials = 300
        for _ in range(trials):
            breaches = rng.random(size=n) < p
            losses = np.where(breaches, var_val * 1.5, 0.0)
            idx = pd.date_range("2020-01-01", periods=n, freq="B")
            result = christoffersen_test(
                pd.Series(var_val, index=idx),
                pd.Series(-losses, index=idx),
                confidence_level=1.0 - p,
                significance=0.05,
            )
            if result.reject_independence:
                rejects += 1

        rejection_rate = rejects / trials
        assert 0.01 < rejection_rate < 0.12

    def test_result_dataclass_shape(self):
        var, ret = _make_series(
            [0.02, 0.02, 0.02], [-0.01, -0.03, 0.01]
        )
        result = christoffersen_test(var, ret, confidence_level=0.95)
        assert isinstance(result, ChristoffersenResult)
        assert result.observations == 2

    def test_rejects_invalid_confidence(self):
        var, ret = _make_series([0.02, 0.02], [-0.01, 0.01])
        with pytest.raises(ValueError):
            christoffersen_test(var, ret, confidence_level=1.0)
        with pytest.raises(ValueError):
            christoffersen_test(var, ret, confidence_level=0.0)

    def test_rejects_single_observation(self):
        var, ret = _make_series([0.02], [-0.01])
        with pytest.raises(ValueError):
            christoffersen_test(var, ret)

    def test_empty_alignment_raises(self):
        idx1 = pd.date_range("2020-01-01", periods=3, freq="B")
        idx2 = pd.date_range("2021-01-01", periods=3, freq="B")
        with pytest.raises(ValueError):
            christoffersen_test(
                pd.Series([0.02, 0.02, 0.02], index=idx1),
                pd.Series([-0.01, -0.01, -0.01], index=idx2),
            )
