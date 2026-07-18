"""Tests for Basel traffic-light zone classification.

Includes an analytical-benchmark-style check against the published
BCBS cumulative-probability table for a 250-day, 99% VaR backtest --
the same "code that runs vs. code that is correct" standard used
throughout this project -- plus the full BCBS capital multiplier
table and generalization to non-standard (observations, confidence)
pairs.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from risk_service.validation.kupiec import kupiec_pof_test
from risk_service.validation.traffic_light import (
    TrafficLightResult,
    basel_multiplier,
    traffic_light_zone,
    traffic_light_zone_from_kupiec,
)


class TestTrafficLightZone:
    @pytest.mark.parametrize(
        ("breaches", "expected_zone"),
        [
            (0, "green"),
            (1, "green"),
            (2, "green"),
            (3, "green"),
            (4, "green"),
            (5, "yellow"),
            (6, "yellow"),
            (7, "yellow"),
            (8, "yellow"),
            (9, "yellow"),
            (10, "red"),
            (11, "red"),
            (25, "red"),
        ],
    )
    def test_matches_bcbs_zone_boundaries(self, breaches, expected_zone):
        # Standard regulatory setup: 250 observations, 99% VaR.
        result = traffic_light_zone(
            breaches=breaches, observations=250, confidence_level=0.99
        )
        assert result.zone == expected_zone

    def test_cumulative_probability_matches_scipy_binomial(self):
        # Not a magic-number check: confirms the implementation is
        # actually computing the binomial CDF, not a hardcoded table.
        for breaches in range(0, 15):
            result = traffic_light_zone(
                breaches=breaches, observations=250, confidence_level=0.99
            )
            expected = float(stats.binom.cdf(breaches, 250, 0.01))
            assert result.cumulative_probability == pytest.approx(expected)

    def test_matches_published_bcbs_probabilities(self):
        for breaches, expected_prob in [
            (0, 0.08106),
            (4, 0.89219),
            (5, 0.95882),
            (9, 0.99975),
            (10, 0.99995),
        ]:
            result = traffic_light_zone(
                breaches=breaches, observations=250, confidence_level=0.99
            )
            assert result.cumulative_probability == pytest.approx(
                expected_prob, abs=5e-4
            )

    def test_result_dataclass_shape(self):
        result = traffic_light_zone(breaches=2, observations=250)
        assert isinstance(result, TrafficLightResult)
        assert result.confidence_level == 0.99

    def test_multiplier_populated_for_standard_setup(self):
        result = traffic_light_zone(
            breaches=6, observations=250, confidence_level=0.99
        )
        assert result.multiplier == pytest.approx(3.50)

    def test_multiplier_none_for_nonstandard_window(self):
        result = traffic_light_zone(
            breaches=2, observations=500, confidence_level=0.99
        )
        assert result.multiplier is None

    def test_multiplier_none_for_nonstandard_confidence(self):
        result = traffic_light_zone(
            breaches=2, observations=250, confidence_level=0.95
        )
        assert result.multiplier is None

    def test_generalizes_to_other_window_and_confidence(self):
        # Zone classification isn't hardcoded to 250/0.99: a 95% VaR
        # over 100 observations should classify using the same
        # cumulative-probability logic against Binomial(100, 0.05).
        for breaches in (2, 8, 15):
            result = traffic_light_zone(
                breaches=breaches, observations=100, confidence_level=0.95
            )
            cum_prob = float(stats.binom.cdf(breaches, 100, 0.05))
            if cum_prob < 0.95:
                expected_zone = "green"
            elif cum_prob < 0.9999:
                expected_zone = "yellow"
            else:
                expected_zone = "red"
            assert result.zone == expected_zone

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValueError):
            traffic_light_zone(breaches=2, observations=250, confidence_level=1.0)
        with pytest.raises(ValueError):
            traffic_light_zone(breaches=2, observations=250, confidence_level=0.0)

    def test_rejects_nonpositive_observations(self):
        with pytest.raises(ValueError):
            traffic_light_zone(breaches=0, observations=0)

    def test_rejects_breaches_out_of_range(self):
        with pytest.raises(ValueError):
            traffic_light_zone(breaches=-1, observations=250)
        with pytest.raises(ValueError):
            traffic_light_zone(breaches=251, observations=250)


class TestBaselMultiplier:
    @pytest.mark.parametrize(
        ("breaches", "expected"),
        [
            (0, 3.00),
            (1, 3.00),
            (4, 3.00),
            (5, 3.40),
            (6, 3.50),
            (7, 3.65),
            (8, 3.75),
            (9, 3.85),
            (10, 4.00),
            (20, 4.00),
        ],
    )
    def test_matches_bcbs_multiplier_table(self, breaches, expected):
        assert basel_multiplier(breaches, observations=250) == pytest.approx(
            expected
        )

    def test_rejects_nonstandard_window(self):
        with pytest.raises(ValueError):
            basel_multiplier(5, observations=100)

    def test_rejects_negative_breaches(self):
        with pytest.raises(ValueError):
            basel_multiplier(-1, observations=250)


class TestTrafficLightFromKupiec:
    def test_extracts_breaches_observations_confidence(self):
        idx = pd.date_range("2020-01-01", periods=250, freq="B")
        var = pd.Series(0.02, index=idx)
        returns = pd.Series(0.0, index=idx)
        returns.iloc[:6] = -0.05  # 6 breaches
        kupiec = kupiec_pof_test(var, returns, confidence_level=0.99)

        result = traffic_light_zone_from_kupiec(kupiec)
        assert result.breaches == 6
        assert result.observations == 250
        assert result.confidence_level == pytest.approx(0.99)
        assert result.zone == "yellow"
        assert result.multiplier == pytest.approx(3.50)
