"""Basel traffic-light backtesting zones.

Kupiec's POF test (`kupiec.py`) gives a p-value: reject or fail to
reject calibration at a chosen significance level. Regulators need
something coarser and more actionable -- a small number of buckets
that map directly to a capital consequence. The Basel Committee's 1996
traffic-light approach does exactly that: given the number of VaR
exceptions (breaches) observed over a 250-day window at 99% confidence,
classify the model as green (no concern), yellow (presumption of a
problem; capital multiplier increases), or red (model rejected; capital
multiplier maxed out).

The zone boundaries are principled, not arbitrary: under H0 (the model
is correctly calibrated), the exception count over `observations` days
is Binomial(observations, 1 - confidence_level). Green covers outcomes
a well-calibrated model produces with high probability (cumulative
probability < 95%); yellow covers outcomes that are increasingly
unlikely under H0 but not yet damning (< 99.99%); red is everything
above that -- results a correctly calibrated model would produce less
than 1 time in 10,000.

The exact capital *multiplier* table (`basel_multiplier`), by contrast,
is a fixed regulatory table BCBS calibrated specifically for a
250-observation window at 99% confidence -- it is not a formula, and
does not generalize to other window sizes or confidence levels the way
the zone classification does.

Reference: Basel Committee on Banking Supervision (1996). "Supervisory
Framework for the Use of Backtesting in Conjunction with the Internal
Models Approach to Market Risk Capital Requirements."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy import stats

from risk_service.validation.kupiec import KupiecResult

Zone = Literal["green", "yellow", "red"]

_GREEN_CUTOFF = 0.95
_RED_CUTOFF = 0.9999

_STANDARD_OBSERVATIONS = 250
_STANDARD_CONFIDENCE_LEVEL = 0.99

_BASE_MULTIPLIER = 3.00
_MAX_INCREASE = 1.00
_YELLOW_INCREASE = {
    5: 0.40,
    6: 0.50,
    7: 0.65,
    8: 0.75,
    9: 0.85,
}


@dataclass(frozen=True)
class TrafficLightResult:
    """Result of a Basel traffic-light backtest classification.

    Attributes
    ----------
    breaches : int
        Number of observed VaR breaches.
    observations : int
        Number of (VaR, return) pairs the breach count is drawn from.
    confidence_level : float
        VaR confidence level the breach count was measured against.
    cumulative_probability : float
        P(X <= breaches) under X ~ Binomial(observations, 1 -
        confidence_level), i.e. H0 that the model is correctly
        calibrated. This is what the zone boundaries are built from.
    zone : "green" | "yellow" | "red"
        green: cumulative_probability < 0.95.
        yellow: 0.95 <= cumulative_probability < 0.9999.
        red: cumulative_probability >= 0.9999.
    multiplier : float | None
        BCBS capital multiplier (base 3.00 plus a zone-dependent
        increase, capped at 4.00 in the red zone). Only defined for the
        standard regulatory setup this table was calibrated for
        (observations=250, confidence_level=0.99); None otherwise.
    """

    breaches: int
    observations: int
    confidence_level: float
    cumulative_probability: float
    zone: Zone
    multiplier: float | None


def traffic_light_zone(
    breaches: int,
    observations: int,
    confidence_level: float = 0.99,
) -> TrafficLightResult:
    """Classify a breach count into a Basel traffic-light zone.

    Parameters
    ----------
    breaches : int
        Number of observed VaR breaches.
    observations : int
        Number of (VaR, return) pairs tested. Must be positive.
    confidence_level : float
        VaR confidence level in (0, 1) the breach count was measured
        against.

    Returns
    -------
    TrafficLightResult

    Raises
    ------
    ValueError
        If confidence_level is out of range, observations <= 0, or
        breaches is not in [0, observations].
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
    if observations <= 0:
        raise ValueError(f"observations must be positive, got {observations}")
    if not 0 <= breaches <= observations:
        raise ValueError(
            f"breaches must be in [0, observations], got breaches={breaches}, "
            f"observations={observations}"
        )

    p = 1.0 - confidence_level
    cumulative_probability = float(stats.binom.cdf(breaches, observations, p))

    zone: Zone
    if cumulative_probability < _GREEN_CUTOFF:
        zone = "green"
    elif cumulative_probability < _RED_CUTOFF:
        zone = "yellow"
    else:
        zone = "red"

    multiplier = None
    if (
        observations == _STANDARD_OBSERVATIONS
        and confidence_level == _STANDARD_CONFIDENCE_LEVEL
    ):
        multiplier = basel_multiplier(breaches, observations)

    return TrafficLightResult(
        breaches=breaches,
        observations=observations,
        confidence_level=confidence_level,
        cumulative_probability=cumulative_probability,
        zone=zone,
        multiplier=multiplier,
    )


def traffic_light_zone_from_kupiec(result: KupiecResult) -> TrafficLightResult:
    """Classify a `KupiecResult` into a Basel traffic-light zone.

    Convenience wrapper: reuses the breach/observation counts a Kupiec
    test already computed rather than recounting breaches separately.
    `confidence_level` is recovered as `1 - result.expected_rate`.

    Parameters
    ----------
    result : KupiecResult
        Output of `kupiec.kupiec_pof_test`.

    Returns
    -------
    TrafficLightResult
    """
    confidence_level = 1.0 - result.expected_rate
    return traffic_light_zone(
        breaches=result.breaches,
        observations=result.observations,
        confidence_level=confidence_level,
    )


def basel_multiplier(breaches: int, observations: int = 250) -> float:
    """BCBS capital multiplier for the standard 250-day, 99% VaR setup.

    Base multiplier 3.00, plus a zone-dependent increase: 0.00 in the
    green zone (0-4 exceptions), a table-specified increase in the
    yellow zone (5-9), capped at 1.00 (total 4.00) in the red zone
    (10+). This table is a fixed regulatory choice, not a formula --
    unlike `traffic_light_zone`, it does not generalize to other window
    sizes or confidence levels.

    Parameters
    ----------
    breaches : int
        Number of observed VaR breaches.
    observations : int
        Must be 250, the window length this table is calibrated for.

    Returns
    -------
    float
        Capital multiplier in [3.00, 4.00].

    Raises
    ------
    ValueError
        If observations != 250 or breaches is negative.
    """
    if observations != _STANDARD_OBSERVATIONS:
        raise ValueError(
            "The BCBS capital multiplier table is calibrated specifically "
            f"for a {_STANDARD_OBSERVATIONS}-observation window; got "
            f"observations={observations}. Zone classification "
            "(traffic_light_zone) generalizes to other window sizes; the "
            "multiplier table does not."
        )
    if breaches < 0:
        raise ValueError(f"breaches must be >= 0, got {breaches}")

    if breaches <= 4:
        increase = 0.0
    elif breaches <= 9:
        increase = _YELLOW_INCREASE[breaches]
    else:
        increase = _MAX_INCREASE

    return _BASE_MULTIPLIER + increase
