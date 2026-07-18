"""Pydantic request/response schemas for the risk service API."""

from __future__ import annotations

from datetime import date as date_type
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ModelName = Literal["historical", "parametric", "monte_carlo"]


class PortfolioReturns(BaseModel):
    """Historical asset returns and portfolio weights.

    `returns` maps each asset name to its return series, aligned
    position-for-position with `dates`. `weights` maps the same asset
    names to portfolio weights (not required to sum to 1, e.g.
    leveraged or net-short books).
    """

    dates: list[date_type]
    returns: dict[str, list[float]]
    weights: dict[str, float]

    @model_validator(mode="after")
    def _check_shapes(self) -> PortfolioReturns:
        if not self.returns:
            raise ValueError("returns must contain at least one asset")
        if set(self.returns) != set(self.weights):
            raise ValueError("returns and weights must have the same asset names")
        for asset, series in self.returns.items():
            if len(series) != len(self.dates):
                raise ValueError(
                    f"returns['{asset}'] has {len(series)} values, "
                    f"expected {len(self.dates)} (len(dates))"
                )
        if any(d2 <= d1 for d1, d2 in zip(self.dates, self.dates[1:], strict=False)):
            raise ValueError("dates must be strictly increasing")
        return self


class VaRRequest(PortfolioReturns):
    model: ModelName = "historical"
    confidence_level: float = Field(default=0.99, gt=0.0, lt=1.0)
    use_shrinkage: bool = True
    dof: float | None = None
    n_simulations: int = Field(default=100_000, ge=1)
    seed: int | None = None


class VaRResponse(BaseModel):
    model: ModelName
    confidence_level: float
    var: float
    cvar: float


class BacktestRequest(PortfolioReturns):
    model: ModelName = "historical"
    confidence_level: float = Field(default=0.99, gt=0.0, lt=1.0)
    window: int = Field(default=250, ge=2)
    significance: float = Field(default=0.05, gt=0.0, lt=1.0)
    use_shrinkage: bool = True
    dof: float | None = None
    n_simulations: int = Field(default=20_000, ge=1)
    seed: int = 0


class KupiecSummary(BaseModel):
    model_config = {"from_attributes": True}

    breaches: int
    observations: int
    breach_rate: float
    expected_rate: float
    lr_statistic: float
    p_value: float
    reject_null: bool


class ChristoffersenSummary(BaseModel):
    model_config = {"from_attributes": True}

    breaches: int
    observations: int
    transition_counts: tuple[int, int, int, int]
    pi01: float
    pi11: float
    lr_independence: float
    p_value_independence: float
    lr_conditional_coverage: float
    p_value_conditional_coverage: float
    reject_independence: bool
    reject_conditional_coverage: bool


class TrafficLightSummary(BaseModel):
    model_config = {"from_attributes": True}

    breaches: int
    observations: int
    confidence_level: float
    cumulative_probability: float
    zone: Literal["green", "yellow", "red"]
    multiplier: float | None


class BacktestResponse(BaseModel):
    model: ModelName
    confidence_level: float
    window: int
    kupiec: KupiecSummary
    christoffersen: ChristoffersenSummary
    traffic_light: TrafficLightSummary


class BreachesRequest(PortfolioReturns):
    model: ModelName = "historical"
    confidence_level: float = Field(default=0.99, gt=0.0, lt=1.0)
    window: int = Field(default=250, ge=2)
    use_shrinkage: bool = True
    dof: float | None = None
    n_simulations: int = Field(default=20_000, ge=1)
    seed: int = 0


class BreachRecord(BaseModel):
    date: date_type
    var_forecast: float
    realised_return: float
    breach: bool


class BreachesResponse(BaseModel):
    model: ModelName
    confidence_level: float
    window: int
    breach_count: int
    records: list[BreachRecord]
