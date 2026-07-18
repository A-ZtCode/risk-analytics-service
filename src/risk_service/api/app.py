"""FastAPI service layer over the VaR/CVaR models and backtests.

Endpoints:
    POST /portfolio/var        One-shot VaR/CVaR for a chosen model.
    POST /portfolio/backtest   Rolling VaR backtest with Kupiec and
                                Christoffersen results.
    POST /portfolio/breaches   Per-date breach detail from the same
                                rolling backtest, for drilldown.

All three take the same core payload shape (dates, per-asset return
series, weights) plus model-specific parameters -- see schemas.py.
Run locally with `uvicorn risk_service.api.app:app --reload`; docs at
`/docs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from risk_service.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    BreachesRequest,
    BreachesResponse,
    BreachRecord,
    ChristoffersenSummary,
    KupiecSummary,
    ModelName,
    PortfolioReturns,
    TrafficLightSummary,
    VaRRequest,
    VaRResponse,
)
from risk_service.models.historical import (
    historical_cvar,
    historical_var,
    rolling_historical_var,
)
from risk_service.models.monte_carlo import (
    monte_carlo_cvar,
    monte_carlo_var,
    rolling_monte_carlo_var,
)
from risk_service.models.parametric import (
    parametric_cvar,
    parametric_var,
    rolling_parametric_var,
)
from risk_service.validation.christoffersen import christoffersen_test
from risk_service.validation.kupiec import kupiec_pof_test
from risk_service.validation.traffic_light import traffic_light_zone_from_kupiec

app = FastAPI(
    title="Risk Analytics Service",
    description=(
        "VaR/CVaR computation and backtesting over historical, "
        "parametric, and Monte Carlo models."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _asset_returns_and_weights(
    payload: PortfolioReturns,
) -> tuple[pd.DataFrame, np.ndarray]:
    index = pd.DatetimeIndex(payload.dates)
    asset_returns = pd.DataFrame(payload.returns, index=index)
    weights = np.array([payload.weights[c] for c in asset_returns.columns])
    return asset_returns, weights


def _portfolio_series(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return pd.Series(asset_returns.to_numpy() @ weights, index=asset_returns.index)


def _one_shot_var_cvar(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    model: ModelName,
    confidence_level: float,
    use_shrinkage: bool,
    dof: float | None,
    n_simulations: int,
    seed: int | None,
) -> tuple[float, float]:
    if model == "historical":
        series = _portfolio_series(asset_returns, weights)
        return (
            historical_var(series, confidence_level=confidence_level),
            historical_cvar(series, confidence_level=confidence_level),
        )
    if model == "parametric":
        return (
            parametric_var(
                asset_returns,
                weights,
                confidence_level=confidence_level,
                use_shrinkage=use_shrinkage,
            ),
            parametric_cvar(
                asset_returns,
                weights,
                confidence_level=confidence_level,
                use_shrinkage=use_shrinkage,
            ),
        )
    return (
        monte_carlo_var(
            asset_returns,
            weights,
            confidence_level=confidence_level,
            dof=dof,
            use_shrinkage=use_shrinkage,
            n_simulations=n_simulations,
            seed=seed,
        ),
        monte_carlo_cvar(
            asset_returns,
            weights,
            confidence_level=confidence_level,
            dof=dof,
            use_shrinkage=use_shrinkage,
            n_simulations=n_simulations,
            seed=seed,
        ),
    )


def _rolling_var(
    asset_returns: pd.DataFrame,
    weights: np.ndarray,
    model: ModelName,
    window: int,
    confidence_level: float,
    use_shrinkage: bool,
    dof: float | None,
    n_simulations: int,
    seed: int,
) -> pd.Series:
    if model == "historical":
        series = _portfolio_series(asset_returns, weights)
        return rolling_historical_var(
            series, window=window, confidence_level=confidence_level
        )
    if model == "parametric":
        return rolling_parametric_var(
            asset_returns,
            weights,
            window=window,
            confidence_level=confidence_level,
            use_shrinkage=use_shrinkage,
        )
    return rolling_monte_carlo_var(
        asset_returns,
        weights,
        window=window,
        confidence_level=confidence_level,
        dof=dof,
        use_shrinkage=use_shrinkage,
        n_simulations=n_simulations,
        seed=seed,
    )


@app.post("/portfolio/var", response_model=VaRResponse)
def compute_var(request: VaRRequest) -> VaRResponse:
    asset_returns, weights = _asset_returns_and_weights(request)
    try:
        var, cvar = _one_shot_var_cvar(
            asset_returns,
            weights,
            request.model,
            request.confidence_level,
            request.use_shrinkage,
            request.dof,
            request.n_simulations,
            request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return VaRResponse(
        model=request.model,
        confidence_level=request.confidence_level,
        var=var,
        cvar=cvar,
    )


@app.post("/portfolio/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> BacktestResponse:
    asset_returns, weights = _asset_returns_and_weights(request)
    portfolio_returns = _portfolio_series(asset_returns, weights)
    try:
        var_series = _rolling_var(
            asset_returns,
            weights,
            request.model,
            request.window,
            request.confidence_level,
            request.use_shrinkage,
            request.dof,
            request.n_simulations,
            request.seed,
        )
        kupiec = kupiec_pof_test(
            var_series,
            portfolio_returns,
            confidence_level=request.confidence_level,
            significance=request.significance,
        )
        christoffersen = christoffersen_test(
            var_series,
            portfolio_returns,
            confidence_level=request.confidence_level,
            significance=request.significance,
        )
        traffic_light = traffic_light_zone_from_kupiec(kupiec)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return BacktestResponse(
        model=request.model,
        confidence_level=request.confidence_level,
        window=request.window,
        kupiec=KupiecSummary.model_validate(kupiec),
        christoffersen=ChristoffersenSummary.model_validate(christoffersen),
        traffic_light=TrafficLightSummary.model_validate(traffic_light),
    )


@app.post("/portfolio/breaches", response_model=BreachesResponse)
def breaches(request: BreachesRequest) -> BreachesResponse:
    asset_returns, weights = _asset_returns_and_weights(request)
    portfolio_returns = _portfolio_series(asset_returns, weights)
    try:
        var_series = _rolling_var(
            asset_returns,
            weights,
            request.model,
            request.window,
            request.confidence_level,
            request.use_shrinkage,
            request.dof,
            request.n_simulations,
            request.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    aligned = pd.concat(
        [var_series.rename("var"), portfolio_returns.rename("ret")],
        axis=1,
        join="inner",
    ).dropna()

    records = [
        BreachRecord(
            date=row.Index.date(),
            var_forecast=float(row.var),
            realised_return=float(row.ret),
            breach=bool(-row.ret > row.var),
        )
        for row in aligned.itertuples()
    ]

    return BreachesResponse(
        model=request.model,
        confidence_level=request.confidence_level,
        window=request.window,
        breach_count=sum(r.breach for r in records),
        records=records,
    )
