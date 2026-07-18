"""Tests for the FastAPI service layer.

These test request/response wiring and validation, not VaR/CVaR
correctness itself -- that's covered by the analytical-benchmark tests
in test_historical.py, test_parametric.py, and test_monte_carlo.py.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from risk_service.api.app import app

client = TestClient(app)


def _synthetic_payload(
    n: int = 400,
    seed: int = 1,
    n_assets: int = 1,
    model: str = "historical",
    **overrides: object,
) -> dict:
    rng = np.random.default_rng(seed=seed)
    start = date(2020, 1, 1)
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n)]
    assets = [f"asset_{i}" for i in range(n_assets)]
    returns = {
        a: (rng.standard_t(df=5, size=n) * 0.007).tolist() for a in assets
    }
    weights = {a: 1.0 / n_assets for a in assets}
    payload: dict = {
        "dates": dates,
        "returns": returns,
        "weights": weights,
        "model": model,
    }
    payload.update(overrides)
    return payload


class TestHealth:
    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestPortfolioVaR:
    @pytest.mark.parametrize("model", ["historical", "parametric", "monte_carlo"])
    def test_returns_var_and_cvar(self, model):
        payload = _synthetic_payload(n=300, model=model)
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["model"] == model
        assert data["var"] >= 0.0
        assert data["cvar"] >= data["var"]

    def test_default_model_is_historical(self):
        payload = _synthetic_payload(n=300)
        del payload["model"]
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 200
        assert resp.json()["model"] == "historical"

    def test_higher_confidence_higher_var(self):
        payload_95 = _synthetic_payload(n=300, confidence_level=0.95)
        payload_99 = _synthetic_payload(n=300, confidence_level=0.99)
        v95 = client.post("/portfolio/var", json=payload_95).json()["var"]
        v99 = client.post("/portfolio/var", json=payload_99).json()["var"]
        assert v99 > v95

    def test_multi_asset_portfolio(self):
        payload = _synthetic_payload(n=300, n_assets=3, model="parametric")
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.json()["var"] > 0.0

    def test_rejects_invalid_confidence_level(self):
        payload = _synthetic_payload(n=50, confidence_level=1.5)
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 422

    def test_rejects_mismatched_weights(self):
        payload = _synthetic_payload(n=50)
        payload["weights"] = {"not_an_asset": 1.0}
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 422

    def test_rejects_wrong_length_returns(self):
        payload = _synthetic_payload(n=50)
        payload["returns"]["asset_0"] = payload["returns"]["asset_0"][:-1]
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 422

    def test_rejects_non_increasing_dates(self):
        payload = _synthetic_payload(n=50)
        payload["dates"][1] = payload["dates"][0]
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 422

    def test_rejects_empty_returns(self):
        payload = {"dates": [], "returns": {}, "weights": {}}
        resp = client.post("/portfolio/var", json=payload)
        assert resp.status_code == 422


class TestPortfolioBacktest:
    def test_returns_kupiec_and_christoffersen(self):
        payload = _synthetic_payload(n=400, window=50)
        resp = client.post("/portfolio/backtest", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["window"] == 50
        assert data["kupiec"]["observations"] == 400 - 50
        assert 0 <= data["kupiec"]["breaches"] <= data["kupiec"]["observations"]
        assert 0.0 <= data["kupiec"]["p_value"] <= 1.0
        assert data["christoffersen"]["observations"] == 400 - 50 - 1
        assert 0.0 <= data["christoffersen"]["p_value_conditional_coverage"] <= 1.0

    def test_traffic_light_matches_kupiec_breach_count(self):
        payload = _synthetic_payload(n=400, window=50)
        resp = client.post("/portfolio/backtest", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["traffic_light"]["breaches"] == data["kupiec"]["breaches"]
        assert data["traffic_light"]["observations"] == data["kupiec"]["observations"]
        assert data["traffic_light"]["zone"] in ("green", "yellow", "red")
        # window=50 != the standard 250-observation BCBS setup, so the
        # fixed capital multiplier table does not apply here.
        assert data["traffic_light"]["multiplier"] is None

    def test_traffic_light_multiplier_populated_for_standard_setup(self):
        # The BCBS multiplier table requires the *tested* backtest
        # period (n - window) to be exactly 250 observations -- not
        # the estimation window itself. n=500, window=250 -> 250
        # tested observations.
        payload = _synthetic_payload(n=500, window=250, confidence_level=0.99)
        resp = client.post("/portfolio/backtest", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["kupiec"]["observations"] == 250
        assert data["traffic_light"]["multiplier"] is not None
        assert 3.00 <= data["traffic_light"]["multiplier"] <= 4.00

    def test_window_too_large_for_data_is_422(self):
        payload = _synthetic_payload(n=20, window=50)
        resp = client.post("/portfolio/backtest", json=payload)
        assert resp.status_code == 422


class TestPortfolioBreaches:
    def test_record_count_matches_window(self):
        payload = _synthetic_payload(n=400, window=50)
        resp = client.post("/portfolio/breaches", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["records"]) == 400 - 50
        assert data["breach_count"] == sum(r["breach"] for r in data["records"])

    def test_breach_flag_matches_loss_exceeds_var(self):
        payload = _synthetic_payload(n=400, window=50, confidence_level=0.90)
        resp = client.post("/portfolio/breaches", json=payload)
        assert resp.status_code == 200
        for record in resp.json()["records"]:
            expected = -record["realised_return"] > record["var_forecast"]
            assert record["breach"] == expected

    def test_records_are_chronological_and_dated(self):
        payload = _synthetic_payload(n=400, window=50)
        resp = client.post("/portfolio/breaches", json=payload)
        dates = [r["date"] for r in resp.json()["records"]]
        assert dates == sorted(dates)
        assert dates[0] == (date(2020, 1, 1) + timedelta(days=50)).isoformat()
