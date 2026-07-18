"""Tests for yfinance-based market data ingestion.

Most tests here are pure (no network): they construct synthetic
DataFrames matching yfinance's actual return shape (verified by hand
against a live call while building this module) and check the
normalization/returns/gap-detection logic in isolation. A handful of
tests marked `network` make real calls against Yahoo Finance and are
excluded by default (see pyproject.toml); run them explicitly with
`pytest -m network`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_service.data.market_data import (
    GapReport,
    MarketData,
    _normalize_prices,
    detect_gaps,
    fetch_prices,
    load_returns,
    prices_to_returns,
)
from risk_service.models.parametric import parametric_var


def _multiindex_raw(
    tickers: list[str], dates: pd.DatetimeIndex, close: dict[str, list[float]]
) -> pd.DataFrame:
    """Build a synthetic frame matching yfinance's MultiIndex download shape."""
    fields = ["Close", "High", "Low", "Open", "Volume"]
    data = {}
    for field in fields:
        for t in tickers:
            data[(field, t)] = close[t] if field == "Close" else [1.0] * len(dates)
    raw = pd.DataFrame(data, index=dates)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
    return raw


class TestNormalizePrices:
    def test_extracts_close_from_multiindex(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        raw = _multiindex_raw(
            ["AAPL", "MSFT"],
            dates,
            {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 199.0, 201.0]},
        )
        prices = _normalize_prices(raw, ["AAPL", "MSFT"])
        assert list(prices.columns) == ["AAPL", "MSFT"]
        assert prices["AAPL"].tolist() == [100.0, 101.0, 102.0]
        assert prices["MSFT"].tolist() == [200.0, 199.0, 201.0]

    def test_reorders_columns_to_match_requested_tickers(self):
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        raw = _multiindex_raw(
            ["AAPL", "MSFT"], dates, {"AAPL": [1.0, 2.0], "MSFT": [3.0, 4.0]}
        )
        # Request MSFT first, even though the raw frame has AAPL first.
        prices = _normalize_prices(raw, ["MSFT", "AAPL"])
        assert list(prices.columns) == ["MSFT", "AAPL"]

    def test_handles_flat_columns_for_single_ticker(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        raw = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0],
                "High": [1.0, 1.0, 1.0],
                "Low": [1.0, 1.0, 1.0],
                "Open": [1.0, 1.0, 1.0],
                "Volume": [1, 1, 1],
            },
            index=dates,
        )
        prices = _normalize_prices(raw, ["AAPL"])
        assert list(prices.columns) == ["AAPL"]
        assert prices["AAPL"].tolist() == [100.0, 101.0, 102.0]

    def test_raises_when_close_field_missing_multiindex(self):
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        raw = pd.DataFrame(
            {("Open", "AAPL"): [1.0, 2.0]},
            index=dates,
        )
        raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
        with pytest.raises(ValueError, match="Close"):
            _normalize_prices(raw, ["AAPL"])

    def test_raises_when_ticker_entirely_nan(self):
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        raw = _multiindex_raw(
            ["AAPL", "GHOST"],
            dates,
            {"AAPL": [1.0, 2.0], "GHOST": [np.nan, np.nan]},
        )
        with pytest.raises(ValueError, match="GHOST"):
            _normalize_prices(raw, ["AAPL", "GHOST"])


class TestPricesToReturns:
    def test_simple_returns_match_manual_calculation(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0, 108.9]}, index=dates)
        returns = prices_to_returns(prices, method="simple")
        expected = [0.10, -0.10, 0.10]
        assert returns["AAPL"].tolist() == pytest.approx(expected, rel=1e-6)
        assert len(returns) == len(prices) - 1

    def test_log_returns_match_manual_calculation(self):
        dates = pd.date_range("2024-01-02", periods=3, freq="B")
        prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0]}, index=dates)
        returns = prices_to_returns(prices, method="log")
        expected = [np.log(110.0 / 100.0), np.log(99.0 / 110.0)]
        assert returns["AAPL"].tolist() == pytest.approx(expected, rel=1e-9)

    def test_rejects_invalid_method(self):
        dates = pd.date_range("2024-01-02", periods=2, freq="B")
        prices = pd.DataFrame({"AAPL": [100.0, 101.0]}, index=dates)
        with pytest.raises(ValueError):
            prices_to_returns(prices, method="nonsense")


class TestDetectGaps:
    def test_no_gaps_for_clean_business_day_range(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="B")
        prices = pd.DataFrame({"AAPL": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        report = detect_gaps(prices)
        assert report.missing_business_days == []
        assert report.per_asset_gaps == {}
        assert report.has_gaps is False

    def test_detects_missing_business_day(self):
        full = pd.date_range("2024-01-02", periods=5, freq="B")
        # Drop the middle business day, as if data were missing for it.
        sparse_index = full.delete(2)
        prices = pd.DataFrame({"AAPL": range(len(sparse_index))}, index=sparse_index)
        report = detect_gaps(prices)
        assert report.missing_business_days == [full[2]]
        assert report.has_gaps is True

    def test_detects_per_asset_nan_gap(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        prices = pd.DataFrame(
            {
                "AAPL": [1.0, 2.0, 3.0, 4.0],
                "NEWCO": [np.nan, np.nan, 3.0, 4.0],  # e.g. late IPO
            },
            index=dates,
        )
        report = detect_gaps(prices)
        assert report.missing_business_days == []
        assert report.per_asset_gaps == {"NEWCO": [dates[0], dates[1]]}
        assert report.has_gaps is True

    def test_rejects_empty_prices(self):
        with pytest.raises(ValueError):
            detect_gaps(pd.DataFrame())


@pytest.mark.network
class TestFetchPricesLive:
    def test_fetches_known_ticker(self):
        prices = fetch_prices("AAPL", start="2024-01-02", end="2024-01-12")
        assert list(prices.columns) == ["AAPL"]
        assert len(prices) > 0
        assert (prices["AAPL"] > 0).all()

    def test_fetches_multiple_tickers_in_requested_order(self):
        prices = fetch_prices(
            ["MSFT", "AAPL"], start="2024-01-02", end="2024-01-12"
        )
        assert list(prices.columns) == ["MSFT", "AAPL"]

    def test_raises_for_invalid_ticker(self):
        with pytest.raises(ValueError):
            fetch_prices(
                "THISISNOTAREALTICKERXYZ", start="2024-01-02", end="2024-01-12"
            )


class TestFetchPricesValidation:
    def test_rejects_empty_tickers(self):
        # No network call is made: validated before yf.download.
        with pytest.raises(ValueError):
            fetch_prices([], start="2024-01-02", end="2024-01-12")


@pytest.mark.network
class TestLoadReturnsLive:
    def test_returns_populated_market_data(self):
        data = load_returns(
            ["AAPL", "MSFT"], start="2023-01-03", end="2023-06-01"
        )
        assert isinstance(data, MarketData)
        assert isinstance(data.gaps, GapReport)
        assert list(data.returns.columns) == ["AAPL", "MSFT"]
        assert len(data.returns) == len(data.prices) - 1

    def test_integrates_with_parametric_var(self):
        # Closes the loop: real market data feeding directly into a
        # model already validated against synthetic data elsewhere.
        data = load_returns(
            ["AAPL", "MSFT"], start="2023-01-03", end="2023-06-01"
        )
        var = parametric_var(
            data.returns, weights=[0.5, 0.5], confidence_level=0.99
        )
        assert var > 0.0
