"""Real market data ingestion via yfinance.

Wraps `yfinance.download` to produce a return matrix in exactly the
shape the rest of this codebase expects (dates index, one column per
asset -- see `models/parametric.py`, `models/monte_carlo.py`, and
`api/schemas.py::PortfolioReturns`), so `load_returns(...).returns`
can be passed directly to `parametric_var`, `monte_carlo_var`, etc.

Two concerns beyond a bare `yfinance.download` call:

- **Corporate-action handling**: prices are fetched with
  `auto_adjust=True`, so stock splits and dividend payments are folded
  into the price series rather than appearing as a fake single-day
  return. See the README for why this matters and what it does not
  cover (special/return-of-capital dividends, spin-offs).
- **Gap detection**: `detect_gaps` flags two distinct kinds of holes --
  business days entirely missing from the fetched range (which may be
  market holidays or genuine data gaps; this module does not
  distinguish the two, see README), and per-asset NaNs on days other
  assets in the same request *do* have data for (a real, asset-specific
  gap: e.g. a late IPO, a trading halt, a delisting).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class GapReport:
    """Gaps found in a fetched price matrix.

    Attributes
    ----------
    missing_business_days : list[pd.Timestamp]
        Business days between the first and last fetched date that are
        entirely absent from the price index. May be market holidays
        or genuine data gaps -- this module does not distinguish them
        (see README known limitations).
    per_asset_gaps : dict[str, list[pd.Timestamp]]
        For each asset, dates within the fetched index where that
        asset's price is NaN. Indicates an asset-specific gap (e.g.
        late IPO, trading halt, delisting) rather than a market-wide
        holiday, since other assets in the same request have data on
        those dates.
    """

    missing_business_days: list[pd.Timestamp]
    per_asset_gaps: dict[str, list[pd.Timestamp]]

    @property
    def has_gaps(self) -> bool:
        return bool(self.missing_business_days) or any(
            self.per_asset_gaps.values()
        )


@dataclass(frozen=True)
class MarketData:
    """Fetched prices, derived returns, and their gap report.

    Attributes
    ----------
    prices : pd.DataFrame
        Split/dividend-adjusted close prices, one column per asset.
    returns : pd.DataFrame
        Returns derived from `prices` (see `prices_to_returns`), ready
        to pass to `parametric_var`, `monte_carlo_var`, or the API's
        `PortfolioReturns` payload.
    gaps : GapReport
        Gap report computed from `prices` before returns were derived.
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    gaps: GapReport


def fetch_prices(
    tickers: str | list[str],
    start: str | date_type,
    end: str | date_type,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download split/dividend-adjusted close prices for `tickers`.

    Parameters
    ----------
    tickers : str | list[str]
        A single ticker or list of tickers.
    start, end : str | date
        Date range, forwarded to `yfinance.download`.
    interval : str
        Bar interval, forwarded to `yfinance.download` (e.g. "1d",
        "1wk"). Not validated here; yfinance rejects unsupported
        values itself.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices, dates index, one column per ticker (in
        the order given).

    Raises
    ------
    ValueError
        If `tickers` is empty, yfinance returns no data at all for the
        request, or a specific ticker's column is entirely NaN (e.g.
        an invalid symbol -- yfinance does not raise for this itself).
    """
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
    if not ticker_list:
        raise ValueError("tickers must not be empty")

    raw = yf.download(
        ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker_list} between "
            f"{start} and {end}"
        )

    return _normalize_prices(raw, ticker_list)


def _normalize_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract a clean {ticker: close price} frame from a yfinance download.

    Handles both the MultiIndex column shape yfinance returns for
    multi-ticker (and, in current versions, single-ticker) requests,
    and the flat column shape older yfinance versions return for a
    single ticker string.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("downloaded data has no 'Close' field")
        prices = raw["Close"]
    else:
        if "Close" not in raw.columns:
            raise ValueError("downloaded data has no 'Close' column")
        prices = raw[["Close"]]
        prices.columns = [tickers[0]]

    prices = prices.reindex(columns=tickers)
    missing = [t for t in tickers if prices[t].isna().all()]
    if missing:
        raise ValueError(f"no data returned for: {missing}")

    return prices


def prices_to_returns(
    prices: pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Compute returns from a price matrix.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices, dates index, one column per asset.
    method : "simple" | "log"
        "simple": P_t / P_{t-1} - 1.
        "log": log(P_t) - log(P_{t-1}).

    Returns
    -------
    pd.DataFrame
        Returns, one row shorter than `prices` (the first date has no
        prior price to compare against).

    Raises
    ------
    ValueError
        If `method` is not "simple" or "log".
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError(f"method must be 'simple' or 'log', got {method!r}")

    return returns.iloc[1:]


def detect_gaps(prices: pd.DataFrame) -> GapReport:
    """Detect missing business days and per-asset NaNs in a price matrix.

    Parameters
    ----------
    prices : pd.DataFrame
        Prices, dates index, one column per asset.

    Returns
    -------
    GapReport

    Raises
    ------
    ValueError
        If `prices` is empty.
    """
    if prices.empty:
        raise ValueError("prices is empty")

    expected = pd.bdate_range(prices.index.min(), prices.index.max())
    missing_business_days = sorted(set(expected) - set(prices.index))

    per_asset_gaps = {
        str(col): list(prices.index[prices[col].isna()])
        for col in prices.columns
        if prices[col].isna().any()
    }

    return GapReport(
        missing_business_days=missing_business_days,
        per_asset_gaps=per_asset_gaps,
    )


def load_returns(
    tickers: str | list[str],
    start: str | date_type,
    end: str | date_type,
    method: Literal["simple", "log"] = "simple",
    interval: str = "1d",
) -> MarketData:
    """Fetch prices, detect gaps, and derive returns in one call.

    Convenience wrapper combining `fetch_prices`, `detect_gaps` (run
    on prices, before any rows are dropped by return computation), and
    `prices_to_returns`.

    Parameters
    ----------
    tickers, start, end, interval
        See `fetch_prices`.
    method
        See `prices_to_returns`.

    Returns
    -------
    MarketData
    """
    prices = fetch_prices(tickers, start, end, interval=interval)
    gaps = detect_gaps(prices)
    returns = prices_to_returns(prices, method=method)
    return MarketData(prices=prices, returns=returns, gaps=gaps)
