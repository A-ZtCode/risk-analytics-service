"""End-to-end demo: rolling VaR backtests with Kupiec, Christoffersen,
and Basel traffic-light zoning.

Uses synthetic returns by default, so the demo runs without network
access. Pass --tickers to fetch real data via yfinance instead:

    PYTHONPATH=src python scripts/demo_backtest.py \\
        --tickers AAPL MSFT --start 2020-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

from risk_service.models.historical import (
    historical_cvar,
    historical_var,
    rolling_historical_var,
)
from risk_service.models.monte_carlo import (
    fit_student_t,
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


def generate_synthetic_returns(
    n: int = 1500,
    seed: int = 20260702,
) -> pd.Series:
    """Fat-tailed synthetic daily returns (Student-t, df=5)."""
    rng = np.random.default_rng(seed=seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    # Scale to ~1% daily vol.
    raw = rng.standard_t(df=5, size=n)
    returns = pd.Series(raw * 0.007, index=idx, name="returns")
    return returns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Fetch real data for these tickers via yfinance instead of "
        "using synthetic returns (requires network access).",
    )
    parser.add_argument("--start", default="2018-01-02")
    parser.add_argument("--end", default=None, help="Defaults to today.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.tickers:
        # Real data via yfinance: fetch, report gaps, equal-weight.
        from risk_service.data.market_data import load_returns

        market_data = load_returns(
            args.tickers, start=args.start, end=args.end or date.today().isoformat()
        )
        asset_returns = market_data.returns
        weights = np.full(len(args.tickers), 1.0 / len(args.tickers))
        returns = pd.Series(
            asset_returns.to_numpy() @ weights, index=asset_returns.index
        )
        print(
            f"Fetched {args.tickers} (equal-weighted): "
            f"{len(returns)} observations, "
            f"{returns.index.min().date()} to {returns.index.max().date()}"
        )
        gaps = market_data.gaps
        if gaps.has_gaps:
            print(
                f"Gap report: {len(gaps.missing_business_days)} missing "
                f"business day(s); per-asset NaNs: "
                f"{ {k: len(v) for k, v in gaps.per_asset_gaps.items()} }"
            )
        t_fit = fit_student_t(asset_returns, weights, use_shrinkage=False)
        print(f"\nFitted Monte Carlo dof: {t_fit.dof:.2f}")
    else:
        returns = generate_synthetic_returns()
        print(f"Return series: {len(returns)} observations, "
              f"{returns.index.min().date()} to {returns.index.max().date()}")

        # Parametric VaR takes a DataFrame of asset returns + weights,
        # since it needs the covariance structure. A single return
        # series is just the one-asset, weight-1.0 special case.
        asset_returns = returns.to_frame(name="portfolio")
        weights = [1.0]

        # Fitted Student-t degrees of freedom, for reference: the
        # generator above uses df=5, so a well-behaved estimator
        # should land nearby.
        t_fit = fit_student_t(asset_returns, weights, use_shrinkage=False)
        print(f"\nFitted Monte Carlo dof (true generator dof=5): {t_fit.dof:.2f}")

    # One-shot VaR/CVaR on full history (for reference only; would be
    # look-through in production; use rolling for real backtests).
    print("\n--- One-shot risk on full sample ---")
    for c in (0.95, 0.99):
        hv = historical_var(returns, confidence_level=c)
        he = historical_cvar(returns, confidence_level=c)
        pv = parametric_var(asset_returns, weights, confidence_level=c)
        pe = parametric_cvar(asset_returns, weights, confidence_level=c)
        mv = monte_carlo_var(asset_returns, weights, confidence_level=c, seed=0)
        me = monte_carlo_cvar(asset_returns, weights, confidence_level=c, seed=0)
        print(
            f"  {int(c*100)}% | historical   VaR={hv:.4%}  CVaR={he:.4%}\n"
            f"       | parametric   VaR={pv:.4%}  CVaR={pe:.4%}\n"
            f"       | monte carlo  VaR={mv:.4%}  CVaR={me:.4%}"
        )

    # Rolling 250-day window VaR (no lookahead), across all three models.
    for label, var_fn in (
        ("historical", lambda c: rolling_historical_var(
            returns, window=250, confidence_level=c
        )),
        ("parametric", lambda c: rolling_parametric_var(
            asset_returns, weights, window=250, confidence_level=c
        )),
        ("monte carlo", lambda c: rolling_monte_carlo_var(
            asset_returns, weights, window=250, confidence_level=c
        )),
    ):
        print(f"\n--- Rolling 250-day {label} VaR backtest ---")
        for c in (0.95, 0.99):
            var_series = var_fn(c)
            kupiec = kupiec_pof_test(
                var_estimates=var_series,
                realised_returns=returns,
                confidence_level=c,
            )
            cc = christoffersen_test(
                var_estimates=var_series,
                realised_returns=returns,
                confidence_level=c,
            )
            tl = traffic_light_zone_from_kupiec(kupiec)
            uc_verdict = "REJECT" if kupiec.reject_null else "pass"
            ind_verdict = "REJECT" if cc.reject_independence else "pass"
            print(
                f"  {int(c*100)}% VaR | "
                f"breaches={kupiec.breaches}/{kupiec.observations} "
                f"({kupiec.breach_rate:.2%}) | expected={kupiec.expected_rate:.2%}\n"
                f"       | Kupiec (coverage)      "
                f"p={kupiec.p_value:.3f}  [{uc_verdict}]\n"
                f"       | Christoffersen (indep) p={cc.p_value_independence:.3f}  "
                f"[{ind_verdict}]  transitions={cc.transition_counts}\n"
                f"       | Basel zone: {tl.zone.upper()}  "
                f"(cumulative prob={tl.cumulative_probability:.4f})"
            )

    # The BCBS capital multiplier table only applies to the literal
    # regulatory setup: exactly 250 *tested* observations (n - window,
    # not window itself) at 99% confidence. The backtest above tests
    # 1250 observations, so its traffic_light.multiplier is None --
    # illustrated here with a dedicated 500-day series (250 for the
    # estimation window, 250 tested) to show the populated case.
    print("\n--- Basel multiplier (standard 250-observation setup) ---")
    standard_returns = generate_synthetic_returns(n=500, seed=1)
    standard_var = rolling_historical_var(
        standard_returns, window=250, confidence_level=0.99
    )
    standard_kupiec = kupiec_pof_test(
        standard_var, standard_returns, confidence_level=0.99
    )
    standard_tl = traffic_light_zone_from_kupiec(standard_kupiec)
    print(
        f"  99% VaR | "
        f"breaches={standard_kupiec.breaches}/{standard_kupiec.observations} "
        f"| zone={standard_tl.zone.upper()} | multiplier={standard_tl.multiplier:.2f}"
    )


if __name__ == "__main__":
    main()
