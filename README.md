# Risk Analytics Service

A production-oriented VaR/CVaR engine with rigorous backtesting of the risk model itself. This project treats the risk model as a testable artefact, not a single number: forecasts are validated statistically against realised outcomes using standard regulatory tests.

## What this is

Most portfolio risk projects compute a VaR figure and stop there. This one implements the piece that risk teams actually own day to day: does the model, when walked forward through history, produce the breach frequency it claims to? A 99% VaR that gets breached 5% of the time is not a 99% VaR, regardless of what the code prints.

Currently implemented:

- **Historical simulation** VaR and CVaR (Expected Shortfall)
- **Parametric (variance-covariance) VaR and CVaR** under a normality assumption, with optional **Ledoit-Wolf shrinkage** on the covariance matrix for multi-asset portfolios
- **Monte Carlo VaR and CVaR with Student-t innovations**, with degrees of freedom fit from the historical sample's excess kurtosis, closing the fat-tail gap parametric VaR leaves open
- **Rolling backtest** with strict no-lookahead guarantee (`shift(1)` before window), for historical, parametric, and Monte Carlo VaR
- **Kupiec Proportion of Failures test** (unconditional coverage) with closed-form LR statistic and chi-squared p-values
- **Christoffersen independence and conditional coverage tests**, closing the gap Kupiec leaves open by detecting breach clustering
- **Basel traffic-light zoning**, classifying a breach count into green/yellow/red via cumulative binomial probability, plus the official BCBS capital multiplier table for the standard 250-observation, 99% VaR setup
- **FastAPI service layer** (`/portfolio/var`, `/portfolio/backtest`, `/portfolio/breaches`) exposing all three VaR models and all three backtests (Kupiec, Christoffersen, traffic light) over HTTP, with request validation and auto-generated docs at `/docs`
- **Real-data ingestion via `yfinance`** (`data/market_data.py`), split/dividend-adjusted, with gap detection distinguishing market-wide missing days from per-asset gaps, feeding straight into the same VaR models and backtests as the synthetic demo
- Full unit tests including analytical benchmarks against known distributions
- CI-ready `pyproject.toml`, typed code, structured docstrings

Everything originally on the roadmap is now implemented; see "Known limitations" below for what each piece explicitly does not do.

## Quick start

```bash
pip install -e ".[dev]"
pytest                      # network tests excluded by default
pytest -m network           # opt-in: exercises real yfinance calls
PYTHONPATH=src python scripts/demo_backtest.py
```

Sample demo output (synthetic Student-t returns, df=5, 1500 observations):

```
Fitted Monte Carlo dof (true generator dof=5): 4.83

--- One-shot risk on full sample ---
  95% | historical   VaR=1.3783%  CVaR=2.0490%
       | parametric   VaR=1.4781%  CVaR=1.8553%
       | monte carlo  VaR=1.3940%  CVaR=2.0232%
  99% | historical   VaR=2.1691%  CVaR=3.4545%
       | parametric   VaR=2.0934%  CVaR=2.3993%
       | monte carlo  VaR=2.3640%  CVaR=3.1418%

--- Rolling 250-day historical VaR backtest ---
  95% VaR | breaches=68/1250 (5.44%) | expected=5.00%
       | Kupiec (coverage)      p=0.481  [pass]
       | Christoffersen (indep) p=0.309  [pass]  transitions=(1115, 66, 66, 2)
       | Basel zone: GREEN  (cumulative prob=0.7842)
  99% VaR | breaches=19/1250 (1.52%) | expected=1.00%
       | Kupiec (coverage)      p=0.086  [pass]
       | Christoffersen (indep) p=0.444  [pass]  transitions=(1211, 19, 19, 0)
       | Basel zone: YELLOW  (cumulative prob=0.9701)

--- Rolling 250-day parametric VaR backtest ---
  95% VaR | breaches=55/1250 (4.40%) | expected=5.00%
       | Kupiec (coverage)      p=0.321  [pass]
       | Christoffersen (indep) p=0.770  [pass]  transitions=(1141, 53, 53, 2)
       | Basel zone: GREEN  (cumulative prob=0.1826)
  99% VaR | breaches=19/1250 (1.52%) | expected=1.00%
       | Kupiec (coverage)      p=0.086  [pass]
       | Christoffersen (indep) p=0.444  [pass]  transitions=(1211, 19, 19, 0)
       | Basel zone: YELLOW  (cumulative prob=0.9701)

--- Rolling 250-day monte carlo VaR backtest ---
  95% VaR | breaches=60/1250 (4.80%) | expected=5.00%
       | Kupiec (coverage)      p=0.744  [pass]
       | Christoffersen (indep) p=0.565  [pass]  transitions=(1131, 58, 58, 2)
       | Basel zone: GREEN  (cumulative prob=0.4047)
  99% VaR | breaches=15/1250 (1.20%) | expected=1.00%
       | Kupiec (coverage)      p=0.491  [pass]
       | Christoffersen (indep) p=0.546  [pass]  transitions=(1219, 15, 15, 0)
       | Basel zone: GREEN  (cumulative prob=0.8070)

--- Basel multiplier (standard 250-observation setup) ---
  99% VaR | breaches=3/250 | zone=GREEN | multiplier=3.00
```

The fitted dof (4.83) nearly recovers the true generator value (5), a useful sanity check that the kurtosis-based estimator works. The 99% historical figure sits close to rejection on Kupiec (p=0.086 against a 5% threshold); Monte Carlo's 99% Kupiec p-value (0.491) is the best-calibrated of the three. All three models pass Christoffersen throughout — expected, since the synthetic generator draws iid Student-t innovations with no built-in volatility clustering for Christoffersen to catch. The *CVaR* column shows the gap Monte Carlo exists to close: parametric 99% CVaR (2.40%) is materially below historical 99% CVaR (3.45%) on this fat-tailed (Student-t, df=5) data, because it assumes normality — VaR (and Kupiec, and Christoffersen) only look at whether/when the threshold was crossed, never by how much, so a model can pass every breach-counting test while still being wrong about the shape of the tail. Monte Carlo's 99% CVaR (3.14%) sits much closer to historical's fat-tail-respecting estimate, because it explicitly models the fat tail via Student-t innovations rather than assuming it away.

The traffic-light row is the sharpest illustration of Kupiec's weakness as a pass/fail gate: at 99%, both historical and parametric land in the **yellow** zone (cumulative probability 0.9701, close to the 0.95 cutoff) despite Kupiec's p=0.086 saying "fail to reject" at the usual 5% significance level. Kupiec's binary reject/fail-to-reject decision and Basel's five-bucket-wide zoning are built from the *same* binomial model but answer different questions — "is this breach count implausible at the 5% level" versus "which of three regulatory-consequence buckets does this breach count fall in" — and they can disagree right at the margin, as they do here. Monte Carlo alone stays green at both confidence levels, consistent with it having the best-calibrated breach count throughout this demo. The dedicated 250-observation section at the bottom shows the case the official capital multiplier table actually applies to (green zone, no capital add-on: multiplier=3.00) — note it needs exactly 250 *tested* observations (`len(returns) - window`), not the 250-day estimation window used throughout the rest of this demo, which is why the multiplier is absent from the three backtests above (they each test 1250 observations).

### Running the API

```bash
uvicorn risk_service.api.app:app --reload
```

Interactive docs (Swagger UI) are then at `http://127.0.0.1:8000/docs`. All three endpoints take the same core payload shape — `dates`, per-asset `returns`, and `weights` — plus a `model` field (`"historical"`, `"parametric"`, or `"monte_carlo"`):

```bash
curl -X POST http://127.0.0.1:8000/portfolio/var \
  -H "Content-Type: application/json" \
  -d '{
    "dates": ["2020-01-01", "2020-01-02", "..."],
    "returns": {"portfolio": [0.004, -0.011, "..."]},
    "weights": {"portfolio": 1.0},
    "model": "monte_carlo",
    "confidence_level": 0.99
  }'
# {"model":"monte_carlo","confidence_level":0.99,"var":0.0229,"cvar":0.0307}
```

`/portfolio/backtest` runs the rolling VaR series for the chosen model through Kupiec, Christoffersen, and Basel traffic-light zoning, returning all three in one call; `/portfolio/breaches` returns the per-date `(var_forecast, realised_return, breach)` records behind those tests, for drilldown into exactly which dates breached.

### Using real market data

Everything above uses synthetic iid Student-t returns by default so the demo and test suite run with no network access. `scripts/demo_backtest.py` also accepts `--tickers` to fetch real data via `yfinance` instead, through `data/market_data.py::load_returns` — the exact same pipeline the API's `PortfolioReturns` payload expects:

```bash
PYTHONPATH=src python scripts/demo_backtest.py \
  --tickers AAPL MSFT GOOGL --start 2019-01-01 --end 2024-01-01
```

Real data surfaces a real breach clustering event that synthetic iid data never will. Equal-weighted AAPL/MSFT/GOOGL, 2019–2024 (spans the March 2020 COVID crash):

```
Fetched ['AAPL', 'MSFT', 'GOOGL'] (equal-weighted): 1257 observations, 2019-01-03 to 2023-12-29
Gap report: 45 missing business day(s); per-asset NaNs: {}

--- Rolling 250-day parametric VaR backtest ---
  99% VaR | breaches=29/1007 (2.88%) | expected=1.00%
       | Kupiec (coverage)      p=0.000  [REJECT]
       | Christoffersen (indep) p=0.008  [REJECT]  transitions=(952, 25, 25, 4)
       | Basel zone: RED  (cumulative prob=1.0000)

--- Rolling 250-day monte carlo VaR backtest ---
  99% VaR | breaches=22/1007 (2.18%) | expected=1.00%
       | Kupiec (coverage)      p=0.001  [REJECT]
       | Christoffersen (indep) p=0.089  [pass]  transitions=(964, 20, 20, 2)
       | Basel zone: YELLOW  (cumulative prob=0.9997)
```

Both models get *rejected* by Kupiec at 99% on real data — the COVID crash produced far more breaches than either model's calibration window anticipated — but parametric also fails Christoffersen (breaches cluster around the crash, as expected: volatility clustering is exactly what a static covariance estimate can't see) and lands in the Basel **red** zone, while Monte Carlo's fatter tails absorb enough of the crash to pass Christoffersen and land only in yellow. This is the demo's synthetic-data story (parametric understates tail risk; Monte Carlo partially closes the gap) playing out on a real historical crisis rather than a manufactured one. These exact numbers will drift as yfinance's underlying data gets revised or extended — this block records one run, not a guarantee.

The `Gap report` line comes from `detect_gaps`: 45 missing business days over this range are calendar effects (US market holidays: New Year's, MLK Day, Presidents' Day, Good Friday, and so on, none of which this module distinguishes from a genuine data gap — see known limitations) and zero per-asset NaNs (all three tickers trade every day the others do, unsurprising for large, liquid, long-listed names).

## Design decisions

**Why historical simulation first.** It makes no distributional assumption, so it is the honest baseline. Any parametric method has to justify itself against the empirical answer.

**Why the `shift(1)` in the rolling window.** VaR forecast for date `t` must be computable strictly from information available before `t`. A test in `tests/test_historical.py::test_no_lookahead` injects a crash on the final date and confirms the VaR forecast for that date is unaffected.

**Why analytical benchmarks in tests.** For X ~ N(0, σ²), 99% VaR has a closed form: `σ · Φ⁻¹(0.99) ≈ 2.326σ`. Expected Shortfall likewise: `σ · φ(Φ⁻¹(α)) / α`. The tests draw large samples from a normal distribution and assert the empirical estimator matches the theoretical value within Monte Carlo error. This is what distinguishes "code that runs" from "code that is correct."

**Why the Kupiec test even though it's weak.** Kupiec tests unconditional coverage (right breach count on average). It does not test whether breaches cluster — a model that produces the correct number of breaches all bunched into one week is clearly bad, and Kupiec will not detect that. `validation/christoffersen.py` closes that gap: it models the breach indicator sequence as a first-order Markov chain and tests whether the breach probability tomorrow depends on whether there was a breach today. `test_christoffersen.py::test_clustered_breaches_rejected_but_kupiec_passes` constructs exactly that scenario — 10 breaches out of 500 (matching the expected rate) but all consecutive — and confirms Kupiec passes it while Christoffersen's independence test rejects it. The tests explain what each test does and does not measure so downstream users are not misled.

**Why Ledoit-Wolf shrinkage for parametric VaR.** Parametric VaR needs a covariance matrix to project asset-level risk onto a portfolio. The raw sample covariance is noisy — and singular when the number of assets approaches or exceeds the number of observations — which is exactly the regime a growing multi-asset book runs into. Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`) shrinks the sample covariance toward a well-conditioned scaled-identity target, trading a small amount of bias for a large reduction in estimation variance. `use_shrinkage=False` is exposed for tests and for direct comparison against the raw sample covariance.

**Why parametric VaR takes a returns *DataFrame* + weights, not a single series.** Historical VaR operates on one already-aggregated portfolio return series because it only needs the empirical distribution of portfolio-level outcomes. Parametric VaR is different: shrinkage and the variance-covariance projection are only meaningful with the per-asset covariance structure, so the function signature reflects that (`portfolio_moments`, `parametric_var`, `parametric_cvar`, `rolling_parametric_var` in `models/parametric.py`). A single-asset portfolio is just the one-column, weight-`[1.0]` special case — see `scripts/demo_backtest.py`.

**Why `ChristoffersenResult.breaches`/`observations` differ slightly from a standalone Kupiec run on the same data.** The independence test only has `N - 1` (t-1, t) transitions to work with (the first observation has no predecessor), so `christoffersen_test`'s internal LR_uc component — used to build the joint `lr_conditional_coverage = LR_uc + LR_ind` — is computed on those `N - 1` transitions, not the full `N` observations `kupiec_pof_test` would use standalone. This keeps LR_uc and LR_ind nested over the *same* sample, which is what makes their sum chi-squared(2) under H0. The difference is one observation and is immaterial for any realistic backtest length; it is called out here so the two breach counts aren't assumed to be a bug if compared side by side.

**Why Monte Carlo reuses `parametric.portfolio_moments` for mean/volatility rather than fitting everything via `scipy.stats.t.fit`.** `scipy.stats.t.fit` would give a one-shot MLE of location, scale, and dof directly from a univariate return series, and is arguably more statistically efficient than the method-of-moments approach used here. But it only works on a single aggregated series, throwing away the whole point of `models/parametric.py`: a multi-asset covariance matrix, Ledoit-Wolf-shrunk, projected through portfolio weights. Reusing `portfolio_moments` keeps the *same* mean/volatility estimate across the parametric and Monte Carlo models, so the one-shot demo comparison above is an apples-to-apples test of "does adding fat tails change the answer", not "do these two models also disagree about volatility for unrelated reasons." Only the tail shape (dof) is fit separately, from the portfolio-level historical series' excess kurtosis (`estimate_dof`).

**Why degrees of freedom are estimated via excess kurtosis (method of moments) rather than full MLE.** It is a closed-form, one-line estimator (`dof = 4 + 6 / excess_kurtosis`) with an obvious failure mode that's easy to reason about and clip (thin-tailed data → large dof → close to normal). Full MLE (`scipy.stats.t.fit`) is more efficient but has no closed form, requires numerical optimization per rolling window, and offers little practical benefit here since `dof` is only used to set tail *shape*, not the portfolio's mean/volatility (those come from `portfolio_moments`, as above). See known limitation 7 for the cost of this choice.

**Why the API takes `dates` + a `dict[asset, list[float]]` + a `dict[asset, weight]` instead of one flat return series.** The three underlying models don't agree on their natural input shape — historical VaR wants one portfolio-level series, parametric and Monte Carlo want a per-asset return matrix — so the API standardizes on the richer multi-asset shape (`api/schemas.py::PortfolioReturns`) and derives the portfolio series internally (`asset_returns @ weights`) wherever a model needs it. This also means the API supports genuine multi-asset portfolios end to end, not just the single-asset special case used in the demo script. A `model_validator` checks `dates` are strictly increasing and that `returns`/`weights` reference exactly the same asset names before any model code runs, so a malformed request fails fast with a 422 and a specific message rather than an obscure error three function calls deep.

**Why `ValueError` from the model layer becomes HTTP 422, not 500.** Every `ValueError` raised by the model/validation functions (e.g. "window too large for the data", "no overlapping non-NaN pairs") reflects a problem with the *request*, not a server fault — the same distinction Pydantic's own validation errors already draw. Routes therefore catch `ValueError` narrowly around the model calls and re-raise as `HTTPException(422, detail=str(exc))`, reusing the exception messages the model layer already writes carefully (see `historical.py`, `kupiec.py`) rather than inventing new API-level error text.

**Why traffic-light *zone* classification generalizes to any (observations, confidence_level) but the *multiplier* table doesn't.** The zone boundaries are a principled statement about a binomial model: green covers breach counts a correctly-calibrated model produces with high probability (cumulative probability < 95%), red covers breach counts it would produce less than 1 time in 10,000. That logic is just `scipy.stats.binom.cdf` and works for any window size or confidence level — `traffic_light_zone` computes it directly rather than hardcoding the classic 250-observation table. The capital *multiplier* (3.00 base, +0.40 to +0.85 in yellow, +1.00 in red) is different: it's a specific number BCBS chose by regulation for exactly the 250-observation, 99% VaR setup, not a formula derived from the binomial model. `basel_multiplier` raises `ValueError` rather than silently extrapolating it to other window sizes, where it has no regulatory meaning.

**Why the multiplier needs `observations == 250`, not `window == 250`.** These are two different numbers in this codebase: `window` is the *estimation* period (how much history each day's VaR forecast is computed from), while `observations` (`kupiec_pof_test`'s output, also what `traffic_light_zone` takes) is the *backtest* period — how many (VaR, realised return) pairs were actually tested, i.e. `len(returns) - window`. The BCBS "250 observations" refers to the latter: one year of daily backtesting, independent of whatever estimation window produced each day's forecast. The demo script's main backtests use `window=250` over 1500 days of data, giving 1250 *tested* observations — not 250 — so their `traffic_light.multiplier` is `None`; the dedicated 500-day example at the bottom of the demo output (`window=250`, 500 total days, so 250 tested observations) is what actually matches the regulatory setup.

**Why prices are fetched with `auto_adjust=True`.** yfinance's `auto_adjust` folds stock splits and (ordinary) dividend payments into the price series, so a 4:1 split or a dividend ex-date doesn't show up as a fake single-day return of -75% or -2%. Without it, `prices_to_returns` would need its own corporate-action logic duplicating what yfinance already does, and every downstream VaR estimate would be corrupted by these events until fixed. This is also why `market_data.py`'s module docstring is explicit about what `auto_adjust` does *not* cover (return-of-capital and special dividends, spin-offs) rather than implying "corporate actions" is a fully solved problem.

**Why `detect_gaps` distinguishes market-wide missing days from per-asset gaps instead of trying to classify holidays.** A real trading-calendar-aware classifier (e.g. `pandas_market_calendars`) would need a specific exchange calendar per ticker and is a meaningfully heavier dependency for what this module needs to guarantee: that a NaN silently reaching `portfolio_moments` doesn't happen unnoticed. Splitting gaps into "missing business day across the whole fetch" (usually a holiday, occasionally a real outage) versus "NaN for one asset while others have data" (almost always asset-specific: late IPO, halt, delisting) gets most of the diagnostic value — knowing *which* assets have real gaps — without the added dependency or the exchange-calendar-per-ticker bookkeeping it would require.

**Why `network`-marked tests are excluded by default (`-m "not network"` in `pyproject.toml`).** This project's own design principle, stated since the very first demo script, is that the default test/demo path never depends on an external service (`generate_synthetic_returns`'s docstring: "so the demo runs without network access"). Yahoo Finance's API is unauthenticated, unversioned, and rate-limits aggressively — exactly the kind of dependency that makes CI flaky for reasons that have nothing to do with this codebase. `tests/test_market_data.py` splits accordingly: pure functions (`_normalize_prices`, `prices_to_returns`, `detect_gaps`) are tested against hand-built DataFrames matching yfinance's real output shape, with no network involved; only `fetch_prices`/`load_returns` themselves are marked `network` and require an explicit `pytest -m network` to run.

## Known limitations

Documented plainly rather than hidden:

1. **Historical VaR is blind to tail events not in the window.** A 250-day window starting in 2019 says nothing about 2008-style events.
2. **VaR is not subadditive.** For two portfolios A and B, it is possible for VaR(A + B) > VaR(A) + VaR(B). This is why CVaR (which is coherent) is preferred by Basel III (FRTB).
3. **Kupiec has low power in small samples.** With n = 250 and p = 0.01, the test struggles to distinguish a well-calibrated model from a mildly miscalibrated one. Longer backtests are needed for reliable inference.
4. **Serial correlation and volatility clustering are ignored** in the current historical implementation. A GARCH filter on returns before applying historical simulation is a standard fix (filtered historical simulation) and is on the roadmap.
5. **Parametric VaR assumes normality**, which understates deep-tail risk on fat-tailed return series (see the demo output above: parametric CVaR sits well below historical CVaR on Student-t data at 99%). Ledoit-Wolf shrinkage improves the covariance *estimate*; it does nothing for the distributional assumption itself. Monte Carlo VaR with Student-t innovations targets this gap directly (see demo output: its CVaR sits much closer to historical's).
6. **Christoffersen's independence test only models first-order dependence** (does yesterday's breach predict today's?). It will miss longer-memory clustering, e.g. a model that reliably breaches every fifth day, or clustering with lags longer than one period. It also inherits Kupiec's low-power problem in small samples, and needs a non-trivial number of transitions into the "breach" state (`n10 + n11 > 0`) to say anything about `pi11` at all — with p = 0.01 and n = 250, that state is visited only ~2.5 times in expectation, so `pi11` is a noisy estimate over short backtests.
7. **Monte Carlo's degrees-of-freedom estimate is a single, global, method-of-moments fit.** Excess kurtosis is a high-variance statistic — noticeably more so than the mean or the variance used for the other moments — so `estimate_dof` can swing considerably between rolling windows on the same underlying process, especially with `window` at the smaller end (250 observations, as used throughout this README). It also fits one `dof` for the whole portfolio, not per asset, which is exact only when every asset shares the same tail shape; a portfolio mixing a fat-tailed asset with a near-normal one gets a single blended dof rather than a distribution reflecting the mixture. Both are standard trade-offs for a first Monte Carlo implementation, not bugs — full MLE fitting and per-asset tail modeling are natural follow-ups if the single-dof approximation proves too coarse in practice.
8. **The API has no authentication, rate limiting, or request-size caps**, and every request recomputes everything from raw returns with no caching or persistence layer (there is no database — each call is stateless in, JSON out). A `POST /portfolio/backtest` with `model=monte_carlo` over a multi-year daily history and the default `n_simulations` runs a fresh simulation per rolling window inside a single synchronous request, which is fine for the demo sizes in this README (a few seconds) but is the first thing to profile before pointing this at a real multi-year, multi-asset backtest. None of this is hidden behind async endpoints or a job queue; both are natural additions once there's a real deployment target.
9. **The traffic-light approach inherits Kupiec's blind spot**, since it's built from the same breach *count* (via the same binomial model) and says nothing about clustering — a model in the green zone can still fail Christoffersen. Read the three backtest results together, not the zone in isolation; the demo output does exactly this by printing all three side by side. The zone also carries the same small-sample caveat as Kupiec (limitation 3): with few tested observations, small changes in breach count can swing the cumulative probability, and therefore the zone, more than the underlying calibration actually changed.
10. **`detect_gaps` cannot tell a market holiday from a genuine data outage.** Both show up identically as a business day absent from the fetched index; distinguishing them needs a real per-exchange trading calendar, which this module deliberately doesn't pull in (see design decisions). In practice, missing business days are overwhelmingly holidays for liquid, established tickers — the AAPL/MSFT/GOOGL example above has 45 missing days and zero per-asset gaps, consistent with that — but this module cannot promise that on its own; a spike in `missing_business_days` beyond what US market holidays would explain is worth investigating by hand, not trusting blindly.
11. **`auto_adjust=True` handles ordinary splits and dividends, not everything a corporate action can do.** Special/return-of-capital dividends, spin-offs, and mergers are not guaranteed to be reflected correctly in yfinance's adjustment, and a delisted or acquired ticker simply stops returning data (survivorship bias: a backtest built only from tickers that still trade today silently excludes every company that failed or was acquired out of the sample, which is exactly the population most likely to have produced a real VaR breach).
12. **Yahoo Finance is an unauthenticated, unofficial, rate-limited data source** with no SLA — it is adequate for research, demos, and this README, not for a production risk desk. `fetch_prices` raises a plain `ValueError` on a bad ticker or an empty response; it does not retry, back off, or cache, so a rate-limited request just fails the calling request.

## Project structure

```
risk-analytics-service/
├── src/risk_service/
│   ├── models/          # VaR/CVaR computation
│   ├── validation/      # Backtests: Kupiec, Christoffersen, Basel traffic light
│   ├── data/            # market_data.py: yfinance ingestion, gap detection
│   └── api/             # FastAPI layer: app.py (routes), schemas.py (request/response models)
├── tests/               # Analytical + property tests
├── scripts/             # End-to-end demos
├── notebooks/           # Exploratory analysis (research artefacts)
└── configs/             # Environment-based configuration
```

## References

- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. (1998). "Evaluating Interval Forecasts." *International Economic Review*, 39(4), 841–862.
- Basel Committee on Banking Supervision (1996). *Supervisory Framework for the Use of Backtesting in Conjunction with the Internal Models Approach to Market Risk Capital Requirements.*
- BCBS (2019). *Minimum capital requirements for market risk* (FRTB).
- Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." *Journal of Multivariate Analysis*, 88(2), 365–411.
