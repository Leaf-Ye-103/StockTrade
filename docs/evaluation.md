# Evaluation diagnostics and interactive reports

Evaluation produces an account and execution audit, historical risk estimates, and an interactive HTML dashboard. The diagnostics describe a held-out run; they do not change the policy, its terminal-wealth training objective, hard risk limits or checkpoint selection.

The charts and metrics remain conditional on observed execution analogs. An intended action, a feasible proposal and a matched historical execution are different records. A chart marker does not establish an actual fill or identify the market impact of inserting a new order.

## Run and inspect

```bash
stocktrade evaluate --bars data/bars.csv --events data/events.jsonl --trades data/trades.csv --checkpoint runs/model/model.pt --funding 100000 --symbols NVDA MSFT --output runs/evaluation
```

Open `runs/evaluation/dashboard.html` as a local file in a browser. It contains interactive market, decision, portfolio and risk views. Evaluation writes the underlying JSON and CSV audit alongside it:

| File | Purpose |
| --- | --- |
| `report.json` | Run summary, wealth, execution support and diagnostic references. |
| `decisions.jsonl` | Every policy decision with feasibility, execution evidence and risk annotations. |
| `diagnostics.json` | Performance, risk, benchmark and market-indicator diagnostics, including unavailable values and reasons. |
| `decision_risk.csv` | Flattened before/projected risk and cost estimates for per-decision inspection. |
| `dashboard.html` | Standalone interactive report; the source audit remains in the accompanying files. |

Use `--no-dashboard` to retain the machine-readable reports without generating HTML. The synthetic demo also exercises the reporting workflow; its plots are software validation, not evidence of investment performance.

An optional benchmark must be a symbol in the same supplied bars CSV. It can be outside the checkpoint's trained stock universe because it is used only for evaluation:

```bash
stocktrade evaluate --bars data/bars-with-spy.csv --events data/events.jsonl --trades data/trades.csv --checkpoint runs/model/model.pt --symbols NVDA MSFT --benchmark SPY --risk-free-rate 0 --risk-lookback 60 --risk-confidence 0.95 --risk-min-observations 20 --output runs/evaluation-with-benchmark
```

Without `--benchmark`, the report constructs an equal-initial-weight normalized raw-price index of the strategy universe. With the option, it uses the named symbol. Both are price-only references, not strategies filled through the matcher, dividend-inclusive total-return indexes, or alternative investment outcomes under identical trading constraints. Missing or misaligned observations cannot be silently forward-filled into a comparison.

Decision records carry `risk_estimate.before`, `risk_estimate.projected` and `risk_estimate.delta`. The `performance` object in `diagnostics.json` contains the run's metric families; its `availability` map explains unavailable metrics.

## Risk at each decision

For each decision, historical scenarios revalue the **entire current portfolio**, including existing holdings when the policy selects hold. The projected view uses the feasible whole-share proposal after hard risk checks. A hold or rejected proposal leaves the projected portfolio unchanged. A permitted proposal's projected risk is conditional on that proposal executing; it does not update the observed account.

The estimator uses only completed observations available by the decision timestamp. By default it considers up to 60 past one-bar simple returns, requires at least 20 aligned return observations, and uses 95% confidence. Across held stocks, it intersects dated return pairs to preserve contemporaneous portfolio dependence. These scenarios apply past price moves to current dollar notionals; they do not reconstruct the account's historical positions. An observed interval can span a session gap, so unequal elapsed durations require care.

| Estimate | Meaning and units |
| --- | --- |
| Historical value at risk | Empirical loss quantile, in USD, using the higher observed quantile. A nonnegative estimate, not a maximum possible loss. |
| Historical expected shortfall | Average observed loss in the quantile tail, in USD, floored at zero. Sparse or unrepresentative tails remain a limitation. |
| Portfolio volatility | Population standard deviation of historical dollar P&L scenarios over one observed bar interval. It is not annualized. |
| Exposure and concentration | Current/projected notionals, gross exposure, gross fraction, cash fraction and largest position fraction. |
| Market stress | Portfolio dollar changes under simultaneous 1%, 5% and 10% price declines across holdings. Scenarios are assumptions, not probabilities or exhaustive risk coverage. |
| Estimated costs | Commission, half-spread and configured slippage estimates for a feasible proposal, reported separately from historical risk. |

Projected holdings are valued at current observed closes, with proposed cash movement based on executable quote notional and excluding the separate cost estimates. Dollar-risk differences are signed projected-minus-before values; a negative change can indicate lower estimated exposure to the sampled scenarios. It does not prove the action is preferable or safer under every market outcome.

Costs are diagnostic estimates. They are not extra ledger charges, policy penalties or additions to VaR/expected shortfall. Matched execution prices and recorded ledger fees remain separate observed-analog accounting. Unknown own-order market impact, changed queue priority and unobserved liquidity are not assigned a reassuring zero.

An all-cash portfolio can have zero market-price scenario risk. Missing prices or too little aligned history produce an unavailable estimate with a reason; malformed decision/configuration records fail validation. Zero, unavailable and unsupported execution must remain distinct. With 60 returns, a 95% tail can contain only a few observations. A 95% historical estimate does not mean a 95% guarantee, especially across regimes, jumps, halts or data failures.

## Read the dashboard with the audit

Use the market chart's symbol selector, pan/zoom and hover controls to inspect candles, volume and decision markers, with SMA 20/50/200 and Bollinger overlays. A separate technical panel has its own stock selector for RSI, MACD/signal/histogram, ATR, OBV and nonannualized rolling volatility. Missing indicator warmup remains blank. Proposal markers are placed at observed decision-bar closes; separate star markers use recorded analog prices and times. A portfolio hold appears in the decision timeline even though it has no stock order. The searchable decision table filters by symbol and status and includes reasons, feasible quantity, current/projected risk, estimated costs, observed fees and source IDs.

Portfolio views show ledger equity, drawdown, gross exposure and cash. Risk views separate current/projected VaR and expected shortfall, their signed changes, and conditional cost estimates versus observed ledger costs. A histogram uses returns between successive ledger observations; a separate heatmap uses only available completed New York daily-return observations. The HTML embeds its plotting library and data, so opening the saved file does not require a chart CDN.

The diagnostic families include:

| Family | Included measures and conventions |
| --- | --- |
| Account performance | Net P&L/return, elapsed time, maximum drawdown, peak/trough/recovery times and time underwater. Open holdings remain marked. |
| Observed-return risk | Interval volatility, dollar changes, fixed 95% return VaR/expected shortfall, skewness and excess kurtosis. Tail counts and the selected return horizon are reported. |
| Annualized performance | Volatility, Sharpe and Sortino with sufficient completed daily observations; CAGR and Calmar with sufficient elapsed time. Undefined denominators produce unavailable values. |
| Execution | Holds, submitted intentions, matches, unmatched/rejected records, rejection reasons, coverage, two-sided turnover, observed fees and signed execution-versus-reference slippage attribution. |
| Closed lots | FIFO closed-lot fragments, realized net P&L, win rate, profit factor, payoff ratio and holding duration. Open lots have no fabricated exits; missing opening lots or fees limit these statistics. |
| Portfolio | Final cash/exposure, time-weighted exposure, per-stock marked weights, largest stock weight and invested-notional concentration (HHI, excluding cash). |
| Benchmark | Raw-price reference return and period excess; beta, correlation, alpha, tracking error and information ratio when completed daily-return pairs align. |
| Per-stock indicators | SMA 20/50/200, Wilder RSI 14, MACD 12/26 with signal 9, Bollinger bands (20-bar mean ± two population standard deviations), Wilder ATR 14, OBV and 20-return realized volatility. |
| Observation quality | Sampling gaps, zero-volume frequency, quote coverage, median spread/dollar volume, open gaps and an Amihud-style absolute-return-per-dollar-volume proxy. |

Indicators use only preceding supplied bars for warmup and are reported over the evaluation interval. Their periods count bars, not days. OBV starts at zero at the first supplied bar. These are a selected set of technical diagnostics, not every possible indicator or a demonstrated trading edge.

Performance and technical indicators describe the supplied observations. They are diagnostics, not additional trading signals fed into the saved policy. A favorable indicator or benchmark comparison does not turn an unmatched proposal into a supported execution.

Annualized volatility, Sharpe and Sortino require at least 20 completed New York daily-close returns; benchmark statistics require at least 20 exactly aligned completed daily-return pairs. The daily value is the last observed mark per New York date and requires a timestamp at or after 16:00 local time; it is not necessarily an exact exchange-close print and can include after-hours marks. The first close is a baseline. Incomplete observed dates and missing/unverified weekdays break return adjacency. No exchange holiday or early-close calendar is inferred, so some otherwise valid comparisons may remain unavailable. Annualization uses 252 trading days. CAGR requires at least 365 elapsed days and positive final equity; Calmar also requires nonzero observed drawdown.

Shorter samples retain nonannualized snapshot-return diagnostics where available. A minute or second observation is not a daily return. `--risk-free-rate` supplies an annual reference assumption where a metric can use it; it does not create observed financing income. Run-level tail statistics require 20 return observations and use a fixed 95% linear quantile. They describe the observed account path; they differ from the configurable-confidence, higher-quantile dollar scenarios at each decision. Signed slippage attribution is already reflected in recorded execution prices and is not deducted again.

Broad indicators still omit unobservable or unmodeled risks: causal market impact, hidden liquidity, broker/counterparty failures, tax and settlement differences, operational outages, model drift and events outside the historical sample. Keep execution coverage, sample quality and those limitations visible beside returns.

## Regenerate an archived report

```bash
stocktrade report --bars data/original-bars.csv --report runs/evaluation/report.json --decisions runs/evaluation/decisions.jsonl --output runs/evaluation-regenerated
```

Use the original matching market observations and a new output directory. This creates an annotated report and dashboard from archived actions and ledger evidence. Supply the original `--benchmark`, `--risk-*` and `--risk-free-rate` reporting options again when reproducing custom diagnostics; otherwise the command uses its defaults. Changing these reporting settings does not change archived actions or hard constraints. Regeneration does not retrain the checkpoint, rerun the policy, fetch market data, submit orders or manufacture execution records. Keep the original audit for comparison.

New reports bind the decision archive through `provenance.decisions_sha256`. Regeneration verifies available hashes and checks decision counts, accounting and portfolio snapshots for consistency. It records source hashes in `regenerated_from`; older archives without the decision binding remain explicitly labelled `legacy_unbound_archive`. These checks preserve audit lineage and do not establish that source market observations are complete or correct.
