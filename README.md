# StockTrade

StockTrade takes total funding, a permitted stock universe, current holdings and point-in-time information, then proposes **buy, sell or hold actions with dollar amounts**. Its trainable neural policy has one financial goal: grow final net account value over a complete trading episode, subject to hard risk constraints. It does not predict a stock price or assign hand-written rewards to individual trades.

Inputs include OHLCV, quotes, observed order-flow/depth metrics, macroeconomic releases, fundamentals, news and investment-platform metrics. Minute and second bars are supported for multiple US equities in USD. Online action inference refreshes registered public information sources automatically; historical training/evaluation use supplied point-in-time archives. No broker orders are submitted.

## Install and run

The package supports Python 3.11 or newer. The pinned CPU development environment was validated on Python 3.12 and Linux x86_64. From the existing checkout:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
stocktrade demo --output runs/demo --epochs 3 --funding 100000 --symbols NVDA MSFT AAPL
pytest
ruff check src tests
```

The demo generates synthetic observations and executions. It validates the workflow, not investment performance. Add `--frequency-seconds 1` to exercise second bars.

Train and evaluate using your normalized archives:

```bash
stocktrade train --bars data/bars.csv --events data/events.jsonl --trades data/trades.csv --funding 100000 --symbols NVDA MSFT --episode-steps 128 --output runs/model --epochs 3
stocktrade evaluate --bars data/bars.csv --events data/events.jsonl --trades data/trades.csv --checkpoint runs/model/model.pt --funding 100000 --symbols NVDA MSFT --output runs/evaluation
```

Request a current action using recent completed bars. Online information collection is enabled by default:

```bash
stocktrade act --bars data/current-bars.csv --events data/events.jsonl --checkpoint runs/model/model.pt --funding 100000 --symbols NVDA MSFT --holdings data/holdings.json --cash 80000 --peak-equity 100000 --day-start-equity 100000 --output runs/action.json
```

Use `--offline` when replaying historical observations or deliberately using supplied events only. Action observations must be after the checkpoint's training/validation cutoff in either mode. Online mode rejects future-dated market bars and bars older than 900 seconds by default; `--max-market-age-seconds` configures that freshness limit. Current collection time can advance the final information clock without inventing a new OHLCV bar.

The holdings file maps trained stock symbols to nonnegative whole-share quantities, for example `{"NVDA": 100, "MSFT": 20}`. Supply actual remaining cash and historical `--peak-equity`/`--day-start-equity` for an existing account; the example values are illustrative and must be replaced with actual account history. Missing either baseline blocks new buys with `risk_history_missing`, while still permitting a chosen hold or risk-checked sale. Only a modeled fresh account with no explicit cash and no owned shares infers both baselines from initial funding. The runtime permitted symbols can be a subset of the trained universe; adding a stock requires retraining.

The policy chooses hold, or one allowed stock and a buy/sell budget of 1%, 5%, 10% or 20% of initial funding. The bins are configurable through the Python API. Risk sizing can reduce or reject an action and rounds down to whole shares: $100 cannot buy one $225 share.

## Financial objective and execution evidence

For a complete episode, the learning signal is:

```text
wealth_growth = final_account_equity / initial_funding - 1
```

Final equity includes remaining cash and observed marks for existing positions, after recorded execution prices and configured fees. There is no assumed final liquidation. Every sampled decision in an admissible trajectory receives the same episode signal through REINFORCE; there are no per-trade grades, imagined alternatives, risk penalties in the financial objective or price-prediction losses.

Buying at $100 and selling at $95 leaves $95 before fees. If the stock subsequently falls to $10, that trajectory preserves much more wealth than holding it, even though the round trip lost $5. The model learns from whole-account outcomes across episodes rather than awarding a separate avoided-loss bonus to the sale.

Hard controls constrain concentration, gross exposure, cash, liquidity, turnover and losses. An episode that breaches observed risk limits or lacks execution evidence for its intended trades is reported and excluded from training and checkpoint selection. Risk breaches do not become invented financial penalties. Default limits are research settings and cannot guarantee safety through gaps or unavailable liquidity.

Execution evaluation requires compatible **recorded trades**, including direction, exact quantity, timing and quote/regime evidence. Unsupported intentions remain unmatched; no bar-price fill is invented. Historical analogs cannot establish the causal market impact or altered queue priority of inserting your order.

Training uses chronological 70% training, 15% validation and 15% held-out partitions by default, with bounded training episodes starting from fresh funding. Training chunks have the same configured bar count (128 by default, reduced if the training prefix is shorter); an incomplete final chunk is left unused. The final bar values the account; it opens no new order. Preprocessing learns only from the training prefix. Validation selects the highest terminal wealth among admissible checkpoints over the same period; held-out evaluation never updates weights.

Keep `model.info.json` beside `model.pt`. Training writes `training.json` and `training_decisions.jsonl`. Evaluation writes `report.json`, annotated `decisions.jsonl`, `diagnostics.json`, `decision_risk.csv` and a standalone interactive `dashboard.html`. Open the HTML file locally to inspect candles, every decision, account valuations, risk and metrics. `--no-dashboard` retains JSON/CSV without HTML. Forecast and earlier action-reward checkpoints must be retrained in the new `direct_wealth_policy` format.

Risk diagnostics include historical VaR/expected shortfall in dollars, volatility, exposure, stress scenarios and separate estimated costs before and after feasible proposals, including the portfolio risk of holding. These estimates use past observations and remain separate from training and hard constraints. Missing inputs produce unavailable values with reasons. See [evaluation and reports](docs/evaluation.md) for assumptions, benchmark comparisons and report regeneration.

## Information collection and platforms

```bash
stocktrade collect-information --symbols NVDA MSFT --output data/online-events.jsonl
stocktrade collect-information --symbols NVDA MSFT --sources data/sources.json --output data/online-events.jsonl
```

Defaults attempt ticker-specific Yahoo Finance RSS and global Federal Reserve monetary-policy news. Configured HTTPS RSS, HTML and JSON sources can extend coverage. Collection is bounded and reports unavailable sources; it does not browse the entire internet. Dated source publication, actual receipt, expiry, deduplication and provenance are preserved. Undated RSS items are skipped; HTML/JSON sources lacking publication metadata fail visibly. Expiry is measured from source publication, so fetching unchanged old content cannot renew its freshness or historical availability.

Investment-platform metrics enter an ordinary neural feature group. Their importance is learned with other information, without a fixed priority multiplier. Provider archives, missingness and timestamps still determine whether those metrics can be used.

Read-only Alpaca imports provide a starter path for minute bars and news:

```bash
stocktrade fetch-alpaca-bars --symbols NVDA MSFT --start 2025-01-02T14:30:00Z --end 2025-01-02T16:00:00Z --feed iex --output data/alpaca-sample
stocktrade fetch-alpaca-news --symbols NVDA MSFT --start 2025-01-02T14:30:00Z --end 2025-01-02T16:00:00Z --output data/alpaca-news.jsonl
```

The adapter needs `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in secure environment settings and the applicable data entitlement. Bars/news do not provide the separate trade/quote evidence needed for execution evaluation. A historical news download receives current retrieval availability, so it cannot be inserted into past decisions. No account signup, purchased plan or forward paper-order integration is implied.

See [architecture](docs/architecture.md), [data formats and collection configuration](docs/data.md), [research methodology](docs/research.md) and [platform options](docs/platforms.md). LEAN requires a custom observed-execution fill model; venue-specific MBO and NautilusTrader are later queue-research options. Current provider pricing and entitlements need verification.

## Research blocks

| Block | Purpose |
| --- | --- |
| [Data](src/stocktrade/data.py) | Point-in-time feature alignment, missingness and expiry. |
| [Collection](src/stocktrade/collection.py) | Bounded automatic public-source refresh and provenance. |
| [Platforms](src/stocktrade/platforms.py) | Read-only provider imports without broker order endpoints. |
| [Policy](src/stocktrade/model.py) | Trainable modality/temporal/account encoders, cross-stock attention and direct actions. |
| [Risk](src/stocktrade/action_risk.py) | Hard account and liquidity constraints before execution. |
| [Execution](src/stocktrade/execution.py) | Observed matches and conditional account ledger. |
| [Objective](src/stocktrade/objective.py) | Terminal net wealth and separate episode feasibility. |
| [Workflow](src/stocktrade/pipeline.py) | Chronological episodes, training and checkpoint/report provenance. |
| [Decision risk](src/stocktrade/decision_risk.py) | Past-only current/projected portfolio diagnostics, including holds. |
| [Analytics](src/stocktrade/analytics.py) | Evaluation risk, performance, benchmark and market indicators. |
| [Visualization](src/stocktrade/visualization.py) | Standalone interactive HTML reports from auditable observations. |
| [Report orchestration](src/stocktrade/diagnostics.py) | Annotated report artifacts and regeneration from archived evidence. |

No financial edge has been demonstrated. Realistic performance claims require licensed historical observations, adequate execution coverage and independent evaluation across market regimes.
