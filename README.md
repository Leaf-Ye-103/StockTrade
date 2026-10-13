# StockTrade

A modular research system that takes total funding, a permitted stock universe, current holdings and point-in-time information, then proposes **buy, sell or hold actions with dollar amounts**. A separate historical evaluator supplies reinforcement feedback from recorded executions and later observed prices. The trading model emits actions directly: it has no price-forecast output or prediction-loss objective.

The workflow supports multiple US equities in USD, with minute or second bars. Inputs can include OHLCV, quotes, measured order-flow and depth metrics, macroeconomic releases, company fundamentals and news. All inputs must have been available at the decision time. No broker orders are submitted.

## Install and run

The package supports Python 3.11 or newer. The pinned CPU development environment was validated on Python 3.12 and Linux x86_64. From the existing checkout:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
stocktrade demo --output runs/demo --epochs 3 --funding 100000 --symbols NVDA MSFT AAPL
pytest
ruff check src tests
```

The demo generates synthetic records and runs training and evaluation offline. It checks the software workflow; its results are not investment performance evidence. Add `--frequency-seconds 1` to exercise second bars.

Train using your normalized observations and recorded executions:

```bash
stocktrade train --bars data/bars.csv --events data/events.jsonl --trades data/trades.csv --funding 100000 --symbols NVDA MSFT --reward-horizon 8 --output runs/model --epochs 3
stocktrade evaluate --bars data/bars.csv --events data/events.jsonl --trades data/trades.csv --checkpoint runs/model/model.pt --funding 100000 --symbols NVDA MSFT --output runs/evaluation
```

Request a current action without training or evaluating against future data:

```bash
stocktrade act --bars data/bars.csv --events data/events.jsonl --checkpoint runs/model/model.pt --funding 100000 --symbols NVDA MSFT --holdings data/holdings.json --cash 80000 --output runs/action.json
```

The holdings file maps stock symbols to nonnegative whole-share quantities, for example `{"NVDA": 100, "MSFT": 20}`. Supply actual remaining cash when supplying holdings. Without holdings, the initial account starts with the funding in cash.

The policy chooses one action per decision: hold, or buy/sell one allowed stock using a budget of 1%, 5%, 10% or 20% of initial funding. These bins are configurable through the Python API. The risk layer may reduce or reject the proposal. Execution uses whole shares: a $100 budget cannot buy one $225 share, and no fractional execution is invented.

Training uses chronological 70% training, 15% validation and 15% held-out partitions by default. Training chunks default to 128 bars with a fresh funded account per chunk. Each chunk/partition reserves its final reward-horizon bars for observing feedback rather than opening new actions. Validation selects checkpoints by its recorded-execution account return minus 0.5 times maximum drawdown; `--selection-drawdown-weight` configures that penalty. Validation does not update weights, and held-out evaluation never trains them. Preprocessing learns only from the training prefix.

Keep `model.info.json` beside `model.pt`. Training writes `training.json` and `training_feedback.jsonl`; evaluation writes `report.json` and `decisions.jsonl`, including source trade IDs and reward components. Earlier forecast-based checkpoints are incompatible and must be retrained as direct-action policies.

## What the feedback means

For each decision, the evaluator compares the subsequent **matched-execution account path** with a baseline that freezes the cash and all shares held immediately before that decision. Both paths use later observed stock prices. A purchase therefore competes with retaining its cash, and a sale with retaining the same existing shares. The account path includes subsequent supported trades and their fees; the baseline needs no alternative trade.

For example, buying one share at $100, actually matching a sale at $95 and observing it later at $10 can reward selling for avoiding the additional $85 decline. The purchase is evaluated using the account's $5 loss after that supported sale, rather than pretending it stayed invested until $10. Each decision has its own baseline: the sale's reward is never added to the purchase's feedback. A rebound above $95 can instead penalize the sale for the foregone recovery. Holds can receive feedback from subsequent supported portfolio choices; if the account continues unchanged, holding has zero feedback.

Default feedback compares the two equity paths at future bars 1, 4 and 8, weighted 25%, 25% and 50%, and adjusts for their difference in downside. Execution fees are already included in the account path and are not deducted again. Changing `--reward-horizon` scales these checkpoints. Feedback is normalized by initial funding. The horizon defines what “later” means; the evaluator does not choose the best endpoint after seeing the outcome. Overlapping feedback sums are diagnostic only: they are neither portfolio profit nor the checkpoint-selection objective.

Historical evaluation requires compatible **recorded trades**, including direction, exact quantity, timing and quote/regime evidence. Unsupported actions remain unmatched and receive no economic reward. Matching is an observational estimate; it cannot establish the queue priority or causal market impact of adding your order.

## Research blocks

| Block | Purpose |
| --- | --- |
| [Data ingestion and features](src/stocktrade/data.py) | Align observed market and information inputs at each decision and preserve missingness. |
| [Action contracts](src/stocktrade/policy_contracts.py) | Represent funding, cash, holdings and dollar-valued buy/sell/hold proposals. |
| [Direct policy](src/stocktrade/model.py) | Encode stock histories and account state, compare stocks and learn action probabilities from evaluator feedback. |
| [Action risk](src/stocktrade/action_risk.py) | Constrain exposure, cash, liquidity and whole-share size before execution evaluation. |
| [Execution evidence](src/stocktrade/execution.py) | Match proposals to observed executions and report unsupported orders. |
| [Action evaluator](src/stocktrade/rewards.py) | Compare supported portfolio continuation with frozen pre-decision cash/holdings at fixed future horizons. |
| [Workflow](src/stocktrade/pipeline.py) | Preserve chronological partitions, mature feedback and checkpoint/report provenance. |

See [architecture](docs/architecture.md), [input formats](docs/data.md) and [research methodology](docs/research.md).

Collect permitted public RSS or Atom news for later ingestion:

```bash
stocktrade collect-news --url https://example.com/news/rss --output data/news.jsonl --symbol NVDA
```

Replace the example with a feed you are allowed to fetch. A fresh scrape records actual collection time and cannot supply historical availability retroactively.

Real queue reconstruction requires exchange-level order lifecycle data and a provider adapter. There is no integrated commercial feed, production trading system or demonstrated financial edge. Historical coverage bias, unmodeled risks and the choice of feedback horizons need further research before any market-performance claim.
