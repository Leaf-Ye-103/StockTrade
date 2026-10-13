# StockTrade

A modular research foundation for stock allocation with point-in-time market and information data, trainable deep learning, risk controls, and execution evaluation supported by observed historical trades. The initial workflow uses US equities and USD, evaluates one symbol at a time, and sends no orders to a broker.

The model learns market representations without hand-labelled trading signals. Its inputs can include OHLCV, quotes and size imbalance, measured order-flow and depth metrics, macroeconomic releases, company fundamentals, and news. A causal temporal encoder means each decision uses only information available by that timestamp; it does not establish causal relationships in the market.

Execution evaluation looks for compatible recorded trades and abstains when the records do not support a proposed order. A recorded match supports a counterfactual estimate, not a claim that inserting your order would preserve the historical queue or reveal its market impact.

## Install and run

The package supports Python 3.11 or newer. The pinned CPU development environment was validated on Python 3.12 and Linux x86_64. From the existing checkout:

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
stocktrade demo --output runs/demo --epochs 3
pytest
ruff check src tests
```

The demo generates synthetic input records and runs training and evaluation offline. It checks the software workflow; its returns are not investment performance evidence. It writes input fixtures under `runs/demo/synthetic`, a checkpoint under `runs/demo/model`, and an auditable report under `runs/demo/evaluation`. Add `--frequency-seconds 1` to exercise second bars.

Train on your normalized data and evaluate against a separate observed trade ledger:

```bash
stocktrade train --bars data/bars.csv --events data/events.jsonl --output runs/model --epochs 3
stocktrade evaluate --bars data/bars.csv --events data/events.jsonl --trades data/trades.csv --checkpoint runs/model/model.pt --output runs/evaluation --cash 100000
```

Training uses chronological 70% training, 15% validation and 15% held-out partitions by default. Keep `model.info.json` next to `model.pt`. Evaluation verifies the original training/validation history and uses subsequent bars. You can append future data; the checkpoint's feature schema stays fixed and future-only feature keys require retraining. The CLI requires exact-quantity observed trade analogs and leaves unsupported proposals unmatched. The main outputs are `report.json`, `decisions.jsonl` and `forecasts.json`.

Collect permitted public RSS or Atom news for later ingestion:

```bash
stocktrade collect-news --url https://example.com/news/rss --output data/news.jsonl --symbol NVDA
```

Replace the example feed URL with a real feed you are allowed to fetch. A fresh scrape records when it became accessible to this process; it cannot supply historical availability retroactively.

## Research blocks

| Block | Purpose |
| --- | --- |
| [Data ingestion and features](src/stocktrade/data.py) | Validate timestamps, align market and information inputs as of each decision, preserve missingness. |
| [Learning](src/stocktrade/model.py) | Train a temporal multimodal representation and a bounded allocation policy. |
| [Risk](src/stocktrade/risk.py) | Bound exposure, turnover and portfolio loss conditions before orders are evaluated. |
| [Execution evaluation](src/stocktrade/execution.py) | Match proposals to observed executions, report unsupported orders and match coverage. |
| [Workflow](src/stocktrade/pipeline.py) | Keep chronological partitions, checkpoint provenance, decision audit records and reports. |

See [architecture](docs/architecture.md), [input formats and data requirements](docs/data.md), and [research methodology and limitations](docs/research.md).

Minute and second bars are supported when their sampling interval is consistent. Risk and training settings are configurable through the Python APIs, including `TrainingConfig`, `RiskConfig` and `MatchingConfig`; the CLI exposes the core workflow. Real queue reconstruction requires exchange-level order lifecycle data and a provider adapter. The current import workflow has no integrated commercial feed, production trading system, or profitability guarantee. It does not yet model every information source or risk.
