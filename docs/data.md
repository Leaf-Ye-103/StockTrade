# Data contracts

The import workflow accepts normalized CSV bars and an optional JSON Lines information stream. Training and historical evaluation also require recorded trades: these supply supported execution evidence for action feedback. Funding, permitted symbols and current cash/holdings are separate policy inputs. Use timezone-aware UTC timestamps with `Z` or `+00:00`. All examples here are illustrative, not historical market records.

## Bars

A bar timestamp is the **end** of its interval, when its OHLCV is available. Use an intentional, consistent positive interval for a run, such as one minute or one second. Each stock's timestamps must increase strictly, and all checkpoint stocks must have exactly matching timestamp sequences; misaligned data is rejected rather than padded. Session breaks, missing intervals and trading halts need explicit preprocessing and interpretation: lookbacks and reward horizons count input observations, whose elapsed time can differ across a gap. A one-second bar is still an aggregate: it does not reconstruct the sequence of quotes and executions inside that second.

```csv
timestamp,symbol,open,high,low,close,volume,bid,ask,bid_size,ask_size
2025-01-02T14:31:00Z,NVDA,100.00,100.10,99.95,100.05,1200,100.04,100.06,300,250
2025-01-02T14:31:00Z,MSFT,400.00,400.20,399.95,400.05,1600,400.04,400.06,400,350
2025-01-02T14:32:00Z,NVDA,100.05,100.15,100.00,100.10,900,100.09,100.11,200,350
2025-01-02T14:32:00Z,MSFT,400.05,400.25,400.00,400.10,1800,400.09,400.11,450,350
```

| Field | Meaning |
| --- | --- |
| `timestamp` | Timezone-aware bar end. |
| `symbol` | Stable identifier used consistently across all files. |
| `open`, `high`, `low`, `close` | Positive prices in the same currency and adjustment basis. |
| `volume` | Nonnegative volume for this interval. |
| `bid`, `ask` | Optional quote at or before the bar end; identify its source and freshness. |
| `bid_size`, `ask_size` | Optional displayed quantities corresponding to the quote. |

An absent optional field represents missing information. Bid/ask sizes provide a top-of-book imbalance proxy, not queue position or full market depth. Avoid mixed currencies or mixed split adjustment conventions.

## Funding and holdings

Pass positive initial funding and the permitted stock symbols through `--funding` and `--symbols`. Funding is both an account input and the fixed scale for action budgets and reward normalization. The default action bins are 1%, 5%, 10% and 20% of that amount. The policy may buy or sell only allowed symbols; it cannot short stocks. Inference may restrict the trained universe to a subset, but expanding it requires retraining. Bars for all checkpoint stocks remain necessary to produce its account and cross-stock context.

For `stocktrade act`, an optional `--holdings` JSON file maps symbols to nonnegative whole-share quantities:

```json
{"NVDA": 100, "MSFT": 20}
```

Supply actual remaining cash using `--cash` when supplying holdings. Every held stock requires an observed current mark for account valuation. Without initial holdings, a new account begins with its funding in cash. Action inference consumes historical observations only and does not require a future trade ledger. Training and historical evaluation need a ledger because unsupported actions cannot generate economic feedback.

Proposed dollar values are gross execution budgets before fees, not share counts or guaranteed fills. Risk sizing rounds down to whole shares using observed quote prices and account limits. A $100 proposal for a stock above $100 can therefore produce no executable quantity; fractional fills are not synthesized.

## Information events

Store one event per line. Every event contains source timestamps and a type. A symbol-specific event has `symbol`; a market-wide event can leave it null.

```json
{"published_at":"2025-01-02T13:00:00Z","available_at":"2025-01-02T13:00:02Z","source":"licensed-macro-feed","kind":"macro","values":{"policy_rate":4.5},"text":"","symbol":null}
{"published_at":"2025-01-02T13:30:00Z","available_at":"2025-01-02T13:30:03Z","source":"company-filing-feed","kind":"fundamentals","values":{"revenue_growth":0.12},"text":"","symbol":"NVDA"}
{"published_at":"2025-01-02T14:30:00Z","available_at":"2025-01-02T14:30:00.050Z","source":"licensed-depth-feed","kind":"orderflow","values":{"buy_volume":600,"sell_volume":450,"cancelled_volume":300,"bid_depth":1500},"text":"","symbol":"NVDA"}
{"published_at":"2025-01-02T14:20:00Z","available_at":"2025-01-02T14:20:10Z","source":"publisher-rss","kind":"news","values":{},"text":"Illustrative company announcement","symbol":"NVDA"}
```

`published_at` describes the source publication. `available_at` describes the earliest time this strategy could access that version of the record. An event is eligible only when both times have passed. Use finite numerical values with consistent names and units.

Supported kinds are `macro`, `market`, `fundamentals`, `orderflow`, and `news`. Numerical macro values become `macro.<key>` features, benchmark values become `macro.market.<key>`, company values become `fundamentals.<key>`, and observed flow values become `orderbook.flow.<key>`. Matching symbol events and global events are eligible. Latest known publications persist; an older release arriving late cannot overwrite a newer publication. Missing values use zero with a false observation mask. A persisted value's presence does not establish its freshness: select appropriate update frequency and stale-data rules for your source.

Order-flow values must be externally observed metrics from a documented provider, such as executed buy/sell volume, cancellations, or measured depth. Identify each metric's interval, units, venues, sampling and delivery time. Metrics computed over an interval become available only after that interval and any provider delay. Do not derive aggressor volume or cancellations from OHLCV alone. These measurements improve input coverage but do not reconstruct individual orders or establish hypothetical queue priority. Full market-by-order lifecycle ingestion remains an extension.

The workflow derives its feature schema from the training prefix and freezes it for validation and evaluation. Unknown future-only numerical keys are ignored until retraining; expected fields absent from a later file remain masked. The lower-level `build_features(..., feature_names=...)` API supports this explicit schema boundary.

News uses 16 deterministic signed token hash features, a decayed count, and log age. Its weight decays exponentially with a six-hour timescale starting when both publication and availability have occurred. This does not infer sentiment, importance, or company relevance. New kinds need an explicit feature mapping; their presence in a file does not alone establish that the model consumes them.

The news collector fetches a permitted HTTPS RSS/Atom source and records actual collection time as availability. It checks the source's robots policy, limits response size, and does not crawl article pages. It does not infer which stocks are impacted. The optional symbol is a user-supplied tag, not evidence of relevance. When a feed supplies no publication date, the collector uses collection time for publication as well. Collection appends records without deduplication; do not repeatedly import duplicate stories as independent evidence. Feed text is untrusted data, not executable instructions.

Do not scrape a current feed and assign an old `available_at` because an article reports an earlier publication date. Such records can be used after collection, but do not constitute point-in-time historical news data. For historical research, obtain an archive with delivery timestamps and enough provenance to detect corrections and duplicate articles.

## Historical execution records

Execution records support the match-based evaluator. They are separate from bars; bar volume does not establish that a requested order could fill. Keep provider and venue provenance alongside the normalized file.

```csv
timestamp,trade_id,symbol,side,quantity,price,bid,ask,spread,volatility
2025-01-02T14:31:00.250Z,example-001,NVDA,buy,100,100.06,100.04,100.06,0.02,0.001
```

| Field | Meaning |
| --- | --- |
| `timestamp` | Actual execution event time with timezone. |
| `trade_id` | Unique identifier across the observation set. |
| `symbol` | Instrument identifier shared with the bars. |
| `side` | Explicit `buy` or `sell` side of the initiating aggressor. |
| `quantity` | Positive integer quantity actually executed. |
| `price` | Positive execution price. |
| `bid`, `ask` | Quotes observed before the execution; required for matching and used to validate the recorded aggressor's marketability. |
| `spread`, `volatility` | Pre-execution regime context; both required under default matching rules. The CLI uses absolute quote spread (`ask - bid`, in price currency units) and rolling standard deviation of per-bar log returns over up to 20 prior intervals. Use matching source conventions. This trade spread is distinct from the model's relative spread feature. |
| `impact_bps` | Optional subsequent movement diagnostic. It is never used to select a match. |

Passive matching additionally requires `queue_ahead`, `queue_depletion`, `queue_observed_from`, `queue_evidence_ids` and `queue_priority_verified`. Evidence IDs are separated by semicolons and reference actual executions at the same level between order arrival and the candidate event. Vendor-supported priority evidence is required. Top-of-book size alone cannot justify setting `queue_priority_verified` to true. The default CLI proposals are aggressive; passive matching is available through the execution module's Python API.

The CLI requires exact observed quantity and searches only before the next bar decision as well as within the execution horizon. The standalone API's `MatchingConfig` defaults search within 60 seconds after a 100 ms assumed arrival delay and permit recorded quantity between 90% and 100% of the proposal. Actual quantity is retained without creating extra shares. Spread and volatility each permit 50% relative deviation. Passive queue position permits 25% relative deviation. These research tolerances are configurable through `MatchingConfig` and are assumptions to examine, not empirically calibrated constants.

Public prints often lack the initiating order side, queue position, depth, or cancellation events. Do not silently treat an inferred aggressor side as the side of an order whose queue lifecycle is known. Missing quote, regime or queue evidence produces abstention. Duplicate IDs and reused source records are rejected. More detailed order lifecycle data is required for queue reconstruction.

## Data quality required before a market claim

- Retain the exchange calendar, sessions, timezones, trading halts and source clock precision. Do not interpolate missing bars and present them as observed trades.
- Use corporate action handling consistently across bars, quotes, trades and holdings. Record split and dividend effects explicitly.
- Use fundamentals as first released, with actual release times and later revisions represented separately. Period-end dates are not disclosure dates.
- Use macro releases and revisions as available at that time, rather than today's revised historical series.
- Track the historical investable universe, listing changes and delistings to avoid survivorship bias.
- Retain market and sector context with a documented timing convention. An index daily close cannot enter an earlier intraday decision.
- Preserve source licensing, provenance, time of collection and missingness. Separate test fixtures and synthetic records from actual observations.

The baseline validates its normalized inputs. It cannot verify whether a vendor's fields satisfy these historical truth conditions. No commercial data service is integrated yet; adapters and credentials should be added for a selected, licensed provider without storing secret values in files or source control.
