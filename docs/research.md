# Research methodology

The policy learns direct trading actions from final account wealth over complete historical episodes. This is constrained offline reinforcement learning, not purely unsupervised representation learning. It has no price-prediction objective, critic head or manually labelled trade quality.

## One financial objective

For an episode with initial funding `F`, remaining cash `C`, final owned quantities `q_s` and observed endpoint prices `P_s`:

```text
final_equity = C + sum(q_s * P_s)
wealth_growth = final_equity / F - 1
```

Supported executions changed cash and holdings at their recorded quantities/prices; their fees have already reduced cash. The objective adds no second fee charge, hand-picked downside coefficient or separate reward for preventing a loss. Open positions are marked, not virtually liquidated.

Buying one share at $100 and selling through a recorded execution at $95 leaves $95 before fees. A subsequent fall to $10 makes that trajectory much better than continuing to hold, whose endpoint wealth would be $10. The model learns differences in complete episode outcomes, rather than assigning positive/negative grades separately to the purchase and sale. A later recovery can change which trajectory preserves more wealth; the fixed episode endpoint is declared before evaluation.

A learning algorithm necessarily needs a measurable objective. Here that scalar is account wealth itself, rather than a price forecast error or an engineered catalogue of “good” trades.

## Trajectory learning and constraints

The actor receives funding, permitted stocks, current cash/holdings and point-in-time stock histories. Modality encoders, a GRU, account/inventory representations and cross-stock attention train end to end. It selects hold or a stock, side and dollar budget.

REINFORCE supplies the same terminal wealth-growth scalar to every sampled action in an eligible trajectory. It uses a prior-episode running financial-return mean for gradient variance reduction. That baseline is not a hypothetical trade or per-action financial reward. Default entropy regularization is zero; exploration comes from sampling action probabilities. The implemented method is a starting point and does not establish that it finds a globally optimal strategy.

Risk limits are hard feasibility requirements. They constrain actions and check observed account paths afterward. Episodes with a measured risk breach or missing intended execution support are excluded from training and checkpoint selection rather than assigned an invented loss. Their measured final wealth remains visible. Current inference needs actual peak/day-start equity for existing accounts; missing loss history blocks additional buys, rather than silently resetting the account's baselines. Filtering changes which behavior can be learned, so report admissibility and rejection rates with performance.

Training episodes start from fresh funding and default to 128 bars, reduced only when the training prefix cannot fit that count. Updates use equal-sized chunks; a remaining short tail is not scored as a shorter episode. Resets limit long-duration inventory learning. The final observation values the account and opens no new order. Chronological 70/15/15 splits separate training, validation and held-out periods. Validation selects highest terminal wealth among admissible trajectories over the same period; held-out data never updates weights or chooses a checkpoint. Schemas and preprocessing fit only the training prefix.

Unequal durations and favorable endpoints can distort comparisons. Report both elapsed steps and calendar time; compare policies on the same held-out interval, not one annualized value from mismatched episodes. Walk-forward experiments, untouched final tests and uncertainty estimates remain necessary.

## Historical execution support

A proposal requires a compatible recorded event within the allowed arrival/window, retaining symbol, side, limit, exact quantity and quote/regime evidence. Passive orders additionally need attributed order-level queue evidence. The earliest compatible event is chosen without ranking subsequent profit. Missing matches cause abstention; aggregate volume or a bar close does not establish a fill.

For example, buying 100 shares at a $225 limit requires a compatible recorded execution of 100 shares at or below $225, after assumed arrival and with the required context. Actual identifiers connect every match to its evidence. The default CLI requires exact size; lower-level APIs may accept a smaller near-size analog while preserving its actual quantity.

Wealth is conditional on this selected evidence. Matching does not identify an inserted order's causal impact, altered queue priority or response by other traders. A game-theoretic market model would require explicit strategic agents and validated response assumptions; the current neural policy does not provide those.

Exact-size coverage, permitted universe and market regimes affect episode eligibility. Rejecting unsupported episodes is preferable to teaching from invented fills, but it can concentrate training on unusually well-covered behavior. Report eligible episodes, unmatched intentions, abstention causes and stock/regime composition. Cash-only or sparse evidence is not proof of a trading edge.

## Information collection and platform features

Historical news, fundamentals and macro inputs need actual publication, delivery, revision and expiry records. Current collection never supplies historical delivery times retroactively. Historical training/evaluation do not automatically fetch current websites into their input periods.

Current `act` automatically refreshes a bounded list of public sources and configured APIs. Default Yahoo ticker RSS and Federal Reserve feeds are discovery inputs, not exhaustive coverage of all information that affects prices. Undated RSS entries are skipped; missing HTML/JSON publication metadata reports failure. Publication-based expiry cannot be renewed by fetching unchanged old content. Failures, blocked domains and missing provider fields must stay visible. Registered sources obey access controls and supported authentication; collectors do not bypass paywalls or create provider accounts.

Investment-platform numerical metrics are ordinary features with learned neural importance. They receive no special priority multiplier. Canonical field registration does not create observations: unseen/missing historical metrics stay masked until the model is trained with valid records. Text uses hashed features, activity and recency; semantic understanding, relevance and resistance to manipulated content still need research.

Alpaca's imported minute bars may contain later archive corrections and do not provide verified historical receipt times. Imported news is available at retrieval, not its original publication. IEX and SIP differ in coverage; neither imported bars alone establish historical order lifecycles. See [platform options](platforms.md) for checked integration evidence and limits.

## Interpreting results

Read terminal wealth together with:

- Risk feasibility and violations, including observed gaps and account exposure.
- Intended trades, supported matches, source IDs and excluded episodes.
- Cash, open positions, endpoint marks, execution fees, drawdown and elapsed time.
- Results by stock/regime and sensitivity to latency, costs and matching tolerances.
- Source availability, expiry, missing features, collector failures and historical provenance.

Evaluation also records historical risk diagnostics for every decision, including holds. Current and projected portfolio VaR/expected shortfall, volatility, concentration and hypothetical stress scenarios use past observations; a feasible proposal's projected risk remains conditional, even if execution later proves unsupported. Estimated commission/spread/slippage costs remain informational and are not extra ledger charges or policy rewards.

The interactive dashboard and JSON/CSV artifacts expose those estimates alongside decisions and execution evidence. Missing data and inadequate samples produce unavailable values with reasons. Optional benchmark comparisons use observed aligned prices as an unexecuted reference; they do not assume an alternative filled strategy. Annualized metrics require defensible sampling and sufficient history. The report does not treat each minute or second as a trading day. See [evaluation diagnostics](evaluation.md).

These diagnostics do not change terminal-wealth learning, hard constraints or checkpoint selection. Selecting models repeatedly after inspecting test charts would still contaminate the held-out test, even though report generation itself performs no learning. Historical VaR, expected shortfall and stress scenarios cannot cover every possible loss or an unobserved queue/impact effect.

Taxes, settlement, financing, dividends and all venue charges are outside the current ledger convention unless explicitly added. Endpoint marks are valuations, not evidence of an executable liquidation. Hard constraints cannot guarantee safety during halts, discontinuities, correlated losses or operational failures.

The synthetic demo validates software integration and input frequency handling. Its records cannot establish profitability, predictive skill, liquidity realism or generalized learning.

Before a market claim, use purged walk-forward evaluation, reconstructed historical universes including delisted stocks, consistent corporate actions and point-in-time revisions. Compare cash and transparent constrained trading baselines under identical execution requirements. Stress test missing/stale information, gaps, liquidity loss and provider outages. Validate full depth/order lifecycle adapters before queue claims and a structural impact model before causal impact claims.

Provider-backed history, semantic news, longer episodes, improved constrained policy algorithms and calibrated portfolio stress tests are possible extensions. No finite source set covers every influence or every risk.
