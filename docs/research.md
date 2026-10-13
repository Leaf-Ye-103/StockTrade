# Research methodology

The implemented workflow is a starting point for experiments. It trains without manually labelled buy/sell targets, constrains proposed allocations, and evaluates whether observed historical executions support those proposals. It cannot establish that the strategy would make money in deployment.

## Learning without labelled decisions

Self-supervised learning derives training targets from the input sequence: reconstruct held-out observations and predict subsequent observations. This is a form of unsupervised representation learning. Any portfolio objective derived from future training-period returns is a further data-derived learning signal; it is not a manually labelled trading decision.

A trainable policy and a historical utility objective are not the same as a validated financial edge. Learn preprocessing parameters only from the training interval. Select model and risk hyperparameters on a validation interval, then evaluate once on a later held-out interval. Report the model's exact features, training dates, random seed and checkpoint with each result.

## Historical matching and what it proves

For a proposed order, the evaluator asks whether sufficiently similar observed executions occurred in a limited subsequent window. Matching preserves the symbol, execution mode, limit price, quantity and relevant market conditions. Aggressive analogs share the aggressor side; passive analogs use the opposite aggressor side and need order-level queue evidence. Incompatible or insufficient records produce abstention. Keep an audit trail from each supported analog to its source records.

For example, an aggressive proposal to buy 100 shares at a $225 limit needs a compatible recorded buy-aggressor execution at or below $225, after the assumed arrival delay, supported by contemporaneous quote and regime context. The CLI requires an actual quantity of 100 shares. The standalone API can permit a smaller near-size analog and retains its quantity unchanged. A close price of $224 or total minute volume of 10,000 alone is insufficient. Even compatible executions do not prove that this additional order would have received those executions.

Results are conditional counterfactual portfolio estimates. The new order may displace existing orders, change depth, reveal information, or trigger other traders' responses. Historical matching does not observe those outcomes. It cannot causally identify your own market impact or a queue changed by insertion of your order.

This restriction should affect reported conclusions. Low support coverage, small samples, concentrated regimes, and loose matching tolerances all weaken an apparent result. An evaluator that frequently abstains may provide useful evidence that the available dataset does not support the strategy.

## Interpreting a report

Assess these together:

- The number of proposals, supported analogs and abstentions, including reasons for rejection.
- Match price, size, time distance, and record reuse protections.
- Net portfolio estimates after explicit costs, with drawdown and exposure history.
- Trades and periods where risks forced a smaller allocation or blocked an order.
- Sensitivity to match tolerances, execution delay, costs and available liquidity.

Portfolio marks at bar closes are valuation assumptions, not execution evidence. Returns from a selected set of matching trades do not describe unsupported proposals or opportunity costs automatically. Report calendar time and cash holdings as well as traded intervals. The baseline ledger includes a one-basis-point execution fee and reports estimated slippage separately; it does not model settlement, taxes, borrow, financing or all venue fees.

The offline synthetic demo is a software integration check. Its generated price paths, executions and news do not measure predictive ability, liquidity realism or future returns. Do not use demo performance to select an investment strategy.

## Validation before expanding scope

A credible research program needs more than a chronological split:

1. Use purged walk-forward evaluation with embargoes sufficient for feature lookbacks, target horizons and execution windows. Check for overlap between training targets and evaluation periods.
2. Evaluate across symbols and market regimes, including unseen securities and periods of stress. Reconstruct the historical universe rather than selecting today's survivors.
3. Compare against transparent baselines such as cash, constrained buy-and-hold and a simple non-neural allocation rule under the same execution evidence requirements.
4. Include fees, spread, financing where applicable, corporate actions and provider-specific execution assumptions. Measure uncertainty and sensitivity rather than reporting one best return.
5. Add full depth order lifecycle data and validated venue adapters before claiming queue realism. Use probabilistic fill and impact models with calibrated uncertainty for unsupported counterfactual effects.
6. Review model stability, distribution shifts, text source reliability, data revisions and operational faults. Stress test volatility jumps, gaps, halts, stale data and unavailable liquidity.

## Roadmap

Useful extensions are provider-backed historical data, richer probabilistic forecasting and uncertainty, a semantic text encoder with entity linking, cross-asset portfolio risk, purged validation utilities, event-driven depth replay and measured production execution calibration. A broker integration would require a separate operational design with reconciliation, monitoring and a tested kill switch.

No finite input set covers every influence on price, and no risk objective captures every loss mechanism. The current modules make assumptions explicit so they can be tested and replaced as better data and evidence become available.
