# Research methodology

The system learns a direct trading policy from a separate observed-data evaluator. It receives no manually labelled trading decisions and has no market prediction objective. This is offline reinforcement learning with shaped delayed feedback, rather than purely unsupervised representation learning. A trainable policy and a coherent reward do not establish a financial edge.

## Why decision feedback differs from trade profit

A round-trip profit cannot tell whether both decisions were good. Buying at $100 and selling at $95 loses $5, but a later decline to $10 makes selling preferable to retaining that exposure. Treating both decisions as equally bad hides this distinction.

The active portfolio evaluator freezes the cash and **all shares held immediately before each decision**, then marks that baseline with later observed prices. The comparison path is the account ledger, including the decision and later supported actions. For a simple account with no subsequent trades, this reduces to:

| Action | Baseline | Fixed-horizon marked advantage |
| --- | --- | --- |
| Buy `q` shares at recorded price `p` | Retain the purchase cash | `q × (future mark − p)` |
| Sell `q` owned shares at recorded price `p` | Retain those same shares | `q × (p − future mark)` |
| Hold | Freeze the existing cash/shares | Zero if no later supported account changes; otherwise evaluate the observed continuation |

These baselines do not assume another buy or sale. Future closes are valuation marks, not claimed exit fills. The evolving comparison path changes its account only through supported recorded executions. Cash has a zero nominal return baseline; financing, interest, taxes and dividends are outside the current reward convention.

At the default eight-step horizon, the evaluator combines account-equity advantages at steps 1, 4 and 8 with weights 25%, 25% and 50%. It adds `0.25 × (baseline downside − ledger downside)`, where each downside is the maximum dollar loss below pre-decision equity over that horizon. This is a starting-equity loss measure, distinct from peak-to-trough percentage drawdown. Execution fees already reduced ledger equity and are not deducted again. Feedback is normalized by initial funding. This can capture the value of replacing one exposure with another through subsequent supported trades, rather than evaluating every purchase as permanently held through its horizon.

For one share sold at $95, a terminal $10 mark gives an $85 advantage over retaining that share, before weights, risk shaping and ledger fees. A terminal $110 mark gives a $15 disadvantage. An earlier purchase at $100 is evaluated from its own pre-decision account: if the subsequent sale at $95 is supported, its continuation has $95 cash, a $5 loss against retaining the original $100. It is not penalized as though it remained invested until $10. Selling can therefore be rewarded for preventing more loss while the purchase receives its own negative feedback.

“Later” has a predeclared meaning. A decline after the configured horizon will not retroactively change feedback. The evaluator does not compare against the best hindsight trade, choose a favorable endpoint, or mark an unmatched action as if it executed. Horizon, weights, fees and downside coefficients are experiment settings selected on training/validation data and frozen for the held-out period.

## Learning from delayed feedback

The neural policy consumes funding, allowed symbols, current cash/holdings and point-in-time stock histories. It outputs a categorical hold or a symbol, side and dollar budget. Modality encoders, a GRU and cross-stock attention are trainable end to end.

Training samples actions and updates their probabilities using REINFORCE with an empirical running reward baseline from prior updates and entropy regularization. The loss is based on observed evaluator feedback, not error between predicted and realized stock prices. Each decision receives one account-path comparison after its complete horizon matures. Later supported actions influence that path, but the trainer never adds their individual shaped feedback to the earlier decision. A hold can receive feedback from subsequently supported portfolio choices, such as waiting before buying; absent later account changes, its feedback is zero. Hard-constraint costs, when present, are separately identified policy penalties rather than invented trading losses.

Decision feedback comparisons overlap economically and are not additive trajectory rewards. In the buy-at-$100, supported-sale-at-$95, later-$10 example, adding the sale's avoided-loss reward to the purchase's account loss could give the poor purchase a positive training signal. The trainer never does this: it computes each decision's own trajectory advantage against its own frozen account baseline. There is no discounted sum of later shaped rewards. A later supported sale affects the earlier actual cash/holdings path, rather than contributing its separate avoided-loss bonus. **Feedback sums are diagnostic only, not portfolio profit or the checkpoint-selection objective.** Maintain and report the cash/holdings ledger separately.

These comparisons evaluate observed continuation against a fixed no-more-trading baseline. They do not replay a policy with the original action removed, and they do not isolate the marginal causal value of a particular order. Later portfolio choices can influence several overlapping decision windows. Coverage and the baseline convention therefore remain material research limitations.

Validation selects a checkpoint using recorded-execution account return minus a maximum-drawdown penalty: `ledger_return − selection_drawdown_weight × max_drawdown`. The weight defaults to 0.5 and is configured by `--selection-drawdown-weight`. This prevents a high sum of overlapping action feedback from being mistaken for successful portfolio performance. The ledger is still conditional on historical matching, not a causal live-strategy return.

Feedback is unavailable until its full horizon has matured. Training rewards, matching windows and future marks remain inside the training partition. Bounded training chunks default to 128 bars and begin with a fresh funded account; weights update after each rollout. These resets limit long-duration inventory learning and must be considered when interpreting results. The final horizon bars in every chunk/partition are reserved for feedback observation rather than new decisions. Validation uses frozen weights and may select a checkpoint; held-out data never updates policy parameters. Feature schemas and scaling use the training prefix only. A later mark can affect evaluator feedback after maturity but cannot enter an earlier policy observation.

Chronological splitting limits leakage but does not demonstrate generalization. Overlapping action horizons create statistical dependence, and repeated validation selection can overfit. Purged walk-forward experiments and untouched final test periods remain necessary for a market-performance claim.

## Historical matching and selection bias

Each proposed execution needs a compatible recorded event within a bounded subsequent window. Preserve symbol, mode, side, price constraints, exact quantity and quote/regime evidence. Aggressive analogs share aggressor side; passive analogs need attributed queue progression. The earliest compatible record is selected without evaluating later profitability. A missing match creates abstention, not a bar-price fill.

For example, an aggressive proposal to buy 100 shares at a $225 limit requires a compatible recorded buy-aggressor execution of 100 shares at or below $225 after assumed arrival, with supporting quote and regime context. A close price of $224 or aggregate volume of 10,000 does not establish an execution. The CLI requires exact quantity; a lower-level API can permit smaller near-size analogs while keeping their actual size.

The policy receives economic feedback only for supported actions. Consequently, it learns from a selected subset of actions and regimes: exact-quantity records, liquidity and data coverage affect what can be rewarded. A high shaped reward with low matching coverage can be misleading. An unsupported action is not a proven bad trade, so it receives no invented economic penalty. Hard-constraint violations can carry an explicitly separate policy cost.

Historical analogs do not identify causal market impact. An inserted order can displace orders, reveal information or trigger responses that are absent from the record. The account is a conditional counterfactual estimate, and matched execution evidence does not turn it into realized or executable strategy returns.

## Interpreting a report

Assess these together:

- Proposed dollar actions, constrained quantities and reasons for any reduction or rejection.
- Supported analogs, unmatched actions, coverage, matching distances and source record IDs.
- Mature economic feedback, the horizon reserved for observing it and separate constraint costs.
- The independently marked account ledger, explicit execution fees, exposure and drawdown.
- Results by stock, regime, holding period and action side, including data availability.
- Sensitivity to reward horizon/weights, execution latency, costs and matching tolerances.

Keep shaped feedback separate from net portfolio estimates. The pipeline's one-basis-point execution fee reduces ledger cash and thus is already included in portfolio feedback; it is not subtracted twice. The standalone per-share `evaluate_action` diagnostic uses a configurable five-basis-point fee by default and does not trace subsequent portfolio trades; active training uses `evaluate_portfolio_action`. Spread enters recorded execution prices. Settlement, taxes, borrow, financing, dividends and all venue charges are not modeled. A reported close-based account value is a valuation assumption, not evidence that it could be liquidated at that price.

The synthetic demo validates software integration, minute/second input handling and feedback mechanics. Its paths, executions and news cannot establish learning quality, liquidity realism or future profitability.

## Validation before expanding scope

1. Use purged walk-forward evaluation with boundaries/embargoes covering lookbacks, execution windows and full reward horizons.
2. Test across market regimes and unseen securities, using the historical investable universe including delisted stocks.
3. Compare cash, constrained buy-and-hold and transparent action rules under the same execution-evidence requirements.
4. Examine reward-hacking behavior: unnecessary churn, collecting overlapping avoided-loss rewards, exploiting tiny actions or trading only well-covered events.
5. Include consistent corporate action handling and provider-specific costs; quantify uncertainty and sensitivity instead of reporting one best result.
6. Validate full depth and order lifecycle adapters before queue claims, and a structural impact model before causal impact claims.
7. Stress test gaps, halts, stale information, missing liquidity, correlated losses, distribution shifts and source failures.

Useful extensions include provider-backed history, finer action sizing, calibrated portfolio stress controls, semantic news encoding, diversified credit assignment and alternative offline policy algorithms. Continuous sizing still needs recorded execution support; expanding a dollar action does not license inventing a fill. Production broker integration requires reconciliation, monitoring and a tested kill switch.

No finite input set covers every price influence or risk. The modules expose assumptions so experiments can test and replace them as better evidence becomes available.
