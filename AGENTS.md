# Working in StockTrade

Use the existing checkout. Cloud tasks are isolated already; do not create a
Git worktree unless the user explicitly asks for one.

Run `bash scripts/bootstrap.sh` if dependencies are missing. Use `.venv/bin`
for Python, pytest, Ruff and the stocktrade CLI. Validate changes with
`.venv/bin/ruff check src tests`, `.venv/bin/pytest`, and a relevant CLI smoke test.

Keep research modules independent. Data contracts belong in schemas.py;
data ingestion must preserve UTC publication AND actual availability times.
Bar timestamps are END times. Freeze feature schemas and fit every statistic
only on the training prefix. Validation may select checkpoints; test feedback
must never train them. Every policy decision must use a strictly past window.

The active model outputs direct buy/sell/hold actions with dollar budgets,
conditioned on total funding, cash, holdings and the allowed stock universe.
Train from complete portfolio terminal wealth using policy gradients. Do not restore
price/return prediction heads, prediction losses or forecast-derived allocation.
Do not grade individual buys, sales, holds, or risk rejections. The sole
financial objective is final net equity divided by initial funding minus one,
over a declared complete period. Fees are already in actual ledger cash.
Enforce risk limits separately; incomplete execution or inadmissible episodes
are not used for learning. No invented exits or penalties. Use one terminal
wealth signal for the full trajectory. Test evaluation never updates weights.

Online sources have publication, actual availability and expiry clocks.
Historical train/evaluate never fetch today's web into past inputs. Live act
can collect ticker sources and advance only the final feature decision clock;
completed bars retain their original market timestamps. Do not prioritize
platform inputs manually; learned encoders decide information importance.

Never replace unsupported execution with invented fills. Preserve observed
trade identifiers, direction, quantity, quote context and latency. Passive
queue evidence needs attributed order lifecycle data. Historical analogs are
observational estimates; do not label them causal market-impact results or
executable strategy returns. Report unmatched intentions and selection bias.

Do not commit market datasets, checkpoints, credentials or generated reports.
Keep them under ignored data/ or runs/. Test fixtures are explicitly synthetic.
No broker order endpoint or live trading is implemented.

Evaluation diagnostics and interactive reports are separate from learning.
Historical VaR/ES, stress scenarios, technical indicators, projected risk and
estimated costs must not enter the objective, alter hard constraints or rank
checkpoints. Include the existing portfolio's risk for holds; distinguish a
conditional feasible proposal from observed analog accounting. Preserve
unavailable values with reasons and do not annualize intraday rows as days.
Report regeneration must preserve original audits and never invent fills.
