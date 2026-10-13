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
Train from mature evaluator feedback using policy gradients. Do not restore
price/return prediction heads, prediction losses or forecast-derived allocation.
Buy feedback compares with keeping cash; sell feedback compares with retaining
the same existing shares at fixed future horizons. Never shorten the horizon
at partition boundaries or select a flattering hindsight endpoint. Keep shaped
action rewards separate from investment profit. Unmatched executions receive
no economic reward. Known feasibility costs must be labelled separately.

Never replace unsupported execution with invented fills. Preserve observed
trade identifiers, direction, quantity, quote context and latency. Passive
queue evidence needs attributed order lifecycle data. Historical analogs are
observational estimates; do not label them causal market-impact results or
executable strategy returns. Report unmatched intentions and selection bias.

Do not commit market datasets, checkpoints, credentials or generated reports.
Keep them under ignored data/ or runs/. Test fixtures are explicitly synthetic.
No broker order endpoint or live trading is implemented.
