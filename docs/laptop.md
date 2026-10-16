# Running research on your laptop

Use Python **3.12**, Git and a fresh virtual environment. The cloud environment
has been tested on Linux x86_64; the Windows/macOS commands below are local setup
instructions, not a claim that those operating systems were tested here.
Start with a few stocks and a short interval, and measure memory and runtime
before increasing the experiment. There is no fixed RAM requirement independent
of dataset size: execution records can be much larger than the price history.

## Install

Clone the source, or pull changes if you already have the checkout:

```bash
git clone https://github.com/Leaf-Ye-103/StockTrade.git
cd StockTrade
```

On **macOS or Linux**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On **Windows PowerShell**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in the following commands; changing your execution policy is unnecessary.
On macOS/Linux, `.venv/bin/python` also works without activation.

With the environment selected, install the application:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" tzdata
python -m pip check
python -m stocktrade --help
```

`tzdata` supplies the IANA timezones that Windows commonly lacks. The package
uses `America/New_York` for trading dates. To avoid unnecessary GPU packages,
Windows/Linux x86_64 users can install `"torch>=2.5,<3"` from
`https://download.pytorch.org/whl/cpu` **before** the editable install. On macOS,
use the ordinary platform wheels from PyPI. Confirm that PyTorch supplies a
compatible wheel for your architecture; this project's `torch>=2.5` requirement
can be a blocker on older Intel Macs. Do not lower the requirement or disable
certificate verification to make an installation appear successful.

`scripts/bootstrap.sh` and `requirements-dev.lock` reproduce the validated Linux
CPU environment; they are not a portable Windows/macOS installation recipe.
The portable install above resolves platform-compatible dependency versions.
Record `python -m pip freeze` with an experiment if you need to reproduce its
exact local environment.

## Check that the software runs

```bash
python -m pytest
python -m stocktrade demo --output runs/local-smoke --epochs 2 --symbols AAPL MSFT NVDA AMZN GOOGL --funding 100000
```

Open `runs/local-smoke/evaluation/dashboard.html` in your browser. It includes
price/decision charts, risk estimates, technical indicators and portfolio metrics.
No web server is required for the standalone HTML.

The demo uses **synthetic** prices and executions, even when the stock symbols
are real company names. It does not download their market history. The CLI demo
always generates 240 bars per stock; more epochs do not add historical data.

For a longer synthetic software test, save this as `local_synthetic.py` in your
checkout and run `python local_synthetic.py`:

```python
from stocktrade.demo import generate_demo
from stocktrade.pipeline import evaluate, train

symbols = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
data = generate_demo("data/local-synthetic", count=500, symbols=symbols)
train(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
      "runs/local-synthetic/model", symbols=symbols, epochs=2)
evaluate(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
         "runs/local-synthetic/model/model.pt", "runs/local-synthetic/evaluation")
```

This generator creates 120 fabricated trade records per stock per bar and uses
continuous timestamps without an exchange calendar. At the settings above it
creates 300,000 trade rows. Increase it cautiously; it is not a substitute for
longer real historical observations. Use new output paths for separate runs.

## Download real Alpaca inputs

Cloud credentials do not automatically appear on your laptop. Set
`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` locally using your password manager
or local environment configuration. A `.env` file is **not automatically loaded**
by this application. Never copy cloud proxy placeholders as actual credentials.

If you prefer not to configure shell variables, the following script prompts
without displaying either credential. Save it as `fetch_local.py`, change the
symbols/dates/output directory as needed, and run `python fetch_local.py`:

```python
import os
from getpass import getpass
from stocktrade.platforms import fetch_alpaca_bars

for name in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
    if not os.environ.get(name):
        os.environ[name] = getpass(f"{name}: ")

result = fetch_alpaca_bars(
    ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    "2025-01-01T00:00:00Z", "2025-02-01T00:00:00Z",
    "data/alpaca-2025-01", feed="sip",
)
print(result)  # Sanitized provenance and counts, not credentials.
```

SIP needs the appropriate entitlement; IEX is a different, venue-specific feed.
Use an output directory that does not already contain imported files. For large
ranges, download separate monthly or smaller chunks. Each import is bounded to
100 stocks and 100 pages of 10,000 bars; a request exceeding that page limit fails
rather than returning a silently truncated training dataset.

The importer does **not** merge chunks, filter sessions, align stocks, attach
quotes, or supply normalized execution records. Preserve provenance and audit
duplicate/missing observations when preparing a dataset. Alpaca documents stock
history since 2016; changing the requested start to 2000 does not create older
history. A historical news download also cannot establish past delivery times.

## Train and evaluate a prepared historical dataset

These commands require files you have already normalized using
[the data contracts](data.md):

| File | Required content |
| --- | --- |
| `data/research/bars.csv` | Completed OHLCV bars for the chosen universe, identical stock clocks, and observed quote context needed for risk admission. |
| `data/research/trades.csv` | Actual execution identifiers, initiating side, quantities, prices, pre-execution quotes and compatible spread/volatility context. |
| `data/research/events.jsonl` | Optional information with defensible historical publication and availability timestamps. |

**Raw Alpaca bars alone cannot run this historical training workflow.** The
repository currently imports Alpaca bars/news, but has no complete automatic
trade/quote normalization adapter. Generic Alpaca trade responses also do not
supply an authoritative initiating aggressor side. Do not invent trades or use
bar closes as fills to get past missing evidence.

Once execution-ready files exist, these one-line commands work in both an
activated POSIX shell and PowerShell:

```bash
python -m stocktrade train --bars data/research/bars.csv --trades data/research/trades.csv --symbols AAPL MSFT NVDA AMZN GOOGL --funding 100000 --lookback 16 --episode-steps 128 --epochs 3 --seed 7 --train-fraction 0.70 --validation-fraction 0.15 --output runs/local-5stocks
python -m stocktrade evaluate --bars data/research/bars.csv --trades data/research/trades.csv --checkpoint runs/local-5stocks/model.pt --output runs/local-5stocks/test
```

Add `--events data/research/events.jsonl` to **both** commands only if you have
that historical archive. Evaluation uses the final 15% chronologically and keeps
the model frozen. Supply the same full historical dataset, including the verified
training/validation prefix; passing only a cropped test file fails provenance
checks. Open `runs/local-5stocks/test/dashboard.html` for the result. Evaluation
indicators and risk estimates do not guide training or checkpoint selection.

Adding symbols requires a newly trained checkpoint. `train` starts a new model;
there is no resume-training CLI. Keep `model.pt` and `model.info.json` together.
Inspect `training.json` and the execution coverage before trusting a run: missing
matches or insufficient complete risk-admissible trajectories can stop training.

## Increase the scope gradually

Increase data length by changing the **input dataset's date range**, and increase
the universe by changing its data and `--symbols`. `--epochs` controls repeated
training passes; `--lookback` controls the past input window; `--episode-steps`
controls a bounded training trajectory. None of those options downloads more data.

Start with a few stocks over several sessions to check execution coverage, then
try 5–10 stocks over a month before attempting longer experiments. Keep episodes
bounded initially. The current implementation uses CPU processing and loads full
histories, execution records, features and audits into memory; buying a GPU alone
does not change that implementation.

A 50-stock, 26-year regular-session minute dataset is roughly 128 million bar
rows before execution data. It needs storage/loading and accounting improvements
before it is a practical laptop run. Current strict alignment would discard early
history for later IPOs if you naively intersected every stock's timestamps. Missing
asset support and split/dividend/delisting accounting must also be addressed before
claiming reliable multi-decade portfolio returns.

Data, trained weights and reports under `data/` and `runs/` are ignored by Git.
Cloning this repository does not download the cloud examples, trained checkpoints
or private datasets. Recreate them locally, or copy the required files separately
while preserving their provenance and licensing.
