"""Synthetic multi-stock records for software validation, not investment evidence."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def generate_demo(output, *, count=240, frequency_seconds=60, seed=7,
                  symbols=("NVDA", "MSFT", "AAPL")):
    if count < 100 or not isinstance(frequency_seconds, int) or frequency_seconds < 1:
        raise ValueError("Demo needs >=100 bars and positive whole-second spacing")
    symbols = tuple(symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Demo stock symbols must be distinct and nonempty")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    origin = datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)
    prices = {symbol: 100.0 + 40 * j for j, symbol in enumerate(symbols)}
    bars, events = [], []
    trade_fields = ["timestamp", "trade_id", "symbol", "side", "quantity", "price",
                    "bid", "ask", "spread", "volatility"]
    with (output / "trades.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trade_fields)
        writer.writeheader()
        for index in range(count):
            timestamp = origin + timedelta(seconds=(index + 1) * frequency_seconds)
            common_return = float(rng.normal(0, .0005))
            for number, symbol in enumerate(symbols):
                previous = prices[symbol]
                drift = .0004 * np.sin(index / 25 + number)
                price = previous * np.exp(common_return + rng.normal(drift, .0013))
                prices[symbol] = price
                spread = price * .0002
                bid, ask = price - spread / 2, price + spread / 2
                bars.append(dict(timestamp=timestamp.isoformat(), symbol=symbol, open=previous,
                    high=max(price, previous) * 1.0005, low=min(price, previous) * .9995,
                    close=price, volume=6000, bid=bid, ask=ask, bid_size=700, ask_size=650))
                # Explicitly fabricated records cover software sizing branches. They
                # never establish that these executions existed on a real exchange.
                for quantity in range(1, 61):
                    for side in ("buy", "sell"):
                        writer.writerow(dict(
                            timestamp=(timestamp + timedelta(milliseconds=150 + quantity)).isoformat(),
                            trade_id=f"SYNTHETIC-{index}-{symbol}-{quantity}-{side}", symbol=symbol,
                            side=side, quantity=quantity, price=ask if side == "buy" else bid,
                            bid=bid, ask=ask, spread=spread, volatility=.0015))
                if index % 20 == 0:
                    events.extend([
                        dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(),
                             source="synthetic", kind="news", symbol=symbol, values={},
                             text=f"Synthetic demand and earnings update for {symbol}"),
                        dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(),
                             source="synthetic", kind="orderflow", symbol=symbol,
                             values={"bid_depth_5": 2000, "ask_depth_5": 1800,
                                     "cancellation_ratio": float(rng.uniform(0, .3))}),
                    ])
                if index % 100 == 0:
                    events.append(dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(),
                        source="synthetic", kind="fundamentals", symbol=symbol,
                        values={"revenue": 1e9 * (1 + number / 10), "debt_to_equity": .3}))
            if index % 20 == 0:
                events.append(dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(),
                    source="synthetic", kind="macro", values={"fed_rate": .04, "market_return": common_return}))
    with (output / "bars.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bars[0]))
        writer.writeheader()
        writer.writerows(bars)
    with (output / "events.jsonl").open("w") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    (output / "provenance.json").write_text(json.dumps(dict(synthetic=True, seed=seed,
        frequency_seconds=frequency_seconds, bars_per_stock=count, symbols=symbols,
        purpose="Software checks only; no historical performance evidence"), indent=2) + "\n")
    return output
