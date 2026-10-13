"""Generate SYNTHETIC fixtures to test software, never historical performance."""

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def generate_demo(output: str | Path, *, count: int = 420, frequency_seconds: int = 60, seed: int = 7):
    if count < 100 or frequency_seconds < 1:
        raise ValueError("Demo needs at least 100 bars and positive whole-second spacing")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    origin = datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)
    price = 100.0
    bars, trades, events = [], [], []
    for i in range(count):
        timestamp = origin + timedelta(seconds=(i + 1) * frequency_seconds)
        previous = price
        change = float(rng.normal(0.00015 + 0.00035 * np.sin(i / 30), 0.0015))
        price *= np.exp(change)
        spread = price * 0.0002
        bid, ask = price - spread / 2, price + spread / 2
        bars.append(dict(timestamp=timestamp.isoformat(), symbol="DEMO", open=previous,
                         high=max(price, previous) * 1.0005, low=min(price, previous) * 0.9995,
                         close=price, volume=int(rng.integers(3000, 12000)), bid=bid, ask=ask,
                         bid_size=int(rng.integers(200, 1200)), ask_size=int(rng.integers(200, 1200))))
        # These are fabricated test records, not observations of a real exchange.
        # They exercise matching independently of whatever sizes a model chooses.
        for j, qty in enumerate((10, 25, 50, 100, 250)):
            for side in ("buy", "sell"):
                trades.append(dict(timestamp=(timestamp + timedelta(milliseconds=200 + j * 20)).isoformat(),
                                   trade_id=f"SYNTHETIC-{i}-{j}-{side}", symbol="DEMO", side=side,
                                   quantity=qty, price=ask if side == "buy" else bid,
                                   bid=bid, ask=ask, spread=spread, volatility=0.0015))
        if i % 20 == 0:
            events.extend([
                dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(), source="synthetic",
                     kind="macro", values={"fed_rate": 0.04, "market_return": float(rng.normal(0, .001))}),
                dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(), source="synthetic",
                     kind="news", symbol="DEMO", values={}, text="Synthetic demand and earnings update"),
                dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(), source="synthetic",
                     kind="orderflow", symbol="DEMO", values={"bid_depth_5": bars[-1]["bid_size"] * 2,
                     "ask_depth_5": bars[-1]["ask_size"] * 2, "cancellation_ratio": float(rng.uniform(0, .3))}),
            ])
        if i % 100 == 0:
            events.append(dict(published_at=timestamp.isoformat(), available_at=timestamp.isoformat(),
                               source="synthetic", kind="fundamentals", symbol="DEMO",
                               values={"revenue": 1e9 * (1 + i / 1000), "debt_to_equity": 0.3}))
    for name, rows in (("bars.csv", bars), ("trades.csv", trades)):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "events.jsonl").open("w") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
    (output / "provenance.json").write_text(json.dumps({"synthetic": True, "seed": seed,
        "frequency_seconds": frequency_seconds, "bars": count,
        "purpose": "Software validation only; no historical performance evidence"}, indent=2) + "\n")
    return output
