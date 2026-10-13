"""Chronological orchestration with audit records and fail-closed execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import build_features, load_bars_csv, load_events_jsonl
from .execution import MatchingConfig, OrderIntent, PortfolioLedger, load_trades_csv, match_order
from .model import TrainingConfig, TrainingResult, train_model
from .risk import AccountState, MarketState, RiskConfig, size_position


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def file_signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_signature(frame, end):
    """Detect changed point-in-time history, order, prices or feature schema."""
    digest = hashlib.sha256()
    digest.update(json.dumps(frame.feature_names).encode())
    digest.update(frame.symbol.encode())
    for timestamp in frame.timestamps[:end]:
        digest.update(timestamp.isoformat().encode())
    for array in (frame.values[:end], frame.observed[:end], frame.prices[:end]):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def load_frame(bars_path, events_path=None, *, feature_names=None):
    bars = sorted(load_bars_csv(bars_path), key=lambda bar: bar.timestamp)
    events = load_events_jsonl(events_path) if events_path else []
    return bars, build_features(bars, events, feature_names=feature_names)


def train(bars_path, events_path, output, *, epochs=3, seed=7, lookback=16,
          train_fraction=0.7, validation_fraction=0.15):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Train and validation fractions must leave a nonempty chronological test interval")
    bars = sorted(load_bars_csv(bars_path), key=lambda bar: bar.timestamp)
    events = load_events_jsonl(events_path) if events_path else []
    train_end = int(len(bars) * train_fraction)
    validation_end = int(len(bars) * (train_fraction + validation_fraction))
    if train_end < lookback + 1:
        raise ValueError("Insufficient bars for the configured training window")
    prefix_events = [event for event in events
                     if max(event.published_at, event.available_at) <= bars[train_end - 1].timestamp]
    schema = build_features(bars[:train_end], prefix_events).feature_names
    frame = build_features(bars, events, feature_names=schema)
    result = train_model(frame, train_end, validation_end,
                         TrainingConfig(epochs=epochs, seed=seed, lookback=lookback))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.save(output / "model.pt")
    metadata = dict(symbol=frame.symbol, feature_names=frame.feature_names,
                    train_end=train_end, validation_end=validation_end, total_bars=len(frame.timestamps),
                    validation_context_sha256=frame_signature(frame, validation_end), seed=seed,
                    train_start=frame.timestamps[0], train_end_time=frame.timestamps[train_end - 1],
                    validation_end_time=frame.timestamps[validation_end - 1],
                    timestamp_convention="bar END / observed availability", frequency="from input timestamps",
                    checkpoint_sha256=file_signature(output / "model.pt"),
                    purpose="Research checkpoint; not evidence of profitable trading")
    write_json(output / "model.info.json", metadata)
    write_json(output / "training.json", dict(history=result.history, metadata=metadata))
    return metadata


def evaluate(bars_path, events_path, trades_path, checkpoint, output, *, cash=100000.0):
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _evaluate(bars_path, events_path, trades_path, checkpoint, output, cash=cash)
    finally:
        torch.set_num_threads(previous_threads)


def _evaluate(bars_path, events_path, trades_path, checkpoint, output, *, cash=100000.0):
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError("Initial cash must be positive and finite")
    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_name("model.info.json").read_text())
    if file_signature(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint and its metadata sidecar are not the saved pair")
    bars, frame = load_frame(bars_path, events_path, feature_names=metadata["feature_names"])
    validation_end = metadata["validation_end"]
    if len(frame.timestamps) <= validation_end:
        raise ValueError("Evaluation requires bars after the checkpoint's validation interval")
    if frame_signature(frame, validation_end) != metadata["validation_context_sha256"]:
        raise ValueError("Checkpoint data/schema differs from supplied point-in-time training history")
    predictor = TrainingResult.load(checkpoint)
    if (predictor.splits["train"][1] != metadata["train_end"]
            or predictor.splits["validation"] != (metadata["train_end"], validation_end)
            or predictor.symbol != metadata["symbol"]
            or predictor.scaler.feature_names != metadata["feature_names"]
            or predictor.training_last_timestamp != metadata["train_end_time"]
            or predictor.validation_last_timestamp != metadata["validation_end_time"]):
        raise ValueError("Checkpoint training/validation boundaries disagree with metadata")
    prints = sorted(load_trades_csv(trades_path), key=lambda record: (record.timestamp, record.trade_id))
    ledger = PortfolioLedger(initial_cash=cash, fee_bps=1.0)
    risk = RiskConfig()
    # Exact size is intentional: actual analog quantity cannot enlarge a risk-capped order.
    matching = MatchingConfig(size_tolerance=0.0)
    peak, day_start, current_day = cash, cash, None
    decision_records, forecasts, reasons = [], [], Counter()
    attempted = matched = 0
    cursor = 0
    for index in range(validation_end - 1, len(bars) - 1):
        bar = bars[index]
        next_timestamp = bars[index + 1].timestamp
        ledger.mark(bar.timestamp, frame.symbol, bar.close)
        peak = max(peak, ledger.equity)
        if current_day != bar.timestamp.date():
            current_day, day_start = bar.timestamp.date(), ledger.equity
        forecast = predictor.predict(frame, index=index)
        actual_return = float(frame.prices[index + 1] / frame.prices[index] - 1)
        forecasts.append(dict(timestamp=bar.timestamp, **asdict(forecast), next_return=actual_return))
        has_quotes = bar.bid is not None and bar.ask is not None
        spread = (bar.ask - bar.bid) if has_quotes else None
        # Last COMPLETED bar's volume is a liquidity proxy, not known future liquidity.
        active_features = predictor.scaler.counts > 0
        quality = float(frame.observed[index, active_features].mean()) if active_features.any() else 0.0
        market = MarketState(frame.symbol, bar.close, bar.volume if has_quotes else 0.0,
                             spread_bps=spread / bar.close * 10000 if has_quotes else 0.0,
                             data_quality=quality, bid=bar.bid, ask=bar.ask)
        account = AccountState(equity=ledger.equity, cash=ledger.cash,
                               position_quantity=ledger.inventory.get(frame.symbol, 0),
                               peak_equity=peak, day_start_equity=day_start)
        sizing = size_position(forecast, market, account, risk)
        record = dict(timestamp=bar.timestamp, forecast=asdict(forecast), sizing=asdict(sizing))
        reasons.update(sizing.reasons)
        if sizing.quantity:
            attempted += 1
            recent = np.diff(np.log(frame.prices[max(0, index - 20):index + 1]))
            volatility = float(np.std(recent)) if len(recent) >= 2 else None
            intent = OrderIntent(bar.timestamp, frame.symbol, sizing.side.lower(), sizing.quantity,
                                 bar.ask if sizing.side == "BUY" else bar.bid, bar.close,
                                 spread=spread, volatility=volatility)
            # No execution selected after the next decision is credited to today's account.
            while cursor < len(prints) and prints[cursor].timestamp < bar.timestamp:
                cursor += 1
            end = cursor
            while end < len(prints) and prints[end].timestamp < next_timestamp:
                end += 1
            match = match_order(intent, prints[cursor:end], matching,
                                used_trade_ids=ledger.used_trade_ids)
            outcome = ledger.apply_match(intent, match)
            record.update(intent=asdict(intent), analog=asdict(match), accounting=asdict(outcome))
            matched += outcome.status == "matched"
        decision_records.append(record)
    ledger.mark(frame.timestamps[-1], frame.symbol, float(frame.prices[-1]))
    result = asdict(ledger.result())
    errors = [row["expected_return"] - row["next_return"] for row in forecasts]
    report = dict(
        interpretation="Observed-trade analog ledger estimate, not executed fills or causal market impact",
        incomplete_execution=matched < attempted,
        source_is_synthetic=bool(prints) and all(record.trade_id.startswith("SYNTHETIC-") for record in prints),
        decisions=len(decision_records), proposed_orders=attempted, matched_orders=matched,
        matching_coverage=matched / attempted if attempted else None,
        risk_reasons=dict(reasons), heldout_return_mae=float(np.mean(np.abs(errors))),
        initial_cash=cash, final_equity=ledger.equity, ledger_return=ledger.equity / cash - 1,
        execution=result,
        provenance=dict(checkpoint_sha256=metadata["checkpoint_sha256"],
                        bars_sha256=file_signature(bars_path),
                        events_sha256=file_signature(events_path) if events_path else None,
                        trades_sha256=file_signature(trades_path),
                        training_config=asdict(predictor.config), risk_config=asdict(risk),
                        matching_config=asdict(matching)),
        limitations=["Unmatched intentions cannot be assigned historical fills or investment outcomes",
                     "Analog selection is retrospective and may have coverage/selection bias",
                     "Quotes/prints do not identify the causal queue or impact of a hypothetical order",
                     "Single-symbol sizing; no portfolio-wide covariance optimization",
                     "Past-bar volume is only a participation proxy; no guarantee of future liquidity"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    with (output / "decisions.jsonl").open("w") as handle:
        for record in decision_records:
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")
    write_json(output / "forecasts.json", forecasts)
    return report
