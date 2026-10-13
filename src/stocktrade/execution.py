"""Auditable historical execution analogs, never simulated exchange fills.

Selection uses observed execution events in a fixed forward observation window.
It is retrospective observational matching, not an executable backtest or an
estimate of the causal effect of placing a new order. Unmatched orders abstain.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _valid_time(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _nonnegative(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class OrderIntent:
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    limit_price: float
    reference_price: float
    mode: str = "aggressive"
    spread: float | None = None
    volatility: float | None = None
    latency_ms: int = 100
    queue_ahead: float | None = None

    def __post_init__(self) -> None:
        if not _valid_time(self.timestamp):
            raise ValueError("Order timestamps must be timezone-aware")
        if not self.symbol or self.side not in {"buy", "sell"}:
            raise ValueError("Order requires a symbol and buy/sell side")
        if self.mode not in {"aggressive", "passive"}:
            raise ValueError("mode must be aggressive or passive")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("Order quantity must be a positive integer")
        if not _positive(self.limit_price) or not _positive(self.reference_price):
            raise ValueError("Order prices must be finite and positive")
        if not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a nonnegative integer")
        for name in ("spread", "volatility", "queue_ahead"):
            value = getattr(self, name)
            if value is not None and not _nonnegative(value):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class TradePrint:
    """One recorded execution; ``side`` is its initiating aggressor's side.

    Quotes/regime fields must describe context observed before this execution.
    ``impact_bps`` is a subsequent market movement diagnostic, never a feature
    used to select an analog. Passive queue evidence requires vendor-supported
    order-level attribution; queue size inferred from a quote is insufficient.
    """

    timestamp: datetime
    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    volatility: float | None = None
    queue_ahead: float | None = None
    queue_depletion: float | None = None
    queue_observed_from: datetime | None = None
    queue_evidence_ids: tuple[str, ...] = ()
    queue_priority_verified: bool = False
    impact_bps: float | None = None

    def __post_init__(self) -> None:
        if not _valid_time(self.timestamp) or not self.trade_id or not self.symbol:
            raise ValueError("Trade requires an aware timestamp, unique ID and symbol")
        if self.side not in {"buy", "sell"}:
            raise ValueError("Trade side must explicitly identify buy/sell aggressor")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("Trade quantity must be a positive integer")
        if not _positive(self.price):
            raise ValueError("Trade price must be finite and positive")
        for name in ("bid", "ask", "spread", "volatility", "queue_ahead", "queue_depletion"):
            value = getattr(self, name)
            if value is not None and not _nonnegative(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.queue_observed_from is not None and not _valid_time(self.queue_observed_from):
            raise ValueError("Queue timestamps must be timezone-aware")
        if self.impact_bps is not None and not math.isfinite(self.impact_bps):
            raise ValueError("impact_bps must be finite")
        if not isinstance(self.queue_evidence_ids, tuple):
            raise ValueError("queue_evidence_ids must be an immutable tuple")


@dataclass(frozen=True)
class MatchingConfig:
    horizon_seconds: float = 60.0
    size_tolerance: float = 0.1
    spread_relative_tolerance: float = 0.5
    volatility_relative_tolerance: float = 0.5
    queue_relative_tolerance: float = 0.25
    require_spread: bool = True
    require_volatility: bool = True

    def __post_init__(self) -> None:
        if not _positive(self.horizon_seconds):
            raise ValueError("horizon_seconds must be finite and positive")
        if not math.isfinite(self.size_tolerance) or not 0 <= self.size_tolerance < 1:
            raise ValueError("size_tolerance must be in [0, 1)")
        for name in ("spread_relative_tolerance", "volatility_relative_tolerance", "queue_relative_tolerance"):
            if not _nonnegative(getattr(self, name)):
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class MatchResult:
    status: str
    reason: str
    requested_quantity: int
    source_trade_ids: tuple[str, ...] = ()
    estimated_price: float | None = None
    estimated_quantity: int = 0
    event_timestamp: datetime | None = None
    observed_impact_bps: float | None = None
    quantity_difference: int = 0
    interpretation: str = "observational execution analog; no actual or hypothetical fill is established"

    def __post_init__(self) -> None:
        if self.status not in {"matched", "unmatched"}:
            raise ValueError("Match status must be matched or unmatched")
        if isinstance(self.requested_quantity, bool) or not isinstance(self.requested_quantity, int) or self.requested_quantity <= 0:
            raise ValueError("Requested quantity must be a positive integer")
        if isinstance(self.estimated_quantity, bool) or not isinstance(self.estimated_quantity, int) or self.estimated_quantity < 0:
            raise ValueError("Observed quantity must be a nonnegative integer")
        if not isinstance(self.source_trade_ids, tuple) or len(set(self.source_trade_ids)) != len(self.source_trade_ids):
            raise ValueError("Source IDs must be an immutable tuple of unique identifiers")
        if self.status == "matched":
            if not _valid_time(self.event_timestamp) or not _positive(self.estimated_price) or not self.source_trade_ids:
                raise ValueError("Matched analog requires observed event time, price and source IDs")
            if not 0 < self.estimated_quantity <= self.requested_quantity:
                raise ValueError("Observed quantity must not exceed the request")
            if self.quantity_difference != self.estimated_quantity - self.requested_quantity:
                raise ValueError("Quantity discrepancy must preserve actual recorded quantity")
        elif self.estimated_quantity or self.estimated_price is not None or self.event_timestamp is not None or self.source_trade_ids:
            raise ValueError("Unmatched results cannot contain invented execution data")


def _close(value: float, reference: float, relative_tolerance: float) -> bool:
    return abs(value - reference) <= max(abs(reference) * relative_tolerance, 1e-12)


def _queue_evidence(
    intent: OrderIntent,
    trade: TradePrint,
    sources: Mapping[str, TradePrint],
    arrival: datetime,
    used_ids: set[str],
    config: MatchingConfig,
) -> str | None:
    if intent.queue_ahead is None:
        return "missing_intent_queue_context"
    if not trade.queue_priority_verified:
        return "unsupported_passive_queue_priority"
    if trade.queue_ahead is None or trade.queue_depletion is None or trade.queue_observed_from is None:
        return "missing_passive_queue_progression"
    if trade.queue_observed_from > arrival:
        return "queue_observation_started_after_arrival"
    if not _close(trade.queue_ahead, intent.queue_ahead, config.queue_relative_tolerance):
        return "incompatible_queue_position"
    if trade.queue_depletion < trade.queue_ahead:
        return "queue_not_depleted"
    if not math.isclose(trade.price, intent.limit_price, rel_tol=0, abs_tol=1e-8):
        return "passive_price_level_mismatch"
    if len(set(trade.queue_evidence_ids)) != len(trade.queue_evidence_ids):
        return "duplicate_queue_evidence"
    executed_ahead = 0
    for evidence_id in trade.queue_evidence_ids:
        evidence = sources.get(evidence_id)
        if evidence is None or evidence_id == trade.trade_id or evidence_id in used_ids:
            return "missing_or_reused_queue_evidence"
        if not (arrival <= evidence.timestamp < trade.timestamp):
            return "invalid_queue_evidence_timing"
        if evidence.symbol != intent.symbol or evidence.side == intent.side:
            return "incompatible_queue_evidence"
        if not math.isclose(evidence.price, intent.limit_price, rel_tol=0, abs_tol=1e-8):
            return "queue_evidence_price_level_mismatch"
        executed_ahead += evidence.quantity
    if executed_ahead < trade.queue_depletion:
        return "unsubstantiated_queue_depletion"
    if trade.queue_ahead > 0 and not trade.queue_evidence_ids:
        return "missing_passive_queue_progression"
    return None


def match_order(
    intent: OrderIntent,
    prints: Iterable[TradePrint],
    config: MatchingConfig | None = None,
    *,
    used_trade_ids: Iterable[str] = (),
) -> MatchResult:
    """Select the earliest compatible recorded event without ranking its outcome.

    Actual recorded quantity is returned unchanged, even if a near-size analog
    differs from the request. No aggregation, interpolation, artificial fill,
    reuse, or optimization on future returns is permitted. The caller supplies
    only context available at decision time in ``intent``. Event matching itself
    is retrospective; it must never feed observations back into past decisions.
    """
    config = config or MatchingConfig()
    used_ids = set(used_trade_ids)
    events = list(prints)
    sources = {trade.trade_id: trade for trade in events}
    if len(sources) != len(events):
        raise ValueError("trade_id must be unique across the complete observation set")
    for name, required in (("spread", config.require_spread), ("volatility", config.require_volatility)):
        if required and getattr(intent, name) is None:
            return MatchResult("unmatched", f"missing_intent_{name}_context", intent.quantity)
    if intent.mode == "passive" and intent.queue_ahead is None:
        return MatchResult("unmatched", "missing_intent_queue_context", intent.quantity)
    arrival = intent.timestamp + timedelta(milliseconds=intent.latency_ms)
    horizon = arrival + timedelta(seconds=config.horizon_seconds)
    rejection_reasons: set[str] = set()
    expected_side = intent.side if intent.mode == "aggressive" else ("sell" if intent.side == "buy" else "buy")
    for trade in sorted(events, key=lambda value: (value.timestamp, value.trade_id)):
        if trade.symbol != intent.symbol:
            continue
        if trade.trade_id in used_ids:
            rejection_reasons.add("source_already_used")
            continue
        if not arrival <= trade.timestamp <= horizon:
            rejection_reasons.add("outside_arrival_window")
            continue
        if trade.side != expected_side:
            rejection_reasons.add("incompatible_aggressor_direction")
            continue
        if trade.quantity > intent.quantity or trade.quantity < intent.quantity * (1 - config.size_tolerance):
            rejection_reasons.add("incompatible_size")
            continue
        if (intent.side == "buy" and trade.price > intent.limit_price) or (intent.side == "sell" and trade.price < intent.limit_price):
            rejection_reasons.add("limit_price_incompatible")
            continue
        if not _positive(trade.bid) or not _positive(trade.ask) or trade.ask < trade.bid:
            rejection_reasons.add("missing_or_invalid_quote_context")
            continue
        if (trade.side == "buy" and trade.price < trade.ask) or (trade.side == "sell" and trade.price > trade.bid):
            rejection_reasons.add("trade_not_marketable_at_observed_quote")
            continue
        rejected = False
        for name, required, tolerance in (
            ("spread", config.require_spread, config.spread_relative_tolerance),
            ("volatility", config.require_volatility, config.volatility_relative_tolerance),
        ):
            reference = getattr(intent, name)
            observed = getattr(trade, name)
            if required and observed is None:
                rejection_reasons.add(f"missing_trade_{name}_context")
                rejected = True
            elif reference is not None and observed is not None and not _close(observed, reference, tolerance):
                rejection_reasons.add(f"incompatible_{name}_regime")
                rejected = True
        if rejected:
            continue
        if intent.mode == "passive":
            reason = _queue_evidence(intent, trade, sources, arrival, used_ids, config)
            if reason is not None:
                rejection_reasons.add(reason)
                continue
        ids = (*trade.queue_evidence_ids, trade.trade_id) if intent.mode == "passive" else (trade.trade_id,)
        return MatchResult(
            "matched", "earliest_compatible_observed_execution", intent.quantity,
            ids, trade.price, trade.quantity, trade.timestamp, trade.impact_bps,
            trade.quantity - intent.quantity,
        )
    reason = ";".join(sorted(rejection_reasons)) or "no_observed_candidate_for_symbol"
    return MatchResult("unmatched", reason, intent.quantity)


@dataclass(frozen=True)
class ReplayRecord:
    intent: OrderIntent
    match: MatchResult
    status: str
    reason: str
    cash_after: float
    quantity_after: int
    fee: float = 0.0


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    gross_exposure: float


@dataclass(frozen=True)
class ReplayResult:
    initial_cash: float
    final_cash: float
    inventory: Mapping[str, int]
    final_equity: float
    matched_orders: int
    unmatched_orders: int
    rejected_orders: int
    coverage: float
    total_fees: float
    gross_notional: float
    estimated_slippage_cost: float
    max_drawdown: float
    gross_exposure: float
    source_trade_ids: tuple[str, ...]
    records: tuple[ReplayRecord, ...]
    equity_curve: tuple[EquityPoint, ...]
    interpretation: str = "observational analog accounting; not an executable strategy return or causal impact estimate"


class PortfolioLedger:
    """Long-only accounting of accepted analogs with audited source exclusions.

    Apply analogs in nondecreasing event-time order. A sequential strategy must
    wait until an observation becomes available before consuming its accounting
    result; otherwise it leaks future execution information into decisions.
    Cash is an accounting balance, not a broker's settled-cash balance. Settlement
    calendars, exchange fees, taxes and borrow are not inferred from trade prints.
    """

    def __init__(self, initial_cash: float, *, fee_bps: float = 0.0, per_trade_fee: float = 0.0) -> None:
        if not _positive(initial_cash) or not _nonnegative(fee_bps) or not _nonnegative(per_trade_fee):
            raise ValueError("initial_cash must be positive; fees must be nonnegative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_bps = fee_bps
        self.per_trade_fee = per_trade_fee
        self.inventory: dict[str, int] = {}
        self.marks: dict[str, float] = {}
        self.used_trade_ids: set[str] = set()
        self.records: list[ReplayRecord] = []
        self.equity_curve: list[EquityPoint] = []
        self.total_fees = 0.0
        self.gross_notional = 0.0
        self.estimated_slippage_cost = 0.0
        self._last_event_time: datetime | None = None

    @property
    def gross_exposure(self) -> float:
        return sum(quantity * self.marks[symbol] for symbol, quantity in self.inventory.items() if quantity)

    @property
    def positions(self) -> dict[str, int]:
        """Read a copy of the long-only inventory for decision-time risk checks."""
        return dict(self.inventory)

    @property
    def equity(self) -> float:
        return self.cash + self.gross_exposure

    def mark(self, timestamp: datetime, symbol: str, price: float) -> None:
        if not _valid_time(timestamp) or not _positive(price):
            raise ValueError("mark requires an aware timestamp and positive finite price")
        if self._last_event_time is not None and timestamp < self._last_event_time:
            raise ValueError("Ledger events must be processed in chronological order")
        self.marks[symbol] = price
        self._last_event_time = timestamp
        self.equity_curve.append(EquityPoint(timestamp, self.equity, self.gross_exposure))

    def apply_match(self, intent: OrderIntent, result: MatchResult) -> ReplayRecord:
        quantity_after = self.inventory.get(intent.symbol, 0)
        reason = result.reason
        status = result.status
        fee = 0.0
        if result.status == "matched":
            if not _valid_time(result.event_timestamp) or not _positive(result.estimated_price) or result.estimated_quantity <= 0 or not result.source_trade_ids:
                raise ValueError("Matched results require event time, positive quantity, price and source IDs")
            if len(set(result.source_trade_ids)) != len(result.source_trade_ids):
                raise ValueError("Matched source IDs must be unique")
            if result.requested_quantity != intent.quantity or result.estimated_quantity > intent.quantity:
                raise ValueError("Matched quantity must not exceed the requested order")
            if result.event_timestamp < intent.timestamp + timedelta(milliseconds=intent.latency_ms):
                raise ValueError("Matched event cannot precede the order's latency-adjusted arrival")
            if (intent.side == "buy" and result.estimated_price > intent.limit_price) or (intent.side == "sell" and result.estimated_price < intent.limit_price):
                raise ValueError("Matched price violates the order limit")
            if self._last_event_time is not None and result.event_timestamp < self._last_event_time:
                raise ValueError("Cannot consume an analog before the current ledger observation time")
            if set(result.source_trade_ids) & self.used_trade_ids:
                status, reason = "rejected", "source_already_used"
            else:
                notional = result.estimated_price * result.estimated_quantity
                fee = notional * self.fee_bps / 10000 + self.per_trade_fee
                if intent.side == "buy" and notional + fee > self.cash + 1e-9:
                    status, reason, fee = "rejected", "insufficient_cash", 0.0
                elif intent.side == "sell" and result.estimated_quantity > quantity_after:
                    status, reason, fee = "rejected", "insufficient_inventory", 0.0
                elif intent.side == "sell" and self.cash + notional < fee:
                    status, reason, fee = "rejected", "insufficient_cash_for_fees", 0.0
                else:
                    sign = 1 if intent.side == "buy" else -1
                    self.cash -= sign * notional + fee
                    quantity_after += sign * result.estimated_quantity
                    self.inventory[intent.symbol] = quantity_after
                    self.used_trade_ids.update(result.source_trade_ids)
                    self.total_fees += fee
                    self.gross_notional += notional
                    self.estimated_slippage_cost += sign * (result.estimated_price - intent.reference_price) * result.estimated_quantity
                    self.mark(result.event_timestamp, intent.symbol, result.estimated_price)
        elif result.status != "unmatched":
            raise ValueError("Match status must be matched or unmatched")
        record = ReplayRecord(intent, result, status, reason, self.cash, quantity_after, fee)
        self.records.append(record)
        return record

    def result(self) -> ReplayResult:
        matched = sum(record.status == "matched" for record in self.records)
        unmatched = sum(record.status == "unmatched" for record in self.records)
        rejected = sum(record.status == "rejected" for record in self.records)
        peak = self.initial_cash
        max_drawdown = 0.0
        for point in self.equity_curve:
            peak = max(peak, point.equity)
            max_drawdown = max(max_drawdown, (peak - point.equity) / peak)
        return ReplayResult(
            self.initial_cash, self.cash, dict(self.inventory), self.equity,
            matched, unmatched, rejected, matched / len(self.records) if self.records else 0.0,
            self.total_fees, self.gross_notional, self.estimated_slippage_cost,
            max_drawdown, self.gross_exposure, tuple(sorted(self.used_trade_ids)),
            tuple(self.records), tuple(self.equity_curve),
        )


def replay_orders(
    orders: Sequence[OrderIntent],
    prints: Sequence[TradePrint],
    *,
    initial_cash: float,
    config: MatchingConfig | None = None,
    fee_bps: float = 0.0,
    per_trade_fee: float = 0.0,
) -> ReplayResult:
    """Replay a frozen intent list; overlapping observation windows abstain.

    This conservative helper requires each decision to occur after the previous
    analog event becomes observable. It rejects overlapping decisions rather
    than letting future cash or inventory silently influence an earlier order.
    Use ``PortfolioLedger`` directly with chronological events for richer flows.
    """
    ledger = PortfolioLedger(initial_cash, fee_bps=fee_bps, per_trade_fee=per_trade_fee)
    for intent in sorted(orders, key=lambda value: value.timestamp):
        if ledger._last_event_time is not None and intent.timestamp < ledger._last_event_time:
            unmatched = MatchResult("unmatched", "decision_precedes_previous_observation", intent.quantity)
            ledger.apply_match(intent, unmatched)
            continue
        ledger.mark(intent.timestamp, intent.symbol, intent.reference_price)
        result = match_order(intent, prints, config, used_trade_ids=ledger.used_trade_ids)
        ledger.apply_match(intent, result)
    return ledger.result()


def load_trades_csv(path: str | Path) -> list[TradePrint]:
    """Load normalized source executions with explicit aggressor and quote data.

    Required columns: timestamp,trade_id,symbol,side,quantity,price. Timestamps
    require an offset (or Z); optional queue_evidence_ids use semicolon-separated
    IDs. Real providers must preserve original identifiers and normalize their
    aggressor convention; missing fields are never invented by this loader.
    """
    def parse_time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not _valid_time(parsed):
            raise ValueError("CSV timestamps require explicit UTC offsets")
        return parsed.astimezone(timezone.utc)

    result: list[TradePrint] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "trade_id", "symbol", "side", "quantity", "price"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Trade CSV is missing required normalized columns")
        for line, row in enumerate(reader, start=2):
            try:
                options: dict[str, object] = {}
                for name in ("bid", "ask", "spread", "volatility", "queue_ahead", "queue_depletion", "impact_bps"):
                    options[name] = float(row[name]) if row.get(name) else None
                options["queue_observed_from"] = parse_time(row["queue_observed_from"]) if row.get("queue_observed_from") else None
                options["queue_evidence_ids"] = tuple(part for part in (row.get("queue_evidence_ids") or "").split(";") if part)
                verified = (row.get("queue_priority_verified") or "false").lower()
                if verified not in {"true", "false", "1", "0"}:
                    raise ValueError("queue_priority_verified must be true/false or 1/0")
                options["queue_priority_verified"] = verified in {"true", "1"}
                result.append(TradePrint(
                    timestamp=parse_time(row["timestamp"]), trade_id=row["trade_id"],
                    symbol=row["symbol"], side=row["side"], quantity=int(row["quantity"]),
                    price=float(row["price"]), **options,
                ))
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid trade CSV row {line}: {exc}") from exc
    if len({trade.trade_id for trade in result}) != len(result):
        raise ValueError("Trade CSV contains duplicate trade IDs")
    return result
