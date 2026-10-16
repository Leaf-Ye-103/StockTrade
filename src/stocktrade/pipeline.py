"""Direct portfolio actions, mature observed-data feedback and policy updates."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .action_risk import ActionDecision, constrain_action
from .data import build_features, load_bars_csv, load_events_jsonl
from .execution import MatchingConfig, OrderIntent, PortfolioLedger, load_trades_csv, match_order
from .model import PolicyAgent, PolicyConfig
from .policy_contracts import PortfolioState
from .objective import evaluate_episode
from .risk import MarketState, RiskConfig


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def file_signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_signature(frame, end):
    digest = hashlib.sha256()
    digest.update(json.dumps(frame.feature_names).encode())
    digest.update(frame.symbol.encode())
    for timestamp in frame.timestamps[:end]:
        digest.update(timestamp.isoformat().encode())
    for array in (frame.values[:end], frame.observed[:end], frame.prices[:end]):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@contextmanager
def _threads(count=1):
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _symbols(requested, available):
    result = tuple(sorted(available)) if requested is None else tuple(requested)
    if not result or len(set(result)) != len(result) or any(s not in available for s in result):
        raise ValueError("Symbols must be a distinct nonempty subset of the supplied stock universe")
    return result


def _funding(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Total funding must be positive and finite")
    return float(value)


def load_market(bars_path, events_path=None, symbols=None, *, schemas=None, train_end=None,
                extra_events=None, as_of=None):
    rows = load_bars_csv(bars_path)
    events = load_events_jsonl(events_path) if events_path else []
    events.extend(extra_events or [])
    universe = _symbols(symbols, {bar.symbol for bar in rows})
    bars = {symbol: [] for symbol in universe}
    for bar in rows:
        if bar.symbol in bars:
            bars[bar.symbol].append(bar)
    for stock in bars.values():
        stock.sort(key=lambda bar: bar.timestamp)
    timestamps = [bar.timestamp for bar in bars[universe[0]]]
    if any([bar.timestamp for bar in bars[symbol]] != timestamps for symbol in universe):
        raise ValueError("All stock bars must have identical timestamps; no implicit fill or alignment")
    frames = {}
    for symbol in universe:
        schema = schemas[symbol] if schemas is not None else None
        if schema is None and train_end is not None:
            if not 1 <= train_end <= len(timestamps):
                raise ValueError("Training prefix must exist in the dataset")
            cutoff = timestamps[train_end - 1]
            known = [event for event in events if max(event.published_at, event.available_at) <= cutoff]
            schema = build_features(bars[symbol][:train_end], known).feature_names
        frames[symbol] = build_features(bars[symbol], events, feature_names=schema, as_of=as_of)
    return bars, frames


def _quotes(agent, bars, frames, index):
    quotes = {}
    for symbol in agent.universe:
        bar = bars[symbol][index]
        has_quotes = bar.bid is not None and bar.ask is not None
        spread = bar.ask - bar.bid if has_quotes else 0.0
        active = agent.scalers[symbol].counts > 0
        quality = float(frames[symbol].observed[index, active].mean()) if active.any() else 0.0
        quotes[symbol] = MarketState(symbol, bar.close, bar.volume if has_quotes else 0.0,
            spread_bps=spread / bar.close * 10000, data_quality=quality, bid=bar.bid, ask=bar.ask)
    return quotes


def _trade_index(prints, universe):
    groups = {symbol: [] for symbol in universe}
    for record in prints:
        if record.symbol in groups:
            groups[record.symbol].append(record)
    for records in groups.values():
        records.sort(key=lambda record: (record.timestamp, record.trade_id))
    return groups, {symbol: [record.timestamp for record in records] for symbol, records in groups.items()}


def rollout(agent, bars, frames, prints, *, start, stop, funding,
            allowed_symbols=None, training=False, risk_config=None):
    """Replay a full portfolio episode; the sole signal is terminal net wealth.

    No decision is scored separately. Risk-rejected requests are known no-ops,
    and unsupported execution has unknown status, never a fabricated fill.
    Incomplete or risk-inadmissible trajectories are not used for learning.
    The final bar only values existing holdings; no unobserved exit is assumed.
    """
    allowed = _symbols(allowed_symbols, set(agent.universe))
    timestamps = frames[agent.universe[0]].timestamps
    if not agent.config.lookback - 1 <= start < stop - 1 < len(timestamps):
        raise ValueError("Rollout needs a past window and at least two complete episode bars")
    risk = risk_config or RiskConfig()
    matching = MatchingConfig(size_tolerance=0.0)
    ledger = PortfolioLedger(initial_cash=funding, fee_bps=risk.commission_bps)
    trade_groups, trade_times = _trade_index(prints, agent.universe)
    records, samples = [], []
    reasons = Counter()
    peak, day_start, current_day = funding, funding, None
    for index in range(start, stop):
        timestamp = timestamps[index]
        for symbol in agent.universe:
            ledger.mark(timestamp, symbol, float(frames[symbol].prices[index]))
        peak = max(peak, ledger.equity)
        date = timestamp.astimezone(ZoneInfo("America/New_York")).date()
        if current_day != date:
            current_day, day_start = date, ledger.equity
        if index == stop - 1:
            continue
        state = PortfolioState(funding, ledger.cash, dict(ledger.positions),
            {symbol: float(frames[symbol].prices[index]) for symbol in agent.universe}, peak, day_start)
        with torch.set_grad_enabled(training):
            sample = agent.act(frames, index, state, deterministic=not training, allowed_symbols=allowed)
        action = sample.action
        decision = constrain_action(action, state, _quotes(agent, bars, frames, index), allowed, risk)
        record = dict(timestamp=timestamp, bar_index=index, policy_action=asdict(action),
            probability=float(sample.log_probability.detach().exp()), feasibility=asdict(decision),
            account_before=dict(equity=state.equity, cash=state.cash, holdings=dict(state.holdings)))
        records.append(record)
        if training:
            samples.append(sample)
        reasons.update(decision.reasons)
        if not decision.accepted:
            record["transition"] = "risk_rejected_no_order"
            continue
        if action.side == "hold":
            record["transition"] = "hold"
            continue
        symbol = action.symbol
        bar = bars[symbol][index]
        recent = np.diff(np.log(frames[symbol].prices[max(0, index - 20):index + 1]))
        volatility = float(np.std(recent)) if len(recent) >= 2 else None
        intent = OrderIntent(timestamp, symbol, action.side, decision.quantity,
            bar.ask if action.side == "buy" else bar.bid, bar.close,
            spread=bar.ask - bar.bid, volatility=volatility)
        lo = bisect_left(trade_times[symbol], timestamp)
        hi = bisect_left(trade_times[symbol], timestamps[index + 1])
        match = match_order(intent, trade_groups[symbol][lo:hi], matching, used_trade_ids=ledger.used_trade_ids)
        accounting = ledger.apply_match(intent, match)
        record.update(intent=asdict(intent), analog=asdict(match), accounting=asdict(accounting),
            transition=accounting.status,
            effective_action=dict(symbol=symbol, side=action.side, value=decision.notional, quantity=decision.quantity))
    objective = evaluate_episode(funding, ledger.equity, equity_curve=ledger.equity_curve,
        risk_config=risk, initial_timestamp=timestamps[start], final_timestamp=timestamps[stop - 1])
    execution = asdict(ledger.result())
    complete = execution["unmatched_orders"] == 0 and execution["rejected_orders"] == 0
    eligible = objective.admissible and complete
    report = dict(
        interpretation="Terminal wealth on an observed-execution analog ledger, not causal market impact",
        universe=list(agent.universe), allowed_symbols=list(allowed), initial_funding=funding,
        decisions=len(records), matched_orders=execution["matched_orders"],
        objective=asdict(objective), trajectory_complete=complete, learning_eligible=eligible,
        final_equity=ledger.equity, ledger_return=objective.wealth_growth,
        execution=execution, risk_reasons=dict(reasons), decision_records=records,
        objective_name="terminal_net_wealth_growth", action_grading=False,
    )
    return report, samples if training and eligible else [], [objective.wealth_growth] * len(samples) if training and eligible else []


def train(bars_path, events_path, trades_path, output, *, epochs=3, seed=7, lookback=16,
          funding=100000.0, symbols=None, episode_steps=128,
          train_fraction=.7, validation_fraction=.15, risk_config=None):
    funding = _funding(funding)
    risk = risk_config or RiskConfig()
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Training epochs must be a positive integer")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Fractions must leave nonempty chronological train, validation and test intervals")
    if not isinstance(episode_steps, int) or isinstance(episode_steps, bool) or episode_steps < 2:
        raise ValueError("Episode steps must contain at least two bars")
    bars, _ = load_market(bars_path, events_path, symbols)
    universe = tuple(bars)
    total = len(bars[universe[0]])
    train_end = int(total * train_fraction)
    validation_end = int(total * (train_fraction + validation_fraction))
    if train_end <= lookback + 1 or validation_end - train_end < 2 or total - validation_end < 2:
        raise ValueError("Insufficient bars for historical windows and complete portfolio episodes")
    # All updates have the same declared time horizon. A short training tail is
    # left unused rather than rewarding an easier or shorter interval.
    episode_length = min(episode_steps, train_end - (lookback - 1))
    bars, frames = load_market(bars_path, events_path, universe, train_end=train_end)
    prints = load_trades_csv(trades_path)
    agent = PolicyAgent.create(frames, universe, train_end, funding, PolicyConfig(lookback=lookback, seed=seed))
    agent.validation_end = validation_end
    agent.validation_last_timestamp = frames[universe[0]].timestamps[validation_end - 1].isoformat()
    agent.risk_config = asdict(risk)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, audit = [], []
    best_score, best_epoch, supported, updated_episodes = -math.inf, 0, 0, 0
    with _threads(agent.config.num_threads), torch.random.fork_rng():
        torch.manual_seed(seed)
        for epoch in range(epochs):
            losses, episode_reports = [], []
            agent.model.train()
            for start in range(lookback - 1, train_end - episode_length + 1, episode_length):
                stop = start + episode_length
                report, samples, wealth = rollout(agent, bars, frames, prints, start=start, stop=stop,
                    funding=funding, training=True, risk_config=risk)
                supported += report["matched_orders"]
                if samples:
                    losses.append(agent.update(samples, wealth))
                    updated_episodes += 1
                episode_reports.append({k: report[k] for k in
                    ("objective", "trajectory_complete", "learning_eligible", "matched_orders")})
                for record in report["decision_records"]:
                    audit.append(dict(epoch=epoch + 1, partition="train", **record))
            agent.model.eval()
            validation, _, _ = rollout(agent, bars, frames, prints, start=train_end, stop=validation_end,
                funding=funding, risk_config=risk)
            score = validation["objective"]["wealth_growth"]
            history.append(dict(epoch=epoch + 1, episodes=episode_reports,
                validation_wealth_growth=score, validation_admissible=validation["learning_eligible"],
                validation_matched_orders=validation["matched_orders"], updates=losses))
            if updated_episodes and supported and validation["learning_eligible"] and score > best_score:
                best_score, best_epoch = score, epoch + 1
                agent.save(output / "model.pt")
    if not supported:
        raise ValueError("No matched executions in training; supply compatible recorded trade data")
    if not updated_episodes or not best_epoch:
        raise ValueError("No complete risk-admissible train/validation trajectory; inspect coverage and risk limits")
    timestamps = frames[universe[0]].timestamps
    metadata = dict(model_kind="direct_wealth_policy", format_version=3, universe=list(universe),
        training_objective="terminal_net_wealth_growth", initial_funding=funding,
        feature_schemas={s: frames[s].feature_names for s in universe},
        train_end=train_end, validation_end=validation_end, total_bars=total,
        validation_context_sha256={s: frame_signature(frames[s], validation_end) for s in universe},
        train_end_time=timestamps[train_end - 1], validation_end_time=timestamps[validation_end - 1],
        checkpoint_sha256=file_signature(output / "model.pt"), seed=seed, selected_epoch=best_epoch,
        training_matched_orders=supported, updated_episodes=updated_episodes,
        risk_config=asdict(risk), episode_steps=episode_length,
        training_inputs_sha256=dict(bars=file_signature(bars_path), trades=file_signature(trades_path),
                                   events=file_signature(events_path) if events_path else None),
        purpose="Maximize terminal net wealth under risk constraints; no action grades or prediction loss")
    write_json(output / "model.info.json", metadata)
    write_json(output / "training.json", dict(history=history, metadata=metadata))
    with (output / "training_decisions.jsonl").open("w") as handle:
        for record in audit:
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")
    return metadata


def _load_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_name("model.info.json").read_text())
    if metadata.get("model_kind") != "direct_wealth_policy" or metadata.get("training_objective") != "terminal_net_wealth_growth":
        raise ValueError("Retrain the old action-reward checkpoint as a terminal-wealth policy")
    if file_signature(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint and metadata are not the saved pair")
    agent = PolicyAgent.load(checkpoint)
    if (list(agent.universe) != metadata["universe"] or agent.train_end != metadata["train_end"]
            or agent.training_last_timestamp != metadata["train_end_time"]
            or agent.initial_funding != metadata["initial_funding"]
            or agent.validation_end != metadata["validation_end"]
            or agent.validation_last_timestamp != metadata["validation_end_time"]
            or agent.risk_config != metadata["risk_config"]
            or {s: agent.scalers[s].feature_names for s in agent.universe} != metadata["feature_schemas"]):
        raise ValueError("Checkpoint and metadata training contracts disagree")
    return agent, metadata


def evaluate(bars_path, events_path, trades_path, checkpoint, output, *, funding=None, symbols=None,
             evaluation_config=None, dashboard=True):
    agent, metadata = _load_checkpoint(checkpoint)
    funding = _funding(metadata["initial_funding"] if funding is None else funding)
    allowed = _symbols(symbols, set(agent.universe))
    bars, frames = load_market(bars_path, events_path, agent.universe, schemas=metadata["feature_schemas"])
    boundary = metadata["validation_end"]
    if len(frames[agent.universe[0]].timestamps) - boundary < 2:
        raise ValueError("Held-out evaluation needs two complete portfolio observations after validation")
    if any(frame_signature(frames[s], boundary) != metadata["validation_context_sha256"][s] for s in agent.universe):
        raise ValueError("Checkpoint data/schema differs from supplied point-in-time training history")
    if frames[agent.universe[0]].timestamps[boundary - 1].isoformat() != metadata["validation_end_time"]:
        raise ValueError("Validation boundary timestamp disagrees with checkpoint metadata")
    prints = load_trades_csv(trades_path)
    risk = RiskConfig(**metadata["risk_config"])
    with _threads(agent.config.num_threads):
        report, _, _ = rollout(agent, bars, frames, prints, start=boundary,
            stop=len(frames[agent.universe[0]].timestamps), funding=funding,
            risk_config=risk, allowed_symbols=allowed)
    report["source_is_synthetic"] = bool(prints) and all(p.trade_id.startswith("SYNTHETIC-") for p in prints)
    report["policy_updated"] = False
    report["provenance"] = dict(checkpoint_sha256=metadata["checkpoint_sha256"],
        bars_sha256=file_signature(bars_path), trades_sha256=file_signature(trades_path),
        events_sha256=file_signature(events_path) if events_path else None,
        risk_config=asdict(risk), selected_epoch=metadata["selected_epoch"])
    records = report.pop("decision_records")
    # Only frozen held-out decisions enter diagnostics. Training/validation and
    # checkpoint selection never call the evaluation reporting modules.
    from .diagnostics import EvaluationConfig, include_benchmark, write_evaluation_artifacts
    config = evaluation_config or EvaluationConfig()
    diagnostic_bars = include_benchmark(bars_path, bars, config.benchmark_symbol)
    return write_evaluation_artifacts(bars=diagnostic_bars, records=records, report=report,
        output=output, start=boundary, stop=len(frames[agent.universe[0]].timestamps),
        risk_config=risk, config=config, dashboard=dashboard)


def _fresh_market(timestamp, now, maximum_age):
    if not math.isfinite(maximum_age) or maximum_age <= 0:
        raise ValueError("Maximum market age must be positive and finite")
    age = (now - timestamp).total_seconds()
    if age < 0:
        raise ValueError("Latest completed market bar is in the future")
    if age > maximum_age:
        raise ValueError("Market data is stale for a current online decision; refresh bars or use explicit --offline research")


def act(bars_path, events_path, checkpoint, *, funding=None, symbols=None, cash=None, holdings=None,
        collect=True, sources_path=None, collection_output="data/online-events.jsonl", max_market_age_seconds=900.,
        peak_equity=None, day_start_equity=None):
    """Acquire current information and propose actions from strictly past inputs.

    Online decisions use actual wall-clock collection availability. Only the
    final feature clock advances: the completed market bar is never retimestamped
    or duplicated. Historical train/evaluate paths never call online collection.
    """
    agent, metadata = _load_checkpoint(checkpoint)
    funding = _funding(metadata["initial_funding"] if funding is None else funding)
    allowed = _symbols(symbols, set(agent.universe))
    bars, frames = load_market(bars_path, events_path, agent.universe, schemas=metadata["feature_schemas"])
    index = len(frames[agent.universe[0]].timestamps) - 1
    market_timestamp = bars[agent.universe[0]][index].timestamp
    if market_timestamp <= datetime.fromisoformat(metadata["validation_end_time"]):
        raise ValueError("Policy actions must be after the checkpoint's training/validation cutoff")
    collection_report = None
    decision_timestamp = market_timestamp
    if collect:
        _fresh_market(market_timestamp, datetime.now(timezone.utc), max_market_age_seconds)
        from .collection import collect_information
        collection_report = collect_information(allowed, collection_output, config_path=sources_path).to_dict()
        decision_timestamp = datetime.now(timezone.utc)
        _fresh_market(market_timestamp, decision_timestamp, max_market_age_seconds)
        extra = load_events_jsonl(collection_output) if Path(collection_output).exists() else []
        bars, frames = load_market(bars_path, events_path, agent.universe,
            schemas=metadata["feature_schemas"], extra_events=extra, as_of=decision_timestamp)
    state = PortfolioState(funding, funding if cash is None else cash, {} if holdings is None else holdings,
        {s: float(frames[s].prices[index]) for s in agent.universe})
    if cash is None and any(state.holdings.values()):
        raise ValueError("Supply available cash separately when providing existing holdings")
    new_account = cash is None and not any(state.holdings.values())
    for value in (peak_equity, day_start_equity):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError("Account peak and day-start equity must be positive and finite")
    state = replace(state,
        peak_equity=peak_equity if peak_equity is not None else funding if new_account else None,
        day_start_equity=day_start_equity if day_start_equity is not None else funding if new_account else None)
    history_missing = state.peak_equity is None or state.day_start_equity is None
    with _threads(agent.config.num_threads), torch.no_grad():
        sample = agent.act(frames, index, state, deterministic=True, allowed_symbols=allowed)
    decision = constrain_action(sample.action, state, _quotes(agent, bars, frames, index), allowed,
                                RiskConfig(**metadata["risk_config"]))
    if history_missing and sample.action.side == "buy":
        decision = ActionDecision(sample.action, "HOLD", 0, 0.0, ("risk_history_missing",), False)
    actions = []
    if decision.accepted and decision.quantity:
        actions.append(dict(side=sample.action.side, symbol=sample.action.symbol,
                            value=decision.notional, quantity=decision.quantity))
    result = dict(timestamp=decision_timestamp, market_timestamp=market_timestamp, total_funding=funding,
        available_cash=state.cash, equity=state.equity, allowed_symbols=list(allowed),
        policy_action=asdict(sample.action), trading_actions=actions, risk_decision=asdict(decision),
        information_collection=collection_report, online=bool(collect),
        risk_history_status="missing" if history_missing else "new_account" if new_account else "provided",
        account_risk_baselines=dict(peak_equity=state.peak_equity, day_start_equity=state.day_start_equity),
        execution_status="Proposal only; no broker order or unsupported historical fill")
    from .decision_risk import estimate_decision_risk
    result["risk_estimate"] = estimate_decision_risk(bars=bars,
        record=dict(timestamp=decision_timestamp, bar_index=index, policy_action=asdict(sample.action),
            feasibility=asdict(decision), account_before=dict(equity=state.equity, cash=state.cash,
                holdings=dict(state.holdings))), risk_config=RiskConfig(**metadata["risk_config"]))
    return result
