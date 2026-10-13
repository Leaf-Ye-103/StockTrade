"""Direct portfolio actions, mature observed-data feedback and policy updates."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .action_risk import constrain_action
from .data import build_features, load_bars_csv, load_events_jsonl
from .execution import MatchingConfig, OrderIntent, PortfolioLedger, load_trades_csv, match_order
from .model import PolicyAgent, PolicyConfig
from .policy_contracts import PortfolioState
from .rewards import PortfolioObservation, RewardConfig, evaluate_action, evaluate_portfolio_action
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


def load_market(bars_path, events_path=None, symbols=None, *, schemas=None, train_end=None):
    rows = load_bars_csv(bars_path)
    events = load_events_jsonl(events_path) if events_path else []
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
        frames[symbol] = build_features(bars[symbol], events, feature_names=schema)
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


def _action_advantages(records):
    """Use each mature portfolio advantage once; never sum shaped advantages.

    Crediting a later sale's avoided-loss reward to its preceding bad purchase
    could reward deliberate loss/rescue cycles. Actual future equity already
    accounts for subsequent matched actions in each decision's own comparison.
    """
    return {position: record["feedback"]["reward"] for position, record in enumerate(records)
            if record["feedback"]["reward"] is not None}


def rollout(agent, bars, frames, prints, *, start, stop, funding, reward_config,
            allowed_symbols=None, training=False, constraint_penalty=.001):
    """One bounded chronological episode. Never update weights during a rollout.

    Decisions stop H bars before the partition end. Pending feedback matures
    only once all H later observations have actually been reached. Later data
    never trains an earlier checkpoint across a partition boundary.
    """
    allowed = _symbols(allowed_symbols, set(agent.universe))
    if not agent.config.lookback - 1 <= start < stop <= len(frames[agent.universe[0]].timestamps):
        raise ValueError("Rollout must have a historical window and a valid exclusive end")
    risk = RiskConfig()
    matching = MatchingConfig(size_tolerance=0.0)
    ledger = PortfolioLedger(initial_cash=funding, fee_bps=reward_config.fee_bps)
    trade_groups, trade_times = _trade_index(prints, agent.universe)
    records, samples, pending, observations = [], {}, deque(), {}
    reasons = Counter()
    peak, day_start, current_day = funding, funding, None
    timestamps = frames[agent.universe[0]].timestamps

    def mature(current_index):
        while pending and pending[0][0] + reward_config.horizon_steps <= current_index:
            decision_index, position, action, match, before = pending.popleft()
            future = [observations[j]
                      for j in range(decision_index + 1, decision_index + reward_config.horizon_steps + 1)]
            supporting = tuple(source_id for ledger_record in ledger.records
                if ledger_record.status == "matched"
                and timestamps[decision_index] < ledger_record.match.event_timestamp <= future[-1].timestamp
                for source_id in ledger_record.match.source_trade_ids)
            feedback = evaluate_portfolio_action(action, match,
                decision_timestamp=timestamps[decision_index], initial_funding=funding,
                cash_before=before.cash, holdings_before=before.holdings, equity_before=before.equity,
                future_observations=future, config=reward_config, supporting_trade_ids=supporting)
            records[position]["feedback"] = asdict(feedback)

    for index in range(start, stop):
        timestamp = timestamps[index]
        for symbol in agent.universe:
            ledger.mark(timestamp, symbol, float(frames[symbol].prices[index]))
        observations[index] = PortfolioObservation(timestamp, ledger.equity,
            {symbol: float(frames[symbol].prices[index]) for symbol in agent.universe})
        peak = max(peak, ledger.equity)
        if current_day != timestamp.date():
            current_day, day_start = timestamp.date(), ledger.equity
        mature(index)
        if index + reward_config.horizon_steps >= stop:
            continue
        state = PortfolioState(funding, ledger.cash, dict(ledger.positions),
            {symbol: float(frames[symbol].prices[index]) for symbol in agent.universe}, peak, day_start)
        with torch.set_grad_enabled(training):
            sample = agent.act(frames, index, state, deterministic=not training, allowed_symbols=allowed)
        action = sample.action
        quotes = _quotes(agent, bars, frames, index)
        decision = constrain_action(action, state, quotes, allowed, risk)
        record = dict(timestamp=timestamp, bar_index=index, policy_action=asdict(action),
            probability=float(sample.log_probability.detach().exp()), feasibility=asdict(decision),
            account_before=dict(equity=state.equity, cash=state.cash, holdings=dict(state.holdings)))
        position = len(records)
        records.append(record)
        if training:
            samples[position] = sample
        reasons.update(decision.reasons)
        if not decision.accepted:
            record["feedback"] = dict(status="constraint", reward=-constraint_penalty,
                maturity_timestamp=timestamp, source_trade_ids=(),
                components={"constraint_cost": -constraint_penalty},
                baseline_description="Known risk/feasibility violation; no invented execution or economic loss")
            continue
        if action.side == "hold":
            pending.append((index, position, action, None, state))
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
            effective_action=dict(symbol=symbol, side=action.side, value=decision.notional, quantity=decision.quantity))
        if accounting.status == "matched":
            pending.append((index, position, action, match, state))
        else:
            record["feedback"] = asdict(evaluate_action(action, None, decision_timestamp=timestamp,
                decision_price=bar.close, future_marks=[], initial_funding=funding,
                equity_before=state.equity, config=reward_config))
    if pending:
        raise RuntimeError("Unmatured feedback at episode boundary; no truncated horizon is allowed")
    advantages = _action_advantages(records)
    eligible = [position for position in range(len(records)) if position in advantages]
    ledger_result = asdict(ledger.result())
    economic_rewards = [record["feedback"]["reward"] for record in records
                        if record["feedback"]["status"] == "supported"]
    constraint_rewards = [record["feedback"]["reward"] for record in records
                          if record["feedback"]["status"] == "constraint"]
    supported_trades = sum(record["feedback"]["status"] == "supported"
                           and record["policy_action"]["side"] != "hold" for record in records)
    report = dict(
        interpretation="Observed-execution action feedback and analog portfolio accounting; not causal impact",
        universe=list(agent.universe), allowed_symbols=list(allowed), initial_funding=funding,
        decisions=len(records), supported_trade_feedback=supported_trades,
        economic_reward_sum=float(sum(economic_rewards)), constraint_reward_sum=float(sum(constraint_rewards)),
        policy_reward_sum=float(sum(economic_rewards) + sum(constraint_rewards)),
        eligible_feedback=len(eligible), unmatched_feedback=sum(r["feedback"]["status"] == "unmatched" for r in records),
        final_equity=ledger.equity, ledger_return=ledger.equity / funding - 1,
        execution=ledger_result, risk_reasons=dict(reasons), feedback_records=records,
        reward_horizon=reward_config.horizon_steps,
        reward_interpretation="Each actual portfolio path versus its frozen pre-action hold baseline; non-additive advantages",
    )
    return report, [samples[position] for position in eligible] if training else [], [advantages[p] for p in eligible]


def train(bars_path, events_path, trades_path, output, *, epochs=3, seed=7, lookback=16,
          funding=100000.0, symbols=None, reward_horizon=8, constraint_penalty=.001,
          episode_steps=128, train_fraction=.7, validation_fraction=.15, selection_drawdown_weight=.5):
    funding = _funding(funding)
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Training epochs must be a positive integer")
    if not math.isfinite(selection_drawdown_weight) or selection_drawdown_weight < 0:
        raise ValueError("Selection drawdown weight must be finite and nonnegative")
    if not math.isfinite(constraint_penalty) or constraint_penalty < 0:
        raise ValueError("Constraint penalty must be finite and nonnegative")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError("Fractions must leave nonempty chronological train, validation and test intervals")
    bars, _ = load_market(bars_path, events_path, symbols)
    universe = tuple(bars)
    total = len(bars[universe[0]])
    train_end = int(total * train_fraction)
    validation_end = int(total * (train_fraction + validation_fraction))
    rewards = RewardConfig(horizon_steps=reward_horizon, fee_bps=1.0)
    if (train_end <= lookback + reward_horizon or validation_end - train_end <= reward_horizon
            or total - validation_end <= reward_horizon):
        raise ValueError("Insufficient bars in each partition for lookback and full reward horizon")
    if not isinstance(episode_steps, int) or episode_steps <= reward_horizon:
        raise ValueError("Episode steps must exceed the full reward horizon")
    bars, frames = load_market(bars_path, events_path, universe, train_end=train_end)
    prints = load_trades_csv(trades_path)
    agent = PolicyAgent.create(frames, universe, train_end, funding, PolicyConfig(lookback=lookback, seed=seed))
    agent.validation_end = validation_end
    agent.validation_last_timestamp = frames[universe[0]].timestamps[validation_end - 1].isoformat()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, audit = [], []
    best_score, best_epoch, supported = -math.inf, 0, 0
    with _threads(agent.config.num_threads), torch.random.fork_rng():
        torch.manual_seed(seed)
        for epoch in range(epochs):
            epoch_samples, epoch_reward = 0, 0.0
            losses = []
            agent.model.train()
            for start in range(lookback - 1, train_end, episode_steps):
                stop = min(start + episode_steps, train_end)
                if stop - start <= reward_horizon:
                    continue
                report, samples, advantages = rollout(agent, bars, frames, prints, start=start, stop=stop,
                    funding=funding, reward_config=rewards, training=True,
                    constraint_penalty=constraint_penalty)
                supported += report["supported_trade_feedback"]
                if samples:
                    losses.append(agent.update(samples, advantages))
                    epoch_samples += len(samples)
                epoch_reward += report["policy_reward_sum"]
                for record in report["feedback_records"]:
                    audit.append(dict(epoch=epoch + 1, partition="train", **record))
            agent.model.eval()
            validation, _, _ = rollout(agent, bars, frames, prints, start=train_end, stop=validation_end,
                funding=funding, reward_config=rewards, constraint_penalty=constraint_penalty)
            score = validation["ledger_return"] - selection_drawdown_weight * validation["execution"]["max_drawdown"]
            history.append(dict(epoch=epoch + 1, trained_samples=epoch_samples,
                training_reward=epoch_reward, validation_portfolio_score=score,
                validation_supported_trades=validation["supported_trade_feedback"], updates=losses))
            if supported and score > best_score:
                best_score, best_epoch = score, epoch + 1
                agent.save(output / "model.pt")
    if supported == 0:
        raise ValueError("No supported economic feedback in training; supply compatible recorded trade data")
    timestamps = frames[universe[0]].timestamps
    metadata = dict(model_kind="direct_action_policy", format_version=2, universe=list(universe),
        initial_funding=funding, feature_schemas={s: frames[s].feature_names for s in universe},
        train_end=train_end, validation_end=validation_end, total_bars=total,
        validation_context_sha256={s: frame_signature(frames[s], validation_end) for s in universe},
        train_end_time=timestamps[train_end - 1], validation_end_time=timestamps[validation_end - 1],
        checkpoint_sha256=file_signature(output / "model.pt"), seed=seed, selected_epoch=best_epoch,
        training_supported_feedback=supported, reward_config=asdict(rewards),
        selection_drawdown_weight=selection_drawdown_weight,
        constraint_penalty=constraint_penalty, episode_steps=episode_steps,
        purpose="Direct policy optimized from mature evaluator feedback; no prediction target or loss")
    write_json(output / "model.info.json", metadata)
    write_json(output / "training.json", dict(history=history, metadata=metadata))
    with (output / "training_feedback.jsonl").open("w") as handle:
        for record in audit:
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")
    return metadata


def _load_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    metadata = json.loads(checkpoint.with_name("model.info.json").read_text())
    if metadata.get("model_kind") != "direct_action_policy":
        raise ValueError("Retrain the old forecast checkpoint as a direct-action policy")
    if file_signature(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint and metadata are not the saved pair")
    agent = PolicyAgent.load(checkpoint)
    if (list(agent.universe) != metadata["universe"] or agent.train_end != metadata["train_end"]
            or agent.training_last_timestamp != metadata["train_end_time"]
            or agent.initial_funding != metadata["initial_funding"]
            or agent.validation_end != metadata["validation_end"]
            or agent.validation_last_timestamp != metadata["validation_end_time"]
            or {s: agent.scalers[s].feature_names for s in agent.universe} != metadata["feature_schemas"]):
        raise ValueError("Checkpoint and metadata training contracts disagree")
    return agent, metadata


def evaluate(bars_path, events_path, trades_path, checkpoint, output, *, funding=None, symbols=None):
    agent, metadata = _load_checkpoint(checkpoint)
    funding = _funding(metadata["initial_funding"] if funding is None else funding)
    allowed = _symbols(symbols, set(agent.universe))
    bars, frames = load_market(bars_path, events_path, agent.universe, schemas=metadata["feature_schemas"])
    boundary = metadata["validation_end"]
    if len(frames[agent.universe[0]].timestamps) - boundary <= metadata["reward_config"]["horizon_steps"]:
        raise ValueError("Held-out evaluation needs a full reward horizon after validation")
    if any(frame_signature(frames[s], boundary) != metadata["validation_context_sha256"][s] for s in agent.universe):
        raise ValueError("Checkpoint data/schema differs from supplied point-in-time training history")
    if frames[agent.universe[0]].timestamps[boundary - 1].isoformat() != metadata["validation_end_time"]:
        raise ValueError("Validation boundary timestamp disagrees with checkpoint metadata")
    payload = dict(metadata["reward_config"])
    payload["horizon_fractions"], payload["horizon_weights"] = tuple(payload["horizon_fractions"]), tuple(payload["horizon_weights"])
    prints = load_trades_csv(trades_path)
    with _threads(agent.config.num_threads):
        report, _, _ = rollout(agent, bars, frames, prints, start=boundary,
            stop=len(frames[agent.universe[0]].timestamps), funding=funding,
            reward_config=RewardConfig(**payload), allowed_symbols=allowed,
            constraint_penalty=metadata["constraint_penalty"])
    report["source_is_synthetic"] = bool(prints) and all(p.trade_id.startswith("SYNTHETIC-") for p in prints)
    report["policy_updated"] = False
    report["provenance"] = dict(checkpoint_sha256=metadata["checkpoint_sha256"],
        bars_sha256=file_signature(bars_path), trades_sha256=file_signature(trades_path),
        events_sha256=file_signature(events_path) if events_path else None,
        reward_config=payload, risk_config=asdict(RiskConfig()), selected_epoch=metadata["selected_epoch"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = report.pop("feedback_records")
    write_json(output / "report.json", report)
    with (output / "decisions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")
    return report


def act(bars_path, events_path, checkpoint, *, funding=None, symbols=None, cash=None, holdings=None):
    """Produce current direct dollar actions without any future-data evaluator."""
    agent, metadata = _load_checkpoint(checkpoint)
    funding = _funding(metadata["initial_funding"] if funding is None else funding)
    allowed = _symbols(symbols, set(agent.universe))
    bars, frames = load_market(bars_path, events_path, agent.universe, schemas=metadata["feature_schemas"])
    holdings = {} if holdings is None else holdings
    index = len(frames[agent.universe[0]].timestamps) - 1
    state = PortfolioState(funding, funding if cash is None else cash, holdings,
        {s: float(frames[s].prices[index]) for s in agent.universe})
    if cash is None and any(state.holdings.values()):
        raise ValueError("Supply available cash separately when providing existing holdings")
    if frames[agent.universe[0]].timestamps[index] <= datetime.fromisoformat(metadata["validation_end_time"]):
        raise ValueError("Policy actions must be after the checkpoint's training/validation cutoff")
    with _threads(agent.config.num_threads), torch.no_grad():
        sample = agent.act(frames, index, state, deterministic=True, allowed_symbols=allowed)
    decision = constrain_action(sample.action, state, _quotes(agent, bars, frames, index), allowed)
    actions = []
    if decision.accepted and decision.quantity:
        actions.append(dict(side=sample.action.side, symbol=sample.action.symbol,
                            value=decision.notional, quantity=decision.quantity))
    return dict(timestamp=frames[agent.universe[0]].timestamps[index], total_funding=funding,
        available_cash=state.cash, equity=state.equity, allowed_symbols=list(allowed),
        policy_action=asdict(sample.action), trading_actions=actions, risk_decision=asdict(decision),
        execution_status="Proposal only; no broker order or unsupported historical fill")
