import csv
from datetime import datetime, timedelta, timezone
import json
import shutil

import pytest
import torch
import numpy as np

from stocktrade.data import build_features
from stocktrade.demo import generate_demo
from stocktrade.execution import TradePrint
from stocktrade.model import PolicyAgent, PolicyConfig, PolicySample
from stocktrade.pipeline import (
    act, evaluate, file_signature, frame_signature, load_market, rollout, train,
)
from stocktrade.policy_contracts import TradingAction
from stocktrade.schemas import MarketBar


@pytest.fixture(scope="module")
def source_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("direct-policy")
    data = generate_demo(root / "data", count=100, symbols=("NVDA", "MSFT"))
    model = root / "model"
    train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", model,
          epochs=1, lookback=4, symbols=("NVDA", "MSFT"))
    return root


@pytest.fixture
def experiment(source_experiment, tmp_path):
    shutil.copytree(source_experiment, tmp_path / "experiment")
    return tmp_path / "experiment"


def paths(root):
    return root / "data", root / "model", root / "evaluation"


def run_evaluation(root, **kwargs):
    data, model, output = paths(root)
    return evaluate(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
                    model / "model.pt", output, **kwargs)


def test_policy_encoders_learn_from_actual_matured_feedback(experiment):
    data, model, _ = paths(experiment)
    history = json.loads((model / "training.json").read_text())
    metadata = history["metadata"]
    assert metadata["training_matched_orders"] > 0
    updates = history["history"][0]["updates"]
    assert updates and all("policy_loss" in row and "value_loss" not in row for row in updates)
    _, frames = load_market(data / "bars.csv", data / "events.jsonl", ("NVDA", "MSFT"),
                           train_end=metadata["train_end"])
    initial = PolicyAgent.create(frames, ("NVDA", "MSFT"), metadata["train_end"], 100000,
                                 PolicyConfig(lookback=4, seed=metadata["seed"]))
    learned = PolicyAgent.load(model / "model.pt")
    encoder_names = [name for name in learned.model.state_dict() if name.startswith("encoders")]
    assert any(not torch.equal(initial.model.state_dict()[name], learned.model.state_dict()[name])
               for name in encoder_names)
    assert not any("next_mean" in name or "reconstruction" in name for name in learned.model.state_dict())


def test_evaluate_reports_actions_feedback_and_never_updates_weights(experiment):
    _, model, output = paths(experiment)
    before = file_signature(model / "model.pt")
    report = run_evaluation(experiment)
    assert report["decisions"] == 14
    assert report["source_is_synthetic"] and not report["policy_updated"]
    assert "heldout_return_mae" not in report
    assert report["universe"] == ["NVDA", "MSFT"]
    assert file_signature(model / "model.pt") == before
    records = [json.loads(row) for row in (output / "decisions.jsonl").read_text().splitlines()]
    assert len(records) == 14
    assert all("policy_action" in record and "forecast" not in record for record in records)


def test_complete_training_episodes_stay_in_training_partition(experiment):
    _, model, _ = paths(experiment)
    metadata = json.loads((model / "model.info.json").read_text())
    cutoff = datetime.fromisoformat(metadata["train_end_time"])
    records = [json.loads(row) for row in (model / "training_decisions.jsonl").read_text().splitlines()]
    assert records and all(datetime.fromisoformat(row["timestamp"]) < cutoff for row in records)
    assert all("feedback" not in row for row in records)
    history = json.loads((model / "training.json").read_text())["history"]
    for epoch in history:
        for episode in epoch["episodes"]:
            assert episode["objective"]["wealth_growth"] == pytest.approx(
                episode["objective"]["final_equity"] / episode["objective"]["initial_funding"] - 1)
        for update in epoch["updates"]:
            assert "episode_wealth_growth" in update and "mean_feedback" not in update


def test_act_outputs_dollar_trade_for_permitted_symbol(experiment, monkeypatch):
    data, model, _ = paths(experiment)

    def fixed_action(*args, **kwargs):
        return PolicySample(TradingAction(1, "NVDA", "buy", 100, .01),
                            torch.tensor(-.5), torch.tensor(0.))

    monkeypatch.setattr(PolicyAgent, "act", fixed_action)
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                   funding=10000, symbols=("NVDA",), collect=False)
    assert proposal["total_funding"] == 10000 and proposal["allowed_symbols"] == ["NVDA"]
    assert proposal["policy_action"]["value"] == 100
    assert len(proposal["trading_actions"]) == 1
    trade = proposal["trading_actions"][0]
    assert trade["symbol"] == "NVDA" and trade["side"] == "buy"
    assert 0 < trade["value"] <= 100 and trade["quantity"] == 1


def test_runtime_subset_is_respected_and_unknown_symbols_rejected(experiment):
    data, model, _ = paths(experiment)
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt", symbols=("MSFT",), collect=False)
    assert all(action["symbol"] == "MSFT" for action in proposal["trading_actions"])
    with pytest.raises(ValueError, match="subset"):
        act(data / "bars.csv", data / "events.jsonl", model / "model.pt", symbols=("UNKNOWN",), collect=False)


def test_existing_holdings_require_separate_cash(experiment):
    data, model, _ = paths(experiment)
    with pytest.raises(ValueError, match="cash"):
        act(data / "bars.csv", data / "events.jsonl", model / "model.pt", holdings={"NVDA": 1}, collect=False)
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                   holdings={"NVDA": 1}, cash=90000, collect=False)
    assert proposal["available_cash"] == 90000 and proposal["equity"] > 90000
    assert act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
               holdings={"NVDA": 0}, collect=False)["available_cash"] == 100000


def test_account_loss_history_is_required_before_existing_account_buys(experiment, monkeypatch):
    data, model, _ = paths(experiment)
    monkeypatch.setattr(PolicyAgent, "act", lambda *a, **k: PolicySample(
        TradingAction(1, "NVDA", "buy", 1000, .01), torch.tensor(-.5), torch.tensor(0.)))
    unknown = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                  cash=70000, holdings={"NVDA": 10}, collect=False)
    assert unknown["risk_history_status"] == "missing"
    assert unknown["risk_decision"]["reasons"] == ("risk_history_missing",)
    assert unknown["trading_actions"] == []
    known = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                cash=70000, holdings={"NVDA": 10}, peak_equity=100000, day_start_equity=100000,
                collect=False)
    assert known["risk_history_status"] == "provided"
    assert {"drawdown_limit", "daily_loss_limit"} <= set(known["risk_decision"]["reasons"])
    assert known["trading_actions"] == []


def test_stale_online_market_data_is_rejected_before_any_web_request(experiment, monkeypatch):
    import stocktrade.collection as collection
    data, model, _ = paths(experiment)
    monkeypatch.setattr(collection, "collect_information", lambda *a, **k: pytest.fail("Stale market triggered HTTP"))
    with pytest.raises(ValueError, match="stale"):
        act(data / "bars.csv", data / "events.jsonl", model / "model.pt")


def test_online_collection_uses_current_feature_clock_without_retiming_bars(experiment, monkeypatch):
    import stocktrade.pipeline as pipeline
    import stocktrade.collection as collection
    data, model, _ = paths(experiment)
    metadata = json.loads((model / "model.info.json").read_text())
    raw_bars, initial = load_market(data / "bars.csv", data / "events.jsonl", ("NVDA", "MSFT"),
                                     schemas=metadata["feature_schemas"])
    market_time = raw_bars["NVDA"][-1].timestamp
    now = market_time + timedelta(seconds=30)
    before_hash = file_signature(data / "bars.csv")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(pipeline, "datetime", Clock)
    captured = {}

    def gather(symbols, output, config_path=None):
        assert set(symbols) == {"NVDA", "MSFT"}
        events = [dict(published_at=(now - timedelta(seconds=20)).isoformat(),
                       available_at=(now - timedelta(seconds=10)).isoformat(),
                       expires_at=(now + timedelta(hours=1)).isoformat(),
                       source="fixture://collected", symbol="NVDA", kind="news", values={}, text="New report"),
                  dict(published_at=(now + timedelta(hours=1)).isoformat(), available_at=now.isoformat(),
                       source="fixture://future", symbol="NVDA", kind="news", values={}, text="Future story")]
        from pathlib import Path
        Path(output).write_text("".join(json.dumps(row) + "\n" for row in events))
        return collection.CollectionReport(events_added=2)

    def inspect(_agent, frames, index, state, **kwargs):
        captured.update(frames)
        return PolicySample(TradingAction(0, None, "hold", 0.), torch.tensor(-.5), torch.tensor(0.))

    monkeypatch.setattr(collection, "collect_information", gather)
    monkeypatch.setattr(PolicyAgent, "act", inspect)
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                   collection_output=experiment / "online.jsonl")
    assert proposal["timestamp"] == now and proposal["market_timestamp"] == market_time
    assert captured["NVDA"].timestamps[-1] == now
    assert captured["NVDA"].prices[-1] == initial["NVDA"].prices[-1]
    assert file_signature(data / "bars.csv") == before_hash
    _, baseline = load_market(data / "bars.csv", data / "events.jsonl", ("NVDA", "MSFT"),
                               schemas=metadata["feature_schemas"], as_of=now)
    column = captured["NVDA"].feature_names.index("news.decayed_count")
    # Exactly the eligible story contributes, while the future publication is excluded.
    assert captured["NVDA"].values[-1, column] - baseline["NVDA"].values[-1, column] == pytest.approx(
        np.exp(-10 / (6 * 3600)), abs=1e-6)


def test_historical_evaluation_never_collects_current_web(experiment, monkeypatch):
    import stocktrade.collection as collection
    monkeypatch.setattr(collection, "collect_information", lambda *a, **k: pytest.fail("Historical HTTP lookup"))
    assert run_evaluation(experiment)["objective_name"] == "terminal_net_wealth_growth"


def test_buy_and_protective_sale_share_only_the_final_portfolio_wealth(monkeypatch):
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    prices = [100, 100, 100, 95] + [10] * 10
    bars = [MarketBar(origin + timedelta(minutes=i), "NVDA", price, price, price, price,
                      10000, bid=price, ask=price, bid_size=500, ask_size=500)
            for i, price in enumerate(prices)]
    frame = build_features(bars, [])
    agent = PolicyAgent.create({"NVDA": frame}, ("NVDA",), 6, 1000, PolicyConfig(lookback=3))

    def controlled_actions(frames, index, state, **kwargs):
        action = (TradingAction(1, "NVDA", "buy", 100, .1) if index == 2 else
                  TradingAction(2, "NVDA", "sell", 95, .095) if index == 3 else
                  TradingAction(0, None, "hold", 0.))
        return PolicySample(action, torch.tensor(-.5), torch.tensor(0.))

    monkeypatch.setattr(agent, "act", controlled_actions)
    prints = []
    for index, side in ((2, "buy"), (3, "sell")):
        volatility = float(np.std(np.diff(np.log(prices[:index + 1]))))
        prints.append(TradePrint(bars[index].timestamp + timedelta(milliseconds=200),
            f"SYNTHETIC-{side}", "NVDA", side, 1, prices[index],
            bid=prices[index], ask=prices[index], spread=0., volatility=volatility))
    report, _, feedback = rollout(agent, {"NVDA": bars}, {"NVDA": frame}, prints,
        start=2, stop=len(bars), funding=1000, training=True)
    buy, sell = report["decision_records"][:2]
    assert report["matched_orders"] == 2
    assert "feedback" not in buy and "feedback" not in sell
    assert buy["account_before"]["holdings"] == {}
    assert sell["account_before"]["holdings"] == {"NVDA": 1}
    assert report["final_equity"] == pytest.approx(994.9805)
    assert report["objective"]["wealth_growth"] == pytest.approx(-.0050195)
    assert feedback and all(value == report["objective"]["wealth_growth"] for value in feedback)
    assert report["learning_eligible"] and not report["action_grading"]


def test_rejects_checkpoint_sidecar_mismatch(experiment):
    _, model, _ = paths(experiment)
    with (model / "model.pt").open("ab") as handle:
        handle.write(b"not-saved-checkpoint")
    with pytest.raises(ValueError, match="saved pair"):
        run_evaluation(experiment)


def test_rejects_validation_boundary_change(experiment):
    data, model, _ = paths(experiment)
    path = model / "model.info.json"
    metadata = json.loads(path.read_text())
    metadata["validation_end"] -= 1
    _, frames = load_market(data / "bars.csv", data / "events.jsonl", ("NVDA", "MSFT"),
                           schemas=metadata["feature_schemas"])
    metadata["validation_end_time"] = frames["NVDA"].timestamps[metadata["validation_end"] - 1].isoformat()
    metadata["validation_context_sha256"] = {s: frame_signature(f, metadata["validation_end"])
                                             for s, f in frames.items()}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="training contracts"):
        run_evaluation(experiment)


def test_rejects_changed_training_price_history(experiment):
    data, _, _ = paths(experiment)
    path = data / "bars.csv"
    rows = list(csv.DictReader(path.open()))
    for key in ("open", "high", "low", "close"):
        rows[1][key] = str(float(rows[1][key]) * 1.01)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="training history"):
        run_evaluation(experiment)


def test_future_only_event_does_not_enter_schema_or_change_learning(experiment):
    data, model, _ = paths(experiment)
    original = PolicyAgent.load(model / "model.pt")
    rows = list(csv.DictReader((data / "bars.csv").open()))
    with (data / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(published_at=rows[-1]["timestamp"], available_at=rows[-1]["timestamp"],
            source="future", kind="fundamentals", symbol="NVDA", values={"unknown_key": 1e12})) + "\n")
    alternative = model.with_name("alternative")
    metadata = train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", alternative,
        epochs=1, lookback=4, symbols=("NVDA", "MSFT"))
    modified = PolicyAgent.load(alternative / "model.pt")
    assert "fundamentals.unknown_key" not in metadata["feature_schemas"]["NVDA"]
    assert all(torch.equal(value, modified.model.state_dict()[name])
               for name, value in original.model.state_dict().items())


def test_unaligned_stock_timestamps_are_rejected(experiment):
    data, _, _ = paths(experiment)
    path = data / "bars.csv"
    rows = list(csv.DictReader(path.open()))
    rows.pop(1)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="identical timestamps"):
        load_market(path, data / "events.jsonl", ("NVDA", "MSFT"))


def test_backdated_policy_actions_are_rejected(experiment):
    data, model, _ = paths(experiment)
    path = data / "bars.csv"
    rows = list(csv.DictReader(path.open()))[:100]  # only first50 timestamps for2stocks
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="cutoff"):
        act(path, data / "events.jsonl", model / "model.pt", collect=False)


def test_insufficient_reward_partition_fails_before_training(experiment):
    data, _, _ = paths(experiment)
    with pytest.raises(ValueError, match="Insufficient bars"):
        train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", experiment / "other",
              lookback=1000)
