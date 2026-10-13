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
    _action_advantages, act, evaluate, file_signature, frame_signature, load_market, rollout, train,
)
from stocktrade.policy_contracts import TradingAction
from stocktrade.rewards import RewardConfig
from stocktrade.schemas import MarketBar


@pytest.fixture(scope="module")
def source_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("direct-policy")
    data = generate_demo(root / "data", count=100, symbols=("NVDA", "MSFT"))
    model = root / "model"
    train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", model,
          epochs=1, lookback=4, reward_horizon=3, symbols=("NVDA", "MSFT"))
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
    assert metadata["training_supported_feedback"] > 0
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
    assert report["decisions"] == 12
    assert report["source_is_synthetic"] and not report["policy_updated"]
    assert "heldout_return_mae" not in report
    assert report["universe"] == ["NVDA", "MSFT"]
    assert file_signature(model / "model.pt") == before
    records = [json.loads(row) for row in (output / "decisions.jsonl").read_text().splitlines()]
    assert len(records) == 12
    assert all("policy_action" in record and "forecast" not in record for record in records)


def test_all_training_feedback_matures_within_training_partition(experiment):
    data, model, _ = paths(experiment)
    metadata = json.loads((model / "model.info.json").read_text())
    _, frames = load_market(data / "bars.csv", data / "events.jsonl", ("NVDA", "MSFT"),
                           schemas=metadata["feature_schemas"])
    cutoff = datetime.fromisoformat(metadata["train_end_time"])
    records = [json.loads(row) for row in (model / "training_feedback.jsonl").read_text().splitlines()]
    for record in records:
        feedback = record["feedback"]
        if feedback["status"] == "supported":
            maturity = datetime.fromisoformat(feedback["maturity_timestamp"])
            assert maturity <= cutoff
            assert maturity == frames["NVDA"].timestamps[record["bar_index"] + 3]
        elif feedback["status"] == "unmatched":
            assert feedback["reward"] is None


def test_good_sale_never_rewards_bad_entry_and_missing_feedback_stays_missing():
    records = [dict(bar_index=0, feedback={"reward": -.05}),
               dict(bar_index=2, feedback={"reward": None}),
               dict(bar_index=5, feedback={"reward": .85})]
    feedback = _action_advantages(records)
    assert feedback == {0: -.05, 2: .85}
    assert 1 not in feedback


def test_act_outputs_dollar_trade_for_permitted_symbol(experiment, monkeypatch):
    data, model, _ = paths(experiment)

    def fixed_action(*args, **kwargs):
        return PolicySample(TradingAction(1, "NVDA", "buy", 100, .01),
                            torch.tensor(-.5), torch.tensor(0.))

    monkeypatch.setattr(PolicyAgent, "act", fixed_action)
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                   funding=10000, symbols=("NVDA",))
    assert proposal["total_funding"] == 10000 and proposal["allowed_symbols"] == ["NVDA"]
    assert proposal["policy_action"]["value"] == 100
    assert len(proposal["trading_actions"]) == 1
    trade = proposal["trading_actions"][0]
    assert trade["symbol"] == "NVDA" and trade["side"] == "buy"
    assert 0 < trade["value"] <= 100 and trade["quantity"] == 1


def test_runtime_subset_is_respected_and_unknown_symbols_rejected(experiment):
    data, model, _ = paths(experiment)
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt", symbols=("MSFT",))
    assert all(action["symbol"] == "MSFT" for action in proposal["trading_actions"])
    with pytest.raises(ValueError, match="subset"):
        act(data / "bars.csv", data / "events.jsonl", model / "model.pt", symbols=("UNKNOWN",))


def test_existing_holdings_require_separate_cash(experiment):
    data, model, _ = paths(experiment)
    with pytest.raises(ValueError, match="cash"):
        act(data / "bars.csv", data / "events.jsonl", model / "model.pt", holdings={"NVDA": 1})
    proposal = act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
                   holdings={"NVDA": 1}, cash=90000)
    assert proposal["available_cash"] == 90000 and proposal["equity"] > 90000
    assert act(data / "bars.csv", data / "events.jsonl", model / "model.pt",
               holdings={"NVDA": 0})["available_cash"] == 100000


def test_actual_loss_cutting_sale_never_turns_the_bad_purchase_reward_positive(monkeypatch):
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
        start=2, stop=len(bars), funding=1000, reward_config=RewardConfig(horizon_steps=8, fee_bps=1.))
    buy, sell = report["feedback_records"][:2]
    assert report["supported_trade_feedback"] == 2
    assert buy["feedback"]["reward"] < 0 < sell["feedback"]["reward"]
    assert feedback[0] == buy["feedback"]["reward"]
    assert feedback[1] == sell["feedback"]["reward"]
    assert buy["account_before"]["holdings"] == {}
    assert sell["account_before"]["holdings"] == {"NVDA": 1}
    assert buy["feedback"]["components"]["terminal_actual_vs_hold_advantage"] == pytest.approx(-5.0195)
    assert sell["feedback"]["components"]["terminal_actual_vs_hold_advantage"] == pytest.approx(84.9905)
    assert buy["feedback"]["source_trade_ids"] == ("SYNTHETIC-buy", "SYNTHETIC-sell")
    assert report["final_equity"] == pytest.approx(994.9805)


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
        epochs=1, lookback=4, reward_horizon=3, symbols=("NVDA", "MSFT"))
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
        act(path, data / "events.jsonl", model / "model.pt")


def test_insufficient_reward_partition_fails_before_training(experiment):
    data, _, _ = paths(experiment)
    with pytest.raises(ValueError, match="Insufficient bars"):
        train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", experiment / "other",
              lookback=1000)
