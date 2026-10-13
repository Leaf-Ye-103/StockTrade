import csv
import json

import pytest
import torch

from stocktrade.demo import generate_demo
from stocktrade.model import TrainingResult
from stocktrade.pipeline import evaluate, frame_signature, load_frame, train
from stocktrade.risk import Forecast, SizingDecision


@pytest.fixture
def experiment(tmp_path):
    data = generate_demo(tmp_path / "data", count=120, frequency_seconds=1)
    model = tmp_path / "model"
    train(data / "bars.csv", data / "events.jsonl", model, epochs=1, lookback=4)
    return data, model, tmp_path / "evaluation"


def run_evaluation(experiment):
    data, model, output = experiment
    return evaluate(data / "bars.csv", data / "events.jsonl", data / "trades.csv",
                    model / "model.pt", output)


def test_real_training_checkpoint_and_heldout_inference(experiment):
    report = run_evaluation(experiment)
    _, _, output = experiment
    assert report["decisions"] == 18
    assert report["source_is_synthetic"] is True
    assert report["heldout_return_mae"] >= 0
    assert len(report["provenance"]["checkpoint_sha256"]) == 64
    assert len((output / "decisions.jsonl").read_text().splitlines()) == 18
    assert json.loads((output / "report.json").read_text())["final_equity"] == report["final_equity"]


def test_future_only_event_key_does_not_change_trained_architecture_or_weights(experiment):
    data, model, _ = experiment
    original = TrainingResult.load(model / "model.pt")
    rows = list(csv.DictReader((data / "bars.csv").open()))
    with (data / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(published_at=rows[-1]["timestamp"], available_at=rows[-1]["timestamp"],
            source="future", kind="fundamentals", symbol="DEMO", values={"previously_unknown": 9e12})) + "\n")
    alternative = model.with_name("alternative")
    metadata = train(data / "bars.csv", data / "events.jsonl", alternative, epochs=1, lookback=4)
    modified = TrainingResult.load(alternative / "model.pt")
    assert metadata["feature_names"] == original.scaler.feature_names
    assert "fundamentals.previously_unknown" not in metadata["feature_names"]
    for name, value in original.model.state_dict().items():
        assert torch.equal(value, modified.model.state_dict()[name])
    assert run_evaluation(experiment)["decisions"] == 18


def test_rejects_changed_training_history(experiment):
    data, _, _ = experiment
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


def test_rejects_checkpoint_sidecar_mismatch(experiment):
    _, model, _ = experiment
    with (model / "model.pt").open("ab") as handle:
        handle.write(b"not-the-saved-checkpoint")
    with pytest.raises(ValueError, match="saved pair"):
        run_evaluation(experiment)


def test_rejects_sidecar_boundaries_even_when_history_hash_matches(experiment):
    data, model, _ = experiment
    path = model / "model.info.json"
    metadata = json.loads(path.read_text())
    metadata["validation_end"] -= 1
    _, frame = load_frame(data / "bars.csv", data / "events.jsonl", feature_names=metadata["feature_names"])
    metadata["validation_context_sha256"] = frame_signature(frame, metadata["validation_end"])
    metadata["validation_end_time"] = frame.timestamps[metadata["validation_end"] - 1].isoformat()
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="boundaries"):
        run_evaluation(experiment)


def test_matched_records_update_next_decision_account_only_after_event(experiment, monkeypatch):
    # A deterministic policy is used to exercise accounting, not to assert performance.
    calls = []

    def fixed_policy(forecast, market, account, config):
        calls.append(account)
        side = "BUY" if len(calls) == 1 else "SELL" if len(calls) == 2 else "HOLD"
        quantity = 10 if len(calls) <= 2 else 0
        return SizingDecision(side, quantity, quantity * market.price, 0.1, ())

    monkeypatch.setattr("stocktrade.pipeline.size_position", fixed_policy)
    monkeypatch.setattr(TrainingResult, "predict", lambda *a, **kw: Forecast(.01, .001, -.003, .9, .1))
    report = run_evaluation(experiment)
    assert report["proposed_orders"] == report["matched_orders"] == 2
    assert calls[0].position_quantity == 0
    assert calls[1].position_quantity == 10
    assert calls[2].position_quantity == 0
    assert report["execution"]["total_fees"] > 0
    records = report["execution"]["records"]
    assert records[0]["match"]["event_timestamp"] < records[1]["intent"]["timestamp"]
    assert len(report["execution"]["source_trade_ids"]) == 2


def test_cli_bad_window_reports_actionable_error(tmp_path, capsys):
    from stocktrade.cli import main
    data = generate_demo(tmp_path / "data", count=100)
    status = main(["train", "--bars", str(data / "bars.csv"), "--output", str(tmp_path / "model"),
                   "--lookback", "1000"])
    assert status == 1
    assert "Insufficient bars" in capsys.readouterr().err
