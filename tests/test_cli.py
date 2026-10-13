"""CLI contract checks; the pipeline tests exercise real policy learning."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from stocktrade import demo, pipeline
from stocktrade.cli import main


@pytest.fixture
def workflows(monkeypatch):
    functions = {
        "train": Mock(return_value={"symbols": ["NVDA", "MSFT"], "training_objective": "action_rewards"}),
        "evaluate": Mock(return_value={
            "decisions": 4, "matched_orders": 2,
            "execution": {"total_fees": .1, "records": [1], "equity_curve": [2], "source_trade_ids": [3]},
        }),
        "act": Mock(return_value={"trading_actions": [{"symbol": "NVDA", "side": "buy", "value": 100}]}),
        "write_json": Mock(),
    }
    for name, function in functions.items():
        monkeypatch.setattr(pipeline, name, function, raising=False)
    return functions


def test_training_requires_recorded_execution_data():
    with pytest.raises(SystemExit) as error:
        main(["train", "--bars", "bars.csv", "--output", "model"])
    assert error.value.code == 2


def test_training_forwards_funding_universe_and_action_reward_settings(workflows, capsys):
    assert main([
        "train", "--bars", "bars.csv", "--events", "events.jsonl", "--trades", "trades.csv",
        "--output", "model", "--funding", "2000", "--symbols", "NVDA", "MSFT",
        "--reward-horizon", "12", "--selection-drawdown-weight", ".9", "--constraint-penalty", ".02",
        "--episode-steps", "32", "--lookback", "8", "--epochs", "2",
        "--train-fraction", ".6", "--validation-fraction", ".2",
    ]) == 0
    workflows["train"].assert_called_once_with(
        "bars.csv", "events.jsonl", "trades.csv", "model", epochs=2, seed=7,
        lookback=8, funding=2000.0, symbols=("NVDA", "MSFT"), reward_horizon=12,
        selection_drawdown_weight=.9, constraint_penalty=.02, episode_steps=32,
        train_fraction=.6, validation_fraction=.2,
    )
    assert json.loads(capsys.readouterr().out)["training_objective"] == "action_rewards"


@pytest.mark.parametrize("funding_flag", ["--funding", "--cash"])
def test_evaluation_passes_funding_and_summarizes_large_evidence(workflows, capsys, funding_flag):
    assert main([
        "evaluate", "--bars", "bars.csv", "--trades", "trades.csv", "--checkpoint", "model.pt",
        "--output", "evaluation", funding_flag, "5000", "--symbols", "NVDA",
    ]) == 0
    workflows["evaluate"].assert_called_once_with(
        "bars.csv", None, "trades.csv", "model.pt", "evaluation", funding=5000.0, symbols=("NVDA",),
    )
    report = json.loads(capsys.readouterr().out)
    assert report["execution"] == {"total_fees": .1}
    assert report["report_path"] == str(Path("evaluation/report.json"))


def test_act_uses_checkpoint_defaults_and_does_not_require_historical_trade_file(workflows, capsys):
    assert main(["act", "--bars", "bars.csv", "--checkpoint", "model.pt"]) == 0
    workflows["act"].assert_called_once_with(
        "bars.csv", None, "model.pt", funding=None, symbols=None, cash=None, holdings=None,
    )
    workflows["write_json"].assert_not_called()
    assert json.loads(capsys.readouterr().out)["trading_actions"][0]["value"] == 100


def test_act_accepts_portfolio_state_and_saves_optional_output(workflows, tmp_path, capsys):
    holdings = tmp_path / "holdings.json"
    holdings.write_text(json.dumps({"NVDA": 3, "MSFT": 2}))
    output = str(tmp_path / "nested/actions.json")
    assert main([
        "act", "--bars", "bars.csv", "--events", "events.jsonl", "--checkpoint", "model.pt",
        "--funding", "2500", "--symbols", "NVDA", "MSFT", "--cash", "1600",
        "--holdings", str(holdings), "--output", output,
    ]) == 0
    workflows["act"].assert_called_once_with(
        "bars.csv", "events.jsonl", "model.pt", funding=2500.0, symbols=("NVDA", "MSFT"),
        cash=1600.0, holdings={"NVDA": 3, "MSFT": 2},
    )
    workflows["write_json"].assert_called_once_with(output, workflows["act"].return_value)
    assert json.loads(capsys.readouterr().out)["report_path"] == output


def test_nonempty_positions_require_explicit_current_cash(workflows, tmp_path, capsys):
    holdings = tmp_path / "holdings.json"
    holdings.write_text('{"NVDA": 3}')
    assert main(["act", "--bars", "bars.csv", "--checkpoint", "model.pt",
                 "--holdings", str(holdings)]) == 1
    assert "--cash is required" in capsys.readouterr().err
    workflows["act"].assert_not_called()


@pytest.mark.parametrize("contents", ["{bad-json", "[]", '{"NVDA": 1.5}', '{"NVDA": true}',
                                       '{"NVDA": -1}', '{"": 1}'])
def test_invalid_holdings_fail_before_action_proposal(workflows, tmp_path, capsys, contents):
    holdings = tmp_path / "holdings.json"
    holdings.write_text(contents)
    assert main(["act", "--bars", "bars.csv", "--checkpoint", "model.pt",
                 "--holdings", str(holdings), "--cash", "1000"]) == 1
    assert capsys.readouterr().err.startswith("stocktrade: ")
    workflows["act"].assert_not_called()


def test_pipeline_invalid_funding_error_is_actionable(workflows, capsys):
    workflows["train"].side_effect = ValueError("Funding must be positive and finite")
    assert main(["train", "--bars", "bars.csv", "--trades", "trades.csv",
                 "--output", "model", "--funding", "-5"]) == 1
    assert "Funding must be positive and finite" in capsys.readouterr().err


def test_demo_shares_funding_and_universe_between_training_and_evaluation(
        workflows, monkeypatch, tmp_path, capsys):
    data = tmp_path / "synthetic"
    generate = Mock(return_value=data)
    monkeypatch.setattr(demo, "generate_demo", generate)
    assert main(["demo", "--output", str(tmp_path), "--epochs", "1", "--frequency-seconds", "1",
                 "--funding", "2500", "--symbols", "NVDA", "MSFT", "--reward-horizon", "4"]) == 0
    generate.assert_called_once_with(tmp_path / "synthetic", frequency_seconds=1, seed=7,
                                     symbols=("NVDA", "MSFT"))
    workflows["train"].assert_called_once_with(
        data / "bars.csv", data / "events.jsonl", data / "trades.csv", tmp_path / "model",
        epochs=1, seed=7, funding=2500.0, symbols=("NVDA", "MSFT"), reward_horizon=4,
    )
    workflows["evaluate"].assert_called_once_with(
        data / "bars.csv", data / "events.jsonl", data / "trades.csv",
        tmp_path / "model/model.pt", tmp_path / "evaluation", funding=2500.0, symbols=("NVDA", "MSFT"),
    )
    report = json.loads(capsys.readouterr().out)
    assert "SYNTHETIC" in report["demo_warning"]
    assert report["report_path"] == str(tmp_path / "evaluation/report.json")


def test_news_collection_remains_available_without_policy_workflow(monkeypatch, capsys):
    collector = Mock(return_value=3)
    monkeypatch.setattr("stocktrade.data.collect_rss", collector)
    assert main(["collect-news", "--url", "https://example.org/feed", "--output", "news.jsonl",
                 "--symbol", "NVDA"]) == 0
    collector.assert_called_once_with("https://example.org/feed", "news.jsonl", symbol="NVDA")
    assert json.loads(capsys.readouterr().out) == {"collected_events": 3}
