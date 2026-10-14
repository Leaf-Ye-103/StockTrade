"""CLI contract checks; the pipeline tests exercise real policy learning."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from stocktrade import collection, demo, pipeline, platforms
from stocktrade.cli import main
from stocktrade.risk import RiskConfig


@pytest.fixture
def workflows(monkeypatch):
    functions = {
        "train": Mock(return_value={"symbols": ["NVDA", "MSFT"], "training_objective": "terminal_net_wealth"}),
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


@pytest.mark.parametrize("flag", ["--reward-horizon", "--constraint-penalty",
                                 "--selection-drawdown-weight", "--gamma", "--platform-weight"])
def test_obsolete_action_grading_or_fixed_platform_priority_flags_are_rejected(flag):
    with pytest.raises(SystemExit) as error:
        main(["train", "--bars", "bars.csv", "--trades", "trades.csv", "--output", "model", flag, "1"])
    assert error.value.code == 2


def test_training_forwards_funding_universe_and_wealth_episode_settings(workflows, capsys):
    assert main([
        "train", "--bars", "bars.csv", "--events", "events.jsonl", "--trades", "trades.csv",
        "--output", "model", "--funding", "2000", "--symbols", "NVDA", "MSFT",
        "--episode-steps", "32", "--lookback", "8", "--epochs", "2",
        "--train-fraction", ".6", "--validation-fraction", ".2",
    ]) == 0
    workflows["train"].assert_called_once_with(
        "bars.csv", "events.jsonl", "trades.csv", "model", epochs=2, seed=7,
        lookback=8, funding=2000.0, symbols=("NVDA", "MSFT"), risk_config=None, episode_steps=32,
        train_fraction=.6, validation_fraction=.2,
    )
    assert json.loads(capsys.readouterr().out)["training_objective"] == "terminal_net_wealth"


def test_training_loads_validated_hard_risk_overrides(workflows, tmp_path):
    risk = tmp_path / "risk.json"
    risk.write_text('{"max_position_fraction": 0.1, "minimum_cash_fraction": 0.3, "lot_size": 2}')
    assert main(["train", "--bars", "bars.csv", "--trades", "trades.csv", "--output", "model",
                 "--risk-config", str(risk)]) == 0
    assert workflows["train"].call_args.kwargs["risk_config"] == RiskConfig(
        max_position_fraction=.1, minimum_cash_fraction=.3, lot_size=2,
    )


@pytest.mark.parametrize("contents,message", [
    ('{"unknown_field": 1}', "Unknown risk configuration fields: unknown_field"),
    ("[]", "Risk configuration must be a JSON object"),
    ('{"max_drawdown": 2}', "Risk fractions must be finite and between zero and one"),
    ('{"minimum_cash_fraction": "high"}', "Risk configuration values must be numeric"),
    ('{"lot_size": true}', "Risk configuration values must be numeric"),
    ('{"lot_size": 1.5}', "lot_size must be a positive integer"),
])
def test_invalid_risk_configuration_fails_before_training(workflows, tmp_path, capsys, contents, message):
    risk = tmp_path / "risk.json"
    risk.write_text(contents)
    assert main(["train", "--bars", "bars.csv", "--trades", "trades.csv", "--output", "model",
                 "--risk-config", str(risk)]) == 1
    assert message in capsys.readouterr().err
    workflows["train"].assert_not_called()


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
        peak_equity=None, day_start_equity=None,
        collect=True, sources_path=None, collection_output="data/online-events.jsonl", max_market_age_seconds=900.0,
    )
    workflows["write_json"].assert_not_called()
    assert json.loads(capsys.readouterr().out)["trading_actions"][0]["value"] == 100


def test_act_accepts_portfolio_state_and_saves_optional_output(workflows, tmp_path, capsys):
    workflows["act"].return_value = {
        "trading_actions": [], "risk_history_status": "missing",
        "risk_decision": {"accepted": False, "reasons": ["risk_history_missing"]},
    }
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
        peak_equity=None, day_start_equity=None,
        collect=True, sources_path=None, collection_output="data/online-events.jsonl", max_market_age_seconds=900.0,
    )
    workflows["write_json"].assert_called_once_with(output, workflows["act"].return_value)
    report = json.loads(capsys.readouterr().out)
    assert report["report_path"] == output
    assert report["trading_actions"] == []
    assert report["risk_history_status"] == "missing"


def test_act_can_disable_online_collection_and_configure_observation_output(workflows):
    assert main(["act", "--bars", "bars.csv", "--checkpoint", "model.pt", "--offline",
                 "--sources", "sources.json", "--collection-output", "data/observed.jsonl",
                 "--max-market-age-seconds", "300"]) == 0
    workflows["act"].assert_called_once_with(
        "bars.csv", None, "model.pt", funding=None, symbols=None, cash=None, holdings=None,
        peak_equity=None, day_start_equity=None,
        collect=False, sources_path="sources.json", collection_output="data/observed.jsonl",
        max_market_age_seconds=300.0,
    )


def test_act_forwards_actual_peak_and_day_start_account_history(workflows, tmp_path):
    holdings = tmp_path / "holdings.json"
    holdings.write_text('{"NVDA": 3}')
    assert main(["act", "--bars", "bars.csv", "--checkpoint", "model.pt", "--offline",
                 "--funding", "2500", "--cash", "1600", "--holdings", str(holdings),
                 "--peak-equity", "2800", "--day-start-equity", "2600"]) == 0
    workflows["act"].assert_called_once_with(
        "bars.csv", None, "model.pt", funding=2500.0, symbols=None, cash=1600.0, holdings={"NVDA": 3},
        peak_equity=2800.0, day_start_equity=2600.0,
        collect=False, sources_path=None, collection_output="data/online-events.jsonl", max_market_age_seconds=900.0,
    )


def test_online_act_reports_stale_market_error_without_output_file(workflows, capsys):
    workflows["act"].side_effect = ValueError("Online actions require recent market bars")
    assert main(["act", "--bars", "bars.csv", "--checkpoint", "model.pt", "--output", "actions.json"]) == 1
    assert "Online actions require recent market bars" in capsys.readouterr().err
    workflows["write_json"].assert_not_called()


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
                 "--funding", "2500", "--symbols", "NVDA", "MSFT", "--episode-steps", "64"]) == 0
    generate.assert_called_once_with(tmp_path / "synthetic", frequency_seconds=1, seed=7,
                                     symbols=("NVDA", "MSFT"))
    workflows["train"].assert_called_once_with(
        data / "bars.csv", data / "events.jsonl", data / "trades.csv", tmp_path / "model",
        epochs=1, seed=7, funding=2500.0, symbols=("NVDA", "MSFT"), episode_steps=64, risk_config=None,
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


@pytest.mark.parametrize("sources", [None, "sources.json"])
def test_information_collection_preserves_partial_success_and_source_errors(monkeypatch, capsys, sources):
    report = collection.CollectionReport(
        events_added=2,
        source_reports=[{"source": "news", "status": "collected", "events_added": 2}],
        errors=[{"source": "fed", "error": "HTTPS proxy denied the request"}],
        requested_domains=["example.org", "www.federalreserve.gov"],
    )
    collector = Mock(return_value=report)
    monkeypatch.setattr(collection, "collect_information", collector)
    args = ["collect-information", "--symbols", "NVDA", "MSFT", "--output", "data/events.jsonl"]
    if sources:
        args.extend(["--sources", sources])
    assert main(args) == 0
    collector.assert_called_once_with(("NVDA", "MSFT"), "data/events.jsonl", config_path=sources)
    assert json.loads(capsys.readouterr().out) == report.to_dict()


def test_information_collection_requires_a_stock_universe():
    with pytest.raises(SystemExit) as error:
        main(["collect-information", "--output", "data/events.jsonl"])
    assert error.value.code == 2


@pytest.mark.parametrize("feed", [None, "sip"])
def test_alpaca_bar_import_routes_read_only_arguments_without_policy_workflow(
        workflows, monkeypatch, capsys, feed):
    importer = Mock(return_value={"provider": "Alpaca", "bars": 12, "observed_execution_supported": False})
    monkeypatch.setattr(platforms, "fetch_alpaca_bars", importer)
    args = ["fetch-alpaca-bars", "--symbols", "NVDA", "MSFT", "--start", "2026-10-01T13:30:00Z",
            "--end", "2026-10-01T20:00:00Z", "--output", "data/alpaca", "--timeout", "20"]
    if feed:
        args.extend(["--feed", feed])
    assert main(args) == 0
    importer.assert_called_once_with(
        ("NVDA", "MSFT"), "2026-10-01T13:30:00Z", "2026-10-01T20:00:00Z", "data/alpaca",
        feed=feed or "iex", timeout=20.0,
    )
    assert json.loads(capsys.readouterr().out) == importer.return_value
    for workflow in workflows.values():
        workflow.assert_not_called()


def test_alpaca_news_import_routes_read_only_arguments_without_policy_workflow(workflows, monkeypatch, capsys):
    importer = Mock(return_value={"provider": "Alpaca", "events": 2, "availability": "retrieval_time"})
    monkeypatch.setattr(platforms, "fetch_alpaca_news", importer)
    assert main(["fetch-alpaca-news", "--symbols", "NVDA", "--start", "2026-10-01T13:30:00+00:00",
                 "--end", "2026-10-01T20:00:00+00:00", "--output", "data/alpaca-news.jsonl"]) == 0
    importer.assert_called_once_with(
        ("NVDA",), "2026-10-01T13:30:00+00:00", "2026-10-01T20:00:00+00:00", "data/alpaca-news.jsonl",
        timeout=15.0,
    )
    assert json.loads(capsys.readouterr().out) == importer.return_value
    for workflow in workflows.values():
        workflow.assert_not_called()


@pytest.mark.parametrize("command", ["fetch-alpaca-bars", "fetch-alpaca-news"])
def test_alpaca_import_requires_a_range_and_handles_missing_environment_bindings(monkeypatch, capsys, command):
    with pytest.raises(SystemExit) as error:
        main([command, "--symbols", "NVDA", "--output", "data/alpaca"])
    assert error.value.code == 2
    capsys.readouterr()
    importer = Mock(side_effect=ValueError("Missing Alpaca environment variables: APCA_API_KEY_ID"))
    monkeypatch.setattr(platforms, command.replace("-", "_"), importer)
    assert main([command, "--symbols", "NVDA", "--start", "2026-10-01T13:30:00Z",
                 "--end", "2026-10-01T20:00:00Z", "--output", "data/alpaca"]) == 1
    assert "Missing Alpaca environment variables: APCA_API_KEY_ID" in capsys.readouterr().err


def test_alpaca_feed_is_limited_to_supported_market_feeds():
    with pytest.raises(SystemExit) as error:
        main(["fetch-alpaca-bars", "--symbols", "NVDA", "--start", "2026-10-01T13:30:00Z",
              "--end", "2026-10-01T20:00:00Z", "--output", "data/alpaca", "--feed", "unsupported"])
    assert error.value.code == 2
