"""Report integration verifies objective isolation and auditable provenance."""

import json

import pytest

from stocktrade.cli import main
from stocktrade.demo import generate_demo
from stocktrade.diagnostics import EvaluationConfig, regenerate_report
from stocktrade.pipeline import evaluate, file_signature, train


@pytest.mark.parametrize("kwargs", [
    {"risk_lookback": True}, {"risk_min_observations": 1}, {"risk_lookback": 10},
    {"risk_confidence": 1}, {"risk_confidence": float("nan")},
    {"risk_free_rate": float("inf")}, {"benchmark_symbol": ""},
])
def test_invalid_diagnostic_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("diagnostic-objective-isolation")
    data = generate_demo(root / "data", count=100, symbols=("NVDA", "MSFT"))
    model = root / "model"
    train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", model,
          epochs=1, lookback=4, symbols=("NVDA", "MSFT"))
    return root


def _evaluate(root, output, **kwargs):
    return evaluate(root / "data/bars.csv", root / "data/events.jsonl", root / "data/trades.csv",
        root / "model/model.pt", output, **kwargs)


def test_diagnostics_change_neither_wealth_nor_checkpoint_or_policy(experiment, tmp_path, monkeypatch):
    import stocktrade.analytics as analytics
    import stocktrade.decision_risk as decision_risk
    from stocktrade.model import PolicyAgent

    checkpoint = experiment / "model/model.pt"
    before = file_signature(checkpoint)
    first = _evaluate(experiment, tmp_path / "original", dashboard=False)
    real_estimator = decision_risk.estimate_decision_risk

    def extravagant_estimate(**kwargs):
        result = real_estimator(**kwargs)
        result["proposal"]["estimates"]["commission_cost"] = 1_000_000_000
        return result

    monkeypatch.setattr(decision_risk, "estimate_decision_risk", extravagant_estimate)
    monkeypatch.setattr(PolicyAgent, "update", lambda *a, **k: pytest.fail("Evaluation must not update weights"))
    second = _evaluate(experiment, tmp_path / "extravagant", dashboard=False)
    assert first["objective"] == second["objective"]
    assert first["execution"] == second["execution"]
    assert file_signature(checkpoint) == before
    assert second["diagnostics"]["used_for_training"] is False
    assert second["diagnostics"]["used_for_checkpoint_selection"] is False
    decisions = [json.loads(line) for line in (tmp_path / "extravagant/decisions.jsonl").read_text().splitlines()]
    assert len(decisions) == second["decisions"]
    assert all(d["risk_estimate"]["proposal"]["estimates"]["commission_cost"] == 1_000_000_000 for d in decisions)
    assert (tmp_path / "extravagant/decision_risk.csv").exists()
    assert callable(analytics.build_performance_report)


def test_training_never_calls_diagnostic_or_indicator_functions(tmp_path, monkeypatch):
    import stocktrade.analytics as analytics
    import stocktrade.decision_risk as decision_risk
    import stocktrade.diagnostics as diagnostics

    def forbidden(*args, **kwargs):
        pytest.fail("Training/checkpoint selection must not consume diagnostic metrics")

    monkeypatch.setattr(analytics, "build_performance_report", forbidden)
    monkeypatch.setattr(decision_risk, "estimate_decision_risk", forbidden)
    monkeypatch.setattr(diagnostics, "write_evaluation_artifacts", forbidden)
    data = generate_demo(tmp_path / "data", count=100, symbols=("NVDA", "MSFT"))
    metadata = train(data / "bars.csv", data / "events.jsonl", data / "trades.csv", tmp_path / "model",
                     epochs=1, lookback=4, symbols=("NVDA", "MSFT"))
    assert metadata["training_objective"] == "terminal_net_wealth_growth"
    assert metadata["updated_episodes"] > 0
    assert "risk_estimate" not in (tmp_path / "model/training_decisions.jsonl").read_text()


def test_archive_regeneration_preserves_source_and_objective(experiment, tmp_path):
    output = tmp_path / "source"
    original = _evaluate(experiment, output, dashboard=False)
    signatures = [file_signature(output / name) for name in ("report.json", "decisions.jsonl")]
    result = regenerate_report(experiment / "data/bars.csv", output / "report.json", output / "decisions.jsonl",
        tmp_path / "rebuilt", config=EvaluationConfig(risk_confidence=.90), dashboard=False)
    assert json.loads(json.dumps(original["objective"])) == result["objective"]
    assert result["diagnostics"]["config"]["risk_confidence"] == .90
    assert result["regenerated_from"]["decision_binding"] == "verified_bound_archive"
    assert result["provenance"]["decisions_sha256"] == file_signature(tmp_path / "rebuilt/decisions.jsonl")
    assert signatures == [file_signature(output / name) for name in ("report.json", "decisions.jsonl")]
    with pytest.raises(ValueError, match="new directory"):
        regenerate_report(experiment / "data/bars.csv", output / "report.json", output / "decisions.jsonl", output)


def test_missing_benchmark_is_unavailable_without_invented_observations(experiment, tmp_path):
    report = _evaluate(experiment, tmp_path, dashboard=False,
                       evaluation_config=EvaluationConfig(benchmark_symbol="SPY"))
    assert report["diagnostics"]["config"]["benchmark_symbol"] == "SPY"
    assert "SPY" not in report["universe"]
    metrics = report["diagnostics"]["performance"]
    assert metrics["benchmark"]["price_return"] is None
    assert metrics["benchmark"]["series"] == []
    assert "not supplied" in metrics["availability"]["benchmark"]


def test_report_rejects_misaligned_decision_and_changed_market_history(experiment, tmp_path):
    output = tmp_path / "source"
    _evaluate(experiment, output, dashboard=False)
    source = experiment / "data/bars.csv"
    altered = tmp_path / "altered-bars.csv"
    altered.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance"):
        regenerate_report(altered, output / "report.json", output / "decisions.jsonl", tmp_path / "reject")
    records = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    records[0]["bar_index"] = 0
    bad = tmp_path / "bad-decisions.jsonl"
    bad.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match="decision audit provenance"):
        regenerate_report(source, output / "report.json", bad, tmp_path / "reject-index", dashboard=False)


def test_legacy_archive_rejects_another_account_path(experiment, tmp_path):
    output = tmp_path / "source"
    _evaluate(experiment, output, dashboard=False)
    report = json.loads((output / "report.json").read_text())
    report["provenance"].pop("decisions_sha256")
    legacy = output / "legacy.json"
    legacy.write_text(json.dumps(report))
    records = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    records[0]["account_before"]["cash"] -= 10
    altered = output / "altered-decisions.jsonl"
    altered.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match="observed portfolio path"):
        regenerate_report(experiment / "data/bars.csv", legacy, altered, tmp_path / "reject", dashboard=False)


def test_cli_diagnostic_options_do_not_pass_to_training(monkeypatch, capsys):
    import stocktrade.pipeline as pipeline
    captured = {}

    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    assert main(["evaluate", "--bars", "bars", "--trades", "trades", "--checkpoint", "checkpoint",
                 "--output", "output", "--benchmark", "SPY", "--risk-confidence", ".90",
                 "--risk-lookback", "100", "--risk-min-observations", "30", "--no-dashboard"]) == 0
    config = captured["evaluation_config"]
    assert config == EvaluationConfig(risk_confidence=.90, risk_lookback=100,
                                      risk_min_observations=30, benchmark_symbol="SPY")
    assert captured["dashboard"] is False
    assert "report_path" in capsys.readouterr().out
