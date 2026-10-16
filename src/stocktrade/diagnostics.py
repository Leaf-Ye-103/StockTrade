"""Evaluation-only risk estimates, report files and archived report regeneration.

This module is deliberately absent from training and checkpoint selection.
Estimated costs never enter the ledger or the terminal-wealth learning signal.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .data import load_bars_csv
from .risk import RiskConfig


@dataclass(frozen=True)
class EvaluationConfig:
    risk_lookback: int = 60
    risk_confidence: float = .95
    risk_min_observations: int = 20
    benchmark_symbol: str | None = None
    risk_free_rate: float = 0.0

    def __post_init__(self):
        for name in ("risk_lookback", "risk_min_observations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.risk_min_observations > self.risk_lookback:
            raise ValueError("Risk lookback must cover the minimum observation count")
        if (isinstance(self.risk_confidence, bool) or not math.isfinite(self.risk_confidence)
                or not .5 < self.risk_confidence < 1):
            raise ValueError("Risk confidence must lie strictly between 0.5 and 1")
        if (isinstance(self.risk_free_rate, bool) or not math.isfinite(self.risk_free_rate)
                or self.risk_free_rate <= -1):
            raise ValueError("Annual risk-free rate must be finite and greater than -1")
        if self.benchmark_symbol is not None and (
                not isinstance(self.benchmark_symbol, str) or not self.benchmark_symbol.strip()):
            raise ValueError("Benchmark must be a nonempty stock symbol")


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def _signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Report timestamps must include a UTC offset")
    return result


def include_benchmark(bars_path, bars, symbol):
    """Add an observed benchmark only if it covers the exact policy clock.

    An unavailable benchmark stays absent, so analytics records missingness.
    Never fill a quote/price or add a reference stock to the policy's universe.
    """
    result = dict(bars)
    if not symbol or symbol in result:
        return result
    source = sorted((bar for bar in load_bars_csv(bars_path) if bar.symbol == symbol),
                    key=lambda bar: bar.timestamp)
    by_time = {bar.timestamp: bar for bar in source}
    clock = [bar.timestamp for bar in next(iter(bars.values()))]
    if len(by_time) == len(source) and all(timestamp in by_time for timestamp in clock):
        result[symbol] = [by_time[timestamp] for timestamp in clock]
    return result


def _csv_value(value):
    # Keep exported untrusted symbols/reasons from becoming spreadsheet formulas.
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _risk_rows(records):
    for record in records:
        estimate = record["risk_estimate"]
        before, projected, delta = (estimate.get(key, {}) for key in ("before", "projected", "delta"))
        proposal = estimate.get("proposal", {})
        costs = proposal.get("estimates", {})
        observed = estimate.get("observed_execution", {})
        action, feasible = record.get("policy_action", {}), record.get("feasibility", {})
        yield dict(timestamp=record["timestamp"], symbol=action.get("symbol"), side=action.get("side"),
            proposed_value=action.get("value"), transition=record.get("transition"),
            accepted=feasible.get("accepted"), admitted_quantity=feasible.get("quantity"),
            risk_reasons=";".join(feasible.get("reasons", [])),
            confidence=estimate.get("confidence"), horizon=estimate.get("horizon"),
            before_var_usd=before.get("value_at_risk"), projected_var_usd=projected.get("value_at_risk"),
            delta_var_usd=delta.get("value_at_risk"),
            before_es_usd=before.get("expected_shortfall"), projected_es_usd=projected.get("expected_shortfall"),
            delta_es_usd=delta.get("expected_shortfall"),
            projected_volatility_usd=projected.get("volatility"),
            projected_gross_fraction=projected.get("gross_fraction"),
            projected_cash_fraction=projected.get("cash_fraction"),
            projected_max_position_fraction=projected.get("max_position_fraction"),
            spread_bps=proposal.get("spread_bps"), volume_participation=proposal.get("volume_participation"),
            estimated_commission_usd=costs.get("commission_cost"),
            estimated_half_spread_usd=costs.get("half_spread_cost"),
            estimated_configured_slippage_usd=costs.get("configured_slippage_cost"),
            observed_matched=observed.get("matched"), observed_fee_usd=observed.get("fee"),
            observed_slippage_usd=observed.get("slippage_cost"),
            risk_observations=projected.get("observations"), unavailable_reason=projected.get("reason"))


def write_evaluation_artifacts(*, bars, records, report, output, start, stop,
                               risk_config=None, config=None, dashboard=True):
    """Annotate frozen decisions and write independent evaluation diagnostics."""
    from .analytics import build_performance_report
    from .decision_risk import estimate_decision_risk

    config = config or EvaluationConfig()
    risk_config = risk_config or RiskConfig()
    if not bars or not 0 <= start < stop <= len(next(iter(bars.values()))):
        raise ValueError("Diagnostic interval must contain observed bars")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result, annotated = deepcopy(report), deepcopy(records)
    strategy_symbols = list(result.get("universe") or bars)
    risk_bars = {symbol: bars[symbol] for symbol in strategy_symbols}
    clock = [bar.timestamp for bar in next(iter(risk_bars.values()))]
    clock_index = {timestamp: index for index, timestamp in enumerate(clock)}
    for record in annotated:
        timestamp = _time(record["timestamp"])
        index = clock_index.get(timestamp)
        if index is None or not start <= index < stop:
            raise ValueError("Decision must reference an observed bar inside the diagnostic interval")
        if record.get("bar_index", index) != index:
            raise ValueError("Decision bar index disagrees with its observed timestamp")
        record["bar_index"] = index
        record["risk_estimate"] = estimate_decision_risk(bars=risk_bars, record=record,
            risk_config=risk_config, lookback=config.risk_lookback,
            confidence=config.risk_confidence, minimum_observations=config.risk_min_observations)
    curve = result.get("execution", {}).get("equity_curve", [])
    metrics = build_performance_report(bars=bars, records=annotated, equity_curve=curve,
        initial_funding=float(result["initial_funding"]), start=start, stop=stop,
        benchmark_symbol=config.benchmark_symbol, risk_free_rate=config.risk_free_rate,
        strategy_symbols=strategy_symbols)
    diagnostics = dict(evaluation_only=True, used_for_training=False, used_for_checkpoint_selection=False,
        config=asdict(config), performance=metrics, decisions=len(annotated),
        interpretation="Risk estimates and market indicators describe frozen decisions; they are not rewards, "
                       "cash charges, execution guarantees or causal market-impact estimates.",
        risk_config=asdict(risk_config), interval=dict(start_index=start, stop_index=stop,
            start_timestamp=clock[start], end_timestamp=clock[stop - 1]))
    result["diagnostics"] = diagnostics
    result["artifacts"] = dict(report_path=str(output / "report.json"),
        decisions_path=str(output / "decisions.jsonl"), diagnostics_path=str(output / "diagnostics.json"),
        decision_risk_csv=str(output / "decision_risk.csv"))
    if dashboard:
        from .visualization import write_dashboard
        paths = write_dashboard(output, bars=bars, records=annotated, report=result, start=start, stop=stop)
        result["artifacts"].update(paths)
    _write_json(output / "diagnostics.json", diagnostics)
    rows = list(_risk_rows(annotated))
    with (output / "decision_risk.csv").open("w", newline="") as handle:
        fieldnames = list(rows[0]) if rows else ["timestamp", "symbol", "side", "before_var_usd", "projected_var_usd"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    with (output / "decisions.jsonl").open("w") as handle:
        for record in annotated:
            handle.write(json.dumps(record, default=_json_default, allow_nan=False) + "\n")
    result.setdefault("provenance", {})["decisions_sha256"] = _signature(output / "decisions.jsonl")
    _write_json(output / "report.json", result)
    return result


def regenerate_report(bars_path, report_path, decisions_path, output, *, config=None, dashboard=True):
    """Rebuild a frozen report without fetching data, sampling policy or training."""
    report_path, decisions_path, output = Path(report_path), Path(decisions_path), Path(output)
    if output.resolve() in {report_path.resolve().parent, decisions_path.resolve().parent}:
        raise ValueError("Write regenerated reports to a new directory to preserve their input audit")
    config = config or EvaluationConfig()
    report = json.loads(report_path.read_text())
    records = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
    provenance = report.get("provenance", {})
    decisions_hash = _signature(decisions_path)
    if provenance.get("decisions_sha256") and decisions_hash != provenance["decisions_sha256"]:
        raise ValueError("Archived report and decision audit provenance disagree")
    if report.get("decisions", len(records)) != len(records):
        raise ValueError("Archived report and decision count disagree")
    decision_times = [_time(record["timestamp"]) for record in records]
    if any(left >= right for left, right in zip(decision_times, decision_times[1:])):
        raise ValueError("Archived evaluation decisions must be strictly chronological")
    execution = report.get("execution", {})
    if "records" in execution and _canonical([record["accounting"] for record in records if "accounting" in record]) != execution["records"]:
        raise ValueError("Archived decisions and observed execution accounting disagree")
    snapshots = {_time(point["timestamp"]): point for point in execution.get("equity_curve", [])}
    for record in records:
        snapshot = snapshots.get(_time(record["timestamp"]))
        if snapshot is not None:
            account = record.get("account_before", {})
            if not (math.isclose(account.get("equity", float("nan")), snapshot["equity"], abs_tol=1e-6)
                    and math.isclose(account.get("cash", float("nan")),
                                     snapshot["equity"] - snapshot["gross_exposure"], abs_tol=1e-6)):
                raise ValueError("Archived decision account and observed portfolio path disagree")
    report["regenerated_from"] = dict(report_sha256=_signature(report_path), decisions_sha256=decisions_hash,
        decision_binding="verified_bound_archive" if provenance.get("decisions_sha256") else "legacy_unbound_archive",
        interpretation="Legacy archives have accounting consistency checks but lack an original decision hash binding.")
    source = load_bars_csv(bars_path)
    universe = report.get("universe")
    if not universe:
        raise ValueError("Archived report must declare its stock universe")
    bars = {symbol: sorted((bar for bar in source if bar.symbol == symbol), key=lambda b: b.timestamp)
            for symbol in universe}
    clock = [bar.timestamp for bar in bars[universe[0]]]
    if not clock or any([bar.timestamp for bar in series] != clock for series in bars.values()):
        raise ValueError("Report stock bars must have identical observed timestamp sequences")
    if provenance.get("bars_sha256") and _signature(bars_path) != provenance["bars_sha256"]:
        raise ValueError("Archived report and bar data provenance disagree")
    existing_interval = report.get("diagnostics", {}).get("interval")
    if existing_interval:
        start, stop = existing_interval["start_index"], existing_interval["stop_index"]
        if (clock[start] != _time(existing_interval["start_timestamp"])
                or clock[stop - 1] != _time(existing_interval["end_timestamp"])):
            raise ValueError("Archived diagnostic interval disagrees with bar data")
    else:
        times = [_time(record["timestamp"]) for record in records]
        curve = report.get("execution", {}).get("equity_curve", [])
        times.extend(_time(point["timestamp"]) for point in curve)
        if not times:
            raise ValueError("Archived report needs observed decisions or an equity path")
        if min(times) not in clock or max(times) not in clock:
            # Execution timestamps can fall between bars; decision timestamps cannot.
            decision_times = [_time(record["timestamp"]) for record in records]
            if any(timestamp not in clock for timestamp in decision_times):
                raise ValueError("Archived decisions do not align with observed bars")
        start = next((i for i, timestamp in enumerate(clock) if timestamp >= min(times)), len(clock))
        stop = next((i for i, timestamp in enumerate(clock) if timestamp > max(times)), len(clock))
    bars = include_benchmark(bars_path, bars, config.benchmark_symbol)
    risk = RiskConfig(**provenance.get("risk_config", report.get("diagnostics", {}).get("risk_config", {})))
    return write_evaluation_artifacts(bars=bars, records=records, report=report, output=output,
        start=start, stop=stop, risk_config=risk, config=config, dashboard=dashboard)
