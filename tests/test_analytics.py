from datetime import datetime, timedelta, timezone
import json
import math
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from stocktrade.analytics import build_performance_report
from stocktrade.schemas import MarketBar


NY = ZoneInfo("America/New_York")


def make_bars(timestamps, prices=None, symbol="TEST"):
    prices = prices if prices is not None else [100.0] * len(timestamps)
    return [MarketBar(timestamp, symbol, price, price + 1, max(0.01, price - 1), price,
                      1000, bid=price - 0.01, ask=price + 0.01)
            for timestamp, price in zip(timestamps, prices)]


def snapshots(timestamps, equity, gross=None):
    gross = gross if gross is not None else [0.0] * len(timestamps)
    return [dict(timestamp=timestamp, equity=value, gross_exposure=exposure)
            for timestamp, value, exposure in zip(timestamps, equity, gross)]


def report(timestamps, equity, *, prices=None, records=None, funding=100.0, gross=None, **kwargs):
    return build_performance_report(
        bars={"TEST": make_bars(timestamps, prices)}, records=records or [],
        equity_curve=snapshots(timestamps, equity, gross),
        initial_funding=funding, start=0, stop=len(timestamps), **kwargs,
    )


def weekdays(count, start=datetime(2024, 1, 2, 16, tzinfo=NY)):
    result = []
    current = start
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.astimezone(timezone.utc))
        current += timedelta(days=1)
    return result


def execution(timestamp, side, quantity, price, fee, status="matched"):
    return dict(
        timestamp=timestamp,
        policy_action=dict(side=side),
        intent=dict(timestamp=timestamp, symbol="TEST", side=side, quantity=quantity,
                    reference_price=price),
        analog=dict(status=status, event_timestamp=timestamp, estimated_quantity=quantity,
                    estimated_price=price),
        accounting=dict(status=status, fee=fee),
    )


def test_hand_computed_drawdown_recovery_duration_and_duplicate_completion():
    timestamps = [datetime(2024, 1, 2, 21, tzinfo=timezone.utc) + timedelta(days=i) for i in range(4)]
    curve = snapshots(timestamps, [100, 80, 90, 110])
    # This partial same-time mark must be discarded in favor of the completed
    # snapshot, including its artificially severe drawdown.
    curve.insert(1, dict(timestamp=timestamps[1], equity=1, gross_exposure=0))
    result = build_performance_report(
        bars={"TEST": make_bars(timestamps)}, records=[], equity_curve=curve,
        initial_funding=100, start=0, stop=4,
    )
    performance = result["performance"]
    assert performance["completed_snapshots"] == 4
    assert performance["net_return"] == pytest.approx(0.10)
    assert performance["drawdown"]["max_drawdown"] == pytest.approx(0.20)
    assert performance["drawdown"]["max_drawdown_duration_seconds"] == 3 * 86400
    assert performance["drawdown"]["recovery_seconds"] == 2 * 86400
    assert performance["drawdown"]["time_underwater_seconds"] == 2 * 86400
    assert performance["drawdown"]["time_underwater_fraction"] == pytest.approx(2 / 3)
    assert result["risk"]["dollar_volatility"] == pytest.approx(np.std([-20, 10, 20], ddof=1))
    assert result["risk"]["cagr"] is None


def test_matched_fifo_statistics_include_fees_and_never_close_open_lots():
    timestamps = [datetime(2024, 1, 2, 15, tzinfo=timezone.utc) + timedelta(minutes=i) for i in range(6)]
    records = [
        execution(timestamps[0], "buy", 10, 10, 1),
        execution(timestamps[1], "sell", 5, 12, 0.5),
        execution(timestamps[2], "sell", 5, 8, 0.5),
        execution(timestamps[3], "buy", 4, 20, 0.4),
        execution(timestamps[4], "buy", 100, 20, 0, status="unmatched"),
        dict(timestamp=timestamps[4], policy_action=dict(side="buy"),
             feasibility=dict(accepted=False, reasons=["cash_limit"]), transition="risk_rejected_no_order"),
        dict(timestamp=timestamps[5], policy_action=dict(side="hold")),
    ]
    result = report(timestamps, [1000, 999, 998, 997.6, 997.6, 997.6],
                    prices=[10, 12, 8, 20, 20, 20], records=records, funding=1000,
                    gross=[0, 60, 0, 80, 80, 80])
    trades, measured = result["trades"], result["execution"]
    assert trades["closed_lot_matches"] == 2
    assert trades["realized_net_pnl"] == pytest.approx(-2)
    assert trades["win_rate"] == 0.5
    assert trades["profit_factor"] == pytest.approx(9 / 11)
    assert trades["payoff_ratio"] == pytest.approx(9 / 11)
    assert trades["open_quantities"] == {"TEST": 4}
    assert measured["gross_notional"] == 280
    assert measured["turnover"] == 0.28
    assert measured["total_fees"] == 2.4
    assert measured["matching_coverage"] == 0.8
    assert measured["risk_rejected_decisions"] == 1
    assert measured["incomplete_execution"]
    assert result["portfolio"]["final_holdings"] == {"TEST": 4}
    assert result["portfolio"]["final_holdings_reconcile_to_snapshot"]
    json.dumps(result, allow_nan=False)


def test_cash_and_no_trades_keep_undefined_ratios_null():
    timestamps = weekdays(25)
    result = report(timestamps, [100.0] * 25)
    assert result["performance"]["net_return"] == 0
    assert result["performance"]["drawdown"]["max_drawdown"] == 0
    assert result["risk"]["historical_var_95"] == 0
    assert result["risk"]["historical_expected_shortfall_95"] == 0
    assert result["risk"]["sharpe_ratio"] is None
    assert result["risk"]["sortino_ratio"] is None
    assert result["trades"]["win_rate"] is None
    assert result["trades"]["profit_factor"] is None
    assert result["execution"]["matching_coverage"] is None
    assert result["portfolio"]["final_cash_fraction"] == 1
    assert result["portfolio"]["invested_hhi"] is None
    assert "risk.sharpe_ratio" in result["availability"]
    assert "execution.matching_coverage" in result["availability"]
    assert result["execution"]["total_fees"] == 0


@pytest.mark.parametrize("seconds", [1, 60])
def test_intraday_sample_count_does_not_become_annualized_days(seconds):
    start = datetime(2024, 1, 2, 10, tzinfo=NY)
    timestamps = [start + timedelta(seconds=index * seconds) for index in range(60)]
    equity = 100 * np.cumprod(1 + np.array([0.001 if i % 2 else -0.002 for i in range(60)]))
    result = report(timestamps, equity)
    assert result["risk"]["return_samples"] == 59
    assert result["risk"]["daily_return_samples"] == 0
    assert result["risk"]["historical_var_95"] == pytest.approx(0.002)
    assert result["risk"]["historical_expected_shortfall_95"] == pytest.approx(0.002)
    assert result["risk"]["confidence_level"] == 0.95
    assert result["risk"]["tail_samples"] > 0
    for key in ("annualized_volatility", "sharpe_ratio", "sortino_ratio", "cagr", "calmar_ratio"):
        assert result["risk"][key] is None
        assert f"risk.{key}" in result["availability"]


def test_ny_dates_not_utc_dates_and_first_partial_day_not_a_daily_return():
    # Both snapshots have the same UTC date but belong to different NY days.
    timestamps = [datetime(2024, 1, 3, 0, tzinfo=timezone.utc),
                  datetime(2024, 1, 3, 22, tzinfo=timezone.utc)]
    result = report(timestamps, [110, 121])
    assert len(result["performance"]["daily_returns"]) == 1
    assert result["performance"]["daily_returns"][0] == dict(
        start_date="2024-01-02", end_date="2024-01-03", value=pytest.approx(0.1))
    timestamps = [datetime(2024, 1, 2, 12, tzinfo=NY),
                  datetime(2024, 1, 3, 16, tzinfo=NY),
                  datetime(2024, 1, 4, 16, tzinfo=NY)]
    result = report(timestamps, [100, 150, 165])
    assert len(result["performance"]["daily_returns"]) == 1
    assert result["performance"]["daily_returns"][0]["value"] == pytest.approx(0.1)
    assert result["performance"]["incomplete_NY_dates"] == ["2024-01-02"]


def test_whole_missing_weekday_breaks_daily_returns_but_weekend_does_not():
    timestamps = [datetime(2024, 1, 5, 16, tzinfo=NY),  # Friday
                  datetime(2024, 1, 8, 16, tzinfo=NY),  # Monday
                  datetime(2024, 1, 10, 16, tzinfo=NY)]  # Missing Tuesday
    result = report(timestamps, [100, 110, 121])
    assert len(result["performance"]["daily_returns"]) == 1
    assert result["performance"]["daily_returns"][0]["value"] == pytest.approx(0.1)
    assert result["performance"]["unverified_weekday_gaps"] == [dict(previous_date="2024-01-08", next_date="2024-01-10")]
    assert "performance.daily_returns" in result["availability"]


def test_aligned_daily_benchmark_beta_correlation_and_tracking_error():
    timestamps = weekdays(30)
    benchmark_returns = np.array([0.01 if index % 2 else -0.005 for index in range(29)])
    prices = np.concatenate(([100.0], 100 * np.cumprod(1 + benchmark_returns)))
    equity = np.concatenate(([100.0], 100 * np.cumprod(1 + 2 * benchmark_returns)))
    bars = {"TEST": make_bars(timestamps), "REF": make_bars(timestamps, prices, "REF")}
    result = build_performance_report(
        bars=bars, records=[], equity_curve=snapshots(timestamps, equity), initial_funding=100,
        start=0, stop=30, benchmark_symbol="REF", strategy_symbols=("TEST",),
    )
    reference = result["benchmark"]
    assert reference["daily_aligned_samples"] == 29
    assert reference["beta"] == pytest.approx(2)
    assert reference["correlation"] == pytest.approx(1)
    assert reference["annualized_alpha"] == pytest.approx(0, abs=1e-12)
    assert reference["annualized_tracking_error"] == pytest.approx(np.std(benchmark_returns, ddof=1) * math.sqrt(252))
    assert not reference["executable"] and not reference["dividends_included"]
    assert result["risk"]["sharpe_ratio"] is not None
    assert result["risk"]["cagr"] is None
    # Remove one whole reference weekday: neither the missing date nor a
    # multi-day reference change may be aligned to a one-day strategy return.
    bars["REF"].pop(10)
    shorter = build_performance_report(
        bars=bars, records=[], equity_curve=snapshots(timestamps, equity), initial_funding=100,
        start=0, stop=30, benchmark_symbol="REF", strategy_symbols=("TEST",),
    )
    assert shorter["benchmark"]["daily_aligned_samples"] == 27


def test_missing_benchmark_and_missing_initial_prices_do_not_fabricate_reference():
    timestamps = weekdays(3)
    result = report(timestamps, [100, 101, 102], benchmark_symbol="SPY")
    assert result["benchmark"]["price_return"] is None
    assert result["benchmark"]["series"] == []
    assert "benchmark" in result["availability"]


def test_technical_indicators_have_warmup_and_hand_computed_values():
    start = datetime(2024, 1, 2, 10, tzinfo=NY)
    timestamps = [start + timedelta(seconds=index) for index in range(220)]
    prices = np.arange(1, 221, dtype=float)
    result = report(timestamps, [100] * 220, prices=prices)
    latest = result["indicators"]["TEST"]["latest"]
    assert latest["sma_20"] == 210.5
    assert latest["sma_50"] == 195.5
    assert latest["sma_200"] == 120.5
    assert latest["rsi_14"] == 100
    assert latest["bollinger_upper"] == pytest.approx(210.5 + 2 * np.std(prices[-20:], ddof=0))
    assert latest["atr_14"] == pytest.approx(2.0, abs=1e-6)
    assert latest["obv"] == 219000
    assert latest["macd"] == pytest.approx(7.0, abs=1e-6)
    assert latest["macd_signal"] == pytest.approx(7.0, abs=1e-6)
    assert result["indicators"]["TEST"]["series"][0]["rsi_14"] is None
    assert result["indicators"]["TEST"]["series"][12]["atr_14"] is None
    short = report(timestamps[:3], [100, 100, 100], prices=prices[:3])
    assert short["indicators"]["TEST"]["latest"]["sma_20"] is None
    assert "indicators.TEST.sma_20" in short["availability"]


def test_no_curve_does_not_invent_flat_performance_and_serialization_is_finite():
    timestamps = weekdays(3)
    result = build_performance_report(
        bars={"TEST": make_bars(timestamps)}, records=[], equity_curve=[],
        initial_funding=100, start=0, stop=3,
    )
    assert result["performance"]["net_return"] is None
    assert result["performance"]["drawdown"]["max_drawdown"] is None
    assert result["performance"]["series"] == []
    assert "performance" in result["availability"]
    json.dumps(result, allow_nan=False)


def test_long_horizon_cagr_uses_elapsed_calendar_time_and_no_invented_exit():
    timestamps = [datetime(2023, 1, 2, 16, tzinfo=NY), datetime(2024, 1, 2, 16, tzinfo=NY)]
    result = report(timestamps, [100, 121])
    assert result["risk"]["cagr"] == pytest.approx(1.21 ** (365.25 / 365) - 1)
    assert result["risk"]["calmar_ratio"] is None
    assert result["risk"]["sharpe_ratio"] is None
    assert not result["methodology"]["training_feedback"]
    assert not result["methodology"]["checkpoint_selection"]


def test_missing_fee_or_opening_lot_does_not_invent_closed_profit():
    timestamps = weekdays(3)
    records = [execution(timestamps[0], "sell", 2, 10, 0),
               execution(timestamps[1], "buy", 2, 10, None),
               execution(timestamps[2], "sell", 2, 11, 0)]
    result = report(timestamps, [100, 100, 102], records=records)
    assert result["trades"]["unsupported_opening_inventory_sales"] == 2
    assert result["trades"]["realized_net_pnl"] is None
    assert result["trades"]["win_rate"] is None
    assert result["execution"]["total_fees"] is None
    assert "execution.total_fees" in result["availability"]
    json.dumps(result, allow_nan=False)
