"""Synthetic tests for descriptive risk, independent of action training."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from stocktrade.decision_risk import estimate_decision_risk
from stocktrade.execution import MatchResult, OrderIntent, ReplayRecord
from stocktrade.risk import RiskConfig
from stocktrade.schemas import MarketBar


def series(symbol, returns, *, final=100, quotes=True):
    prices = [final / float(np.prod(1 + np.asarray(returns, dtype=float)))]
    for value in returns:
        prices.append(prices[-1] * (1 + value))
    start = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)
    return [MarketBar(start + timedelta(minutes=index), symbol, price, price, price, price, 10000,
                      bid=price - 1 if quotes else None, ask=price + 1 if quotes else None)
            for index, price in enumerate(prices)]


def fixture():
    bars = {"A": series("A", [-.1, .02, .1, -.05])}
    record = dict(timestamp=bars["A"][-1].timestamp, bar_index=4,
                  policy_action=dict(action_id=0, symbol=None, side="hold", value=0),
                  feasibility=dict(accepted=True, quantity=0, notional=0, reasons=[]),
                  account_before=dict(equity=2000, cash=1000, holdings={"A": 10}))
    return bars, record


def estimate(bars, record, **kwargs):
    return estimate_decision_risk(bars=bars, record=record, risk_config=kwargs.pop("risk_config", RiskConfig()),
                                  lookback=4, minimum_observations=4, **kwargs)


def trade(record, *, side, quantity, notional, accepted=True, symbol="A"):
    result = deepcopy(record)
    result["policy_action"] = dict(action_id=1, symbol=symbol, side=side, value=notional)
    result["feasibility"] = dict(accepted=accepted, quantity=quantity, notional=notional, reasons=[])
    return result


def test_known_one_stock_empirical_var_es_volatility_and_stress():
    bars, record = fixture()
    result = estimate(bars, record, confidence=.5)
    before = result["before"]
    assert before["value_at_risk"] == pytest.approx(50)
    assert before["expected_shortfall"] == pytest.approx(75)
    assert before["volatility"] == pytest.approx(np.std([-100, 20, 100, -50]))
    assert before["observations"] == 4 and before["reason"] is None
    assert before["gross_exposure"] == 1000
    assert before["gross_fraction"] == .5 and before["cash_fraction"] == .5
    assert before["max_position_fraction"] == pytest.approx(.5)
    assert before["market_stress"] == {"-1%": -10, "-5%": -50, "-10%": -100}
    assert result["horizon"] == "one_observed_bar" and result["unit"] == "USD"


def test_hold_preserves_full_portfolio_risk_instead_of_reporting_zero_action_risk():
    bars, record = fixture()
    result = estimate(bars, record)
    for name in ("value_at_risk", "expected_shortfall", "volatility", "gross_exposure"):
        assert result["projected"][name] == result["before"][name]
        assert result["delta"][name] == 0
    assert result["before"]["value_at_risk"] == pytest.approx(100)
    assert result["projected"]["projected_not_an_execution"]


def test_sale_reduces_exposure_and_dollar_risk_without_ranking_the_action():
    bars, record = fixture()
    result = estimate(bars, trade(record, side="sell", quantity=5, notional=500))
    assert result["projected"]["value_at_risk"] == pytest.approx(50)
    assert result["delta"]["value_at_risk"] == pytest.approx(-50)
    assert result["projected"]["gross_exposure"] == 500
    assert result["projected"]["cash"] == 1500
    assert result["proposal"]["projected_notional"] == 500
    assert not any(name in result for name in ("reward", "penalty", "ranking", "loss"))


def test_rejected_proposal_keeps_before_portfolio_even_with_nonzero_requested_size():
    bars, record = fixture()
    result = estimate(bars, trade(record, side="buy", quantity=10, notional=1000, accepted=False))
    assert result["projected"]["value_at_risk"] == result["before"]["value_at_risk"]
    assert result["proposal"]["applied_quantity"] == 0
    assert result["proposal"]["projected_notional"] == 0
    assert result["proposal"]["estimates"]["commission_cost"] == 0


def test_current_notional_scenarios_preserve_cross_stock_correlation():
    returns = [-.1, .02, .1, -.05]
    bars, record = fixture()
    bars["B"] = series("B", [-value for value in returns])
    record["account_before"]["holdings"] = {"A": 5, "B": 5}
    result = estimate(bars, record)
    assert result["before"]["value_at_risk"] == pytest.approx(0, abs=1e-10)
    assert result["before"]["volatility"] == pytest.approx(0, abs=1e-10)
    assert result["before"]["max_position_fraction"] == pytest.approx(.25)


def test_return_pairs_intersect_without_inventing_missing_stock_observations():
    bars, record = fixture()
    bars["B"] = series("B", [.1, -.02, -.1, .05])
    del bars["B"][2]
    record["account_before"]["holdings"] = {"A": 5, "B": 5}
    result = estimate(bars, record)
    assert result["before"]["observations"] == 2
    assert result["before"]["value_at_risk"] is None
    assert result["before"]["reason"] == "insufficient_common_historical_returns"


def test_future_bars_after_index_and_after_timestamp_cannot_change_diagnostics():
    bars, record = fixture()
    original = estimate(bars, record)
    bars["A"].append(replace(bars["A"][-1], timestamp=record["timestamp"] + timedelta(minutes=1),
                             close=float("nan"), volume=float("inf")))
    assert estimate(bars, record) == original
    record["bar_index"] = 5
    assert estimate(bars, record) == original
    bars["A"][-1] = replace(bars["A"][-1], close=1e200, bid=1e150, ask=1e250)
    assert estimate(bars, record) == original


def test_advanced_live_feature_clock_uses_last_completed_bar_without_new_market_data():
    bars, record = fixture()
    original = estimate(bars, record)
    record["timestamp"] += timedelta(seconds=45)
    result = estimate(bars, record)
    assert result["before"] == original["before"]
    assert result["projected"] == original["projected"]
    assert result["timestamp"] == record["timestamp"].isoformat()


def test_lookback_limits_scenarios_to_recent_returns():
    bars, record = fixture()
    bars["A"] = series("A", [-.99] + [.01] * 4)
    record["bar_index"] = 5
    record["timestamp"] = bars["A"][-1].timestamp
    result = estimate(bars, record)
    assert result["before"]["observations"] == 4
    assert result["before"]["value_at_risk"] == 0


@pytest.mark.parametrize("failure", ["short", "missing", "invalid"])
def test_insufficient_or_missing_history_is_null_with_reason(failure):
    bars, record = fixture()
    if failure == "short":
        bars["A"] = bars["A"][-3:]
    elif failure == "missing":
        bars = {}
    else:
        bars["A"][2] = replace(bars["A"][2], close=float("nan"))
    result = estimate(bars, record)
    assert result["before"]["value_at_risk"] is None
    assert result["before"]["expected_shortfall"] is None
    assert result["before"]["volatility"] is None
    assert result["before"]["reason"]
    assert result["delta"]["value_at_risk"] is None
    json.dumps(result, allow_nan=False)


def test_cash_only_risk_is_legitimately_zero_without_historical_data():
    _, record = fixture()
    record["account_before"] = dict(equity=2000, cash=2000, holdings={})
    result = estimate({}, record)
    assert result["before"]["value_at_risk"] == 0
    assert result["before"]["expected_shortfall"] == 0
    assert result["before"]["volatility"] == 0
    assert result["before"]["reason"] == "cash_only_portfolio"
    assert result["before"]["cash_fraction"] == 1


def test_new_stock_without_observed_prices_has_unknown_projected_risk_not_cash_only_zero():
    _, record = fixture()
    record["account_before"] = dict(equity=2000, cash=2000, holdings={})
    result = estimate({}, trade(record, side="buy", quantity=1, notional=100))
    assert result["before"]["value_at_risk"] == 0
    assert result["projected"]["value_at_risk"] is None
    assert "missing_price_history" in result["projected"]["reason"]
    assert result["delta"]["value_at_risk"] is None


def test_conditional_fee_spread_and_slippage_estimates_are_not_charged_again():
    bars, record = fixture()
    record = trade(record, side="buy", quantity=1, notional=100)
    risk = RiskConfig(commission_bps=100, slippage_bps=200)
    without_execution = estimate(bars, record, risk_config=risk)
    record["accounting"] = dict(status="matched", fee=.7,
                                match=dict(estimated_price=100.5, estimated_quantity=1))
    record["intent"] = dict(side="buy", reference_price=100)
    result = estimate(bars, record, risk_config=risk)
    assert result["proposal"]["estimates"] == dict(commission_cost=1, half_spread_cost=1,
                                                    configured_slippage_cost=2)
    assert result["proposal"]["volume_participation"] == .0001
    assert result["proposal"]["spread_bps"] == pytest.approx(200)
    assert result["proposal"]["estimates_are_charged"] is False
    assert result["projected"]["cash"] == 900
    assert result["projected"]["equity"] == pytest.approx(2000)
    assert result["before"] == without_execution["before"]
    assert result["projected"] == without_execution["projected"]
    assert result["observed_execution"]["fee"] == .7
    assert result["observed_execution"]["slippage_cost"] == .5
    assert result["observed_execution"]["reason"] is None


def test_projected_quote_notional_and_current_close_marks_are_explicitly_conditional():
    bars, record = fixture()
    result = estimate(bars, trade(record, side="buy", quantity=1, notional=101))
    assert result["projected"]["cash"] == 899
    assert result["projected"]["equity"] == pytest.approx(1999)
    assert result["projected"]["projected_not_an_execution"]
    assert result["projected"]["cash_excludes_cost_estimates"]


def test_execution_costs_require_a_matched_ledger_record():
    bars, record = fixture()
    record["accounting"] = dict(status="unmatched", fee=999,
                                match=dict(estimated_price=90, estimated_quantity=10))
    record["intent"] = dict(side="sell", reference_price=100)
    result = estimate(bars, record)
    assert not result["observed_execution"]["matched"]
    assert result["observed_execution"]["fee"] is None
    assert result["observed_execution"]["slippage_cost"] is None


def test_observed_costs_accept_the_ledger_dataclass_without_a_duplicate_top_level_intent():
    bars, record = fixture()
    intent = OrderIntent(record["timestamp"], "A", "sell", 1, 99, 100)
    match = MatchResult("matched", "synthetic_fixture", 1, ("fixture-1",), 99.5, 1,
                        record["timestamp"] + timedelta(seconds=1))
    record["accounting"] = ReplayRecord(intent, match, "matched", "synthetic_fixture", 1099.4, 9, .1)
    result = estimate(bars, record)
    assert result["observed_execution"]["fee"] == .1
    assert result["observed_execution"]["slippage_cost"] == .5


def test_missing_quotes_never_fabricate_a_spread_or_half_spread_cost():
    bars, record = fixture()
    bars["A"][-1] = replace(bars["A"][-1], bid=None, ask=None, volume=0)
    result = estimate(bars, trade(record, side="buy", quantity=1, notional=100))
    proposal = result["proposal"]
    assert proposal["spread_bps"] is None and proposal["estimates"]["half_spread_cost"] is None
    assert proposal["quote_reason"] == "missing_executable_quotes"
    assert proposal["volume_participation"] is None
    assert proposal["liquidity_reason"] == "no_positive_observed_volume"
    json.dumps(result, allow_nan=False)


def test_overflowing_conditional_cost_estimates_are_null_and_explained():
    bars, record = fixture()
    record["account_before"] = dict(equity=1e308, cash=1e308, holdings={})
    result = estimate(bars, trade(record, side="buy", quantity=1, notional=1e308),
                      risk_config=RiskConfig(commission_bps=1e308, slippage_bps=1e308))
    assert result["proposal"]["estimates"]["commission_cost"] is None
    assert result["proposal"]["estimate_reasons"]["commission_cost"] == "nonfinite_cost_estimate"
    json.dumps(result, allow_nan=False)


def test_mutated_risk_configuration_is_revalidated():
    bars, record = fixture()
    config = RiskConfig()
    object.__setattr__(config, "commission_bps", float("nan"))
    with pytest.raises(ValueError):
        estimate(bars, record, risk_config=config)


@pytest.mark.parametrize("kwargs", [
    {"lookback": 0}, {"lookback": True}, {"minimum_observations": 0},
    {"minimum_observations": 61}, {"confidence": 1}, {"confidence": 0},
    {"confidence": float("nan")},
])
def test_invalid_configuration_is_rejected(kwargs):
    bars, record = fixture()
    with pytest.raises(ValueError):
        estimate_decision_risk(bars=bars, record=record, risk_config=RiskConfig(), **kwargs)


@pytest.mark.parametrize("section,field,value", [
    ("account_before", "cash", float("nan")),
    ("account_before", "equity", float("inf")),
    ("account_before", "holdings", {"A": -1}),
    ("policy_action", "value", float("inf")),
    ("feasibility", "notional", float("nan")),
    ("feasibility", "quantity", 1.5),
])
def test_invalid_finite_or_long_only_record_values_are_rejected(section, field, value):
    bars, record = fixture()
    record[section][field] = value
    with pytest.raises(ValueError):
        estimate(bars, record)


def test_proposed_shorts_and_purchase_beyond_cash_are_not_valid_projections():
    bars, record = fixture()
    with pytest.raises(ValueError, match="sales cannot exceed"):
        estimate(bars, trade(record, side="sell", quantity=11, notional=1100))
    with pytest.raises(ValueError, match="exceeds observed available cash"):
        estimate(bars, trade(record, side="buy", quantity=11, notional=1100))


def test_function_does_not_mutate_input_records_or_bars():
    bars, record = fixture()
    original_bars, original_record = deepcopy(bars), deepcopy(record)
    estimate(bars, record)
    assert bars == original_bars and record == original_record
