from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from stocktrade.execution import (
    MatchResult,
    MatchingConfig,
    OrderIntent,
    PortfolioLedger,
    TradePrint,
    load_trades_csv,
    match_order,
    replay_orders,
)


BASE = datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)


def intent(**kwargs):
    values = dict(
        timestamp=BASE, symbol="NVDA", side="buy", quantity=100,
        limit_price=100.1, reference_price=100.0, spread=0.1,
        volatility=0.02, latency_ms=100,
    )
    values.update(kwargs)
    return OrderIntent(**values)


def trade(trade_id="actual-1", **kwargs):
    values = dict(
        timestamp=BASE + timedelta(seconds=1), trade_id=trade_id,
        symbol="NVDA", side="buy", quantity=100, price=100.0,
        bid=99.9, ask=100.0, spread=0.1, volatility=0.02,
    )
    values.update(kwargs)
    return TradePrint(**values)


def test_no_print_means_no_fill():
    result = match_order(intent(), [])
    assert result.status == "unmatched"
    assert result.estimated_quantity == 0
    assert result.estimated_price is None
    assert result.source_trade_ids == ()


@pytest.mark.parametrize("seconds", [-1, 0, 0.099, 60.101])
def test_timing_latency_and_finite_horizon(seconds):
    result = match_order(intent(), [trade(timestamp=BASE + timedelta(seconds=seconds))])
    assert result.status == "unmatched"
    assert "outside_arrival_window" in result.reason


def test_arrival_boundary_is_supported():
    source = trade(timestamp=BASE + timedelta(milliseconds=100))
    assert match_order(intent(), [source]).status == "matched"


@pytest.mark.parametrize("changes,reason", [
    ({"side": "sell", "price": 99.9}, "incompatible_aggressor_direction"),
    ({"quantity": 1}, "incompatible_size"),
    ({"quantity": 101}, "incompatible_size"),
    ({"price": 100.2, "ask": 100.2}, "limit_price_incompatible"),
    ({"spread": 0.3}, "incompatible_spread_regime"),
    ({"volatility": 0.2}, "incompatible_volatility_regime"),
    ({"bid": None}, "missing_or_invalid_quote_context"),
    ({"ask": 100.05}, "trade_not_marketable_at_observed_quote"),
])
def test_mismatch_abstains(changes, reason):
    result = match_order(intent(), [trade(**changes)])
    assert result.status == "unmatched"
    assert reason in result.reason


def test_near_size_uses_actual_recorded_quantity_without_rounding_up():
    result = match_order(intent(), [trade(quantity=95)])
    assert result.status == "matched"
    assert result.requested_quantity == 100
    assert result.estimated_quantity == 95
    assert result.quantity_difference == -5


def test_exact_size_mode_is_available():
    config = MatchingConfig(size_tolerance=0)
    assert match_order(intent(), [trade(quantity=99)], config).status == "unmatched"


@pytest.mark.parametrize("name", ["spread", "volatility"])
def test_missing_predecision_context_abstains(name):
    result = match_order(replace(intent(), **{name: None}), [trade()])
    assert result.reason == f"missing_intent_{name}_context"


@pytest.mark.parametrize("name", ["spread", "volatility"])
def test_missing_event_regime_abstains(name):
    result = match_order(intent(), [replace(trade(), **{name: None})])
    assert result.status == "unmatched"
    assert f"missing_trade_{name}_context" in result.reason


def test_earliest_candidate_selected_without_future_impact_or_price_ranking():
    earlier = trade("earlier", price=100.1, impact_bps=-1000)
    later = trade("later", timestamp=BASE + timedelta(seconds=2), price=100.0, impact_bps=1000)
    result = match_order(intent(), [later, earlier])
    assert result.source_trade_ids == ("earlier",)
    assert result.observed_impact_bps == -1000
    changed = match_order(intent(), [replace(later, impact_bps=-9999), replace(earlier, impact_bps=9999)])
    assert changed.source_trade_ids == result.source_trade_ids


def test_duplicate_ids_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="unique"):
        match_order(intent(), [trade(), trade()])


def test_used_source_never_matches_again():
    result = match_order(intent(), [trade()], used_trade_ids={"actual-1"})
    assert result.status == "unmatched"
    assert result.reason == "source_already_used"


def passive_fixture():
    order = intent(mode="passive", limit_price=99.9, reference_price=99.9, queue_ahead=50)
    ahead = trade("ahead", timestamp=BASE + timedelta(milliseconds=200), side="sell", price=99.9, quantity=50)
    candidate = trade(
        "passive", side="sell", price=99.9, queue_ahead=50,
        queue_depletion=50, queue_observed_from=BASE,
        queue_evidence_ids=("ahead",), queue_priority_verified=True,
    )
    return order, ahead, candidate


def test_passive_without_order_level_priority_abstains():
    order, ahead, candidate = passive_fixture()
    result = match_order(order, [ahead, replace(candidate, queue_priority_verified=False)])
    assert result.status == "unmatched"
    assert "unsupported_passive_queue_priority" in result.reason


def test_passive_has_opposite_aggressor_and_auditable_progression():
    order, ahead, candidate = passive_fixture()
    result = match_order(order, [ahead, candidate])
    assert result.status == "matched"
    assert result.source_trade_ids == ("ahead", "passive")
    assert result.estimated_price == 99.9


@pytest.mark.parametrize("changes,reason", [
    ({"queue_evidence_ids": ()}, "unsubstantiated_queue_depletion"),
    ({"queue_evidence_ids": ("invented",)}, "missing_or_reused_queue_evidence"),
    ({"queue_depletion": 49}, "queue_not_depleted"),
    ({"queue_observed_from": BASE + timedelta(milliseconds=101)}, "queue_observation_started_after_arrival"),
    ({"queue_evidence_ids": ("ahead", "ahead")}, "duplicate_queue_evidence"),
])
def test_passive_invalid_progression_abstains(changes, reason):
    order, ahead, candidate = passive_fixture()
    result = match_order(order, [ahead, replace(candidate, **changes)])
    assert result.status == "unmatched"
    assert reason in result.reason


def test_passive_cannot_reuse_queue_evidence():
    order, ahead, candidate = passive_fixture()
    result = match_order(order, [ahead, candidate], used_trade_ids={"ahead"})
    assert result.status == "unmatched"
    assert "missing_or_reused_queue_evidence" in result.reason


def test_ledger_actual_quantity_cash_costs_drawdown_and_source_reuse():
    order = intent()
    analog = match_order(order, [trade(quantity=95)])
    ledger = PortfolioLedger(20000, fee_bps=10, per_trade_fee=1)
    accepted = ledger.apply_match(order, analog)
    assert accepted.status == "matched"
    assert ledger.positions == {"NVDA": 95}
    assert ledger.cash == pytest.approx(20000 - 9500 - 10.5)
    duplicate = ledger.apply_match(order, analog)
    assert duplicate.status == "rejected"
    assert duplicate.reason == "source_already_used"
    ledger.mark(BASE + timedelta(seconds=2), "NVDA", 90)
    result = ledger.result()
    assert result.total_fees == pytest.approx(10.5)
    assert result.coverage == 0.5
    assert result.rejected_orders == 1
    assert result.final_equity == pytest.approx(20000 - 10.5 - 950)
    assert result.max_drawdown == pytest.approx((10.5 + 950) / 20000)
    assert result.source_trade_ids == ("actual-1",)


def test_ledger_rejects_unfunded_buy_without_consuming_source():
    order = intent()
    ledger = PortfolioLedger(100)
    result = ledger.apply_match(order, match_order(order, [trade()]))
    assert result.status == "rejected"
    assert result.reason == "insufficient_cash"
    assert ledger.cash == 100
    assert ledger.used_trade_ids == set()


def test_no_shorting_and_round_trip_accounting():
    ledger = PortfolioLedger(20000)
    sell_order = intent(side="sell", limit_price=99.8)
    sell = trade("sell", side="sell", price=99.9)
    record = ledger.apply_match(sell_order, match_order(sell_order, [sell]))
    assert record.reason == "insufficient_inventory"
    ledger.apply_match(intent(), match_order(intent(), [trade()]))
    later_sell_order = intent(timestamp=BASE + timedelta(seconds=2), side="sell", limit_price=101, reference_price=101)
    later_sell = trade("sell-later", timestamp=BASE + timedelta(seconds=3), side="sell", price=101, bid=101, ask=101.1)
    assert ledger.apply_match(later_sell_order, match_order(later_sell_order, [later_sell])).status == "matched"
    assert ledger.cash == 20100
    assert ledger.positions == {"NVDA": 0}


def test_chronological_ledger_and_replay_prevent_future_cash_leak():
    orders = [intent(), intent(timestamp=BASE + timedelta(milliseconds=500))]
    result = replay_orders(orders, [trade()], initial_cash=20000)
    assert result.matched_orders == 1
    assert result.unmatched_orders == 1
    assert result.records[-1].reason == "decision_precedes_previous_observation"
    ledger = PortfolioLedger(20000)
    ledger.mark(BASE + timedelta(seconds=2), "NVDA", 100)
    with pytest.raises(ValueError, match="observation time"):
        ledger.apply_match(intent(), match_order(intent(), [trade()]))


def test_loader_preserves_source_identifiers_and_rejects_naive_timestamps(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text("timestamp,trade_id,symbol,side,quantity,price,bid,ask,spread,volatility\n"
                    "2026-10-08T14:00:01Z,venue-unique-1,NVDA,buy,100,100,99.9,100,.1,.02\n")
    source = load_trades_csv(path)
    assert source[0].trade_id == "venue-unique-1"
    assert match_order(intent(), source).status == "matched"
    path.write_text("timestamp,trade_id,symbol,side,quantity,price\n"
                    "2026-10-08T14:00:01,venue-unique-1,NVDA,buy,100,100\n")
    with pytest.raises(ValueError, match="offset"):
        load_trades_csv(path)


def test_matched_result_cannot_invent_fractional_or_oversized_quantities():
    with pytest.raises(ValueError, match="integer"):
        MatchResult("matched", "fake", 100, ("real",), 100, 1.5, BASE)
    with pytest.raises(ValueError, match="exceed"):
        MatchResult("matched", "fake", 100, ("real",), 100, 101, BASE, quantity_difference=1)
    with pytest.raises(ValueError, match="invented"):
        MatchResult("unmatched", "no event", 100, estimated_price=100)
