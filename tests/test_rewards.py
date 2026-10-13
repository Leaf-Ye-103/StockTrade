"""Synthetic tests for observed action advantages and mature credit only."""

from datetime import datetime, timedelta, timezone
import pytest

from stocktrade.execution import MatchResult
from stocktrade.policy_contracts import TradingAction
from stocktrade.rewards import PortfolioObservation, RewardConfig, evaluate_action, evaluate_portfolio_action


TIME = datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc)
EXECUTION_TIME = TIME + timedelta(milliseconds=100)


def action(side="buy", *, price=100.0, quantity=1):
    if side == "hold":
        return TradingAction(7, None, side, 0.0)
    return TradingAction(7, "NVDA", side, price * quantity, 0.1)


def matched(price=100.0, *, quantity=1, execution_time=EXECUTION_TIME):
    return MatchResult(
        "matched", "compatible", quantity, ("observed-17",), price, quantity, execution_time
    )


def marks(prices, *, anchor=EXECUTION_TIME):
    return [(anchor + timedelta(minutes=index + 1), price) for index, price in enumerate(prices)]


def score(side, prices, *, price=100.0, quantity=1, config=None, initial_funding=1000.0):
    return evaluate_action(
        action(side, price=price, quantity=quantity),
        matched(price, quantity=quantity),
        decision_timestamp=TIME,
        decision_price=price,
        future_marks=marks(prices),
        initial_funding=initial_funding,
        equity_before=initial_funding,
        config=config,
    )


def test_sale_at_loss_can_receive_reward_for_avoiding_far_greater_loss():
    # The historical cost of 100 is deliberately not a reward target. Selling
    # existing shares at 95 is valuable relative to holding shares marked at 10.
    feedback = score("sell", [10.0] * 8, price=95.0)
    assert feedback.status == "supported"
    assert feedback.reward > 0
    assert feedback.components["terminal_marked_advantage"] == 85.0
    assert feedback.source_trade_ids == ("observed-17",)
    assert feedback.maturity_timestamp == marks([10.0] * 8)[-1][0]
    assert "same existing shares" in feedback.baseline_description


def test_purchase_before_collapse_is_penalized_separately_from_sale():
    purchase = score("buy", [95.0, 90.0, 80.0, 60.0, 40.0, 20.0, 15.0, 10.0])
    assert purchase.reward < 0
    assert purchase.components["terminal_marked_advantage"] == -90.0
    assert purchase.components["downside_adjustment"] == -22.5


def test_shaped_feedback_components_are_not_a_round_trip_profit_report():
    purchase = score("buy", [95.0] * 8)
    sale = score("sell", [10.0] * 8, price=95.0)
    for feedback in (purchase, sale):
        parts = feedback.components
        assert parts["reward_dollars"] == pytest.approx(
            parts["weighted_marked_advantage"] + parts["downside_adjustment"] - parts["fees"]
        )
        assert "profit" not in parts
    # Sale avoids a later loss that was never realized by the example ledger.
    # Adding these action advantages therefore cannot be called its PnL of -5.
    assert purchase.components["reward_dollars"] + sale.components["reward_dollars"] != -5.0


def test_sale_that_misses_later_gains_is_penalized():
    feedback = score("sell", [100.0] * 8, price=95.0)
    assert feedback.reward < 0
    assert feedback.components["terminal_marked_advantage"] == -5.0
    assert feedback.components["downside_adjustment"] == 0.0


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_flat_prices_reward_only_deducts_fee_once(side):
    feedback = score(side, [100.0] * 8)
    assert feedback.components["weighted_marked_advantage"] == 0.0
    assert feedback.components["downside_adjustment"] == 0.0
    assert feedback.components["fees"] == pytest.approx(0.05)
    assert feedback.reward == pytest.approx(-0.05 / 1000)
    assert score(side, [100.0] * 8, config=RewardConfig(fee_bps=0)).reward == 0.0


def test_fixed_checkpoints_do_not_choose_best_hindsight_price():
    config = RewardConfig(
        horizon_steps=4,
        horizon_fractions=(0.5, 1.0),
        horizon_weights=(0.5, 0.5),
        downside_weight=0,
        fee_bps=0,
    )
    feedback = score("buy", [1000.0, 90.0, 1000.0, 90.0], config=config)
    assert feedback.reward == pytest.approx(-10.0 / 1000.0)
    assert feedback.components["terminal_marked_advantage"] == -10.0


def test_downside_path_penalizes_purchase_and_rewards_removed_exposure():
    prices = [100.0, 100.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    buying = score("buy", prices, config=RewardConfig(fee_bps=0))
    selling = score("sell", prices, config=RewardConfig(fee_bps=0))
    assert buying.components["weighted_marked_advantage"] == 0.0
    assert buying.reward == -selling.reward
    assert buying.reward < 0
    assert selling.reward > 0


def test_larger_notional_has_larger_reward_under_fixed_funding():
    small = score("buy", [110.0] * 8, quantity=1)
    large = score("buy", [110.0] * 8, quantity=10)
    assert large.reward == pytest.approx(small.reward * 10)


def test_equity_snapshot_does_not_change_fixed_funding_normalization():
    kwargs = dict(
        decision_timestamp=TIME,
        decision_price=100.0,
        future_marks=marks([110.0] * 8),
        initial_funding=1000.0,
    )
    low_equity = evaluate_action(action(), matched(), equity_before=200.0, **kwargs)
    high_equity = evaluate_action(action(), matched(), equity_before=2000.0, **kwargs)
    assert low_equity.reward == high_equity.reward


@pytest.mark.parametrize("execution", [None, MatchResult("unmatched", "no_similar_trade", 1)])
def test_unsupported_execution_does_not_invent_loss_profit_or_fill(execution):
    feedback = evaluate_action(
        action(), execution,
        decision_timestamp=TIME,
        decision_price=100.0,
        future_marks=marks([1000.0] * 8),
        initial_funding=1000.0,
        equity_before=1000.0,
    )
    assert feedback.status == "unmatched"
    assert feedback.reward is None
    assert feedback.source_trade_ids == ()
    assert feedback.components == {}


def test_dataset_end_cannot_shorten_reward_horizon():
    feedback = score("buy", [1000.0] * 7)
    assert feedback.status == "immature"
    assert feedback.reward is None
    assert feedback.maturity_timestamp is None
    assert feedback.source_trade_ids == ("observed-17",)


def test_prices_beyond_fixed_horizon_cannot_change_reward():
    short = score("buy", [110.0] * 8)
    longer = score("buy", [110.0] * 8 + [1.0, 1000000.0])
    assert longer.reward == short.reward
    assert longer.maturity_timestamp == short.maturity_timestamp


def test_exact_recorded_quantity_is_used_for_near_size_match():
    observed = MatchResult("matched", "near_size", 10, ("recorded-9",), 100.0, 9, EXECUTION_TIME,
                           quantity_difference=-1)
    feedback = evaluate_action(
        action(quantity=10), observed,
        decision_timestamp=TIME,
        decision_price=100.0,
        future_marks=marks([110.0] * 8),
        initial_funding=1000.0,
        equity_before=1000.0,
        config=RewardConfig(fee_bps=0),
    )
    assert feedback.reward == pytest.approx(90.0 / 1000.0)
    assert feedback.components["gross_execution_value"] == 900.0


@pytest.mark.parametrize("bad_marks", [
    [(EXECUTION_TIME, 100.0)],
    [(TIME, 100.0)],
    [(EXECUTION_TIME + timedelta(minutes=1), 100.0)] * 8,
    [(datetime(2025, 1, 1), 100.0)],
    marks([float("nan")] * 8),
    marks([0.0] * 8),
])
def test_marks_must_follow_execution_with_valid_prices_and_times(bad_marks):
    with pytest.raises(ValueError):
        evaluate_action(
            action(), matched(),
            decision_timestamp=TIME,
            decision_price=100.0,
            future_marks=bad_marks,
            initial_funding=1000.0,
            equity_before=1000.0,
        )


def test_execution_before_decision_is_rejected():
    with pytest.raises(ValueError, match="precede"):
        evaluate_action(
            action(), matched(execution_time=TIME - timedelta(seconds=1)),
            decision_timestamp=TIME,
            decision_price=100.0,
            future_marks=marks([100.0] * 8),
            initial_funding=1000.0,
            equity_before=1000.0,
        )


def test_hold_does_not_assume_an_alternative_execution():
    feedback = evaluate_action(
        action("hold"), None,
        decision_timestamp=TIME,
        decision_price=0.0,
        future_marks=marks([1000.0] * 8, anchor=TIME),
        initial_funding=1000.0,
        equity_before=1000.0,
    )
    assert feedback.reward == 0.0
    assert feedback.source_trade_ids == ()
    assert feedback.execution_timestamp is None
    assert "unobserved" in feedback.baseline_description


@pytest.mark.parametrize("kwargs", [
    {"horizon_steps": 0},
    {"horizon_steps": True},
    {"horizon_fractions": (0.5,), "horizon_weights": (1.0,)},
    {"horizon_weights": (0.1, 0.1, 0.1)},
    {"horizon_weights": (-0.5, 0.5, 1.0)},
    {"downside_weight": -1},
    {"fee_bps": float("nan")},
])
def test_reward_settings_reject_invalid_or_incomplete_horizons(kwargs):
    with pytest.raises(ValueError):
        RewardConfig(**kwargs)


def portfolio_marks(equities, prices, *, anchor=EXECUTION_TIME):
    return [
        PortfolioObservation(anchor + timedelta(minutes=index + 1), equity, {"NVDA": price})
        for index, (equity, price) in enumerate(zip(equities, prices, strict=True))
    ]


def portfolio_score(
    side,
    equities,
    prices,
    *,
    cash_before=100.0,
    holdings_before=None,
    equity_before=100.0,
    price=100.0,
    config=None,
    supporting_trade_ids=(),
):
    return evaluate_portfolio_action(
        action(side, price=price),
        matched(price) if side != "hold" else None,
        decision_timestamp=TIME,
        initial_funding=1000.0,
        cash_before=cash_before,
        holdings_before=holdings_before or {},
        equity_before=equity_before,
        future_observations=portfolio_marks(equities, prices),
        config=config,
        supporting_trade_ids=supporting_trade_ids,
    )


def test_active_feedback_penalizes_bad_entry_even_when_later_sale_avoids_collapse():
    # Actual observed purchase at 100 and later recorded sale at 95 leave cash
    # 95. Entry is compared with retaining its original 100 cash, never with an
    # invented perpetual holding marked at 10. Fees omitted in this fixture.
    entry = portfolio_score(
        "buy", [95.0] * 8, [10.0] * 8, supporting_trade_ids=("recorded-sale-at-95",)
    )
    assert entry.reward == pytest.approx(-6.25 / 1000.0)
    assert entry.components["actual_vs_hold_advantage"] == -5.0
    assert entry.components["downside_improvement"] == -1.25
    assert entry.source_trade_ids == ("observed-17", "recorded-sale-at-95")

    protective_sale = portfolio_score(
        "sell", [95.0] * 8, [10.0] * 8,
        cash_before=0.0, holdings_before={"NVDA": 1}, equity_before=95.0, price=95.0,
    )
    assert protective_sale.reward > 0
    assert protective_sale.components["actual_vs_hold_advantage"] == 85.0
    assert protective_sale.components["downside_improvement"] == 21.25
    # These overlapping action-continuation targets describe distinct causal
    # questions and must not be added as the investment's realized profit.
    assert entry.reward < 0 < protective_sale.reward


def test_actual_later_policy_loss_remains_in_entry_feedback():
    feedback = portfolio_score(
        "buy", [95.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0], [10.0] * 8,
        supporting_trade_ids=("later-supported-order",),
    )
    assert feedback.components["terminal_actual_vs_hold_advantage"] == -60.0
    assert feedback.reward < 0


def test_portfolio_fees_are_not_deducted_again():
    feedback = portfolio_score(
        "sell", [99.9] * 8, [100.0] * 8,
        cash_before=0.0, holdings_before={"NVDA": 1},
        config=RewardConfig(fee_bps=500, downside_weight=0),
    )
    assert feedback.reward == pytest.approx(-0.1 / 1000.0)
    assert feedback.components["fee_already_in_ledger"] == 1.0
    assert "fees" not in feedback.components


def test_retaining_portfolio_is_neutral_even_when_held_stock_declines():
    feedback = portfolio_score(
        "hold", [150.0] * 8, [50.0] * 8,
        cash_before=100.0, holdings_before={"NVDA": 1}, equity_before=200.0,
    )
    assert feedback.reward == 0.0
    assert feedback.components["baseline_worst_drawdown"] == 50.0
    assert feedback.components["actual_worst_drawdown"] == 50.0
    assert feedback.source_trade_ids == ()


def test_hold_credit_can_include_actual_supported_policy_continuation():
    feedback = portfolio_score(
        "hold", [90.0] * 8, [10.0] * 8,
        supporting_trade_ids=("later-buy", "later-sell"),
    )
    assert feedback.reward < 0
    assert feedback.execution_timestamp is None
    assert feedback.source_trade_ids == ("later-buy", "later-sell")


def test_portfolio_baseline_marks_all_pre_action_holdings():
    observations = [
        PortfolioObservation(EXECUTION_TIME + timedelta(minutes=index + 1), 150.0,
                             {"NVDA": 10.0, "AAPL": 20.0})
        for index in range(8)
    ]
    feedback = evaluate_portfolio_action(
        action("sell", price=95.0), matched(95.0),
        decision_timestamp=TIME,
        initial_funding=1000.0,
        cash_before=0.0,
        holdings_before={"NVDA": 1, "AAPL": 2},
        equity_before=200.0,
        future_observations=observations,
    )
    # The baseline is 1 * 10 + 2 * 20 = 50, not just the sold stock's 10.
    assert feedback.components["actual_vs_hold_advantage"] == 100.0
    assert feedback.components["baseline_worst_drawdown"] == 150.0


def test_portfolio_fixed_checkpoints_and_full_horizon():
    config = RewardConfig(
        horizon_steps=4, horizon_fractions=(0.5, 1.0), horizon_weights=(0.5, 0.5), downside_weight=0,
    )
    feedback = portfolio_score("buy", [1000.0, 90.0, 1000.0, 90.0], [100.0] * 4, config=config)
    assert feedback.reward == pytest.approx(-10.0 / 1000.0)
    immature = portfolio_score("buy", [1000.0] * 3, [100.0] * 3, config=config)
    assert immature.status == "immature"
    assert immature.reward is None


def test_portfolio_unknown_execution_is_not_rewarded_by_later_equity():
    feedback = evaluate_portfolio_action(
        action(), MatchResult("unmatched", "missing_observation", 1),
        decision_timestamp=TIME,
        initial_funding=1000.0,
        cash_before=100.0,
        holdings_before={},
        equity_before=100.0,
        future_observations=portfolio_marks([1000.0] * 8, [1000.0] * 8),
        supporting_trade_ids=("unrelated-later-order",),
    )
    assert feedback.status == "unmatched"
    assert feedback.reward is None
    assert feedback.source_trade_ids == ()


def test_portfolio_sources_deduplicate_without_losing_execution_evidence():
    feedback = portfolio_score(
        "buy", [110.0] * 8, [110.0] * 8,
        supporting_trade_ids=("later-first", "observed-17", "later-first", "later-second"),
    )
    assert feedback.source_trade_ids == ("observed-17", "later-first", "later-second")


def test_portfolio_path_downside_risk_affects_feedback_without_endpoint_optimization():
    feedback = portfolio_score(
        "buy", [100.0, 100.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0], [100.0] * 8,
    )
    assert feedback.components["actual_vs_hold_advantage"] == 0.0
    assert feedback.components["actual_worst_drawdown"] == 50.0
    assert feedback.reward == pytest.approx(-12.5 / 1000.0)


def test_portfolio_sale_requires_preexisting_shares():
    with pytest.raises(ValueError, match="existing holdings"):
        portfolio_score("sell", [100.0] * 8, [10.0] * 8)


def test_portfolio_baseline_requires_observed_marks_for_existing_holdings():
    with pytest.raises(ValueError, match="Missing future observed mark"):
        evaluate_portfolio_action(
            action("hold"), None,
            decision_timestamp=TIME,
            initial_funding=1000.0,
            cash_before=100.0,
            holdings_before={"AAPL": 1},
            equity_before=200.0,
            future_observations=portfolio_marks([200.0] * 8, [100.0] * 8),
        )


def test_portfolio_observations_cannot_precede_execution():
    with pytest.raises(ValueError, match="after execution"):
        evaluate_portfolio_action(
            action(), matched(),
            decision_timestamp=TIME,
            initial_funding=1000.0,
            cash_before=100.0,
            holdings_before={},
            equity_before=100.0,
            future_observations=[PortfolioObservation(TIME, 100.0, {"NVDA": 100.0})],
        )


@pytest.mark.parametrize("kwargs", [
    {"timestamp": TIME.replace(tzinfo=None)},
    {"equity": float("nan")},
    {"equity": -1.0},
    {"prices": {"NVDA": 0.0}},
])
def test_portfolio_observation_validation(kwargs):
    defaults = {"timestamp": TIME, "equity": 100.0, "prices": {"NVDA": 100.0}}
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        PortfolioObservation(**defaults)
