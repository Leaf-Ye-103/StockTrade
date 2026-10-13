"""Synthetic checks of direct-action, portfolio-wide observed-risk limits."""

from dataclasses import replace

import pytest

from stocktrade.action_risk import constrain_action
from stocktrade.policy_contracts import PortfolioState, TradingAction
from stocktrade.risk import MarketState, RiskConfig


def inputs():
    state = PortfolioState(100000, 100000, {}, {"NVDA": 100, "AMD": 50})
    quote = MarketState("NVDA", 100, 10000, bid=100, ask=100)
    action = TradingAction(1, "NVDA", "buy", 20000, 0.2)
    return action, state, {"NVDA": quote}


def test_direct_action_is_limited_without_predicting_returns():
    action, state, quotes = inputs()
    decision = constrain_action(action, state, quotes, ("NVDA", "AMD"))
    assert decision.action is action
    assert decision.accepted and decision.side == "BUY"
    assert decision.quantity == 100
    assert decision.notional == 10000
    assert "turnover_cap" in decision.reasons
    assert not any("forecast" in reason for reason in decision.reasons)


def test_dollar_notional_excludes_fees_but_cash_limits_include_them():
    action, state, quotes = inputs()
    decision = constrain_action(replace(action, value=100), state, quotes, ("NVDA",))
    assert decision.quantity == 1
    assert decision.notional == 100
    reserve_state = replace(state, cash=20500, holdings={"AMD": 1590})
    config = RiskConfig(max_gross_fraction=1, max_position_fraction=1)
    decision = constrain_action(action, reserve_state, quotes, ("NVDA",), config)
    assert decision.quantity == 4
    fees = decision.notional * (config.commission_bps + config.slippage_bps) / 10000
    cash_after = reserve_state.cash - decision.notional - fees
    assert cash_after >= config.minimum_cash_fraction * (reserve_state.equity - fees)


def test_all_stock_holdings_count_towards_gross_exposure_even_outside_universe():
    action, state, quotes = inputs()
    state = replace(state, cash=60000, holdings={"AMD": 800})
    config = RiskConfig(max_position_fraction=0.5, max_turnover_fraction=1, max_participation=1)
    decision = constrain_action(replace(action, value=50000), state, quotes, ("NVDA",), config)
    assert decision.side == "BUY" and decision.quantity == 199
    assert "gross_exposure_cap" in decision.reasons
    fees = decision.notional * (config.commission_bps + config.slippage_bps) / 10000
    exposure_after = state.holdings["AMD"] * state.prices["AMD"] + decision.notional
    assert exposure_after <= config.max_gross_fraction * (state.equity - fees)


def test_concentration_limit_does_not_turn_a_buy_into_a_sale():
    action, state, quotes = inputs()
    state = replace(state, cash=70000, holdings={"NVDA": 300})
    decision = constrain_action(action, state, quotes, ("NVDA",))
    assert not decision.accepted and decision.side == "HOLD"
    assert decision.quantity == 0 and decision.action.side == "buy"
    assert "concentration_cap" in decision.reasons


def test_loss_limits_suppress_new_buys_and_leave_sales_as_policy_choices():
    action, state, quotes = inputs()
    state = replace(state, cash=60000, holdings={"NVDA": 200},
                    peak_equity=100000, day_start_equity=85000)
    buy = constrain_action(action, state, quotes, ("NVDA",))
    assert not buy.accepted and buy.side == "HOLD" and buy.quantity == 0
    assert {"drawdown_limit", "daily_loss_limit"}.issubset(buy.reasons)
    sell_action = replace(action, side="sell", value=10000)
    sell = constrain_action(sell_action, state, quotes, ("NVDA",))
    assert sell.accepted and sell.side == "SELL" and sell.quantity == 80
    assert {"drawdown_limit", "daily_loss_limit"}.issubset(sell.reasons)
    dry = {"NVDA": replace(quotes["NVDA"], available_volume=0)}
    assert not constrain_action(sell_action, state, dry, ("NVDA",)).accepted


def test_sell_cannot_short_or_exceed_requested_dollar_amount():
    action, state, quotes = inputs()
    sale = replace(action, side="sell", value=350)
    empty = constrain_action(sale, state, quotes, ("NVDA",))
    assert not empty.accepted and "holding_cap" in empty.reasons
    state = replace(state, holdings={"NVDA": 2})
    decision = constrain_action(sale, state, quotes, ("NVDA",))
    assert decision.accepted and decision.quantity == 2 and decision.notional == 200
    assert decision.notional <= sale.value


def test_past_volume_whole_lots_and_sub_share_budget_constrain_actions():
    action, state, quotes = inputs()
    quote = replace(quotes["NVDA"], available_volume=1000)
    config = RiskConfig(lot_size=3)
    decision = constrain_action(action, state, {"NVDA": quote}, ("NVDA",), config)
    assert decision.quantity == 9 and "participation_cap" in decision.reasons
    tiny = constrain_action(replace(action, value=99), state, quotes, ("NVDA",))
    assert not tiny.accepted and "below_lot_size" in tiny.reasons


@pytest.mark.parametrize("mark,bid,ask", [(100, 100, 100.5), (100, 99, 99.5), (99.5, 99.5, 100)])
def test_spread_and_fees_preserve_reserves_at_current_and_executable_marks(mark, bid, ask):
    action = TradingAction(1, "NVDA", "buy", 90000, 0.9)
    state = PortfolioState(100000, 70000, {"NVDA": 100, "AMD": 400},
                           {"NVDA": mark, "AMD": 50})
    quote = MarketState("NVDA", mark, 1000000, bid=bid, ask=ask)
    config = RiskConfig(max_spread_bps=60, max_position_fraction=0.4,
                        max_participation=1, max_turnover_fraction=1)
    decision = constrain_action(action, state, {"NVDA": quote}, ("NVDA",), config)
    assert decision.accepted and decision.side == "BUY"
    fee = decision.notional * (config.commission_bps + config.slippage_bps) / 10000
    cash_after = state.cash - decision.notional - fee
    other = state.holdings["AMD"] * state.prices["AMD"]
    for resulting_mark in (mark, ask):
        position_after = (state.holdings["NVDA"] + decision.quantity) * resulting_mark
        equity_after = cash_after + position_after + other
        assert cash_after >= config.minimum_cash_fraction * equity_after
        assert position_after <= config.max_position_fraction * equity_after
        assert position_after + other <= config.max_gross_fraction * equity_after
    assert decision.notional == decision.quantity * ask
    assert decision.notional <= action.value


@pytest.mark.parametrize("field,value,reason", [
    ("price", float("nan"), "invalid_market_state"),
    ("available_volume", -1, "invalid_market_state"),
    ("available_volume", 0, "no_observed_liquidity"),
    ("data_quality", 0.49, "insufficient_data_quality"),
    ("data_quality", 1.1, "insufficient_data_quality"),
    ("spread_bps", 51, "spread_limit"),
    ("bid", None, "incomplete_executable_quotes"),
    ("ask", 99, "invalid_executable_quotes"),
    ("ask", float("inf"), "invalid_executable_quotes"),
    ("symbol", "AMD", "inconsistent_market_symbol"),
])
def test_invalid_market_information_suppresses_actions(field, value, reason):
    action, state, quotes = inputs()
    quote = replace(quotes["NVDA"], **{field: value})
    decision = constrain_action(action, state, {"NVDA": quote}, ("NVDA",))
    assert not decision.accepted and decision.side == "HOLD" and reason in decision.reasons


def test_allowed_universe_and_available_marks_are_required():
    action, state, quotes = inputs()
    assert "symbol_not_allowed" in constrain_action(action, state, quotes, ("AMD",)).reasons
    assert "missing_market_state" in constrain_action(action, state, {}, ("NVDA",)).reasons
    no_mark = replace(state, prices={"AMD": 50})
    assert "missing_market_mark" in constrain_action(action, no_mark, quotes, ("NVDA",)).reasons
    mismarked = replace(state, prices={"NVDA": 101, "AMD": 50})
    assert "inconsistent_market_mark" in constrain_action(action, mismarked, quotes, ("NVDA",)).reasons


def test_explicit_hold_has_no_quote_or_execution_requirement():
    _, state, _ = inputs()
    action = TradingAction(0, None, "hold", 0)
    decision = constrain_action(action, state, {}, ("NVDA",))
    assert decision.accepted and decision.side == "HOLD"
    assert decision.quantity == 0 and decision.notional == 0


@pytest.mark.parametrize("field,value", [
    ("value", float("nan")), ("value", -100), ("value", float("inf")),
    ("fraction", float("nan")), ("fraction", -1), ("action_id", -1),
])
def test_corrupted_action_values_are_rejected(field, value):
    action, state, quotes = inputs()
    object.__setattr__(action, field, value)
    assert "invalid_action" in constrain_action(action, state, quotes, ("NVDA",)).reasons


@pytest.mark.parametrize("field,value,reason", [
    ("cash", float("nan"), "invalid_portfolio"),
    ("cash", -1, "invalid_portfolio"),
    ("initial_funding", float("inf"), "invalid_portfolio"),
    ("holdings", {"NVDA": -1}, "invalid_portfolio_holdings"),
    ("holdings", {"NVDA": 1.5}, "invalid_portfolio_holdings"),
    ("holdings", {"NVDA": True}, "invalid_portfolio_holdings"),
    ("holdings", {"UNKNOWN": 1}, "missing_holding_price"),
    ("prices", {"NVDA": float("nan")}, "invalid_portfolio_prices"),
    ("peak_equity", float("nan"), "invalid_loss_baseline"),
    ("day_start_equity", 0, "invalid_loss_baseline"),
])
def test_corrupted_portfolio_fields_are_rejected(field, value, reason):
    action, state, quotes = inputs()
    object.__setattr__(state, field, value)
    assert reason in constrain_action(action, state, quotes, ("NVDA",)).reasons


def test_nonfinite_aggregate_equity_and_extreme_fees_are_rejected():
    action, state, quotes = inputs()
    state.holdings["NVDA"] = 10**400
    assert "nonfinite_portfolio_equity" in constrain_action(action, state, quotes, ("NVDA",)).reasons
    _, state, _ = inputs()
    config = RiskConfig(commission_bps=1e308, slippage_bps=1e308)
    assert "nonfinite_sizing" in constrain_action(action, state, quotes, ("NVDA",), config).reasons


def test_sales_with_costs_exceeding_proceeds_cannot_create_negative_cash():
    action, _, quotes = inputs()
    state = PortfolioState(1250, 250, {"NVDA": 10}, {"NVDA": 100})
    config = RiskConfig(commission_bps=20000, slippage_bps=0, max_turnover_fraction=1)
    decision = constrain_action(replace(action, side="sell", value=1000), state, quotes, ("NVDA",), config)
    assert decision.accepted and decision.quantity == 2
    assert "sale_cost_cash_cap" in decision.reasons
    cash_after = state.cash + decision.notional - decision.notional * 2
    assert cash_after >= 0
