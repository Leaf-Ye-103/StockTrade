from dataclasses import replace

import pytest

from stocktrade.risk import AccountState, Forecast, MarketState, RiskConfig, size_position


def inputs():
    return (
        Forecast(0.02, 0.01, -0.02, 0.9, 0.20),
        MarketState("TEST", 100, 10000, spread_bps=2),
        AccountState(100000, 100000, peak_equity=100000, day_start_equity=100000),
    )


def test_position_sizing_respects_liquidity_turnover_and_cash():
    forecast, market, account = inputs()
    order = size_position(forecast, market, account)
    assert order.side == "BUY"
    assert order.quantity == 99  # Turnover uses an executable full-spread price.
    assert order.target_fraction == pytest.approx(0.20)
    config = RiskConfig(lot_size=7, max_turnover_fraction=0.08)
    order = size_position(forecast, market, account, config)
    assert order.quantity == 77
    assert order.quantity % 7 == 0
    low_cash = replace(account, cash=20500)
    order = size_position(forecast, market, low_cash)
    assert order.quantity == 4  # Retain the 20% equity reserve and budget fees.
    assert order.notional <= low_cash.cash - low_cash.equity * 0.2


@pytest.mark.parametrize("field,value,reason", [
    ("confidence", 0.1, "insufficient_confidence"),
    ("expected_return", 0.00001, "edge_below_transaction_cost"),
    ("volatility", float("nan"), "invalid_forecast"),
])
def test_forecast_gates_block_new_orders(field, value, reason):
    forecast, market, account = inputs()
    order = size_position(replace(forecast, **{field: value}), market, account)
    assert order.side == "HOLD" and order.quantity == 0
    assert reason in order.reasons


@pytest.mark.parametrize("field,value,reason", [
    ("available_volume", 0, "no_observed_liquidity"),
    ("data_quality", 0.1, "insufficient_data_quality"),
    ("spread_bps", 100, "spread_limit"),
    ("price", float("nan"), "nonfinite_market_or_account"),
])
def test_market_gates(field, value, reason):
    forecast, market, account = inputs()
    order = size_position(forecast, replace(market, **{field: value}), account)
    assert order.side == "HOLD" and reason in order.reasons


def test_drawdown_and_daily_loss_reduce_risk_without_inventing_liquidity():
    forecast, market, _ = inputs()
    account = AccountState(80000, 60000, position_quantity=200, peak_equity=100000, day_start_equity=85000)
    order = size_position(forecast, market, account)
    assert order.side == "SELL" and order.quantity == 80
    assert {"drawdown_limit", "daily_loss_limit"}.issubset(order.reasons)
    assert order.target_fraction == 0
    dry = size_position(forecast, replace(market, available_volume=0), account)
    assert dry.side == "HOLD" and dry.quantity == 0


def test_tail_volatility_and_other_positions_limit_target_exposure():
    forecast, market, account = inputs()
    volatile = replace(forecast, volatility=0.20, downside_quantile=-0.80)
    order = size_position(volatile, market, account)
    assert order.target_fraction == pytest.approx(0.025)
    other_positions = replace(account, other_exposure=59000)
    order = size_position(forecast, market, other_positions)
    assert order.target_fraction == pytest.approx(0.01)
    assert order.notional <= 1000


def test_bad_configuration_and_negative_position_are_rejected():
    with pytest.raises(ValueError):
        RiskConfig(max_participation=1.1)
    with pytest.raises(ValueError):
        RiskConfig(lot_size=0)
    forecast, market, account = inputs()
    order = size_position(forecast, market, replace(account, position_quantity=-1))
    assert order.side == "HOLD"


def test_executable_ask_and_fees_preserve_cash_and_exposure_reserves():
    forecast = Forecast(0.02, 0.01, -0.02, 0.9, 1.0)
    market = MarketState("TEST", 100, 1000000, spread_bps=50, bid=100, ask=100.5)
    account = AccountState(1000000, 1000000)
    config = RiskConfig(
        max_position_fraction=1.0, max_gross_fraction=1.0,
        minimum_cash_fraction=0.8, max_participation=1.0,
        max_turnover_fraction=1.0, volatility_budget=1.0, tail_loss_budget=1.0,
    )
    order = size_position(forecast, market, account, config)
    assert order.side == "BUY"
    fees = order.notional * (config.commission_bps + config.slippage_bps) / 10000
    cash_after = account.cash - order.notional - fees
    # Check both execution marking (ledger convention) and continued close
    # marking. Full quote spread plus fees were budgeted before the trade.
    for mark in (market.price, market.ask):
        exposure = order.quantity * mark
        equity_after = cash_after + exposure
        assert cash_after >= config.minimum_cash_fraction * equity_after
        assert exposure <= order.target_fraction * equity_after
    assert order.notional == order.quantity * market.ask


def test_existing_position_and_other_exposure_use_executable_ask_for_buy_caps():
    forecast, market, account = inputs()
    market = replace(market, bid=99.5, ask=100, price=99.5, spread_bps=50)
    account = replace(account, cash=50000, position_quantity=100, other_exposure=40000)
    config = RiskConfig(max_turnover_fraction=1, max_participation=1, max_spread_bps=60)
    order = size_position(forecast, market, account, config)
    assert order.side == "BUY"
    fee = order.notional * (config.commission_bps + config.slippage_bps) / 10000
    for mark in (market.price, market.ask):
        equity_after = account.equity + account.position_quantity * (mark - market.price) - order.quantity * (market.ask - mark) - fee
        position = (account.position_quantity + order.quantity) * mark
        assert position <= config.max_position_fraction * equity_after
        assert position + account.other_exposure <= config.max_gross_fraction * equity_after


def test_invalid_or_crossed_executable_quotes_abstain():
    forecast, market, account = inputs()
    for bid, ask in ((100, 99), (None, 100), (100, float("nan"))):
        order = size_position(forecast, replace(market, bid=bid, ask=ask), account)
        assert order.side == "HOLD" and order.quantity == 0
