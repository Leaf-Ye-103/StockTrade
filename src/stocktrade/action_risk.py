"""Portfolio-wide safeguards for directly proposed trading actions.

This layer uses observed account and quote information only. It neither
forecasts returns nor changes a rejected buy into an unchosen sale. A sizing
decision is an intention; recorded execution evidence is still required.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

from .policy_contracts import PortfolioState, TradingAction
from .risk import MarketState, RiskConfig


@dataclass(frozen=True)
class ActionDecision:
    action: TradingAction
    side: str
    quantity: int
    notional: float
    reasons: tuple[str, ...]
    accepted: bool


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def constrain_action(
    action: TradingAction,
    state: PortfolioState,
    quotes: dict[str, MarketState],
    allowed_symbols: tuple[str, ...],
    config: RiskConfig | None = None,
) -> ActionDecision:
    """Constrain a long-only action without a prediction-derived target.

    ``value`` bounds dollar notional before fees, for buys and sales. Cash
    reserve limits include fees. All held stocks contribute to equity
    and gross exposure, including holdings outside the current trading universe.
    Whole-share, observed-liquidity and quote limits apply before historical
    matching. ``available_volume`` must be past observed volume, not future data.
    """
    config = config or RiskConfig()

    def reject(reason: str | tuple[str, ...]) -> ActionDecision:
        reasons = (reason,) if isinstance(reason, str) else reason
        return ActionDecision(action, "HOLD", 0, 0.0, reasons, False)

    if (not isinstance(action.action_id, int) or isinstance(action.action_id, bool)
            or action.action_id < 0 or action.side not in ("buy", "sell", "hold")
            or not _finite(action.value) or action.value < 0
            or not _finite(action.fraction) or action.fraction < 0):
        return reject("invalid_action")
    if (not isinstance(state.holdings, Mapping) or not isinstance(state.prices, Mapping)
            or not _finite(state.initial_funding) or state.initial_funding <= 0
            or not _finite(state.cash) or state.cash < 0):
        return reject("invalid_portfolio")
    if any(not isinstance(symbol, str) or not symbol or not _finite(price) or price <= 0
           for symbol, price in state.prices.items()):
        return reject("invalid_portfolio_prices")
    for symbol, quantity in state.holdings.items():
        if (not isinstance(symbol, str) or not symbol or not isinstance(quantity, int)
                or isinstance(quantity, bool) or quantity < 0):
            return reject("invalid_portfolio_holdings")
        if quantity > 0 and symbol not in state.prices:
            return reject("missing_holding_price")
    try:
        gross_exposure = math.fsum(
            quantity * state.prices[symbol]
            for symbol, quantity in state.holdings.items() if quantity > 0
        )
        equity = state.cash + gross_exposure
    except OverflowError:
        return reject("nonfinite_portfolio_equity")
    if not math.isfinite(equity) or equity <= 0:
        return reject("nonfinite_portfolio_equity")

    loss_reasons: list[str] = []
    for name, baseline, limit in (
        ("drawdown_limit", state.peak_equity, config.max_drawdown),
        ("daily_loss_limit", state.day_start_equity, config.max_daily_loss),
    ):
        if baseline is not None:
            if not _finite(baseline) or baseline <= 0:
                return reject("invalid_loss_baseline")
            if 1 - equity / baseline >= limit:
                loss_reasons.append(name)

    if action.side == "hold":
        if action.value != 0 or action.fraction != 0:
            return reject("invalid_hold_action")
        if action.symbol is not None and action.symbol not in allowed_symbols:
            return reject("symbol_not_allowed")
        return ActionDecision(action, "HOLD", 0, 0.0, tuple(loss_reasons), True)
    if action.symbol not in allowed_symbols:
        return reject("symbol_not_allowed")
    if action.value == 0:
        return reject("zero_action_budget")
    if action.symbol not in quotes:
        return reject("missing_market_state")
    market = quotes[action.symbol]
    if market.symbol != action.symbol:
        return reject("inconsistent_market_symbol")
    if (not all(_finite(value) for value in (
            market.price, market.available_volume, market.spread_bps, market.data_quality,
    )) or market.price <= 0 or market.available_volume < 0 or market.spread_bps < 0):
        return reject("invalid_market_state")
    if action.symbol not in state.prices:
        return reject("missing_market_mark")
    if not math.isclose(state.prices[action.symbol], market.price, rel_tol=1e-10, abs_tol=1e-10):
        return reject("inconsistent_market_mark")
    if market.available_volume == 0:
        return reject("no_observed_liquidity")
    if not 0 <= market.data_quality <= 1 or market.data_quality < config.minimum_data_quality:
        return reject("insufficient_data_quality")
    if (market.bid is None) != (market.ask is None):
        return reject("incomplete_executable_quotes")
    if market.bid is not None:
        if (not _finite(market.bid) or not _finite(market.ask)
                or market.bid <= 0 or market.ask < market.bid):
            return reject("invalid_executable_quotes")
        bid, ask = market.bid, market.ask
        effective_spread_bps = max(market.spread_bps, (ask - bid) / market.price * 10000)
    else:
        # The mark need not be a midpoint: conservatively budget the full spread.
        ask = market.price * (1 + market.spread_bps / 10000)
        bid = market.price * (1 - market.spread_bps / 10000)
        effective_spread_bps = market.spread_bps
    if (not math.isfinite(ask) or not math.isfinite(bid) or bid <= 0
            or not math.isfinite(effective_spread_bps)
            or effective_spread_bps > config.max_spread_bps + 1e-10):
        return reject("spread_limit")
    if action.side == "buy" and loss_reasons:
        return reject(tuple(loss_reasons))

    holding = state.holdings.get(action.symbol, 0)
    other_exposure = gross_exposure - holding * market.price
    cost_fraction = (config.commission_bps + config.slippage_bps) / 10000
    if not math.isfinite(cost_fraction):
        return reject("nonfinite_sizing")
    execution_price = ask if action.side == "buy" else bid
    requested = action.value / execution_price
    reasons = list(loss_reasons)
    limits = [
        ("participation_cap", market.available_volume * config.max_participation),
        ("turnover_cap", equity * config.max_turnover_fraction / execution_price),
    ]
    if action.side == "buy":
        # Check both the decision-time mark and immediate executable-price mark.
        # Exposure uses a lower bound on post-trade equity, cash an upper bound.
        exposure_base = min(equity, equity + holding * (ask - market.price))
        exposure_unit_loss = max(0.0, ask - market.price) + ask * cost_fraction
        exposure_mark = max(market.price, ask)
        cash_base = max(equity, equity + holding * (ask - market.price))
        cash_unit_loss = min(ask * cost_fraction, ask - market.price + ask * cost_fraction)
        cash_denominator = ask * (1 + cost_fraction) - config.minimum_cash_fraction * cash_unit_loss
        cap_values = (
            exposure_base, exposure_unit_loss, exposure_mark, cash_base, cash_unit_loss, cash_denominator,
        )
        if not all(math.isfinite(value) for value in cap_values) or cash_denominator <= 0:
            return reject("nonfinite_sizing")
        limits.extend([
            ("concentration_cap", max(0.0,
                (config.max_position_fraction * exposure_base - holding * exposure_mark)
                / (exposure_mark + config.max_position_fraction * exposure_unit_loss))),
            ("gross_exposure_cap", max(0.0,
                (config.max_gross_fraction * exposure_base - other_exposure - holding * exposure_mark)
                / (exposure_mark + config.max_gross_fraction * exposure_unit_loss))),
            ("cash_reserve_cap", max(0.0,
                (state.cash - config.minimum_cash_fraction * cash_base) / cash_denominator)),
        ])
    else:
        limits.append(("holding_cap", holding))
        if cost_fraction > 1:
            limits.append(("sale_cost_cash_cap", state.cash / (bid * (cost_fraction - 1))))
    if not math.isfinite(requested) or any(not math.isfinite(limit) for _, limit in limits):
        return reject("nonfinite_sizing")
    reasons.extend(name for name, limit in limits if limit < requested)
    desired = max(0.0, min(requested, *(limit for _, limit in limits)))
    quantity = math.floor(desired / config.lot_size) * config.lot_size
    if quantity < config.lot_size:
        return reject(tuple(reasons + ["below_lot_size"]))
    notional = quantity * execution_price
    if not math.isfinite(notional):
        return reject("nonfinite_sizing")
    return ActionDecision(action, action.side.upper(), quantity, notional, tuple(reasons), True)
