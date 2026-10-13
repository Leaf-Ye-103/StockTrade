"""Conservative, deterministic limits around a learned research allocator.

These limits cover selected observable risks. They cannot eliminate market,
execution, model, or operational risk and do not authorize a live trade.
Return and volatility estimates refer to one model prediction interval.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Forecast:
    expected_return: float
    volatility: float
    downside_quantile: float
    confidence: float
    allocation_fraction: float


@dataclass(frozen=True)
class RiskConfig:
    max_position_fraction: float = 0.20
    max_gross_fraction: float = 0.60
    minimum_cash_fraction: float = 0.20
    max_participation: float = 0.01
    max_turnover_fraction: float = 0.10
    max_drawdown: float = 0.15
    max_daily_loss: float = 0.03
    minimum_confidence: float = 0.50
    minimum_data_quality: float = 0.50
    max_spread_bps: float = 50.0
    volatility_budget: float = 0.01
    tail_loss_budget: float = 0.02
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    lot_size: int = 1

    def __post_init__(self) -> None:
        fractions = (
            self.max_position_fraction, self.max_gross_fraction,
            self.minimum_cash_fraction, self.max_participation,
            self.max_turnover_fraction, self.max_drawdown, self.max_daily_loss,
            self.minimum_confidence, self.minimum_data_quality,
            self.volatility_budget, self.tail_loss_budget,
        )
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in fractions):
            raise ValueError("Risk fractions must be finite and between zero and one")
        if self.lot_size < 1 or not isinstance(self.lot_size, int):
            raise ValueError("lot_size must be a positive integer")
        if any(not math.isfinite(v) or v < 0 for v in (
            self.max_spread_bps, self.commission_bps, self.slippage_bps,
        )):
            raise ValueError("Costs and spread limits must be finite and nonnegative")


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    position_quantity: float = 0.0
    peak_equity: float | None = None
    day_start_equity: float | None = None
    other_exposure: float = 0.0


@dataclass(frozen=True)
class MarketState:
    symbol: str
    price: float
    available_volume: float
    spread_bps: float = 0.0
    data_quality: float = 1.0
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class SizingDecision:
    side: str
    quantity: int
    notional: float
    target_fraction: float
    reasons: tuple[str, ...]


def size_position(
    forecast: Forecast,
    market: MarketState,
    account: AccountState,
    config: RiskConfig | None = None,
) -> SizingDecision:
    """Propose a long-only order subject to cash, liquidity, and risk budgets.

    ``available_volume`` must be observed liquidity for the contemplated
    interval, not future volume. Loss limits request risk reduction; actual
    execution is still limited by liquidity and must pass observed-trade
    matching. Invalid or stale observations suppress all orders.
    """
    config = config or RiskConfig()

    def hold(reason: str, fraction: float = 0.0) -> SizingDecision:
        return SizingDecision("HOLD", 0, 0.0, fraction, (reason,))

    state_values = (
        market.price, market.available_volume, market.spread_bps,
        market.data_quality, account.equity, account.cash,
        account.position_quantity, account.other_exposure,
    )
    if any(not math.isfinite(v) for v in state_values):
        return hold("nonfinite_market_or_account")
    if (market.price <= 0 or account.equity <= 0 or account.cash < 0
            or account.position_quantity < 0 or account.other_exposure < 0):
        return hold("invalid_market_or_account")
    current_fraction = account.position_quantity * market.price / account.equity
    if market.available_volume <= 0:
        return hold("no_observed_liquidity", current_fraction)
    if not 0 <= market.data_quality <= 1 or market.data_quality < config.minimum_data_quality:
        return hold("insufficient_data_quality", current_fraction)
    if (market.bid is None) != (market.ask is None):
        return hold("incomplete_executable_quotes", current_fraction)
    if market.bid is not None:
        if (not math.isfinite(market.bid) or not math.isfinite(market.ask)
                or market.bid <= 0 or market.ask < market.bid):
            return hold("invalid_executable_quotes", current_fraction)
        bid, ask = market.bid, market.ask
        effective_spread_bps = max(market.spread_bps, (ask - bid) / market.price * 10000)
    else:
        # A mark can sit at either edge of the spread. Budget the full spread,
        # rather than assuming that it is the midpoint of an executable quote.
        ask = market.price * (1 + market.spread_bps / 10000)
        bid = market.price * (1 - market.spread_bps / 10000)
        effective_spread_bps = market.spread_bps
    if market.spread_bps < 0 or bid <= 0 or effective_spread_bps > config.max_spread_bps + 1e-10:
        return hold("spread_limit", current_fraction)
    forecast_values = (
        forecast.expected_return, forecast.volatility, forecast.downside_quantile,
        forecast.confidence, forecast.allocation_fraction,
    )
    if any(not math.isfinite(v) for v in forecast_values) or forecast.volatility < 0:
        return hold("invalid_forecast", current_fraction)
    if not 0 <= forecast.confidence <= 1:
        return hold("invalid_forecast_confidence", current_fraction)

    reasons: list[str] = []
    killed = False
    for name, baseline, limit in (
        ("drawdown_limit", account.peak_equity, config.max_drawdown),
        ("daily_loss_limit", account.day_start_equity, config.max_daily_loss),
    ):
        if baseline is not None:
            if not math.isfinite(baseline) or baseline <= 0:
                return hold("invalid_loss_baseline", current_fraction)
            if 1 - account.equity / baseline >= limit:
                reasons.append(name)
                killed = True

    if killed:
        target = 0.0
    elif forecast.confidence < config.minimum_confidence:
        return hold("insufficient_confidence", current_fraction)
    else:
        target = min(max(forecast.allocation_fraction, 0.0), config.max_position_fraction)
        exposure_budget = max(0.0, config.max_gross_fraction - account.other_exposure / account.equity)
        target = min(target, exposure_budget, 1 - config.minimum_cash_fraction)
        if forecast.volatility > 0:
            target = min(target, config.volatility_budget / forecast.volatility)
        if forecast.downside_quantile < 0:
            target = min(target, config.tail_loss_budget / -forecast.downside_quantile)
        costs = (config.commission_bps + config.slippage_bps + effective_spread_bps) / 10000
        if forecast.expected_return <= 0:
            target = 0.0
            reasons.append("nonpositive_expected_return")
        elif target > current_fraction and forecast.expected_return <= costs:
            return hold("edge_below_transaction_cost", current_fraction)

    cost_fraction = (config.commission_bps + config.slippage_bps) / 10000
    holding = account.position_quantity
    # Budget against both a continued decision-time mark and an immediate
    # execution-price mark. Known spread/fee losses must not consume cash or
    # exposure reserves before any future market move occurs.
    buy_base_equity = min(account.equity, account.equity + holding * (ask - market.price))
    buy_unit_loss = max(0.0, ask - market.price) + ask * cost_fraction
    buy_mark = max(market.price, ask)
    buy_delta = (target * buy_base_equity - holding * buy_mark) / (buy_mark + target * buy_unit_loss)
    if buy_delta > 0:
        delta = buy_delta
        execution_price = ask
    else:
        sell_base_equity = min(account.equity, account.equity + holding * (bid - market.price))
        sell_unit_loss = max(0.0, market.price - bid) + bid * cost_fraction
        sell_mark = max(market.price, bid)
        denominator = sell_mark - target * sell_unit_loss
        sell_delta = (holding * sell_mark - target * sell_base_equity) / denominator if denominator > 0 else holding
        delta = -max(0.0, sell_delta)
        execution_price = bid
    if abs(delta) < config.lot_size:
        return SizingDecision("HOLD", 0, 0.0, target, tuple(reasons + ["below_lot_size"]))
    quantity_limit = min(
        market.available_volume * config.max_participation,
        account.equity * config.max_turnover_fraction / execution_price,
    )
    if delta > 0:
        # A reserve fraction must use an upper bound on resulting equity,
        # unlike an exposure fraction, which uses a lower bound. Marking an
        # existing holding at a higher ask can raise equity before fees.
        cash_base_equity = max(account.equity, account.equity + holding * (ask - market.price))
        cash_unit_loss = min(ask * cost_fraction, ask - market.price + ask * cost_fraction)
        cash_budget = max(0.0, account.cash - config.minimum_cash_fraction * cash_base_equity)
        cash_denominator = ask * (1 + cost_fraction) - config.minimum_cash_fraction * cash_unit_loss
        cash_limit = cash_budget / cash_denominator
        gross_numerator = config.max_gross_fraction * buy_base_equity - account.other_exposure - holding * buy_mark
        gross_limit = max(0.0, gross_numerator / (buy_mark + config.max_gross_fraction * buy_unit_loss))
        quantity_limit = min(quantity_limit, cash_limit, gross_limit)
        side = "BUY"
    else:
        quantity_limit = min(quantity_limit, account.position_quantity)
        if cost_fraction > 1:
            quantity_limit = min(quantity_limit, account.cash / (bid * (cost_fraction - 1)))
        side = "SELL"
    desired = min(abs(delta), quantity_limit)
    quantity = math.floor(desired / config.lot_size) * config.lot_size
    if quantity < config.lot_size:
        return SizingDecision("HOLD", 0, 0.0, target, tuple(reasons + ["cash_or_liquidity_limit"]))
    if desired < abs(delta):
        reasons.append("cash_liquidity_or_turnover_cap")
    return SizingDecision(side, quantity, quantity * execution_price, target, tuple(reasons))
