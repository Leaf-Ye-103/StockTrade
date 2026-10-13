"""Mature, observed-data action feedback for direct policy learning.

The active portfolio reward compares the actual observed-analog ledger path,
including subsequent supported actions, with retaining the pre-action cash and
shares. This rewards a protective sale while retaining the cost of a bad entry
in that entry's feedback. The isolated-action scorer remains a diagnostic; its
independent buy/sell advantages must never be summed to assign policy credit.
Both no-action baselines need no additional trade or invented execution. Price
marks are valuation observations, never assumed executable exits.

Rewards are shaped observational estimates, not additive portfolio profits or
causal market-impact estimates. Their predeclared horizons and downside weights
must be frozen before evaluating a validation/test interval. An unsupported
execution receives no economic feedback; the end of a dataset is not a shorter
reward horizon. There is deliberately no forecast dependency in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from .execution import MatchResult

if TYPE_CHECKING:
    from .policy_contracts import TradingAction


def _aware(timestamp: datetime) -> bool:
    return isinstance(timestamp, datetime) and timestamp.utcoffset() is not None


def _finite_positive(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class RewardConfig:
    """Fixed future-bar checkpoints and risk shaping, chosen before a rollout.

    Each fraction maps to ``ceil(fraction * horizon_steps)`` observed bars after
    the execution. Coalescing checkpoints at small horizons is intentional.
    A reward requires the entire final horizon, even when an early checkpoint
    alone would have favored an action. ``fee_bps`` applies once to the recorded
    gross execution value; spread/slippage already enter through its price.
    """

    horizon_steps: int = 8
    horizon_fractions: tuple[float, ...] = (0.125, 0.5, 1.0)
    horizon_weights: tuple[float, ...] = (0.25, 0.25, 0.5)
    downside_weight: float = 0.25
    fee_bps: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.horizon_steps, bool) or not isinstance(self.horizon_steps, int) or self.horizon_steps < 1:
            raise ValueError("horizon_steps must be a positive integer")
        if not isinstance(self.horizon_fractions, tuple) or not isinstance(self.horizon_weights, tuple):
            raise ValueError("Horizon fractions and weights must be immutable tuples")
        if not self.horizon_fractions or len(self.horizon_fractions) != len(self.horizon_weights):
            raise ValueError("Every horizon fraction needs exactly one weight")
        if any(not math.isfinite(value) or not 0 < value <= 1 for value in self.horizon_fractions):
            raise ValueError("Horizon fractions must be finite and in (0, 1]")
        if tuple(sorted(self.horizon_fractions)) != self.horizon_fractions or self.horizon_fractions[-1] != 1:
            raise ValueError("Horizons must be increasing and include the full horizon")
        if any(not math.isfinite(value) or value < 0 for value in self.horizon_weights):
            raise ValueError("Horizon weights must be finite and nonnegative")
        if not math.isclose(sum(self.horizon_weights), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Horizon weights must sum to one")
        for name in ("downside_weight", "fee_bps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ActionFeedback:
    action_id: int
    symbol: str | None
    side: str
    decision_timestamp: datetime
    execution_timestamp: datetime | None
    maturity_timestamp: datetime | None
    status: str
    reward: float | None
    components: dict[str, float]
    source_trade_ids: tuple[str, ...]
    baseline_description: str


@dataclass(frozen=True)
class PortfolioObservation:
    """Observed-analog ledger equity and available stock marks at one bar end.

    Equity must already include every supported execution's actual recorded
    price, quantity, and fee. The evaluator does not construct those fills.
    """

    timestamp: datetime
    equity: float
    prices: dict[str, float]

    def __post_init__(self) -> None:
        if not _aware(self.timestamp):
            raise ValueError("Portfolio timestamp must be timezone-aware")
        if not math.isfinite(self.equity) or self.equity < 0:
            raise ValueError("Portfolio equity must be finite and nonnegative")
        if any(not symbol or not _finite_positive(price) for symbol, price in self.prices.items()):
            raise ValueError("Portfolio marks need symbols and finite positive prices")


def evaluate_portfolio_action(
    action: TradingAction,
    match: MatchResult | None,
    *,
    decision_timestamp: datetime,
    initial_funding: float,
    cash_before: float,
    holdings_before: dict[str, int],
    equity_before: float,
    future_observations: Sequence[PortfolioObservation],
    config: RewardConfig | None = None,
    supporting_trade_ids: tuple[str, ...] = (),
) -> ActionFeedback:
    """Evaluate the realized policy continuation versus retaining its portfolio.

    The fixed baseline retains ``cash_before`` and ``holdings_before`` without
    any hypothetical trade. The actual path may contain subsequent orders only
    when supported by recorded execution evidence. This prevents a protective
    sale's avoided-loss advantage from being added to a bad entry's feedback:
    buying at 100 and later selling at 95 leaves that entry's actual path below
    its original 100 cash baseline, even if the stock subsequently falls to 10.

    Each action receives its own complete-horizon continuation feedback. These
    overlapping advantages are training targets, not additive portfolio PnL.
    Additional source identifiers must refer to actual supported later orders
    within this horizon. The caller preserves the underlying execution ledger.
    Fees and spread are already in actual equity and are never charged again;
    ``config.fee_bps`` applies only to the isolated ``evaluate_action`` scorer.
    """
    config = config or RewardConfig()
    if not _aware(decision_timestamp):
        raise ValueError("Decision timestamp must be timezone-aware")
    if not _finite_positive(initial_funding) or not _finite_positive(equity_before):
        raise ValueError("Funding and equity must be finite and positive")
    if not math.isfinite(cash_before) or cash_before < 0:
        raise ValueError("Pre-action cash must be finite and nonnegative")
    if any(
        not symbol or isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0
        for symbol, quantity in holdings_before.items()
    ):
        raise ValueError("Pre-action holdings must contain symbols and nonnegative integer quantities")
    if action.side not in {"buy", "sell", "hold"}:
        raise ValueError("Action side must be buy, sell or hold")
    if not isinstance(supporting_trade_ids, tuple) or any(
        not isinstance(source_id, str) or not source_id for source_id in supporting_trade_ids
    ):
        raise ValueError("Supporting execution IDs must be a tuple of nonempty identifiers")

    baseline = (
        "Retain pre-action cash and existing shares without additional trades; "
        "compare with the actual observed-analog ledger including supported later actions."
    )
    own_ids = match.source_trade_ids if match is not None else ()
    source_ids = tuple(dict.fromkeys((*own_ids, *supporting_trade_ids)))

    def feedback(
        status: str,
        reward: float | None = None,
        components: dict[str, float] | None = None,
        *,
        maturity: datetime | None = None,
    ) -> ActionFeedback:
        return ActionFeedback(
            action_id=action.action_id,
            symbol=action.symbol,
            side=action.side,
            decision_timestamp=decision_timestamp,
            execution_timestamp=match.event_timestamp if match is not None else None,
            maturity_timestamp=maturity,
            status=status,
            reward=reward,
            components=components or {},
            source_trade_ids=source_ids if status != "unmatched" else (),
            baseline_description=baseline,
        )

    if action.side != "hold" and (match is None or match.status != "matched"):
        return feedback("unmatched")
    if action.side == "hold" and match is not None:
        raise ValueError("Hold actions must not contain their own execution evidence")
    anchor = decision_timestamp
    if match is not None:
        assert match.event_timestamp is not None
        anchor = match.event_timestamp
        if anchor < decision_timestamp:
            raise ValueError("Execution cannot precede its decision")
        if action.side == "sell" and match.estimated_quantity > holdings_before.get(action.symbol, 0):
            raise ValueError("A sale cannot exceed pre-action existing holdings")

    previous_timestamp = anchor
    for observation in future_observations:
        if observation.timestamp <= previous_timestamp:
            raise ValueError("Portfolio observations must be strictly increasing and after execution/decision")
        for symbol, quantity in holdings_before.items():
            if quantity and symbol not in observation.prices:
                raise ValueError(f"Missing future observed mark for existing holding {symbol}")
        previous_timestamp = observation.timestamp
    if len(future_observations) < config.horizon_steps:
        return feedback("immature")
    observations = future_observations[: config.horizon_steps]
    baseline_equities = [
        cash_before + sum(quantity * observation.prices[symbol]
                          for symbol, quantity in holdings_before.items() if quantity)
        for observation in observations
    ]
    weighted_advantage = sum(
        weight * (observations[index].equity - baseline_equities[index])
        for fraction, weight in zip(config.horizon_fractions, config.horizon_weights, strict=True)
        for index in (math.ceil(fraction * config.horizon_steps) - 1,)
    )
    baseline_downside = max(0.0, equity_before - min(baseline_equities))
    actual_downside = max(0.0, equity_before - min(observation.equity for observation in observations))
    downside_improvement = config.downside_weight * (baseline_downside - actual_downside)
    reward_dollars = weighted_advantage + downside_improvement
    components = {
        "actual_vs_hold_advantage": weighted_advantage,
        "terminal_actual_vs_hold_advantage": observations[-1].equity - baseline_equities[-1],
        "downside_improvement": downside_improvement,
        "baseline_worst_drawdown": baseline_downside,
        "actual_worst_drawdown": actual_downside,
        "fee_already_in_ledger": 1.0,
        "reward_dollars": reward_dollars,
        "equity_before": equity_before,
        "normalization_funding": initial_funding,
    }
    return feedback(
        "supported", reward_dollars / initial_funding, components, maturity=observations[-1].timestamp
    )


def evaluate_action(
    action: TradingAction,
    match: MatchResult | None,
    *,
    decision_timestamp: datetime,
    decision_price: float,
    future_marks: Sequence[tuple[datetime, float]],
    initial_funding: float,
    equity_before: float,
    config: RewardConfig | None = None,
) -> ActionFeedback:
    """Score supported actions from actual execution and fixed later marks.

    ``future_marks`` must contain strictly increasing, timezone-aware observed
    bars *after the execution*, or after the decision for a hold. The caller
    must keep their timestamps within the relevant chronological partition.
    The scorer never predicts a price, assumes a future exit, chooses the best
    hindsight endpoint, or provides feedback before the full horizon matures.

    Cash has a zero nominal return baseline. Funding is a fixed denominator for
    the complete experiment; normalizing by action value would let tiny orders
    earn the same reward as economically consequential ones. ``equity_before``
    is retained as an audit diagnostic and never changes that denominator.
    """
    config = config or RewardConfig()
    if not _aware(decision_timestamp):
        raise ValueError("Decision timestamp must be timezone-aware")
    if action.side not in {"buy", "sell", "hold"}:
        raise ValueError("Action side must be buy, sell or hold")
    if not _finite_positive(initial_funding) or not _finite_positive(equity_before):
        raise ValueError("Funding and equity must be finite and positive")
    if action.side != "hold" and not _finite_positive(decision_price):
        raise ValueError("Decision price must be finite and positive")

    baseline = {
        "buy": "Retain the purchase cash; value the observed acquired shares at fixed future marks.",
        "sell": "Retain the same existing shares; compare their future marks with observed sale cash.",
        "hold": "No action versus no action; no unobserved acquisition or liquidation is assumed.",
    }[action.side]

    def feedback(
        status: str,
        reward: float | None = None,
        components: dict[str, float] | None = None,
        *,
        maturity: datetime | None = None,
    ) -> ActionFeedback:
        return ActionFeedback(
            action_id=action.action_id,
            symbol=action.symbol,
            side=action.side,
            decision_timestamp=decision_timestamp,
            execution_timestamp=match.event_timestamp if match is not None else None,
            maturity_timestamp=maturity,
            status=status,
            reward=reward,
            components=components or {},
            source_trade_ids=match.source_trade_ids if match is not None else (),
            baseline_description=baseline,
        )

    if action.side != "hold" and (match is None or match.status != "matched"):
        return feedback("unmatched")
    if action.side == "hold" and match is not None:
        raise ValueError("Hold actions must not contain execution evidence")
    anchor = decision_timestamp
    if match is not None:
        # MatchResult validates its matched execution fields at construction.
        assert match.event_timestamp is not None
        anchor = match.event_timestamp
        if anchor < decision_timestamp:
            raise ValueError("Execution cannot precede its decision")

    previous_timestamp = anchor
    for timestamp, price in future_marks:
        if not _aware(timestamp) or timestamp <= previous_timestamp:
            raise ValueError("Future marks must be strictly increasing and after execution/decision")
        if not _finite_positive(price):
            raise ValueError("Future marks must have finite positive prices")
        previous_timestamp = timestamp
    if len(future_marks) < config.horizon_steps:
        return feedback("immature")
    observations = future_marks[: config.horizon_steps]
    maturity = observations[-1][0]
    if action.side == "hold":
        return feedback("supported", 0.0, {"reward_dollars": 0.0}, maturity=maturity)

    assert match is not None and match.estimated_price is not None
    observed_price = match.estimated_price
    quantity = match.estimated_quantity
    direction = 1.0 if action.side == "buy" else -1.0
    marked_advantage = sum(
        weight
        * direction
        * quantity
        * (observations[math.ceil(fraction * config.horizon_steps) - 1][1] - observed_price)
        for fraction, weight in zip(config.horizon_fractions, config.horizon_weights, strict=True)
    )
    # An added long position introduces this downside exposure; removing the
    # same shares avoids it. The minimum is a fixed-window risk statistic, not
    # a selected sale price or a promise the position could exit there.
    downside_dollars = quantity * max(0.0, observed_price - min(price for _, price in observations))
    downside_adjustment = -direction * config.downside_weight * downside_dollars
    fees = quantity * observed_price * config.fee_bps / 10_000
    reward_dollars = marked_advantage + downside_adjustment - fees
    components = {
        "weighted_marked_advantage": marked_advantage,
        "downside_adjustment": downside_adjustment,
        "fees": fees,
        "reward_dollars": reward_dollars,
        "terminal_marked_advantage": direction * quantity * (observations[-1][1] - observed_price),
        "gross_execution_value": quantity * observed_price,
        "decision_price": decision_price,
        "equity_before": equity_before,
        "normalization_funding": initial_funding,
    }
    return feedback("supported", reward_dollars / initial_funding, components, maturity=maturity)
