"""Observed-market contracts and hard limits for direct action policies."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskConfig:
    max_position_fraction: float = 0.20
    max_gross_fraction: float = 0.60
    minimum_cash_fraction: float = 0.20
    max_participation: float = 0.01
    max_turnover_fraction: float = 0.10
    max_drawdown: float = 0.15
    max_daily_loss: float = 0.03
    minimum_data_quality: float = 0.50
    max_spread_bps: float = 50.0
    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    lot_size: int = 1

    def __post_init__(self):
        fractions = (self.max_position_fraction, self.max_gross_fraction,
                      self.minimum_cash_fraction, self.max_participation,
                      self.max_turnover_fraction, self.max_drawdown, self.max_daily_loss,
                      self.minimum_data_quality)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in fractions):
            raise ValueError("Risk fractions must be finite and between zero and one")
        if isinstance(self.lot_size, bool) or not isinstance(self.lot_size, int) or self.lot_size < 1:
            raise ValueError("lot_size must be a positive integer")
        if any(not math.isfinite(value) or value < 0
               for value in (self.max_spread_bps, self.commission_bps, self.slippage_bps)):
            raise ValueError("Costs and spread limits must be finite and nonnegative")


@dataclass(frozen=True)
class MarketState:
    symbol: str
    price: float
    available_volume: float
    spread_bps: float = 0.0
    data_quality: float = 1.0
    bid: float | None = None
    ask: float | None = None
