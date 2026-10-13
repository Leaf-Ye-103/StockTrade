"""Direct cash-valued actions and the portfolio context supplied to the policy."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TradingAction:
    action_id: int
    symbol: str | None
    side: str
    value: float
    fraction: float = 0.0

    def __post_init__(self):
        if self.side not in {"buy", "sell", "hold"}:
            raise ValueError("Action side must be buy, sell, or hold")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("Action value must be a finite nonnegative dollar notional")
        if not math.isfinite(self.fraction) or self.fraction < 0:
            raise ValueError("Action fraction must be finite and nonnegative")
        if self.side == "hold" and (self.symbol is not None or self.value != 0):
            raise ValueError("Hold has no symbol and zero trade value")
        if self.side != "hold" and (not self.symbol or self.value <= 0):
            raise ValueError("Buy/sell actions require a symbol and positive dollar notional")


@dataclass(frozen=True)
class PortfolioState:
    initial_funding: float
    cash: float
    holdings: dict[str, int]
    prices: dict[str, float]
    peak_equity: float | None = None
    day_start_equity: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.initial_funding) or self.initial_funding <= 0:
            raise ValueError("Total funding must be positive and finite")
        if not math.isfinite(self.cash) or self.cash < 0:
            raise ValueError("Cash must be nonnegative and finite")
        for symbol, quantity in self.holdings.items():
            if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 0:
                raise ValueError("Holdings must contain nonnegative whole-share quantities")
            if quantity and symbol not in self.prices:
                raise ValueError("Every held stock requires a current observed price")
        if any(not math.isfinite(price) or price <= 0 for price in self.prices.values()):
            raise ValueError("Observed prices must be positive and finite")

    @property
    def equity(self):
        return self.cash + sum(quantity * self.prices[symbol]
                               for symbol, quantity in self.holdings.items() if quantity)
