"""Normalized, point-in-time research contracts.

All timestamps must be timezone-aware. A bar timestamp is its END, when its
OHLCV is available, never its opening time. Information must retain both its
publication timestamp and the earliest timestamp accessible to the strategy.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class MarketBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None


@dataclass(frozen=True)
class InformationEvent:
    published_at: datetime
    available_at: datetime
    source: str
    kind: str
    values: dict[str, float | None] = field(default_factory=dict)
    text: str = ""
    symbol: str | None = None


@dataclass(frozen=True)
class FeatureFrame:
    timestamps: list[datetime]
    values: np.ndarray
    observed: np.ndarray
    feature_names: list[str]
    groups: dict[str, list[int]]
    prices: np.ndarray
    symbol: str
