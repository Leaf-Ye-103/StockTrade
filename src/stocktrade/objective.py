"""A portfolio episode's sole financial objective: terminal net asset growth.

This module never grades an individual trade, invents a counterfactual action,
or subtracts a hand-written risk penalty. Commissions and execution costs are
already reflected in the ledger's equity. Risk feasibility is reported
separately from the observed financial outcome.

Compare episodes over the same period. Annualizing or selecting a convenient
future endpoint would change the training objective and favor unequal episodes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from numbers import Real
from zoneinfo import ZoneInfo

from .execution import EquityPoint
from .risk import RiskConfig


@dataclass(frozen=True)
class EpisodeObjective:
    initial_funding: float
    final_equity: float
    wealth_growth: float
    elapsed_seconds: float
    elapsed_steps: int
    admissible: bool
    risk_violations: tuple[str, ...]
    max_drawdown: float
    max_daily_loss: float
    max_gross_fraction: float
    minimum_cash_fraction: float

    @property
    def risk_feasible(self) -> bool:
        """Whether the observed trajectory satisfied the checked constraints."""
        return self.admissible


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _aware_time(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def evaluate_episode(
    initial_funding: float,
    final_equity: float,
    *,
    equity_curve: Sequence[EquityPoint],
    risk_config: RiskConfig | None = None,
    initial_timestamp: datetime | None = None,
    final_timestamp: datetime | None = None,
    trading_timezone: str = "America/New_York",
) -> EpisodeObjective:
    """Measure net wealth and observed risk over a complete portfolio episode.

    The learning signal is ``final_equity / initial_funding - 1`` for the
    entire trajectory. It is unchanged by risk breaches: callers must enforce
    feasibility separately rather than train on a fabricated financial penalty.
    In particular, a loss-cutting sale is not assigned an independent reward.

    All held assets must be marked at each observation. The last snapshot at
    a repeated timestamp is retained so sequential per-symbol mark calls do
    not mistake partially updated portfolios for complete observations. Each
    retained snapshot is one step; elapsed steps count the intervals between
    those snapshots. Explicit timestamps can include an initial cash-only
    period but cannot omit any observed portion of the episode.

    Drawdown uses the running peak, including initial funding. Daily loss uses
    initial funding on the episode's starting trading day and the first full
    snapshot on each subsequent trading day. Overnight losses remain captured
    by drawdown. Default trading dates follow the US stock market timezone.

    Gross exposure and cash reserve checks apply to the complete long-only
    portfolio. Position concentration, turnover, spread, participation, and
    data quality cannot be recovered from an aggregate equity curve; they must
    still be constrained when proposing actions. Gaps can breach a constraint
    despite action-time checks, so feasibility is an observation, not a promise.
    """
    if not _finite_real(initial_funding) or initial_funding <= 0:
        raise ValueError("initial_funding must be positive and finite")
    if not _finite_real(final_equity) or final_equity < 0:
        raise ValueError("final_equity must be nonnegative and finite")
    if not equity_curve:
        raise ValueError("equity_curve must contain at least one complete snapshot")
    config = risk_config or RiskConfig()
    trading_tz = ZoneInfo(trading_timezone)
    snapshots: list[EquityPoint] = []
    for point in equity_curve:
        if not isinstance(point, EquityPoint) or not _aware_time(point.timestamp):
            raise ValueError("equity snapshots require timezone-aware timestamps")
        if (not _finite_real(point.equity) or point.equity < 0
                or not _finite_real(point.gross_exposure) or point.gross_exposure < 0):
            raise ValueError("snapshot equity and gross exposure must be nonnegative and finite")
        # Normalize before comparisons: wall times repeat when daylight saving
        # ends, and Python compares same-zone datetime folds by their wall time.
        point = EquityPoint(point.timestamp.astimezone(timezone.utc), point.equity, point.gross_exposure)
        if snapshots and point.timestamp < snapshots[-1].timestamp:
            raise ValueError("equity snapshots must be in chronological order")
        if snapshots and point.timestamp == snapshots[-1].timestamp:
            snapshots[-1] = point
        else:
            snapshots.append(point)
    start = snapshots[0].timestamp if initial_timestamp is None else initial_timestamp
    end = snapshots[-1].timestamp if final_timestamp is None else final_timestamp
    if not _aware_time(start) or not _aware_time(end):
        raise ValueError("episode boundaries must be timezone-aware timestamps")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    if start > snapshots[0].timestamp or end < snapshots[-1].timestamp or end < start:
        raise ValueError("episode boundaries must include the complete equity curve")
    if not math.isclose(final_equity, snapshots[-1].equity, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError("final_equity must match the final portfolio snapshot")
    wealth_growth = float(final_equity) / float(initial_funding) - 1.0
    if not math.isfinite(wealth_growth):
        raise ValueError("terminal wealth growth must be finite")

    peak = float(initial_funding)
    day = start.astimezone(trading_tz).date()
    day_start_equity = float(initial_funding)
    max_drawdown = 0.0
    max_daily_loss = 0.0
    max_gross_fraction = 0.0
    minimum_cash_fraction = 1.0
    nonpositive_equity = False
    for point in snapshots:
        point_day = point.timestamp.astimezone(trading_tz).date()
        if point_day != day:
            day, day_start_equity = point_day, point.equity
        peak = max(peak, point.equity)
        max_drawdown = max(max_drawdown, 1.0 - point.equity / peak)
        if day_start_equity > 0:
            max_daily_loss = max(max_daily_loss, 1.0 - point.equity / day_start_equity)
        if point.equity == 0:
            nonpositive_equity = True
            gross_fraction = math.inf if point.gross_exposure > 0 else 0.0
            cash_fraction = 0.0
        else:
            gross_fraction = point.gross_exposure / point.equity
            cash_fraction = 1.0 - gross_fraction
        max_gross_fraction = max(max_gross_fraction, gross_fraction)
        minimum_cash_fraction = min(minimum_cash_fraction, cash_fraction)

    tolerance = 1e-10
    violations: list[str] = []
    if max_drawdown > config.max_drawdown + tolerance:
        violations.append("drawdown_limit")
    if max_daily_loss > config.max_daily_loss + tolerance:
        violations.append("daily_loss_limit")
    if max_gross_fraction > config.max_gross_fraction + tolerance:
        violations.append("gross_exposure_cap")
    if minimum_cash_fraction < config.minimum_cash_fraction - tolerance:
        violations.append("cash_reserve_cap")
    if nonpositive_equity:
        violations.append("nonpositive_equity")
    return EpisodeObjective(
        float(initial_funding), float(final_equity), wealth_growth,
        (end - start).total_seconds(), len(snapshots) - 1, not violations,
        tuple(violations), max_drawdown, max_daily_loss,
        max_gross_fraction, minimum_cash_fraction,
    )
