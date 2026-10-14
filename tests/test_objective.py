"""Synthetic complete-portfolio episodes; no individual action grades."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from stocktrade.execution import EquityPoint
from stocktrade.objective import evaluate_episode
from stocktrade.risk import RiskConfig


START = datetime(2026, 10, 1, 13, 30, tzinfo=timezone.utc)


def points(*values: float, exposure: float = 0.0):
    return tuple(EquityPoint(START + timedelta(minutes=i), value, exposure)
                 for i, value in enumerate(values))


def test_entire_losing_episode_stays_negative_despite_good_exit():
    # Buy at $100, exit at $95; a later fall to $10 does not give the sale
    # a separate positive score or erase the episode's actual $5 loss.
    curve = (
        EquityPoint(START, 1000, 0),
        EquityPoint(START + timedelta(minutes=1), 1000, 100),
        EquityPoint(START + timedelta(minutes=2), 995, 95),
        EquityPoint(START + timedelta(minutes=3), 995, 0),
        EquityPoint(START + timedelta(minutes=4), 995, 0),
    )
    objective = evaluate_episode(1000, 995, equity_curve=curve)
    assert objective.wealth_growth == pytest.approx(-0.005)
    assert objective.admissible


def test_all_cash_episode_has_zero_growth():
    objective = evaluate_episode(1000, 1000, equity_curve=points(1000, 1000, 1000))
    assert objective.wealth_growth == 0
    assert objective.admissible
    assert objective.risk_feasible is objective.admissible
    assert objective.elapsed_steps == 2
    assert objective.elapsed_seconds == 120


def test_fees_are_already_in_terminal_wealth_and_not_subtracted_again():
    # Ledger equity represents $10 gain minus $2 fees.
    objective = evaluate_episode(1000, 1008, equity_curve=points(1000, 1008))
    assert objective.wealth_growth == pytest.approx(0.008)


def test_positive_terminal_growth_does_not_hide_intermediate_risk_breach():
    objective = evaluate_episode(1000, 1100, equity_curve=points(1000, 800, 1100))
    assert objective.wealth_growth == pytest.approx(0.1)
    assert not objective.admissible
    assert set(objective.risk_violations) == {"drawdown_limit", "daily_loss_limit"}
    assert objective.max_drawdown == pytest.approx(0.2)
    assert objective.max_daily_loss == pytest.approx(0.2)


def test_drawdown_is_measured_from_running_peak():
    objective = evaluate_episode(1000, 1040, equity_curve=points(1000, 1250, 1040))
    assert objective.wealth_growth == pytest.approx(0.04)
    assert objective.max_drawdown == pytest.approx(0.168)
    assert objective.risk_violations == ("drawdown_limit",)


def test_gross_and_cash_limits_are_reported_without_objective_penalty():
    config = RiskConfig(max_gross_fraction=0.9)
    objective = evaluate_episode(1000, 1000, equity_curve=points(1000, exposure=850),
                                 risk_config=config)
    assert objective.wealth_growth == 0
    assert objective.risk_violations == ("cash_reserve_cap",)
    assert objective.minimum_cash_fraction == pytest.approx(0.15)
    objective = evaluate_episode(1000, 1000, equity_curve=points(1000, exposure=650))
    assert objective.risk_violations == ("gross_exposure_cap",)


def test_limits_allow_the_boundary():
    config = RiskConfig(max_daily_loss=0.15, max_gross_fraction=0.6, minimum_cash_fraction=0.4)
    objective = evaluate_episode(1000, 850, equity_curve=(
        EquityPoint(START, 1000, 600),
        EquityPoint(START + timedelta(seconds=1), 850, 510),
    ), risk_config=config)
    assert objective.admissible
    assert objective.max_drawdown == pytest.approx(0.15)


def test_terminal_bankruptcy_is_negative_and_infeasible():
    objective = evaluate_episode(1000, 0, equity_curve=points(1000, 0))
    assert objective.wealth_growth == -1
    assert not objective.admissible
    assert "nonpositive_equity" in objective.risk_violations
    assert objective.max_drawdown == 1


def test_repeated_timestamp_uses_complete_last_portfolio_mark():
    curve = (
        EquityPoint(START, 1000, 0),
        EquityPoint(START + timedelta(seconds=1), 750, 0),
        EquityPoint(START + timedelta(seconds=1), 1010, 0),
        EquityPoint(START + timedelta(seconds=2), 1020, 0),
    )
    objective = evaluate_episode(1000, 1020, equity_curve=curve)
    assert objective.admissible
    assert objective.max_drawdown == 0
    assert objective.elapsed_steps == 2
    assert objective.elapsed_seconds == 2


def test_daily_loss_restarts_at_first_observation_of_new_trading_day():
    curve = (
        EquityPoint(START, 1000, 0),
        EquityPoint(START + timedelta(hours=1), 1010, 0),
        EquityPoint(START + timedelta(days=1), 1000, 0),
        EquityPoint(START + timedelta(days=1, hours=1), 969, 0),
    )
    objective = evaluate_episode(1000, 969, equity_curve=curve)
    assert objective.max_daily_loss == pytest.approx(0.031)
    assert objective.risk_violations == ("daily_loss_limit",)


def test_trading_day_does_not_reset_at_utc_midnight():
    # October 1, 19:59 and 20:01 New York time span UTC midnight.
    before_midnight = datetime(2026, 10, 1, 23, 59, tzinfo=timezone.utc)
    curve = (
        EquityPoint(before_midnight, 1000, 0),
        EquityPoint(before_midnight + timedelta(minutes=2), 960, 0),
    )
    objective = evaluate_episode(1000, 960, equity_curve=curve)
    assert objective.max_daily_loss == pytest.approx(0.04)
    assert objective.risk_violations == ("daily_loss_limit",)
    utc_objective = evaluate_episode(1000, 960, equity_curve=curve, trading_timezone="UTC")
    assert utc_objective.max_daily_loss == 0
    assert utc_objective.admissible


def test_explicit_period_records_full_requested_horizon():
    objective = evaluate_episode(1000, 1000, equity_curve=points(1000, 1000),
                                 initial_timestamp=START - timedelta(minutes=1),
                                 final_timestamp=START + timedelta(minutes=2))
    assert objective.elapsed_seconds == 180


def test_elapsed_time_and_steps_use_actual_time_across_repeated_dst_hour():
    market_zone = ZoneInfo("America/New_York")
    before = datetime(2026, 11, 1, 1, 30, tzinfo=market_zone, fold=0)
    after = datetime(2026, 11, 1, 1, 30, tzinfo=market_zone, fold=1)
    objective = evaluate_episode(1000, 1001, equity_curve=(
        EquityPoint(before, 1000, 0), EquityPoint(after, 1001, 0),
    ))
    assert objective.elapsed_seconds == 3600
    assert objective.elapsed_steps == 1


@pytest.mark.parametrize("initial, final", [
    (0, 0), (-1, 1), (float("nan"), 1), (True, 1), (1000, -1),
    (1000, float("inf")), (1000, True),
])
def test_invalid_funding_or_final_equity_is_rejected(initial, final):
    with pytest.raises(ValueError):
        evaluate_episode(initial, final, equity_curve=points(final))


@pytest.mark.parametrize("curve", [
    (),
    (EquityPoint(START.replace(tzinfo=None), 1000, 0),),
    (EquityPoint(START, -1, 0),),
    (EquityPoint(START, float("inf"), 0),),
    (EquityPoint(START, 1000, -1),),
    (EquityPoint(START, 1000, float("nan")),),
    (EquityPoint(START, 1000, 0), EquityPoint(START - timedelta(seconds=1), 1000, 0)),
])
def test_invalid_equity_curves_are_rejected(curve):
    with pytest.raises(ValueError):
        evaluate_episode(1000, 1000, equity_curve=curve)


def test_final_snapshot_must_match_final_equity():
    with pytest.raises(ValueError, match="final portfolio snapshot"):
        evaluate_episode(1000, 1000, equity_curve=points(1000, 500))


@pytest.mark.parametrize("kwargs", [
    {"initial_timestamp": START + timedelta(seconds=1)},
    {"final_timestamp": START - timedelta(seconds=1)},
    {"initial_timestamp": START.replace(tzinfo=None)},
    {"final_timestamp": START.replace(tzinfo=None)},
])
def test_episode_boundaries_cannot_hide_observed_path(kwargs):
    with pytest.raises(ValueError):
        evaluate_episode(1000, 1000, equity_curve=points(1000), **kwargs)
