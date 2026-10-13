import pytest

from stocktrade.risk import RiskConfig


@pytest.mark.parametrize("kwargs", [
    {"max_position_fraction": float("nan")}, {"minimum_cash_fraction": 1.1},
    {"max_participation": -0.1}, {"lot_size": True}, {"lot_size": 0},
    {"commission_bps": -1}, {"slippage_bps": float("inf")},
])
def test_invalid_hard_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RiskConfig(**kwargs)
