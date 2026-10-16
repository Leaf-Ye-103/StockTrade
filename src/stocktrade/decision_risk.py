"""Evaluation-only, decision-time portfolio risk diagnostics.

These historical scenarios do not forecast prices, grade actions, alter hard
limits, or supply a learning objective. Proposals are conditional portfolios,
not executions. Costs are reported separately, never added to portfolio risk
or charged again to ledger cash.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import math
from numbers import Real

import numpy as np

from .risk import RiskConfig
from .schemas import MarketBar


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _time(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Decision timestamp must be an aware datetime or ISO timestamp") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Decision timestamp must be timezone-aware")
    return value


def _history(bars, symbol, index, timestamp, lookback):
    eligible = []
    history = bars.get(symbol, [])
    # Walk backwards to avoid copying/scanning an entire seconds-level prefix
    # for every decision. Future-dated rows inside that prefix remain excluded.
    for position in range(min(index, len(history) - 1), -1, -1):
        bar = history[position]
        try:
            observed = _time(bar.timestamp)
        except (AttributeError, ValueError):
            return [], "invalid_bar_timestamp"
        if observed <= timestamp:
            eligible.append(bar)
            if len(eligible) == lookback + 1:
                break
    eligible.reverse()
    if not eligible:
        return [], "missing_price_history"
    if any(bar.symbol != symbol or not _finite(bar.close) or bar.close <= 0 for bar in eligible):
        return [], "invalid_price_history"
    if any(left.timestamp >= right.timestamp for left, right in zip(eligible, eligible[1:])):
        return [], "nonchronological_price_history"
    return eligible, None


def _risk(holdings, cash, equity, histories, history_reasons, confidence, minimum_observations):
    active = {symbol: quantity for symbol, quantity in holdings.items() if quantity}
    notionals = {
        symbol: quantity * histories[symbol][-1].close if histories.get(symbol) else None
        for symbol, quantity in active.items()
    }
    if any(value is not None and not _finite(value) for value in notionals.values()):
        notionals = {symbol: value if value is None or _finite(value) else None
                     for symbol, value in notionals.items()}
    gross = equity - cash if equity is not None else None
    def ratio(value):
        return value / equity if value is not None and equity is not None and equity > 0 else None
    concentration = (max(notionals.values(), default=0.0)
                     if all(value is not None for value in notionals.values()) else None)
    result = dict(
        value_at_risk=None, expected_shortfall=None, volatility=None, observations=0, reason=None,
        scenario_interval_seconds=None, warnings=[],
        notionals=notionals, equity=equity, cash=cash, gross_exposure=gross,
        gross_fraction=ratio(gross), cash_fraction=ratio(cash), max_position_fraction=ratio(concentration),
        market_stress={f"-{percentage}%": -percentage / 100 * gross if gross is not None else None
                       for percentage in (1, 5, 10)},
    )
    if not active:
        result.update(value_at_risk=0.0, expected_shortfall=0.0, volatility=0.0, reason="cash_only_portfolio")
        return result
    missing = [symbol for symbol in active if notionals[symbol] is None]
    if missing:
        result["reason"] = ";".join(f"{symbol}:{history_reasons.get(symbol) or 'nonfinite_current_notional'}"
                                    for symbol in sorted(missing))
        return result
    scenarios = {}
    for symbol in active:
        history = histories[symbol]
        scenarios[symbol] = {
            (left.timestamp, right.timestamp): right.close / left.close - 1
            for left, right in zip(history, history[1:])
        }
    common = set.intersection(*(set(values) for values in scenarios.values()))
    result["observations"] = len(common)
    if common:
        durations = [(right - left).total_seconds() for left, right in common]
        result["scenario_interval_seconds"] = dict(minimum=min(durations), median=float(np.median(durations)),
                                                   maximum=max(durations))
        if min(durations) != max(durations):
            result["warnings"].append("Historical return intervals have unequal elapsed durations; "
                                      "this is an observed-bar scenario distribution, not a fixed-minute forecast")
    if len(common) < minimum_observations:
        result["reason"] = "insufficient_common_historical_returns"
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        pnl = np.asarray([sum(notionals[symbol] * scenarios[symbol][pair] for symbol in active)
                          for pair in sorted(common)], dtype=float)
        losses = -pnl
        if not np.isfinite(losses).all():
            result["reason"] = "nonfinite_historical_scenarios"
            return result
        threshold = float(np.quantile(losses, confidence, method="higher"))
        expected_shortfall = max(0.0, float(losses[losses >= threshold].mean()))
        volatility = float(pnl.std(ddof=0))
    if not all(math.isfinite(value) for value in (threshold, expected_shortfall, volatility)):
        result["reason"] = "nonfinite_historical_statistics"
        return result
    result.update(value_at_risk=max(0.0, threshold), expected_shortfall=expected_shortfall,
                  volatility=volatility)
    return result


def _proposal_details(bar, quantity, notional, risk_config):
    details = dict(
        spread_bps=None, volume_participation=None, quote_reason=None, liquidity_reason=None,
        estimates=dict(commission_cost=notional * (risk_config.commission_bps / 10000),
                       half_spread_cost=None,
                       configured_slippage_cost=notional * (risk_config.slippage_bps / 10000)),
        estimate_reasons={},
        estimates_are_charged=False,
        cost_interpretation="Conditional cost estimates; not ledger charges or additions to VaR/ES",
    )
    for name, value in details["estimates"].items():
        if value is not None and not _finite(value):
            details["estimates"][name] = None
            details["estimate_reasons"][name] = "nonfinite_cost_estimate"
    if quantity == 0:
        details["estimates"]["half_spread_cost"] = 0.0
        return details
    if bar is None:
        details.update(quote_reason="missing_current_bar", liquidity_reason="missing_current_bar")
        return details
    if _finite(bar.volume) and bar.volume > 0:
        details["volume_participation"] = quantity / bar.volume
    else:
        details["liquidity_reason"] = "no_positive_observed_volume"
    if bar.bid is None or bar.ask is None:
        details["quote_reason"] = "missing_executable_quotes"
    elif not _finite(bar.bid) or not _finite(bar.ask) or bar.bid <= 0 or bar.ask < bar.bid:
        details["quote_reason"] = "invalid_executable_quotes"
    else:
        width = bar.ask - bar.bid
        details["spread_bps"] = width / (bar.bid + width / 2) * 10000
        half_spread = quantity * (width / 2)
        if _finite(half_spread):
            details["estimates"]["half_spread_cost"] = half_spread
        else:
            details["estimate_reasons"]["half_spread_cost"] = "nonfinite_cost_estimate"
    return details


def _observed_execution(record):
    accounting = record.get("accounting") or {}
    if is_dataclass(accounting):
        accounting = asdict(accounting)
    if not isinstance(accounting, dict):
        accounting = {}
    matched = accounting.get("status") == "matched"
    result = dict(matched=matched, fee=None, slippage_cost=None, reason="no_matched_execution",
                  interpretation="Observed-match ledger amounts; fees already included in ledger cash")
    if not matched:
        return result
    fee = accounting.get("fee")
    result["fee"] = float(fee) if _finite(fee) and fee >= 0 else None
    match = accounting.get("match") or {}
    intent = record.get("intent") or accounting.get("intent") or {}
    if is_dataclass(match):
        match = asdict(match)
    if is_dataclass(intent):
        intent = asdict(intent)
    price, quantity, reference = match.get("estimated_price"), match.get("estimated_quantity"), intent.get("reference_price")
    if (all(_finite(value) and value > 0 for value in (price, quantity, reference))
            and intent.get("side") in ("buy", "sell")):
        sign = 1 if intent["side"] == "buy" else -1
        slippage = sign * (price - reference) * quantity
        result["slippage_cost"] = slippage if _finite(slippage) else None
    result["reason"] = (None if result["fee"] is not None and result["slippage_cost"] is not None
                        else "incomplete_matched_execution_cost_fields")
    return result


def _json_safe(value):
    """Final guard against overflow in diagnostic arithmetic, never emit NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, (bool, int)) and not _finite(value):
        return None
    return value


def estimate_decision_risk(
    *, bars: dict[str, list[MarketBar]], record: dict, risk_config: RiskConfig,
    lookback=60, confidence=.95, minimum_observations=20,
) -> dict:
    """Compare current and feasible proposed portfolios using past one-bar PnL.

    Up to ``lookback`` simple-return scenarios are intersected by their complete
    timestamp pairs across held symbols. Current dollar notionals preserve
    observed cross-stock correlation. VaR is the empirical upper loss quantile
    (``higher`` convention); ES averages losses at least that quantile. Both
    are floored at zero. Volatility is population dollar-PnL standard deviation.
    Metrics are neither annualized nor conditional price predictions.
    """
    if not isinstance(risk_config, RiskConfig):
        raise ValueError("risk_config must be a RiskConfig")
    RiskConfig(**asdict(risk_config))  # Revalidate configuration, including mutated instances.
    for name, value in (("lookback", lookback), ("minimum_observations", minimum_observations)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if minimum_observations > lookback:
        raise ValueError("minimum_observations must not exceed lookback")
    if not _finite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between zero and one")
    timestamp = _time(record.get("timestamp"))
    index = record.get("bar_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("bar_index must be a nonnegative integer")
    account = record.get("account_before") or {}
    holdings = account.get("holdings")
    equity, cash = account.get("equity"), account.get("cash")
    if (not isinstance(holdings, dict) or not _finite(equity) or not _finite(cash)
            or equity < 0 or cash < 0 or cash > equity + 1e-8):
        raise ValueError("account_before requires finite long-only equity, cash and holdings")
    if any(not isinstance(symbol, str) or not symbol or not isinstance(quantity, int)
           or isinstance(quantity, bool) or quantity < 0 or not _finite(quantity)
           for symbol, quantity in holdings.items()):
        raise ValueError("Holdings must contain nonnegative finite whole-share quantities")
    action = record.get("policy_action") or {}
    side, symbol = action.get("side"), action.get("symbol")
    if side not in ("buy", "sell", "hold") or (side != "hold" and (not isinstance(symbol, str) or not symbol)):
        raise ValueError("A direct buy, sell or hold policy_action is required")
    if not _finite(action.get("value")) or action["value"] < 0:
        raise ValueError("Policy action dollar value must be finite and nonnegative")
    if (side == "hold" and (symbol is not None or action["value"] != 0)
            or side != "hold" and action["value"] <= 0):
        raise ValueError("Hold has no symbol or value; buy/sell requires a positive dollar value")
    feasibility = record.get("feasibility") or {}
    accepted = feasibility.get("accepted")
    quantity, notional = feasibility.get("quantity", 0), feasibility.get("notional", 0.0)
    if (not isinstance(accepted, bool) or not isinstance(quantity, int) or isinstance(quantity, bool)
            or not _finite(quantity) or quantity < 0 or not _finite(notional) or notional < 0):
        raise ValueError("Feasibility requires an accepted flag, whole-share quantity and finite notional")
    active_proposal = accepted and side != "hold" and quantity > 0
    quantity, notional = (quantity, float(notional)) if active_proposal else (0, 0.0)
    if active_proposal and notional <= 0:
        raise ValueError("A feasible nonzero quantity requires positive notional")
    if active_proposal and side == "sell" and quantity > holdings.get(symbol, 0):
        raise ValueError("Projected sales cannot exceed observed holdings")
    histories, history_reasons = {}, {}
    for stock in set(holdings) | ({symbol} if symbol else set()):
        histories[stock], history_reasons[stock] = _history(bars, stock, index, timestamp, lookback)
    before = _risk(holdings, cash, equity, histories, history_reasons, confidence, minimum_observations)
    projected_holdings = dict(holdings)
    projected_cash, projected_equity = float(cash), float(equity)
    if active_proposal:
        sign = 1 if side == "buy" else -1
        projected_holdings[symbol] = projected_holdings.get(symbol, 0) + sign * quantity
        projected_cash -= sign * notional
        if projected_cash < -1e-8:
            raise ValueError("Projected purchase exceeds observed available cash")
        projected_cash = max(0.0, projected_cash)
        selected = histories[symbol]
        projected_equity = equity + sign * (quantity * selected[-1].close - notional) if selected else None
    projected = _risk(projected_holdings, projected_cash, projected_equity, histories,
                      history_reasons, confidence, minimum_observations)
    projected.update(projected_not_an_execution=True, cash_excludes_cost_estimates=True)
    delta = {name: projected[name] - before[name] if projected[name] is not None and before[name] is not None else None
             for name in ("value_at_risk", "expected_shortfall", "volatility", "gross_exposure",
                          "gross_fraction", "cash_fraction", "max_position_fraction")}
    delta["interpretation"] = "Signed projected minus before; descriptive changes, not action scores"
    bar = histories.get(symbol, [])[-1] if histories.get(symbol) else None
    proposal = dict(accepted=accepted, applied_quantity=quantity, projected_notional=notional,
                    projected_not_an_execution=True, **_proposal_details(bar, quantity, notional, risk_config))
    return _json_safe(dict(
        timestamp=timestamp.astimezone(timezone.utc).isoformat(), unit="USD", confidence=float(confidence),
        horizon="one_observed_bar", lookback=lookback, minimum_observations=minimum_observations,
        quantile_method="higher", volatility_convention="population_standard_deviation",
        interpretation="Evaluation-only historical scenarios at current notionals; no prediction or action grading",
        before=before, projected=projected, delta=delta, proposal=proposal,
        observed_execution=_observed_execution(record),
    ))
