"""Evaluation-only descriptive analytics for a completed observed-trade ledger.

This module has no dependency on a policy, optimizer or training pipeline.
Prices are raw prices: corporate actions, dividends, executable benchmarks,
and unobserved fills cannot be inferred from these inputs.
"""

from collections import Counter, deque
from datetime import datetime, time, timezone
import math
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np

from .schemas import MarketBar


_NY = ZoneInfo("America/New_York")
_MIN_DAILY = 20
_YEAR_SECONDS = 365.25 * 86400


def _timestamp(value) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Analytics timestamps must have an explicit timezone")
    return timestamp.astimezone(timezone.utc)


def _number(value):
    if isinstance(value, (bool, str)) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _clean(value):
    """Produce strict JSON data, including for unavailable numerical results."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _daily_returns(points):
    """NY-day closes, requiring an observed mark at/after 16:00 local time.

    The first observed close is only a baseline. No funding-to-partial-first-day
    return is annualized; partial final days supply no daily close. A missing
    close breaks return adjacency rather than creating a multi-day daily return.
    No holiday calendar or unobserved session close is invented.
    """
    by_day = {}
    for timestamp, value in points:
        local = timestamp.astimezone(_NY)
        by_day[local.date()] = (timestamp, value)
    returns, closes = [], []
    previous = None
    incomplete = []
    gaps = []
    for date, (timestamp, value) in sorted(by_day.items()):
        if timestamp.astimezone(_NY).time().replace(tzinfo=None) < time(16):
            incomplete.append(date.isoformat())
            previous = None
            continue
        closes.append(dict(date=date.isoformat(), timestamp=timestamp, value=value))
        if previous is not None and np.busday_count(previous[0].isoformat(), date.isoformat()) > 1:
            gaps.append(dict(previous_date=previous[0].isoformat(), next_date=date.isoformat()))
            previous = None
        if previous is not None and previous[1] > 0 and value >= 0:
            returns.append(dict(start_date=previous[0].isoformat(), end_date=date.isoformat(),
                                value=value / previous[1] - 1))
        previous = (date, value)
    return returns, closes, incomplete, gaps


def _sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else None


def _drawdowns(points, funding):
    if not points:
        return dict(max_drawdown=None, peak_timestamp=None, trough_timestamp=None,
                    recovery_timestamp=None, max_drawdown_duration_seconds=None,
                    recovery_seconds=None, longest_underwater_episode_seconds=None,
                    time_underwater_seconds=None, time_underwater_fraction=None,
                    currently_underwater=None), []
    peak = funding
    peak_time = points[0]["timestamp"] if points else None
    max_drawdown = 0.0
    worst_peak = worst_trough = recovery = None
    worst_peak_value = None
    drawdowns = []
    episodes = []
    active_peak = None
    for point in points:
        timestamp, equity = point["timestamp"], point["equity"]
        if equity >= peak:
            if active_peak is not None:
                episodes.append((active_peak, timestamp))
                active_peak = None
            if worst_trough is not None and recovery is None and equity >= worst_peak_value:
                recovery = timestamp
            peak, peak_time = equity, timestamp
        drawdown = max(0.0, 1 - equity / peak) if peak > 0 else 0.0
        drawdowns.append(drawdown)
        if drawdown > 0 and active_peak is None:
            active_peak = peak_time
        if drawdown > max_drawdown:
            max_drawdown, worst_peak, worst_trough = drawdown, peak_time, timestamp
            worst_peak_value, recovery = peak, None
    if active_peak is not None and points:
        episodes.append((active_peak, points[-1]["timestamp"]))
    durations = [(end - start).total_seconds() for start, end in episodes]
    underwater = sum(
        (points[index + 1]["timestamp"] - points[index]["timestamp"]).total_seconds()
        for index in range(len(points) - 1) if drawdowns[index] > 0
    )
    elapsed = (points[-1]["timestamp"] - points[0]["timestamp"]).total_seconds() if len(points) > 1 else 0
    return dict(
        max_drawdown=max_drawdown,
        peak_timestamp=worst_peak, trough_timestamp=worst_trough, recovery_timestamp=recovery,
        max_drawdown_duration_seconds=(
            ((recovery or points[-1]["timestamp"]) - worst_peak).total_seconds()
            if worst_peak is not None else 0.0
        ),
        recovery_seconds=(recovery - worst_trough).total_seconds() if recovery and worst_trough else None,
        longest_underwater_episode_seconds=max(durations, default=0.0),
        time_underwater_seconds=underwater,
        time_underwater_fraction=underwater / elapsed if elapsed > 0 else None,
        currently_underwater=bool(drawdowns and drawdowns[-1] > 0),
    ), drawdowns


def _risk_metrics(returns, daily, equity, elapsed, funding, final_equity, drawdown, availability, rf):
    daily_values = [row["value"] for row in daily]
    # Prefer daily risk when enough completed sessions exist. Otherwise use
    # observed snapshot returns with an explicit nonannualized horizon label.
    sample = daily_values if len(daily_values) >= _MIN_DAILY else returns
    horizon = "observed_NY_day_end_return" if len(daily_values) >= _MIN_DAILY else "observed_snapshot_interval_return"
    std = _sample_std(sample)
    result = dict(
        return_samples=len(sample), return_horizon=horizon, confidence_level=0.95,
        interval_volatility=std,
        dollar_volatility=_sample_std(np.diff(equity)),
        historical_var_95=None, historical_expected_shortfall_95=None,
        tail_samples=0, skewness=None, excess_kurtosis=None,
        daily_return_samples=len(daily_values), annualization_days=252,
        annualized_volatility=None, sharpe_ratio=None, sortino_ratio=None,
        cagr=None, calmar_ratio=None,
    )
    if len(sample) >= 20:
        losses = -np.asarray(sample, dtype=float)
        quantile = float(np.quantile(losses, 0.95, method="linear"))
        tail = losses[losses >= quantile]
        result.update(historical_var_95=max(0.0, quantile),
                      historical_expected_shortfall_95=max(0.0, float(np.mean(tail))),
                      tail_samples=len(tail))
        centered = np.asarray(sample, dtype=float) - np.mean(sample)
        m2 = float(np.mean(centered ** 2))
        n = len(sample)
        if m2 > 1e-24:
            g1 = float(np.mean(centered ** 3)) / m2 ** 1.5
            g2 = float(np.mean(centered ** 4)) / m2 ** 2 - 3
            result["skewness"] = math.sqrt(n * (n - 1)) / (n - 2) * g1
            result["excess_kurtosis"] = (n - 1) * ((n + 1) * g2 + 6) / ((n - 2) * (n - 3))
        else:
            availability["risk.skewness"] = "Zero return variance"
            availability["risk.excess_kurtosis"] = "Zero return variance"
    else:
        for key in ("historical_var_95", "historical_expected_shortfall_95", "skewness", "excess_kurtosis"):
            availability[f"risk.{key}"] = "Requires at least 20 observed returns; tail estimates are descriptive"
    if std is None:
        availability["risk.interval_volatility"] = "Requires at least two observed returns"
    if result["dollar_volatility"] is None:
        availability["risk.dollar_volatility"] = "Requires at least two observed dollar changes"
    if len(daily_values) >= _MIN_DAILY:
        daily_std = _sample_std(daily_values)
        result["annualized_volatility"] = daily_std * math.sqrt(252)
        daily_rf = math.expm1(math.log1p(rf) / 252)
        excess = np.asarray(daily_values) - daily_rf
        if daily_std > 1e-12:
            result["sharpe_ratio"] = float(np.mean(excess)) / daily_std * math.sqrt(252)
        else:
            availability["risk.sharpe_ratio"] = "Zero daily return variance"
        downside = math.sqrt(float(np.mean(np.minimum(excess, 0) ** 2)))
        if downside > 1e-12:
            result["sortino_ratio"] = float(np.mean(excess)) / downside * math.sqrt(252)
        else:
            availability["risk.sortino_ratio"] = "No below-target daily observations"
    else:
        for key in ("annualized_volatility", "sharpe_ratio", "sortino_ratio"):
            availability[f"risk.{key}"] = "Requires at least 20 completed NY daily-close returns; intraday counts are not trading days"
    if elapsed >= 365 * 86400 and final_equity > 0:
        result["cagr"] = math.expm1(math.log(final_equity / funding) * _YEAR_SECONDS / elapsed)
        if drawdown is not None and drawdown > 0:
            result["calmar_ratio"] = result["cagr"] / drawdown
        else:
            availability["risk.calmar_ratio"] = "Zero observed maximum drawdown"
    else:
        availability["risk.cagr"] = "Requires at least 365 elapsed days and positive ending equity"
        availability["risk.calmar_ratio"] = "Requires an available CAGR and nonzero observed maximum drawdown"
    return result


def _execution_statistics(records, funding, availability):
    statuses = Counter()
    rejected_reasons = Counter()
    matches = []
    fees, slippage, gross_notional = 0.0, 0.0, 0.0
    missing_fees = missing_reference = malformed_matches = 0
    attempts = holds = risk_rejected = 0
    for record in records:
        action = record.get("policy_action", {})
        if action.get("side", "").lower() == "hold":
            holds += 1
        feasibility = record.get("feasibility", {})
        if record.get("transition") == "risk_rejected_no_order" or feasibility.get("accepted") is False:
            risk_rejected += 1
            rejected_reasons.update(feasibility.get("reasons", ()))
        if "intent" not in record:
            continue
        attempts += 1
        intent, analog, accounting = record["intent"], record.get("analog", {}), record.get("accounting", {})
        status = accounting.get("status", analog.get("status", "unknown"))
        statuses[status] += 1
        if status != "matched":
            rejected_reasons.update([accounting.get("reason", analog.get("reason", status))])
            continue
        price, quantity = _number(analog.get("estimated_price")), _number(analog.get("estimated_quantity"))
        side = str(intent.get("side", "")).lower()
        if price is None or price <= 0 or quantity is None or quantity <= 0 or side not in ("buy", "sell"):
            malformed_matches += 1
            continue
        fee = _number(accounting.get("fee"))
        if fee is None:
            missing_fees += 1
        else:
            fees += fee
        notional = price * quantity
        gross_notional += notional
        reference = _number(intent.get("reference_price"))
        if reference is None:
            missing_reference += 1
        else:
            slippage += (1 if side == "buy" else -1) * (price - reference) * quantity
        timestamp = analog.get("event_timestamp") or accounting.get("timestamp") or intent.get("timestamp") or record.get("timestamp")
        matches.append(dict(symbol=intent["symbol"], side=side, price=price, quantity=quantity,
                            fee=fee, timestamp=_timestamp(timestamp)))
    if not attempts:
        availability["execution.matching_coverage"] = "No submitted order intentions"
    if missing_fees:
        availability["execution.total_fees"] = f"Fee missing for {missing_fees} matched executions"
    if missing_reference:
        availability["execution.signed_slippage_cost"] = f"Reference mark missing for {missing_reference} matched executions"
    if malformed_matches:
        availability["execution.gross_notional"] = f"Incomplete matched execution details for {malformed_matches} records"
    result = dict(
        decisions=len(records), submitted_orders=attempts, holds=holds,
        matched_orders=statuses["matched"], unmatched_orders=statuses["unmatched"],
        accounting_rejected_orders=statuses["rejected"], risk_rejected_decisions=risk_rejected,
        other_order_statuses={key: count for key, count in statuses.items() if key not in ("matched", "unmatched", "rejected")},
        matching_coverage=statuses["matched"] / attempts if attempts else None,
        rejection_reasons=dict(rejected_reasons),
        gross_notional=gross_notional if not malformed_matches else None,
        turnover=gross_notional / funding if not malformed_matches else None,
        total_fees=fees if not missing_fees else None,
        known_fees=fees,
        signed_slippage_cost=slippage if not missing_reference else None,
        attributed_cost=fees + slippage if not missing_fees and not missing_reference else None,
        incomplete_execution=bool(statuses["unmatched"] or statuses["rejected"] or malformed_matches),
    )
    return result, sorted(matches, key=lambda match: match["timestamp"])


def _trade_statistics(matches, availability):
    lots = {}
    closed = []
    unsupported_sales = 0.0
    for match in matches:
        symbol = match["symbol"]
        queue = lots.setdefault(symbol, deque())
        if match["side"] == "buy":
            queue.append(dict(quantity=match["quantity"], price=match["price"],
                              fee_per_share=match["fee"] / match["quantity"] if match["fee"] is not None else None,
                              timestamp=match["timestamp"]))
            continue
        remaining = match["quantity"]
        exit_fee = match["fee"] / match["quantity"] if match["fee"] is not None else None
        while remaining > 1e-10 and queue:
            lot = queue[0]
            quantity = min(remaining, lot["quantity"])
            fee_known = lot["fee_per_share"] is not None and exit_fee is not None
            basis = quantity * (lot["price"] + lot["fee_per_share"]) if fee_known else None
            pnl = quantity * (match["price"] - lot["price"] - lot["fee_per_share"] - exit_fee) if fee_known else None
            closed.append(dict(symbol=symbol, quantity=quantity, entry_timestamp=lot["timestamp"],
                               exit_timestamp=match["timestamp"], entry_price=lot["price"],
                               exit_price=match["price"], realized_net_pnl=pnl,
                               return_on_entry_cost=pnl / basis if basis is not None and basis > 0 else None,
                               holding_seconds=(match["timestamp"] - lot["timestamp"]).total_seconds()))
            lot["quantity"] -= quantity
            remaining -= quantity
            if lot["quantity"] < 1e-10:
                queue.popleft()
        unsupported_sales += remaining
    pnl = [lot["realized_net_pnl"] for lot in closed if lot["realized_net_pnl"] is not None]
    wins, losses = [value for value in pnl if value > 0], [value for value in pnl if value < 0]
    result = dict(
        convention="FIFO matched closed-lot fragments; open holdings are not liquidated",
        closed_lot_matches=len(closed), priced_closed_lot_matches=len(pnl),
        realized_net_pnl=sum(pnl) if len(pnl) == len(closed) else None,
        win_rate=len(wins) / len(pnl) if pnl else None,
        profit_factor=sum(wins) / -sum(losses) if losses else None,
        payoff_ratio=float(np.mean(wins)) / -float(np.mean(losses)) if wins and losses else None,
        mean_closed_lot_pnl=float(np.mean(pnl)) if pnl else None,
        median_holding_seconds=float(np.median([lot["holding_seconds"] for lot in closed])) if closed else None,
        unsupported_opening_inventory_sales=unsupported_sales,
        open_quantities={symbol: sum(lot["quantity"] for lot in queue) for symbol, queue in lots.items() if queue},
        closed_lots=closed,
    )
    if not pnl:
        for key in ("win_rate", "profit_factor", "payoff_ratio", "mean_closed_lot_pnl"):
            availability[f"trades.{key}"] = "No matched closed FIFO lots with known entry/exit fees"
    else:
        if not losses:
            availability["trades.profit_factor"] = "No realized losing FIFO lots; an infinite ratio is not reported"
        if not wins or not losses:
            availability["trades.payoff_ratio"] = "Requires both winning and losing closed FIFO lots"
    if unsupported_sales:
        availability["trades.realized_net_pnl"] = "Some sales have no observed opening lot; their realized result is unknown"
        result["realized_net_pnl"] = None
    return result


def _rolling(values, period, statistic="mean"):
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    # O(T) prefix sums, with centering to reduce cancellation. Finite counts
    # ensure a missing observation never becomes an invented zero value.
    finite = np.isfinite(values)
    origin = float(values[np.flatnonzero(finite)[0]]) if finite.any() else 0.0
    centered = np.where(finite, values - origin, 0)
    count = np.concatenate(([0], np.cumsum(finite)))
    total = np.concatenate(([0.0], np.cumsum(centered)))
    valid = (count[period:] - count[:-period]) == period
    sums = total[period:] - total[:-period]
    means = sums / period
    if statistic == "mean":
        computed = means + origin
    else:
        squared = np.concatenate(([0.0], np.cumsum(centered ** 2)))
        numerator = (squared[period:] - squared[:-period]) - sums ** 2 / period
        computed = np.sqrt(np.maximum(numerator, 0) / (period - (statistic == "sample_std")))
    result[period - 1:] = np.where(valid, computed, np.nan)
    return result


def _ema(values, period):
    result = np.full(len(values), np.nan)
    previous = None
    for index in range(len(values)):
        if not math.isfinite(float(values[index])):
            previous = None
        elif previous is None:
            if index + 1 >= period and np.isfinite(values[index + 1 - period:index + 1]).all():
                previous = float(np.mean(values[index + 1 - period:index + 1]))
        else:
            previous += 2 / (period + 1) * (float(values[index]) - previous)
        if previous is not None:
            result[index] = previous
    return result


def _technical_indicators(series, start_time, end_time, availability, symbol):
    selected = [bar for bar in series if _timestamp(bar.timestamp) <= end_time]
    selected.sort(key=lambda bar: _timestamp(bar.timestamp))
    count = len(selected)
    close = np.array([bar.close for bar in selected], dtype=float)
    high = np.array([bar.high for bar in selected], dtype=float)
    low = np.array([bar.low for bar in selected], dtype=float)
    volume = np.array([bar.volume for bar in selected], dtype=float)
    timestamps = [_timestamp(bar.timestamp) for bar in selected]
    simple = np.full(count, np.nan)
    if count > 1:
        simple[1:] = np.divide(close[1:], close[:-1], out=np.full(count - 1, np.nan), where=close[:-1] > 0) - 1
    sma = {period: _rolling(close, period) for period in (20, 50, 200)}
    band_std = _rolling(close, 20, "std")
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    rsi, atr, obv = np.full(count, np.nan), np.full(count, np.nan), np.zeros(count)
    true_range = high - low
    if count > 1:
        true_range[1:] = np.maximum.reduce((high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    gain = loss = None
    for index in range(count):
        if index:
            obv[index] = obv[index - 1] + np.sign(close[index] - close[index - 1]) * volume[index]
        if index == 13:
            atr[index] = float(np.mean(true_range[:14]))
        elif index > 13:
            atr[index] = (atr[index - 1] * 13 + true_range[index]) / 14
        if index == 14:
            changes = np.diff(close[:15])
            gain, loss = float(np.mean(np.maximum(changes, 0))), float(np.mean(np.maximum(-changes, 0)))
        elif index > 14:
            change = close[index] - close[index - 1]
            gain, loss = (gain * 13 + max(change, 0)) / 14, (loss * 13 + max(-change, 0)) / 14
        if gain is not None:
            rsi[index] = 100 - 100 / (1 + gain / loss) if loss > 0 else (100 if gain > 0 else 50)
    realized = _rolling(simple, 20, "sample_std")
    rows = []
    for index, timestamp in enumerate(timestamps):
        if timestamp < start_time:
            continue
        rows.append(dict(timestamp=timestamp, close=close[index], volume=volume[index],
                         sma_20=sma[20][index], sma_50=sma[50][index], sma_200=sma[200][index],
                         rsi_14=rsi[index], macd=macd[index], macd_signal=signal[index],
                         macd_histogram=macd[index] - signal[index],
                         bollinger_middle=sma[20][index], bollinger_upper=sma[20][index] + 2 * band_std[index],
                         bollinger_lower=sma[20][index] - 2 * band_std[index],
                         atr_14=atr[index], obv=obv[index], realized_volatility_20=realized[index]))
    observed = [bar for bar in selected if _timestamp(bar.timestamp) >= start_time]
    gaps = np.diff([_timestamp(bar.timestamp).timestamp() for bar in observed])
    positive_gaps = gaps[gaps > 0]
    median_gap = float(np.median(positive_gaps)) if len(positive_gaps) else None
    spreads = [(bar.ask - bar.bid) / bar.close * 10000 for bar in observed
               if bar.ask is not None and bar.bid is not None and bar.close > 0 and 0 < bar.bid <= bar.ask]
    dollar_volume = [bar.close * bar.volume for bar in observed if bar.close > 0 and bar.volume >= 0]
    illiquidity = []
    opening_gaps = []
    for index, bar in enumerate(selected):
        if index and _timestamp(bar.timestamp) >= start_time and selected[index - 1].close > 0:
            opening_gaps.append(bar.open / selected[index - 1].close - 1)
            if bar.volume > 0 and bar.close > 0:
                illiquidity.append(abs(bar.close / selected[index - 1].close - 1) / (bar.close * bar.volume))
    latest = {key: value for key, value in rows[-1].items() if key not in ("timestamp", "close", "volume")} if rows else {}
    for key, value in latest.items():
        if not math.isfinite(value):
            availability[f"indicators.{symbol}.{key}"] = "Insufficient preceding finite bars for this indicator's full warmup"
    return dict(
        latest=latest, series=rows,
        diagnostics=dict(evaluation_bars=len(observed), history_bars=count,
                         median_interval_seconds=median_gap,
                         maximum_interval_seconds=float(max(positive_gaps)) if len(positive_gaps) else None,
                         gaps_over_five_median_intervals=int(np.sum(gaps > 5 * median_gap)) if median_gap else 0,
                         nonincreasing_timestamps=int(np.sum(gaps <= 0)),
                         zero_volume_fraction=sum(bar.volume == 0 for bar in observed) / len(observed) if observed else None,
                         quote_coverage=len(spreads) / len(observed) if observed else None,
                         median_spread_bps=float(np.median(spreads)) if spreads else None,
                         median_dollar_volume=float(np.median(dollar_volume)) if dollar_volume else None,
                         mean_absolute_open_gap=float(np.mean(np.abs(opening_gaps))) if opening_gaps else None,
                         amihud_abs_return_per_dollar=float(np.mean(illiquidity)) if illiquidity else None),
    )


def _benchmark(bars, symbols, requested, start_time, end_time, points, strategy_daily, availability, rf):
    source_symbols = [requested] if requested else list(symbols)
    result = dict(
        kind="supplied_symbol_raw_price_index" if requested else "equally_initial_weighted_universe_raw_price_index",
        symbol=requested, source_symbols=source_symbols, executable=False,
        dividends_included=False, transaction_costs_included=False,
        price_return=None, excess_period_return=None,
        daily_aligned_samples=0, beta=None, correlation=None, annualized_alpha=None,
        annualized_tracking_error=None, information_ratio=None, series=[],
    )
    if not source_symbols or any(symbol not in bars for symbol in source_symbols):
        availability["benchmark"] = "Requested benchmark prices were not supplied; no substitute ticker is fabricated"
        return result
    prices = {symbol: {_timestamp(bar.timestamp): bar.close for bar in bars[symbol]
                       if start_time <= _timestamp(bar.timestamp) <= end_time and bar.close > 0}
              for symbol in source_symbols}
    common = set.intersection(*(set(prices[symbol]) for symbol in source_symbols))
    if start_time not in common or end_time not in common:
        availability["benchmark"] = "Exact evaluation start/end benchmark observations are missing; no forward fill"
        return result
    normalized = [(timestamp, float(np.mean([prices[symbol][timestamp] / prices[symbol][start_time]
                                             for symbol in source_symbols]))) for timestamp in sorted(common)]
    result["series"] = [dict(timestamp=timestamp, normalized_price_index=value) for timestamp, value in normalized]
    result["price_return"] = normalized[-1][1] - 1
    if points:
        result["excess_period_return"] = points[-1]["normalized_equity"] - normalized[-1][1]
    benchmark_daily, _, _, _ = _daily_returns(normalized)
    lookup = {(row["start_date"], row["end_date"]): row["value"] for row in benchmark_daily}
    paired = [(row["value"], lookup[(row["start_date"], row["end_date"])]) for row in strategy_daily
              if (row["start_date"], row["end_date"]) in lookup]
    result["daily_aligned_samples"] = len(paired)
    if len(paired) < _MIN_DAILY:
        for key in ("beta", "correlation", "annualized_alpha", "annualized_tracking_error", "information_ratio"):
            availability[f"benchmark.{key}"] = "Requires at least 20 exactly aligned completed NY daily-return pairs"
        return result
    strategy, reference = np.array(paired, dtype=float).T
    strategy_std, reference_std = _sample_std(strategy), _sample_std(reference)
    if reference_std > 1e-12:
        result["beta"] = float(np.cov(strategy, reference, ddof=1)[0, 1]) / reference_std ** 2
        daily_rf = math.expm1(math.log1p(rf) / 252)
        result["annualized_alpha"] = 252 * (float(np.mean(strategy - daily_rf)) - result["beta"] * float(np.mean(reference - daily_rf)))
        if strategy_std > 1e-12:
            result["correlation"] = float(np.corrcoef(strategy, reference)[0, 1])
        else:
            availability["benchmark.correlation"] = "Zero strategy daily variance"
    else:
        for key in ("beta", "correlation", "annualized_alpha"):
            availability[f"benchmark.{key}"] = "Zero benchmark daily variance"
    active = strategy - reference
    active_std = _sample_std(active)
    result["annualized_tracking_error"] = active_std * math.sqrt(252)
    if active_std > 1e-12:
        result["information_ratio"] = float(np.mean(active)) / active_std * math.sqrt(252)
    else:
        availability["benchmark.information_ratio"] = "Zero active-return variance"
    return result


def build_performance_report(
    *, bars: dict[str, list[MarketBar]], records: list[dict], equity_curve: list[dict],
    initial_funding: float, start: int, stop: int,
    benchmark_symbol: str | None = None, risk_free_rate: float = 0.0,
    strategy_symbols: Sequence[str] | None = None,
) -> dict:
    """Describe a completed evaluation, without model updates or invented fills.

    ``start:stop`` selects the evaluation's half-open bar range on the first
    strategy symbol. Technical warmup uses only earlier supplied bars. All
    subsequent series are filtered to that exact timestamp interval. Additional
    bars may supply an explicitly named reference outside the strategy universe.
    """
    funding = _number(initial_funding)
    if funding is None or funding <= 0:
        raise ValueError("initial_funding must be positive and finite")
    if not isinstance(start, int) or not isinstance(stop, int) or start < 0 or stop <= start:
        raise ValueError("Require nonnegative half-open start < stop bar indices")
    rf = _number(risk_free_rate)
    if rf is None or rf <= -1:
        raise ValueError("Annual risk-free rate must be finite and greater than -1")
    symbols = tuple(strategy_symbols) if strategy_symbols is not None else tuple(symbol for symbol in bars if symbol != benchmark_symbol)
    if not symbols:
        symbols = tuple(bars)
    if not symbols or symbols[0] not in bars or stop > len(bars[symbols[0]]):
        raise ValueError("Strategy bars must contain the requested evaluation interval")
    reference = bars[symbols[0]]
    start_time, end_time = _timestamp(reference[start].timestamp), _timestamp(reference[stop - 1].timestamp)
    if end_time < start_time:
        raise ValueError("Evaluation timestamps must be chronological")
    availability = {}
    snapshots = {}
    dropped = 0
    for row in equity_curve:
        timestamp = _timestamp(row["timestamp"])
        equity, gross = _number(row.get("equity")), _number(row.get("gross_exposure"))
        if start_time <= timestamp <= end_time:
            if equity is None or equity < 0 or gross is None or gross < 0:
                dropped += 1
                continue
            snapshots[timestamp] = dict(timestamp=timestamp, equity=equity, gross_exposure=gross)
    points = [snapshots[timestamp] for timestamp in sorted(snapshots)]
    if not points:
        availability["performance"] = "No finite completed portfolio snapshots; no performance path is fabricated"
    if dropped:
        availability["performance.snapshot_quality"] = f"Excluded {dropped} nonfinite or negative snapshots"
    drawdown, drawdowns = _drawdowns(points, funding)
    interval_returns = []
    for index, point in enumerate(points):
        point.update(normalized_equity=point["equity"] / funding,
                     net_return=point["equity"] / funding - 1,
                     cash=point["equity"] - point["gross_exposure"], drawdown=-drawdowns[index])
        if index and points[index - 1]["equity"] > 0:
            interval_returns.append(point["equity"] / points[index - 1]["equity"] - 1)
    final_equity = points[-1]["equity"] if points else None
    elapsed = (end_time - start_time).total_seconds()
    daily, closes, incomplete, gaps = _daily_returns([(point["timestamp"], point["equity"]) for point in points])
    if gaps:
        availability["performance.daily_returns"] = "Unverified missing weekdays break daily-return adjacency; holidays/early closes require an exchange calendar"
    performance = dict(
        initial_funding=funding, final_equity=final_equity,
        net_pnl=final_equity - funding if final_equity is not None else None,
        net_return=final_equity / funding - 1 if final_equity is not None else None,
        start_timestamp=start_time, end_timestamp=end_time, elapsed_seconds=elapsed,
        completed_snapshots=len(points), excluded_invalid_snapshots=dropped,
        series=points, daily_returns=daily, daily_closes=closes,
        incomplete_NY_dates=incomplete,
        unverified_weekday_gaps=gaps,
        drawdown=drawdown,
    )
    risk = _risk_metrics(interval_returns, daily, [point["equity"] for point in points], elapsed,
                         funding, final_equity or 0, drawdown["max_drawdown"], availability, rf)
    snapshot_intervals = np.diff([point["timestamp"].timestamp() for point in points])
    risk["snapshot_interval_seconds"] = dict(
        minimum=float(np.min(snapshot_intervals)) if len(snapshot_intervals) else None,
        median=float(np.median(snapshot_intervals)) if len(snapshot_intervals) else None,
        maximum=float(np.max(snapshot_intervals)) if len(snapshot_intervals) else None,
    )
    risk["irregular_snapshot_intervals"] = bool(len(snapshot_intervals) > 1 and not np.allclose(snapshot_intervals, snapshot_intervals[0], rtol=1e-6, atol=1e-6))
    if risk["irregular_snapshot_intervals"] and risk["return_horizon"] == "observed_snapshot_interval_return":
        availability["risk.return_horizon"] = "Snapshot intervals are irregular; these descriptive return/tail metrics do not represent a fixed holding period"
    execution, matches = _execution_statistics(records, funding, availability)
    trades = _trade_statistics(matches, availability)
    holdings = {}
    if records:
        holdings.update(records[0].get("account_before", {}).get("holdings", {}))
    for match in matches:
        holdings[match["symbol"]] = holdings.get(match["symbol"], 0) + (1 if match["side"] == "buy" else -1) * match["quantity"]
    unknown_positions = [symbol for symbol, quantity in holdings.items() if quantity < 0]
    if unknown_positions:
        availability["portfolio.final_holdings"] = "Matched sales exceed observed opening inventory; affected positions cannot be reconstructed"
    final_marks = {}
    for symbol, series in bars.items():
        history = [bar for bar in series if _timestamp(bar.timestamp) <= end_time]
        if history:
            final_marks[symbol] = history[-1].close
    notionals = {symbol: quantity * final_marks[symbol] for symbol, quantity in holdings.items()
                 if quantity > 0 and symbol in final_marks}
    known_gross = sum(notionals.values())
    weights = {symbol: notional / final_equity for symbol, notional in notionals.items()} if final_equity and final_equity > 0 else {}
    intervals = [(points[index + 1]["timestamp"] - point["timestamp"]).total_seconds()
                 for index, point in enumerate(points[:-1])]
    weights_time = sum(intervals)
    exposure_ratios = [point["gross_exposure"] / point["equity"] if point["equity"] > 0 else None for point in points]
    valid_weighted = [(ratio, duration) for ratio, duration in zip(exposure_ratios, intervals) if ratio is not None]
    valid_duration = sum(duration for _, duration in valid_weighted)
    portfolio = dict(
        final_cash=points[-1]["cash"] if points else None,
        final_gross_exposure=points[-1]["gross_exposure"] if points else None,
        final_gross_exposure_fraction=exposure_ratios[-1] if points else None,
        final_cash_fraction=1 - exposure_ratios[-1] if points and exposure_ratios[-1] is not None else None,
        time_weighted_gross_exposure_fraction=sum(ratio * duration for ratio, duration in valid_weighted) / valid_duration if valid_duration > 0 else None,
        final_holdings={symbol: quantity for symbol, quantity in holdings.items() if quantity > 0},
        unknown_position_symbols=unknown_positions,
        final_symbol_notionals=notionals, final_symbol_equity_weights=weights,
        largest_symbol_equity_weight=max(weights.values(), default=0.0) if final_equity else None,
        invested_hhi=sum((value / known_gross) ** 2 for value in notionals.values()) if known_gross > 0 else None,
        final_holdings_reconcile_to_snapshot=bool(points and math.isclose(known_gross, points[-1]["gross_exposure"], rel_tol=1e-8, abs_tol=1e-8)),
    )
    if not weights_time:
        availability["portfolio.time_weighted_gross_exposure_fraction"] = "Requires positive elapsed snapshot intervals"
    if not known_gross:
        availability["portfolio.invested_hhi"] = "No observed marked long holdings"
    indicators = {symbol: _technical_indicators(series, start_time, end_time, availability, symbol) for symbol, series in bars.items()}
    benchmark = _benchmark(bars, symbols, benchmark_symbol, start_time, end_time, points, daily, availability, rf)
    return _clean(dict(
        performance=performance, risk=risk, execution=execution, trades=trades,
        portfolio=portfolio, benchmark=benchmark, indicators=indicators,
        availability=availability,
        methodology=dict(
            evaluation_only=True, training_feedback=False, checkpoint_selection=False,
            trading_day_timezone="America/New_York", serialized_timezone="UTC",
            daily_close="Observed NY-day-end mark, available only at or after 16:00 local; may include after-hours marks, not an exact exchange closing tick; first day is baseline",
            annualization="252 observed trading days; >=20 complete daily-return observations; no intraday annualization",
            cagr_minimum_elapsed_days=365, risk_free_annual_rate=rf,
            snapshot_policy="Duplicate UTC timestamps collapse to the last completed portfolio snapshot",
            terminal_valuation="Open holdings remain marked; no terminal liquidation is invented",
            benchmark="Raw-price normalized index, no executable fills, dividends, corporate-action adjustments or costs inferred",
            cost_attribution="Fees are ledger charges; signed execution-versus-reference slippage is attribution, not an additional deduction",
            limitations=["Descriptive finite-sample metrics do not establish future risk or profitability",
                         "Observed-execution analog coverage and selection bias remain; unmatched actions have unknown outcomes",
                         "No exchange holiday/early-close calendar is inferred; missing dates and partial sessions may reduce availability",
                         "Raw-price splits/dividends can distort returns, indicators and benchmark comparisons",
                         "No deposits/withdrawals are inferred; net return assumes only initial funding",
                         "This is a selected indicator set, not all possible technical indicators"],
        ),
        definitions=dict(
            net_return="final_equity / initial_funding - 1, including observed fees and marked open positions",
            max_drawdown="Largest peak-to-trough equity loss as a nonnegative fraction of the running peak including initial funding",
            dollar_volatility="Sample standard deviation of consecutive completed-snapshot equity changes; not annualized",
            historical_var_95="Nonnegative 95th percentile of observed loss returns, linear quantile; descriptive horizon is reported",
            historical_expected_shortfall_95="Nonnegative mean observed loss at/above the raw 95th percentile; finite tail count reported",
            turnover="Sum of actual matched buy and sell notionals / initial funding; two-sided",
            win_rate="Positive net realized FIFO closed-lot fragments / fragments with known entry and exit fees",
            sortino_ratio="Annualized mean daily excess return divided by root-mean-square negative daily excess return",
            invested_hhi="Sum of squared symbol shares of observed invested gross notional; cash excluded",
            realized_volatility_20="Sample standard deviation of 20 simple close-to-close bar returns; not annualized",
            obv="On-balance volume starts at zero at the first supplied bar",
            rsi_14="Wilder 14-change RSI; flat zero-gain/zero-loss window is defined as 50",
        ),
    ))
