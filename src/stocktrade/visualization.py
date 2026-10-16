"""Offline, interactive diagnostics for an observed-execution evaluation.

This module reads evaluation outputs; it never invokes a model, changes a
checkpoint, supplies an execution, or turns a risk estimate into a reward.
"""

from __future__ import annotations

import bisect
import html
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

_COLORS = {"matched": "#168453", "unmatched": "#d89216", "rejected": "#7a8497",
           "proposed": "#336ac0", "hold": "#7963b5"}


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _date(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _value(value):
    return "Unavailable" if value is None else str(value)


def _money(value):
    value = _number(value)
    return "Unavailable" if value is None else f"${value:,.2f}"


def _percent(value):
    value = _number(value)
    return "Unavailable" if value is None else f"{value:.2%}"


def _safe_json(value):
    # JSON in an HTML script element needs HTML-safe escaping in addition to
    # JSON quoting. textContent is used for all subsequent table rendering.
    return (json.dumps(value, cls=PlotlyJSONEncoder, allow_nan=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _status(record):
    action = _mapping(record.get("policy_action"))
    if action.get("side") == "hold":
        return "hold"
    transition = record.get("transition")
    accounting = _mapping(record.get("accounting"))
    if transition == "risk_rejected_no_order" or _mapping(record.get("feasibility")).get("accepted") is False:
        return "rejected"
    status = accounting.get("status") or transition
    return status if status in {"matched", "unmatched", "rejected"} else "proposed"


def _row(record):
    action = _mapping(record.get("policy_action"))
    feasibility = _mapping(record.get("feasibility"))
    risk = _mapping(record.get("risk_estimate"))
    before, projected, delta = (_mapping(risk.get(name)) for name in ("before", "projected", "delta"))
    proposal = _mapping(risk.get("proposal"))
    estimates = _mapping(proposal.get("estimates"))
    accounting = _mapping(record.get("accounting"))
    analog = _mapping(record.get("analog"))
    reasons = feasibility.get("reasons") or accounting.get("reason") or analog.get("reason") or ""
    if isinstance(reasons, (list, tuple)):
        reasons = "; ".join(str(reason) for reason in reasons)
    costs = [_number(estimates.get(key)) for key in
             ("commission_cost", "half_spread_cost", "configured_slippage_cost")]
    conditional_cost = sum(costs) if all(value is not None for value in costs) else None
    matched = _status(record) == "matched"
    return {
        "time": _date(record.get("timestamp")).isoformat(),
        "symbol": action.get("symbol") or "Portfolio", "action": str(action.get("side", "unknown")).upper(),
        "requested_value": _money(action.get("value")), "quantity": _value(feasibility.get("quantity")),
        "status": _status(record), "reason": str(reasons),
        "var_before": _money(before.get("value_at_risk")),
        "var_projected": _money(projected.get("value_at_risk")), "var_delta": _money(delta.get("value_at_risk")),
        "conditional_cost": _money(conditional_cost),
        "spread_bps": _value(_number(proposal.get("spread_bps"))),
        "volume_participation": _percent(proposal.get("volume_participation")),
        "observed_fee": _money(accounting.get("fee")) if matched else "Unavailable",
        "source_ids": ", ".join(str(value) for value in analog.get("source_trade_ids", ())) if matched else "",
    }


def _hover(row):
    pairs = (("Time", row["time"]), ("Action", f"{row['action']} {row['symbol']}"),
             ("Proposal status", row["status"]), ("Requested", row["requested_value"]),
             ("Feasible quantity", row["quantity"]), ("Reason", row["reason"]),
             ("VaR before / projected", f"{row['var_before']} / {row['var_projected']}"),
             ("VaR change", row["var_delta"]), ("Conditional cost estimate", row["conditional_cost"]),
             ("Observed quote spread (bps)", row["spread_bps"]),
             ("Proposal / observed volume", row["volume_participation"]),
             ("Observed fee", row["observed_fee"]), ("Observed source IDs", row["source_ids"]))
    return "<br>".join(f"{label}: {html.escape(str(value))}" for label, value in pairs)


def _style(figure, *, height=430):
    figure.update_layout(template="plotly_white", height=height, margin=dict(l=65, r=25, t=55, b=50),
                         paper_bgcolor="white", font=dict(family="Arial, sans-serif", color="#26334a"),
                         hovermode="closest", legend=dict(orientation="h", y=-.13, x=0))
    return figure


def _indicator_rows(report, bars):
    analytics = _mapping(_mapping(report.get("diagnostics")).get("performance"))
    canonical = _mapping(analytics.get("indicators"))
    result = {}
    for symbol, stock in bars.items():
        if not stock:
            result[symbol] = []
            continue
        first, last = _date(stock[0].timestamp), _date(stock[-1].timestamp)
        rows = []
        for item in _mapping(canonical.get(symbol)).get("series", ()):
            item = _mapping(item)
            timestamp = _date(item.get("timestamp"))
            if timestamp is not None and first <= timestamp <= last:
                rows.append({**item, "timestamp": timestamp.isoformat()})
        result[symbol] = sorted(rows, key=lambda item: item["timestamp"])
    return result


def _market_figure(bars, records, indicators):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.065,
                           row_heights=[.60, .17, .23], subplot_titles=("Observed OHLC prices", "Observed volume",
                                                                      "Every policy decision (proposals)"))
    symbols = sorted(bars)
    owners = []

    def add(trace, row, owner):
        figure.add_trace(trace, row=row, col=1)
        owners.append(owner)

    for symbol, stock in sorted(bars.items()):
        times = [bar.timestamp.astimezone(timezone.utc) for bar in stock]
        x = [timestamp.isoformat() for timestamp in times]
        label = html.escape(symbol)
        add(go.Candlestick(x=x, open=[bar.open for bar in stock], high=[bar.high for bar in stock],
                          low=[bar.low for bar in stock], close=[bar.close for bar in stock],
                          name=f"{label} OHLC", increasing_line_color="#168453", decreasing_line_color="#c34d58"),
            1, symbol)
        add(go.Bar(x=x, y=[bar.volume for bar in stock], name=f"{label} volume", marker_color="#b4c5e0",
                   showlegend=False, hovertemplate="%{x}<br>Observed volume: %{y:,.0f}<extra></extra>"), 2, symbol)
        technical = indicators.get(symbol, [])
        for key, name, color, dash in (
            ("sma_20", "SMA 20", "#336ac0", "solid"), ("sma_50", "SMA 50", "#7963b5", "solid"),
            ("sma_200", "SMA 200", "#d89216", "solid"),
            ("bollinger_upper", "Bollinger upper", "#9aa8ba", "dot"),
            ("bollinger_middle", "Bollinger middle", "#9aa8ba", "dot"),
            ("bollinger_lower", "Bollinger lower", "#9aa8ba", "dot"),
        ):
            if technical:
                add(go.Scatter(x=[item["timestamp"] for item in technical],
                               y=[_number(item.get(key)) for item in technical], name=name, mode="lines",
                               connectgaps=False, line=dict(color=color, dash=dash, width=1.4),
                               hovertemplate=f"%{{x}}<br>{name}: $%{{y:,.3f}}<extra></extra>"), 1, symbol)
        selected = [record for record in records if _mapping(record.get("policy_action")).get("symbol") == symbol]
        for side in ("buy", "sell", "hold"):
            for status in _COLORS:
                matching = [record for record in selected if _mapping(record.get("policy_action")).get("side") == side
                            and _status(record) == status]
                if not matching:
                    continue
                timestamps = [_date(record["timestamp"]) for record in matching]
                texts = [_hover(_row(record)) for record in matching]
                marker = dict(color=_COLORS[status], size=10, line=dict(width=1, color="white"),
                              symbol={"buy": "triangle-up", "sell": "triangle-down", "hold": "circle"}[side])
                name = f"{side.upper()} proposal · {status}"
                positions = [bisect.bisect_right(times, timestamp) - 1 for timestamp in timestamps]
                valid = [index for index, position in enumerate(positions) if position >= 0]
                add(go.Scatter(x=[timestamps[index].isoformat() for index in valid],
                               y=[stock[positions[index]].close for index in valid], text=[texts[index] for index in valid],
                               mode="markers", marker=marker, name=name, legendgroup=f"{side}-{status}",
                               hovertemplate="%{text}<extra>Decision at observed bar price</extra>"), 1, symbol)
                add(go.Scatter(x=[timestamp.isoformat() for timestamp in timestamps], y=[side] * len(matching),
                               text=texts, mode="markers", marker=marker, name=name, showlegend=False,
                               legendgroup=f"{side}-{status}", hovertemplate="%{text}<extra>Proposal</extra>"), 3, symbol)
        observed = []
        for record in selected:
            analog = _mapping(record.get("analog"))
            timestamp = _date(analog.get("event_timestamp"))
            price = _number(analog.get("estimated_price"))
            quantity = _number(analog.get("estimated_quantity"))
            if (_status(record) == "matched" and timestamp is not None and price is not None and price > 0
                    and quantity is not None and quantity > 0 and analog.get("source_trade_ids")
                    and times and times[0] <= timestamp <= times[-1]):
                observed.append((timestamp, price, quantity, record))
        if observed:
            add(go.Scatter(x=[item[0].isoformat() for item in observed], y=[item[1] for item in observed],
                           mode="markers", marker=dict(symbol="star-open", size=16, color="#172a49", line_width=2),
                           name="Observed matched execution", text=[_hover(_row(item[3])) +
                           f"<br>Observed price: {_money(item[1])}<br>Observed shares: {item[2]:g}" for item in observed],
                           hovertemplate="%{text}<extra>Recorded analog; not causal impact</extra>"), 1, symbol)
    portfolio = [record for record in records if _mapping(record.get("policy_action")).get("symbol") not in bars]
    if portfolio:
        add(go.Scatter(x=[_date(record["timestamp"]).isoformat() for record in portfolio],
                       y=[str(_mapping(record.get("policy_action")).get("side", "unknown")) for record in portfolio],
                       text=[_hover(_row(record)) for record in portfolio], mode="markers", name="Portfolio HOLD / other",
                       marker=dict(symbol="circle", size=9, color=_COLORS["hold"]),
                       hovertemplate="%{text}<extra>Portfolio decision</extra>"), 3, None)
    if symbols:
        for trace, owner in zip(figure.data, owners):
            trace.visible = owner in (None, symbols[0])
        figure.update_layout(updatemenus=[dict(buttons=[dict(label=html.escape(symbol), method="update",
            args=[{"visible": [owner in (None, symbol) for owner in owners]}]) for symbol in symbols],
            direction="down", x=1, xanchor="right", y=1.11, yanchor="top")])
    else:
        figure.add_annotation(text="No market bars in the declared interval", showarrow=False)
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_yaxes(title_text="USD / share", row=1, col=1)
    figure.update_yaxes(title_text="Shares", row=2, col=1)
    figure.update_yaxes(categoryorder="array", categoryarray=["sell", "hold", "buy"], row=3, col=1)
    return _style(figure, height=690)


def _technical_figure(indicators):
    figure = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=.055, subplot_titles=(
        "RSI 14", "MACD (12, 26), signal (9) and histogram", "Average true range 14",
        "On-balance volume", "Realized volatility 20 (per bar, not annualized)"))
    owners = []
    any_values = False
    definitions = (
        (1, "rsi_14", "RSI 14", "#7963b5", "solid"),
        (2, "macd", "MACD", "#336ac0", "solid"),
        (2, "macd_signal", "MACD signal", "#d89216", "solid"),
        (2, "macd_histogram", "MACD histogram", "#b4c5e0", "bar"),
        (3, "atr_14", "ATR 14", "#c34d58", "solid"),
        (4, "obv", "OBV", "#168453", "solid"),
        (5, "realized_volatility_20", "Realized volatility 20", "#172a49", "solid"),
    )
    for symbol, rows in sorted(indicators.items()):
        times = [row["timestamp"] for row in rows]
        for panel, key, name, color, style in definitions:
            values = [_number(row.get(key)) for row in rows]
            any_values |= any(value is not None for value in values)
            hover = f"%{{x}}<br>{name}: " + ("%{y:.3%}" if panel == 5 else "%{y:,.4f}") + "<extra></extra>"
            if style == "bar":
                trace = go.Bar(x=times, y=values, name=name, marker_color=color, hovertemplate=hover)
            else:
                trace = go.Scatter(x=times, y=values, name=name, mode="lines", connectgaps=False,
                                   line=dict(color=color), hovertemplate=hover)
            figure.add_trace(trace, row=panel, col=1)
            owners.append(symbol)
    symbols = sorted(indicators)
    if symbols:
        for trace, owner in zip(figure.data, owners):
            trace.visible = owner == symbols[0]
        figure.update_layout(updatemenus=[dict(buttons=[dict(label=html.escape(symbol), method="update",
            args=[{"visible": [owner == symbol for owner in owners]}]) for symbol in symbols],
            direction="down", x=1, xanchor="right", y=1.07, yanchor="top")])
    if not any_values:
        figure.add_annotation(text="Technical indicator history unavailable; missing warmup values remain blank",
                              showarrow=False, x=.5, y=1.055, xref="paper", yref="paper")
    figure.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1)
    figure.update_yaxes(title_text="USD", row=2, col=1)
    figure.update_yaxes(title_text="USD", row=3, col=1)
    figure.update_yaxes(title_text="Signed shares", row=4, col=1)
    figure.update_yaxes(title_text="Per-bar return", tickformat=".2%", row=5, col=1)
    return _style(figure, height=950)


def _equity_points(report, first, last):
    points = {}
    for point in _mapping(report.get("execution")).get("equity_curve", ()):
        point = _mapping(point)
        timestamp = _date(point.get("timestamp"))
        equity, gross = _number(point.get("equity")), _number(point.get("gross_exposure"))
        if timestamp is not None and equity is not None and (first is None or first <= timestamp <= last):
            points[timestamp] = (equity, gross)
    return [(timestamp, *points[timestamp]) for timestamp in sorted(points)]


def _portfolio_figure(points, funding):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07,
                           subplot_titles=("Observed-analog ledger equity", "Drawdown", "Gross exposure and cash"))
    times = [point[0].isoformat() for point in points]
    equity = [point[1] for point in points]
    drawdown, peak = [], funding or (equity[0] if equity else 0)
    for value in equity:
        peak = max(peak, value)
        drawdown.append(100 * (peak - value) / peak if peak > 0 else None)
    figure.add_trace(go.Scatter(x=times, y=equity, name="Ledger equity", line_color="#336ac0"), row=1, col=1)
    figure.add_trace(go.Scatter(x=times, y=drawdown, name="Drawdown %", fill="tozeroy", line_color="#c34d58"),
                     row=2, col=1)
    for name, values, color in (("Gross exposure", [point[2] for point in points], "#7963b5"),
                                ("Cash (equity − gross)", [point[1] - point[2] if point[2] is not None else None
                                                          for point in points], "#168453")):
        figure.add_trace(go.Scatter(x=times, y=values, name=name, line_color=color), row=3, col=1)
    if not points:
        figure.add_annotation(text="Ledger history unavailable", showarrow=False)
    figure.update_yaxes(title_text="USD", row=1, col=1)
    figure.update_yaxes(title_text="% below peak", row=2, col=1)
    figure.update_yaxes(title_text="USD", row=3, col=1)
    return _style(figure, height=630)


def _risk_figure(records):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08, subplot_titles=(
        "One-bar portfolio risk estimates (USD)", "Projected change in risk (USD)",
        "Conditional proposal costs and observed ledger costs (USD)"))
    times = [_date(record["timestamp"]).isoformat() for record in records]
    definitions = [
        (1, "Before VaR", ("before", "value_at_risk"), "#336ac0", "solid"),
        (1, "Projected VaR", ("projected", "value_at_risk"), "#336ac0", "dash"),
        (1, "Before ES", ("before", "expected_shortfall"), "#c34d58", "solid"),
        (1, "Projected ES", ("projected", "expected_shortfall"), "#c34d58", "dash"),
        (2, "VaR change", ("delta", "value_at_risk"), "#336ac0", "solid"),
        (2, "ES change", ("delta", "expected_shortfall"), "#c34d58", "solid"),
        (3, "Conditional commission", ("proposal", "estimates", "commission_cost"), "#7963b5", "dash"),
        (3, "Conditional half-spread", ("proposal", "estimates", "half_spread_cost"), "#d89216", "dash"),
        (3, "Conditional slippage", ("proposal", "estimates", "configured_slippage_cost"), "#c34d58", "dash"),
        (3, "Observed fee", ("observed_execution", "fee"), "#168453", "solid"),
        (3, "Observed slippage cost", ("observed_execution", "slippage_cost"), "#172a49", "solid"),
    ]
    count = 0
    for row, name, path, color, dash in definitions:
        values = []
        for record in records:
            value = _mapping(record.get("risk_estimate"))
            for key in path:
                value = _mapping(value).get(key)
            values.append(_number(value))
        if any(value is not None for value in values):
            count += 1
            figure.add_trace(go.Scatter(x=times, y=values, name=name, mode="lines+markers", connectgaps=False,
                                       line=dict(color=color, dash=dash), marker_size=4), row=row, col=1)
    if not count:
        figure.add_annotation(text="Decision risk estimates unavailable", showarrow=False)
    figure.add_hline(y=0, line_width=1, line_color="#cbd3df", row=2, col=1)
    figure.update_yaxes(title_text="USD")
    return _style(figure, height=730)


def _return_figure(points):
    returns = [100 * (current[1] / previous[1] - 1) for previous, current in zip(points, points[1:])
               if previous[1] > 0]
    figure = go.Figure()
    if returns:
        figure.add_trace(go.Histogram(x=returns, name="Observed ledger intervals", marker_color="#336ac0",
                                     nbinsx=min(30, max(5, len(returns) // 3))))
    else:
        figure.add_annotation(text="Insufficient ledger observations for a return distribution", showarrow=False)
    figure.update_layout(xaxis_title="Return between successive ledger observations (%)", yaxis_title="Count")
    return _style(figure, height=340)


def _daily_figure(report, first, last):
    analytics = _mapping(_mapping(report.get("diagnostics")).get("performance"))
    canonical = _mapping(analytics.get("performance")).get("daily_returns", ())
    daily = {}
    for row in canonical:
        row = _mapping(row)
        try:
            start, end = date.fromisoformat(row.get("start_date", "")), date.fromisoformat(row.get("end_date", ""))
        except (TypeError, ValueError):
            continue
        value = _number(row.get("value"))
        if (value is not None and first is not None and start < end
                and first.astimezone(ZoneInfo("America/New_York")).date() <= start
                and end <= last.astimezone(ZoneInfo("America/New_York")).date()):
            daily[end] = 100 * value
    figure = go.Figure()
    if not daily:
        figure.add_annotation(text="Daily heatmap unavailable: at least two completed NY session closes are required",
                              showarrow=False, x=.5, y=.5, xref="paper", yref="paper")
    else:
        months = sorted({day.strftime("%Y-%m") for day in daily})
        values = [[next((value for date, value in daily.items() if date.strftime("%Y-%m") == month
                         and date.day == day), None) for day in range(1, 32)] for month in months]
        figure.add_trace(go.Heatmap(x=list(range(1, 32)), y=months, z=values, zmid=0, colorscale="RdYlGn",
                                   colorbar=dict(title="Return %"), hoverongaps=False,
                                   hovertemplate="%{y}-%{x}<br>Observed daily return: %{z:.3f}%<extra></extra>"))
        figure.update_xaxes(title_text="Day of month", dtick=2)
    return _style(figure, height=340)


def _metric_rows(report, indicators):
    analytics = _mapping(_mapping(report.get("diagnostics")).get("performance"))
    rows = []

    def visit(value, prefix, depth=0):
        if isinstance(value, dict) and depth <= 3:
            for key, child in value.items():
                visit(child, f"{prefix} / {key}" if prefix else str(key), depth + 1)
        elif value is None or isinstance(value, (str, bool, int, float)):
            display = f"{value:,.6g}" if _number(value) is not None else _value(value)
            rows.append((prefix.replace("_", " "), display))

    for section in ("performance", "risk", "execution", "trades", "portfolio", "benchmark", "availability",
                    "methodology", "definitions"):
        visit(analytics.get(section, {}), section)
    canonical = _mapping(analytics.get("indicators"))
    for symbol, series in sorted(indicators.items()):
        latest = {key: value for key, value in series[-1].items() if key not in {"timestamp", "close", "volume"}} \
                 if series else {}
        visit(latest, f"indicators / {symbol} / latest")
        visit(_mapping(canonical.get(symbol)).get("diagnostics", {}), f"indicators / {symbol} / diagnostics")
    return rows


def write_dashboard(output_dir, *, bars, records, report, start, stop):
    """Write a self-contained HTML report for exactly ``[start, stop)`` bars."""
    if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or not isinstance(stop, int)
            or start < 0 or stop < start or any(stop > len(stock) for stock in bars.values())):
        raise ValueError("dashboard start/stop must select a valid common bar interval")
    selected = {symbol: list(stock[start:stop]) for symbol, stock in bars.items()}
    times = [bar.timestamp.astimezone(timezone.utc) for stock in selected.values() for bar in stock]
    first, last = (min(times), max(times)) if times else (None, None)
    decisions = []
    for record in records:
        timestamp = _date(record.get("timestamp"))
        index = record.get("bar_index")
        if (timestamp is not None and first is not None and first <= timestamp <= last
                and (index is None or isinstance(index, int) and start <= index < stop)):
            decisions.append(record)
    decisions.sort(key=lambda record: _date(record["timestamp"]))
    points = _equity_points(report, first, last) if first is not None else []
    funding = _number(report.get("initial_funding"))
    indicators = _indicator_rows(report, selected)
    figures = {"market": _market_figure(selected, decisions, indicators),
               "technical": _technical_figure(indicators), "portfolio": _portfolio_figure(points, funding),
               "risk": _risk_figure(decisions), "returns": _return_figure(points),
               "daily": _daily_figure(report, first, last)}
    payload = {"figures": {name: figure.to_plotly_json() for name, figure in figures.items()},
               "decisions": [_row(record) for record in decisions]}
    execution = _mapping(report.get("execution"))
    analytics = _mapping(_mapping(report.get("diagnostics")).get("performance"))
    canonical_drawdown = _mapping(_mapping(analytics.get("performance")).get("drawdown"))
    drawdown = canonical_drawdown.get("max_drawdown", execution.get("max_drawdown"))
    coverage = _mapping(analytics.get("execution")).get("matching_coverage", execution.get("coverage"))
    cards = (("Initial funding", _money(funding)), ("Final ledger equity", _money(report.get("final_equity"))),
             ("Ledger return", _percent(report.get("ledger_return"))),
             ("Maximum drawdown", _percent(drawdown)),
             ("Observed match coverage", _percent(coverage)),
             ("Observed fees", _money(execution.get("total_fees"))))
    card_html = "".join(f'<div class="card"><span>{label}</span><strong>{html.escape(value)}</strong></div>'
                        for label, value in cards)
    provenance = {key: report.get(key) for key in ("data_source", "data_provenance", "source", "dataset")}
    synthetic = (report.get("source_is_synthetic") is True or report.get("synthetic") is True
                 or report.get("data_is_synthetic") is True
                 or "synthetic" in str(provenance).lower())
    badge = "SYNTHETIC DATA · SOFTWARE DEMONSTRATION" if synthetic else "SUPPLIED HISTORICAL DATA · VERIFY PROVENANCE"
    metric_html = "".join(f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>"
                          for name, value in _metric_rows(report, indicators))
    if not metric_html:
        metric_html = '<tr><td colspan="2">Extended analytics unavailable for this report.</td></tr>'
    period = "No market bars selected" if first is None else f"{first.isoformat()} to {last.isoformat()}"
    context = ("<p><strong>" + html.escape(str(report["evaluation_notice"])) + "</strong></p>"
               if report.get("evaluation_notice") else "")
    replacements = {"PLOTLY": get_plotlyjs(), "PAYLOAD": _safe_json(payload), "BADGE": badge, "CONTEXT": context,
                    "PERIOD": html.escape(period), "CARDS": card_html, "METRICS": metric_html}
    document = re.sub(r"__(PLOTLY|PAYLOAD|BADGE|PERIOD|CARDS|METRICS|CONTEXT)__",
                      lambda match: replacements[match.group(1)], _HTML)
    destination = Path(output_dir) / "dashboard.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return {"dashboard_html": str(destination)}


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StockTrade · Evaluation dashboard</title><style>
:root{font-family:Arial,sans-serif;color:#26334a;background:#f3f5f9}*{box-sizing:border-box}
body{margin:0}main{max-width:1500px;margin:auto;padding:28px 24px 55px}h1{margin:12px 0 8px;font-size:30px}
h2{font-size:21px;margin:0 0 8px}p{line-height:1.6;margin:7px 0}.muted{color:#627087;font-size:14px}
.badge{font-size:11px;font-weight:bold;letter-spacing:1px;color:#78561b;background:#fff2d5;padding:7px 10px;
display:inline-block;border-radius:5px}.notice{background:#eaf0fb;border-left:4px solid #336ac0;padding:14px 18px;
margin:20px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.card,section{background:#fff;border:1px solid #e0e5ed;border-radius:9px}.card{padding:18px}.card span{display:block;
font-size:12px;color:#627087}.card strong{display:block;margin-top:10px;font-size:23px;font-weight:600}
section{padding:22px;margin:18px 0}.chart{width:100%;min-height:250px}.two{display:grid;grid-template-columns:1fr 1fr;
gap:18px}.two section{min-width:0;margin:0}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
input,select{font:inherit;padding:10px;border:1px solid #cbd3df;border-radius:5px}input{min-width:260px;flex:1}
.table-wrap{overflow:auto;max-height:550px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{
padding:11px 12px;border-bottom:1px solid #e7ebf1;text-align:left;vertical-align:top;white-space:nowrap}
th{position:sticky;top:0;background:#f5f7fa;z-index:1}td.reason{white-space:normal;min-width:220px}
details summary{cursor:pointer;font-weight:bold}.metrics td{white-space:normal;min-width:200px}
@media(max-width:850px){main{padding:18px 10px}.two{grid-template-columns:1fr}section{padding:16px 8px}}
</style></head><body><main><span class="badge">__BADGE__</span><h1>Trading evaluation</h1>
<p class="muted">__PERIOD__ · All chart timestamps are UTC.</p>
<div class="notice">__CONTEXT__<strong>Observed-execution analog research</strong><p>Proposals and recorded matching executions
are shown separately. Unmatched or rejected requests are never drawn as fills. This conditional ledger does not establish
executable strategy returns, queue priority, or causal market impact.</p><p>Risk measures and projected post-action holdings
are estimates, not action rewards or penalties. Conditional cost estimates are not additional ledger charges. Missing
measurements remain unavailable. Raw-price results omit unprovided dividends, taxes and corporate-action adjustments.</p></div>
<div class="cards">__CARDS__</div>
<section><h2>Market and trading decisions</h2><p class="muted">Choose a stock in the chart dropdown. Hover for action size,
risk, reasons and source IDs. Proposal markers use the observed decision-bar close; stars use actual recorded analog prices.
Portfolio HOLD decisions remain visible in the decision lane. Drag to zoom, use the toolbar to pan, double-click to reset,
and click legend items to toggle a series.</p><div id="market" class="chart"></div></section>
<section><h2>Technical indicators</h2><p class="muted">Select a stock in this chart's dropdown. Price overlays above show
SMA 20/50/200 and Bollinger bands (20 bars, two standard deviations). These panels use the analytics report's canonical
indicator series; missing warmup values remain blank. Windows count supplied bars, so 20 minute bars and 20 second bars
represent different horizons. Indicators describe observed prices and volumes; they are not action scores or predictions.</p>
<div id="technical" class="chart"></div></section>
<section><h2>Portfolio path</h2><p class="muted">Equity includes observed holdings valued at supplied market marks.
Repeated marks at one timestamp use the last complete ledger snapshot. No terminal sale is invented.</p>
<div id="portfolio" class="chart"></div></section>
<section><h2>Decision risk and costs</h2><p class="muted">VaR and expected shortfall (ES) are one-bar USD estimates.
Dashed projected lines describe the proposed feasible quantity, not a completed trade. Risk change is projected minus
current risk. Dashed proposal costs are conditional estimates; solid observed costs belong to recorded ledger executions.</p>
<div id="risk" class="chart"></div></section>
<div class="two"><section><h2>Ledger return distribution</h2><p class="muted">Intervals follow actual ledger observations;
they are not assumed to be equally spaced or annualized.</p><div id="returns" class="chart"></div></section>
<section><h2>Completed-session daily returns</h2><p class="muted">The analytics report's canonical NY-close returns,
requiring completed session marks. Partial or missing sessions and the initial date remain blank.</p>
<div id="daily" class="chart"></div></section></div>
<section><h2>Decision audit</h2><p class="muted">Every policy decision in this interval appears here, including HOLD,
unmatched and rejected intentions. Projected risk and proposal costs are conditional estimates in USD.</p>
<div class="controls"><input id="search" aria-label="Search decisions" placeholder="Search symbol, reason, source ID…">
<select id="symbol" aria-label="Filter decisions by stock"><option value="">All stocks / portfolio</option></select>
<select id="status" aria-label="Filter decisions by status"><option value="">All statuses</option>
<option>matched</option><option>unmatched</option><option>rejected</option><option>proposed</option><option>hold</option></select>
</div><p id="count" class="muted" aria-live="polite"></p><div class="table-wrap"><table><thead><tr>
<th>UTC decision time</th><th>Stock</th><th>Action</th><th>Requested USD</th><th>Feasible shares</th><th>Status</th><th>Reason</th>
<th>VaR before</th><th>Projected VaR</th><th>VaR change</th><th>Conditional cost</th><th>Observed fee</th><th>Observed source IDs</th>
</tr></thead><tbody id="decisions"></tbody></table></div></section>
<section><details><summary>Extended metrics and data availability</summary><p class="muted">Metric names and units follow
the machine-readable analytics report. Unavailable statistics are not replaced with zero.</p>
<div class="table-wrap"><table class="metrics"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
__METRICS__</tbody></table></div></details></section></main>
<script>__PLOTLY__</script><script type="application/json" id="dashboard-data">__PAYLOAD__</script><script>
"use strict";
const dashboard=JSON.parse(document.getElementById('dashboard-data').textContent);
for(const [id,figure] of Object.entries(dashboard.figures)){
  Plotly.newPlot(id,figure.data,figure.layout,{responsive:true,displaylogo:false,scrollZoom:true});
}
const symbol=document.getElementById('symbol'),status=document.getElementById('status'),search=document.getElementById('search');
for(const name of [...new Set(dashboard.decisions.map(row=>row.symbol))].sort()){
  const option=document.createElement('option');option.value=name;option.textContent=name;symbol.appendChild(option);
}
const columns=['time','symbol','action','requested_value','quantity','status','reason','var_before','var_projected',
               'var_delta','conditional_cost','observed_fee','source_ids'];
function drawTable(){
  const query=search.value.toLocaleLowerCase();
  const rows=dashboard.decisions.filter(row=>(!symbol.value||row.symbol===symbol.value)&&
    (!status.value||row.status===status.value)&&(!query||Object.values(row).join(' ').toLocaleLowerCase().includes(query)));
  const body=document.getElementById('decisions');body.replaceChildren();
  for(const row of rows){const tr=document.createElement('tr');for(const column of columns){
    const td=document.createElement('td');td.textContent=row[column];if(column==='reason')td.className='reason';tr.appendChild(td);
  }body.appendChild(tr);}
  document.getElementById('count').textContent=`Showing ${rows.length} of ${dashboard.decisions.length} decisions`;
}
for(const control of [symbol,status,search])control.addEventListener('input',drawTable);
drawTable();
</script></body></html>"""
