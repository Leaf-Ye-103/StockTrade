"""A dashboard must preserve observed evidence and remain safe and offline."""

import copy
import json
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import pytest

from stocktrade.schemas import MarketBar
from stocktrade.visualization import write_dashboard

BASE = datetime(2024, 10, 1, 13, tzinfo=timezone.utc)


def bars(symbol="NVDA", count=6):
    return [MarketBar(timestamp=BASE + timedelta(minutes=index), symbol=symbol,
                      open=100 + index, high=102 + index, low=99 + index,
                      close=101 + index, volume=1000 + index) for index in range(count)]


def decision(index, side, status, *, symbol="NVDA", reason="observed context"):
    record = {"timestamp": BASE + timedelta(minutes=index), "bar_index": index,
              "policy_action": {"side": side, "symbol": None if side == "hold" else symbol,
                                "value": 0 if side == "hold" else 100},
              "feasibility": {"accepted": status != "rejected", "quantity": 1, "reasons": [reason]},
              "transition": "risk_rejected_no_order" if status == "rejected" else status,
              "risk_estimate": {"before": {"value_at_risk": 5, "expected_shortfall": 8},
                                "projected": {"value_at_risk": 6, "expected_shortfall": 9},
                                "delta": {"value_at_risk": 1, "expected_shortfall": 1},
                                "proposal": {"estimates": {"commission_cost": .1, "half_spread_cost": .2,
                                                           "configured_slippage_cost": .3}},
                                "observed_execution": {"fee": .1 if status == "matched" else None}}}
    if status == "matched":
        record["accounting"] = {"status": "matched", "fee": .1}
        record["analog"] = {"event_timestamp": record["timestamp"] + timedelta(seconds=1),
                            "estimated_price": 100.5, "estimated_quantity": 1,
                            "source_trade_ids": ["observed-trade-123"]}
    return record


def report(count=6):
    return {"source_is_synthetic": True, "initial_funding": 1000, "final_equity": 1010,
            "ledger_return": .01, "execution": {"coverage": .5, "max_drawdown": .02, "total_fees": .1,
            "equity_curve": [{"timestamp": BASE + timedelta(minutes=index), "equity": 1000 + index,
                              "gross_exposure": index * 10} for index in range(count)]},
            "diagnostics": {"performance": {"risk": {"sharpe_ratio": None},
                                               "availability": {"daily_statistics": False}}}}


def dashboard(tmp_path, **kwargs):
    options = {"bars": {"NVDA": bars(), "MSFT": bars("MSFT")}, "records": [],
               "report": report(), "start": 0, "stop": 6}
    options.update(kwargs)
    paths = write_dashboard(tmp_path, **options)
    document = (tmp_path / "dashboard.html").read_text()
    match = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>', document, re.S)
    assert match is not None
    return paths, document, json.loads(match.group(1))


class ScriptParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.scripts.append(dict(attrs))


def test_self_contained_interactive_report_and_synthetic_badge(tmp_path):
    paths, document, payload = dashboard(tmp_path)
    assert paths == {"dashboard_html": str(tmp_path / "dashboard.html")}
    parser = ScriptParser()
    parser.feed(document)
    assert all("src" not in script for script in parser.scripts)
    assert len(parser.scripts) == 3  # Inline Plotly library, JSON, controller.
    assert "plotly.js v3.1.1" in document
    assert "SYNTHETIC DATA" in document
    assert "Plotly.newPlot" in document
    assert "scrollZoom:true" in document
    assert "drawTable" in document
    buttons = payload["figures"]["market"]["layout"]["updatemenus"][0]["buttons"]
    assert [button["label"] for button in buttons] == ["MSFT", "NVDA"]
    assert all(len(button["args"][0]["visible"]) == len(payload["figures"]["market"]["data"])
               for button in buttons)


def test_all_decisions_and_statuses_visible_and_observed_execution_distinct(tmp_path):
    records = [decision(0, "buy", "matched"), decision(1, "sell", "unmatched"),
               decision(2, "buy", "rejected"), decision(3, "hold", "hold")]
    _, _, payload = dashboard(tmp_path, records=records)
    assert [row["status"] for row in payload["decisions"]] == ["matched", "unmatched", "rejected", "hold"]
    assert [row["action"] for row in payload["decisions"]] == ["BUY", "SELL", "BUY", "HOLD"]
    traces = payload["figures"]["market"]["data"]
    observed = [trace for trace in traces if trace["name"] == "Observed matched execution"]
    assert len(observed) == 1
    assert observed[0]["y"] == [100.5]
    proposals = [trace for trace in traces if trace["name"] == "BUY proposal · matched"]
    assert proposals[0]["y"] == [101]  # Observed close for a proposal; not a fill.
    holds = [trace for trace in traces if trace["name"] == "Portfolio HOLD / other"]
    assert holds[0]["y"] == ["hold"]
    assert holds[0]["visible"] is True  # Independent of selected stock.
    assert payload["decisions"][0]["source_ids"] == "observed-trade-123"
    assert payload["decisions"][1]["observed_fee"] == "Unavailable"


def test_no_observed_execution_marker_for_rejected_analog(tmp_path):
    record = decision(0, "buy", "matched")
    record["accounting"]["status"] = "rejected"
    record["transition"] = "rejected"
    _, _, payload = dashboard(tmp_path, records=[record])
    assert payload["decisions"][0]["status"] == "rejected"
    assert payload["decisions"][0]["source_ids"] == ""
    assert all(trace["name"] != "Observed matched execution" for trace in payload["figures"]["market"]["data"])


def test_declared_slice_excludes_past_future_prices_decisions_and_equity(tmp_path):
    records = [decision(index, "buy", "unmatched", reason=f"decision-{index}") for index in range(6)]
    _, _, payload = dashboard(tmp_path, records=records, start=2, stop=4)
    candles = [trace for trace in payload["figures"]["market"]["data"] if trace["type"] == "candlestick"]
    assert all(trace["close"] == [103, 104] for trace in candles)
    assert [row["reason"] for row in payload["decisions"]] == ["decision-2", "decision-3"]
    equity = payload["figures"]["portfolio"]["data"][0]
    assert equity["y"] == [1002, 1003]
    serialized = json.dumps(payload)
    assert (BASE + timedelta(minutes=5)).isoformat() not in serialized
    assert BASE.isoformat() not in serialized


def test_escape_untrusted_strings_without_script_or_html_injection(tmp_path):
    attack = '</script><script id="injected">alert(1)</script><img src=x onerror="alert(2)">__CARDS__'
    record = decision(0, "buy", "unmatched", reason=attack)
    metadata = report()
    metadata["diagnostics"]["performance"]["risk"]["untrusted"] = attack
    _, document, payload = dashboard(tmp_path, records=[record], report=metadata)
    parser = ScriptParser()
    parser.feed(document)
    assert len(parser.scripts) == 3
    assert '<script id="injected">' not in document
    assert '<img src=x onerror=' not in document
    assert payload["decisions"][0]["reason"] == attack
    assert "__CARDS__" in payload["decisions"][0]["reason"]
    assert "td.textContent=row[column]" in document
    assert "innerHTML" not in document.split('"use strict";')[-1]


def test_template_tokens_in_untrusted_fields_round_trip_without_a_second_substitution(tmp_path):
    tokens = "__PLOTLY__ __PAYLOAD__ __BADGE__ __PERIOD__ __CARDS__ __METRICS__ __CONTEXT__"
    record = decision(0, "buy", "unmatched", reason=tokens)
    metadata = report()
    metadata["diagnostics"]["performance"]["definitions"] = {"untrusted": tokens}
    _, document, payload = dashboard(tmp_path, records=[record], report=metadata)
    assert payload["decisions"][0]["reason"] == tokens
    assert f"<td>{tokens}</td>" in document


def test_canonical_technical_indicators_are_plotted_without_recalculation_and_warmup_stays_null(tmp_path):
    metadata = report()
    rows = [{"timestamp": BASE + timedelta(minutes=index), "sma_20": 80 + index,
             "sma_50": None, "sma_200": None, "bollinger_middle": 80 + index,
             "bollinger_upper": 90 + index, "bollinger_lower": 70 + index,
             "rsi_14": None if index == 0 else 42, "macd": 1.2, "macd_signal": 1.0,
             "macd_histogram": .2, "atr_14": 2.3, "obv": 500 + index,
             "realized_volatility_20": .01} for index in range(6)]
    metadata["diagnostics"]["performance"]["indicators"] = {
        "NVDA": {"series": rows, "latest": {"rsi_14": 42}, "diagnostics": {"quote_coverage": .9}}}
    _, document, payload = dashboard(tmp_path, bars={"NVDA": bars()}, report=metadata)
    market = {trace["name"]: trace for trace in payload["figures"]["market"]["data"]}
    assert market["SMA 20"]["y"] == [80, 81, 82, 83, 84, 85]
    assert market["SMA 50"]["y"] == [None] * 6
    assert market["SMA 200"]["connectgaps"] is False
    assert market["Bollinger upper"]["y"] == [90, 91, 92, 93, 94, 95]
    technical = {trace["name"]: trace for trace in payload["figures"]["technical"]["data"]}
    assert technical["RSI 14"]["y"] == [None, 42, 42, 42, 42, 42]
    assert technical["MACD"]["y"] == [1.2] * 6
    assert technical["MACD signal"]["y"] == [1.0] * 6
    assert technical["MACD histogram"]["type"] == "bar"
    assert technical["ATR 14"]["y"] == [2.3] * 6
    assert technical["OBV"]["y"] == [500, 501, 502, 503, 504, 505]
    assert technical["Realized volatility 20"]["y"] == [.01] * 6
    assert 'id="technical"' in document
    assert "indicators / NVDA / latest / rsi 14" in document
    assert "indicators / NVDA / diagnostics / quote coverage" in document


def test_technical_series_and_latest_display_cannot_leak_outside_declared_interval(tmp_path):
    metadata = report()
    rows = [{"timestamp": BASE + timedelta(minutes=index), "sma_20": 200 + index,
             "rsi_14": 40 + index} for index in range(6)]
    metadata["diagnostics"]["performance"]["indicators"] = {
        "NVDA": {"series": rows, "latest": {"rsi_14": 987654321}, "diagnostics": {"quote_coverage": .9}}}
    _, document, payload = dashboard(tmp_path, bars={"NVDA": bars()}, report=metadata, start=2, stop=4)
    market = {trace["name"]: trace for trace in payload["figures"]["market"]["data"]}
    assert market["SMA 20"]["y"] == [202, 203]
    technical = {trace["name"]: trace for trace in payload["figures"]["technical"]["data"]}
    assert technical["RSI 14"]["y"] == [42, 43]
    assert "indicators / NVDA / latest / rsi 14</td><td>43</td>" in document
    assert "987654321" not in document
    serialized = json.dumps(payload)
    assert BASE.isoformat() not in serialized
    assert (BASE + timedelta(minutes=5)).isoformat() not in serialized


def test_technical_figure_has_per_symbol_selection(tmp_path):
    metadata = report()
    metadata["diagnostics"]["performance"]["indicators"] = {
        symbol: {"series": [{"timestamp": BASE, "rsi_14": value}]} for symbol, value in (("MSFT", 30), ("NVDA", 70))}
    _, _, payload = dashboard(tmp_path, report=metadata)
    figure = payload["figures"]["technical"]
    buttons = figure["layout"]["updatemenus"][0]["buttons"]
    assert [button["label"] for button in buttons] == ["MSFT", "NVDA"]
    assert sum(buttons[0]["args"][0]["visible"]) == 7
    assert sum(buttons[1]["args"][0]["visible"]) == 7
    assert buttons[0]["args"][0]["visible"] != buttons[1]["args"][0]["visible"]


def test_empty_records_and_missing_diagnostics_are_honest(tmp_path):
    metadata = {"source_is_synthetic": False}
    _, document, payload = dashboard(tmp_path, records=[], report=metadata)
    assert payload["decisions"] == []
    assert "SUPPLIED HISTORICAL DATA" in document
    assert "SYNTHETIC DATA" not in document
    assert "Extended analytics unavailable" in document
    assert "Decision risk estimates unavailable" in json.dumps(payload["figures"]["risk"])
    assert "at least two completed NY session closes" in json.dumps(payload["figures"]["daily"])
    assert "Unavailable" in document


def test_no_submitted_orders_show_unavailable_coverage_instead_of_failed_zero_percent(tmp_path):
    metadata = report()
    metadata["execution"]["coverage"] = 0
    metadata["diagnostics"]["performance"]["execution"] = {"matching_coverage": None}
    _, document, _ = dashboard(tmp_path, records=[decision(0, "hold", "hold")], report=metadata)
    assert "Observed match coverage</span><strong>Unavailable" in document


def test_risk_estimates_and_conditional_costs_stay_separate_from_observed_costs(tmp_path):
    _, _, payload = dashboard(tmp_path, records=[decision(0, "buy", "matched"), decision(1, "hold", "hold")])
    traces = {trace["name"]: trace for trace in payload["figures"]["risk"]["data"]}
    assert traces["Projected VaR"]["y"] == [6, 6]
    assert traces["VaR change"]["y"] == [1, 1]
    assert traces["Conditional commission"]["line"]["dash"] == "dash"
    assert traces["Observed fee"]["y"] == [.1, None]
    assert traces["Observed fee"]["line"]["dash"] == "solid"
    assert payload["decisions"][0]["conditional_cost"] == "$0.60"


def test_daily_heatmap_uses_canonical_completed_session_series_and_preserves_missing_days(tmp_path):
    stock = [MarketBar(timestamp=BASE + timedelta(hours=7, days=day), symbol="NVDA", open=100, high=102,
                       low=99, close=101, volume=1000) for day in (0, 1, 3)]
    metadata = report()
    metadata["execution"]["equity_curve"] = [{"timestamp": item.timestamp, "equity": equity,
                                              "gross_exposure": 100} for item, equity in zip(stock, (1000, 1010, 990))]
    metadata["diagnostics"]["performance"]["performance"] = {"daily_returns": [
        {"start_date": "2024-10-01", "end_date": "2024-10-02", "value": .01}]}
    _, _, payload = dashboard(tmp_path, bars={"NVDA": stock}, report=metadata, start=0, stop=3)
    heatmap = payload["figures"]["daily"]["data"][0]
    assert heatmap["type"] == "heatmap"
    assert heatmap["y"] == ["2024-10"]
    assert heatmap["z"][0][0] is None
    assert heatmap["z"][0][1] == pytest.approx(1)
    assert heatmap["z"][0][2] is None
    assert heatmap["z"][0][3] is None  # Missing Oct 3 breaks a completed daily return.


def test_daily_heatmap_does_not_derive_daily_returns_from_partial_session_marks(tmp_path):
    stock = [MarketBar(timestamp=BASE + timedelta(days=day), symbol="NVDA", open=100, high=102,
                       low=99, close=101, volume=1000) for day in (0, 1, 2)]
    metadata = report()
    metadata["execution"]["equity_curve"] = [{"timestamp": item.timestamp, "equity": 1000 + index,
                                              "gross_exposure": 100} for index, item in enumerate(stock)]
    _, _, payload = dashboard(tmp_path, bars={"NVDA": stock}, report=metadata, stop=3)
    assert payload["figures"]["daily"]["data"] == []


def test_inputs_are_not_mutated(tmp_path):
    stocks, records, metadata = {"NVDA": bars()}, [decision(0, "buy", "matched")], report()
    before = copy.deepcopy((stocks, records, metadata))
    dashboard(tmp_path, bars=stocks, records=records, report=metadata)
    assert (stocks, records, metadata) == before


@pytest.mark.parametrize("start,stop", [(-1, 2), (3, 2), (0, 7), (True, 4)])
def test_invalid_slice_is_rejected(tmp_path, start, stop):
    with pytest.raises(ValueError, match="valid common bar interval"):
        dashboard(tmp_path, start=start, stop=stop)
