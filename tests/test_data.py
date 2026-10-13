"""Data-contract and point-in-time tests; no remote network access required."""

import io
import json
import math
import urllib.request
import urllib.robotparser
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from stocktrade.data import _SameOriginHTTPSRedirect, build_features, collect_rss, load_bars_csv, load_events_jsonl
from stocktrade.schemas import InformationEvent, MarketBar

BASE = datetime(2024, 10, 8, 14, 30, tzinfo=timezone.utc)


def bar(minutes=0, close=100.0, **kwargs):
    return MarketBar(timestamp=BASE + timedelta(minutes=minutes), symbol="NVDA",
                     open=close, high=close + 1, low=close - 1, close=close, volume=1000, **kwargs)


def event(kind, *, published=0, available=0, values=None, text="", symbol="NVDA"):
    return InformationEvent(published_at=BASE + timedelta(minutes=published),
                            available_at=BASE + timedelta(minutes=available),
                            source="observed-source", kind=kind, values=values or {}, text=text, symbol=symbol)


def column(frame, name):
    index = frame.feature_names.index(name)
    return frame.values[:, index], frame.observed[:, index]


def test_information_requires_both_publication_and_availability():
    frame = build_features([bar(0), bar(1), bar(2), bar(3)], [
        event("macro", published=-10, available=2, values={"policy_rate": 5.0}, symbol=None),
        event("fundamentals", published=3, available=1, values={"revenue": 10.0}),
        event("fundamentals", published=-2, available=-1, values={"other_company": 99.0}, symbol="AAPL"),
    ])
    rate, rate_observed = column(frame, "macro.policy_rate")
    revenue, revenue_observed = column(frame, "fundamentals.revenue")
    np.testing.assert_array_equal(rate, [0, 0, 5, 5])
    np.testing.assert_array_equal(rate_observed, [False, False, True, True])
    np.testing.assert_array_equal(revenue, [0, 0, 0, 10])
    np.testing.assert_array_equal(revenue_observed, [False, False, False, True])
    assert "fundamentals.other_company" not in frame.feature_names


def test_delayed_old_release_does_not_overwrite_newer_known_release():
    frame = build_features([bar(0), bar(1), bar(2)], [
        event("fundamentals", published=-5, available=-1, values={"eps": 2.0}),
        event("fundamentals", published=-20, available=1, values={"eps": 1.0}),
        event("fundamentals", published=1, available=2, values={"eps": 3.0}),
    ])
    np.testing.assert_array_equal(column(frame, "fundamentals.eps")[0], [2, 2, 3])


def test_missing_quotes_null_events_and_initial_returns_have_masks():
    frame = build_features([bar(1, 101), bar(0, 100)], [
        event("macro", values={"missing": None}),
        event("market", values={"log_return": 0.2}),
    ])
    values, observed = column(frame, "market.log_return")
    np.testing.assert_array_equal(observed, [False, True])
    assert values[1] == pytest.approx(math.log(101 / 100))
    np.testing.assert_allclose(column(frame, "macro.market.log_return")[0], [0.2, 0.2])
    assert not column(frame, "macro.missing")[1].any()
    assert not frame.observed[:, frame.groups["orderbook"]].any()
    assert np.isfinite(frame.values).all()
    assert set(frame.groups) == {"market", "orderbook", "macro", "fundamentals", "news"}


def test_observed_quotes_provide_snapshot_not_fabricated_queue():
    frame = build_features([bar(bid=99, ask=101, bid_size=30, ask_size=10)], [])
    assert column(frame, "orderbook.relative_spread")[0][0] == pytest.approx(0.02)
    assert column(frame, "orderbook.imbalance")[0][0] == pytest.approx(0.5)
    assert frame.observed[:, frame.groups["orderbook"]].all()
    assert not any("queue" in name for name in frame.feature_names)


def test_measured_orderflow_is_point_in_time_and_uses_orderbook_group():
    frame = build_features([bar(0), bar(1), bar(2)], [
        event("orderflow", published=0, available=1,
              values={"l2_bid_depth": 1200, "queue_ahead": 400, "signed_trade_volume": -50}),
    ])
    depth, depth_observed = column(frame, "orderbook.flow.l2_bid_depth")
    np.testing.assert_array_equal(depth, [0, 1200, 1200])
    np.testing.assert_array_equal(depth_observed, [False, True, True])
    for name in ("l2_bid_depth", "queue_ahead", "signed_trade_volume"):
        index = frame.feature_names.index(f"orderbook.flow.{name}")
        assert index in frame.groups["orderbook"]
        assert index not in frame.groups["macro"]
    assert not frame.observed[:, frame.groups["orderbook"][:4]].any()


def test_future_orderflow_keys_do_not_expand_frozen_training_schema():
    bars = [bar(0), bar(1), bar(2)]
    known = event("orderflow", published=0, available=0, values={"l2_bid_depth": 100})
    future = event("orderflow", published=2, available=2, values={"new_queue_measurement": 200})
    training_schema = build_features(bars[:2], [known]).feature_names
    frame = build_features(bars, [known, future], feature_names=training_schema)
    assert frame.feature_names == training_schema
    assert "orderbook.flow.new_queue_measurement" not in frame.feature_names
    np.testing.assert_array_equal(column(frame, "orderbook.flow.l2_bid_depth")[0], [100, 100, 100])


def test_news_is_causal_stable_and_decays_from_collection():
    bars = [bar(0), bar(1), bar(361)]
    events = [event("news", published=-1000, available=1, text="Company profit increases sharply.")]
    frame = build_features(bars, events)
    again = build_features(bars, events)
    np.testing.assert_array_equal(frame.values, again.values)
    news_indices = frame.groups["news"]
    assert not frame.observed[0, news_indices].any()
    assert frame.observed[1:, news_indices].all()
    count, _ = column(frame, "news.decayed_count")
    assert count[1] == 1
    assert count[2] == pytest.approx(math.exp(-1))
    age, _ = column(frame, "news.log_seconds_since_latest")
    assert age[1] == 0
    assert age[2] == pytest.approx(math.log1p(6 * 3600))


def test_frozen_training_schema_discards_future_only_event_keys():
    bars = [bar(0), bar(1), bar(2)]
    initial = event("macro", values={"policy_rate": 5.0}, symbol=None)
    future = event("fundamentals", published=2, available=2, values={"future_metric": 99.0})
    training_schema = build_features(bars[:2], [initial]).feature_names
    frozen = build_features(bars, [initial, future], feature_names=training_schema)
    assert frozen.feature_names == training_schema
    assert frozen.values.shape[1] == len(training_schema)
    assert "fundamentals.future_metric" not in frozen.feature_names
    assert frozen.groups["fundamentals"] == []
    np.testing.assert_array_equal(frozen.values[:2], build_features(bars[:2], [initial]).values)


def test_frozen_schema_preserves_order_and_masks_unavailable_columns():
    names = ["fundamentals.eps", "market.log_return", "macro.policy_rate", "orderbook.imbalance"]
    frame = build_features([bar(0), bar(1, 101)], [], feature_names=names)
    assert frame.feature_names == names
    assert frame.groups == {"market": [1], "orderbook": [3], "macro": [2], "fundamentals": [0], "news": []}
    assert not frame.observed[:, [0, 2, 3]].any()
    assert not frame.values[:, [0, 2, 3]].any()
    assert frame.observed[1, 1]
    names.append("news.decayed_count")
    assert len(frame.feature_names) == 4


@pytest.mark.parametrize("names,error", [
    ([], "nonempty"),
    (["market.log_return", "market.log_return"], "unique"),
    (["unknown.foo"], "recognized"),
    (["macro."], "recognized"),
    ([None], "recognized"),
])
def test_frozen_schema_rejects_invalid_names(names, error):
    with pytest.raises(ValueError, match=error):
        build_features([bar()], [], feature_names=names)


@pytest.mark.parametrize("bad_bars,error", [
    ([bar(), bar()], "unique"),
    ([bar(), MarketBar(BASE, "AAPL", 100, 101, 99, 100, 100)], "one symbol"),
    ([MarketBar(BASE.replace(tzinfo=None), "NVDA", 100, 101, 99, 100, 100)], "timezone-aware"),
    ([MarketBar(BASE, "NVDA", 100, 99, 98, 100, 100)], "enclose"),
    ([bar(bid=101, ask=100)], "bid cannot exceed"),
    ([MarketBar(BASE, "NVDA", 100, 101, 99, float("nan"), 100)], "finite"),
])
def test_build_rejects_invalid_data(bad_bars, error):
    with pytest.raises(ValueError, match=error):
        build_features(bad_bars, [])


def test_csv_jsonl_loaders_preserve_missingness_and_availability(tmp_path):
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text("timestamp,symbol,open,high,low,close,volume,bid,ask\n"
                        "2024-10-08T14:30:00Z,NVDA,100,101,99,100,1000,,\n", encoding="utf-8")
    bars = load_bars_csv(csv_path)
    assert bars[0].bid is None
    path = tmp_path / "events.jsonl"
    payload = {"published_at": "2024-10-07T00:00:00Z", "available_at": "2024-10-09T00:00:00Z",
               "source": "release", "kind": "fundamentals", "values": {"eps": None}}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    events = load_events_jsonl(path)
    assert events[0].available_at > bars[0].timestamp
    assert events[0].values["eps"] is None
    frame = build_features(bars, events)
    assert not column(frame, "fundamentals.eps")[1].any()


def test_loader_rejects_naive_dates_and_missing_available_at(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("timestamp,symbol,open,high,low,close,volume\n"
                    "2024-10-08T14:30:00,NVDA,100,101,99,100,1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2.*timezone-aware"):
        load_bars_csv(path)
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(json.dumps({"published_at": "2024-10-08T00:00:00Z", "source": "x", "kind": "news"}), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1.*available_at"):
        load_events_jsonl(event_path)


class FakeOpener:
    def __init__(self, robots, feed):
        self.robots = robots
        self.feed = feed
        self.urls = []

    def open(self, request, timeout):
        self.urls.append(request.full_url)
        return io.BytesIO(self.robots if request.full_url.endswith("/robots.txt") else self.feed)


def test_rss_collection_uses_actual_availability_and_keeps_provenance(tmp_path, monkeypatch):
    feed = b"<rss><channel><item><title>Profit rises</title><description>Quarterly release</description><pubDate>Tue, 08 Oct 2024 14:00:00 GMT</pubDate></item></channel></rss>"
    fake = FakeOpener(b"User-agent: *\nAllow: /\n", feed)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: fake)
    before = datetime.now(timezone.utc)
    output = tmp_path / "news.jsonl"
    assert collect_rss("https://example.org/feed", output, symbol="NVDA") == 1
    result = load_events_jsonl(output)[0]
    assert result.available_at >= before
    assert result.published_at == datetime(2024, 10, 8, 14, tzinfo=timezone.utc)
    assert result.source == "https://example.org/feed"
    assert result.symbol == "NVDA"
    assert fake.urls == ["https://example.org/robots.txt", "https://example.org/feed"]


def test_rss_obeys_robots_and_rejects_insecure_fetches(tmp_path, monkeypatch):
    fake = FakeOpener(b"User-agent: *\nDisallow: /feed\n", b"<rss/>")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: fake)
    with pytest.raises(PermissionError, match="disallows"):
        collect_rss("https://example.org/feed", tmp_path / "news.jsonl")
    assert fake.urls == ["https://example.org/robots.txt"]
    with pytest.raises(ValueError, match="HTTPS"):
        collect_rss("http://example.org/feed", tmp_path / "news.jsonl")
    assert not (tmp_path / "news.jsonl").exists()


def test_rss_rejects_xml_entity_declarations(tmp_path, monkeypatch):
    fake = FakeOpener(b"User-agent: *\nAllow: /\n", b"<!DOCTYPE rss><rss/>")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: fake)
    with pytest.raises(ValueError, match="DTD"):
        collect_rss("https://example.org/feed", tmp_path / "news.jsonl")


def test_rss_rejects_utf16_entity_declarations(tmp_path, monkeypatch):
    payload = ('<?xml version="1.0" encoding="UTF-16"?>'
               '<!DOCTYPE rss [<!ENTITY injected "expanded text">]>'
               '<rss><channel><item><title>&injected;</title></item></channel></rss>').encode("utf-16")
    fake = FakeOpener(b"User-agent: *\nAllow: /\n", payload)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: fake)
    with pytest.raises(ValueError, match="DTD"):
        collect_rss("https://example.org/feed", tmp_path / "news.jsonl")
    assert not (tmp_path / "news.jsonl").exists()


@pytest.mark.parametrize("destination", ["http://example.org/feed", "https://other.org/feed", "https://user:pass@example.org/feed"])
def test_rss_redirect_rejects_protocol_origin_or_credentials(destination):
    handler = _SameOriginHTTPSRedirect()
    request = urllib.request.Request("https://example.org/feed")
    with pytest.raises(ValueError, match="same HTTPS origin"):
        handler.redirect_request(request, None, 302, "Moved", {}, destination)


def test_rss_redirect_checks_destination_robots_policy():
    handler = _SameOriginHTTPSRedirect()
    robots = urllib.robotparser.RobotFileParser("https://example.org/robots.txt")
    robots.parse(["User-agent: *", "Disallow: /private"])
    handler.robots = robots
    handler.user_agent = "StockTradeResearch/0.1"
    request = urllib.request.Request("https://example.org/feed")
    with pytest.raises(PermissionError, match="redirected"):
        handler.redirect_request(request, None, 302, "Moved", {}, "https://example.org/private/news")
