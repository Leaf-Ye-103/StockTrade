"""Synthetic read-only platform imports, clocks, pagination, and secret handling."""

import io
import json
import ssl
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from stocktrade import platforms
from stocktrade.data import build_features, load_bars_csv, load_events_jsonl
from stocktrade.schemas import MarketBar

BASE = datetime(2026, 10, 1, 13, tzinfo=timezone.utc)
START = BASE - timedelta(hours=1)
END = BASE + timedelta(hours=1)


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-private-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-private-secret")
    monkeypatch.setattr(platforms, "_now", lambda: END)


def vendor_bar(timestamp=BASE, **kwargs):
    return {"t": timestamp.isoformat(), "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1200, **kwargs}


def vendor_news(timestamp=BASE, **kwargs):
    return {"id": 1, "created_at": timestamp.isoformat(), "headline": "Observed report",
            "summary": "Actual information", "content": "<p>Published content.</p>",
            "symbols": ["NVDA"], "url": "https://publisher.example/article", **kwargs}


def responses(monkeypatch, *pages):
    queue = iter(pages)
    calls = []

    def request(path, params, **kwargs):
        calls.append((path, dict(params), kwargs))
        return next(queue)

    monkeypatch.setattr(platforms, "_request_json", request)
    return calls


def test_bars_paginate_shift_open_to_end_and_preserve_missing_quotes(tmp_path, monkeypatch, credentials):
    calls = responses(monkeypatch,
                      {"bars": {"NVDA": [vendor_bar()]}, "next_page_token": "next"},
                      {"bars": {"AAPL": [vendor_bar(BASE + timedelta(minutes=1))]}})
    result = platforms.fetch_alpaca_bars(["nvda", "AAPL"], START.isoformat(), END.isoformat(), tmp_path)
    bars = load_bars_csv(result["bars_path"])
    assert [bar.symbol for bar in bars] == ["NVDA", "AAPL"]
    assert [bar.timestamp for bar in bars] == [BASE + timedelta(minutes=1), BASE + timedelta(minutes=2)]
    assert bars[0].close == 100.5 and bars[0].volume == 1200
    assert all(bar.bid is None and bar.ask is None and bar.bid_size is None and bar.ask_size is None for bar in bars)
    assert all(path == "/v2/stocks/bars" for path, _, _ in calls)
    assert calls[0][1]["timeframe"] == "1Min"
    assert calls[0][1]["adjustment"] == "raw"
    assert calls[0][1]["feed"] == "iex"
    assert "page_token" not in calls[0][1] and calls[1][1]["page_token"] == "next"
    assert result["bars"] == 2 and result["pages"] == 2
    assert result["observed_execution_supported"] is False
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert provenance["recorded_at"] == END.isoformat()
    assert provenance["historical_availability"] == "bar_end_approximation"
    assert any("corrections" in item for item in provenance["limitations"])
    assert "test-private" not in json.dumps(provenance)


def test_bars_only_completed_before_request_end_and_retrieval_clock(tmp_path, monkeypatch, credentials):
    monkeypatch.setattr(platforms, "_now", lambda: BASE + timedelta(seconds=45))
    responses(monkeypatch, {"bars": {"NVDA": [
        vendor_bar(BASE - timedelta(minutes=1)), vendor_bar(BASE), vendor_bar(END),
        vendor_bar(START - timedelta(minutes=1)),
    ]}})
    result = platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert result["bars"] == 1 and result["discarded_outside_completed_range"] == 3
    assert load_bars_csv(result["bars_path"])[0].timestamp == BASE


def test_requested_end_excludes_bar_that_finishes_after_end(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"bars": {"NVDA": [vendor_bar(BASE), vendor_bar(BASE + timedelta(minutes=1))]}})
    result = platforms.fetch_alpaca_bars(["NVDA"], START, BASE + timedelta(minutes=1), tmp_path)
    assert result["bars"] == 1


def test_duplicate_identical_bars_are_not_repeated(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"bars": {"NVDA": [vendor_bar(), vendor_bar()]}})
    assert platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)["bars"] == 1


def test_conflicting_bars_reject_without_writing_archive(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"bars": {"NVDA": [vendor_bar(), vendor_bar(c=100)]}})
    with pytest.raises(ValueError, match="conflicting duplicate"):
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert not (tmp_path / "bars.csv").exists()


@pytest.mark.parametrize("bad", [
    {"v": -1}, {"c": float("nan")}, {"l": 102}, {"o": True}, {"o": 0},
    {"t": "2026-10-01T13:00:00"},
])
def test_invalid_bar_values_reject(tmp_path, monkeypatch, credentials, bad):
    responses(monkeypatch, {"bars": {"NVDA": [vendor_bar(**bad)]}})
    with pytest.raises(ValueError):
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert not (tmp_path / "bars.csv").exists()


def test_unexpected_universe_stock_rejects(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"bars": {"AAPL": [vendor_bar()]}})
    with pytest.raises(ValueError, match="unexpected stock"):
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)


def test_empty_market_interval_retains_provenance(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"bars": None})
    result = platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert result["bars"] == 0
    assert (tmp_path / "bars.csv").read_text().startswith("timestamp,symbol,open")


@pytest.mark.parametrize("missing", ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"])
def test_credentials_missing_reports_names_without_existing_secret(tmp_path, monkeypatch, credentials, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError) as exc:
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert missing in str(exc.value)
    assert "test-private" not in str(exc.value)
    assert not (tmp_path / "bars.csv").exists()


@pytest.mark.parametrize("symbols,start,end,timeout", [
    ([], START, END, 15), (["NVDA?token=secret"], START, END, 15),
    (["NVDA"], "2026-10-01T12:00:00", END, 15), (["NVDA"], END, START, 15),
    (["NVDA"], START, END, True), (["NVDA"], START, END, float("inf")),
])
def test_bad_inputs_reject_before_requests(tmp_path, symbols, start, end, timeout):
    with pytest.raises(ValueError):
        platforms.fetch_alpaca_bars(symbols, start, end, tmp_path, timeout=timeout)


def test_unknown_feed_rejects(tmp_path, credentials):
    with pytest.raises(ValueError, match="feed must"):
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path, feed="unknown")


def test_existing_output_is_preserved(tmp_path, credentials):
    saved = tmp_path / "bars.csv"
    saved.write_text("original user archive")
    with pytest.raises(ValueError, match="already exist"):
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert saved.read_text() == "original user archive"


@pytest.mark.parametrize("news", [False, True])
def test_pagination_token_loops_fail_without_output(tmp_path, monkeypatch, credentials, news):
    field = "news" if news else "bars"
    contents = [vendor_news()] if news else {"NVDA": [vendor_bar()]}
    responses(monkeypatch, {field: contents, "next_page_token": "same"},
              {field: contents, "next_page_token": "same"})
    with pytest.raises(ValueError, match="repeated a pagination token"):
        if news:
            platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
        else:
            platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert not list(tmp_path.iterdir())


def test_page_limit_requires_smaller_range(tmp_path, monkeypatch, credentials):
    monkeypatch.setattr(platforms, "_MAX_PAGES", 1)
    responses(monkeypatch, {"bars": {"NVDA": [vendor_bar()]}, "next_page_token": "next"})
    with pytest.raises(ValueError, match="page limit"):
        platforms.fetch_alpaca_bars(["NVDA"], START, END, tmp_path)
    assert not list(tmp_path.iterdir())


def test_news_preserves_actual_availability_and_publication_expiry(tmp_path, monkeypatch, credentials):
    now = BASE + timedelta(minutes=5)
    monkeypatch.setattr(platforms, "_now", lambda: now)
    calls = responses(monkeypatch, {"news": [vendor_news()]})
    result = platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    events = load_events_jsonl(result["events_path"])
    event = events[0]
    assert event.published_at == BASE
    assert event.available_at == now
    assert event.expires_at == BASE + timedelta(days=1)
    assert event.symbol == "NVDA" and event.source == "https://publisher.example/article"
    assert "Published content." in event.text and "<p>" not in event.text
    assert result["historical_availability"] == "actual_retrieval_only"
    assert calls[0][0] == "/v1beta1/news" and calls[0][1]["include_content"] == "true"
    assert "test-private" not in (tmp_path / "news.jsonl").read_text()


def test_news_at_1300_cannot_inform_0900_or_unobserved_1300(tmp_path, monkeypatch, credentials):
    now = BASE + timedelta(minutes=5)
    monkeypatch.setattr(platforms, "_now", lambda: now)
    responses(monkeypatch, {"news": [vendor_news()]})
    result = platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    bars = [MarketBar(timestamp, "NVDA", 100, 101, 99, 100, 1000)
            for timestamp in (BASE - timedelta(hours=4), BASE, now)]
    frame = build_features(bars, load_events_jsonl(result["events_path"]))
    mask = frame.observed[:, frame.feature_names.index("news.decayed_count")]
    assert list(mask) == [False, False, True]


def test_old_news_archive_cannot_become_fresh_on_retrieval(tmp_path, monkeypatch, credentials):
    now = BASE + timedelta(days=10)
    monkeypatch.setattr(platforms, "_now", lambda: now)
    responses(monkeypatch, {"news": [vendor_news()]})
    result = platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    events = load_events_jsonl(result["events_path"])
    assert events[0].available_at == now and events[0].expires_at < now
    bars = [MarketBar(timestamp, "NVDA", 100, 101, 99, 100, 1000) for timestamp in (BASE, now)]
    frame = build_features(bars, events)
    assert not frame.observed[:, frame.groups["news"]].any()


def test_news_paginates_filters_tickers_and_duplicates(tmp_path, monkeypatch, credentials):
    responses(monkeypatch,
              {"news": [vendor_news(symbols=["NVDA", "AAPL"])], "next_page_token": "next"},
              {"news": [vendor_news(symbols=["NVDA", "AAPL"]), vendor_news(id=2, symbols=["TSLA"])]})
    result = platforms.fetch_alpaca_news(["NVDA", "AAPL"], START, END, tmp_path / "news.jsonl")
    assert result["events"] == 2 and result["pages"] == 2 and result["discarded_news"] == 1
    assert {event.symbol for event in load_events_jsonl(result["events_path"])} == {"NVDA", "AAPL"}


def test_news_with_future_publication_is_discarded(tmp_path, monkeypatch, credentials):
    monkeypatch.setattr(platforms, "_now", lambda: BASE)
    responses(monkeypatch, {"news": [vendor_news(BASE + timedelta(minutes=1))]})
    result = platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    assert result["events"] == 0 and result["discarded_news"] == 1


def test_news_existing_capture_is_never_overwritten(tmp_path, credentials):
    saved = tmp_path / "news.jsonl"
    saved.write_text("original capture")
    with pytest.raises(ValueError, match="already exists"):
        platforms.fetch_alpaca_news(["NVDA"], START, END, saved)
    assert saved.read_text() == "original capture"


@pytest.mark.parametrize("response", [{"error": "do not print this"}, {"news": None}])
def test_news_error_responses_do_not_become_events(tmp_path, monkeypatch, credentials, response):
    responses(monkeypatch, response)
    with pytest.raises(ValueError, match="missing its news list"):
        platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    assert not list(tmp_path.iterdir())


def test_fixed_data_host_tls_and_auth_headers(monkeypatch):
    captured = {}

    class Opener:
        def open(self, request, timeout):
            captured["request"], captured["timeout"] = request, timeout
            return io.BytesIO(b'{"bars": {}}')

    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(platforms.urllib.request, "build_opener", build)
    result = platforms._request_json("/v2/stocks/bars", {"symbols": "NVDA"}, timeout=15,
                                     credentials=("private-key", "private-secret"))
    assert result == {"bars": {}}
    assert captured["request"].full_url == "https://data.alpaca.markets/v2/stocks/bars?symbols=NVDA"
    assert captured["request"].get_header("Apca-api-key-id") == "private-key"
    tls = next(handler for handler in captured["handlers"] if isinstance(handler, platforms.urllib.request.HTTPSHandler))
    assert tls._context.verify_mode == ssl.CERT_REQUIRED and tls._context.check_hostname
    assert any(isinstance(handler, platforms._RejectRedirect) for handler in captured["handlers"])


@pytest.mark.parametrize("error", [
    urllib.error.URLError("private-key and private-secret must not be logged"),
    urllib.error.HTTPError("https://bad.example/?token=private-secret", 403,
                           "private-key echoed", {}, io.BytesIO(b"private-secret echoed body")),
    ValueError("redirect leaked private-secret"),
])
def test_request_errors_are_sanitized(monkeypatch, error):
    class Opener:
        def open(self, request, timeout):
            raise error

    monkeypatch.setattr(platforms.urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(ValueError) as exc:
        platforms._request_json("/v2/stocks/bars", {}, timeout=15, credentials=("private-key", "private-secret"))
    assert "private-key" not in str(exc.value) and "private-secret" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_redirect_rejected_without_printing_destination():
    with pytest.raises(ValueError) as exc:
        platforms._RejectRedirect().redirect_request(None, None, 302, None, {},
                                                     "https://bad.example/?token=private-secret")
    assert "private-secret" not in str(exc.value)


def test_broker_order_endpoints_are_rejected():
    with pytest.raises(ValueError, match="Unsupported read-only"):
        platforms._request_json("/v2/orders", {}, timeout=15, credentials=("key", "secret"))


def test_news_html_script_is_removed(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"news": [vendor_news(content="<script>hidden</script><p>Visible.</p>")]})
    result = platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    text = load_events_jsonl(result["events_path"])[0].text
    assert "Visible." in text and "hidden" not in text


def test_news_credential_urls_are_not_archived(tmp_path, monkeypatch, credentials):
    responses(monkeypatch, {"news": [vendor_news(url="https://example.com/?api_key=private-secret")]})
    result = platforms.fetch_alpaca_news(["NVDA"], START, END, tmp_path / "news.jsonl")
    event = load_events_jsonl(result["events_path"])[0]
    assert event.source == "https://data.alpaca.markets/v1beta1/news"
    assert "private-secret" not in (tmp_path / "news.jsonl").read_text()
