"""Collection boundaries, source timestamps, freshness and secret handling."""

import io
import json
import urllib.error
from datetime import datetime, timedelta, timezone

import pytest

from stocktrade import collection
from stocktrade.collection import CollectionConfig, CollectionSource, collect_information
from stocktrade.data import build_features, load_events_jsonl
from stocktrade.schemas import MarketBar

NOW = datetime(2024, 10, 1, 14, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr(collection, "_now", lambda: NOW)


def configuration(tmp_path, sources):
    path = tmp_path / "sources.json"
    path.write_text(json.dumps({"sources": sources}))
    return path


def news(published=NOW - timedelta(hours=1), text="NVDA new product announcement"):
    return {"published_at": published.isoformat(), "kind": "news", "text": text, "values": {}}


def source(**values):
    if values.get("type") == "json":
        values.setdefault("published_at_path", "timestamp")
    return {"name": "ticker-news", "type": "rss", "url": "https://news.example/feed/{symbol}",
            "refresh_seconds": 60, "max_age_seconds": 7200, **values}


def test_ticker_discovery_and_global_source_are_bounded_and_explicit():
    defaults = CollectionConfig.defaults()
    assert len(defaults.sources) == 2
    assert "{symbol}" in defaults.sources[0].url
    assert "{symbol}" not in defaults.sources[1].url
    assert all(item.type == "rss" for item in defaults.sources)


def test_collects_each_ticker_once_and_global_source_once(tmp_path, monkeypatch):
    calls = []

    def records(config, url, agent, symbol):
        calls.append((url, symbol))
        return [news()]

    monkeypatch.setattr(collection, "_records", records)
    config = configuration(tmp_path, [source(), source(name="global", url="https://macro.example/feed")])
    output = tmp_path / "events.jsonl"
    result = collect_information(["NVDA", "MSFT", "NVDA"], output, config)
    assert len(calls) == 3
    assert result.events_added == 3
    assert result.requested_domains == ["macro.example", "news.example"]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["symbol"] for row in rows} == {"NVDA", "MSFT", None}
    assert all(row["available_at"] == NOW.isoformat() for row in rows)
    assert all(row["expires_at"] == (NOW + timedelta(hours=1)).isoformat() for row in rows)


def test_publication_at_1300_cannot_influence_0900_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "_records", lambda *args: [news()])
    output = tmp_path / "events.jsonl"
    collect_information(["NVDA"], output, configuration(tmp_path, [source()]))
    bars = [MarketBar(timestamp=NOW.replace(hour=hour), symbol="NVDA", open=100,
                      high=101, low=99, close=100, volume=1000) for hour in (9, 13, 14, 15)]
    frame = build_features(bars, load_events_jsonl(output))
    index = frame.feature_names.index("news.decayed_count")
    assert frame.observed[:, index].tolist() == [False, False, True, False]
    assert frame.values[2, index] == 1


def test_fresh_source_cache_avoids_network_then_deduplicates_without_backdating(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collection, "_records", lambda *args: calls.append(args) or [news()])
    output = tmp_path / "events.jsonl"
    config = configuration(tmp_path, [source()])
    assert collect_information(["NVDA"], output, config).events_added == 1
    before = output.read_bytes()
    cached = collect_information(["NVDA"], output, config)
    assert cached.events_added == 0
    assert cached.source_reports[0]["status"] == "fresh_cache"
    assert cached.requested_domains == []
    assert len(calls) == 1
    monkeypatch.setattr(collection, "_now", lambda: NOW + timedelta(seconds=61))
    refreshed = collect_information(["NVDA"], output, config)
    assert refreshed.source_reports[0]["status"] == "collected"
    assert refreshed.events_added == 0
    assert len(calls) == 2
    assert output.read_bytes() == before


def test_stale_or_future_publications_are_not_recast_as_fresh(tmp_path, monkeypatch):
    rows = [news(NOW - timedelta(hours=3)), news(NOW + timedelta(minutes=1)), news()]
    monkeypatch.setattr(collection, "_records", lambda *args: rows)
    report = collect_information(["NVDA"], tmp_path / "events.jsonl", configuration(tmp_path, [source()]))
    assert report.events_added == 1
    assert report.source_reports[0]["stale_or_future_items_skipped"] == 2


def test_failed_source_does_not_prevent_other_information_or_get_cached(tmp_path, monkeypatch):
    calls = []

    def records(config, *args):
        calls.append(config.name)
        if config.name == "blocked":
            raise urllib.error.HTTPError(config.url, 403, "forbidden", {}, None)
        return [news()]

    monkeypatch.setattr(collection, "_records", records)
    config = configuration(tmp_path, [source(name="blocked"), source(name="works")])
    output = tmp_path / "events.jsonl"
    result = collect_information(["NVDA"], output, config)
    assert result.events_added == 1
    assert result.errors[0]["error"] == "source request returned HTTP 403"
    collect_information(["NVDA"], output, config)
    assert calls.count("blocked") == 2
    assert calls.count("works") == 1


def test_changed_source_policy_invalidates_the_refresh_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collection, "_records", lambda *args: calls.append(args) or [news()])
    output = tmp_path / "events.jsonl"
    config = configuration(tmp_path, [source()])
    collect_information(["NVDA"], output, config)
    configuration(tmp_path, [source(max_articles=1)])
    collect_information(["NVDA"], output, config)
    assert len(calls) == 2


def test_rss_uses_existing_verified_robot_aware_collector(tmp_path, monkeypatch):
    calls = []

    def rss(url, output, **kwargs):
        calls.append((url, kwargs))
        output.write_text(json.dumps({**news(), "available_at": (NOW - timedelta(hours=1)).isoformat(),
                                     "symbol": kwargs["symbol"], "source": url}) + "\n")
        return 1

    monkeypatch.setattr(collection, "collect_rss", rss)
    config = configuration(tmp_path, [source(user_agent="ExampleResearch/1", max_articles=4, request_timeout=8)])
    output = tmp_path / "events.jsonl"
    report = collect_information(["NVDA"], output, config)
    assert report.events_added == 1
    assert calls == [("https://news.example/feed/NVDA", {"symbol": "NVDA", "timeout": 8,
                                                      "user_agent": "ExampleResearch/1", "max_articles": 4,
                                                      "require_publication_time": True, "ttl_seconds": 7200})]
    assert json.loads(output.read_text())["available_at"] == NOW.isoformat()


class Opener:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return io.BytesIO(item)


def test_undated_rss_never_refreshes_content_into_a_new_dated_event(tmp_path, monkeypatch):
    rss = b"<rss><channel><item><title>Unchanged undated headline</title></item></channel></rss>"
    opener = Opener([b"User-agent: *\nAllow: /\n", rss])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    output = tmp_path / "events.jsonl"
    config = configuration(tmp_path, [source()])
    first = collect_information(["NVDA"], output, config)
    assert first.events_added == 0
    assert first.source_reports[0]["note"] == "no eligible dated fresh items"
    monkeypatch.setattr(collection, "_now", lambda: NOW + timedelta(days=1))
    opener.responses.extend([b"User-agent: *\nAllow: /\n", rss])
    second = collect_information(["NVDA"], output, config)
    assert second.events_added == 0
    assert second.source_reports[0]["note"] == "no eligible dated fresh items"
    assert len(opener.requests) == 4
    assert output.read_text() == ""


def test_html_article_respects_robots_and_preserves_publication(tmp_path, monkeypatch):
    opener = Opener([b"User-agent: *\nDisallow: /private\n", b"""<html><head>
        <meta property="article:published_time" content="2024-10-01T13:00:00Z">
        </head><body><nav>ignore navigation</nav><article>Company announces a new chip.
        <script>ignore code</script></article></body></html>"""])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="html", url="https://news.example/article")])
    output = tmp_path / "events.jsonl"
    assert collect_information(["NVDA"], output, config).events_added == 1
    row = json.loads(output.read_text())
    assert row["text"] == "Company announces a new chip."
    assert row["published_at"] == "2024-10-01T13:00:00+00:00"
    assert row["available_at"] == NOW.isoformat()
    assert row["symbol"] is None
    assert [request.full_url for request, _ in opener.requests] == ["https://news.example/robots.txt",
                                                                  "https://news.example/article"]


@pytest.mark.parametrize("response", [b"User-agent: *\nDisallow: /\n", urllib.error.URLError("unreachable")])
def test_html_with_denied_or_unknown_robot_permission_never_fetches_article(tmp_path, monkeypatch, response):
    opener = Opener([response])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="html", url="https://news.example/article")])
    result = collect_information(["NVDA"], tmp_path / "events.jsonl", config)
    assert len(result.errors) == 1
    assert len(opener.requests) == 1


def test_json_snapshot_uses_env_headers_and_source_time_not_collection_as_publication(tmp_path, monkeypatch):
    secret = "synthetic-secret-do-not-output"
    monkeypatch.setenv("PROVIDER_SECRET", secret)
    payload = {"latestTrade": {"p": 225.0, "t": (NOW - timedelta(seconds=30)).isoformat()}, "unrelated": 777}
    opener = Opener([json.dumps(payload).encode()])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="json", headers_env={"Authorization": "PROVIDER_SECRET"},
                                           metrics={"last_price": "latestTrade.p"},
                                           published_at_path="latestTrade.t", max_age_seconds=60)])
    output = tmp_path / "events.jsonl"
    result = collect_information(["NVDA"], output, config)
    assert result.events_added == 1
    row = json.loads(output.read_text())
    assert row["kind"] == "platform"
    assert row["values"] == {"last_price": 225}
    assert row["available_at"] == NOW.isoformat()
    assert row["expires_at"] == (NOW + timedelta(seconds=30)).isoformat()
    assert opener.requests[0][0].get_header("Authorization") == secret
    assert secret not in json.dumps(result.to_dict()) + output.read_text() + config.read_text()
    assert secret not in output.with_name(output.name + ".collection-state.json").read_text()


def test_undated_json_snapshot_cannot_refresh_itself_each_collection(tmp_path, monkeypatch):
    opener = Opener([b'{"ratio": 18.5}'])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="json", metrics={"trailing_pe": "ratio"})])
    output = tmp_path / "events.jsonl"
    first = collect_information(["NVDA"], output, config)
    assert first.events_added == 0
    assert first.errors[0]["error"] == "source publication time is unknown"
    assert not output.exists()
    assert not output.with_name(output.name + ".collection-state.json").exists()
    monkeypatch.setattr(collection, "_now", lambda: NOW + timedelta(days=1))
    opener.responses.append(b'{"ratio": 18.5}')
    second = collect_information(["NVDA"], output, config)
    assert second.events_added == 0
    assert second.errors[0]["error"] == "source publication time is unknown"
    assert len(opener.requests) == 2
    assert not output.exists()


def test_json_source_requires_an_explicit_publication_path():
    with pytest.raises(ValueError, match="require published_at_path"):
        CollectionSource(name="platform", type="json", url="https://news.example/snapshot",
                         metrics={"last_price": "price"})


def test_undated_html_article_never_becomes_fresh_on_a_later_day(tmp_path, monkeypatch):
    article = b"<html><article>Unchanged content without a publication date.</article></html>"
    opener = Opener([b"User-agent: *\nAllow: /\n", article])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="html", url="https://news.example/article")])
    output = tmp_path / "events.jsonl"
    first = collect_information(["NVDA"], output, config)
    assert first.events_added == 0
    assert first.errors[0]["error"] == "source publication time is unknown"
    monkeypatch.setattr(collection, "_now", lambda: NOW + timedelta(days=1))
    opener.responses.extend([b"User-agent: *\nAllow: /\n", article])
    second = collect_information(["NVDA"], output, config)
    assert second.events_added == 0
    assert second.errors[0]["error"] == "source publication time is unknown"
    assert len(opener.requests) == 4
    assert not output.exists()


@pytest.mark.parametrize("timestamp", ["2024-10-01T13:00:00", "not a date", 1727787600])
def test_platform_publication_is_validated_instead_of_replaced_with_now(tmp_path, monkeypatch, timestamp):
    opener = Opener([json.dumps({"price": 225, "timestamp": timestamp}).encode()])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="json", metrics={"last_price": "price"})])
    output = tmp_path / "events.jsonl"
    report = collect_information(["NVDA"], output, config)
    assert report.events_added == 0
    assert len(report.errors) == 1
    assert not output.exists()


def test_missing_api_secret_reports_name_without_requesting_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_PROVIDER_SECRET", raising=False)
    opener = Opener([])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="json", headers_env={"Authorization": "MISSING_PROVIDER_SECRET"},
                                           metrics={"last_price": "price"})])
    result = collect_information(["NVDA"], tmp_path / "events.jsonl", config)
    assert result.errors[0]["error"] == "missing environment variable MISSING_PROVIDER_SECRET"
    assert opener.requests == []


@pytest.mark.parametrize("value", ["225", True, float("nan"), [225]])
def test_json_metric_is_a_number_not_coerced_or_invented(tmp_path, monkeypatch, value):
    opener = Opener([json.dumps({"price": value}).encode()])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="json", metrics={"last_price": "price"})])
    result = collect_information(["NVDA"], tmp_path / "events.jsonl", config)
    assert result.events_added == 0
    assert "must be a finite JSON number" in result.errors[0]["error"]


def test_oversized_http_response_is_rejected(tmp_path, monkeypatch):
    opener = Opener([b"x" * (collection._MAX_BYTES + 1)])
    monkeypatch.setattr(collection.urllib.request, "build_opener", lambda *args: opener)
    config = configuration(tmp_path, [source(type="json", metrics={"last_price": "price"})])
    result = collect_information(["NVDA"], tmp_path / "events.jsonl", config)
    assert "2 MB" in result.errors[0]["error"]


@pytest.mark.parametrize("url", ["http://news.example/rss", "https://token:secret@news.example/rss",
                                 "https://news.example/rss?api_key=secret", "https://localhost/rss",
                                 "https://127.0.0.1/rss", "https://news.example/{date}/rss"])
def test_source_url_requires_public_https_without_embedded_secrets(url):
    with pytest.raises(ValueError):
        CollectionSource(name="bad", url=url)


@pytest.mark.parametrize("kwargs", [{"type": "json", "metrics": {"invented_edge": "x"}},
                                   {"request_timeout": 61}, {"refresh_seconds": 0},
                                   {"max_articles": True}, {"max_age_seconds": -1},
                                   {"headers_env": {"Authorization": "literal secret"}},
                                   {"headers_env": {"Host": "ENV_NAME"}}, {"user_agent": "bad\nagent"}])
def test_invalid_source_requirements_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CollectionSource(name="bad", url="https://news.example/rss", **kwargs)


def test_example_config_is_valid_and_authenticated_api_is_disabled():
    config = CollectionConfig.from_json("config/sources.example.json")
    api = next(item for item in config.sources if item.type == "json")
    assert not api.enabled
    assert api.metrics == {"last_price": "latestTrade.p"}


def test_existing_invalid_archive_is_preserved_and_not_appended(tmp_path):
    output = tmp_path / "events.jsonl"
    output.write_text("invalid original file\n")
    with pytest.raises(ValueError, match="invalid event"):
        collect_information(["NVDA"], output)
    assert output.read_text() == "invalid original file\n"


@pytest.mark.parametrize("symbols", ["NVDA", [], ["nvda"], ["NVDA?api_key=secret"], [["NVDA"]]])
def test_invalid_universe_is_rejected_before_any_network(symbols):
    with pytest.raises(ValueError, match="symbols|ticker"):
        collect_information(symbols, "/tmp/unused-stocktrade-collection.jsonl")


def test_no_api_to_backdate_live_collection():
    with pytest.raises(TypeError, match="now"):
        collect_information(["NVDA"], "/tmp/unused-stocktrade-collection.jsonl", now=NOW)
