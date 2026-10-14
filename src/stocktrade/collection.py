"""Bounded, autonomous information collection for a live decision.

The policy may refresh configured sources for its allowed ticker universe.
Collection never supplies historical availability: every new observation is
available at the actual collection time, even if its publication predates it.
Only public HTTPS RSS/article sources and explicitly configured JSON APIs are
supported. This module contains no broker order or unrestricted browsing tool.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

from .data import PLATFORM_VALUE_KEYS, _SameOriginHTTPSRedirect, collect_rss

_MAX_BYTES = 2_000_000
_COLLECTION_POLICY_VERSION = 2
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.\-^]{0,19}$")
_CREDENTIAL_QUERY_KEYS = {"api_key", "apikey", "token", "access_token", "secret", "password", "key"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source publication time must be a nonempty timestamp string")
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        try:
            result = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("source publication time is not a supported timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("source publication time must include a UTC offset")
    return result.astimezone(timezone.utc)


def _url(value: str) -> str:
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise ValueError("source URL must be a valid HTTPS URL")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("source URL must be HTTPS without credentials or a fragment")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname == "localhost" or parsed.hostname.endswith((".local", ".localhost")):
            raise ValueError("source URL must identify a public source")
    else:
        if not address.is_global:
            raise ValueError("source URL must identify a public source")
    if any(key.lower() in _CREDENTIAL_QUERY_KEYS for key, _ in urllib.parse.parse_qsl(parsed.query)):
        raise ValueError("source credentials belong in environment-backed headers, never URL parameters")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("source URL has an invalid port") from exc
    return value


def _number(value: object, label: str, *, maximum: float, integer: bool = False) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite positive number")
    if not 0 < value <= maximum or integer and not isinstance(value, int):
        raise ValueError(f"{label} must be {'an integer' if integer else 'a number'} in (0, {maximum}]")
    return value


@dataclass(frozen=True)
class CollectionSource:
    name: str
    url: str
    type: str = "rss"
    provider: str = ""
    enabled: bool = True
    refresh_seconds: float = 300
    max_age_seconds: float = 86400
    max_articles: int = 50
    request_timeout: float = 15
    user_agent: str | None = None
    headers_env: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, str] = field(default_factory=dict)
    published_at_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
            raise ValueError("source name must be a nonempty string of at most 100 characters")
        if not isinstance(self.provider, str) or len(self.provider) > 100:
            raise ValueError("source provider must be a string of at most 100 characters")
        if self.type not in {"rss", "html", "json"}:
            raise ValueError("source type must be rss, html or json")
        if not isinstance(self.enabled, bool):
            raise ValueError("source enabled must be a boolean")
        if not isinstance(self.url, str):
            raise ValueError("source URL must be a valid HTTPS URL")
        _url(self.url.replace("{symbol}", "NVDA"))
        if "{" in self.url.replace("{symbol}", "") or "}" in self.url.replace("{symbol}", ""):
            raise ValueError("source URL supports only the {symbol} template")
        _number(self.refresh_seconds, "refresh_seconds", maximum=31 * 86400)
        _number(self.max_age_seconds, "max_age_seconds", maximum=31 * 86400)
        _number(self.max_articles, "max_articles", maximum=1000, integer=True)
        _number(self.request_timeout, "request_timeout", maximum=60)
        if self.user_agent is not None:
            _user_agent(self.user_agent)
        if not isinstance(self.headers_env, dict):
            raise ValueError("headers_env must map header names to environment variable names")
        for header, env_name in self.headers_env.items():
            if (not isinstance(header, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", header)
                    or header.lower() in {"host", "cookie", "content-length", "user-agent"}
                    or not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name)):
                raise ValueError("headers_env contains an unsupported header or environment variable name")
        if self.headers_env and self.type != "json":
            raise ValueError("environment-backed authentication is supported only for JSON APIs")
        if not isinstance(self.metrics, dict):
            raise ValueError("metrics must map canonical platform metric names to JSON paths")
        if any(key not in PLATFORM_VALUE_KEYS or not _valid_path(path) for key, path in self.metrics.items()):
            raise ValueError("metrics must use canonical platform names and nonempty dot-separated JSON paths")
        if self.type == "json" and not self.metrics:
            raise ValueError("JSON sources require at least one canonical platform metric")
        if self.type == "json" and self.published_at_path is None:
            raise ValueError("JSON sources require published_at_path for a verified source timestamp")
        if self.type != "json" and (self.metrics or self.published_at_path is not None):
            raise ValueError("metrics and published_at_path apply only to JSON sources")
        if self.published_at_path is not None and not _valid_path(self.published_at_path):
            raise ValueError("published_at_path must be a nonempty dot-separated JSON path")


def _user_agent(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError("a valid collector User-Agent is required")
    return value


def _valid_path(value: object) -> bool:
    return isinstance(value, str) and bool(value) and all(part for part in value.split("."))


@dataclass(frozen=True)
class CollectionConfig:
    sources: tuple[CollectionSource, ...]
    user_agent: str = "StockTradeResearch/0.3"

    def __post_init__(self) -> None:
        _user_agent(self.user_agent)
        if not 1 <= len(self.sources) <= 25 or any(not isinstance(source, CollectionSource) for source in self.sources):
            raise ValueError("collection requires 1 to 25 configured sources")
        if len({source.name for source in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")

    @classmethod
    def from_json(cls, path: str | Path) -> CollectionConfig:
        """Read an explicit source allowlist; secret values are never configuration."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) - {"sources", "user_agent"}:
            raise ValueError("collection configuration supports only sources and user_agent")
        if not isinstance(raw.get("sources"), list):
            raise ValueError("collection configuration requires a sources list")
        try:
            sources = tuple(CollectionSource(**row) for row in raw["sources"])
        except TypeError as exc:
            raise ValueError("source configuration has unsupported fields or missing required fields") from exc
        return cls(sources=sources, user_agent=raw.get("user_agent", "StockTradeResearch/0.3"))

    @classmethod
    def defaults(cls) -> CollectionConfig:
        """Discover ticker-specific headlines and global monetary-policy news."""
        return cls(sources=(
            CollectionSource(name="yahoo-ticker-news", provider="Yahoo Finance", max_age_seconds=86400,
                             url="https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"),
            CollectionSource(name="fed-policy-news", provider="Federal Reserve", refresh_seconds=900,
                             max_age_seconds=7 * 86400,
                             url="https://www.federalreserve.gov/feeds/press_monetary.xml"),
        ))


@dataclass
class CollectionReport:
    events_added: int = 0
    source_reports: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    requested_domains: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _read(opener, url: str, user_agent: str, timeout: float, *, headers: dict | None = None) -> bytes:
    request_headers = {"User-Agent": user_agent, "Accept": "application/json,text/html,application/xml"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(_MAX_BYTES + 1)
    if len(payload) > _MAX_BYTES:
        raise ValueError("source response exceeds the 2 MB request limit")
    return payload


def _fetch(source: CollectionSource, url: str, user_agent: str) -> bytes:
    redirect = _SameOriginHTTPSRedirect()
    opener = urllib.request.build_opener(redirect)
    if source.type == "html":
        parsed = urllib.parse.urlparse(url)
        robots_url = urllib.parse.urlunparse(("https", parsed.netloc, "/robots.txt", "", "", ""))
        robots = urllib.robotparser.RobotFileParser(robots_url)
        try:
            robots.parse(_read(opener, robots_url, user_agent, source.request_timeout)
                         .decode("utf-8", errors="replace").splitlines())
        except urllib.error.HTTPError as exc:
            if exc.code not in {404, 410}:
                raise PermissionError(f"cannot establish robots permission: HTTP {exc.code}") from exc
            robots.parse([])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PermissionError("cannot establish robots permission; article was not requested") from exc
        if not robots.can_fetch(user_agent, url):
            raise PermissionError("robots.txt disallows this article")
        redirect.robots, redirect.user_agent = robots, user_agent
        delay = float(robots.crawl_delay(user_agent) or 0)
        rate = robots.request_rate(user_agent)
        if rate is not None:
            delay = max(delay, rate.seconds / rate.requests)
        if delay > 60:
            raise PermissionError("robots delay requires a separately scheduled collector")
        if delay:
            time.sleep(delay)
    # JSON APIs are explicitly configured service endpoints, rather than a
    # website crawl. Their authentication remains in runtime-injected headers.
    headers = {}
    for header, env_name in source.headers_env.items():
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"missing environment variable {env_name}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"invalid header binding in environment variable {env_name}")
        headers[header] = value
    return _read(opener, url, user_agent, source.request_timeout, headers=headers)


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.article_parts: list[str] = []
        self.skip = 0
        self.article = 0
        self.published: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag in {"script", "style", "noscript", "nav", "footer", "header"}:
            self.skip += 1
        if tag in {"article", "main"}:
            self.article += 1
        if tag == "meta":
            name = (attributes.get("property") or attributes.get("name") or "").lower()
            if name in {"article:published_time", "datepublished", "pubdate", "publication_date"}:
                self.published = attributes.get("content") or self.published
        if tag == "time" and not self.published:
            self.published = attributes.get("datetime")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "nav", "footer", "header"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"article", "main"}:
            self.article = max(0, self.article - 1)

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data.strip())
            if self.article:
                self.article_parts.append(data.strip())


def _path(payload: object, path: str) -> object:
    value = payload
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def _records(source: CollectionSource, url: str, user_agent: str, symbol: str | None) -> list[dict]:
    if source.type == "rss":
        with tempfile.TemporaryDirectory(prefix="stocktrade-rss-") as directory:
            temporary = Path(directory) / "events.jsonl"
            collect_rss(url, temporary, symbol=symbol, timeout=source.request_timeout,
                        user_agent=user_agent, max_articles=source.max_articles, require_publication_time=True,
                        ttl_seconds=source.max_age_seconds)
            return [json.loads(line) for line in temporary.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = _fetch(source, url, user_agent)
    if source.type == "html":
        article = _ArticleParser()
        article.feed(payload.decode("utf-8", errors="replace"))
        text = " ".join(article.article_parts or article.parts)[:120_000]
        if not text:
            raise ValueError("article contains no readable text")
        if not article.published:
            raise ValueError("source publication time is unknown")
        published = _timestamp(article.published)
        return [{"published_at": published.isoformat(), "kind": "news", "values": {}, "text": text}]
    try:
        parsed = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("platform response is not valid JSON") from exc
    values = {}
    for key, path in source.metrics.items():
        value = _path(parsed, path)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"platform metric {key} must be a finite JSON number")
        values[key] = float(value)
    if not values:
        raise ValueError("platform response contains none of the configured numeric metrics")
    publication = _path(parsed, source.published_at_path)
    if publication is None:
        raise ValueError("source publication time is unknown")
    published = _timestamp(publication)
    return [{"published_at": published.isoformat(), "kind": "platform", "values": values, "text": ""}]


def _fingerprint(row: dict) -> str:
    # Availability/expiry are deliberately absent: rereading the same article
    # may neither refresh its age nor rewrite when we first observed it.
    fields = {key: row.get(key) for key in ("source", "symbol", "kind", "published_at", "text", "values")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            if raw.strip():
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise ValueError("event must be an object")
                    seen.add(_fingerprint(row))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"existing collection archive has an invalid event at line {line}") from exc
    return seen


def _error(exc: Exception) -> str:
    # Do not include request headers, remote bodies or arbitrary exception
    # reprs: credentials can otherwise be exposed by client validation errors.
    if isinstance(exc, urllib.error.HTTPError):
        return f"source request returned HTTP {exc.code}"
    if isinstance(exc, (ValueError, PermissionError)):
        return str(exc)
    if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)):
        return "source request failed; check network access and source availability"
    return "source collection failed"


def collect_information(symbols, output_path: str | Path, config_path: str | Path | None = None) -> CollectionReport:
    """Refresh relevant sources and append newly observed, expiring events.

    Invoke for present-time decisions only. Historical training/evaluation uses
    archived events and must never call this to recreate past information.
    There is intentionally no caller-supplied timestamp or historical mode.
    Failed sources are reported and do not prevent other sources from updating.
    """
    if isinstance(symbols, str):
        raise ValueError("symbols must be a collection of ticker strings")
    supplied = list(symbols)
    if not supplied or any(not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol) for symbol in supplied):
        raise ValueError("collection requires 1 to 100 valid uppercase ticker symbols")
    universe = list(dict.fromkeys(supplied))
    if len(universe) > 100:
        raise ValueError("collection requires 1 to 100 valid uppercase ticker symbols")
    config = CollectionConfig.from_json(config_path) if config_path is not None else CollectionConfig.defaults()
    endpoint_count = sum(len(universe) if "{symbol}" in source.url else 1
                         for source in config.sources if source.enabled)
    if endpoint_count > 1000:
        raise ValueError("collection is limited to 1000 configured source/ticker requests")
    destination = Path(output_path)
    seen = _existing(destination)
    state_path = destination.with_name(destination.name + ".collection-state.json")
    state = {}
    if destination.exists() and state_path.exists():
        try:
            candidate = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                state = candidate
        except (ValueError, OSError):
            pass  # A damaged cache only causes a conservative fresh collection.
    report = CollectionReport()
    domains = set()
    for source in config.sources:
        if not source.enabled:
            continue
        for symbol in universe if "{symbol}" in source.url else [None]:
            url = _url(source.url.replace("{symbol}", urllib.parse.quote(symbol or "", safe="")))
            source_id = hashlib.sha256(json.dumps({"policy_version": _COLLECTION_POLICY_VERSION,
                                                  "source": asdict(source), "url": url},
                                                 sort_keys=True).encode()).hexdigest()
            observation = {"name": source.name, "provider": source.provider,
                           "symbol": symbol, "domain": urllib.parse.urlparse(url).hostname}
            last = state.get(source_id)
            if isinstance(last, str):
                try:
                    age = (_now() - _timestamp(last)).total_seconds()
                except ValueError:
                    age = -1
                if 0 <= age < min(source.refresh_seconds, source.max_age_seconds):
                    report.source_reports.append({**observation, "status": "fresh_cache", "events_added": 0})
                    continue
            domains.add(observation["domain"])
            try:
                rows = _records(source, url, source.user_agent or config.user_agent, symbol)
                collected = _now()
                additions = []
                stale = 0
                eligible = 0
                for row in rows[:source.max_articles]:
                    published = _timestamp(row["published_at"])
                    # TTL uses the earlier of publication and observation. An
                    # old article newly retrieved today does not become fresh.
                    expires = min(published, collected) + timedelta(seconds=source.max_age_seconds)
                    if published > collected or expires <= collected:
                        stale += 1
                        continue
                    eligible += 1
                    normalized = {"published_at": published.isoformat(), "available_at": collected.isoformat(),
                                  "expires_at": expires.isoformat(), "source": url,
                                  "kind": row["kind"], "values": row.get("values", {}),
                                  "text": row.get("text", ""), "symbol": symbol}
                    identity = _fingerprint(normalized)
                    if identity not in seen:
                        additions.append(normalized)
                        seen.add(identity)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("a", encoding="utf-8") as stream:
                    for row in additions:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                state[source_id] = collected.isoformat()
                report.events_added += len(additions)
                source_report = {**observation, "status": "collected", "events_added": len(additions),
                                 "stale_or_future_items_skipped": stale}
                if not eligible:
                    source_report["note"] = "no eligible dated fresh items"
                report.source_reports.append(source_report)
            except (ValueError, PermissionError, urllib.error.URLError, TimeoutError, OSError) as exc:
                failure = {**observation, "error": _error(exc)}
                report.errors.append(failure)
                report.source_reports.append({**failure, "status": "failed", "events_added": 0})
    if destination.exists():
        state_path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    report.requested_domains = sorted(domains)
    return report
