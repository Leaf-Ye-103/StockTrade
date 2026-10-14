"""Point-in-time data ingestion and causal multimodal feature construction.

CSV bars must have UTC offsets in their timestamps. Information is usable only
after both its publication time and the time at which the dataset received it.
Quote columns describe a top-of-book snapshot; they never imply queue priority.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from .schemas import FeatureFrame, InformationEvent, MarketBar

NEWS_HASH_DIM = 16
PLATFORM_VALUE_KEYS = (
    "last_price", "market_cap", "trailing_pe", "forward_pe", "price_to_book",
    "dividend_yield", "analyst_target_price", "analyst_recommendation",
    "short_float_fraction", "short_interest_ratio", "implied_volatility",
    "borrow_rate", "borrow_available", "retail_sentiment", "discussion_volume",
)
PLATFORM_METRICS = PLATFORM_VALUE_KEYS
DEFAULT_INFORMATION_MAX_AGE_SECONDS = {"orderflow": 60.0, "platform": 3600.0}
_EVENT_KINDS = {"macro", "market", "fundamentals", "news", "orderflow", "platform"}
_BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
_QUOTE_COLUMNS = ("bid", "ask", "bid_size", "ask_size")
_MAX_RESPONSE_BYTES = 2_000_000


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO 8601 timestamp with a UTC offset") from exc
    return _utc(result, label)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _validate_bar(bar: MarketBar) -> None:
    _utc(bar.timestamp, "bar timestamp")
    if not isinstance(bar.symbol, str) or not bar.symbol.strip():
        raise ValueError("bar symbol must be a nonempty string")
    prices = {name: _finite(getattr(bar, name), name) for name in ("open", "high", "low", "close")}
    if min(prices.values()) <= 0:
        raise ValueError("OHLC prices must be positive")
    if prices["high"] < max(prices.values()) or prices["low"] > min(prices.values()):
        raise ValueError("bar high/low must enclose open and close")
    if _finite(bar.volume, "volume") < 0:
        raise ValueError("bar volume must be nonnegative")
    for name in _QUOTE_COLUMNS:
        value = getattr(bar, name)
        if value is not None:
            number = _finite(value, name)
            if number < 0 or (name in {"bid", "ask"} and number == 0):
                raise ValueError(f"{name} must be {'positive' if name in {'bid', 'ask'} else 'nonnegative'}")
    if bar.bid is not None and bar.ask is not None and bar.bid > bar.ask:
        raise ValueError("bid cannot exceed ask")


def _validate_event(event: InformationEvent) -> None:
    _utc(event.published_at, "published_at")
    _utc(event.available_at, "available_at")
    if event.expires_at is not None:
        if _utc(event.expires_at, "expires_at") <= event.published_at:
            raise ValueError("expires_at must be later than published_at")
    if event.kind not in _EVENT_KINDS:
        raise ValueError(f"unsupported information kind {event.kind!r}; expected one of {sorted(_EVENT_KINDS)}")
    if not isinstance(event.source, str) or not event.source.strip():
        raise ValueError("event source must be a nonempty string")
    if event.symbol is not None and (not isinstance(event.symbol, str) or not event.symbol.strip()):
        raise ValueError("event symbol must be a nonempty string or null")
    if not isinstance(event.text, str) or not isinstance(event.values, dict):
        raise ValueError("event text must be a string and values must be an object")
    for key, value in event.values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("event value names must be nonempty strings")
        if value is not None:
            _finite(value, f"event value {key}")


def load_bars_csv(path: str | Path) -> list[MarketBar]:
    """Load observed bars; optional quote fields remain missing when absent.

    Timestamps denote when the complete bar is available (normally its closing
    boundary). A feed with bar-open timestamps must shift them before ingestion.
    """
    result: list[MarketBar] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(_BAR_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"bars CSV is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            try:
                numbers = {name: _finite(row[name], name) for name in _BAR_COLUMNS[2:]}
                quotes = {
                    name: None if row.get(name) in (None, "") else _finite(row[name], name)
                    for name in _QUOTE_COLUMNS
                }
                bar = MarketBar(timestamp=_parse_timestamp(row["timestamp"], "timestamp"),
                                symbol=row["symbol"].strip(), **numbers, **quotes)
                _validate_bar(bar)
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"invalid bar at CSV line {line}: {exc}") from exc
            result.append(bar)
    if not result:
        raise ValueError("bars CSV contains no bars")
    return result


def load_events_jsonl(path: str | Path) -> list[InformationEvent]:
    """Load news, macro/benchmark, and fundamental releases with provenance.

    Example kinds are ``macro`` (policy_rate), ``market`` (benchmark_return),
    ``fundamentals`` (revenue), ``news`` (text), ``platform`` (investment-platform
    snapshots), and ``orderflow`` (provider-measured L2 depth, queue, or trade-flow
    aggregates). ``available_at`` is required:
    publication timestamps alone do not establish historical data availability.
    Order-flow measurements require a real source and retain that source's
    aggregation interval; OHLCV is never used to reconstruct queue priority.
    """
    result: list[InformationEvent] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                event = InformationEvent(
                    published_at=_parse_timestamp(row["published_at"], "published_at"),
                    available_at=_parse_timestamp(row["available_at"], "available_at"),
                    source=row["source"], kind=row["kind"], values=row.get("values", {}),
                    text=row.get("text", ""), symbol=row.get("symbol"),
                    expires_at=(_parse_timestamp(row["expires_at"], "expires_at")
                                if row.get("expires_at") is not None else None),
                )
                _validate_event(event)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid information event at JSONL line {line}: {exc}") from exc
            result.append(event)
    return result


def _hashed_text(text: str) -> np.ndarray:
    """Stable signed feature hashing; no fitting on future articles is needed."""
    vector = np.zeros(NEWS_HASH_DIM, dtype=np.float64)
    tokens = re.findall(r"\b[\w'-]+\b", text.casefold())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest[:4], "little") % NEWS_HASH_DIM] += 1 if digest[4] & 1 else -1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _event_column(event: InformationEvent, key: str) -> str:
    # Benchmark releases belong to macro, and cannot shadow OHLCV columns.
    if event.kind == "market":
        return f"macro.market.{key}"
    if event.kind == "orderflow":
        return f"orderbook.flow.{key}"
    return f"{event.kind}.{key}"


def build_features(bars: Iterable[MarketBar], events: Iterable[InformationEvent], *,
                   feature_names: list[str] | None = None,
                   information_max_age_seconds: dict[str, float] | None = None,
                   as_of: datetime | None = None) -> FeatureFrame:
    """Construct single-symbol features using only information available then.

    Feature values are finite and missing entries are zero with ``observed=False``.
    No queue depth, trade direction, or fill availability is inferred from OHLCV.
    News decays over six hours from *availability*, avoiding stale-news timing
    leakage. Explicit ``expires_at`` boundaries invalidate every information
    kind. Order-flow/platform snapshots expire after at most 60/3600 seconds
    from publication, even when their explicit expiry is later; collection cannot
    renew an old measurement. ``information_max_age_seconds`` can override these
    defaults or set a fallback TTL for other kinds. Macro/fundamentals otherwise
    persist until a newer known release. Providers must timestamp order-flow
    observations at their aggregation end, and explicitly supply subsequent
    measurements; these inputs cannot establish historical fills by themselves.

    Freeze ``feature_names`` from the training prefix for later evaluation. The
    exact requested order is retained, additional event keys are ignored, and
    requested columns absent from this dataset contain zero with a false mask.

    For a current decision, ``as_of`` may advance only the final feature clock
    beyond the latest completed bar. Raw bar timestamps, OHLCV, returns, and
    prices remain unchanged; no synthetic market observation is appended.
    Historical training/evaluation should omit this current-clock override.
    """
    max_ages = dict(DEFAULT_INFORMATION_MAX_AGE_SECONDS)
    if information_max_age_seconds is not None:
        if not isinstance(information_max_age_seconds, dict):
            raise ValueError("information_max_age_seconds must be a kind-to-seconds dictionary")
        for kind, seconds in information_max_age_seconds.items():
            if kind not in _EVENT_KINDS or _finite(seconds, "information max age") <= 0:
                raise ValueError("information max ages must use recognized kinds and positive finite seconds")
            max_ages[kind] = float(seconds)

    def expiry_for(event: InformationEvent) -> datetime | None:
        if event.expires_at is not None:
            explicit = _utc(event.expires_at, "expires_at")
            if event.kind in {"orderflow", "platform"}:
                maximum = _utc(event.published_at, "published_at") + timedelta(seconds=max_ages[event.kind])
                return min(explicit, maximum)
            return explicit
        if event.kind in max_ages:
            return _utc(event.published_at, "published_at") + timedelta(seconds=max_ages[event.kind])
        return None

    frozen_names = None
    allowed_groups = {"market", "orderbook", "macro", "fundamentals", "news", "platform"}
    if feature_names is not None:
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("feature_names must be a nonempty list")
        frozen_names = list(feature_names)
        if any(not isinstance(name, str) or not name.partition(".")[2]
               or name.partition(".")[0] not in allowed_groups for name in frozen_names):
            raise ValueError("feature_names must have a recognized group prefix and nonempty name")
        if len(set(frozen_names)) != len(frozen_names):
            raise ValueError("feature_names must be unique")
    frozen_set = set(frozen_names) if frozen_names is not None else None
    bar_list = list(bars)
    event_list = list(events)
    if not bar_list:
        raise ValueError("at least one bar is required")
    for bar in bar_list:
        _validate_bar(bar)
    for event in event_list:
        _validate_event(event)
    symbols = {bar.symbol for bar in bar_list}
    if len(symbols) != 1:
        raise ValueError("build_features accepts one symbol at a time")
    symbol = bar_list[0].symbol
    bar_list.sort(key=lambda bar: bar.timestamp)
    times = [_utc(bar.timestamp, "bar timestamp") for bar in bar_list]
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("bar timestamps must be unique for the symbol")
    if as_of is not None:
        decision_time = _utc(as_of, "as_of")
        if decision_time < times[-1]:
            raise ValueError("as_of cannot precede the latest completed bar")
        times[-1] = decision_time
    applicable = [event for event in event_list if event.symbol in (None, symbol)]
    applicable.sort(key=lambda event: (max(event.published_at, event.available_at), event.published_at))

    names: list[str] = []
    groups: dict[str, list[int]] = {}

    def register(group: str, columns: list[str]) -> None:
        groups[group] = list(range(len(names), len(names) + len(columns)))
        names.extend(columns)

    register("market", ["market.log_return", "market.open_gap", "market.log_range",
                        "market.close_position", "market.log_volume", "market.log_volume_change"])
    orderflow_keys = sorted({_event_column(event, key) for event in applicable if event.kind == "orderflow"
                            for key in event.values if frozen_set is None or _event_column(event, key) in frozen_set})
    register("orderbook", ["orderbook.relative_spread", "orderbook.imbalance",
                           "orderbook.log_bid_size", "orderbook.log_ask_size"] + orderflow_keys)
    macro_keys = sorted({_event_column(event, key) for event in applicable if event.kind in {"macro", "market"}
                         for key in event.values if frozen_set is None or _event_column(event, key) in frozen_set})
    fundamental_keys = sorted({f"fundamentals.{key}" for event in applicable if event.kind == "fundamentals"
                               for key in event.values if frozen_set is None or f"fundamentals.{key}" in frozen_set})
    register("macro", macro_keys)
    register("fundamentals", fundamental_keys)
    register("news", [f"news.hash_{index:02d}" for index in range(NEWS_HASH_DIM)]
                       + ["news.decayed_count", "news.log_seconds_since_latest"])
    # These canonical fields exist before any collection, keeping checkpoints
    # ready for newly observed platform data without discovering future keys.
    platform_keys = {f"platform.{key}" for key in PLATFORM_VALUE_KEYS}
    platform_keys.update(_event_column(event, key) for event in applicable if event.kind == "platform"
                         for key in event.values if frozen_set is None or _event_column(event, key) in frozen_set)
    register("platform", sorted(platform_keys))
    columns = {name: index for index, name in enumerate(names)}
    values = np.zeros((len(bar_list), len(names)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=bool)
    persistent: dict[tuple[str, str], tuple[datetime, datetime, datetime | None, float | None]] = {}
    news_vector = np.zeros(NEWS_HASH_DIM, dtype=np.float64)
    news_count = 0.0
    active_news: dict[int, tuple[datetime, np.ndarray]] = {}
    news_expirations: list[tuple[datetime, int]] = []
    news_latest: list[tuple[float, int]] = []
    cursor = 0

    for row, (bar, timestamp) in enumerate(zip(bar_list, times)):
        def put(name: str, value: float) -> None:
            if name not in columns:
                return
            values[row, columns[name]] = value
            observed[row, columns[name]] = True

        if row:
            previous = bar_list[row - 1]
            news_decay = math.exp(-(timestamp - times[row - 1]).total_seconds() / (6 * 3600))
            news_vector *= news_decay
            news_count *= news_decay
            put("market.log_return", math.log(bar.close / previous.close))
            put("market.open_gap", math.log(bar.open / previous.close))
            put("market.log_volume_change", math.log1p(bar.volume) - math.log1p(previous.volume))
        put("market.log_range", math.log(bar.high / bar.low))
        put("market.close_position", (bar.close - bar.low) / (bar.high - bar.low) if bar.high > bar.low else 0.5)
        put("market.log_volume", math.log1p(bar.volume))
        if bar.bid is not None and bar.ask is not None:
            put("orderbook.relative_spread", (bar.ask - bar.bid) / ((bar.ask + bar.bid) / 2))
        if bar.bid_size is not None:
            put("orderbook.log_bid_size", math.log1p(bar.bid_size))
        if bar.ask_size is not None:
            put("orderbook.log_ask_size", math.log1p(bar.ask_size))
        if bar.bid_size is not None and bar.ask_size is not None and bar.bid_size + bar.ask_size > 0:
            put("orderbook.imbalance", (bar.bid_size - bar.ask_size) / (bar.bid_size + bar.ask_size))
        while cursor < len(applicable):
            event = applicable[cursor]
            usable_at = max(_utc(event.published_at, "published_at"), _utc(event.available_at, "available_at"))
            if usable_at > timestamp:
                break
            if event.kind == "news":
                expiry = expiry_for(event)
                if expiry is None or timestamp < expiry:
                    encoded = _hashed_text(event.text)
                    news_decay = math.exp(-(timestamp - usable_at).total_seconds() / (6 * 3600))
                    news_vector += news_decay * encoded
                    news_count += news_decay
                    active_news[cursor] = (usable_at, encoded)
                    heapq.heappush(news_latest, (-usable_at.timestamp(), cursor))
                    if expiry is not None:
                        heapq.heappush(news_expirations, (expiry, cursor))
            else:
                for key, value in event.values.items():
                    name = _event_column(event, key)
                    source_key = (name, event.source)
                    previous_release = persistent.get(source_key)
                    release_order = (_utc(event.published_at, "published_at"), _utc(event.available_at, "available_at"))
                    if previous_release is None or release_order >= previous_release[:2]:
                        persistent[source_key] = (*release_order, expiry_for(event),
                                                  _finite(value, name) if value is not None else None)
            cursor += 1
        latest_values = {}
        for (name, source), snapshot in persistent.items():
            # Retain expired/new-null snapshots as tombstones: an older delayed
            # release must not resurrect an invalidated value, even cross-source.
            release_order = (*snapshot[:2], source)
            if name not in latest_values or release_order > latest_values[name][0]:
                latest_values[name] = (release_order, snapshot)
        for name, (_, (_, _, expiry, value)) in latest_values.items():
            if value is not None and (expiry is None or timestamp < expiry):
                put(name, value)
        while news_expirations and news_expirations[0][0] <= timestamp:
            _, article = heapq.heappop(news_expirations)
            available, encoded = active_news.pop(article)
            decay = math.exp(-(timestamp - available).total_seconds() / (6 * 3600))
            news_vector -= decay * encoded
            news_count = max(0.0, news_count - decay)
        while news_latest and news_latest[0][1] not in active_news:
            heapq.heappop(news_latest)
        if news_latest:
            latest_news_at = active_news[news_latest[0][1]][0]
            vector = news_vector / max(news_count, 1e-12)
            for index, value in enumerate(vector):
                put(f"news.hash_{index:02d}", float(value))
            put("news.decayed_count", news_count)
            put("news.log_seconds_since_latest", math.log1p((timestamp - latest_news_at).total_seconds()))
    if not np.isfinite(values).all():
        raise ValueError("feature values exceed representable finite float32 range")
    if frozen_names is not None:
        aligned_values = np.zeros((len(bar_list), len(frozen_names)), dtype=np.float32)
        aligned_observed = np.zeros_like(aligned_values, dtype=bool)
        aligned_groups = {group: [] for group in groups}
        for target, name in enumerate(frozen_names):
            aligned_groups[name.partition(".")[0]].append(target)
            if name in columns:
                source = columns[name]
                aligned_values[:, target] = values[:, source]
                aligned_observed[:, target] = observed[:, source]
        names, values, observed, groups = frozen_names, aligned_values, aligned_observed, aligned_groups
    return FeatureFrame(timestamps=times, values=values, observed=observed,
                        feature_names=names, groups=groups,
                        prices=np.asarray([bar.close for bar in bar_list], dtype=np.float64), symbol=symbol)


class _SameOriginHTTPSRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow feed redirects onto another host without its robots policy."""

    max_redirections = 3
    max_repeats = 1

    def __init__(self):
        super().__init__()
        self.robots = None
        self.user_agent = ""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        original, target = urllib.parse.urlparse(req.full_url), urllib.parse.urlparse(newurl)
        if (target.scheme != "https" or target.username or target.password
                or (target.hostname, target.port) != (original.hostname, original.port)):
            raise ValueError("RSS redirects must remain on the same HTTPS origin; use the final feed URL explicitly")
        if self.robots is not None and not self.robots.can_fetch(self.user_agent, newurl):
            raise PermissionError("robots.txt disallows the redirected feed URL")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_https(opener, url: str, user_agent: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/xml,text/xml,text/plain"})
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ValueError("RSS/robots response exceeds the 2 MB request limit")
    return payload


def collect_rss(url: str, output_path: str | Path, *, symbol: str | None = None,
                timeout: float = 15.0, user_agent: str = "StockTradeResearch/0.1",
                max_articles: int = 250, ttl_seconds: float | None = 86400.0,
                require_publication_time: bool = True) -> int:
    """Fetch one permitted HTTPS RSS/Atom feed and append provenance-rich JSONL.

    This bounded collector performs a robots request and a feed request, respects
    TLS verification, and refuses cross-origin or HTTP redirects. It does not
    crawl article pages or bypass paywalls. Newly collected news is available
    now, even when a feed gives an earlier publication date. Articles expire
    after ``ttl_seconds`` from publication (one day by default); already expired
    items are skipped. Explicit ``ttl_seconds=None`` allows archival collection
    while preserving actual collection time as the availability boundary.
    Undated entries are skipped by default, preventing repeated collection from
    making an old article appear newly published. Explicit legacy estimation
    via ``require_publication_time=False`` is marked in the saved provenance.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("RSS URL must be HTTPS with a hostname and without embedded credentials")
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 60:
        raise ValueError("timeout must be between 0 and 60 seconds")
    if not isinstance(max_articles, int) or isinstance(max_articles, bool) or not 1 <= max_articles <= 1000:
        raise ValueError("max_articles must be between 1 and 1000")
    if ttl_seconds is not None and _finite(ttl_seconds, "ttl_seconds") <= 0:
        raise ValueError("ttl_seconds must be positive")
    if not isinstance(require_publication_time, bool):
        raise ValueError("require_publication_time must be a boolean")
    if not user_agent.strip() or "\n" in user_agent or "\r" in user_agent:
        raise ValueError("a valid crawler User-Agent is required")
    if symbol is not None and (not isinstance(symbol, str) or not symbol.strip()):
        raise ValueError("symbol must be a nonempty string or null")
    redirect_handler = _SameOriginHTTPSRedirect()
    opener = urllib.request.build_opener(redirect_handler)
    robots_url = urllib.parse.urlunparse(("https", parsed.netloc, "/robots.txt", "", "", ""))
    robots = urllib.robotparser.RobotFileParser(robots_url)
    try:
        robots.parse(_read_https(opener, robots_url, user_agent, timeout).decode("utf-8", errors="replace").splitlines())
    except urllib.error.HTTPError as exc:
        if exc.code not in {404, 410}:
            raise PermissionError(f"cannot establish robots permission: HTTP {exc.code}") from exc
        robots.parse([])
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise PermissionError("cannot establish robots permission; feed was not requested") from exc
    if not robots.can_fetch(user_agent, url):
        raise PermissionError("robots.txt disallows this feed for the collector")
    redirect_handler.robots = robots
    redirect_handler.user_agent = user_agent
    delay = float(robots.crawl_delay(user_agent) or 0)
    rate = robots.request_rate(user_agent)
    if rate is not None:
        delay = max(delay, rate.seconds / rate.requests)
    if delay > 60:
        raise PermissionError("robots.txt requires a delay above 60 seconds; use a separately scheduled collector")
    if delay:
        time.sleep(delay)
    payload = _read_https(opener, url, user_agent, timeout)
    try:
        root = SafeET.fromstring(payload, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except DefusedXmlException as exc:
        raise ValueError("RSS documents with DTDs or entity declarations are unsupported") from exc
    except ET.ParseError as exc:
        raise ValueError("feed response is not valid RSS/Atom XML") from exc
    collected = datetime.now(timezone.utc)
    items = [item for item in root.iter() if item.tag.rsplit("}", 1)[-1] in {"item", "entry"}]
    rows = []
    for item in items[:max_articles]:
        fields = {child.tag.rsplit("}", 1)[-1]: " ".join(child.itertext()).strip() for child in item}
        text = " ".join(fields.get(key, "") for key in ("title", "description", "summary", "content")).strip()
        if not text:
            continue
        date_text = fields.get("pubDate") or fields.get("published") or fields.get("updated")
        if not date_text and require_publication_time:
            continue
        published = collected
        if date_text:
            try:
                published = _parse_timestamp(date_text, "feed publication time")
            except ValueError:
                try:
                    published = _utc(parsedate_to_datetime(date_text), "feed publication time")
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"feed item has an invalid publication time: {date_text!r}") from exc
        expiry = published + timedelta(seconds=ttl_seconds) if ttl_seconds is not None else None
        if expiry is not None and expiry <= collected:
            continue
        rows.append({"published_at": published.isoformat(), "available_at": collected.isoformat(),
                     "source": url, "kind": "news", "values": {}, "text": text, "symbol": symbol,
                     "expires_at": expiry.isoformat() if expiry is not None else None,
                     "publication_time_estimated": not bool(date_text)})
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return len(rows)
