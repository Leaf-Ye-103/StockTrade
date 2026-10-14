"""Read-only, provenance-preserving Alpaca market-data imports.

No broker/account/order endpoints are used. Historical bars use raw prices and
END timestamps; missing quotes and execution evidence remain missing. Alpaca's
historical endpoint is a present-day archive, not an as-of revision archive.
Downloaded news becomes available only when it is actually retrieved.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

_HOST = "https://data.alpaca.markets"
_MAX_PAGES = 100
_MAX_BYTES = 10_000_000
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.\-]{0,14}$")
_CREDENTIAL_NAMES = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
_BAR_FIELDS = ("timestamp", "symbol", "open", "high", "low", "close", "volume",
               "bid", "ask", "bid_size", "ask_size")
_BAR_LIMITATIONS = [
    "Bar end is an availability approximation, not a verified historical receipt timestamp.",
    "Raw historical bars can still contain later corrections; full point-in-time revisions are unresolved.",
    "IEX data covers one exchange; SIP requires the applicable Alpaca entitlement.",
    "Bars provide no quotes, trade aggressor, execution IDs, or queue lifecycle evidence. "
    "Observed execution matching is unsupported until separately supplied.",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: str | datetime, label: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{label} must be an ISO 8601 timestamp with a UTC offset") from None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return value.astimezone(timezone.utc)


def _inputs(symbols: Iterable[str], start: str | datetime, end: str | datetime,
            timeout: float) -> tuple[list[str], datetime, datetime]:
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    try:
        universe = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    except (AttributeError, TypeError):
        raise ValueError("symbols must be a nonempty collection of stock codes") from None
    if not universe or len(universe) > 100 or any(not _SYMBOL.fullmatch(symbol) for symbol in universe):
        raise ValueError("symbols must contain 1 to 100 valid stock codes")
    first, last = _timestamp(start, "start"), _timestamp(end, "end")
    if first >= last:
        raise ValueError("start must precede end")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
        raise ValueError("timeout must be a finite number in (0, 120]")
    if not 0 < timeout <= 120:
        raise ValueError("timeout must be a finite number in (0, 120]")
    return universe, first, last


def _credentials() -> tuple[str, str]:
    values = tuple(os.environ.get(name, "").strip() for name in _CREDENTIAL_NAMES)
    missing = [name for name, value in zip(_CREDENTIAL_NAMES, values) if not value]
    if missing:
        raise ValueError("Missing Alpaca environment variables: " + ", ".join(missing)
                         + ". Set their values securely in environment settings.")
    return values


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward authentication headers to a redirect destination."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Alpaca data requests must not redirect")


def _request_json(path: str, params: dict, *, timeout: float,
                  credentials: tuple[str, str]) -> dict:
    if path not in {"/v2/stocks/bars", "/v1beta1/news"}:
        raise ValueError("Unsupported read-only Alpaca data endpoint")
    request = urllib.request.Request(_HOST + path + "?" + urllib.parse.urlencode(params), headers={
        "APCA-API-KEY-ID": credentials[0], "APCA-API-SECRET-KEY": credentials[1],
        "Accept": "application/json", "User-Agent": "StockTrade-Research/0.2",
    })
    # Default proxy discovery is retained. TLS certificate verification remains enabled.
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()), _RejectRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(_MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"Alpaca data request failed with HTTP status {exc.code}; "
                         "check credentials, feed entitlement, and request range.") from None
    except (urllib.error.URLError, OSError, ValueError):
        # Do not expose server bodies, redirect destinations, request headers, or reasons.
        raise ValueError("Alpaca data request failed; check connectivity and the permitted data domain.") from None
    if len(payload) > _MAX_BYTES:
        raise ValueError("Alpaca data response exceeds the supported size")
    try:
        result = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Alpaca data response is not valid JSON") from None
    if not isinstance(result, dict):
        raise ValueError("Alpaca data response must be an object")
    return result


def _pages(path: str, params: dict, *, timeout: float, credentials: tuple[str, str]):
    seen: set[str] = set()
    for _ in range(_MAX_PAGES):
        result = _request_json(path, params, timeout=timeout, credentials=credentials)
        yield result
        token = result.get("next_page_token")
        if token in (None, ""):
            return
        if not isinstance(token, str) or len(token) > 4096:
            raise ValueError("Alpaca response has an invalid pagination token")
        if token in seen:
            raise ValueError("Alpaca response repeated a pagination token")
        seen.add(token)
        params = {**params, "page_token": token}
    raise ValueError("Alpaca import reached its page limit; request a smaller time range")


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Alpaca {label} must be a finite number")
    return float(value)


def _bar(raw: dict, symbol: str) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Alpaca bar must be an object")
    try:
        closed = _timestamp(raw["t"], "bar timestamp") + timedelta(minutes=1)
        values = {name: _number(raw[key], name)
                  for name, key in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v"))}
    except KeyError:
        raise ValueError("Alpaca bar is missing a required field") from None
    prices = [values[name] for name in ("open", "high", "low", "close")]
    if min(prices) <= 0 or values["high"] < max(prices) or values["low"] > min(prices) or values["volume"] < 0:
        raise ValueError("Alpaca bar contains invalid OHLCV values")
    return {"timestamp": closed.isoformat(), "symbol": symbol, **values,
            "bid": "", "ask": "", "bid_size": "", "ask_size": ""}


def fetch_alpaca_bars(symbols: Iterable[str], start: str | datetime, end: str | datetime,
                      output_dir: str | Path, *, feed: str = "iex", timeout: float = 15) -> dict:
    """Import raw 1Min bars, preserving absent quotes and retrieval provenance.

    The output directory must not already contain bars.csv/provenance.json.
    Only completed bars at or before the requested end and retrieval clock are
    retained. Account creation, paper orders, and broker orders are outside this
    read-only adapter. Returned counts may be zero for a closed-market interval.
    """
    universe, first, last = _inputs(symbols, start, end, timeout)
    if feed not in {"iex", "sip"}:
        raise ValueError("feed must be iex or sip")
    target = Path(output_dir)
    bars_path, provenance_path = target / "bars.csv", target / "provenance.json"
    if bars_path.exists() or provenance_path.exists():
        raise ValueError("Alpaca output files already exist; choose a new output directory")
    credentials = _credentials()
    started = _now()
    bound = min(last, started)
    params = {"symbols": ",".join(universe), "timeframe": "1Min", "start": first.isoformat(),
              "end": last.isoformat(), "adjustment": "raw", "feed": feed, "limit": 10000, "sort": "asc"}
    rows: dict[tuple[str, str], dict] = {}
    discarded = 0
    pages = 0
    for result in _pages("/v2/stocks/bars", params, timeout=timeout, credentials=credentials):
        pages += 1
        if "bars" not in result or not isinstance(result["bars"], (dict, type(None))):
            raise ValueError("Alpaca response is missing its bars object")
        for symbol, records in (result["bars"] or {}).items():
            if symbol not in universe or not isinstance(records, list):
                raise ValueError("Alpaca response contains unexpected stock bars")
            for raw in records:
                row = _bar(raw, symbol)
                closed = _timestamp(row["timestamp"], "bar end")
                if closed - timedelta(minutes=1) < first or closed > bound:
                    discarded += 1
                    continue
                key = (row["timestamp"], symbol)
                if key in rows and rows[key] != row:
                    raise ValueError("Alpaca response contains conflicting duplicate bars")
                rows[key] = row
    retrieved = _now()
    provenance = {"provider": "Alpaca", "feed": feed, "timeframe": "1Min", "adjustment": "raw",
                  "symbols": universe, "requested_start": first.isoformat(), "requested_end": last.isoformat(),
                  "request_started_at": started.isoformat(), "recorded_at": retrieved.isoformat(),
                  "bar_timestamp": "end_of_minute", "historical_availability": "bar_end_approximation",
                  "bars": len(rows), "discarded_outside_completed_range": discarded, "pages": pages,
                  "observed_execution_supported": False, "limitations": _BAR_LIMITATIONS}
    target.mkdir(parents=True, exist_ok=True)
    with bars_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_BAR_FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    with provenance_path.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    return {**provenance, "bars_path": str(bars_path), "provenance_path": str(provenance_path)}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > 500_000:
        raise ValueError("Alpaca news text must be a bounded string")
    parser = _Text()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def _news_source(value: object) -> str:
    if not isinstance(value, str):
        return _HOST + "/v1beta1/news"
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return _HOST + "/v1beta1/news"
    sensitive = {"api_key", "apikey", "token", "access_token", "secret", "password", "key"}
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or any(key.lower() in sensitive for key, _ in urllib.parse.parse_qsl(parsed.query))):
        return _HOST + "/v1beta1/news"
    return urllib.parse.urlunparse(parsed._replace(fragment=""))


def fetch_alpaca_news(symbols: Iterable[str], start: str | datetime, end: str | datetime,
                      output_path: str | Path, *, timeout: float = 15) -> dict:
    """Archive vendor news with actual retrieval availability, never backdated.

    ``created_at`` establishes publication, not historical receipt. Downloading
    old news today cannot insert it into a past decision. News expires 24 hours
    after publication; retrieval cannot refresh an old article. Use a previously
    captured archive for historical point-in-time news research.
    """
    universe, first, last = _inputs(symbols, start, end, timeout)
    target = Path(output_path)
    if target.exists():
        raise ValueError("Alpaca news output already exists; choose a new archive path")
    credentials = _credentials()
    params = {"symbols": ",".join(universe), "start": first.isoformat(), "end": last.isoformat(),
              "limit": 50, "include_content": "true", "sort": "asc"}
    raw_items: list[dict] = []
    pages = 0
    for result in _pages("/v1beta1/news", params, timeout=timeout, credentials=credentials):
        pages += 1
        if not isinstance(result.get("news"), list):
            raise ValueError("Alpaca response is missing its news list")
        raw_items.extend(result["news"])
    retrieved = _now()
    rows: dict[tuple, dict] = {}
    discarded = 0
    for raw in raw_items:
        if not isinstance(raw, dict) or "created_at" not in raw:
            raise ValueError("Alpaca news is missing its publication timestamp")
        published = _timestamp(raw["created_at"], "news publication")
        if published < first or published > last or published > retrieved:
            discarded += 1
            continue
        tickers = raw.get("symbols", [])
        if not isinstance(tickers, list) or any(not isinstance(symbol, str) for symbol in tickers):
            raise ValueError("Alpaca news symbols must be a list of stock codes")
        applicable = [symbol for symbol in universe if symbol in tickers] if tickers else [None]
        text = "\n".join(filter(None, (_text(raw.get(field)) for field in ("headline", "summary", "content"))))
        if not text or not applicable:
            discarded += 1
            continue
        source = _news_source(raw.get("url"))
        news_id = raw.get("id")
        if news_id is not None and (isinstance(news_id, bool) or not isinstance(news_id, int) or news_id < 0):
            raise ValueError("Alpaca news ID must be a nonnegative integer")
        for symbol in applicable:
            key = (published, source, text, symbol)
            rows[key] = {"published_at": published.isoformat(), "available_at": retrieved.isoformat(),
                         "recorded_at": retrieved.isoformat(), "expires_at": (published + timedelta(days=1)).isoformat(),
                         "source": source, "kind": "news", "values": {}, "text": text, "symbol": symbol,
                         "provider": "Alpaca", "vendor_news_id": news_id}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for row in sorted(rows.values(), key=lambda row: (row["published_at"], row["symbol"] or "", row["source"])):
            stream.write(json.dumps(row) + "\n")
    return {"provider": "Alpaca", "events_path": str(target), "events": len(rows), "pages": pages,
            "symbols": universe, "requested_start": first.isoformat(), "requested_end": last.isoformat(),
            "recorded_at": retrieved.isoformat(), "discarded_news": discarded,
            "historical_availability": "actual_retrieval_only",
            "limitations": ["Downloaded news cannot be backdated into historical strategy inputs.",
                            "News expires 24 hours after publication, including already-expired archive entries."]}
