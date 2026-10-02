"""Web tools: `search_web` and `fetch_url`.

Both are keyless by design so the whole project runs on a free tier:
  - search  -> DuckDuckGo's HTML endpoint, or the Wikipedia API
  - fetch   -> plain HTTP GET + readability-style HTML-to-text

`fetch_url` returns a `Source` so the citation validator can later check that
claims cite documents actually retrieved in this run.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from citecheck.tools.base import Source, ToolContext, ToolResult

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_DDG_HTML = "https://html.duckduckgo.com/html/"

# Tags whose contents should never become readable text.
_DROP_TAGS = ("script", "style", "noscript", "svg", "template", "iframe")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def search_web(query: str, ctx: ToolContext, *, max_results: int = 6) -> ToolResult:
    """Search the web and return titles + URLs. Keyless backends only."""
    s = ctx.settings
    query = (query or "").strip()
    if not query:
        return ToolResult.failure("empty query")
    if s.search_backend == "off":
        return ToolResult.failure("search disabled (CITECHECK_SEARCH_BACKEND=off)")

    try:
        if s.search_backend == "wikipedia":
            results = _search_wikipedia(query, max_results, s.fetch_timeout_s)
        else:
            results = _search_duckduckgo(query, max_results, s.fetch_timeout_s)
    except httpx.HTTPError as e:
        return ToolResult.failure(f"search request failed: {type(e).__name__}")

    if not results:
        return ToolResult(
            data={"query": query, "results": [], "note": "no results; try rephrasing"},
            meta={"backend": s.search_backend},
        )
    return ToolResult(
        data={"query": query, "results": results},
        meta={"backend": s.search_backend, "result_count": len(results)},
    )


def _search_wikipedia(query: str, limit: int, timeout: float) -> list[dict[str, str]]:
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": str(limit),
    }
    r = httpx.get(_WIKI_API, params=params, timeout=timeout, follow_redirects=True,
                  headers={"User-Agent": "citecheck/0.1 (research agent)"})
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    return [
        {
            "title": h["title"],
            "url": f"https://en.wikipedia.org/wiki/{h['title'].replace(' ', '_')}",
            "snippet": re.sub(r"<[^>]+>", "", h.get("snippet", ""))[:300],
        }
        for h in hits
    ]


def _search_duckduckgo(query: str, limit: int, timeout: float) -> list[dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; citecheck/0.1; +research)"}
    r = httpx.post(_DDG_HTML, data={"q": query}, headers=headers,
                   timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return _parse_ddg(r.text, limit)


def _parse_ddg(html: str, limit: int) -> list[dict[str, str]]:
    """DuckDuckGo's HTML endpoint has no API; parse the result anchors."""
    out: list[dict[str, str]] = []
    # Anchors look like: <a class="result__a" href="//duckduckgo.com/l/?uddg=<encoded>">
    for m in re.finditer(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    ):
        href, title_html = m.group(1), m.group(2)
        url = _clean_ddg_url(href)
        if not url:
            continue
        out.append({"title": _strip_tags(title_html), "url": url, "snippet": ""})
        if len(out) >= limit:
            break
    return out


def _clean_ddg_url(href: str) -> str | None:
    from urllib.parse import parse_qs, unquote

    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href:
        qs = parse_qs(urlparse(href).query)
        target = qs.get("uddg")
        return unquote(target[0]) if target else None
    return href if href.startswith("http") else None


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------
def fetch_url(url: str, ctx: ToolContext, *, max_chars: int = 12_000) -> ToolResult:
    """Fetch a URL, extract readable text, and register it as a citable source."""
    s = ctx.settings
    url = (url or "").strip()
    if not url:
        return ToolResult.failure("empty url")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ToolResult.failure(f"unsupported scheme: {parsed.scheme or 'none'}")
    if not s.allow_private_network:
        verdict = check_public_host(parsed.hostname or "")
        if verdict is not None:
            return ToolResult.failure(f"blocked_host: {verdict}", blocked=True)

    try:
        resp = httpx.get(
            url,
            timeout=s.fetch_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; citecheck/0.1; +research)"},
        )
    except httpx.HTTPError as e:
        return ToolResult.failure(f"fetch failed: {type(e).__name__}: {e}")

    if resp.status_code >= 400:
        return ToolResult.failure(f"http_{resp.status_code}", status=resp.status_code)

    raw = resp.text
    if len(raw.encode("utf-8", errors="ignore")) > s.max_fetch_bytes:
        return ToolResult.failure(f"response exceeds {s.max_fetch_bytes} bytes", blocked=True)

    content_type = resp.headers.get("content-type", "")
    if "html" in content_type.lower() or "<html" in raw[:2000].lower():
        title, text = html_to_text(raw)
    else:
        title = parsed.hostname or url
        text = raw

    text = text[:max_chars]
    source = Source(
        id=f"S{_content_hash(url + text)[:8]}",
        uri=url,
        title=title,
        text=text,
        kind="web",
        content_hash=_content_hash(text),
    )
    return ToolResult(
        data={"source_id": source.id, "uri": source.uri, "title": source.title,
              "chars": len(text), "text": text},
        source=source,
        meta={"status": resp.status_code, "content_type": content_type},
    )


def check_public_host(hostname: str) -> str | None:
    """Return a reason string if `hostname` resolves to a non-public address.

    SSRF guard: a research agent fetching a model-chosen URL could otherwise be
    pointed at localhost, the LAN, or cloud metadata endpoints (169.254.169.254).
    Returns None when the host is acceptable.
    """
    if not hostname:
        return "missing hostname"
    lowered = hostname.lower()
    if lowered in ("localhost", "localhost.localdomain") or lowered.endswith(".local"):
        return f"{hostname} is a local host"
    if lowered.endswith(".internal") or lowered.endswith(".localdomain"):
        return f"{hostname} looks internal"

    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return f"cannot resolve {hostname}"
        ips = [ipaddress.ip_address(i[4][0]) for i in infos]
    else:
        ips = [addr]

    for ip in ips:
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return f"{hostname} resolves to non-public {ip}"
    return None


def html_to_text(html: str) -> tuple[str, str]:
    """Minimal HTML-to-text conversion. No external dependency, no BS4."""
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = _strip_tags(title_m.group(1)) if title_m else "Untitled"

    # Prefer <main>/<article> body if present -- strips nav and boilerplate.
    body = html
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.IGNORECASE | re.DOTALL)
        if m and len(_strip_tags(m.group(1))) > 500:
            body = m.group(1)
            break

    for t in _DROP_TAGS:
        body = re.sub(rf"<{t}[^>]*>.*?</{t}>", " ", body, flags=re.IGNORECASE | re.DOTALL)

    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", body, flags=re.IGNORECASE)
    body = re.sub(r"<li[^>]*>", "- ", body, flags=re.IGNORECASE)
    body = re.sub(r"<h([1-6])[^>]*>", lambda m: "\n" + "#" * int(m.group(1)) + " ", body,
                  flags=re.IGNORECASE)
    text = _strip_tags(body)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title.strip()[:200], text.strip()


def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    return (
        s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    )


def decode_entities(s: str) -> str:  # small public helper used by the eval harness
    return _strip_tags(s) if "<" in s else s


def summarise_source(src: Source, limit: int = 300) -> dict[str, Any]:
    return {"id": src.id, "uri": src.uri, "title": src.title,
            "excerpt": src.excerpt(limit), "kind": src.kind}