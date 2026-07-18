"""HTTP helpers for scraping Pro-Football-Reference and similar sites.

PFR / Sports Reference often block automated clients with Cloudflare (403 +
"Just a moment..."). Prefer curl_cffi Chrome impersonation when installed;
fall back to requests. You can also pass a browser Cookie header via the
PFR_COOKIE env var or headers={\"Cookie\": \"...\"}.

For Cloudflare blocks, save the page in a browser and pass ``html_path`` to
``load_html`` / ``read_html`` (CLI: ``python -m src.scrape … --html file.htm``).

Always parse tables with StringIO + lxml (no html5lib; avoids the FutureWarning
from passing a raw HTML string to pd.read_html).
"""

from __future__ import annotations

import os
import time
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}

# Sports Reference asks scrapers to pause between requests.
DEFAULT_SLEEP = 3.0

_BLOCK_HINT = (
    "Pro-Football-Reference is behind Cloudflare and is blocking this client.\n"
    "Options:\n"
    "  1. Open the URL in your browser → Save Page As… →\n"
    "       python -m src.scrape <command> --html /path/to/page.htm\n"
    "  2. Copy the Cookie request header from DevTools and retry:\n"
    "       export PFR_COOKIE='your cookie string'\n"
    "       python -m src.scrape <command>\n"
    "  3. Wait a few minutes and retry (rate limits are common)."
)


def _looks_like_challenge(html: str) -> bool:
    head = html[:2000].lower()
    return "just a moment" in head or "cf-browser-verification" in head


def get_html(
    url: str,
    *,
    sleep: float = DEFAULT_SLEEP,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    cookie: str | None = None,
) -> str:
    """Fetch page HTML. Raises on HTTP errors or an obvious Cloudflare interstitial."""
    if sleep:
        time.sleep(sleep)

    hdrs = dict(DEFAULT_HEADERS)
    env_cookie = os.environ.get("PFR_COOKIE")
    if cookie or env_cookie:
        hdrs["Cookie"] = cookie or env_cookie
    if headers:
        hdrs.update(headers)

    parsed = urlparse(url)
    hdrs.setdefault("Referer", f"{parsed.scheme}://{parsed.netloc}/")

    html: str
    try:
        from curl_cffi import requests as crequests

        resp = crequests.get(
            url,
            headers=hdrs,
            timeout=timeout,
            impersonate="chrome",
        )
        html = resp.text
        status = resp.status_code
    except ImportError:
        import requests

        resp = requests.get(url, headers=hdrs, timeout=timeout)
        html = resp.text
        status = resp.status_code

    blocked = status >= 400 or _looks_like_challenge(html)
    if blocked:
        raise RuntimeError(f"HTTP {status} for {url}\n{_BLOCK_HINT}")
    return html


def load_html(
    url: str | None = None,
    *,
    html_path: str | Path | None = None,
    sleep: float = DEFAULT_SLEEP,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    cookie: str | None = None,
) -> str:
    """Return page HTML from a local file or a live fetch.

    Prefer ``html_path`` when Cloudflare blocks automated clients: save the
    page in a browser, then pass the path here (or via CLI ``--html``).
    """
    if html_path is not None:
        path = Path(html_path)
        return path.read_text(encoding="utf-8", errors="replace")
    if not url:
        raise ValueError("Either url or html_path is required")
    return get_html(
        url, sleep=sleep, timeout=timeout, headers=headers, cookie=cookie
    )


def read_html(
    url: str | None = None,
    *,
    html_path: str | Path | None = None,
    sleep: float = DEFAULT_SLEEP,
    flavor: str = "lxml",
    cookie: str | None = None,
    **kwargs: Any,
) -> list[pd.DataFrame]:
    """Like pd.read_html(url), via load_html + StringIO (supports ``html_path``)."""
    html = load_html(url, html_path=html_path, sleep=sleep, cookie=cookie)
    return pd.read_html(StringIO(html), flavor=flavor, **kwargs)
