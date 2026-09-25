"""Reading a web page's text into the conversation.

Two things need this: links users post directly to the bot, and digging into a
search result when Tavily's snippet is too thin to answer from.

The page itself is fetched by Tavily's Extract API, on Tavily's infrastructure.
That matters: this process never opens a connection to the URL it is reading,
so asking for a link cannot be used to reach addresses inside our own network.
A local fetch fallback for pages Tavily can't read would forfeit that property
and needs real SSRF defences (DNS checks, redirect re-validation, egress
isolation), so it is not part of this module.
"""

from urllib.parse import urlparse

from langchain_core.tools import tool
from langchain_tavily import TavilyExtract

from synthea.config import Config
from synthea.utilities import inference_logger

ALLOWED_SCHEMES: set[str] = {"http", "https"}


class UrlReadError(Exception):
    """Raised when Tavily could not extract a URL."""


def validate_url(url: str) -> str:
    """Returns ``url`` cleaned up, or raises ValueError if it isn't readable.

    Only absolute http(s) links with a host name are accepted. This is about
    handing Tavily a sane URL - since Tavily does the fetching, there is no
    local address check to make here.
    """
    cleaned = (url or "").strip().strip("<>")
    parsed = urlparse(cleaned)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Only http/https links can be read, got {cleaned[:100]!r}.",
        )
    if not parsed.hostname:
        raise ValueError(f"{cleaned[:100]!r} has no host name to read from.")
    return cleaned


# one Tavily client per api key: building one spins up its HTTP session
_TAVILY_TOOLS: dict[str, TavilyExtract] = {}


def _get_tavily_extract(api_key: str) -> TavilyExtract:
    if api_key not in _TAVILY_TOOLS:
        _TAVILY_TOOLS[api_key] = TavilyExtract(
            tavily_api_key=api_key,
            handle_tool_error=True,  # complaints come back as a string, not a raise
        )
    return _TAVILY_TOOLS[api_key]


async def _extract_with_tavily(url: str, query: str | None) -> str | None:
    """Extracts a page through Tavily, or returns None if it couldn't be read."""
    api_key = Config().tavily_api_key
    if not api_key:
        return None

    params: dict = {"urls": [url], "extract_depth": "basic"}
    if query:
        params["query"] = query

    try:
        result = await _get_tavily_extract(api_key).ainvoke(params)
    except Exception:
        inference_logger.info("Tavily extract failed for %s", url, exc_info=True)
        return None

    if not isinstance(result, dict):
        # handle_tool_error turned the failure into a message string
        inference_logger.info("Tavily extract could not read %s: %s", url, result)
        return None

    results = result.get("results") or []
    if not results:
        inference_logger.info("Tavily extract found no content for %s", url)
        return None

    page: dict = results[0]
    content = page.get("raw_content") or page.get("content") or ""
    return content.strip() or None


def _truncate(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return (
        content[:max_chars].rstrip()
        + f"\n\n[truncated: this page is {len(content)} characters; "
        + f"only the first {max_chars} are shown]"
    )


async def read_url_content(url: str, query: str | None = None) -> str:
    """Reads a web page as plain text through Tavily Extract.

    Raises ValueError for links we refuse to read and UrlReadError when Tavily
    came up empty.
    """
    target = validate_url(url)
    max_chars = Config().url_reader_max_chars

    content = await _extract_with_tavily(target, query)

    if not content:
        raise UrlReadError(
            f"Could not read {target}. The page may be down, empty, behind a "
            "login, or blocking automated readers - don't keep retrying it; "
            "try another link or a search instead.",
        )

    return _truncate(content, max_chars)


@tool
async def read_url(url: str, query: str | None = None) -> str:
    """Reads the full text of one web page and returns it as plain text.

    Use this when a user posts a link, or after a search when a result's
    snippet is not enough to answer the question properly: search first, then
    read the most promising result instead of guessing from its summary.

    Reads one URL per call. Only http/https links are supported, and very long
    pages are truncated.

    Args:
        url: The absolute http(s) URL of the page to read.
        query: Optional short question or topic to focus the extraction on.
    """
    return await read_url_content(url, query=query)
