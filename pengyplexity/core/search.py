"""Web search and URL fetch orchestration for Pengyplexity.

This module provides:

* :class:`SearchResult` — a single search hit (title, URL, snippet).
* :class:`SearchService` — the abstract interface for web search + fetch.
  The agent's tool executor calls this.
* :class:`DDGSSearchService` — the real implementation using ``ddgs``
  (DuckDuckGo). Used in production.
* :class:`FakeSearchService` — offline test double. Returns canned results.
  The whole test suite uses this so ``pytest -q`` is green with no network.

Design notes:
* The search service is **injectable** into the tool executor. In production
  the app factory wires :class:`DDGSSearchService`; in tests, tests swap
  in :class:`FakeSearchService`.
* ``web_search`` returns formatted text (numbered results with title/URL/snippet)
  suitable for the model to read.
* ``fetch_url`` returns the page body (truncated to ``max_chars``).
* Source/citation extraction happens in the agent (subtask 8) — this module
  just provides the raw data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> Dict[str, str]:
        """As a dict (for JSON serialization / source tracking)."""
        return {"title": self.title, "url": self.url}


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SearchService(ABC):
    """Abstract search + fetch interface.

    Implementations:
    * :class:`DDGSSearchService` — real DuckDuckGo search.
    * :class:`FakeSearchService` — offline test double.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> str:
        """Search the web and return formatted results (numbered text).

        The return format is the same as what the model expects to see:
        ``1. Title\\n   URL: https://...\\n   snippet\\n...``
        """
        ...

    @abstractmethod
    def fetch(self, url: str, max_chars: Optional[int] = None) -> str:
        """Fetch a URL and return its text content (truncated)."""
        ...


# ---------------------------------------------------------------------------
# Real DuckDuckGo implementation
# ---------------------------------------------------------------------------


class DDGSSearchService(SearchService):
    """Real web search using ``ddgs`` (DuckDuckGo, no API key needed).

    Parameters
    ----------
    timeout:
        Search timeout in seconds (default 10).
    """

    def __init__(self, timeout: int = 10, user_agent: str = "Mozilla/5.0 (Pengyplexity)") -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    def search(self, query: str, max_results: int = 5) -> str:
        """Search using DDGS and format results."""
        import concurrent.futures

        from ddgs import DDGS

        def _do_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            results = executor.submit(_do_search).result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            return f"Web search timed out after {self.timeout} seconds."
        except Exception as e:
            return f"Error performing web search: {e}"
        finally:
            executor.shutdown(wait=False)

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}")
            lines.append(f"   URL: {r.get('href', '')}")
            body = r.get("body", "")
            if body:
                lines.append(f"   {body}")
            lines.append("")
        return "\n".join(lines).strip()

    def fetch(self, url: str, max_chars: Optional[int] = None) -> str:
        """Fetch a URL and return its text content."""
        import urllib.request
        import re

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error fetching {url}: {e}"

        # Strip HTML tags for a text approximation.
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated...]"
        return text


# ---------------------------------------------------------------------------
# Fake search for offline tests
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchService(SearchService):
    """Offline test double for :class:`SearchService`.

    Parameters
    ----------
    results:
        Default search results to return (as a list of
        ``(title, url, snippet)`` tuples or :class:`SearchResult`).
        If a dict is given keyed by query substring, matching queries
        return their specific results.
    fetch_bodies:
        Optional dict mapping URL → body text for ``fetch()``.
        Default: returns the URL as the body.
    """

    results: List[Any] = field(default_factory=list)
    fetch_bodies: Dict[str, str] = field(default_factory=dict)
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def search(self, query: str, max_results: int = 5) -> str:
        """Record the call and return formatted canned results."""
        self.calls.append({"op": "search", "query": query, "max_results": max_results})

        # Check for query-specific results (dict keyed by substring).
        if self.fetch_bodies and isinstance(self.results, dict):
            for key, res in self.results.items():
                if key.lower() in query.lower():
                    return self._format(res, max_results)

        if not self.results:
            return "No results found."

        # Results can be SearchResults or tuples.
        formatted_results = []
        for r in self.results[:max_results]:
            if isinstance(r, SearchResult):
                formatted_results.append(r)
            elif isinstance(r, (tuple, list)):
                title = r[0] if len(r) > 0 else ""
                url = r[1] if len(r) > 1 else ""
                snippet = r[2] if len(r) > 2 else ""
                formatted_results.append(SearchResult(title, url, snippet))
            elif isinstance(r, dict):
                formatted_results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", r.get("href", "")),
                        snippet=r.get("snippet", r.get("body", "")),
                    )
                )
        return self._format(formatted_results, max_results)

    def fetch(self, url: str, max_chars: Optional[int] = None) -> str:
        """Record the call and return the canned body."""
        self.calls.append({"op": "fetch", "url": url, "max_chars": max_chars})
        body = self.fetch_bodies.get(url, f"[fetched content from {url}]")
        if max_chars and len(body) > max_chars:
            body = body[:max_chars] + "\n[...truncated...]"
        return body

    def _format(self, results: List[SearchResult], max_results: int) -> str:
        """Format results in the same shape the model expects."""
        lines = []
        for i, r in enumerate(results[:max_results], 1):
            lines.append(f"{i}. {r.title}")
            lines.append(f"   URL: {r.url}")
            if r.snippet:
                lines.append(f"   {r.snippet}")
            lines.append("")
        return "\n".join(lines).strip()

    @property
    def search_calls(self) -> List[Dict[str, Any]]:
        """All search() calls recorded."""
        return [c for c in self.calls if c["op"] == "search"]

    @property
    def fetch_calls(self) -> List[Dict[str, Any]]:
        """All fetch() calls recorded."""
        return [c for c in self.calls if c["op"] == "fetch"]


# ---------------------------------------------------------------------------
# Helper: extract sources from a formatted search result string
# ---------------------------------------------------------------------------


def extract_sources(search_result: str) -> List[Dict[str, str]]:
    """Extract ``{title, url}`` pairs from a formatted search result string.

    This is a utility the agent (or UI) can use to build the "Sources" block.
    Handles the numbered format: ``1. Title\\n   URL: https://...\\n...``
    """
    import re

    sources: List[Dict[str, str]] = []
    lines = search_result.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for "N. Title" pattern
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            title = m.group(1).strip()
            url = ""
            # Check next line for "URL: ..."
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                um = re.match(r"^\s+URL:\s+(https?://\S+)", next_line)
                if um:
                    url = um.group(1)
                    i += 1
            if url:
                sources.append({"title": title, "url": url})
        i += 1
    return sources


def extract_source_from_url(url: str) -> Dict[str, str]:
    """Create a minimal source entry from a URL (for fetch_url citations)."""
    return {"title": url, "url": url}
