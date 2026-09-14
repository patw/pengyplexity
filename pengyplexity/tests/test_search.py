"""Tests for :mod:`pengyplexity.core.search`.

These test:
* ``FakeSearchService`` returns canned results in the right format.
* ``extract_sources`` parses the numbered search format correctly.
* ``SearchResult`` / dataclasses behave correctly.
* The ``SearchService`` ABC is satisfied by both real and fake.
* Fetch returns canned bodies and respects ``max_chars``.

All offline — no network, no real ddgs call.
"""

from __future__ import annotations

import pytest

from pengyplexity.core.search import (
    DDGSSearchService,
    FakeSearchService,
    SearchService,
    SearchResult,
    extract_source_from_url,
    extract_sources,
)


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_fields(self):
        r = SearchResult(title="Page", url="https://x.com", snippet="snippet")
        assert r.title == "Page"
        assert r.url == "https://x.com"
        assert r.snippet == "snippet"

    def test_to_dict(self):
        r = SearchResult(title="Page", url="https://x.com")
        d = r.to_dict()
        assert d == {"title": "Page", "url": "https://x.com"}


# ---------------------------------------------------------------------------
# SearchService interface
# ---------------------------------------------------------------------------


class TestSearchServiceInterface:
    def test_fake_is_search_service(self):
        assert isinstance(FakeSearchService(), SearchService)

    def test_ddgs_is_search_service(self):
        assert isinstance(DDGSSearchService(), SearchService)


# ---------------------------------------------------------------------------
# FakeSearchService: search
# ---------------------------------------------------------------------------


class TestFakeSearch:
    def test_no_results_returns_no_results_found(self):
        fs = FakeSearchService()
        result = fs.search("anything")
        assert result == "No results found."

    def test_returns_formatted_results(self):
        fs = FakeSearchService(
            results=[
                ("Cat Facts", "https://cats.example.com", "Cats are great."),
                ("Kittens 101", "https://kittens.example.org", "Learn about kittens."),
            ]
        )
        result = fs.search("cats")
        assert "1. Cat Facts" in result
        assert "URL: https://cats.example.com" in result
        assert "Cats are great." in result
        assert "2. Kittens 101" in result
        assert "URL: https://kittens.example.org" in result

    def test_max_results_respected(self):
        fs = FakeSearchService(
            results=[
                ("R1", "https://1.com", ""),
                ("R2", "https://2.com", ""),
                ("R3", "https://3.com", ""),
                ("R4", "https://4.com", ""),
            ]
        )
        result = fs.search("q", max_results=2)
        assert "1. R1" in result
        assert "2. R2" in result
        assert "3." not in result

    def test_search_result_objects(self):
        fs = FakeSearchService(
            results=[
                SearchResult(title="T1", url="https://a.com", snippet="S1"),
                SearchResult(title="T2", url="https://b.com", snippet="S2"),
            ]
        )
        result = fs.search("query")
        assert "T1" in result
        assert "https://a.com" in result
        assert "T2" in result
        assert "https://b.com" in result

    def test_dict_results(self):
        fs = FakeSearchService(
            results=[
                {"title": "D1", "href": "https://d1.com", "body": "Body1"},
            ]
        )
        result = fs.search("q")
        assert "D1" in result
        assert "https://d1.com" in result

    def test_records_search_calls(self):
        fs = FakeSearchService(results=[("T", "https://x.com", "")])
        fs.search("hello world", max_results=3)
        assert len(fs.search_calls) == 1
        assert fs.search_calls[0]["query"] == "hello world"
        assert fs.search_calls[0]["max_results"] == 3

    def test_multiple_search_calls(self):
        fs = FakeSearchService()
        fs.search("a")
        fs.search("b")
        assert len(fs.search_calls) == 2


# ---------------------------------------------------------------------------
# FakeSearchService: fetch
# ---------------------------------------------------------------------------


class TestFakeFetch:
    def test_default_fetch_body(self):
        fs = FakeSearchService()
        result = fs.fetch("https://example.com")
        assert "https://example.com" in result

    def test_canned_fetch_body(self):
        fs = FakeSearchService(
            fetch_bodies={"https://example.com": "<h1>Hello</h1>"}
        )
        result = fs.fetch("https://example.com")
        assert result == "<h1>Hello</h1>"

    def test_fetch_max_chars(self):
        fs = FakeSearchService(
            fetch_bodies={"https://long.com": "A" * 1000}
        )
        result = fs.fetch("https://long.com", max_chars=100)
        assert len(result) <= 120  # 100 + "\n[...truncated...]" (18 chars)
        assert "truncated" in result

    def test_records_fetch_calls(self):
        fs = FakeSearchService()
        fs.fetch("https://x.com", max_chars=500)
        assert len(fs.fetch_calls) == 1
        assert fs.fetch_calls[0]["url"] == "https://x.com"
        assert fs.fetch_calls[0]["max_chars"] == 500

    def test_search_and_fetch_tracked_separately(self):
        fs = FakeSearchService(results=[("T", "https://x.com", "")])
        fs.search("q")
        fs.fetch("https://y.com")
        assert len(fs.search_calls) == 1
        assert len(fs.fetch_calls) == 1
        assert len(fs.calls) == 2


# ---------------------------------------------------------------------------
# extract_sources — parsing the formatted search result
# ---------------------------------------------------------------------------


class TestExtractSources:
    def test_extracts_from_formatted_result(self):
        text = (
            "1. Cat Facts\n"
            "   URL: https://cats.example.com\n"
            "   Cats are great pets.\n"
            "\n"
            "2. Kittens 101\n"
            "   URL: https://kittens.example.org\n"
            "   Learn about kittens."
        )
        sources = extract_sources(text)
        assert len(sources) == 2
        assert sources[0] == {"title": "Cat Facts", "url": "https://cats.example.com"}
        assert sources[1] == {"title": "Kittens 101", "url": "https://kittens.example.org"}

    def test_no_sources_in_empty_result(self):
        assert extract_sources("No results found.") == []

    def test_no_url_line(self):
        text = "1. Some Result\n   No URL here\n"
        sources = extract_sources(text)
        assert sources == []  # no URL found

    def test_single_result(self):
        text = "1. Only\n   URL: https://one.com\n   snippet"
        sources = extract_sources(text)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://one.com"

    def test_real_fake_search_output(self):
        """Integration: extract_sources on FakeSearchService output."""
        fs = FakeSearchService(
            results=[
                ("Article A", "https://a.example.com", "About A"),
                ("Article B", "https://b.example.com", "About B"),
            ]
        )
        formatted = fs.search("test query")
        sources = extract_sources(formatted)
        assert len(sources) == 2
        assert sources[0]["title"] == "Article A"
        assert sources[0]["url"] == "https://a.example.com"
        assert sources[1]["title"] == "Article B"
        assert sources[1]["url"] == "https://b.example.com"


# ---------------------------------------------------------------------------
# extract_source_from_url
# ---------------------------------------------------------------------------


class TestExtractSourceFromUrl:
    def test_basic(self):
        s = extract_source_from_url("https://example.com/page")
        assert s == {"title": "https://example.com/page", "url": "https://example.com/page"}


# ---------------------------------------------------------------------------
# DDGSSearchService (no network — just construction)
# ---------------------------------------------------------------------------


class TestDDGSSearchServiceConstruction:
    def test_constructable_without_network(self):
        """Creating the service doesn't hit the network."""
        s = DDGSSearchService(timeout=5)
        assert s.timeout == 5
        assert isinstance(s, SearchService)

    def test_is_search_service_subclass(self):
        s = DDGSSearchService()
        assert isinstance(s, SearchService)


# ---------------------------------------------------------------------------
# Integration: search → extract → sources list
# ---------------------------------------------------------------------------


class TestSearchToSourcesPipeline:
    def test_full_pipeline(self):
        """Simulate: search → format → extract sources."""
        fs = FakeSearchService(
            results=[
                ("Python Docs", "https://docs.python.org", "Official Python docs"),
                ("Flask Guide", "https://flask.palletsprojects.com", "Flask web framework"),
                ("DuckDuckGo", "https://duckduckgo.com", "Privacy search"),
            ]
        )

        # Step 1: Search
        formatted = fs.search("python web frameworks", max_results=5)
        assert "Python Docs" in formatted

        # Step 2: Extract sources
        sources = extract_sources(formatted)
        assert len(sources) == 3
        assert sources[0] == {
            "title": "Python Docs",
            "url": "https://docs.python.org",
        }

        # Step 3: Format for the answer
        sources_block = "\n".join(
            f"[{i+1}] {s['title']} — {s['url']}" for i, s in enumerate(sources)
        )
        assert "[1] Python Docs — https://docs.python.org" in sources_block


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
