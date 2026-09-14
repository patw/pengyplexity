"""Tests for :mod:`pengyplexity.core.deepresearch`.

These test the multi-query research loop with **fake model** and **fake
search** — fully offline. Verify:

* The model is called twice: once for decomposition, once for synthesis.
* The search is called once per sub-query (bounded by query_budget).
* Sources are collected from all searches.
* The report is markdown (contains headings).
* The query budget is enforced.
* Fallback behavior when the model returns empty queries.
"""

from __future__ import annotations

import json

import pytest

from pengyplexity.core.deepresearch import (
    DeepResearchService,
    ResearchResult,
    run_research,
)
from pengyplexity.core.modelclient import ChatResponse, FakeModelClient
from pengyplexity.core.search import FakeSearchService, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_research_fakes(
    queries_response: str = "query one\nquery two\nquery three",
    report_response: str = "# Report\n\nFindings here.",
    search_results: list | None = None,
) -> tuple[FakeModelClient, FakeSearchService]:
    """Create a fake model + search for a research test."""
    # The model returns: first call = decomposition, second call = report.
    model = FakeModelClient(
        responses=[
            ChatResponse(content=queries_response),
            ChatResponse(content=report_response),
        ]
    )

    # The search returns canned results for each query.
    if search_results is None:
        search_results = [
            SearchResult("Result A", "https://a.example.com", "About A"),
            SearchResult("Result B", "https://b.example.com", "About B"),
        ]
    search = FakeSearchService(
        results=search_results,
    )
    return model, search


# ---------------------------------------------------------------------------
# Basic research flow
# ---------------------------------------------------------------------------


class TestBasicResearch:
    def test_returns_research_result(self):
        model, search = _make_research_fakes()
        service = DeepResearchService(model=model, search=search)
        result = service.research("Tell me about quantum computing")

        assert isinstance(result, ResearchResult)
        assert result.report_markdown == "# Report\n\nFindings here."
        assert result.query_count == 3
        assert len(result.queries) == 3

    def test_model_called_twice(self):
        model, search = _make_research_fakes()
        service = DeepResearchService(model=model, search=search)
        service.research("q")

        # First call: decomposition. Second: synthesis.
        assert model.call_count == 2

    def test_search_called_per_query(self):
        model, search = _make_research_fakes(
            queries_response="q1\nq2\nq3"
        )
        service = DeepResearchService(model=model, search=search)
        service.research("topic")

        # One search per sub-query.
        assert len(search.search_calls) == 3
        assert search.search_calls[0]["query"] == "q1"
        assert search.search_calls[1]["query"] == "q2"
        assert search.search_calls[2]["query"] == "q3"

    def test_queries_parsed_from_model_response(self):
        model, search = _make_research_fakes(
            queries_response="  1. quantum computing history\n2. quantum supremacy\n- quantum error correction"
        )
        service = DeepResearchService(model=model, search=search)
        result = service.research("quantum computing")

        assert "quantum computing history" in result.queries
        assert "quantum supremacy" in result.queries
        assert "quantum error correction" in result.queries

    def test_sources_collected(self):
        model, search = _make_research_fakes(
            search_results=[
                SearchResult("Source A", "https://a.com", "snippet"),
                SearchResult("Source B", "https://b.com", "snippet"),
            ]
        )
        service = DeepResearchService(model=model, search=search)
        result = service.research("q")

        # Sources from all searches are collected.
        urls = {s["url"] for s in result.sources}
        assert "https://a.com" in urls
        assert "https://b.com" in urls

    def test_sources_deduplicated(self):
        """If the same source appears in multiple searches, it's only listed once."""
        model, search = _make_research_fakes(
            search_results=[
                SearchResult("Same", "https://same.com", "x"),
            ]
        )
        service = DeepResearchService(model=model, search=search)
        result = service.research("q")

        # Only one entry for the duplicate URL.
        same_sources = [s for s in result.sources if s["url"] == "https://same.com"]
        assert len(same_sources) == 1


# ---------------------------------------------------------------------------
# Query budget enforcement
# ---------------------------------------------------------------------------


class TestQueryBudget:
    def test_budget_limits_queries(self):
        """Model returns 10 queries, but budget is 3 → only 3 searched."""
        model, search = _make_research_fakes(
            queries_response="\n".join(f"query_{i}" for i in range(10))
        )
        service = DeepResearchService(model=model, search=search, query_budget=3)
        result = service.research("big topic")

        assert result.query_count == 3
        assert len(result.queries) == 3
        assert len(search.search_calls) == 3

    def test_budget_of_1(self):
        model, search = _make_research_fakes(
            queries_response="a\nb\nc\nd"
        )
        service = DeepResearchService(model=model, search=search, query_budget=1)
        result = service.research("q")

        assert result.query_count == 1
        assert len(search.search_calls) == 1

    def test_fewer_queries_than_budget(self):
        """If model returns fewer queries than the budget, all are used."""
        model, search = _make_research_fakes(
            queries_response="just_one_query"
        )
        service = DeepResearchService(model=model, search=search, query_budget=6)
        result = service.research("q")

        assert result.query_count == 1
        assert len(search.search_calls) == 1


# ---------------------------------------------------------------------------
# Fallback: empty queries from model
# ---------------------------------------------------------------------------


class TestFallback:
    def test_empty_queries_falls_back_to_question(self):
        """If the model returns an empty response, the question itself is used."""
        model, search = _make_research_fakes(
            queries_response="",  # empty
            report_response="# Fallback Report",
        )
        service = DeepResearchService(model=model, search=search)
        result = service.research("What is dark matter?")

        # The question is used as the single query.
        assert result.query_count == 1
        assert result.queries == ["What is dark matter?"]

    def test_whitespace_only_queries(self):
        model, search = _make_research_fakes(
            queries_response="   \n  \n",
        )
        service = DeepResearchService(model=model, search=search)
        result = service.research("fallback question")

        assert result.queries == ["fallback question"]


# ---------------------------------------------------------------------------
# Report content
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_is_the_model_synthesis(self):
        report_md = "# Detailed Report\n\n## Summary\n\nContent here.\n\n## Sources\n\n- [1] A"
        model, search = _make_research_fakes(
            report_response=report_md,
        )
        service = DeepResearchService(model=model, search=search)
        result = service.research("q")

        assert result.report_markdown == report_md

    def test_synthesis_prompt_includes_findings(self):
        """The second model call includes the search results in its prompt."""
        model, search = _make_research_fakes(
            search_results=[SearchResult("Finding X", "https://x.com", "data")],
        )
        service = DeepResearchService(model=model, search=search)
        service.research("test question")

        # The second call (synthesis) should include the findings.
        synthesis_call = model.calls[1]
        messages = synthesis_call["messages"]
        user_msg = [m for m in messages if m["role"] == "user"][0]
        # The search result content should be in the prompt.
        assert "https://x.com" in user_msg["content"] or "Finding X" in user_msg["content"]


# ---------------------------------------------------------------------------
# run_research convenience
# ---------------------------------------------------------------------------


class TestRunResearch:
    def test_convenience_wrapper(self):
        model, search = _make_research_fakes()
        result = run_research(model, search, "quantum physics", query_budget=4)

        assert isinstance(result, ResearchResult)
        assert result.report_markdown != ""
        assert result.query_count <= 4


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_research_produces_actionable_report(self):
        """End-to-end: question → decompose → search → synthesize → report."""
        model = FakeModelClient(
            responses=[
                # Decomposition
                ChatResponse(content=(
                    "history of neural networks\n"
                    "transformer architecture\n"
                    "large language models 2024"
                )),
                # Synthesis
                ChatResponse(content=(
                    "# AI History Report\n\n"
                    "## Executive Summary\n"
                    "Neural networks evolved from perceptrons to transformers.\n\n"
                    "## Key Findings\n"
                    "1. Early perceptrons (1957)\n"
                    "2. Transformer breakthrough (2017)\n\n"
                    "## Sources\n"
                    "- [1] History of AI — https://ai.example.com\n"
                    "- [2] Transformer Paper — https://arxiv.example.com"
                )),
            ]
        )
        search = FakeSearchService(
            results=[
                SearchResult("History of AI", "https://ai.example.com", "From 1950s to now"),
                SearchResult("Transformer Paper", "https://arxiv.example.com", "Attention is all you need"),
            ]
        )

        service = DeepResearchService(model=model, search=search, query_budget=6)
        result = service.research("History of AI from neural nets to LLMs")

        # Report is the full synthesis.
        assert "# AI History Report" in result.report_markdown
        assert "Executive Summary" in result.report_markdown
        assert "Key Findings" in result.report_markdown

        # Queries were the 3 decomposed ones.
        assert len(result.queries) == 3

        # Sources were collected.
        assert len(result.sources) >= 1

        # Query count matches.
        assert result.query_count == 3
        assert len(search.search_calls) == 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
