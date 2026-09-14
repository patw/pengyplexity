"""Deep research: multi-query research loop that synthesizes a report.

This module provides:

* :class:`ResearchResult` — the output of a research session (report
  markdown, queries used, sources).
* :class:`DeepResearchService` — the bounded research loop that:
  1. Asks the model to decompose the question into sub-queries.
  2. Executes web searches for each sub-query (bounded by
     ``query_budget``).
  3. Asks the model to synthesize all findings into a report.
  4. Returns the report markdown + all sources.

Design notes:
* Both the model and the search service are **injectable** — tests use
  fakes so the suite runs offline.
* The query budget (default 6) is a hard cap on total web searches.
* The model is called at most twice: once for query decomposition, once
  for synthesis. (A more sophisticated version could loop, but the spec
  says "bounded".)
* The report is markdown — the UI renders it to HTML and offers PDF
  download via the artifacts module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .modelclient import ChatResponse, ModelClient
from .search import SearchService


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ResearchResult:
    """Result of a deep research session.

    Attributes
    ----------
    report_markdown:
        The synthesized report (markdown format).
    queries:
        The sub-queries that were executed.
    sources:
        All sources collected across all searches.
    query_count:
        Number of searches actually performed (≤ query_budget).
    """

    report_markdown: str
    queries: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    query_count: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DeepResearchService:
    """Bounded multi-query research loop.

    Parameters
    ----------
    model:
        A :class:`ModelClient` (real or fake).
    search:
        A :class:`SearchService` (real or fake).
    query_budget:
        Maximum number of web searches to perform (default 6).
    """

    def __init__(
        self,
        model: ModelClient,
        search: SearchService,
        query_budget: int = 6,
    ) -> None:
        self.model = model
        self.search = search
        self.query_budget = query_budget

    def research(self, question: str) -> ResearchResult:
        """Execute a deep research session.

        Steps:
        1. Ask the model to decompose *question* into sub-queries.
        2. Search each sub-query (up to ``query_budget`` total).
        3. Ask the model to synthesize a report from all results.
        4. Return the :class:`ResearchResult`.

        Parameters
        ----------
        question:
            The user's research question.

        Returns
        -------
        ResearchResult
        """
        # Step 1: Decompose into sub-queries.
        queries = self._decompose(question)

        # Enforce the budget.
        queries = queries[: self.query_budget]

        # Step 2: Execute searches.
        all_results: List[str] = []
        all_sources: List[Dict[str, str]] = []
        for query in queries:
            result_text = self.search.search(query, max_results=5)
            all_results.append(f"### Search: {query}\n\n{result_text}")
            # Extract sources from the result.
            from .search import extract_sources
            sources = extract_sources(result_text)
            for s in sources:
                if s not in all_sources:
                    all_sources.append(s)

        # Step 3: Synthesize the report.
        report = self._synthesize(question, queries, all_results, all_sources)

        return ResearchResult(
            report_markdown=report,
            queries=queries,
            sources=all_sources,
            query_count=len(queries),
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _decompose(self, question: str) -> List[str]:
        """Ask the model to decompose a question into 3-6 sub-queries.

        Returns a list of search query strings.
        """
        prompt = (
            f"Decompose the following research question into 3-6 specific "
            f"web search queries that together will provide enough "
            f"information to write a comprehensive report.\n\n"
            f"Question: {question}\n\n"
            f"Return ONLY the queries, one per line, no numbering, no "
            f"explanation. Each query should be a natural search phrase."
        )
        response = self.model.chat([
            {"role": "system", "content": "You are a research planning assistant."},
            {"role": "user", "content": prompt},
        ])
        # Parse the queries from the response (one per line).
        queries = [
            line.strip().lstrip("-•0123456789. )").strip()
            for line in response.content.splitlines()
            if line.strip() and len(line.strip()) > 1
        ]
        if not queries:
            # Fallback: use the question itself as a single query.
            queries = [question]
        return queries

    def _synthesize(
        self,
        question: str,
        queries: List[str],
        results: List[str],
        sources: List[Dict[str, str]],
    ) -> str:
        """Ask the model to synthesize all search results into a report.

        Returns the report as markdown.
        """
        # Build the synthesis prompt.
        findings = "\n\n".join(results)
        sources_text = "\n".join(
            f"- {s['title']}: {s['url']}" for s in sources[:20]
        )

        prompt = (
            f"Write a comprehensive research report answering the following "
            f"question, based on the research findings below.\n\n"
            f"Question: {question}\n\n"
            f"## Research Findings\n\n{findings}\n\n"
            f"## Sources\n\n{sources_text}\n\n"
            f"Write the report in markdown format with:\n"
            f"- A title (# heading)\n"
            f"- An executive summary\n"
            f"- Key findings in sections (## headings)\n"
            f"- A 'Sources' section at the end listing the references\n"
            f"- Be thorough but concise. Cite sources inline where appropriate."
        )
        response = self.model.chat([
            {"role": "system", "content": (
                "You are a research report writer. Write clear, well-structured "
                "markdown reports based on the provided research findings."
            )},
            {"role": "user", "content": prompt},
        ])
        return response.content


# ---------------------------------------------------------------------------
# Convenience: single-call research (for the UI action)
# ---------------------------------------------------------------------------


def run_research(
    model: ModelClient,
    search: SearchService,
    question: str,
    query_budget: int = 6,
) -> ResearchResult:
    """One-shot deep research (convenience wrapper)."""
    service = DeepResearchService(model=model, search=search, query_budget=query_budget)
    return service.research(question)
