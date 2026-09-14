"""Tests for memory search relevance.

With four memories saved, *every* search returned all four. Reciprocal Rank
Fusion scores a document by its **rank**, not its similarity, so on a small
corpus the results were near-identical rank artifacts (1/61, 1/62, 1/63...)
and searching for "pizza" surfaced everything.

:meth:`MemoryStore.search` fixes that with BotTalk's approach: rank with RRF,
but decide whether to return a result at all from the raw semantic cosine,
which is absolute and comparable across queries. These tests pin the filtering
and the reporting; they use an injected fake ranking rather than a live
embedding model, so the suite stays offline and fast.
"""

from __future__ import annotations

import pytest

from pengyplexity.core.memory import (
    SIGNAL_CONFIDENT,
    SIGNAL_FLOOR,
    MemorySearchResults,
    MemoryStore,
)


class _ScriptedStore(MemoryStore):
    """A store whose hybrid leg returns a canned ranking.

    Lets the filtering be tested against exact cosine values without loading
    a 422 MB embedding model in the test suite.
    """

    def __init__(self, rows, corpus=None, floor=SIGNAL_FLOOR, confident=SIGNAL_CONFIDENT):
        # Deliberately skips MemoryStore.__init__: nothing here touches disk.
        self._rows = rows
        self._corpus = len(rows) if corpus is None else corpus
        self.enable_semantic = True
        self.signal_floor = floor
        self.signal_confident = confident

    def search_hybrid(self, owner, query, limit=10, statuses=None, with_scores=False):
        rows = self._rows[:limit]
        if with_scores:
            return rows
        return [(doc, score) for doc, score, _legs in rows]

    def count(self, owner, statuses=None):
        return self._corpus


def _row(title, rrf, semantic=None, lexical=None):
    return (
        {"_id": title, "title": title, "summary": f"about {title}", "tags": []},
        rrf,
        {"semantic": semantic, "lexical": lexical},
    )


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


class TestRelevanceFloor:
    def test_irrelevant_matches_are_dropped(self):
        # What the live corpus actually returned for "pizza": four memories
        # about cars, an essay, and a forum thread, every cosine under 0.18.
        store = _ScriptedStore([
            _row("grand-tour", 0.01639, semantic=0.1752),
            _row("grand-tour-update", 0.01613, semantic=0.1678),
            _row("universe-essay", 0.01587, semantic=0.1408),
            _row("hn-muse", 0.01562, semantic=0.1270),
        ])

        found = store.search("pat", "pizza")

        assert found.results == []
        assert found.examined == 4
        assert found.filtered == 4

    def test_a_real_match_survives(self):
        store = _ScriptedStore([
            _row("hn-muse", 0.03279, semantic=0.6130),
            _row("universe-essay", 0.01613, semantic=0.3089),
            _row("grand-tour", 0.01587, semantic=0.1317),
        ])

        found = store.search("pat", "meta muse hacker news")

        assert [h.doc["_id"] for h in found.results] == ["hn-muse"]
        assert found.confident is True
        assert found.results[0].confidence == "strong"

    def test_borderline_match_is_kept_but_flagged_weak(self):
        store = _ScriptedStore([
            _row("grand-tour-update", 0.03279, semantic=0.4887),
            _row("hn-muse", 0.01613, semantic=0.0665),
        ])

        found = store.search("pat", "grand tour reboot reception")

        # Between the floor and the confidence bar: worth showing, not worth
        # trusting.
        assert [h.doc["_id"] for h in found.results] == ["grand-tour-update"]
        assert found.results[0].confidence == "weak"
        assert found.confident is False

    def test_floor_sits_below_the_confidence_bar(self):
        assert SIGNAL_FLOOR < SIGNAL_CONFIDENT

    def test_defaults_separate_this_corpus(self):
        """The measured cosines the defaults were derived from.

        Probed against the live corpus; the floor has to sit in the gap
        between the best true negative and the worst true positive. BotTalk's
        own floor (0.45) is inside the positive range here and cut 5 of 7
        genuine matches, which is why these are not its numbers.
        """
        worst_true_positive = 0.375   # "catbee universe article"
        best_true_negative = 0.308    # "how do I fix my car engine"

        assert best_true_negative < SIGNAL_FLOOR < worst_true_positive

    def test_thresholds_are_per_store(self):
        rows = [_row("a", 0.03, semantic=0.40)]

        assert len(_ScriptedStore(rows, floor=0.33).search("pat", "q")) == 1
        # A deployment with a different corpus can tighten it without a code
        # change — see the calibration note in core/memory.py.
        assert len(_ScriptedStore(rows, floor=0.45).search("pat", "q")) == 0

    def test_min_signal_zero_returns_everything(self):
        store = _ScriptedStore([
            _row("a", 0.016, semantic=0.17),
            _row("b", 0.015, semantic=0.12),
        ])

        found = store.search("pat", "pizza", min_signal=0)

        # The escape hatch for "is there anything even close?"
        assert len(found.results) == 2
        assert found.filtered == 0

    def test_min_signal_can_be_raised(self):
        store = _ScriptedStore([
            _row("strong", 0.03, semantic=0.61),
            _row("weak", 0.02, semantic=0.48),
        ])

        found = store.search("pat", "q", min_signal=0.55)

        assert [h.doc["_id"] for h in found.results] == ["strong"]


# ---------------------------------------------------------------------------
# Lexical-only results are never measured against the floor
# ---------------------------------------------------------------------------


class TestLexicalSignal:
    def test_lexical_only_hit_is_kept(self):
        # BM25 has no absolute scale, so a lexical-only hit cannot be judged
        # against a cosine floor — dropping it would lose exact-keyword
        # matches the semantic leg happened not to rank.
        store = _ScriptedStore([_row("kw", 0.016, lexical=3.2)])

        found = store.search("pat", "some exact keyword")

        assert len(found.results) == 1
        assert found.results[0].signal_kind == "relative"
        assert found.results[0].confidence == "unscored"

    def test_lexical_signal_is_relative_to_the_best_hit(self):
        store = _ScriptedStore([
            _row("best", 0.016, lexical=4.0),
            _row("half", 0.015, lexical=2.0),
        ])

        found = store.search("pat", "q")

        assert found.results[0].signal == pytest.approx(1.0)
        assert found.results[1].signal == pytest.approx(0.5)

    def test_semantic_wins_when_both_legs_scored_a_document(self):
        store = _ScriptedStore([_row("both", 0.03, semantic=0.62, lexical=9.9)])

        found = store.search("pat", "q")

        # The cosine is the comparable one; the BM25 value is display only.
        assert found.results[0].signal_kind == "cosine"
        assert found.results[0].signal == pytest.approx(0.62)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestAdvisory:
    def test_empty_result_reports_denominators(self):
        store = _ScriptedStore([_row("a", 0.016, semantic=0.17)], corpus=4)

        found = store.search("pat", "pizza")

        # An agent told a bare "nothing found" cannot tell an empty corpus
        # from a failed lookup, so it guesses. The numbers prevent that.
        assert "1 candidate" in found.advisory
        assert "4-memory corpus" in found.advisory
        assert "min_signal=0" in found.advisory

    def test_weak_only_result_warns_against_relying_on_it(self):
        store = _ScriptedStore([_row("a", 0.03, semantic=0.48)])

        found = store.search("pat", "q")

        assert found.results
        assert "confidence bar" in found.advisory

    def test_confident_result_needs_no_advisory(self):
        store = _ScriptedStore([_row("a", 0.03, semantic=0.71)])

        found = store.search("pat", "q")

        assert found.advisory == ""

    def test_counts_add_up(self):
        store = _ScriptedStore([
            _row("keep", 0.03, semantic=0.60),
            _row("drop", 0.02, semantic=0.10),
        ], corpus=7)

        found = store.search("pat", "q")

        assert (found.examined, found.surfaced, found.filtered) == (2, 1, 1)
        assert found.corpus == 7
        assert len(found) == 1
        assert found.docs == [found.results[0].doc]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class TestResultContainer:
    def test_is_falsey_when_empty(self):
        assert not MemorySearchResults().results
        assert len(MemorySearchResults()) == 0

    def test_iterates_over_hits(self):
        store = _ScriptedStore([_row("a", 0.03, semantic=0.7)])
        found = store.search("pat", "q")
        assert [h.doc["_id"] for h in found] == ["a"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
