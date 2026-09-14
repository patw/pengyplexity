"""Tests for :mod:`pengyplexity.core.memory`.

Exercises CRUD, per-owner scoping, update history, and all three search
modes (lexical/semantic/hybrid) against a real moofile collection in a temp
dir. The embedding model is cached locally on this host (see
``~/.cache/moofile/models``), so semantic search runs with no network call —
same offline guarantee as the rest of the suite.
"""

from __future__ import annotations

import pytest

from pengyplexity.core.memory import MemoryStore, normalize_tag


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.bson")
    yield s
    s.close()


@pytest.fixture
def lexical_store(tmp_path):
    """A store with semantic search disabled — lexical only, no model load."""
    s = MemoryStore(tmp_path / "memories_lex.bson", enable_semantic=False)
    yield s
    s.close()


class TestNormalizeTag:
    def test_lowercases_and_hyphenates(self):
        assert normalize_tag("Open Source") == "open-source"

    def test_collapses_repeats(self):
        assert normalize_tag("a--b..c") == "a-b.c"

    def test_strips_edges(self):
        assert normalize_tag("-tag-") == "tag"


class TestCRUD:
    def test_create_returns_stored_doc(self, store):
        doc = store.create("alice", "Likes cats", "Alice loves cats.", tags=["Pets", "pets"])
        assert doc["_id"]
        assert doc["owner"] == "alice"
        assert doc["title"] == "Likes cats"
        assert doc["tags"] == ["pets"]  # normalized + deduped
        assert doc["status"] == "active"
        assert "search_embedding" not in doc  # internal field stripped

    def test_get_respects_ownership(self, store):
        doc = store.create("alice", "T", "S")
        assert store.get(doc["_id"], owner="alice") is not None
        assert store.get(doc["_id"], owner="bob") is None
        assert store.get("nonexistent") is None

    def test_list_scoped_to_owner_newest_first(self, store):
        store.create("alice", "First", "S1")
        store.create("alice", "Second", "S2")
        store.create("bob", "Other", "S3")
        docs = store.list("alice")
        assert [d["title"] for d in docs] == ["Second", "First"]

    def test_list_filters_by_status(self, store):
        a = store.create("alice", "A", "S")
        store.update(a["_id"], owner="alice", status="deprecated")
        assert store.list("alice", statuses=("active",)) == []
        assert len(store.list("alice", statuses=("deprecated",))) == 1

    def test_update_appends_history_with_prior_value(self, store):
        doc = store.create("alice", "Old title", "Old summary")
        updated = store.update(doc["_id"], owner="alice", editor="alice", title="New title")
        assert updated["title"] == "New title"
        assert len(updated["update_history"]) == 1
        record = updated["update_history"][0]
        assert record["changed"] == ["title"]
        assert record["prior"]["title"] == "Old title"
        assert record["editor"] == "alice"

    def test_update_rejects_wrong_owner(self, store):
        doc = store.create("alice", "T", "S")
        assert store.update(doc["_id"], owner="bob", title="Hacked") is None
        assert store.get(doc["_id"])["title"] == "T"

    def test_update_no_changes_returns_doc_unmodified(self, store):
        doc = store.create("alice", "T", "S")
        same = store.update(doc["_id"], owner="alice", title="T")
        assert same["update_history"] == []

    def test_delete_respects_ownership(self, store):
        doc = store.create("alice", "T", "S")
        assert store.delete(doc["_id"], owner="bob") is False
        assert store.get(doc["_id"]) is not None
        assert store.delete(doc["_id"], owner="alice") is True
        assert store.get(doc["_id"]) is None


class TestSearch:
    def test_lexical_finds_keyword_match(self, store):
        store.create("alice", "Deadline", "Alice has a project deadline on Friday.")
        store.create("alice", "Unrelated", "The sky is blue today.")
        results = store.search_lexical("alice", "deadline")
        assert results
        assert results[0][0]["title"] == "Deadline"

    def test_lexical_scoped_to_owner(self, store):
        store.create("bob", "Deadline", "Bob has a deadline too.")
        results = store.search_lexical("alice", "deadline")
        assert results == []

    def test_semantic_finds_conceptual_match(self, store):
        store.create("alice", "Pet preference", "Alice loves cats and has two of them.")
        results = store.search_semantic("alice", "what pets does she like")
        assert results
        assert results[0][0]["title"] == "Pet preference"

    def test_hybrid_fuses_both_legs(self, store):
        store.create("alice", "Pet preference", "Alice loves cats and has two of them.")
        store.create("alice", "Work schedule", "Alice works remotely on Tuesdays.")
        results = store.search_hybrid("alice", "feline companions")
        assert results
        assert results[0][0]["title"] == "Pet preference"

    def test_semantic_disabled_falls_back_to_lexical_only(self, lexical_store):
        lexical_store.create("alice", "Deadline", "Alice has a project deadline.")
        assert lexical_store.search_semantic("alice", "due date") == []
        # Hybrid degrades to lexical when semantic is off.
        results = lexical_store.search_hybrid("alice", "deadline")
        assert results
        assert results[0][0]["title"] == "Deadline"
