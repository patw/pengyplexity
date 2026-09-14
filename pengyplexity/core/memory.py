"""Per-user memory store for Pengyplexity.

Modeled on BotTalk's (``~/Personal/BotTalk``) messageboard memory design: a
moofile collection with BM25 text indexes plus a semantic vector index (via
moofile's auto-embed), queried with lexical, semantic, or Reciprocal-Rank-
Fusion hybrid search. Unlike BotTalk's shared multi-bot board, memories here
are private and scoped per Pengyplexity user (``owner``), and support the
same append-only-history lifecycle so an edit is never a silent overwrite:

* ``status`` — one of :data:`VALID_STATUSES` (``active``/``superseded``/
  ``deprecated``); memories are never hard-deleted by an edit, only marked.
* ``update_history`` — every edit appends a record of what changed and what
  the prior value was, so the memory's evolution is auditable.

This is deliberately a *different* capability from the raw ``bottalk`` skill
(posting to the shared, cross-bot message bus, which stays on
``FORBIDDEN_TOOL_NAMES`` in ``sandbox/toolpolicy.py`` since it is an outbound
network side effect to a shared external system) — this store never leaves
the app's own local database and is never shared between users.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import moofile

VALID_STATUSES: Tuple[str, ...] = ("active", "superseded", "deprecated")

_TAG_CLEAN_RE = re.compile(r"[^a-z0-9.]+")
_TAG_DASH_RE = re.compile(r"-{2,}")
_TAG_DOT_RE = re.compile(r"\.{2,}")
MAX_TAG_LEN = 50

# Fields the internal ``search_text`` (what the semantic leg embeds) is
# rebuilt from whenever either changes.
_SEARCH_TEXT_FIELDS = ("summary", "body")

# moofile auto-embed config for the semantic leg, matching BotTalk's
# voyage-4-nano setup (512-dim MRL truncation, int8 quantized, asymmetric
# query/doc prefixes). The model is cached locally after first use — no
# network hit on subsequent opens.
AUTO_EMBED_CONFIG: Dict[str, Any] = {
    "search_text": {
        "target": "search_embedding",
        "dims": 512,
        "precision": "int8",
        "normalize": True,
        "max_length": 1024,
        "query_prefix": "Represent the query for retrieving supporting documents: ",
        "doc_prefix": "",
    },
}

_INTERNAL_FIELDS = ("search_text", "search_embedding")

# ---------------------------------------------------------------------------
# Confidence calibration (approach ported from BotTalk: ``bot_talk/routes.py``)
# ---------------------------------------------------------------------------
#
# Reciprocal Rank Fusion scores a document by its **rank**, not its similarity.
# On a small corpus that means every memory comes back for every query with
# near-identical scores (1/61, 1/62, 1/63...) — four saved memories, and
# searching for "pizza" returned all four. BotTalk measured the same effect on
# 25 hand-labelled queries that have an answer against 9 that do not: the
# fused RRF score could not separate them (AUC 0.74 — a correct answer and a
# query with no answer both score ~0.032) while the raw semantic cosine could
# (AUC 0.947).
#
# So ranking stays RRF, which fuses the two legs well, but *whether a result
# is returned at all* is decided by the absolute cosine, which is comparable
# across queries in a way a rank never is.
#
# The thresholds themselves are corpus- and model-specific and are NOT
# BotTalk's numbers: its floor of 0.45 was calibrated on its own longer posts
# and, applied here, cut 5 of 7 genuine matches. These defaults come from
# probing this app's corpus with 7 queries that should match and 5 that should
# not:
#
#   should match     0.375 - 0.571   (lowest: "catbee universe article")
#   should not match 0.174 - 0.308   (highest: "how do I fix my car engine")
#
# 0.33 sits in the gap: above every negative, below every positive. Treat that
# as a starting point, not a settled result — it is a dozen probes on a
# four-memory corpus, not a labelled audit, and the right value will drift as
# the corpus grows. Both thresholds are therefore configurable per deployment
# (``PENGYPLEXITY_MEMORY_SIGNAL_FLOOR`` / ``..._CONFIDENT``, and the admin
# settings page) rather than baked in here.
#
# The floor leans toward recall on purpose: these results are read by an LLM,
# which discards an irrelevant memory cheaply but cannot recover a relevant
# one it was never shown.
SIGNAL_CONFIDENT = 0.50
SIGNAL_FLOOR = 0.33


@dataclass
class MemoryHit:
    """One search result, with the evidence for how relevant it actually is.

    Attributes
    ----------
    doc:
        The memory document (internal search fields stripped).
    score:
        The fused RRF score — good for *ordering*, useless as a relevance
        measure (see the calibration note above).
    signal:
        The match signal on a 0-1 scale, or None when neither leg scored it.
    signal_kind:
        ``"cosine"`` — an absolute semantic similarity, comparable across
        queries, and the only kind the floor is applied to. ``"relative"`` —
        a BM25 score expressed as a fraction of the best lexical hit *in this
        result set*, so the top one is 1.0 by construction even when it is
        junk. Kept for display, never compared against the floor.
    confidence:
        ``"strong"`` (cosine >= SIGNAL_CONFIDENT), ``"weak"``, or
        ``"unscored"`` when the signal is not a cosine.
    legs:
        The raw per-leg scores: ``{"semantic": float|None, "lexical": float|None}``.
    """

    doc: Dict[str, Any]
    score: float
    signal: Optional[float] = None
    signal_kind: Optional[str] = None
    confidence: str = "unscored"
    legs: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass
class MemorySearchResults:
    """A search's surviving hits plus the denominators behind them.

    Reporting ``examined``/``surfaced``/``corpus`` rather than a bare empty
    list is BotTalk's "denominator discipline": an agent told *nothing was
    found* cannot tell an empty corpus from a failed lookup, and guesses. Told
    "0 of 4 candidates in a 4-memory corpus cleared the floor", it can say so.
    """

    results: List[MemoryHit] = field(default_factory=list)
    query: str = ""
    confident: bool = False
    examined: int = 0
    surfaced: int = 0
    filtered: int = 0
    corpus: int = 0
    floor: float = SIGNAL_FLOOR
    advisory: str = ""

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    @property
    def docs(self) -> List[Dict[str, Any]]:
        """Just the documents, for callers that only want to list them."""
        return [hit.doc for hit in self.results]


def normalize_tag(tag: str) -> str:
    """Normalize a raw tag to lowercase kebab/dotted-case form."""
    t = _TAG_CLEAN_RE.sub("-", tag.strip().lower())
    t = _TAG_DASH_RE.sub("-", t)
    t = _TAG_DOT_RE.sub(".", t)
    t = t.strip("-.")
    return t[:MAX_TAG_LEN] if len(t) > MAX_TAG_LEN else t


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _build_search_text(summary: str, body: str) -> str:
    return f"{summary or ''}\n\n{body or ''}".strip()


def _clean_tags(tags: Optional[Sequence[str]]) -> List[str]:
    if not tags:
        return []
    out = []
    for t in tags:
        norm = normalize_tag(str(t))
        if norm and norm not in out:
            out.append(norm)
    return out


def strip_internal(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *doc* without the internal search fields.

    ``search_embedding`` is a 512-float vector — never worth sending to a
    template or a tool result.
    """
    return {k: v for k, v in doc.items() if k not in _INTERNAL_FIELDS}


class MemoryStore:
    """Wraps a moofile collection of per-user memories.

    Parameters
    ----------
    path:
        Path to the ``.bson`` file backing the collection.
    enable_semantic:
        When True (default), configures the vector index + auto-embed so
        :meth:`search_semantic`/:meth:`search_hybrid` work. When False,
        lexical search still works fully; semantic/hybrid degrade to
        lexical-only. Tests that don't want to load the embedding model can
        pass False.

    The underlying moofile ``Collection`` is opened lazily (on first use, via
    the :attr:`col` property) so simply constructing a ``MemoryStore`` — e.g.
    as part of wiring up the app — costs nothing until a memory is actually
    read or written.
    """

    def __init__(
        self,
        path: Union[str, Path],
        enable_semantic: bool = True,
        signal_floor: float = SIGNAL_FLOOR,
        signal_confident: float = SIGNAL_CONFIDENT,
    ) -> None:
        self._path = str(path)
        self.enable_semantic = enable_semantic
        # Per-deployment overrides of the module defaults. Tunable because
        # the right cut point depends on the corpus and the embedding model —
        # see the calibration note at the top of this module.
        self.signal_floor = signal_floor
        self.signal_confident = signal_confident
        self._col: Optional[moofile.Collection] = None

    @property
    def col(self) -> moofile.Collection:
        if self._col is None:
            vector_indexes = {"search_embedding": 512} if self.enable_semantic else None
            auto_embed = AUTO_EMBED_CONFIG if self.enable_semantic else None
            self._col = moofile.Collection(
                self._path,
                indexes=["owner"],
                text_indexes=["title", "summary", "tags", "body"],
                vector_indexes=vector_indexes,
                auto_embed=auto_embed,
            )
        return self._col

    def close(self) -> None:
        if self._col is not None:
            self._col.close()
            self._col = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        owner: str,
        title: str,
        summary: str,
        tags: Optional[Sequence[str]] = None,
        body: str = "",
        status: str = "active",
    ) -> Dict[str, Any]:
        """Insert a new memory and return the stored document (with ``_id``)."""
        if status not in VALID_STATUSES:
            status = "active"
        doc = {
            "owner": owner,
            "title": title,
            "summary": summary,
            "tags": _clean_tags(tags),
            "body": body or "",
            "status": status,
            "created": _utcnow(),
            "updated": None,
            "update_history": [],
            "search_text": _build_search_text(summary, body or ""),
        }
        return strip_internal(self.col.insert(doc))

    def get(self, memory_id: str, owner: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a memory by id, or None. If *owner* is given, also checks
        ownership (returns None rather than someone else's memory)."""
        doc = self.col.find_one({"_id": memory_id})
        if doc is None:
            return None
        if owner is not None and doc.get("owner") != owner:
            return None
        return strip_internal(doc)

    def list(
        self,
        owner: str,
        statuses: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List an owner's memories, newest first."""
        filt = self._filter(owner, statuses, tags)
        docs = self.col.find(filt).sort("created", descending=True).to_list()
        return [strip_internal(d) for d in docs]

    def update(
        self,
        memory_id: str,
        owner: str,
        editor: str = "user",
        **fields: Any,
    ) -> Optional[Dict[str, Any]]:
        """Edit a memory the caller owns. Only recognized, changed fields in
        ``fields`` (``title``/``summary``/``tags``/``body``/``status``) are
        applied; every change is appended to ``update_history`` with the
        prior value, so nothing is silently overwritten without a trace.
        Returns the updated document, or None if not found/not owned.
        """
        doc = self.col.find_one({"_id": memory_id})
        if doc is None or doc.get("owner") != owner:
            return None

        set_fields: Dict[str, Any] = {}
        changed: List[str] = []
        for key in ("title", "summary", "tags", "body", "status"):
            if key not in fields or fields[key] is None:
                continue
            new_val = fields[key]
            if key == "tags":
                new_val = _clean_tags(new_val)
            if key == "status" and new_val not in VALID_STATUSES:
                continue
            if new_val != doc.get(key):
                set_fields[key] = new_val
                changed.append(key)

        if not changed:
            return strip_internal(doc)

        if "summary" in set_fields or "body" in set_fields:
            set_fields["search_text"] = _build_search_text(
                set_fields.get("summary", doc.get("summary", "")),
                set_fields.get("body", doc.get("body", "")),
            )

        now = _utcnow()
        set_fields["updated"] = now
        prior = {k: doc.get(k) for k in changed}
        history = list(doc.get("update_history") or [])
        history.append({
            "editor": editor,
            "timestamp": now,
            "changed": changed,
            "prior": prior,
        })
        set_fields["update_history"] = history

        self.col.update_one({"_id": memory_id}, set=set_fields)
        return self.get(memory_id, owner=owner)

    def delete(self, memory_id: str, owner: str) -> bool:
        """Hard-delete a memory the caller owns. Returns True if deleted."""
        doc = self.col.find_one({"_id": memory_id})
        if doc is None or doc.get("owner") != owner:
            return False
        return self.col.delete_one({"_id": memory_id})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_lexical(
        self,
        owner: str,
        query: str,
        limit: int = 10,
        statuses: Optional[Sequence[str]] = None,
        with_scores: bool = False,
    ) -> List[Tuple]:
        """BM25 search across title/summary/tags/body, title-boosted, deduped."""
        pre = self._filter(owner, statuses)
        best: Dict[str, Tuple[Dict[str, Any], float]] = {}
        for field in ("title", "summary", "tags", "body"):
            try:
                results = self.col.find(pre).text_search(field, query, limit=limit).to_list()
            except Exception:
                continue
            boost = 1.5 if field == "title" else 1.0
            for doc, score in results:
                did = doc["_id"]
                current = best.get(did, (None, float("-inf")))[1]
                if score * boost > current:
                    best[did] = (doc, score * boost)
        ranked = sorted(best.values(), key=lambda x: x[1], reverse=True)[:limit]
        if with_scores:
            return [
                (strip_internal(doc), score, {"semantic": None, "lexical": score})
                for doc, score in ranked
            ]
        return [(strip_internal(doc), score) for doc, score in ranked]

    def search_semantic(
        self,
        owner: str,
        query: str,
        limit: int = 10,
        statuses: Optional[Sequence[str]] = None,
        with_scores: bool = False,
    ) -> List[Tuple]:
        """Semantic (cosine) search on ``search_text`` (summary+body)."""
        if not self.enable_semantic:
            return []
        pre = self._filter(owner, statuses)
        results = self.col.find(pre).semantic("search_text", query, limit=limit).to_list()
        if with_scores:
            return [
                (strip_internal(doc), score, {"semantic": score, "lexical": None})
                for doc, score in results
            ]
        return [(strip_internal(doc), score) for doc, score in results]

    def search_hybrid(
        self,
        owner: str,
        query: str,
        limit: int = 10,
        statuses: Optional[Sequence[str]] = None,
        with_scores: bool = False,
    ) -> List[Tuple]:
        """BM25 + semantic search fused via Reciprocal Rank Fusion.

        Ranks well, but the fused score is derived from rank alone and says
        nothing about relevance — use :meth:`search` unless you specifically
        want the unfiltered ranking. With ``with_scores=True`` each row is
        ``(doc, rrf_score, {"semantic": cos|None, "lexical": bm25|None})``.

        Falls back to lexical-only when semantic search is disabled.
        """
        if not self.enable_semantic:
            return self.search_lexical(
                owner, query, limit=limit, statuses=statuses, with_scores=with_scores
            )

        pre = self._filter(owner, statuses)
        semantic_results = (
            self.col.find(pre).semantic("search_text", query, limit=limit * 3).to_list()
        )

        lexical: Dict[str, Tuple[Dict[str, Any], float]] = {}
        for field in ("title", "summary", "tags", "body"):
            try:
                results = self.col.find(pre).text_search(field, query, limit=limit * 3).to_list()
            except Exception:
                continue
            boost = 1.5 if field == "title" else 1.0
            for doc, score in results:
                did = doc["_id"]
                current = lexical.get(did, (None, float("-inf")))[1]
                if score * boost > current:
                    lexical[did] = (doc, score * boost)

        k = 60  # canonical RRF constant
        rrf: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        for rank, (doc, _) in enumerate(semantic_results):
            rrf[doc["_id"]] = rrf.get(doc["_id"], 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc["_id"]] = doc
        for rank, (doc, _) in enumerate(
            sorted(lexical.values(), key=lambda x: x[1], reverse=True)
        ):
            rrf[doc["_id"]] = rrf.get(doc["_id"], 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc["_id"]] = doc

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:limit]
        if not with_scores:
            return [(strip_internal(doc_map[did]), score) for did, score in fused]

        # The per-leg raw scores, kept alongside the fused rank score. The
        # semantic one is an absolute cosine and is the only thing that can
        # answer "is this actually relevant?" — see the calibration note at
        # the top of this module.
        sem_scores = {doc["_id"]: sc for doc, sc in semantic_results}
        lex_scores = {did: sc for did, (_doc, sc) in lexical.items()}
        return [
            (
                strip_internal(doc_map[did]),
                score,
                {"semantic": sem_scores.get(did), "lexical": lex_scores.get(did)},
            )
            for did, score in fused
        ]

    # ------------------------------------------------------------------
    # The search callers should use
    # ------------------------------------------------------------------

    def search(
        self,
        owner: str,
        query: str,
        limit: int = 10,
        statuses: Optional[Sequence[str]] = None,
        min_signal: Optional[float] = None,
    ) -> MemorySearchResults:
        """Search *owner*'s memories and return only the ones that clear the floor.

        This is what the ``search_memory`` tool and the memories page call.
        :meth:`search_hybrid` ranks well but scores every document on every
        query, because a Reciprocal Rank Fusion score describes a document's
        *position*, not its similarity — so on a small corpus every memory came
        back for every query, "pizza" included. The filtering here is BotTalk's
        (``bot_talk/routes.py``): judge relevance by the raw semantic cosine,
        which is absolute and comparable across queries.

        Parameters
        ----------
        min_signal:
            Override :data:`SIGNAL_FLOOR`. ``0`` disables filtering entirely
            and returns the near misses, which is the way to check whether
            something is in the corpus at all.
        """
        floor = self.signal_floor if min_signal is None else float(min_signal)
        scored_rows = self.search_hybrid(
            owner, query, limit=limit, statuses=statuses, with_scores=True
        )

        # A BM25 score means nothing on its own, so express it as a fraction
        # of the best lexical hit in THIS result set. That makes the top
        # lexical hit 1.0 by construction even when it is junk, which is
        # exactly why it is shown but never compared against the floor.
        lex_values = [
            legs.get("lexical")
            for _doc, _score, legs in scored_rows
            if legs.get("lexical") is not None
        ]
        max_lex = max(lex_values) if lex_values else None

        hits: List[MemoryHit] = []
        for doc, score, legs in scored_rows:
            sem = legs.get("semantic")
            lex = legs.get("lexical")
            if sem is not None:
                signal, kind = max(0.0, min(1.0, sem)), "cosine"
                confidence = "strong" if signal >= self.signal_confident else "weak"
            elif lex is not None and max_lex:
                signal, kind = max(0.0, min(1.0, lex / max_lex)), "relative"
                confidence = "unscored"
            else:
                signal, kind, confidence = None, None, "unscored"
            hits.append(
                MemoryHit(
                    doc=doc, score=score, signal=signal,
                    signal_kind=kind, confidence=confidence, legs=legs,
                )
            )

        kept = [h for h in hits if not (h.signal_kind == "cosine" and h.signal < floor)]
        confident = any(
            h.signal_kind == "cosine" and h.signal >= self.signal_confident for h in kept
        )
        corpus = self.count(owner, statuses=statuses)

        advisory = ""
        if not kept:
            # Never a bare "nothing found": an agent cannot tell an empty
            # corpus from a failed lookup, and fills the gap by guessing.
            advisory = (
                f"No memory scored above the relevance floor ({floor:g}). Examined "
                f"{len(hits)} candidate(s) from a {corpus}-memory corpus and surfaced "
                "0. There is most likely nothing saved on this topic — say so rather "
                "than guessing. Re-run with min_signal=0 to see the near misses."
            )
        elif not confident:
            advisory = (
                f"Nothing cleared the confidence bar ({self.signal_confident:g}); these are "
                "the closest memories available, not necessarily answers. Treat them "
                "as leads and verify before relying on them."
            )

        return MemorySearchResults(
            results=kept,
            query=query,
            confident=confident,
            examined=len(hits),
            surfaced=len(kept),
            filtered=len(hits) - len(kept),
            corpus=corpus,
            floor=floor,
            advisory=advisory,
        )

    def count(self, owner: str, statuses: Optional[Sequence[str]] = None) -> int:
        """How many memories *owner* has (optionally limited to *statuses*)."""
        try:
            return len(self.col.find(self._filter(owner, statuses)).to_list())
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter(
        owner: str,
        statuses: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        filt: Dict[str, Any] = {"owner": owner}
        if statuses:
            filt["status"] = {"$in": list(statuses)}
        if tags:
            filt["tags"] = {"$elemMatch": {"$in": _clean_tags(tags)}}
        return filt
