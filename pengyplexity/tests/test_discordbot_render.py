"""Tests for the Discord bot's pure text shaping (``discordbot/render.py``)."""

from __future__ import annotations

from types import SimpleNamespace

from pengyplexity.discordbot.render import (
    MESSAGE_LIMIT,
    clean_question,
    describe_api_error,
    format_sources,
    progress_text,
    quote_context,
    render_answer,
    split_message,
    thread_name,
)

BOT = 1234


class TestCleanQuestion:
    def test_strips_the_bots_own_mention_in_both_forms(self):
        assert clean_question(f"<@{BOT}> what is rust?", BOT) == "what is rust?"
        assert clean_question(f"<@!{BOT}> what is rust?", BOT) == "what is rust?"

    def test_names_other_mentions(self):
        text = f"<@{BOT}> did <@55> or <@&66> post in <#77>? <:party:88>"
        cleaned = clean_question(text, BOT, users={55: "alice"}, roles={66: "mods"}, channels={77: "news"})
        assert cleaned == "did @alice or @mods post in #news? :party:"

    def test_unknown_ids_fall_back_to_generic_names(self):
        assert clean_question("<@9> <#8>", BOT) == "@user #channel"


def test_quote_context_quotes_every_line():
    out = quote_context("bob", "line one\nline two", "is this true?")
    assert out == "bob wrote:\n> line one\n> line two\n\nis this true?"


def test_quote_context_truncates_huge_quotes():
    out = quote_context("bob", "x" * 10_000, "tl;dr?")
    assert len(out) < 4_100


class TestThreadName:
    def test_first_non_blank_line_collapsed(self):
        assert thread_name("\n  what   changed\nin 3.13") == "what changed"

    def test_capped_at_discords_limit(self):
        name = thread_name("word " * 100)
        assert len(name) <= 100 and name.endswith("…")

    def test_never_empty(self):
        assert thread_name("   ") == "Question"


class TestSplitMessage:
    def test_short_message_is_one_chunk(self):
        assert split_message("hello") == ["hello"]

    def test_empty_message_is_no_chunks(self):
        assert split_message("  ") == []

    def test_every_chunk_fits(self):
        text = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
        chunks = split_message(text)
        assert len(chunks) > 1
        assert all(len(c) <= MESSAGE_LIMIT for c in chunks)

    def test_no_newlines_still_splits(self):
        chunks = split_message("y" * 5000)
        assert all(len(c) <= MESSAGE_LIMIT for c in chunks)
        assert "".join(chunks) == "y" * 5000

    def test_code_fences_stay_balanced_across_chunks(self):
        code = "```python\n" + "\n".join(f"print({i})" for i in range(600)) + "\n```"
        chunks = split_message("Here:\n" + code)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.count("```") % 2 == 0, chunk[:80]


class TestProgressText:
    def test_shows_activity_before_the_answer_starts(self):
        text = progress_text("Searching the web…", "")
        assert "Searching the web…" in text and "⏹️" in text

    def test_shows_the_tail_of_a_long_answer_within_the_limit(self):
        text = progress_text("x", "a" * 5000 + "THE END")
        assert len(text) <= MESSAGE_LIMIT
        assert text.startswith("…") and "THE END" in text

    def test_closes_an_open_code_fence(self):
        text = progress_text("x", "```py\nprint(1)")
        assert text.split("\n-#")[0].count("```") == 2


class TestRenderAnswer:
    def test_answer_with_numbered_sources_that_do_not_unfurl(self):
        message = {
            "content": "Cats are mammals.",
            "sources": [
                {"title": "Cat [Wiki]", "url": "https://cats.example.com"},
                {"title": "dupe", "url": "https://cats.example.com"},
                {"title": "not a link", "url": "javascript:alert(1)"},
            ],
        }
        out = render_answer(message)
        assert out.startswith("Cats are mammals.")
        assert "1. [Cat Wiki](<https://cats.example.com>)" in out
        assert "2." not in out and "javascript" not in out

    def test_error_without_content(self):
        assert "⚠️ model exploded" in render_answer(None, error="model exploded")

    def test_error_with_partial_content_does_not_repeat_it(self):
        out = render_answer({"content": "partial\n\n_[Error]_"}, error="boom")
        assert "boom" not in out

    def test_nothing_at_all(self):
        assert render_answer(None).startswith("⚠️")

    def test_notes_come_last(self):
        assert render_answer({"content": "hi"}, notes=["-# note"]).endswith("-# note")

    def test_sources_are_capped(self):
        sources = [{"title": str(i), "url": f"https://e.com/{i}"} for i in range(30)]
        assert format_sources(sources).count("\n") == 10


def test_describe_api_error_is_specific():
    def err(code, status=409, retry_after=None):
        return SimpleNamespace(code=code, status=status, message="m", retry_after=retry_after)

    assert "still answering" in describe_api_error(err("turn_in_progress"))
    assert "12s" in describe_api_error(err("rate_limited", 429, 12))
    assert "API key" in describe_api_error(err("invalid_api_key", 401))
    assert "can't reach" in describe_api_error(err("unreachable", 0))
    assert "m" in describe_api_error(err("weird", 500))
