"""Tests for the Discord bot's pure text shaping (``discordbot/render.py``)."""

from __future__ import annotations

from types import SimpleNamespace

from pengyplexity.discordbot import render
from pengyplexity.discordbot.render import (
    MESSAGE_LIMIT,
    PENGUIN_ACTIVITIES,
    Speaker,
    build_question,
    clean_question,
    describe_api_error,
    describe_attachments,
    format_history,
    format_roster,
    penguin_activity,
    progress_text,
    quote_context,
    render_answer,
    split_message,
    thread_name,
)

BOT = 1234
ALICE = Speaker("Alice", "alice_dev", 111)
BOB = Speaker("bob", "bob", 222)


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


class TestPenguinActivity:
    def test_always_comes_from_the_list(self):
        assert all(penguin_activity() in PENGUIN_ACTIVITIES for _ in range(100))

    def test_never_repeats_the_line_already_on_screen(self):
        current = PENGUIN_ACTIVITIES[0]
        assert all(penguin_activity(exclude=current) != current for _ in range(100))

    def test_falls_back_rather_than_crashing_when_excluding_the_only_option(self, monkeypatch):
        monkeypatch.setattr(render, "PENGUIN_ACTIVITIES", ("Only one…",))
        assert render.penguin_activity(exclude="Only one…") == "Only one…"


class TestProgressText:
    def test_shows_a_penguin_line_before_the_answer_starts(self):
        assert progress_text("Dreaming of fish…", "") == "-# 🐧 Dreaming of fish…"

    def test_has_no_stop_hint(self):
        assert "⏹️" not in progress_text("Dreaming of fish…", "")
        assert "⏹️" not in progress_text("Dreaming of fish…", "partial")

    def test_the_status_line_sits_under_the_answer_so_far(self):
        text = progress_text("Dreaming of fish…", "partial answer")
        assert text.startswith("partial answer")
        assert text.endswith("-# 🐧 Dreaming of fish…")

    def test_an_empty_label_still_says_something(self):
        assert progress_text("", "").startswith("-# 🐧 ")

    def test_shows_the_tail_of_a_long_answer_within_the_limit(self):
        text = progress_text("x", "a" * 5000 + "THE END")
        assert len(text) <= MESSAGE_LIMIT
        assert text.startswith("…") and "THE END" in text

    def test_closes_an_open_code_fence(self):
        text = progress_text("x", "```py\nprint(1)")
        assert text.split("\n-#")[0].count("```") == 2


class TestRenderAnswer:
    def test_sources_are_not_appended(self):
        # A numbered footer under every answer is the most chatbot-looking
        # thing in a channel. A link belongs inside the sentence instead.
        message = {
            "content": "Cats are mammals.",
            "sources": [{"title": "Cat Wiki", "url": "https://cats.example.com"}],
        }
        assert render_answer(message) == "Cats are mammals."

    def test_error_without_content(self):
        assert "⚠️ model exploded" in render_answer(None, error="model exploded")

    def test_error_with_partial_content_does_not_repeat_it(self):
        out = render_answer({"content": "partial\n\n_[Error]_"}, error="boom")
        assert "boom" not in out

    def test_nothing_at_all(self):
        assert render_answer(None).startswith("⚠️")

    def test_notes_come_last(self):
        assert render_answer({"content": "hi"}, notes=["-# note"]).endswith("-# note")


def test_describe_api_error_is_specific():
    def err(code, status=409, retry_after=None):
        return SimpleNamespace(code=code, status=status, message="m", retry_after=retry_after)

    assert "still answering" in describe_api_error(err("turn_in_progress"))
    assert "12s" in describe_api_error(err("rate_limited", 429, 12))
    assert "API key" in describe_api_error(err("invalid_api_key", 401))
    assert "can't reach" in describe_api_error(err("unreachable", 0))
    assert "m" in describe_api_error(err("weird", 500))


class TestSpeaker:
    def test_alias_and_handle_when_they_differ(self):
        assert ALICE.label == "Alice (@alice_dev)"
        assert ALICE.full == "Alice (@alice_dev), id 111"

    def test_a_repeated_name_is_not_said_twice(self):
        assert BOB.label == "@bob"
        assert Speaker("BOB", "bob", 1).label == "@bob"

    def test_missing_pieces_degrade_instead_of_breaking(self):
        assert Speaker("Alice", "", 5).label == "Alice"
        assert Speaker("", "", 0).label == "someone"
        # No id to add, so the long form is just the short one.
        assert Speaker("Alice", "alice_dev", 0).full == "Alice (@alice_dev)"


class TestFormatHistory:
    def test_keeps_chronological_order_and_identifies_speakers(self):
        out = format_history([(ALICE, "first"), (BOB, "second")])
        assert out == "Alice (@alice_dev): first\n@bob: second"

    def test_collapses_whitespace_and_drops_empty_messages(self):
        out = format_history([(ALICE, "a\n\n  b"), (BOB, "   ")])
        assert out == "Alice (@alice_dev): a b"

    def test_long_messages_are_trimmed(self):
        out = format_history([(ALICE, "x" * 2000)])
        assert len(out) < 600 and out.endswith("…")

    def test_over_budget_drops_the_oldest_not_the_newest(self):
        entries = [(Speaker(f"u{i}", f"u{i}", i), "x" * 400) for i in range(100)]
        entries.append((Speaker("newest", "newest", 999), "the last thing said"))
        out = format_history(entries)
        assert len(out) <= 8000
        # The messages nearest the question are the ones that explain it.
        assert "@newest: the last thing said" in out
        assert "@u0:" not in out

    def test_nothing_to_show_is_empty(self):
        assert format_history([]) == ""


class TestFormatRoster:
    def test_one_line_each_with_the_permanent_id(self):
        assert format_roster([ALICE, BOB]) == (
            "- Alice (@alice_dev), id 111\n- @bob, id 222"
        )

    def test_a_person_who_spoke_twice_is_listed_once(self):
        assert format_roster([ALICE, BOB, ALICE]).count("alice_dev") == 1

    def test_capped(self):
        many = [Speaker(f"u{i}", f"u{i}", i) for i in range(1, 60)]
        assert len(format_roster(many).splitlines()) == 20

    def test_nobody_is_empty(self):
        assert format_roster([]) == ""


class TestBuildQuestion:
    def test_no_asker_and_no_history_leaves_the_question_alone(self):
        assert build_question("why?") == "why?"

    def test_the_question_comes_last_and_names_who_asked(self):
        out = build_question(
            "what did they decide?", asker=ALICE, history="@bob: lunch?",
            roster=format_roster([ALICE, BOB]), channel="general",
        )
        assert "#general" in out
        assert out.index("@bob: lunch?") < out.index("[The question")
        assert "Alice (@alice_dev), id 111" in out
        assert out.endswith("what did they decide?")

    def test_it_forbids_the_phrase_that_ruins_memories(self):
        # The symptom this exists to fix: memories saved about "the user"
        # when there are dozens of people in the conversation.
        out = build_question("hi", asker=ALICE, channel="general")
        assert '"the user"' in out and "never" in out

    def test_a_dm_is_not_described_as_a_channel(self):
        out = build_question("hi", asker=BOB)
        assert "direct message" in out and "#" not in out.splitlines()[0]


class TestDescribeAttachments:
    def test_no_images_is_empty(self):
        assert describe_attachments([]) == ""

    def test_names_the_tools_that_actually_work(self):
        # A bare URL sends the agent to fetch_url, which only returns text.
        out = describe_attachments([("cat.png", "https://cdn.example/cat.png")])
        assert "download_file" in out and "read_image" in out
        assert "- cat.png: https://cdn.example/cat.png" in out

    def test_too_many_images_are_capped_and_counted(self):
        items = [(f"{i}.png", f"https://cdn.example/{i}.png") for i in range(10)]
        out = describe_attachments(items)
        assert out.count("https://cdn.example/") == 4
        assert "and 6 more" in out
