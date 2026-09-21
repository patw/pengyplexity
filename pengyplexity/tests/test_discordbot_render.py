"""Tests for the Discord bot's pure text shaping (``discordbot/render.py``)."""

from __future__ import annotations

from types import SimpleNamespace

from pengyplexity.discordbot import render
from pengyplexity.discordbot.render import (
    MESSAGE_LIMIT,
    PENGUIN_ACTIVITIES,
    Image,
    Speaker,
    build_question,
    clean_question,
    describe_api_error,
    describe_images,
    format_history,
    image_links,
    looks_like_image_url,
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

    def test_it_asks_for_the_askers_memories_by_name(self):
        # A rolling channel conversation rolls over; what the bot knows about
        # a person is in its memories, not in the thread it happens to be in.
        out = build_question("hi", asker=ALICE, channel="general")
        assert "search_memory" in out and "Alice (@alice_dev)" in out

    def test_pictures_sit_between_the_history_and_the_question(self):
        out = build_question(
            "what is that?", asker=ALICE, history="@bob: look",
            images=describe_images([Image("cat.png", "https://cdn.example/cat.png")]),
            channel="general",
        )
        assert out.index("@bob: look") < out.index("cat.png") < out.index("[The question")

    def test_pictures_alone_are_enough_to_build_a_question(self):
        images = describe_images([Image("cat.png", "https://cdn.example/cat.png")])
        assert "cat.png" in build_question("what is that?", images=images)

    def test_a_dm_is_not_described_as_a_channel(self):
        out = build_question("hi", asker=BOB)
        assert "direct message" in out and "#" not in out.splitlines()[0]


class TestLooksLikeImageUrl:
    def test_plain_image_urls(self):
        assert looks_like_image_url("https://cdn.example/cat.PNG")
        assert looks_like_image_url("cat.jpeg")
        assert not looks_like_image_url("https://example.com/article")
        assert not looks_like_image_url("notes.pdf")

    def test_a_discord_cdn_link_keeps_its_query_string(self):
        # Every Discord attachment URL looks like this; an endswith() check
        # would see none of them.
        assert looks_like_image_url(
            "https://cdn.discordapp.com/attachments/1/2/cat.png?ex=abc&is=def&hm=99"
        )

    def test_nothing_is_not_an_image(self):
        assert not looks_like_image_url("")
        assert not looks_like_image_url(None)


class TestImageLinks:
    def test_finds_a_pasted_picture(self):
        assert image_links("look at https://i.example/cat.png please") == [
            ("cat.png", "https://i.example/cat.png")
        ]

    def test_ignores_ordinary_links(self):
        assert image_links("https://example.com/some/article") == []

    def test_trailing_punctuation_is_not_part_of_the_url(self):
        assert image_links("see https://i.example/cat.png.") == [
            ("cat.png", "https://i.example/cat.png")
        ]

    def test_a_link_in_angle_brackets_still_counts(self):
        # Discord's way of posting a link without an embed.
        assert image_links("<https://i.example/cat.gif>") == [
            ("cat.gif", "https://i.example/cat.gif")
        ]

    def test_the_same_link_twice_is_one_picture(self):
        text = "https://i.example/cat.png and https://i.example/cat.png"
        assert len(image_links(text)) == 1

    def test_no_text_is_no_links(self):
        assert image_links("") == [] and image_links(None) == []


class TestDescribeImages:
    def test_no_images_is_empty(self):
        assert describe_images([]) == ""

    def test_names_the_tools_that_actually_work(self):
        # A bare URL sends the agent to fetch_url, which only returns text.
        out = describe_images([Image("cat.png", "https://cdn.example/cat.png")])
        assert "download_file" in out and "read_image" in out
        assert "- cat.png: https://cdn.example/cat.png" in out

    def test_says_where_each_picture_came_from(self):
        out = describe_images([
            Image("cat.png", "https://cdn.example/cat.png", "posted earlier here by Alice"),
        ])
        assert "- cat.png (posted earlier here by Alice): https://cdn.example/cat.png" in out

    def test_the_same_url_is_listed_once(self):
        # A picture can be both attached to the replied-to message and in the
        # history window; the agent should not download it twice.
        out = describe_images([
            Image("cat.png", "https://cdn.example/cat.png", "with the question"),
            Image("cat.png", "https://cdn.example/cat.png", "posted earlier here by Bob"),
        ])
        assert out.count("https://cdn.example/cat.png") == 1
        assert "more, not listed" not in out

    def test_too_many_images_are_capped_and_counted(self):
        items = [Image(f"{i}.png", f"https://cdn.example/{i}.png") for i in range(10)]
        out = describe_images(items)
        assert out.count("https://cdn.example/") == 6
        assert "and 4 more" in out

    def test_the_first_ones_survive_the_cap(self):
        items = [Image(f"{i}.png", f"https://cdn.example/{i}.png") for i in range(10)]
        out = describe_images(items, max_files=2)
        assert "0.png" in out and "1.png" in out and "2.png" not in out
