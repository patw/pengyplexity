"""Tests for the SAFE_TOOLS allowlist and the tool/skills audit.

These assert the *policy half* of the escape-proof contract: the model is only
ever handed the curated, re-scoped set of tools, the sudo/elevation parameter is
stripped, no host skill can appear as a raw tool, and every allowed tool has a
documented rationale. All offline — no Pengy import is required (a guarded
integration test cross-checks the real ``pengy.core.tools.TOOLS`` when present).
"""

from __future__ import annotations

import importlib.util

import pytest

from pengyplexity.sandbox.toolpolicy import (
    EXCLUDED,
    FORBIDDEN_TOOL_NAMES,
    RATIONALES,
    SAFE_TOOLS,
    build_safe_tool_schemas,
    is_safe_tool,
    scrub_tool_schema,
)

EXPECTED_SAFE = {
    "read_file",
    "read_multiple_files",
    "read_image",
    "directory_tree",
    "search_content",
    "glob",
    "web_search",
    "fetch_url",
    "download_file",
    "write_file",
    "replace_in_file",
    "apply_changes",
    "run_python",
    "run_bash",
    "todowrite",
    "ask_user_question",
    "make_chart",
    "generate_image",
    "edit_image",
    "create_report",
    "save_memory",
    "search_memory",
    "edit_memory",
    "delete_memory",
}


def test_safe_tools_is_exactly_the_curated_set():
    assert SAFE_TOOLS == frozenset(EXPECTED_SAFE)


def test_is_safe_tool_membership():
    assert is_safe_tool("web_search")
    assert is_safe_tool("run_bash")
    assert not is_safe_tool("plot")
    assert not is_safe_tool("elevated")
    assert not is_safe_tool("totally_unknown")


def test_every_safe_tool_has_an_activity_label():
    """Each SAFE_TOOLS name maps to a generic, user-facing activity label."""
    from pengyplexity.sandbox.toolpolicy import TOOL_ACTIVITY_LABELS, tool_activity_label
    for name in SAFE_TOOLS:
        assert name in TOOL_ACTIVITY_LABELS, f"missing activity label for {name}"
        assert TOOL_ACTIVITY_LABELS[name].strip()


def test_tool_activity_label_is_generic_and_safe():
    from pengyplexity.sandbox.toolpolicy import tool_activity_label
    # Research tool -> generic phrase, never the raw tool name.
    assert "web" in tool_activity_label("web_search").lower()
    # Unknown tool -> neutral fallback, never raises.
    assert tool_activity_label("bogus").strip()



def test_every_allowed_tool_has_a_rationale():
    missing = [name for name in SAFE_TOOLS if not RATIONALES.get(name, "").strip()]
    assert missing == []
    # The allowlist and the rationale keys agree exactly.
    assert set(RATIONALES) == set(SAFE_TOOLS)


def test_no_forbidden_skill_or_elevated_tool_is_allowed():
    # The allowlist must be disjoint from the forbidden skill/capability names.
    assert SAFE_TOOLS.isdisjoint(FORBIDDEN_TOOL_NAMES)
    # 'elevated' / 'sudo' must never be a tool name.
    assert "elevated" in FORBIDDEN_TOOL_NAMES
    assert "sudo" in FORBIDDEN_TOOL_NAMES
    assert "elevated" not in SAFE_TOOLS
    assert "sudo" not in SAFE_TOOLS


def test_audit_documents_the_excluded_skills():
    # Every excluded skill/capability in the audit is also on the forbidden list,
    # except for the run_bash.elevated parameter note (dotted, not a tool name).
    for name, reason in EXCLUDED.items():
        assert reason.strip(), f"missing rationale for excluded {name!r}"
        if "." in name:
            continue
        assert name in FORBIDDEN_TOOL_NAMES, f"{name} excluded but not forbidden"
        assert name not in SAFE_TOOLS


def test_scrub_strips_elevated_from_run_bash():
    schema = _make_schema("run_bash", with_elevated=True)
    scrubbed = scrub_tool_schema(schema)
    props = scrubbed["function"]["parameters"]["properties"]
    assert "elevated" not in props
    assert "elevated" not in scrubbed["function"]["parameters"].get("required", [])
    # The original is not mutated.
    assert "elevated" in schema["function"]["parameters"]["properties"]
    # The rewritten description drops sudo wording and states the sandbox.
    desc = scrubbed["function"]["description"].lower()
    assert "sudo" not in desc
    assert "sandbox" in desc


def test_scrub_adds_workspace_note_to_file_tools():
    schema = _make_schema("read_file")
    scrubbed = scrub_tool_schema(schema)
    assert "workspace" in scrubbed["function"]["description"].lower()
    # read_multiple_files uses a `paths` list parameter — that gets the note too.
    multi = _make_schema("read_multiple_files", param="paths")
    scrubbed_multi = scrub_tool_schema(multi)
    assert "workspace" in scrubbed_multi["function"]["parameters"]["properties"]["paths"]["description"].lower()


def test_scrub_is_a_noop_for_disallowed_tools():
    # A tool not on the allowlist is returned unchanged (so build can drop it).
    schema = _make_schema("plot")
    assert scrub_tool_schema(schema) is schema


def test_build_filters_and_scrubs_a_stub_tool_list():
    stub = [
        _make_schema("web_search"),
        _make_schema("read_file"),
        _make_schema("run_bash", with_elevated=True),
        _make_schema("plot"),          # skill -> dropped
        _make_schema("sudo"),          # forbidden -> dropped
    ]
    result = build_safe_tool_schemas(stub)
    names = {s["function"]["name"] for s in result}
    assert names == {"web_search", "read_file", "run_bash"}
    bash = next(s for s in result if s["function"]["name"] == "run_bash")
    assert "elevated" not in bash["function"]["parameters"]["properties"]


def _make_schema(name, with_elevated=False, param="path"):
    props = {param: {"type": "string", "description": f"the {name} arg"}}
    required = [param]
    if with_elevated:
        props["elevated"] = {"type": "boolean", "description": "invoke sudo"}
        required = required + ["elevated"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} does a thing. To invoke sudo, set elevated=true.",
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


@pytest.mark.skipif(
    importlib.util.find_spec("pengy") is None,
    reason="pengy not installed (offline-safe: allowlist tested via stubs above)",
)
def test_real_pengy_tools_are_all_allowed_or_dropped():
    """When Pengy is importable, every one of its TOOLS lands in SAFE_TOOLS and
    the scrubbed output contains no forbidden names and no elevated param."""
    from pengy.core.tools import TOOLS

    real_names = {t["function"]["name"] for t in TOOLS}
    # All of Pengy's current tools are on the allowlist (they get re-scoped,
    # not dropped) — and none are a forbidden skill name.
    assert real_names.issubset(SAFE_TOOLS)
    assert real_names.isdisjoint(FORBIDDEN_TOOL_NAMES)

    scrubbed = build_safe_tool_schemas(TOOLS)
    scrubbed_names = {s["function"]["name"] for s in scrubbed}
    assert scrubbed_names.isdisjoint(FORBIDDEN_TOOL_NAMES)
    bash = next((s for s in scrubbed if s["function"]["name"] == "run_bash"), None)
    if bash is not None:
        assert "elevated" not in bash["function"]["parameters"]["properties"]
