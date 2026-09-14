"""The safety boundary of Pengyplexity.

This package is where the "can't escape" contract lives and is asserted by the
test suite:

- :mod:`pengyplexity.sandbox.confine` — ``resolve(root, raw_path)`` confines every
  file path to the per-thread workspace.
- :mod:`pengyplexity.sandbox.executors` — the ``Runner`` interface plus the real
  ``BwrapRunner`` (bubblewrap confinement for ``run_python`` / ``run_bash``) and a
  ``FakeRunner`` for offline tests.
- :mod:`pengyplexity.sandbox.toolpolicy` — the explicit ``SAFE_TOOLS`` allowlist with
  a per-tool rationale and the full tool/skills audit.

Subtasks 2–4 fill these in: the ``SAFE_TOOLS`` allowlist (subtask 2) and the
path :func:`~pengyplexity.sandbox.confine.resolve` / :func:`~pengyplexity.sandbox.confine.workspace`
confinement guard (subtask 3) are implemented; the ``bwrap`` runner (subtask 4)
lands next. The package always imports cleanly (pure data, no heavy deps at
import time).
"""
