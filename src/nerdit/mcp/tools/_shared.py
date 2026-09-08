"""Shared read-bound constants for the MCP tool implementations.

Split out of ``mcp/server.py`` (Track B WP24, pure motion). These pairs are
``inputSchema``-visible tool-signature defaults — a leaf inside ``tools/`` so
domain modules can import them without depending on ``mcp.server`` (which
would cycle: ``server`` imports the domain modules for ``build_server``).

(P33) It also owns :data:`SANDBOX_NOTE` — the one sandbox sentence every deploy
tool description carries — for the same leaf-module reason.
"""

from __future__ import annotations

from textwrap import wrap
from typing import TypeVar

# Bounds for the read tools. Agents must not be able to page the daemon to death
# or pull unbounded logs through a single tool call.
DEFAULT_LOG_TAIL = 100
MAX_LOG_TAIL = 1000
# Services page through a cursor server-side; cap the page the daemon caps at (≤200).
DEFAULT_SERVICE_LIMIT = 50
MAX_SERVICE_LIMIT = 200
# Models page like services (GET /api/models caps its ``limit`` at ≤200).
DEFAULT_MODEL_LIMIT = 50
MAX_MODEL_LIMIT = 200
# Databases page like models (GET /api/databases caps its ``limit`` at ≤200).
DEFAULT_DATABASE_LIMIT = 50
MAX_DATABASE_LIMIT = 200
# Audit pages through a cursor server-side; cap the page the daemon caps at (≤200).
DEFAULT_AUDIT_LIMIT = 50
MAX_AUDIT_LIMIT = 200
# The durable event feed pages like the audit log (GET /api/events clamps its
# ``limit`` at ≤200). There is deliberately no MAX for ``types``: the daemon
# honours the first 10 and clamping here would silently drop a filter an agent
# passed, where the server-side clamp is at least uniform and documented.
DEFAULT_EVENT_LIMIT = 50
MAX_EVENT_LIMIT = 200
# Routes page like services (GET /api/routes clamps its ``limit`` at ≤200).
DEFAULT_ROUTE_LIMIT = 50
MAX_ROUTE_LIMIT = 200
# Diagnose bundles a log tail; the daemon clamps ``log_tail`` to [1, 200].
DEFAULT_DIAGNOSE_LOG_TAIL = 50
MAX_DIAGNOSE_LOG_TAIL = 200
# The one-off run tool (P20) returns a bounded tail of the command's output; the
# daemon clamps ``log_tail`` to [1, 200], a compile-time bound, so mirroring it
# here is exact. There is deliberately NO ``MAX_RUN_TIMEOUT_S``: the timeout's
# ceiling is config-driven (``[services].run_timeout_max_s``), so a client-side
# clamp would silently truncate the request whenever an operator raises the cap,
# and the authoritative 422 ``run.timeout_too_large`` could never reach an MCP
# caller. The tool floors ``timeout_s`` at 1 and passes it through.
DEFAULT_RUN_TIMEOUT_S = 300
DEFAULT_RUN_LOG_TAIL = 200
MAX_RUN_LOG_TAIL = 200
# The managed-database dump tool (P37, D-P37-8). 300 s rather than the route's
# own 900 because the binding constraint here is the CLIENT, not the daemon: the
# reference streamable-HTTP MCP client reads with a 300 s ``sse_read_timeout``,
# so a tool default above it would turn every large dump into a transport error
# an agent reads as a failure. It is not one — the daemon runs the dump to
# completion regardless of who is still listening, and the tar then shows up in
# ``list_database_dumps``. Both tool descriptions say so. Like
# ``DEFAULT_RUN_TIMEOUT_S`` there is deliberately no MAX: the ceiling is
# config-driven (``[services].dump_timeout_max_s``) and a client-side clamp
# would hide the authoritative 422 ``dump.timeout_too_large`` from MCP callers.
DEFAULT_DUMP_TIMEOUT_S = 300

# --- The shared sandbox sentence (P33, field failure 2026-08-23) -------------
#
# A tool docstring IS its MCP description, so this is where an agent finds out
# what its image must survive BEFORE it writes a Dockerfile — the field failure
# this closes was an agent deploying a stock ``FROM nginx`` static site, whose
# root entrypoint died on ``chown(...) Operation not permitted``, burned the
# restart budget, and got ``fix_start_command`` back.
#
# ONE constant interpolated into every deploy-shaped tool rather than copies:
# four docstrings drifting apart is exactly how an agent ends up trusting the
# stalest one. Kept value-free and mechanically checkable (the golden in
# ``tests/data/mcp_tools_golden.json`` pins the rendered text).
#
# The capability drop is the DEFAULT profile, not a law:
# ``[containers].drop_all_caps`` / ``[containers].no_new_privileges`` are
# operator settings (``core/launch.py``) and ``capabilities.sandbox`` reports
# the live values, so the note states the default and points at the live source
# instead of asserting the drop categorically.
SANDBOX_NOTE = (
    "SANDBOX: by DEFAULT containers run with all Linux capabilities dropped "
    '(``cap_drop=["ALL"]``) and no-new-privileges — both are operator settings '
    "(``[containers].drop_all_caps`` / ``[containers].no_new_privileges``), so "
    "read the live profile in ``capabilities.sandbox`` rather than assuming: an "
    "operator may have relaxed it. Under that default, an image whose entrypoint "
    "does root work — ``chown``/``chmod`` of its own runtime dirs, dropping "
    "privileges via ``setuid``/``setgid``, writing ``/var/run`` — dies before "
    "your app ever starts. Stock ``nginx`` is the classic trap. Use a non-root "
    "image and a non-root ``USER`` that owns the files it writes: for a static "
    "site, ``nginxinc/nginx-unprivileged`` (listens on **8080**, so set "
    "``port=8080``) or any tiny static server. A crash from this reports "
    "remediation code ``image_needs_privileges`` in ``diagnose_service``. "
    "PORT CONTRACT: the "
    "container MUST listen on the declared ``port`` (the tool argument, else "
    "``nerdit.toml`` ``[deploy].port``, else **8000**) — that is the only port "
    "the daemon publishes and health-checks, so listening anywhere else fails "
    "the health probe with a healthy-looking container. A port below 1024 is "
    "NOT the reason a root entrypoint is needed: every container runs on the "
    "docker bridge in its own network namespace (never host networking), where "
    "Docker sets ``net.ipv4.ip_unprivileged_port_start=0``, so binding 80 needs "
    "no capability and no root — the chown/setuid work does."
)

SANDBOX_NOTE_PLACEHOLDER = "{SANDBOX_NOTE}"

_F = TypeVar("_F")


def apply_sandbox_note(fn: _F) -> _F:
    """Interpolate :data:`SANDBOX_NOTE` into ``fn.__doc__`` at the placeholder's indent.

    A docstring must be a string *literal* to become ``__doc__`` at all, so the
    shared sentence cannot be an f-string inside the ``def``; it is stamped in
    afterwards instead. The placeholder line's own indentation is reused for
    every wrapped line so ``inspect.cleandoc`` (what FastMCP and the golden test
    both apply) still strips one uniform prefix.

    A no-op when the docstring is absent (``python -OO``) or carries no
    placeholder, so it can be mapped over a whole ``TOOLS`` tuple.
    """
    doc = getattr(fn, "__doc__", None)
    if not doc or SANDBOX_NOTE_PLACEHOLDER not in doc:
        return fn
    lines: list[str] = []
    # ``split`` not ``splitlines``: the docstring's trailing newline before the
    # closing quotes is part of every other tool's description and must survive.
    for line in doc.split("\n"):
        if SANDBOX_NOTE_PLACEHOLDER not in line:
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines.extend(indent + chunk for chunk in _wrap(SANDBOX_NOTE))
    fn.__doc__ = "\n".join(lines)  # type: ignore[attr-defined]
    return fn


def _wrap(text: str, width: int = 76) -> list[str]:
    """Greedy word-wrap — deterministic, so the golden description is stable."""
    return wrap(" ".join(text.split()), width=width, break_long_words=False, break_on_hyphens=False)
