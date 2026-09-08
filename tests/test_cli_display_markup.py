"""Markup-safety and envelope-surfacing regressions for ``cli/display.py`` (P20 WP4/WP5).

The fourth rediscovery of "``console.print`` on a server-derived string is a
bug" (lessons.md, P20 WP1+WP2: *a helper is not a fix until every call site in
the module uses it*). This time the untrusted string is the **error envelope**:
the run route interpolates the caller-supplied ``{ident}`` into five of its
messages, so ``nerdit services run '[/x]' -- ls`` against a 404 raised
``MarkupError: closing tag '[/x]' …`` instead of printing "Not found".

Each test below drives a raw bracketed value through one renderer and asserts
it survives verbatim — the escape must be invisible, not just crash-free.
"""

from __future__ import annotations

import io

import httpx
import pytest

from nerdit.cli import display

# Values that are legal server output and illegal Rich markup: a closing tag
# with no opener is the shape that raises, a bare `[word]` is the shape that
# silently disappears.
MARKUP_BOMB = "[/x]"


def _capture(func, *args, **kwargs) -> str:
    """Run *func* with the display console redirected to a string buffer."""
    buf = io.StringIO()
    original = display.console
    display.console = display.Console(file=buf, force_terminal=False, width=200)
    try:
        func(*args, **kwargs)
    finally:
        display.console = original
    return buf.getvalue()


def _envelope_error(status: int, **envelope) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://gpu:9321/api/services/x/run")
    response = httpx.Response(status, request=request, json=envelope)
    return httpx.HTTPStatusError("err", request=request, response=response)


# --- markup safety: every interpolated field of the error envelope ----------


def test_envelope_message_hint_and_detail_survive_markup():
    """A 404 naming a bracketed ident renders as text, not a traceback."""
    exc = _envelope_error(
        404,
        code="not_found",
        message=f"Service '{MARKUP_BOMB}' not found",
        detail=f"no row for ident '{MARKUP_BOMB}'",
        hint=f"List them with `nerdit services list {MARKUP_BOMB}`.",
    )
    out = _capture(display.render_client_error, exc)
    assert "Not found" in out
    assert f"Service '{MARKUP_BOMB}' not found" in out
    assert f"no row for ident '{MARKUP_BOMB}'" in out
    assert f"nerdit services list {MARKUP_BOMB}" in out


def test_bare_detail_fallback_survives_markup():
    """The envelope-less 422 path renders `{detail}` through the same escape."""
    request = httpx.Request("POST", "http://gpu:9321/x")
    response = httpx.Response(422, request=request, json={"detail": "bad [/uvicorn] value"})
    exc = httpx.HTTPStatusError("err", request=request, response=response)
    out = _capture(display.render_client_error, exc)
    assert "bad [/uvicorn] value" in out


@pytest.mark.parametrize(
    "exc",
    [
        httpx.TransportError(f"socket died {MARKUP_BOMB}"),
        ValueError(f"weird {MARKUP_BOMB}"),
    ],
)
def test_non_http_errors_survive_markup(exc):
    """The transport and catch-all tails interpolate `exc` — escaped too."""
    out = _capture(display.render_client_error, exc)
    assert MARKUP_BOMB in out


def test_service_table_survives_markup_in_every_data_cell():
    """Table cells are markup-rendered as well, so they take the same escape."""
    out = _capture(
        display.display_service_table,
        [
            {
                "name": f"app{MARKUP_BOMB}",
                "status": "wei[rd]",  # unmapped status ⇒ bare cell (P5 finding)
                "restart_count": 0,
                "gpu_count": 0,
                "endpoint": {"url": f"127.0.0.1:8000{MARKUP_BOMB}"},
            }
        ],
    )
    assert f"app{MARKUP_BOMB}" in out
    assert "wei[rd]" in out


def test_deploy_result_survives_markup():
    out = _capture(
        display.display_deploy_result,
        {"name": f"app{MARKUP_BOMB}", "status": "building", "endpoint": {}},
    )
    assert f"app{MARKUP_BOMB}" in out


# --- P20: an actionable 5xx envelope must not be thrown away ----------------


def test_actionable_5xx_envelope_is_surfaced():
    """`run.interrupted` says the command MAY have partially applied.

    P20 is the first phase with actionable 5xx codes; collapsing them into
    "Daemon error (503). Check the daemon logs." is the difference between an
    operator knowing a migration half-ran and not.
    """
    exc = _envelope_error(
        503,
        code="run.interrupted",
        message="The run was interrupted; the command may have partially applied.",
        hint="Inspect the log tail, then re-run when it is safe to repeat.",
    )
    out = _capture(display.render_client_error, exc)
    assert "may have partially applied" in out
    assert "Inspect the log tail" in out
    assert "Check the daemon logs" not in out


def test_5xx_without_envelope_keeps_the_generic_fallback():
    """Hoisting the envelope read must not change the envelope-less path."""
    request = httpx.Request("GET", "http://gpu:9321/x")
    response = httpx.Response(500, request=request, text="upstream exploded")
    exc = httpx.HTTPStatusError("err", request=request, response=response)
    out = _capture(display.render_client_error, exc)
    assert "Daemon error (500)" in out
    assert "Check the daemon logs" in out


# --- P20: `nerdit services run` with a null exit code -----------------------


def _run_verb_output(result: dict) -> tuple[str, int | None]:
    """Drive the run verb's renderer over a canned response; return (out, exit)."""
    import asyncio

    import typer

    import nerdit.cli.client as client_mod
    import nerdit.cli.commands.services as services_mod

    class _FakeClient:
        async def run_service_command(self, *a, **k):
            return result

    original_factory = client_mod.get_configured_client
    client_mod.get_configured_client = lambda *a, **k: _FakeClient()  # type: ignore[assignment]
    # The command module binds `console` at import time, so redirecting
    # `display.console` alone would leave the output on real stdout.
    original_console = services_mod.console
    buf = io.StringIO()
    services_mod.console = display.Console(file=buf, force_terminal=False, width=200)
    seen: dict[str, int | None] = {"code": 0}
    try:
        try:
            asyncio.run(services_mod._run_async("api", ["ls"], timeout_s=10, env=None, log_tail=10))
        except typer.Exit as exc:
            seen["code"] = exc.exit_code
        return buf.getvalue(), seen["code"]
    finally:
        client_mod.get_configured_client = original_factory
        services_mod.console = original_console


def test_run_null_exit_code_prints_an_honest_line():
    """`exit_code: null` used to print "Exit  in 0.4s" — a malformed line.

    The schema documents null as "neither the bounded wait nor the post-mortem
    inspect could produce one", which is a reportable outcome, not a blank.
    """
    out, code = _run_verb_output(
        {"exit_code": None, "timed_out": False, "duration_s": 0.4, "log_tail": []}
    )
    assert "Exit unknown" in out
    assert "Exit  in" not in out
    assert "may have partially applied" in out
    assert code == 1  # a null exit code is still a failure


def test_run_nonzero_exit_code_is_unchanged():
    out, code = _run_verb_output(
        {"exit_code": 2, "timed_out": False, "duration_s": 1.0, "log_tail": []}
    )
    assert "Exit 2" in out
    assert code == 1


def test_run_zero_exit_code_is_unchanged():
    out, code = _run_verb_output(
        {"exit_code": 0, "timed_out": False, "duration_s": 1.0, "log_tail": []}
    )
    assert "Exit 0" in out
    assert code == 0


def _run_verb_raising(exc: Exception) -> tuple[str, int | None]:
    """Drive the run verb when the client raises; return (output, exit code)."""
    import asyncio

    import typer

    import nerdit.cli.client as client_mod
    import nerdit.cli.commands.services as services_mod

    class _FakeClient:
        async def run_service_command(self, *a, **k):
            raise exc

    original_factory = client_mod.get_configured_client
    client_mod.get_configured_client = lambda *a, **k: _FakeClient()  # type: ignore[assignment]
    original_console = services_mod.console
    buf = io.StringIO()
    services_mod.console = display.Console(file=buf, force_terminal=False, width=200)
    original_display_console = display.console
    display.console = services_mod.console
    seen: dict[str, int | None] = {"code": 0}
    try:
        try:
            asyncio.run(services_mod._run_async("api", ["ls"], timeout_s=10, env=None, log_tail=10))
        except typer.Exit as e:
            seen["code"] = e.exit_code
        return buf.getvalue(), seen["code"]
    finally:
        client_mod.get_configured_client = original_factory
        services_mod.console = original_console
        display.console = original_display_console


def _http_error(status: int, body: dict) -> Exception:
    import httpx

    request = httpx.Request("POST", "http://d/api/services/api/run")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_run_interrupted_prints_the_partial_output():
    """A 503 ``run.interrupted`` carries the ONLY surviving record of the run.

    ``run_once`` raises before it reaches ``set_last_run``, so ``nerdit
    diagnose`` shows the previous run, not this one — and ``render_client_error``
    prints message/detail/hint only. Without the special case the operator is
    told the command may have half-applied and shown nothing to judge it by.
    The tail is server-derived container stdout, so it must also survive Rich.
    """
    out, code = _run_verb_raising(
        _http_error(
            503,
            {
                "code": "run.interrupted",
                "message": "the run container was lost",
                "hint": "may have PARTIALLY applied",
                "container_started": True,
                "log_tail": ["Running upgrade abc -> def", "INFO [alembic] applying"],
            },
        )
    )
    assert "partial output" in out
    assert "Running upgrade abc -> def" in out
    assert "INFO [alembic] applying" in out  # markup-bearing, rendered verbatim
    assert "may have PARTIALLY applied" in out
    assert code == 1


def test_other_run_errors_print_no_partial_output_block():
    """The block is specific to run.interrupted — every other error has no tail."""
    out, _ = _run_verb_raising(
        _http_error(409, {"code": "service.run_in_progress", "message": "already running"})
    )
    assert "partial output" not in out
    assert "already running" in out


@pytest.mark.parametrize("value, expected", [(None, "-"), ("", ""), (0, "0"), ("[/x]", r"\[/x]")])
def test_plain_preserves_missing_values_and_markup(value, expected):
    assert display.plain(value) == expected
    assert display._plain(value) == ("" if value is None else expected)
