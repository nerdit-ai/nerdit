"""Tests for CLI error rendering and destructive-op confirmation prompts."""

from __future__ import annotations

import io

import httpx
import pytest
import typer

from nerdit.cli import display


def _capture(func, *args) -> str:
    """Run *func* with the display console redirected to a string buffer."""
    buf = io.StringIO()
    original = display.console
    display.console = display.Console(file=buf, force_terminal=False, width=200)
    try:
        func(*args)
    finally:
        display.console = original
    return buf.getvalue()


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://gpu:9321/jobs")
    response = httpx.Response(code, request=request, json={"detail": "boom"})
    return httpx.HTTPStatusError("err", request=request, response=response)


def _envelope_error(status: int, **envelope) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://gpu:9321/services")
    response = httpx.Response(status, request=request, json=envelope)
    return httpx.HTTPStatusError("err", request=request, response=response)


# --- render_client_error ---


def test_render_401_points_to_token():
    out = _capture(display.render_client_error, _status_error(401))
    assert "Authentication required" in out
    assert "nerdit token" in out


def test_render_403_invalid_token():
    out = _capture(display.render_client_error, _status_error(403))
    assert "Invalid or expired token" in out


def test_render_404_not_found():
    out = _capture(display.render_client_error, _status_error(404))
    assert "Not found" in out


def test_render_422_shows_detail():
    out = _capture(display.render_client_error, _status_error(422))
    assert "boom" in out


def test_render_500_server_error():
    out = _capture(display.render_client_error, _status_error(500))
    assert "Daemon error (500)" in out


# --- structured envelope surfacing on all 4xx (P13a) ---


def test_render_422_envelope_surfaces_message_and_hint():
    """A structured 422 envelope shows its message + hint (previously lost)."""
    exc = _envelope_error(
        422,
        code="deploy.invalid",
        message="Invalid deploy request",
        detail="name must be a DNS label",
        hint="Use lowercase letters, digits and hyphens.",
    )
    out = _capture(display.render_client_error, exc)
    assert "Invalid request" in out
    assert "Invalid deploy request" in out
    assert "name must be a DNS label" in out
    assert "Use lowercase letters, digits and hyphens." in out


def test_render_409_envelope_surfaces_message_and_hint():
    """A structured 409 conflict now surfaces its envelope hint (was generic)."""
    exc = _envelope_error(
        409,
        code="idempotency.conflict",
        message="Idempotency key already used",
        hint="Retry with a fresh Idempotency-Key.",
    )
    out = _capture(display.render_client_error, exc)
    assert "Conflict" in out
    assert "Idempotency key already used" in out
    assert "Retry with a fresh Idempotency-Key." in out


def test_render_403_envelope_surfaces_permission_message():
    """A 403 with an envelope is a permission denial, not a bad token."""
    exc = _envelope_error(
        403,
        code="forbidden",
        message="admin role required",
        hint="Ask an admin for a scoped token.",
    )
    out = _capture(display.render_client_error, exc)
    assert "Forbidden" in out
    assert "admin role required" in out
    assert "Invalid or expired token" not in out


def test_render_connect_error_includes_target():
    request = httpx.Request("GET", "http://gpu:9321/jobs")
    exc = httpx.ConnectError("refused", request=request)
    out = _capture(display.render_client_error, exc)
    assert "Cannot reach the daemon" in out
    assert "gpu:9321" in out


def test_render_timeout():
    request = httpx.Request("GET", "http://gpu:9321/jobs")
    exc = httpx.ReadTimeout("slow", request=request)
    out = _capture(display.render_client_error, exc)
    assert "did not respond in time" in out


def test_render_unknown_falls_back():
    out = _capture(display.render_client_error, ValueError("weird"))
    assert "weird" in out


# --- destructive-op confirmation ---


def test_service_rm_data_purge_aborts_when_not_confirmed(monkeypatch):
    """`nerdit services rm --purge data` without --yes bails out if the user declines."""
    import nerdit.cli.commands.services as services_mod

    def _decline(*a, **k):
        raise typer.Abort()

    monkeypatch.setattr(services_mod.typer, "confirm", _decline)
    # If it reached the client it would hit the network; the abort must come first.
    monkeypatch.setattr(
        services_mod.asyncio,
        "run",
        lambda *a, **k: pytest.fail("client called despite declined confirmation"),
    )
    with pytest.raises(typer.Abort):
        services_mod.services_rm("my-app", purge="data", force=False, yes=False)


def test_service_rm_skips_prompt_with_yes(monkeypatch):
    """--yes runs the destructive purge without prompting."""
    import nerdit.cli.commands.services as services_mod

    called = {"ran": False}

    def _fake_confirm(*a, **k):
        pytest.fail("confirm should be skipped with --yes")

    def _fake_run(coro):
        called["ran"] = True
        coro.close()  # avoid "coroutine was never awaited" warning

    monkeypatch.setattr(services_mod.typer, "confirm", _fake_confirm)
    monkeypatch.setattr(services_mod.asyncio, "run", _fake_run)
    services_mod.services_rm("my-app", purge="data", force=False, yes=True)
    assert called["ran"] is True


# --- display_logs markup safety (P20, found by the live agentic run) ---------
#
# A log line is the most untrusted string the CLI renders, and `display_logs`
# used to interpolate it straight into a Rich markup string. Both failure modes
# below were observed live against a real daemon, not constructed.


def test_release_prefix_survives_rich_rendering():
    """``[release]`` must not be eaten as a markup tag.

    The whole feature marks its output with this prefix; swallowing it made the
    migration tail indistinguishable from ordinary app output in ``nerdit logs``
    — the one place users read it.
    """
    out = _capture(
        display.display_logs,
        [
            {
                "stream": "system",
                "message": "[release] running (timeout 300s)",
                "timestamp": "2026-08-03T10:00:00",
            }
        ],
    )
    assert "[release] running (timeout 300s)" in out


def test_a_closing_tag_shape_in_a_log_line_does_not_crash_the_command():
    """``INFO [/uvicorn] done`` used to raise ``MarkupError`` and kill ``nerdit logs``.

    npm, pip and uvicorn all emit bracketed shapes, so this was reachable from
    any ordinary build log — not just a release tail.
    """
    out = _capture(
        display.display_logs,
        [
            {
                "stream": "stdout",
                "message": "INFO [/uvicorn] startup complete",
                "timestamp": "2026-08-03T10:00:00",
            }
        ],
    )
    assert "INFO [/uvicorn] startup complete" in out


def test_every_stream_style_is_markup_safe():
    """stderr/system/stdout all take the same escaped path."""
    entries = [
        {"stream": s, "message": f"[{s}] tag-ish [/x]", "timestamp": "2026-08-03T10:00:00"}
        for s in ("stdout", "stderr", "system")
    ]
    out = _capture(display.display_logs, entries)
    for s in ("stdout", "stderr", "system"):
        assert f"[{s}] tag-ish [/x]" in out
