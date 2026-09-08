"""Test daemon restart with CliRunner and a mocked configured client.

Require confirmation, mint an idempotency key, display 202 drain counts and render
restart-in-progress 409s with exit 1. --wait accepts a new boot only when
now - uptime_s >= t_post; truncated uptime from the old process must not pass.
A lost POST response means probable restart only after a successful baseline GET;
an initially unreachable daemon exits 1. Cover both wait modes and MCP wrapper
registration, key handling and structured errors.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.app import app
from nerdit.cli.commands import daemon as daemon_mod

runner = CliRunner()

PROBABLY = "probably restarting"
# The daemon-down-only hint for the `[daemon].port` handoff dead-end (PR #105).
PORT_HINT = "still on the OLD port"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Keep Rich from wrapping the sentences the assertions look for."""
    monkeypatch.setenv("COLUMNS", "300")


def _client(**methods) -> AsyncMock:
    fake = AsyncMock()
    for name, value in methods.items():
        setattr(fake, name, AsyncMock(**value))
    return fake


def _accepted(**over) -> dict:
    body = {
        "restarting": True,
        "in_flight_builds": 2,
        "in_flight_runs": 1,
        "drain_timeout_s": 30,
    }
    body.update(over)
    return body


def _http_error(status: int, body: dict) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:9321/api/daemon/restart")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def _connect_error() -> httpx.ConnectError:
    return httpx.ConnectError(
        "connection refused",
        request=httpx.Request("POST", "http://localhost:9321/api/daemon/restart"),
    )


class _Clock:
    """A stand-in for the ``time`` module exposing only ``monotonic``.

    Patched onto the command module (never onto ``time`` itself) so the event
    loop keeps its real clock while the poll loop runs on this one.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t


def _install_clock(monkeypatch, clock: _Clock) -> None:
    """Freeze the command module's clock and make its ``sleep`` advance it."""

    async def _sleep(delay: float) -> None:
        clock.t += delay

    monkeypatch.setattr(daemon_mod, "time", clock)
    monkeypatch.setattr(daemon_mod, "asyncio", types.SimpleNamespace(run=asyncio.run, sleep=_sleep))


# --- confirmation gate --------------------------------------------------------


def test_restart_aborts_without_confirmation():
    fake = _client(
        get_capabilities={"return_value": {"version": "0.3.0", "uptime_s": 900}},
        restart_daemon={"return_value": _accepted()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart"], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    fake.restart_daemon.assert_not_awaited()


def test_restart_proceeds_on_confirmation():
    fake = _client(
        get_capabilities={"return_value": {"version": "0.3.0", "uptime_s": 900}},
        restart_daemon={"return_value": _accepted()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart"], input="y\n")
    assert result.exit_code == 0
    fake.restart_daemon.assert_awaited_once()


# --- happy path ---------------------------------------------------------------


def test_restart_mints_key_and_prints_counts():
    fake = _client(
        get_capabilities={"return_value": {"version": "0.3.0", "uptime_s": 900}},
        restart_daemon={"return_value": _accepted()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--drain-timeout-s", "30"])
    assert result.exit_code == 0
    kwargs = fake.restart_daemon.await_args.kwargs
    assert kwargs["drain_timeout_s"] == 30
    # uuid4().hex, minted at the command layer (the `nerdit gc` precedent).
    assert kwargs["idempotency_key"] and len(kwargs["idempotency_key"]) == 32
    # The 202 body is the operator's only view of what is being drained.
    assert "in-flight builds: 2" in result.output
    assert "in-flight runs: 1" in result.output
    assert "30" in result.output


def test_restart_survives_a_failed_baseline_and_still_posts():
    fake = _client(
        get_capabilities={"side_effect": _connect_error()},
        restart_daemon={"return_value": _accepted()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes"])
    assert result.exit_code == 0
    fake.restart_daemon.assert_awaited_once()


# --- structured failures ------------------------------------------------------


def test_restart_conflict_exits_1_with_envelope():
    fake = _client(
        get_capabilities={"return_value": {"version": "0.3.0", "uptime_s": 900}},
        restart_daemon={
            "side_effect": _http_error(
                409,
                {
                    "code": "daemon.restart_in_progress",
                    "message": "A daemon restart is already in progress.",
                    "hint": "Wait for the daemon to come back up; poll GET /capabilities.uptime_s.",
                },
            )
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes"])
    assert result.exit_code == 1
    assert "already in progress" in result.output
    assert PROBABLY not in result.output


def test_restart_forbidden_exits_1():
    fake = _client(
        get_capabilities={"return_value": {"version": "0.3.0", "uptime_s": 900}},
        restart_daemon={
            "side_effect": _http_error(
                403, {"code": "forbidden", "message": "Admin role required."}
            )
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes"])
    assert result.exit_code == 1
    assert "Admin role required." in result.output


# --- (d) transport-error branches --------------------------------------------


def test_lost_202_after_a_good_baseline_exits_0():
    """The `--drain-timeout-s 0` case: SIGTERM beats the 202 off the wire."""
    fake = _client(
        get_capabilities={"return_value": {"version": "0.3.0", "uptime_s": 900}},
        restart_daemon={"side_effect": _connect_error()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--drain-timeout-s", "0"])
    assert result.exit_code == 0
    assert PROBABLY in result.output
    # The port-handoff hint belongs to the daemon-down branch only.
    assert PORT_HINT not in result.output


def test_daemon_down_exits_1_and_never_claims_a_restart():
    """Baseline GET and POST both raise ⇒ no evidence a daemon was ever live."""
    fake = _client(
        get_capabilities={"side_effect": _connect_error()},
        restart_daemon={"side_effect": _connect_error()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes"])
    assert result.exit_code == 1
    assert PROBABLY not in result.output
    assert "Cannot reach the daemon" in result.output
    # …plus the one-line `[daemon].port` handoff hint (no detection attempted).
    assert PORT_HINT in result.output
    assert "[daemon].port" in result.output


def test_daemon_down_with_wait_still_exits_1(monkeypatch):
    """`--wait` must not rescue the daemon-down branch into a 0."""
    _install_clock(monkeypatch, _Clock())
    fake = _client(
        get_capabilities={"side_effect": _connect_error()},
        restart_daemon={"side_effect": _connect_error()},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--wait"])
    assert result.exit_code == 1
    assert PROBABLY not in result.output
    # It exits before polling: the wait banner is never printed.
    assert "waiting up to" not in result.output


# --- (e) boot-time freshness predicate ---------------------------------------


def _daemon_caps(clock: _Clock, boot: float, version: str) -> dict:
    """What a daemon booted at ``boot`` reports now — ``uptime_s`` is int()-truncated."""
    return {"version": version, "uptime_s": int(clock.t - boot)}


def test_wait_accepts_a_freshly_booted_daemon(monkeypatch):
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    # The old process booted 1.1 s before the POST — exactly the case that
    # answers an early poll with a truncated `uptime_s == 1` and satisfies the
    # rejected `uptime_s < elapsed + 1` form. The fresh one boots at t_post+0.2.
    old_boot, fresh_boot = 998.9, 1000.2
    calls = {"n": 0}

    async def _caps() -> dict:
        calls["n"] += 1
        if calls["n"] == 1:  # the pre-POST baseline
            return _daemon_caps(clock, old_boot, "0.3.0")
        if calls["n"] <= 4:  # the OLD process, still reachable mid-drain
            return _daemon_caps(clock, old_boot, "0.3.0")
        return _daemon_caps(clock, fresh_boot, "0.3.1")

    fake = _client(restart_daemon={"return_value": _accepted()})
    fake.get_capabilities = AsyncMock(side_effect=_caps)
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--wait"])
    assert result.exit_code == 0
    assert "Daemon back" in result.output
    assert "0.3.1" in result.output
    # The three mid-drain answers from the old process were all rejected.
    assert calls["n"] == 5


def test_wait_rejects_the_old_process_until_the_timeout(monkeypatch):
    """The regression the old predicate failed: an old daemon answers every poll.

    Its `uptime_s` is truncated downward, so the derived boot never rises above
    its true boot + 1 s — and its true boot is before the POST, so no answer is
    ever accepted and the wait honestly times out.
    """
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    old_boot = 998.9

    async def _caps() -> dict:
        return _daemon_caps(clock, old_boot, "0.3.0")

    fake = _client(restart_daemon={"return_value": _accepted()})
    fake.get_capabilities = AsyncMock(side_effect=_caps)
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--wait", "--wait-timeout", "5"])
    assert result.exit_code == 1
    assert "has not answered within 5s" in result.output
    assert "Daemon back" not in result.output


def test_wait_after_a_lost_202_polls_and_succeeds(monkeypatch):
    """(d) + (e): a lost 202 with a good baseline falls through to the poll."""
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    calls = {"n": 0}

    async def _caps() -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return _daemon_caps(clock, 900.0, "0.3.0")
        if calls["n"] == 2:
            raise _connect_error()  # mid-restart: expected, ignored
        return _daemon_caps(clock, 1000.4, "0.3.1")

    fake = _client(restart_daemon={"side_effect": _connect_error()})
    fake.get_capabilities = AsyncMock(side_effect=_caps)
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--wait"])
    assert result.exit_code == 0
    assert PROBABLY in result.output
    assert "Daemon back" in result.output


def test_wait_reports_an_error_status_when_the_daemon_only_ever_answers_5xx(monkeypatch):
    """A daemon that answers — badly — must not be reported as "never answered".

    The poll swallows the ``HTTPStatusError`` and keeps going (a booting daemon
    may legitimately answer ``503`` for a while), but it remembers the status so
    the timeout sentence says the daemon replied with an error, and names it.
    """
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    calls = {"n": 0}

    async def _caps() -> dict:
        calls["n"] += 1
        if calls["n"] == 1:  # a healthy pre-POST baseline
            return _daemon_caps(clock, 900.0, "0.3.0")
        raise _http_error(500, {"code": "internal_error", "message": "boom"})

    fake = _client(restart_daemon={"return_value": _accepted()})
    fake.get_capabilities = AsyncMock(side_effect=_caps)
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--wait", "--wait-timeout", "5"])
    assert result.exit_code == 1
    assert "error status (500)" in result.output
    assert "has not answered within" not in result.output
    assert "Check `nerdit doctor`" in result.output
    assert "Daemon back" not in result.output
    # It really did keep polling rather than bailing on the first 500.
    assert calls["n"] > 2


def test_wait_transport_error_then_fresh_boot_still_exits_zero(monkeypatch):
    """The narrowed swallow keeps the expected signal: a refused socket, then a boot."""
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    calls = {"n": 0}

    async def _caps() -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            return _daemon_caps(clock, 900.0, "0.3.0")
        if calls["n"] <= 4:  # socket down mid-restart — expected, swallowed
            raise _connect_error()
        return _daemon_caps(clock, 1000.4, "0.3.1")

    fake = _client(restart_daemon={"return_value": _accepted()})
    fake.get_capabilities = AsyncMock(side_effect=_caps)
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["daemon", "restart", "--yes", "--wait"])
    assert result.exit_code == 0
    assert "Daemon back" in result.output
    assert "0.3.1" in result.output
    assert "error status" not in result.output


# --- MCP half -----------------------------------------------------------------


def _mcp_client(handler) -> object:
    from nerdit.cli.client import NerditClient

    return NerditClient(
        host="localhost", port=9321, token=None, transport=httpx.MockTransport(handler)
    )


def test_mcp_tool_is_registered():
    from nerdit.mcp.tools.system import TOOLS

    assert TOOLS[-1].__name__ == "restart_daemon"


@pytest.mark.asyncio
async def test_mcp_impl_mints_and_forwards_keys():
    from nerdit.mcp.tools.system import _restart_daemon_impl

    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("idempotency-key"))
        return httpx.Response(202, json={"restarting": True})

    await _restart_daemon_impl(_mcp_client(handler))
    await _restart_daemon_impl(_mcp_client(handler), idempotency_key="from-caller")
    assert seen[0] and len(seen[0]) >= 32  # minted (there is no dry run here)
    assert seen[1] == "from-caller"


@pytest.mark.asyncio
async def test_mcp_impl_returns_structured_409():
    from nerdit.mcp.tools.system import _restart_daemon_impl

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "daemon.restart_in_progress",
                "message": "A daemon restart is already in progress.",
            },
        )

    result = await _restart_daemon_impl(_mcp_client(handler))
    assert result["error"]["code"] == "daemon.restart_in_progress"
    assert result["error"]["status"] == 409
