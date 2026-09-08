"""CLI tests for ``nerdit doctor`` (P13b WP6).

Two layers: the ``NerditClient.get_doctor`` method over ``httpx.MockTransport``
and the async command body driven with a fake client. The load-bearing
assertions: a healthy run exits 0, a top ``fail`` exits 1, and an unreachable
daemon renders a single local ``daemon`` fail row (no traceback) and exits 1.
"""

from __future__ import annotations

import httpx

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.doctor import _doctor_async

# -- client method -------------------------------------------------------------


async def test_get_doctor_hits_the_api_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"status": "ok", "checks": []})

    client = NerditClient(token="t", transport=httpx.MockTransport(handler))
    data = await client.get_doctor()
    assert seen["path"] == "/api/doctor"
    assert data["status"] == "ok"


# -- command body --------------------------------------------------------------


class _FakeClient:
    def __init__(self, data=None, exc=None) -> None:
        self._data = data
        self._exc = exc

    async def get_doctor(self) -> dict:
        if self._exc is not None:
            raise self._exc
        return self._data


async def test_doctor_healthy_exits_zero(monkeypatch):
    fake = _FakeClient(
        data={
            "status": "ok",
            "checks": [{"name": "docker", "status": "ok", "detail": "ok", "latency_ms": 3}],
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    assert await _doctor_async() == 0


async def test_doctor_renders_a_warn_docker_row_without_exiting_nonzero(monkeypatch, capsys):
    """(BUG-1) A buildx-less node warns; only a top ``fail`` sets the exit code.

    The docker row is deliberately ``warn`` when the BuildKit builder is
    missing — a node serving only pre-built images is healthy — so
    ``nerdit doctor`` must still exit 0 while showing the advice.
    """
    fake = _FakeClient(
        data={
            "status": "warn",
            "checks": [
                {
                    "name": "docker",
                    "status": "warn",
                    "detail": (
                        "Docker runtime responding; the BuildKit builder (docker buildx) "
                        "is NOT available to this daemon"
                    ),
                    "latency_ms": 4,
                }
            ],
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    assert await _doctor_async() == 0
    out = capsys.readouterr().out
    assert "buildx" in out


async def test_doctor_renders_the_never_linked_warn_and_still_exits_zero(monkeypatch, capsys):
    """(P34 D4) The never-linked row is advice, not a failure — and it renders.

    Two things are pinned here. First the exit code: an unlinked node is fully
    functional (D-ENT-2), the row is a ``warn``, and ``nerdit doctor`` must
    still exit 0 — nothing in this programme turns the absence of a Nerdit
    account into a gate. Second the render: the detail is the first doctor
    string to carry a quoted command with option dashes, and every server-
    derived string on this surface goes through ``_plain``/``escape`` (the P6
    ``MarkupError`` lesson), so the resume command must survive to the terminal
    intact rather than being eaten as Rich markup.
    """
    fake = _FakeClient(
        data={
            "status": "warn",
            "checks": [
                {
                    "name": "link",
                    "status": "warn",
                    "detail": (
                        "not linked — no Nerdit account attached; run 'nerdit link --device' (free)"
                    ),
                    "latency_ms": 0,
                }
            ],
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    assert await _doctor_async() == 0
    out = capsys.readouterr().out
    assert "--device" in out


async def test_doctor_top_fail_exits_one(monkeypatch):
    fake = _FakeClient(
        data={
            "status": "fail",
            "checks": [
                {"name": "docker", "status": "fail", "detail": "stub runtime", "latency_ms": 1}
            ],
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    assert await _doctor_async() == 1


async def test_doctor_unreachable_daemon_renders_local_fail_row(monkeypatch):
    fake = _FakeClient(exc=httpx.ConnectError("connection refused"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    # Keep the local registry deterministic (no real docker/socket/httpx probing).
    monkeypatch.setattr("nerdit.cli.checks.run_checks", lambda *a, **k: [])
    # No traceback escapes — the command body swallows it into a local fail row.
    assert await _doctor_async() == 1


async def test_doctor_unreachable_falls_back_to_local_registry(monkeypatch):
    from nerdit.cli import checks

    fake = _FakeClient(exc=httpx.ConnectError("connection refused"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)

    called = {}

    def fake_run_checks(*args, **kwargs):
        called["hit"] = True
        return [
            checks.CheckResult("Docker Engine", "fail", "Docker daemon not running", "start it"),
            checks.CheckResult("Disk space", "ok", "50.0% free"),
        ]

    monkeypatch.setattr("nerdit.cli.checks.run_checks", fake_run_checks)
    # A dead daemon still yields the local registry rows (useful diagnostics),
    # and the exit code stays 1.
    assert await _doctor_async() == 1
    assert called.get("hit") is True


async def test_doctor_remote_daemon_down_banners_local_scope(monkeypatch):
    # When configured for a remote daemon (nerdit connect) that is down, the
    # local fallback checks describe THIS machine — a banner must say so.
    from nerdit.cli.commands import doctor as doctor_mod

    class _RemoteClient(_FakeClient):
        host = "192.168.1.50"

    fake = _RemoteClient(exc=httpx.ConnectError("connection refused"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    monkeypatch.setattr("nerdit.cli.checks.run_checks", lambda *a, **k: [])

    printed: list[str] = []
    monkeypatch.setattr(
        doctor_mod.console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    )

    assert await _doctor_async() == 1
    joined = "\n".join(printed)
    assert "192.168.1.50" in joined
    assert "THIS machine" in joined


async def test_doctor_local_daemon_down_no_remote_banner(monkeypatch):
    from nerdit.cli.commands import doctor as doctor_mod

    class _LocalClient(_FakeClient):
        host = "127.0.0.1"

    fake = _LocalClient(exc=httpx.ConnectError("connection refused"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)
    monkeypatch.setattr("nerdit.cli.checks.run_checks", lambda *a, **k: [])

    printed: list[str] = []
    monkeypatch.setattr(
        doctor_mod.console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a))
    )

    assert await _doctor_async() == 1
    assert "THIS machine" not in "\n".join(printed)


async def test_doctor_local_registry_error_never_crashes(monkeypatch):
    fake = _FakeClient(exc=httpx.ConnectError("connection refused"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: fake, raising=False)

    def boom(*args, **kwargs):
        raise RuntimeError("registry blew up")

    monkeypatch.setattr("nerdit.cli.checks.run_checks", boom)
    # The fallback swallows its own error into a line, not a traceback; exit 1.
    assert await _doctor_async() == 1
