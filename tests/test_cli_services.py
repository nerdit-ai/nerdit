"""Unit tests for the P2 service CLI: client methods, table render, logs resolution."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from nerdit.cli.client import NerditClient
from nerdit.cli.display import display_service_submitted, display_service_table


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


# ---- client method header / payload behavior ----


@pytest.mark.asyncio
async def test_create_service_sends_idempotency_key_when_present():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo", "status": "building"})

    client = _make_client(handler)
    await client.create_service(name="demo", image="demo:1", idempotency_key="key-123")

    assert seen["path"] == "/api/services"
    assert seen["idem"] == "key-123"
    assert seen["body"]["name"] == "demo"
    assert seen["body"]["image"] == "demo:1"
    # Defaults are sent explicitly.
    assert seen["body"]["port"] == 8000
    assert seen["body"]["gpus"] == 0
    assert seen["body"]["restart_policy"] == "always"
    # Unset optional fields are omitted.
    for omitted in ("command", "script_path", "env", "vendor", "health_check"):
        assert omitted not in seen["body"]


@pytest.mark.asyncio
async def test_create_service_omits_idempotency_key_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    await client.create_service(name="demo", image="demo:1")
    assert seen["idem"] is None


@pytest.mark.asyncio
async def test_create_service_includes_optional_fields():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    await client.create_service(
        name="demo",
        image="demo:1",
        command="python app.py",
        script_path="/app/run.py",
        env={"K": "V"},
        vendor="nvidia",
        health_check={"path": "/healthz"},
    )
    body = seen["body"]
    assert body["command"] == "python app.py"
    assert body["script_path"] == "/app/run.py"
    assert body["env"] == {"K": "V"}
    assert body["vendor"] == "nvidia"
    assert body["health_check"] == {"path": "/healthz"}


@pytest.mark.asyncio
async def test_list_services_hits_api_prefix_with_limit():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = _make_client(handler)
    await client.list_services(status="running")
    assert seen["path"] == "/api/services"
    assert seen["query"] == {"limit": "50", "status": "running"}


@pytest.mark.asyncio
async def test_resolve_service_returns_none_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "no service"})

    client = _make_client(handler)
    assert await client.resolve_service("not-a-service") is None


@pytest.mark.asyncio
async def test_resolve_service_returns_dict_on_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/services/demo"
        return httpx.Response(200, json={"id": "svc-1", "name": "demo", "status": "running"})

    client = _make_client(handler)
    svc = await client.resolve_service("demo")
    assert svc is not None and svc["id"] == "svc-1"


@pytest.mark.asyncio
async def test_resolve_service_propagates_non_404_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.resolve_service("demo")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,verb,suffix",
    [
        ("stop_service", "POST", "/stop"),
        ("restart_service", "POST", "/restart"),
        ("remove_service", "DELETE", ""),
    ],
)
async def test_service_write_methods_target_and_verb(method, verb, suffix):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"id": "svc-1"})

    client = _make_client(handler)
    await getattr(client, method)("demo")
    assert seen["method"] == verb
    assert seen["path"] == f"/api/services/demo{suffix}"


# ---- table render ----


def test_display_service_table_renders_url_and_fields(capsys):
    services = [
        {
            "name": "demo",
            "status": "running",
            "restart_count": 2,
            "gpu_count": 1,
            "endpoint": {"host_port": 9400, "url": "http://127.0.0.1:9400"},
        },
        {"name": "bare", "status": "building", "restart_count": 0, "gpu_count": 0},
    ]
    display_service_table(services)
    out = capsys.readouterr().out
    assert "demo" in out
    assert "running" in out
    assert "9400" in out
    assert "bare" in out
    assert "building" in out


def test_display_service_table_renders_unmapped_status(capsys):
    # P5-runbook regression: a status outside the style map (e.g. 'completed')
    # used to render as "[]completed[/]" and crash Rich with a MarkupError.
    display_service_table(
        [{"name": "old", "status": "completed", "restart_count": 0, "gpu_count": 0}]
    )
    assert "completed" in capsys.readouterr().out


def test_display_service_table_prefers_public_url(capsys):
    services = [
        {
            "name": "demo",
            "status": "running",
            "restart_count": 0,
            "gpu_count": 0,
            "endpoint": {
                "host_port": 9400,
                "url": "http://127.0.0.1:9400",
                "public_url": "https://demo.example.com",
            },
        },
    ]
    display_service_table(services)
    out = capsys.readouterr().out
    assert "https://demo.example.com" in out
    assert "127.0.0.1:9400" not in out


def test_display_service_submitted_prefers_public_url(capsys):
    service = {
        "name": "demo",
        "status": "running",
        "gpu_count": 0,
        "endpoint": {"url": "http://127.0.0.1:9400", "public_url": "https://demo.example.com"},
    }
    display_service_submitted(service)
    out = capsys.readouterr().out
    assert "https://demo.example.com" in out
    assert "http://127.0.0.1:9400" in out


def test_display_service_submitted_loopback_only(capsys):
    service = {
        "name": "demo",
        "status": "running",
        "gpu_count": 0,
        "endpoint": {"url": "http://127.0.0.1:9400"},
    }
    display_service_submitted(service)
    out = capsys.readouterr().out
    assert "http://127.0.0.1:9400" in out
    assert "Local:" not in out


# ---- logs: service routes only + kind-aware terminal ----


class _FakeClient:
    """Minimal client double for the logs follower.

    Deliberately exposes ONLY the service surface (``get_service_logs`` /
    ``get_service``): the batch names ``get_logs``/``get_job``/
    ``resolve_service`` are absent, so any regression back to the batch routes
    blows up with an ``AttributeError`` instead of passing quietly (WP1).
    """

    def __init__(self, *, statuses: list[str]):
        self._statuses = statuses
        self.logs_calls: list[str] = []
        self.service_calls: list[str] = []

    async def get_service_logs(
        self, ident: str, since_id: int = 0, tail=None, *, grep=None, since=None
    ):
        # Mirrors ``NerditClient.get_service_logs`` including the P24a filter
        # keywords: the CLI passes them through unconditionally (the client owns
        # the omit-when-falsy rule), so a narrower double would fail on shape.
        self.logs_calls.append(ident)
        return []

    async def get_service_logs_page(
        self, ident: str, *, since_id: int = 0, tail=None, grep=None, since=None
    ):
        # The follower's read: entries plus the scan watermark
        # (``X-Nerdit-Scan-Watermark``, ``None`` on an unfiltered page).
        self.logs_calls.append(ident)
        return [], None

    async def get_service(self, ident: str):
        self.service_calls.append(ident)
        # Walk the scripted status sequence, then settle on the last value.
        status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]
        return {"id": ident, "status": status}


@pytest.mark.asyncio
async def test_logs_oneshot_uses_the_service_logs_route(monkeypatch):
    """One-shot ``nerdit logs <service>`` pins ``GET /api/services/{ident}/logs``.

    The identifier is passed through verbatim — the route resolves id *or*
    name, so there is no client-side resolve hop any more.
    """
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod

    fake = _FakeClient(statuses=["running"])
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    await logs_mod._logs_async("demo", follow=False)
    assert fake.logs_calls == ["demo"]
    # A one-shot read never polls the status route.
    assert fake.service_calls == []


@pytest.mark.asyncio
async def test_logs_follow_pins_both_service_routes(monkeypatch):
    """``--follow`` pins BOTH calls to the service plane and terminates.

    Regression guard: repointing only the log fetch and leaving the status poll
    on ``GET /jobs/{id}`` is how this silently 404s once the batch routes go.
    """
    import asyncio

    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod

    fake = _FakeClient(statuses=["running", "stopped"])
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))

    await logs_mod._logs_async("demo", follow=True)

    # Status polled through the service route, logs fetched through the service
    # route, and the loop stopped on the service-terminal ``stopped``.
    assert fake.service_calls == ["demo", "demo"]
    assert fake.logs_calls == ["demo", "demo", "demo"]


@pytest.mark.asyncio
async def test_follow_polling_does_not_stop_on_transient_states(monkeypatch):
    # restarting → degraded → building are transient; only ``stopped`` ends it.
    import asyncio

    from nerdit.cli.commands import logs as logs_mod

    fake = _FakeClient(statuses=["restarting", "degraded", "building", "stopped"])

    # Make the inter-poll sleep a no-op so the loop runs instantly.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))

    await logs_mod._follow_polling(fake, "svc-1")

    # It polled through all three transient states before the terminal ``stopped``.
    assert len(fake.logs_calls) >= 4


@pytest.mark.parametrize(
    "fn, client_method",
    [
        ("_stop_async", "stop_service"),
        ("_restart_async", "restart_service"),
        ("_rm_async", "remove_service"),
    ],
)
@pytest.mark.asyncio
async def test_lifecycle_commands_mint_idempotency_key(monkeypatch, fn, client_method):
    # stop/restart/rm must send a fresh Idempotency-Key (like serve/MCP), or they
    # fail on a daemon with require_idempotency_key=true (Codex review).
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    seen: dict = {}

    class _C:
        async def _record(self, name, *, idempotency_key=None, **_kw):
            seen["name"], seen["key"] = name, idempotency_key
            return {"id": name, "status": "stopped"}

        stop_service = restart_service = remove_service = _record

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _C())
    # ``rm`` widened for P14b WP-B1: it forwards purge/force keyword args.
    if fn == "_rm_async":
        await svc_mod._rm_async("demo", purge="secrets", force=False)
    else:
        await getattr(svc_mod, fn)("demo")
    assert seen["name"] == "demo"
    assert seen["key"]  # a non-empty key was minted


@pytest.mark.asyncio
async def test_follow_polling_service_stops_on_completed(monkeypatch):
    # A clean ``on-failure``/``no`` exit settles a service to ``completed``, which
    # IS service-terminal — ``logs -f`` must stop, not poll forever (Codex review).
    import asyncio

    from nerdit.cli.commands import logs as logs_mod

    fake = _FakeClient(statuses=["completed", "completed"])

    monkeypatch.setattr(asyncio, "sleep", AsyncMock(return_value=None))

    await logs_mod._follow_polling(fake, "svc-1")

    # The follower terminated on ``completed`` (initial fetch + one final drain,
    # no infinite loop).
    assert len(fake.logs_calls) == 2
    from nerdit.db.enums import TERMINAL_STATUSES

    assert "completed" in TERMINAL_STATUSES
    assert "stopped" in TERMINAL_STATUSES


# ---- nerdit services wait: the exit contract, one mapping, two callers (P23 WP2) ----


class _WaitClient:
    """Client double for ``services wait``: records the call, returns one outcome."""

    def __init__(self, outcome: dict | Exception):
        self._outcome = outcome
        self.wait_calls: list[dict] = []

    async def wait_for_service(self, ident: str, *, version=None, timeout: int = 60):
        self.wait_calls.append({"ident": ident, "version": version, "timeout": timeout})
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("converged", 0),
        ("failed", 1),
        ("superseded", 1),
        ("timeout", 3),
        # An outcome the CLI has never heard of must still exit 1 — never crash,
        # and never 2 (Typer owns 2 for usage errors).
        ("something-new", 1),
    ],
)
@pytest.mark.asyncio
async def test_services_wait_exit_codes(monkeypatch, outcome, expected):
    import typer

    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    fake = _WaitClient(
        {
            "outcome": outcome,
            "service_name": "demo",
            "version": 5,
            "phase": "healthy",
            "status": "running",
            "public_url": "https://host/demo",
            "waited_s": 1.0,
        }
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    if expected == 0:
        await svc_mod._wait_async("demo", timeout=60, version=None)
    else:
        with pytest.raises(typer.Exit) as ei:
            await svc_mod._wait_async("demo", timeout=60, version=None)
        assert ei.value.exit_code == expected
    assert fake.wait_calls == [{"ident": "demo", "version": None, "timeout": 60}]


@pytest.mark.asyncio
async def test_services_wait_forwards_version_and_timeout(monkeypatch):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    fake = _WaitClient({"outcome": "converged", "service_name": "demo"})
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    await svc_mod._wait_async("demo", timeout=120, version=7)

    # ``--version`` pins the generation server-side; ``--timeout`` is passed
    # through unclamped (the daemon owns the [1, 300] clamp).
    assert fake.wait_calls == [{"ident": "demo", "version": 7, "timeout": 120}]


@pytest.mark.asyncio
async def test_services_wait_request_error_exits_one(monkeypatch):
    import typer

    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    fake = _WaitClient(
        httpx.ConnectError(
            "daemon is down",
            request=httpx.Request("GET", "http://localhost:9321/api/services/demo/wait"),
        )
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    with pytest.raises(typer.Exit) as ei:
        await svc_mod._wait_async("demo", timeout=60, version=None)
    assert ei.value.exit_code == 1


@pytest.mark.asyncio
async def test_services_wait_escapes_a_markup_hostile_name(monkeypatch):
    """A caller-supplied ident is Rich-escaped before it reaches the spinner.

    Regression for the P6 ``MarkupError`` lesson: an ident like ``[/x]`` is a
    dangling closing tag. Unescaped it blows up inside ``console.status`` —
    which sits *inside* the try — so the request is never even made and the
    user gets a bogus "request failed". The assertion is therefore that the
    client IS called: no local short-circuit.
    """
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    fake = _WaitClient({"outcome": "converged", "service_name": "[/x]"})
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    await svc_mod._wait_async("[/x]", timeout=60, version=None)

    assert fake.wait_calls == [{"ident": "[/x]", "version": None, "timeout": 60}]


def test_wait_exit_code_maps_every_outcome():
    """The extracted mapping itself: 0 / 3 / 1, never 2, unknown falls to 1."""
    from nerdit.cli.display import wait_exit_code

    assert wait_exit_code({"outcome": "converged"}) == 0
    assert wait_exit_code({"outcome": "timeout"}) == 3
    assert wait_exit_code({"outcome": "failed"}) == 1
    assert wait_exit_code({"outcome": "superseded"}) == 1
    assert wait_exit_code({"outcome": "who-knows"}) == 1
    assert wait_exit_code({}) == 1


@pytest.mark.parametrize(
    "outcome, expected",
    [("converged", 0), ("failed", 1), ("superseded", 1), ("timeout", 3)],
)
@pytest.mark.asyncio
async def test_deploy_wait_still_maps_through_the_extracted_helper(monkeypatch, outcome, expected):
    """Regression: ``deploy --wait`` keeps its codes now that the mapping moved.

    ``_maybe_wait`` no longer inlines the branches — it calls
    ``wait_exit_code``, the same helper ``services wait`` uses.
    """
    import typer

    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import deploy as deploy_mod

    fake = _WaitClient({"outcome": outcome, "service_name": "demo", "waited_s": 1.0})
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    service = {"last_deploy": {"version": 5}, "build_version": 5}
    if expected == 0:
        await deploy_mod._maybe_wait(fake, "demo", service, True, 60)
    else:
        with pytest.raises(typer.Exit) as ei:
            await deploy_mod._maybe_wait(fake, "demo", service, True, 60)
        assert ei.value.exit_code == expected
    assert fake.wait_calls == [{"ident": "demo", "version": 5, "timeout": 60}]
