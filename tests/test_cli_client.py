"""Unit tests for ``nerdit.cli.client.NerditClient`` using ``httpx.MockTransport``."""

from __future__ import annotations

import contextlib
import json

import httpx
import pytest

from nerdit.cli.client import NerditClient, get_configured_client


def _make_client(handler, *, token: str | None = None) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=token,
        transport=httpx.MockTransport(handler),
    )


# ---- base URL construction ----


@pytest.mark.parametrize(
    "host, expected",
    [
        ("127.0.0.1", "http://127.0.0.1:9321"),
        ("localhost", "http://localhost:9321"),
        ("nerdit.example", "http://nerdit.example:9321"),
        ("::1", "http://[::1]:9321"),
        ("[::1]", "http://[::1]:9321"),
        ("fd00::5", "http://[fd00::5]:9321"),
    ],
)
def test_base_url_brackets_ipv6_literals(host, expected):
    # An unbracketed `http://::1:9321` makes httpx raise InvalidURL, which is
    # NOT an HTTPError and so escapes every caller's error mapping.
    client = NerditClient(host=host, port=9321)
    assert client._base_url == expected
    assert httpx.URL(client._base_url).port == 9321


# ---- simple GET / POST happy paths ----


@pytest.mark.asyncio
async def test_health_returns_parsed_json():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"status": "ok", "version": "0.2.0"})

    client = _make_client(handler)
    result = await client.health()
    assert result == {"status": "ok", "version": "0.2.0"}
    assert captured == {"method": "GET", "path": "/health"}


@pytest.mark.asyncio
async def test_health_raises_on_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Missing token"})

    client = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.health()
    assert exc_info.value.response.status_code == 401


@pytest.mark.asyncio
async def test_health_raises_on_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.health()
    assert exc_info.value.response.status_code == 500


@pytest.mark.asyncio
async def test_token_sent_as_bearer_header():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"status": "ok"})

    client = _make_client(handler, token="abc123")
    await client.health()
    assert seen["auth"] == "Bearer abc123"


@pytest.mark.asyncio
async def test_list_gpus():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/gpus"
        return httpx.Response(200, json=[{"id": "GPU-0"}, {"id": "GPU-1"}])

    client = _make_client(handler)
    gpus = await client.list_gpus()
    assert [g["id"] for g in gpus] == ["GPU-0", "GPU-1"]


# ---- service logs (P6) ----


@pytest.mark.asyncio
async def test_get_service_logs_hits_api_path_with_params():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json=[{"id": 7, "message": "up"}])

    client = _make_client(handler)
    logs = await client.get_service_logs("my-app", since_id=5, tail=100)
    assert logs == [{"id": 7, "message": "up"}]
    assert captured["method"] == "GET"
    assert captured["path"] == "/api/services/my-app/logs"
    assert captured["query"] == {"since_id": "5", "tail": "100"}


@pytest.mark.asyncio
async def test_get_service_logs_omits_tail_when_none():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = _make_client(handler)
    await client.get_service_logs("svc-1")
    assert seen["query"] == {"since_id": "0"}


# ---- audit log (P6) ----


@pytest.mark.asyncio
async def test_get_audit_default_params():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = _make_client(handler)
    page = await client.get_audit()
    assert page == {"items": [], "next_cursor": None}
    assert captured["method"] == "GET"
    assert captured["path"] == "/api/audit"
    assert captured["query"] == {"limit": "50"}


@pytest.mark.asyncio
async def test_get_audit_forwards_filters_and_cursor():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = _make_client(handler)
    await client.get_audit(
        action="deploy.create",
        result="ok",
        target="my-app",
        target_type="service",
        cursor="c123",
        limit=10,
    )
    assert seen["query"] == {
        "limit": "10",
        "action": "deploy.create",
        "result": "ok",
        "target": "my-app",
        "target_type": "service",
        "cursor": "c123",
    }


@pytest.mark.asyncio
async def test_get_audit_raises_on_403():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"code": "auth.forbidden", "message": "admin only"})

    client = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.get_audit()
    assert exc_info.value.response.status_code == 403


# ---- secrets idempotency plumbing (P6) ----


@pytest.mark.asyncio
async def test_set_secrets_sends_idempotency_key_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"service": "my-app", "keys": ["API_KEY"]})

    client = _make_client(handler)
    await client.set_secrets("my-app", {"API_KEY": "s3cret"}, idempotency_key="idem-1")
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/secrets/my-app"
    assert captured["idem"] == "idem-1"
    assert captured["body"] == {"values": {"API_KEY": "s3cret"}}


@pytest.mark.asyncio
async def test_set_secrets_omits_header_when_no_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service": "my-app", "keys": []})

    client = _make_client(handler)
    await client.set_secrets("my-app", {"A": "b"})
    assert seen["idem"] is None


@pytest.mark.asyncio
async def test_delete_secret_sends_idempotency_key_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"deleted": ["API_KEY"]})

    client = _make_client(handler)
    await client.delete_secret("my-app", "API_KEY", idempotency_key="idem-2")
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/api/secrets/my-app/API_KEY"
    assert captured["idem"] == "idem-2"


@pytest.mark.asyncio
async def test_delete_secrets_sends_idempotency_key_header():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"deleted": ["A", "B"]})

    client = _make_client(handler)
    await client.delete_secrets("my-app", idempotency_key="idem-3")
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/api/secrets/my-app"
    assert captured["idem"] == "idem-3"


# ---- service delete: heavier since P14b (F9) ----


@pytest.mark.asyncio
async def test_remove_service_uses_long_httpx_timeout():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["query"] = dict(request.url.params)
        captured["idem"] = request.headers.get("idempotency-key")
        # httpx surfaces the per-request timeout as a {connect,read,write,pool}
        # dict on the request extensions — the same probe used in
        # tests/test_cli_deploy.py for the deploy legs.
        captured["httpx_timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"id": "svc1", "name": "my-app", "deleted": True})

    client = _make_client(handler)
    out = await client.remove_service(
        "my-app",
        purge="secrets,data,images",
        force=True,
        idempotency_key="idem-del",
    )
    assert out["deleted"] is True
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/api/services/my-app"
    assert captured["query"] == {"purge": "secrets,data,images", "force": "true"}
    assert captured["idem"] == "idem-del"
    # The DELETE route runs teardown + purge (data-dir rmtree, a removal per
    # image tag, two full docker image listings) synchronously inside the
    # request: the pre-P14b 15 s budget aborted the CLI mid-delete while the
    # server went on to succeed.
    assert captured["httpx_timeout"]["read"] == 120.0


@pytest.mark.asyncio
async def test_remove_service_defaults_omit_force():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["query"] = dict(request.url.params)
        captured["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"id": "svc1", "name": "my-app", "deleted": True})

    client = _make_client(handler)
    await client.remove_service("my-app")
    assert captured["query"] == {"purge": "secrets"}
    assert captured["idem"] is None


# ---- restart_daemon (P23 WP3) ----


@pytest.mark.asyncio
async def test_restart_daemon_sends_body_and_idempotency_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        captured["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(
            202,
            json={
                "restarting": True,
                "in_flight_builds": 1,
                "in_flight_runs": 0,
                "drain_timeout_s": 30,
            },
        )

    client = _make_client(handler)
    out = await client.restart_daemon(drain_timeout_s=30, idempotency_key="idem-restart")
    assert out["restarting"] is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/daemon/restart"
    assert captured["idem"] == "idem-restart"
    assert captured["body"] == {"drain_timeout_s": 30}


@pytest.mark.asyncio
async def test_restart_daemon_omits_body_and_header_when_unset():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["idem"] = request.headers.get("idempotency-key")
        captured["content"] = request.content
        return httpx.Response(202, json={"restarting": True})

    client = _make_client(handler)
    await client.restart_daemon()
    # No body at all: the server's own 60 s drain default stays the server's,
    # never a client-side copy of it. No key ⇒ no header (the daemon's own
    # 400 idempotency_key_required envelope must be able to reach a caller).
    assert captured["content"] == b""
    assert captured["idem"] is None


# ---- get_configured_client + config.toml loading ----


def test_get_configured_client_falls_back_to_localhost(tmp_path, monkeypatch):
    # No config.toml → localhost + no token
    monkeypatch.setenv("HOME", str(tmp_path))
    # Ensure load_settings inside the helper picks up the env
    client = get_configured_client()
    assert client._base_url == "http://127.0.0.1:9321"
    assert client._headers == {}


def test_get_configured_client_reads_remote_section(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".nerdit"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(
        "[client]\n"
        'remote_host = "my-box.example.com"\n'
        "remote_port = 4242\n"
        'auth_token = "topsecret"\n'
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    client = get_configured_client()
    assert client._base_url == "http://my-box.example.com:4242"
    assert client._headers == {"Authorization": "Bearer topsecret"}


def test_get_configured_client_uses_daemon_token_when_no_remote(tmp_path, monkeypatch):
    cfg_dir = tmp_path / ".nerdit"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text('[daemon]\nauth_token = "localtoken"\n')
    monkeypatch.setenv("HOME", str(tmp_path))

    client = get_configured_client()
    assert client._base_url == "http://127.0.0.1:9321"
    assert client._headers == {"Authorization": "Bearer localtoken"}


def test_get_configured_client_uses_daemon_port_locally(monkeypatch):
    # Regression (P23 WP0): the local fallback used to hardcode DEFAULT_PORT,
    # so a CLI on a non-default-port install silently dialled 9321.
    from nerdit.config import settings as settings_mod

    patched = settings_mod.NerditSettings(
        daemon=settings_mod.DaemonSettings(port=9333, auth_token="t")
    )
    monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **kw: patched)

    client = get_configured_client()
    assert client._base_url.endswith(":9333")
    assert client._base_url == "http://127.0.0.1:9333"
    assert client._headers == {"Authorization": "Bearer t"}


def test_get_configured_client_remote_section_beats_daemon_port(monkeypatch):
    # A configured [client] remote still wins over [daemon].port.
    from nerdit.config import settings as settings_mod

    patched = settings_mod.NerditSettings(
        daemon=settings_mod.DaemonSettings(port=9333, auth_token="localtoken"),
        client=settings_mod.ClientSettings(
            remote_host="my-box.example.com", remote_port=4242, auth_token="remotetoken"
        ),
    )
    monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **kw: patched)

    client = get_configured_client()
    assert client._base_url == "http://my-box.example.com:4242"
    assert client._headers == {"Authorization": "Bearer remotetoken"}


# ---- P25 self-service token surface (D-P25-4) ----


@pytest.mark.asyncio
async def test_get_self_token_hits_the_self_path():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json={"id": "tok-1", "rotatable": True})

    client = _make_client(handler)
    assert await client.get_self_token() == {"id": "tok-1", "rotatable": True}
    assert captured["method"] == "GET"
    assert captured["path"] == "/api/tokens/self"


@pytest.mark.asyncio
async def test_rotate_self_token_sends_extend_and_no_idempotency_key():
    """No key by design: a replay withholds the plaintext (D-P25-4 trade 2)."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "tok-1", "token": "nrd_new"})

    client = _make_client(handler)
    await client.rotate_self_token(extend=True, expires_in_s=3600)
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/tokens/self/rotate"
    assert captured["idem"] is None
    assert captured["body"] == {"extend": True, "expires_in_s": 3600}


@pytest.mark.asyncio
async def test_rotate_self_token_omits_expires_in_s_when_unset():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "tok-1", "token": "nrd_new"})

    client = _make_client(handler)
    await client.rotate_self_token()
    assert seen["body"] == {"extend": False}


# ---- workspace URL encoding (P29 D11) ----
#
# The four workspace methods interpolate two caller-free-form segments. Before
# the fix they went in raw, so '#' started a fragment (wrong path + 404), '?'
# truncated into a query, a literal '%2f' decoded server-side into a slash, and
# a name containing '/' re-routed the URL entirely. Encoded, every one of them
# reaches the daemon and earns its structured 422/404.


@pytest.mark.asyncio
async def test_read_workspace_file_percent_encodes_the_path():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw_path"] = request.url.raw_path.decode()
        seen["decoded"] = request.url.path
        return httpx.Response(200, text="print('hi')")

    client = _make_client(handler)
    await client.read_workspace_file("app", "a#b?c%2fd.py")
    assert seen["raw_path"] == "/api/workspaces/app/files/a%23b%3Fc%252fd.py"
    # Round-trips: the daemon decodes back to exactly the requested path.
    assert seen["decoded"] == "/api/workspaces/app/files/a#b?c%2fd.py"


@pytest.mark.asyncio
async def test_read_workspace_file_keeps_real_slashes():
    """Slashes are real structure for the route's ``{path:path}`` capture."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(200, text="x")

    client = _make_client(handler)
    await client.read_workspace_file("app", "src/lib/main.py")
    assert seen["raw_path"] == "/api/workspaces/app/files/src/lib/main.py"


@pytest.mark.asyncio
async def test_workspace_name_is_one_encoded_segment():
    """A hostile name stays ONE path segment so the daemon can refuse it."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        raw = request.url.raw_path.decode()
        seen.setdefault("paths", []).append(raw)
        if "/files/" in raw:
            return httpx.Response(200, text="x")
        return httpx.Response(200, json={})

    client = _make_client(handler)
    await client.get_workspace("a/b")
    await client.read_workspace_file("a/b", "main.py")
    await client.write_workspace_files("a/b", {"main.py": "x"})
    await client.deploy_workspace("a/b")

    assert seen["paths"] == [
        "/api/workspaces/a%2Fb",
        "/api/workspaces/a%2Fb/files/main.py",
        "/api/workspaces/a%2Fb/files",
        "/api/workspaces/a%2Fb/deploy",
    ]


# ---- dot segments survive to the daemon (P29 review round-2, Codex 3803596898)
#
# ``quote`` leaves '.' literal and httpx applies RFC 3986 remove_dot_segments
# when it builds the request target — ON THE WIRE. So ``foo/../main.py`` used to
# be sent as ``…/files/main.py``: ``read_app_file`` returned a DIFFERENT valid
# file instead of the promised structured 422, and a '.'/'..' workspace NAME
# rerouted the URL the same way. Pure-dot segments now ride as ``%2E``, which
# httpx leaves alone and Starlette decodes without ever collapsing.


@pytest.mark.asyncio
async def test_read_workspace_file_preserves_dot_segments():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(422, json={"code": "workspace.invalid_path", "message": "no"})

    client = _make_client(handler)
    with contextlib.suppress(httpx.HTTPStatusError):
        await client.read_workspace_file("app", "foo/../main.py")
    assert seen["raw_path"] == "/api/workspaces/app/files/foo/%2E%2E/main.py"


@pytest.mark.asyncio
async def test_dot_workspace_names_stay_one_segment():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("paths", []).append(request.url.raw_path.decode())
        return httpx.Response(200, json={})

    client = _make_client(handler)
    await client.get_workspace("..")
    await client.get_workspace(".")

    assert seen["paths"] == ["/api/workspaces/%2E%2E", "/api/workspaces/%2E"]


@pytest.mark.asyncio
async def test_dotted_but_not_pure_dot_segments_are_unchanged():
    """Only pure-dot segments are re-encoded — remove_dot_segments touches
    nothing else, so every other path stays byte-identical to before."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(200, text="x")

    client = _make_client(handler)
    await client.read_workspace_file("app", "src/.hidden/a.b.c/...py")
    assert seen["raw_path"] == "/api/workspaces/app/files/src/.hidden/a.b.c/...py"


@pytest.mark.asyncio
async def test_wait_for_service_preserves_query_and_long_poll_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/services/demo/wait"
        assert dict(request.url.params) == {"timeout": "75", "version": "0"}
        assert request.extensions["timeout"]["read"] == 105
        return httpx.Response(200, json={"outcome": "ready"})

    result = await _make_client(handler).wait_for_service("demo", version=0, timeout=75)
    assert result == {"outcome": "ready"}


@pytest.mark.asyncio
async def test_wait_for_service_floors_only_the_transport_deadline():
    """A `--timeout -100` must still reach the daemon and hit its clamp.

    The transport deadline is `max(1, timeout) + 30`: unfloored it would be
    negative, and httpx fails a negative timeout before any I/O — so
    `nerdit services wait --timeout -100` answered `connection_error` against a
    healthy daemon instead of the documented [1, 300] clamp. The request value
    stays unfloored on purpose: the daemon owns the clamp and answers with the
    value it actually used.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == {"timeout": "-100"}
        assert request.extensions["timeout"]["read"] == 31
        return httpx.Response(200, json={"outcome": "converged", "timeout": 1})

    result = await _make_client(handler).wait_for_service("demo", timeout=-100)
    assert result["outcome"] == "converged"
