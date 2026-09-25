"""CLI tests for ``nerdit link`` / ``nerdit unlink`` (P27 WP-C2).

Same three-layer structure as ``test_cli_trust.py``: the ``NerditClient``
methods over ``httpx.MockTransport`` (wire shape: path, bearer,
``Idempotency-Key``, exact JSON body), the async command bodies against a fake
client, and argv-level parsing through Typer's ``CliRunner``.

The security-relevant assertion, repeated on every path that can print: the
link **code value never appears in output** — not on success, not on a daemon
refusal, not on a transport failure. The code is a low-entropy human-typed
secret; a console line is a scrollback, a CI log and a screenshot.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands import link as link_mod
from nerdit.cli.commands.link import (
    DEFAULT_LINK_API_URL,
    DEFAULT_LINK_RELAY_URL,
    _link_async,
    _unlink_async,
)

CODE = "ZZTOPSECRET42"


@pytest.fixture(autouse=True)
def _no_daemon_restart(monkeypatch):
    """Never restart a real daemon from a CLI rendering test."""
    restart = AsyncMock(return_value=0)
    monkeypatch.setattr(link_mod, "_restart_daemon", restart)
    monkeypatch.setattr(
        "nerdit.config.settings.get_client_config", lambda: ("127.0.0.1", 9321, None)
    )
    return restart


_CLAIM_OK = {
    "node_id": "11111111-2222-3333-4444-555555555555",
    "slug": "brave-otter",
    "relay_url": "wss://relay.localhost/link",
    "enabled": True,
    "verifier_fingerprint": "ab12cd34ef56",
    "requires_restart": True,
    "restart_keys": ["link.node_id", "link.slug", "link.enabled"],
}

#: A ``LinkDeviceStartView`` as the route contract pins it: display material,
#: the daemon-local machine facts, and a non-secret opaque ``session`` that
#: selects the daemon's pending slot. The 160-bit ``device_code`` is absent by
#: construction — it never leaves the daemon process (D-P34-1).
_DEVICE_START_OK = {
    "session": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
    "user_code": "7Q3M-X2VF-9KHT",
    "verification_uri": "https://app.localhost/link",
    "verification_uri_complete": "https://app.localhost/link#7Q3M-X2VF-9KHT",
    # Full 64 hex, as the daemon returns it; the CLI shows the first eight.
    "credential_fingerprint": "3f9c2a1b" + "e" * 56,
    "hostname": "buildbox-01",
    "daemon_version": "0.6.0",
    "os": "linux",
    "expires_in": 900,
    "interval": 5,
}

#: The approved answer: ``status: "linked"`` plus every ``LinkClaimView`` field.
_DEVICE_LINKED = {
    "status": "linked",
    "node_id": "a1b2c3d4e5f6",
    "slug": "corvid-mesa",
    "relay_url": "wss://relay.localhost/link",
    "enabled": True,
    "verifier_fingerprint": "ab12cd34ef56",
    "nodes_base_domain": "nodes.example",
    "requires_restart": True,
    "restart_keys": ["link.enabled", "link.node_id", "link.slug", "mcp.http_enabled"],
    "mcp_http_enabled": True,
    "mcp_skipped_reason": None,
}

_DEVICE_PENDING = {"status": "pending", "interval": 5}
_DEVICE_SLOW_DOWN = {"status": "slow_down", "interval": 10}

#: A persisted ``[link]`` section for a node that IS linked — the state the
#: D-P34-3 short-circuit exists to answer with one line and exit 0.
_LINKED_SECTION = {
    "section": "link",
    "values": {"node_id": "11111111-2222-3333-4444-555555555555", "slug": "brave-otter"},
    "etag": "e",
}

#: A pre-auth key shaped like the shipped cloud grammar (``nk_`` + 32 Crockford
#: symbols). Its VALUE is the assertion target of the custody tests below: it
#: must reach the daemon and appear nowhere else.
PREAUTH_KEY = "nk_" + "ABCDEFGHJKMNPQRSTVWXYZ0123456789"

_REFRESH_OK = {
    "node_id": "11111111-2222-3333-4444-555555555555",
    "slug": "brave-otter",
    "nodes_base_domain": "nodes.example",
    "changed": True,
    "requires_restart": True,
    "restart_keys": ["link.nodes_base_domain"],
}


# -- client methods ---------------------------------------------------------------


async def test_claim_link_posts_code_with_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_CLAIM_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.claim_link(
        code=CODE, api_url="https://app.localhost", idempotency_key="key-1"
    )

    assert result == _CLAIM_OK
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/link/claim"
    assert seen["auth"] == "Bearer t"
    assert seen["idem"] == "key-1"
    # relay_url is ABSENT when not supplied — omitting it means "keep the
    # stored one", which is not the same statement as sending it again.
    assert seen["body"] == {"code": CODE, "api_url": "https://app.localhost", "enable": True}


async def test_claim_link_includes_relay_url_and_enable_flag_when_given():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_CLAIM_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.claim_link(
        code=CODE,
        api_url="https://app.localhost",
        relay_url="wss://relay.localhost/link",
        enable=False,
    )
    assert seen["body"] == {
        "code": CODE,
        "api_url": "https://app.localhost",
        "enable": False,
        "relay_url": "wss://relay.localhost/link",
    }


async def test_claim_link_sends_a_preauth_key_under_the_key_field_the_daemon_schema_accepts():
    """(D-P34-6) The cross-layer drift pin the two mocked suites cannot be.

    The CLI suite stubs the client and the route suite posts JSON by hand, so
    each half can be self-consistent while the wire between them is broken —
    which is exactly how a pre-auth key once travelled as ``code`` and met a
    validator that refuses every ``nk_`` value. This test drives the REAL
    client and then validates the exact bytes it emitted against the REAL
    daemon request schema: the key must ride its own ``key`` field, and the
    daemon model must accept the body verbatim.
    """
    from nerdit.daemon.schemas.link import LinkClaimRequest

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_CLAIM_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.claim_link(
        key=PREAUTH_KEY, api_url="https://app.localhost", idempotency_key="key-1"
    )

    assert seen["body"] == {"key": PREAUTH_KEY, "api_url": "https://app.localhost", "enable": True}
    parsed = LinkClaimRequest.model_validate(seen["body"])
    assert parsed.key == PREAUTH_KEY
    assert parsed.code is None


async def test_claim_link_refuses_both_grants_or_neither_before_any_request():
    """The client mirrors the route's exactly-one-of, without a network hop.

    A programming error that supplies both grants (or neither) should fail in
    the caller's own process rather than burn a round trip to be told the same
    thing by the route's 422 — and the transport below proves nothing left.
    """

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request may leave the client")

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="exactly one"):
        await client.claim_link(api_url="https://app.localhost")
    with pytest.raises(ValueError, match="exactly one"):
        await client.claim_link(code=CODE, key=PREAUTH_KEY, api_url="https://app.localhost")


async def test_start_and_poll_device_link_post_the_paths_and_bodies_the_routes_bind():
    """(P34 D1) The device pair exists on the real client and speaks the routes' shapes.

    The CLI device tests attach ``AsyncMock`` attributes to a stub, which would
    absorb ANY method name — so this is the test that goes red if the real
    ``NerditClient`` loses (or renames) ``start_device_link``/``poll_device_link``
    again. Both emitted bodies are validated against the daemon's own request
    schemas, the same closed loop as the claim's key-field pin above.
    """
    from nerdit.daemon.schemas.link import LinkDevicePollRequest, LinkDeviceStartRequest

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "path": request.url.path,
                "idem": request.headers.get("idempotency-key"),
                "body": httpx.Response(200, content=request.content).json(),
            }
        )
        return httpx.Response(200, json={"status": "pending", "interval": 5})

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.start_device_link(
        api_url="https://app.localhost",
        relay_url="wss://relay.localhost/link",
        idempotency_key="start-1",
    )
    await client.poll_device_link(session="0f" * 16, idempotency_key="poll-1")

    assert [s["path"] for s in seen] == ["/api/link/device", "/api/link/device/poll"]
    assert [s["idem"] for s in seen] == ["start-1", "poll-1"]
    start_body, poll_body = seen[0]["body"], seen[1]["body"]
    assert start_body == {
        "api_url": "https://app.localhost",
        "enable": True,
        "relay_url": "wss://relay.localhost/link",
    }
    assert poll_body == {"session": "0f" * 16}
    LinkDeviceStartRequest.model_validate(start_body)
    LinkDevicePollRequest.model_validate(poll_body)


async def test_claim_link_refusal_raises_http_status_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "link.claim_refused",
                "message": "The cloud refused this link code.",
                "cloud_code": "link_code_expired",
                "cloud_status": 410,
            },
        )

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await client.claim_link(code=CODE, api_url="https://app.localhost")
    assert excinfo.value.response.status_code == 409


async def test_unlink_node_deletes_with_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(
            200,
            json={
                "was_linked": True,
                "tunnel_stopped": True,
                "key_removed": True,
                "requires_restart": True,
            },
        )

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.unlink_node(idempotency_key="key-2")
    assert result["was_linked"] is True
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/link"
    assert seen["idem"] == "key-2"


async def test_refresh_link_posts_the_api_url_with_idempotency_key():
    """(P26 WP-H) The refresh carries NO code — only the console origin."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_REFRESH_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.refresh_link(api_url="https://app.localhost", idempotency_key="key-3")

    assert result == _REFRESH_OK
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/link/refresh"
    assert seen["idem"] == "key-3"
    assert seen["body"] == {"api_url": "https://app.localhost"}


# -- async command bodies ----------------------------------------------------------


def _fake_client(**overrides) -> SimpleNamespace:
    calls = {
        "claim_link": AsyncMock(return_value=dict(_CLAIM_OK)),
        "unlink_node": AsyncMock(
            return_value={
                "was_linked": True,
                "tunnel_stopped": True,
                "key_removed": True,
                "requires_restart": True,
            }
        ),
        "get_config": AsyncMock(return_value={"section": "link", "values": {}, "etag": "e"}),
        "get_capabilities": AsyncMock(return_value={"link": {"enabled": False}}),
        "refresh_link": AsyncMock(return_value=dict(_REFRESH_OK)),
        # (P34 D1) The device pair. The CLI is a thin renderer over these two
        # daemon routes, so the fake answers with the view shapes the route
        # contract pins and nothing else — no ``device_code`` field exists to
        # leak, here or on the wire.
        "start_device_link": AsyncMock(return_value=dict(_DEVICE_START_OK)),
        "poll_device_link": AsyncMock(return_value=dict(_DEVICE_LINKED)),
    }
    calls.update(overrides)
    return SimpleNamespace(**calls)


@pytest.fixture
def fake_client(monkeypatch):
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    return client


@pytest.mark.parametrize(
    ("reason", "must_say", "must_not_say"),
    [
        (
            "no_auth_token: no [daemon].auth_token — the transport is bearer-only",
            "nerdit init --auth-token-only",
            "http_enabled=true",
        ),
        (
            "mcp_extra_missing: the mcp extra is not installed in this build",
            "nerdit[mcp]",
            "http_enabled=true",
        ),
        ("config_error: could not stage [mcp].http_enabled: x", "http_enabled=true", ""),
    ],
)
async def test_claim_mcp_skip_remedy_matches_the_missing_prerequisite(
    fake_client, capsys, reason, must_say, must_not_say
):
    """Codex round 1: never hand the operator the flag while the precondition
    the daemon refuses to boot without is still missing."""
    fake_client.claim_link.return_value = {
        **_CLAIM_OK,
        "mcp_http_enabled": False,
        "mcp_skipped_reason": reason,
    }
    await _link_async(CODE, "https://app.localhost", None, True)
    out = capsys.readouterr().out
    assert "Remote MCP: not enabled" in out
    assert must_say in out
    if must_not_say:
        assert must_not_say not in out


async def test_claim_reports_remote_mcp_enabled(fake_client, capsys):
    fake_client.claim_link.return_value = {**_CLAIM_OK, "mcp_http_enabled": True}
    await _link_async(CODE, "https://app.localhost", None, True)
    assert "Remote MCP: enabled" in capsys.readouterr().out


async def test_claim_restarts_without_a_service_unit(fake_client, capsys, _no_daemon_restart):
    """A detached daemon can apply its saved link without launchd registration."""
    await _link_async(CODE, "https://app.localhost", None, True)

    fake_client.claim_link.assert_awaited_once()
    kwargs = fake_client.claim_link.await_args.kwargs
    assert kwargs["code"] == CODE
    assert kwargs["api_url"] == "https://app.localhost"
    assert kwargs["enable"] is True
    assert kwargs["idempotency_key"]  # minted per invocation

    out = capsys.readouterr().out
    assert "brave-otter" in out
    assert _CLAIM_OK["node_id"] in out
    assert "Daemon restarted" in out
    _no_daemon_restart.assert_awaited_once_with(
        drain_timeout_s=60, yes=True, wait=True, wait_timeout=60
    )
    assert CODE not in out


async def test_claim_defaults_to_production_endpoints(fake_client, capsys):
    """(P30 D-P30-10) The two-command onboarding: no flags, production cloud."""
    await _link_async(CODE, None, None, True)

    kwargs = fake_client.claim_link.await_args.kwargs
    assert kwargs["api_url"] == DEFAULT_LINK_API_URL == "https://app.nerdit.ai"
    assert kwargs["relay_url"] == DEFAULT_LINK_RELAY_URL == "wss://relay.nerdit.ai/v1/connect"
    # Pinned to the path the relay actually serves (``relay/control.py``
    # ``NODE_CONNECT_PATH``) and the only path the production edge publishes:
    # the previous default ``/link`` 404-looped every fresh install
    # (E2E on 2026-08-23, fixed in 0.5.3).
    assert CODE not in capsys.readouterr().out


async def test_reclaim_keeps_a_configured_relay(monkeypatch, capsys):
    """Omitting --relay-url must not repoint a self-hosted node at the
    production relay: the persisted value wins over the P30 default."""
    client = _fake_client(
        get_config=AsyncMock(
            return_value={"values": {"relay_url": "wss://relay.internal.example/link"}}
        ),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(CODE, "https://console.internal.example", None, True)

    assert client.claim_link.await_args.kwargs["relay_url"] == "wss://relay.internal.example/link"


async def test_first_claim_falls_back_to_the_production_relay(monkeypatch, capsys):
    client = _fake_client(get_config=AsyncMock(return_value={"values": {"relay_url": ""}}))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(CODE, None, None, True)

    assert client.claim_link.await_args.kwargs["relay_url"] == DEFAULT_LINK_RELAY_URL


async def test_explicit_relay_flag_skips_the_config_read(monkeypatch, capsys):
    def _never(*args, **kwargs):  # pragma: no cover — must not be reached
        raise AssertionError("an explicit --relay-url needs no config read")

    client = _fake_client(get_config=AsyncMock(side_effect=_never))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(CODE, None, "wss://relay.localhost/link", True)

    assert client.claim_link.await_args.kwargs["relay_url"] == "wss://relay.localhost/link"


async def test_remote_target_does_not_restart_automatically(
    fake_client, monkeypatch, capsys, _no_daemon_restart
):
    monkeypatch.setattr(
        "nerdit.config.settings.get_client_config", lambda: ("gpu-box", 9321, "tok")
    )
    await _link_async(CODE, "https://app.localhost", None, True)
    _no_daemon_restart.assert_not_awaited()
    out = capsys.readouterr().out
    assert "configured for a remote daemon" in out
    assert "nerdit daemon restart --yes --wait" in " ".join(out.split())
    assert CODE not in out


async def test_claim_restart_failure_reports_activation_pending(
    fake_client, capsys, _no_daemon_restart
):
    _no_daemon_restart.return_value = 1
    await _link_async(CODE, "https://app.localhost", None, True)
    out = " ".join(capsys.readouterr().out.split())
    assert "Link saved, but tunnel activation is pending" in out
    assert "nerdit daemon restart --yes --wait" in " ".join(out.split())
    assert "tunnel is starting" not in out
    assert CODE not in out


async def test_claim_no_enable_warns_the_flag_is_still_false(monkeypatch, capsys):
    client = _fake_client(
        claim_link=AsyncMock(return_value={**_CLAIM_OK, "enabled": False}),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(CODE, "https://app.localhost", None, False)
    out = capsys.readouterr().out
    assert "[link].enabled is still false" in out
    assert CODE not in out


async def test_claim_daemon_refusal_exits_1_without_echoing_the_code(monkeypatch, capsys):
    response = httpx.Response(
        409,
        json={
            "code": "link.claim_refused",
            "message": "The cloud refused this link code.",
            "hint": "Ask the console for a fresh code.",
            "cloud_code": "link_code_expired",
        },
        request=httpx.Request("POST", "http://localhost:9321/api/link/claim"),
    )
    client = _fake_client(
        claim_link=AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "refused", request=response.request, response=response
            )
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _link_async(CODE, "https://app.localhost", None, True)
    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "refused" in out.lower()
    assert CODE not in out


async def test_status_reads_config_and_capabilities_when_unlinked(fake_client, capsys):
    await _link_async(None, None, None, True)

    fake_client.get_config.assert_awaited_once_with("link")
    fake_client.get_capabilities.assert_awaited_once()
    fake_client.claim_link.assert_not_awaited()
    out = capsys.readouterr().out
    assert "linked      = no" in out
    # (D-P34-2 / OD-P34-4, D5 copy) Bare ``nerdit link`` stays a read, and the
    # hint is where the device flow becomes discoverable — so the unlinked
    # status must name it, and must no longer send an operator hunting for a
    # console code as the only way in.
    assert "nerdit link --device" in out
    assert "nerdit link <code>" not in out


async def test_status_linked_and_enabled_but_manager_absent_warns(monkeypatch, capsys):
    client = _fake_client(
        get_config=AsyncMock(
            return_value={
                "section": "link",
                "values": {
                    "enabled": True,
                    "relay_url": "wss://relay.localhost/link",
                    "node_id": _CLAIM_OK["node_id"],
                    "slug": "brave-otter",
                },
            }
        ),
        get_capabilities=AsyncMock(return_value={"link": {"enabled": False}}),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "linked      = yes" in out
    assert "brave-otter" in out
    # The yellow restart-needed line. Asserted on a wrap-safe prefix: Rich
    # hard-wraps at 80 columns off a tty, so a substring near the fold is a
    # flaky assertion, not a stricter one.
    assert "Linked but the tunnel manager is not running" in out


async def test_status_renders_the_live_session_when_connected(monkeypatch, capsys):
    client = _fake_client(
        get_config=AsyncMock(
            return_value={
                "section": "link",
                "values": {"enabled": True, "node_id": _CLAIM_OK["node_id"], "slug": "otter"},
            }
        ),
        get_capabilities=AsyncMock(
            return_value={
                "link": {
                    "enabled": True,
                    "state": "connected",
                    "connected_at": "2026-08-14T10:00:00+00:00",
                    "capability_expires_at": "2026-08-14T10:10:00+00:00",
                }
            }
        ),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "connected" in out
    assert "2026-08-14T10:10:00+00:00" in out
    assert "Linked but the tunnel manager is not running" not in out


def _entitlement_client(link_block: dict) -> SimpleNamespace:
    """A fake client whose running daemon reports the given ``link`` block."""
    return _fake_client(
        get_config=AsyncMock(
            return_value={
                "section": "link",
                "values": {"enabled": True, "node_id": _CLAIM_OK["node_id"], "slug": "otter"},
            }
        ),
        get_capabilities=AsyncMock(return_value={"link": {"enabled": True, **link_block}}),
    )


async def test_status_renders_a_live_entitlement_with_its_age(monkeypatch, capsys):
    """(P32) "yes" alone would not tell an operator whether the cloud is still
    talking to this daemon, and the mirror ages out after 24 h."""
    asserted_at = datetime.now(UTC) - timedelta(minutes=12)
    client = _entitlement_client(
        {
            "state": "connected",
            "hosted_public_entitled": True,
            "hosted_public_entitled_at": asserted_at.isoformat(),
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "public" in out
    assert "yes (asserted 12m ago)" in out


async def test_status_distinguishes_never_asserted_from_expired(monkeypatch, capsys):
    """The two "no"s an operator debugging a refused public share must tell
    apart: the cloud never spoke, versus it spoke and the value aged out."""
    client = _entitlement_client(
        {
            "state": "connected",
            "hosted_public_entitled": False,
            "hosted_public_entitled_at": None,
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    await _link_async(None, None, None, True)
    assert "no (never asserted)" in capsys.readouterr().out

    stale = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    client = _entitlement_client(
        {
            "state": "connected",
            "hosted_public_entitled": False,
            "hosted_public_entitled_at": stale,
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    await _link_async(None, None, None, True)
    assert "no (asserted 1d ago, expired)" in capsys.readouterr().out


async def test_status_reports_a_fresh_denial_without_calling_it_expired(monkeypatch, capsys):
    """A recent ``false`` is the cloud saying no — not the lease running out."""
    fresh = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
    client = _entitlement_client(
        {
            "state": "connected",
            "hosted_public_entitled": False,
            "hosted_public_entitled_at": fresh,
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "no (asserted 3m ago)" in out
    assert "expired" not in out


async def test_status_omits_the_entitlement_line_on_an_older_daemon(monkeypatch, capsys):
    """A daemon that predates the key prints one line fewer rather than a
    confident falsehood — the key is read, never inferred from the state."""
    client = _entitlement_client({"state": "connected", "capability_expires_at": None})
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "connected" in out
    assert "public" not in out


async def test_status_renders_the_github_line_with_count_and_lead(monkeypatch, capsys):
    """(P33 D-GH-7) ``github = <n> installation(s), expires in 42m`` — a count
    and the soonest expiry, read off the secret-free summaries."""
    soon = (datetime.now(UTC) + timedelta(minutes=42, seconds=30)).isoformat()
    later = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
    client = _entitlement_client(
        {
            "state": "connected",
            "github_installations": [
                {"installation_id": 11, "expires_at": later, "repos_count": 2},
                {"installation_id": 22, "expires_at": soon, "repos_count": 1},
            ],
        }
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "github      = 2 installations, expires in 42m" in out


async def test_status_renders_github_none_when_the_tunnel_is_down(monkeypatch, capsys):
    """``none`` is now reserved for the case it is honest about (P34 D3).

    With the link not up, the cloud has had no opportunity to push anything, so
    the empty list says nothing about whether the App is installed — and the
    line must not pretend otherwise.
    """
    for state in ("connecting", "backoff", "displaced", "terminal"):
        client = _entitlement_client({"state": state, "github_installations": []})
        monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda client=client: client)

        await _link_async(None, None, None, True)
        assert "github      = none" in capsys.readouterr().out, state


async def test_github_line_connected_zero_mirrored(monkeypatch, capsys):
    """(P34 D3) A working tunnel with zero installations is its own state.

    Before D3 this rendered ``none``, identical to an offline node — the two
    situations need opposite actions (fix the link vs install the App) and the
    operator had no way to tell which one they were in. The wording still
    matches the ``github += .*installation`` assertion in the cross-repo gate
    that lives in the private cloud repository.
    """
    client = _entitlement_client({"state": "connected", "github_installations": []})
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out

    assert "github      = connected on Nerdit Cloud, 0 installation(s) mirrored" in out
    # The cross-repo gate's regex, applied here so the compatibility claim is a
    # test rather than a note in the plan.
    assert re.search(r"github += .*installation", out) is not None


def test_the_zero_state_never_claims_a_link_the_daemon_did_not_report():
    """The new line is keyed on the machine ``state`` token, never inferred.

    A daemon that predates the key still prints one line fewer, and a live block
    with no ``state`` at all falls back to ``none`` rather than asserting a
    connection nobody claimed.
    """
    from nerdit.cli.commands.link import _github_line

    assert _github_line({"github_installations": []}) == "none"
    assert _github_line({"state": None, "github_installations": []}) == "none"
    assert _github_line({"state": "connected"}) is None


async def test_status_omits_the_github_line_on_an_older_daemon(monkeypatch, capsys):
    client = _entitlement_client({"state": "connected", "capability_expires_at": None})
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)
    assert "github" not in capsys.readouterr().out


def test_the_github_line_never_renders_a_token_or_a_repo_name():
    """A daemon that (wrongly) shipped more than the summary would still not
    reach the terminal: the renderer reads three known keys and nothing else."""
    from nerdit.cli.commands.link import _github_line

    line = _github_line(
        {
            "github_installations": [
                {
                    "installation_id": 1,
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1, minutes=5)).isoformat(),
                    "repos_count": 1,
                    "repos": ["acme/web"],
                    "token": "ghs_should_never_render",
                }
            ]
        }
    )
    assert line == "1 installation, expires in 1h"
    assert "acme" not in line
    assert "ghs_" not in line


def _terminal_client(reason: str) -> SimpleNamespace:
    """A fake client whose running daemon reports a terminal link."""
    return _fake_client(
        get_config=AsyncMock(
            return_value={
                "section": "link",
                "values": {"enabled": True, "node_id": _CLAIM_OK["node_id"], "slug": "otter"},
            }
        ),
        get_capabilities=AsyncMock(
            return_value={
                "link": {
                    "enabled": True,
                    "state": "terminal",
                    "terminal_reason": reason,
                    "connected_at": None,
                    "capability_expires_at": None,
                }
            }
        ),
    )


async def test_status_renders_the_entitlement_hint_on_a_terminal_link(monkeypatch, capsys):
    """(P27 WP-C4, recopy P34 D-X16-23) The refusal an operator can fix, told with its fix.

    The advisory is keyed on the machine token from ``/capabilities``, not on
    any rendered text, so an older daemon that does not carry the field simply
    prints the state line. Asserted on wrap-safe fragments: Rich hard-wraps at
    80 columns off a tty.
    """
    monkeypatch.setattr(
        "nerdit.cli.client.get_configured_client", lambda: _terminal_client("entitlement_required")
    )

    await _link_async(None, None, None, True)
    out = capsys.readouterr().out
    assert "terminal    = entitlement_required" in out
    assert "not active" in out
    assert "subscription" not in out
    assert "restart the daemon" in out
    assert "(nerdit daemon restart)" in out


async def test_status_renders_the_recovery_on_a_revoked_link(monkeypatch, capsys):
    """(P34) The revoked node's exit, on the surface an operator actually runs.

    The doctor row carries this recovery; this line is why it is here too. An
    operator who only runs ``nerdit link`` never opens the doctor, so leaving
    this surface at a bare ``terminal    = revoked`` shows them the dead end
    and hides the way out of it.

    Both commands, in this order. Revocation clears nothing locally: the
    persisted ``node_id`` printed a few lines above still names the revoked
    node, and while it is set ``_linked_slug`` short-circuits (and the daemon
    answers ``409 link.already_linked``), so a re-link on its own prints
    "Already linked as …" and exits 0 — a no-op dressed as success. Asserted on
    whitespace-normalised output because Rich hard-wraps at 80 columns off a
    tty and either command can straddle a line break.
    """
    monkeypatch.setattr(
        "nerdit.cli.client.get_configured_client", lambda: _terminal_client("revoked")
    )

    await _link_async(None, None, None, True)
    out = " ".join(capsys.readouterr().out.split())
    assert "terminal = revoked" in out
    assert "nerdit unlink" in out
    assert "nerdit link --device" in out
    assert out.index("nerdit unlink") < out.index("nerdit link --device")
    # The custody cost is stated where the operator decides, not discovered at
    # the confirmation prompt: unlink wipes the key, so the re-link enrols a
    # new verifier rather than resuming the revoked identity.
    assert "wipes the node identity key" in out


async def test_status_renders_a_terminal_reason_without_a_hint_generically(monkeypatch, capsys):
    """Generic reasons get the reason line only — never invented guidance.

    The sample reason moved off ``revoked`` in P34 (it has a real recovery
    now); ``auth_failed`` replaces it because it is a reason with no single
    known fix, which is the property this test exists to pin. The pin is the
    absence of guidance for reasons we cannot advise on, never a rule that any
    particular reason must stay silent.
    """
    monkeypatch.setattr(
        "nerdit.cli.client.get_configured_client", lambda: _terminal_client("auth_failed")
    )

    await _link_async(None, None, None, True)
    out = " ".join(capsys.readouterr().out.split())
    assert "terminal = auth_failed" in out
    assert "not active" not in out
    assert "nerdit unlink" not in out


async def test_unlink_decline_makes_zero_client_calls(monkeypatch, capsys):
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)

    with pytest.raises(typer.Exit) as excinfo:
        await _unlink_async(False)
    assert excinfo.value.exit_code == 1
    client.unlink_node.assert_not_awaited()
    assert "Aborted" in capsys.readouterr().out


async def test_unlink_yes_calls_the_route_with_a_minted_key(fake_client, capsys):
    await _unlink_async(True)

    fake_client.unlink_node.assert_awaited_once()
    assert fake_client.unlink_node.await_args.kwargs["idempotency_key"]
    out = capsys.readouterr().out
    assert "Unlinked." in out
    assert "live tunnel dropped" in out
    assert "node identity key deleted" in out


async def test_unlink_when_not_linked_is_a_no_op_line(monkeypatch, capsys):
    client = _fake_client(
        unlink_node=AsyncMock(
            return_value={
                "was_linked": False,
                "tunnel_stopped": False,
                "key_removed": False,
                "requires_restart": True,
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _unlink_async(True)
    assert "was not linked" in capsys.readouterr().out


async def test_unlink_key_removal_failure_warns(monkeypatch, capsys):
    client = _fake_client(
        unlink_node=AsyncMock(
            return_value={
                "was_linked": True,
                "tunnel_stopped": False,
                "key_removed": False,
                "requires_restart": True,
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _unlink_async(True)
    out = capsys.readouterr().out
    assert "no live tunnel" in out
    assert "could not be removed" in out


async def test_unlink_daemon_error_exits_1(monkeypatch):
    response = httpx.Response(
        403,
        json={"code": "forbidden", "message": "Admin role required."},
        request=httpx.Request("DELETE", "http://localhost:9321/api/link"),
    )
    client = _fake_client(
        unlink_node=AsyncMock(
            side_effect=httpx.HTTPStatusError("nope", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _unlink_async(True)
    assert excinfo.value.exit_code == 1


# -- argv-level ---------------------------------------------------------------------

_runner = CliRunner()


def test_cli_link_argv_parses_every_option(monkeypatch):
    """The D-P30-10 regression pin: explicit URLs are passed through untouched,
    never replaced by the production defaults."""
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(
        app,
        [
            "link",
            CODE,
            "--api-url",
            "https://app.localhost",
            "--relay-url",
            "wss://r/link",
            "--no-enable",
        ],
    )
    assert result.exit_code == 0, result.output
    kwargs = client.claim_link.await_args.kwargs
    assert kwargs["code"] == CODE
    assert kwargs["api_url"] == "https://app.localhost"
    assert kwargs["relay_url"] == "wss://r/link"
    assert kwargs["enable"] is False


def test_cli_link_argv_without_code_takes_the_status_path(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link"])
    assert result.exit_code == 0, result.output
    client.get_config.assert_awaited_once_with("link")
    client.claim_link.assert_not_awaited()


def test_cli_link_argv_with_only_a_code_claims_against_production(monkeypatch):
    """(P30 D-P30-10) ``nerdit link <code>`` — the whole second onboarding step."""
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", CODE])
    assert result.exit_code == 0, result.output
    kwargs = client.claim_link.await_args.kwargs
    assert kwargs["api_url"] == DEFAULT_LINK_API_URL
    assert kwargs["relay_url"] == DEFAULT_LINK_RELAY_URL
    assert CODE not in result.output


def test_cli_unlink_argv_yes_skips_the_prompt(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["unlink", "-y"])
    assert result.exit_code == 0, result.output
    client.unlink_node.assert_awaited_once()


async def test_claim_options_without_a_code_prompt_for_it_hidden(fake_client, monkeypatch, capsys):
    """Claim options without a code prompt (hidden) — never a silent status.

    Round 3 turned the refusal into the argv-free path: the prompted code
    reaches neither shell history nor the process command line, where a
    still-valid code would linger if the claim failed before spending it.
    ``--enable`` alone is indistinguishable from the default and stays exempt.
    """
    prompts: list[dict] = []

    def fake_prompt(label, **kwargs):
        prompts.append(kwargs)
        return CODE

    monkeypatch.setattr(typer, "prompt", fake_prompt)

    await _link_async(None, "https://app.localhost", None, True)
    assert prompts and prompts[0].get("hide_input") is True
    assert fake_client.claim_link.await_args.kwargs["code"] == CODE
    assert CODE not in capsys.readouterr().out

    # An empty prompt answer is a refusal, not a claim with "".
    monkeypatch.setattr(typer, "prompt", lambda label, **kw: "  ")
    with pytest.raises(typer.Exit) as excinfo:
        await _link_async(None, None, "wss://relay.localhost/link", True)
    assert excinfo.value.exit_code == 1
    assert "No link code entered" in capsys.readouterr().out

    # The bare no-arg form still renders status, prompting for nothing.
    monkeypatch.setattr(
        typer, "prompt", lambda *a, **kw: pytest.fail("status mode must not prompt")
    )
    await _link_async(None, None, None, True)
    assert "Link (persisted)" in capsys.readouterr().out


# -- refresh mode (P26 WP-H) --------------------------------------------------------


async def test_refresh_changed_restarts_the_service(fake_client, monkeypatch, capsys):
    """(S2) ``[link]`` is restart-required as a whole section, so a refresh that
    actually moved the domain must bounce the daemon exactly as a claim does."""
    restarts: list[int] = []
    monkeypatch.setattr(
        link_mod, "_restart_for_tunnel", AsyncMock(side_effect=lambda: restarts.append(1))
    )

    await _link_async("refresh", None, None, True)

    fake_client.refresh_link.assert_awaited_once()
    kwargs = fake_client.refresh_link.await_args.kwargs
    assert kwargs["api_url"] == DEFAULT_LINK_API_URL
    assert kwargs["idempotency_key"]
    fake_client.claim_link.assert_not_awaited()
    assert restarts == [1]
    assert "nodes.example" in capsys.readouterr().out


async def test_refresh_unchanged_does_not_restart(monkeypatch, capsys):
    client = _fake_client(
        refresh_link=AsyncMock(return_value={**_REFRESH_OK, "changed": False}),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(
        link_mod,
        "_restart_for_tunnel",
        lambda: pytest.fail("an unchanged refresh must not restart"),
    )

    await _link_async("refresh", None, None, True)

    assert "Already current" in capsys.readouterr().out


async def test_refresh_is_case_insensitive_and_honours_api_url(fake_client, monkeypatch, capsys):
    monkeypatch.setattr(link_mod, "_restart_for_tunnel", AsyncMock())

    await _link_async("  REFRESH  ", "https://app.localhost", None, True)

    assert fake_client.refresh_link.await_args.kwargs["api_url"] == "https://app.localhost"
    fake_client.claim_link.assert_not_awaited()


async def test_refresh_never_prompts_for_a_code(fake_client, monkeypatch, capsys):
    """The refresh check precedes the claim-intent gate, so ``link refresh
    --api-url …`` can never fall through to the hidden code prompt."""
    monkeypatch.setattr(link_mod, "_restart_for_tunnel", AsyncMock())
    monkeypatch.setattr(
        typer, "prompt", lambda *a, **kw: pytest.fail("refresh mode must not prompt")
    )

    await _link_async("refresh", "https://app.localhost", None, False)

    fake_client.refresh_link.assert_awaited_once()


async def test_refresh_on_a_pre_p26_cloud_renders_the_refusal(monkeypatch, capsys):
    """A cloud without the metadata endpoint 404s; the daemon turns that into a
    structured ``link.refresh_unsupported`` and the CLI prints its hint."""
    response = httpx.Response(
        409,
        json={
            "code": "link.refresh_unsupported",
            "message": "The cloud does not publish hosted metadata yet.",
            "hint": "Upgrade the cloud, or set the key through PUT /config/daemon/link.",
            "detail": "The cloud does not publish hosted metadata yet.",
        },
        request=httpx.Request("POST", "http://localhost:9321/api/link/refresh"),
    )
    client = _fake_client(
        refresh_link=AsyncMock(
            side_effect=httpx.HTTPStatusError("409", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _link_async("refresh", None, None, True)
    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "does not publish hosted metadata" in out


async def test_status_renders_the_hosted_domain(monkeypatch, capsys):
    client = _fake_client(
        get_config=AsyncMock(
            return_value={
                "values": {
                    "node_id": "n-1",
                    "slug": "brave-otter",
                    "nodes_base_domain": "nodes.example",
                    "enabled": True,
                }
            }
        ),
        get_capabilities=AsyncMock(return_value={"link": {"enabled": True, "state": "connected"}}),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _link_async(None, None, None, True)

    assert "nodes.example" in capsys.readouterr().out


def test_cli_link_refresh_argv(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(link_mod, "_restart_for_tunnel", AsyncMock())

    result = _runner.invoke(app, ["link", "refresh", "--api-url", "http://app.localhost"])
    assert result.exit_code == 0, result.output
    assert client.refresh_link.await_args.kwargs["api_url"] == "http://app.localhost"
    client.claim_link.assert_not_awaited()


# -- the device flow and the pre-auth key (P34 D1) ----------------------------------
#
# The device tests drive the verb through ``CliRunner`` rather than calling the
# async body, because on this flow the EXIT CODE is half the contract: 0 linked
# or already linked, 1 refused or interrupted, 3 the local --timeout elapsing
# while the cloud still says pending (the ``--wait`` house codes). Asserting a
# printed line without asserting the code it exits with would pin the friendly
# half of the copy and miss the half automation reads.


def _flat(text: str) -> str:
    """Collapse Rich's word wrapping so a sentence can be asserted whole.

    ``console`` wraps at 80 columns off a terminal, so a copy assertion long
    enough to be meaningful is also long enough to straddle a line break. The
    wrap only ever inserts whitespace, so re-joining on whitespace restores the
    sentence exactly.
    """
    return " ".join(text.split())


def _daemon_error(status: int, code: str, **extra: object) -> httpx.HTTPStatusError:
    """A daemon refusal shaped exactly as the link routes return it.

    The ``message`` is deliberately a marker string: several tests below assert
    that a branch with its own copy prints *that* copy and never the envelope's
    words, which is the CLI-side half of D11 (the cloud's body never travels).
    """
    request = httpx.Request("POST", "http://localhost:9321/api/link/device/poll")
    response = httpx.Response(
        status,
        json={"code": code, "message": "CLOUD-BODY-MUST-NOT-BE-ECHOED", **extra},
        request=request,
    )
    return httpx.HTTPStatusError("refused", request=request, response=response)


@pytest.fixture
def instant_polls(monkeypatch):
    """Pin the poll loop's clock and make its waits free.

    The loop reads time through ``_monotonic`` and waits through
    ``_poll_sleep`` — two seams that exist for exactly this: a sleep advances
    the pinned clock instead of the wall clock, so a 600 s budget is exercised
    in microseconds and the number of polls is deterministic rather than
    timing-dependent. Returns the list of requested sleep durations, which is
    how the cadence tests observe an adopted ``slow_down`` interval.
    """
    clock = {"t": 0.0}
    slept: list[float] = []
    monkeypatch.setattr(link_mod, "_monotonic", lambda: clock["t"])

    async def _advance(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    monkeypatch.setattr(link_mod, "_poll_sleep", _advance)
    return slept


@pytest.fixture
def device_client(monkeypatch):
    """A fake client wired in, with the tunnel restart stubbed and recorded."""
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    restarts: list[int] = []
    monkeypatch.setattr(
        link_mod, "_restart_for_tunnel", AsyncMock(side_effect=lambda: restarts.append(1))
    )
    client.restarts = restarts
    return client


def test_cli_link_device_renders_url_code_and_fingerprint_and_exits_zero_on_link(
    device_client, instant_polls
):
    """The D-P34-4 approval frame, field by field, then the verdict.

    Four things must be on screen and one must not: the one-click URL carrying
    the code in its FRAGMENT (D-X16-O24), the grouped code for the retype path,
    the daemon-local machine line (D-P34-1 — the CLI cannot recompute those
    facts for a possibly-remote daemon), and the first eight characters of the
    credential fingerprint. The remaining 56 must not appear: the console
    truncates to the same eight, and printing more only invites comparing
    strings that were never meant to match.
    """
    from nerdit.cli.app import app

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "https://app.localhost/link#7Q3M-X2VF-9KHT" in out
    assert "code 7Q3M-X2VF-9KHT" in out
    assert "machine buildbox-01 (nerditd 0.6.0, linux)" in out
    assert "credential 3f9c2a1b" in out
    assert _DEVICE_START_OK["credential_fingerprint"] not in out
    assert "Waiting for approval" in out
    assert "Linked as node a1b2c3d4e5f6 (slug corvid-mesa)" in out
    assert "hosted domain nodes.example" in out
    assert "restart required: [link].enabled" in out
    assert "[mcp].http_enabled" in out
    assert device_client.restarts == [1]


def test_the_device_credential_line_is_a_cross_check_and_never_claims_to_stop_phishing(
    device_client, instant_polls
):
    """(D-X16-O4 as amended by security review H3) A copy pin, deliberately.

    The eight characters help the operator sitting at this terminal confirm the
    console is showing THIS node. They are not an anti-phishing control — a
    phished victim has no out-of-band reference to compare against — and copy
    that implies otherwise teaches a defence that does not exist. The words
    below are the ones a well-meaning rewrite would reach for.
    """
    from nerdit.cli.app import app

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    out = _flat(result.output).lower()
    assert "the console will show the same 8 characters" in out
    for forbidden in ("phishing", "secure", "verify that", "make sure the site"):
        assert forbidden not in out


def test_cli_link_device_already_linked_short_circuit_prints_and_exits_zero(monkeypatch):
    """(D-P34-3) The line that makes the installer idempotent, on the read path.

    A node that is already linked must cost one config read and nothing else:
    no cloud row minted, no code printed for a human to approve pointlessly,
    and an exit code automation reads as success.
    """
    from nerdit.cli.app import app

    client = _fake_client(get_config=AsyncMock(return_value=dict(_LINKED_SECTION)))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    assert _flat(result.output) == "Already linked as brave-otter."
    client.start_device_link.assert_not_awaited()
    client.poll_device_link.assert_not_awaited()


def test_cli_link_device_timeout_exits_three_and_names_the_resume_command(
    device_client, instant_polls
):
    """(D-P34-5) Exit 3, and copy that does not lie about the code.

    Once this terminal stops polling nothing ever consumes an approval: the
    daemon's pending slot is read only by CLI-driven polls. So the message must
    deny that a later approval links this machine — the softer "your code is
    still valid" wording produces an operator who approves into a cloud node
    that never connects — and it must name the command that starts a fresh
    flow, since there is no way to resume this one.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(return_value=dict(_DEVICE_PENDING))

    result = _runner.invoke(app, ["link", "--device", "--timeout", "10"])

    assert result.exit_code == 3, result.output
    out = _flat(result.output)
    assert "Stopped polling after 10s." in out
    assert "will no longer link this machine" in out
    assert "nerdit link --device" in out
    assert device_client.restarts == []
    # 0 s, 5 s and 10 s: the poll at the deadline is a last chance, not a bug.
    # Two, not three: the budget is now consulted BEFORE each hop, so the loop
    # no longer starts a poll it has no time left to wait for. That final poll
    # was the overrun — `poll_device_link` allows 60 s per request, so firing
    # one with zero budget is precisely how the command outran its own
    # `--timeout`.
    assert device_client.poll_device_link.await_count == 2


def test_cli_link_device_ctrl_c_says_a_later_approval_will_not_link_this_machine(
    device_client, instant_polls
):
    """(D-P34-5) The interrupt is caught around the outermost ``asyncio.run``.

    Same honesty as the timeout — abandoning the wait abandons the flow — and
    never a traceback: the operator pressed Ctrl-C on purpose, and a stack
    dump would suggest they broke something.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(side_effect=KeyboardInterrupt())

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "Stopped waiting" in out
    assert "will no longer link this machine" in out
    assert "nerdit link --device" in out
    assert "Traceback" not in result.output


def test_cli_link_device_maps_a_denied_request_to_exit_one_without_echoing_the_cloud_body(
    device_client, instant_polls
):
    """(D-P34-5, D11) One written line per terminal code; never the cloud's words."""
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(409, "link.device_denied", cloud_code="link_code_denied")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "The request was denied in the console." in out
    assert "CLOUD-BODY-MUST-NOT-BE-ECHOED" not in out
    assert device_client.restarts == []


def test_cli_link_device_expiry_tells_the_operator_to_start_a_new_flow(
    device_client, instant_polls
):
    """A code that died before approval is terminal — but recoverable, and the
    copy has to say which command recovers it."""
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(409, "link.device_expired")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "The code expired before it was approved" in out
    assert "nerdit link --device" in out


def test_cli_link_device_superseded_tells_this_terminal_its_code_is_dead(
    device_client, instant_polls
):
    """(D-P34-1 session binding) Two SSH sessions, one pending slot.

    The daemon refuses the older session's polls rather than silently
    servicing them with the newer flow's code — which would have this terminal
    display code A while approving code B, enrolling an orphan cloud node and
    firing two service restarts. The CLI's job is to say plainly which terminal
    lost.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(409, "link.device_superseded")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "replaced this request" in out
    assert "the newer terminal owns the flow" in out


def test_cli_link_device_renders_the_route_hint_when_the_daemon_has_no_pending_slot(
    device_client, instant_polls
):
    """``link.device_not_started`` carries its own hint — the route knows why.

    Restating the guidance here would only drift from the route that owns the
    distinction between "never started", "expired locally" and "the credential
    changed under the flow", so this branch renders the envelope as-is.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(
            409,
            "link.device_not_started",
            hint="run 'nerdit link --device' to start a new request",
        )
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    assert "start a new request" in _flat(result.output)


def test_cli_link_device_adopts_the_slow_down_interval_and_keeps_polling(
    device_client, instant_polls
):
    """(D-P34-5, RFC 8628) ``slow_down`` is a cadence instruction, not a failure.

    The cloud's gate fires before any database read and is oracle-free, so
    racing it is both rude and useless: the returned interval is adopted for
    the rest of the session and the poller carries on.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=[dict(_DEVICE_SLOW_DOWN), dict(_DEVICE_LINKED)]
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    assert instant_polls == [10.0]
    assert device_client.poll_device_link.await_count == 2


def test_cli_link_device_keeps_polling_through_a_briefly_unreachable_cloud(
    device_client, instant_polls
):
    """A mid-poll cloud failure is NOT terminal: the daemon still holds the
    pending slot, so the poller widens its interval and carries on until the
    --timeout budget runs out. Only the START hop failing is fatal, because
    there is no session to come back to."""
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=[_daemon_error(502, "link.cloud_unreachable"), dict(_DEVICE_LINKED)]
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    assert instant_polls == [10.0]  # 5 s advertised, doubled on the failure
    assert device_client.restarts == [1]


def test_cli_link_device_start_failure_is_immediately_terminal(device_client, instant_polls):
    """No session exists yet, so retrying could only mint a second cloud row."""
    from nerdit.cli.app import app

    device_client.start_device_link = AsyncMock(
        side_effect=_daemon_error(502, "link.cloud_unreachable", hint="retry shortly")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    device_client.poll_device_link.assert_not_awaited()


def test_cli_link_device_a_late_already_linked_conflict_still_exits_zero(
    device_client, instant_polls
):
    """(D-P34-3) The lost-response recovery, rendered as the success it is.

    The commit branch clears the daemon's pending slot, so a poll retried after
    a link that actually SUCCEEDED lands on the D8 refusal. The wire stays
    honest — an unconditional local "success" would mislabel a node linked by a
    different flow — and the operator still sees one line and exit 0.
    """
    from nerdit.cli.app import app

    unlinked = {"section": "link", "values": {}, "etag": "e"}
    # Three reads in order: the pre-flight short-circuit, the relay-seam read,
    # and — after the refusal — the read that names what this node became.
    device_client.get_config = AsyncMock(side_effect=[unlinked, unlinked, dict(_LINKED_SECTION)])
    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(409, "link.already_linked")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    assert "Already linked as brave-otter." in _flat(result.output)


def test_a_late_already_linked_conflict_warns_the_printed_code_is_still_live(
    device_client, instant_polls
):
    """Success, but this run PRINTED a code and that code outlives the verdict.

    The pre-start short-circuit mints nothing and owes no such warning. This
    path does: the terminal already showed a URL and a code, the cloud row
    behind them stays approvable until it expires, and approving it now enrolls
    a second, unwanted node in the account. Exit 0 and the ordinary line stay
    exactly as they were — the warning is added, never substituted.
    """
    from nerdit.cli.app import app

    unlinked = {"section": "link", "values": {}, "etag": "e"}
    device_client.get_config = AsyncMock(side_effect=[unlinked, unlinked, dict(_LINKED_SECTION)])
    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(409, "link.already_linked")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    assert "Already linked as brave-otter." in out
    assert "still live until it expires" in out
    assert "do not approve it in the console" in out


def test_cli_link_device_survives_a_transport_failure_and_links_on_the_next_poll(
    device_client, instant_polls
):
    """A CLI→daemon transport failure is not a verdict from anyone.

    A restarting daemon or a dropped socket produces no HTTP response at all,
    so nothing was decided: the cloud row is still pending and still
    approvable. Treating that as terminal threw away a live approval request
    the operator was in the middle of using. The poller says so once — not once
    per poll — and carries on inside the same --timeout budget.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.ConnectError("connection refused"),
            dict(_DEVICE_LINKED),
        ]
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 0, result.output
    out = _flat(result.output)
    # Announced once, however many polls fail.
    assert out.count("Cannot reach the daemon right now") == 1
    assert device_client.poll_device_link.await_count == 3
    # The cadence is untouched by a transport failure — it is not a rate limit.
    assert instant_polls == [5.0, 5.0]
    assert device_client.restarts == [1]


def test_cli_link_device_an_unknown_envelope_code_stops_and_warns_about_the_live_code(
    device_client, instant_polls
):
    """A refusal this CLI has never heard of is a reason to stop — loudly.

    A daemon newer than this CLI can answer with a code that is in no local
    table; hammering it would be pointless. But stopping leaves a pending code
    behind exactly as a timeout or a Ctrl-C does, and this was the one exit
    that said nothing about it.
    """
    from nerdit.cli.app import app

    device_client.poll_device_link = AsyncMock(
        side_effect=_daemon_error(409, "link.device_something_new", hint="upgrade the CLI")
    )

    result = _runner.invoke(app, ["link", "--device"])

    assert result.exit_code == 1, result.output
    out = _flat(result.output)
    assert "will no longer link this machine" in out
    assert "nerdit link --device" in out


def test_cli_link_device_and_a_positional_code_is_a_usage_error_before_any_network_hop(
    monkeypatch,
):
    """(D-P34-2) Mutually exclusive, and refused before anything is built.

    Exit 2 — the house code for a mistyped invocation — and, more importantly,
    zero client calls: a conflicting invocation must never mint a cloud row or
    spend the code it was handed by mistake.
    """
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", CODE, "--device"])

    assert result.exit_code == 2, result.output
    client.start_device_link.assert_not_awaited()
    client.claim_link.assert_not_awaited()
    client.get_config.assert_not_awaited()
    assert CODE not in result.output


def test_cli_link_device_and_key_stdin_together_are_a_usage_error(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", "--device", "--key-stdin"], input=f"{PREAUTH_KEY}\n")

    assert result.exit_code == 2, result.output
    client.start_device_link.assert_not_awaited()
    client.claim_link.assert_not_awaited()


def test_cli_link_key_stdin_reads_one_line_and_the_key_never_appears_in_argv_or_output(
    monkeypatch,
):
    """(D-X16-O11) The key crosses on a pipe fd and nowhere else.

    Three assertions, one per hop the installer controls: the CLI exposes no
    option that would take the key as a VALUE (``/proc/<pid>/cmdline`` is
    world-readable on Linux while environ is owner-only, so argv is the one
    channel never used); the key does reach the daemon in the request body; and
    it appears nowhere in what this process printed.
    """
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(link_mod, "_restart_for_tunnel", AsyncMock())

    # There is no ``--key``: the key can never be an option value.
    rejected = _runner.invoke(app, ["link", "--key", PREAUTH_KEY])
    assert rejected.exit_code == 2, rejected.output
    client.claim_link.assert_not_awaited()

    result = _runner.invoke(app, ["link", "--key-stdin"], input=f"  {PREAUTH_KEY}  \n")

    assert result.exit_code == 0, result.output
    kwargs = client.claim_link.await_args.kwargs
    # Surrounding whitespace stripped and nothing else: the key is strict-shape
    # by design, so folding case or dropping separators could only turn a typo
    # into a different, well-formed, wrong key. And it travels as ``key=``,
    # never ``code=`` (D-P34-6): the daemon dispatches on which body field is
    # set, and its code validator pins an alphabet no ``nk_`` key can satisfy —
    # sending the key as the code would 422 every valid key before the cloud.
    assert kwargs["key"] == PREAUTH_KEY
    assert "code" not in kwargs
    assert kwargs["api_url"] == DEFAULT_LINK_API_URL
    assert kwargs["idempotency_key"]
    assert PREAUTH_KEY not in result.output
    assert PREAUTH_KEY not in rejected.output


def test_cli_link_hidden_prompt_dispatches_an_nk_key_into_the_key_field(monkeypatch):
    """(D-P34-6) The hidden prompt accepts an ``nk_`` key too — made true, not free.

    The claim path's hidden prompt predates the pre-auth key, and the daemon
    dispatches on which BODY FIELD is set — so an ``nk_`` secret pasted there
    must travel as ``key=``, where the plain claim's uppercase code alphabet
    would otherwise refuse the underscore before the cloud ever saw it. The
    dispatch is the client-side ``nk_`` prefix check, the recognisability
    D-X16-O8 minted the prefix for.
    """
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(link_mod, "_restart_for_tunnel", AsyncMock())

    result = _runner.invoke(
        app, ["link", "--api-url", "https://app.localhost"], input=f"{PREAUTH_KEY}\n"
    )

    assert result.exit_code == 0, result.output
    kwargs = client.claim_link.await_args.kwargs
    assert kwargs["key"] == PREAUTH_KEY
    assert "code" not in kwargs
    assert PREAUTH_KEY not in result.output


def test_cli_link_key_stdin_already_linked_short_circuit_prints_and_exits_zero(monkeypatch):
    """(D-P34-3) The same one line on the key path — the installer re-run case.

    ``nerdit update`` re-executes the recorded installer with the skip flag,
    but Ansible re-converges, retried cloud-init and a human re-running the
    one-liner do not set it, and every one of them must terminate as a cheap
    no-op success without spending the key again.
    """
    from nerdit.cli.app import app

    client = _fake_client(get_config=AsyncMock(return_value=dict(_LINKED_SECTION)))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", "--key-stdin"], input=f"{PREAUTH_KEY}\n")

    assert result.exit_code == 0, result.output
    assert _flat(result.output) == "Already linked as brave-otter."
    client.claim_link.assert_not_awaited()


def test_cli_link_key_stdin_with_an_empty_line_refuses_without_calling_the_daemon(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", "--key-stdin"], input="   \n")

    assert result.exit_code == 1, result.output
    assert "No link key entered." in _flat(result.output)
    client.claim_link.assert_not_awaited()


def test_cli_link_key_stdin_maps_a_refused_key_to_exit_one_without_echoing_it(monkeypatch):
    """One message for every key death (D-X16-O9) and never the key itself.

    The cloud deliberately collapses unknown, malformed, revoked, expired,
    exhausted and race-lost into one refusal; distinguishing them here would
    reopen client-side the oracle the cloud closed.
    """
    from nerdit.cli.app import app

    client = _fake_client(
        claim_link=AsyncMock(
            side_effect=_daemon_error(
                409,
                "link.claim_refused",
                cloud_code="link_key_invalid",
                hint="The link key is not recognised (it may be revoked, expired, or exhausted).",
            )
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link", "--key-stdin"], input=f"{PREAUTH_KEY}\n")

    assert result.exit_code == 1, result.output
    assert "not recognised" in _flat(result.output)
    assert PREAUTH_KEY not in result.output


def test_a_bare_nerdit_link_still_renders_status_and_starts_no_device_flow(monkeypatch):
    """(OD-P34-4, ratified) The verb's most-run invocation stays a READ.

    Making a bare ``nerdit link`` auto-start the flow on an unlinked node would
    be friendlier to retype and would turn a documented, scripted read into a
    write that mints cloud rows and can commit ``[link]`` config — on exactly
    the class of node that automation re-runs blindly. This pin exists so that
    change cannot be made by accident, only on purpose.
    """
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["link"])

    assert result.exit_code == 0, result.output
    client.start_device_link.assert_not_awaited()
    client.poll_device_link.assert_not_awaited()
    client.claim_link.assert_not_awaited()
    assert "nerdit link --device" in _flat(result.output)


async def test_claim_prints_the_mcp_section_name_instead_of_eating_it(fake_client, capsys):
    """``[mcp]`` is valid Rich markup, so the unescaped form printed nothing.

    The line read ``Remote MCP: enabled (.http_enabled — …)``: Rich consumed
    ``[mcp]`` as a style tag and dropped it, leaving a config key with no
    section. Pre-existing on the claim path, and now also on ``--key-stdin``,
    which is what makes it worth pinning rather than tolerating.
    """
    fake_client.claim_link.return_value = {**_CLAIM_OK, "mcp_http_enabled": True}
    await _link_async(CODE, "https://app.localhost", None, True)
    out = capsys.readouterr().out
    assert "[mcp].http_enabled" in out
    assert "(.http_enabled" not in out


async def test_restart_for_tunnel_uses_target_api_port_and_waits_for_fresh_boot(monkeypatch):
    """Restart the selected daemon, independent of another installed service."""
    from nerdit.cli.commands.daemon import _restart_async

    requests = []
    polls = 0

    def handler(request):
        nonlocal polls
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(202, json={"in_flight_builds": 0, "in_flight_runs": 0})
        polls += 1
        return httpx.Response(200, json={"version": "test", "uptime_s": 100 if polls == 1 else 0})

    client = NerditClient(
        "127.0.0.1", 9333, token="test-only", transport=httpx.MockTransport(handler)
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(link_mod, "_restart_daemon", _restart_async)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: pytest.fail("restart must not prompt"))
    await link_mod._restart_for_tunnel()

    assert polls >= 2
    assert all(request.url.port == 9333 for request in requests)
    restart = next(request for request in requests if request.method == "POST")
    assert restart.url.path == "/api/daemon/restart"
    assert restart.headers["authorization"] == "Bearer test-only"
    assert restart.headers["idempotency-key"]


async def test_a_slow_poll_cannot_overrun_the_advertised_device_timeout(capsys):
    """`--timeout` is a promise about when the operator gets their terminal back.

    `poll_device_link` allows a request up to 60 s, so a poll started with two
    seconds of budget left could sit on a slow cloud well past the total wall
    clock this command advertised before it was allowed to answer. A budget
    consulted only BETWEEN polls is not that promise.
    """

    class _SlowClient:
        async def poll_device_link(self, **_kw):
            await asyncio.sleep(30)
            raise AssertionError("the poll should have been cut off by the budget")

    start = {
        "session": "s" * 32,
        "user_code": "ABCD-EFGH-JKMN",
        "verification_uri_complete": "https://app.example.test/link#ABCD-EFGH-JKMN",
        "credential_fingerprint": "f" * 64,
        "expires_in": 900,
        "interval": 5,
    }

    began = link_mod._monotonic()
    with pytest.raises(typer.Exit) as excinfo:
        await link_mod._poll_for_approval(_SlowClient(), start, 1)
    elapsed = link_mod._monotonic() - began

    assert excinfo.value.exit_code == 3
    assert elapsed < 10, f"overran the 1s budget by {elapsed:.1f}s"
    assert "Stopped polling after" in capsys.readouterr().out


def test_cli_link_refuses_a_positional_preauth_key_before_any_network_hop(monkeypatch):
    """(D-X16-O11) ``nerdit link nk_…`` exits 2 before any client exists, never echoing the key."""
    from nerdit.cli.app import app

    client = _fake_client()
    built = []
    monkeypatch.setattr(
        "nerdit.cli.client.get_configured_client", lambda: built.append(1) or client
    )

    result = _runner.invoke(app, ["link", PREAUTH_KEY])

    assert result.exit_code == 2
    assert not built
    client.claim_link.assert_not_awaited()
    # CI forces a colour terminal: Rich styles the flag, so match on plain text.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert PREAUTH_KEY not in plain
    assert "--key-stdin" in plain
