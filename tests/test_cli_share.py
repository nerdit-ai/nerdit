"""CLI tests for ``nerdit share`` / ``nerdit unshare`` (P26 WP-H).

Same three-layer structure as ``test_cli_link.py``: the ``NerditClient``
methods over ``httpx.MockTransport`` (wire shape: method, path, bearer,
``Idempotency-Key``, exact JSON body), the async command bodies against a fake
client, and argv-level parsing through Typer's ``CliRunner``.

Two properties are asserted on every printing path:

* **The daemon is the judge.** ``--public`` without ``--consent`` still reaches
  the daemon as typed — there is no local pre-check to drift — and the refusal
  the operator sees is the daemon's own hint.
* **Server-derived strings are escaped.** A hosted URL is composed from a
  service name and a cloud domain; a hostile one containing Rich markup must
  render, not raise ``MarkupError`` (the P6 lesson).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.share import _share_async, _unshare_async

_SHARE_OK = {
    "service_name": "demo",
    "access": "private",
    "url": "https://demo--gpu-box.nodes.test/",
    "state": "ready",
    "created_at": "2026-08-21T10:00:00Z",
}


# -- client methods ---------------------------------------------------------------


async def test_set_share_puts_access_and_consent_with_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_SHARE_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.set_share("demo", access="public", consent=True, idempotency_key="k1")

    assert result == _SHARE_OK
    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/services/demo/share"
    assert seen["auth"] == "Bearer t"
    assert seen["idem"] == "k1"
    # Both fields always travel: the daemon's model is strict (extra=forbid) and
    # its defaults are not this client's to re-state by omission.
    assert seen["body"] == {"access": "public", "consent": True}


async def test_set_share_defaults_to_private_without_consent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_SHARE_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.set_share("demo")

    assert seen["body"] == {"access": "private", "consent": False}


async def test_get_share_reads_without_a_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json=_SHARE_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    assert await client.get_share("demo") == _SHARE_OK
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/services/demo/share"
    assert seen["idem"] is None


async def test_remove_share_deletes_with_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service_name": "demo", "removed": True})

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.remove_share("demo", idempotency_key="k2")

    assert result == {"service_name": "demo", "removed": True}
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/services/demo/share"
    assert seen["idem"] == "k2"


async def test_share_refusal_raises_http_status_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "share.unprotected",
                "message": "Public means public.",
                "hint": "Add [deploy].edge_auth or pass consent=true.",
                "detail": "Public means public.",
            },
        )

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await client.set_share("demo", access="public", idempotency_key="k3")
    assert excinfo.value.response.status_code == 409


# -- async command bodies ----------------------------------------------------------


def _fake_client(**overrides) -> SimpleNamespace:
    calls = {
        "set_share": AsyncMock(return_value=dict(_SHARE_OK)),
        "get_share": AsyncMock(return_value=dict(_SHARE_OK)),
        "remove_share": AsyncMock(return_value={"service_name": "demo", "removed": True}),
    }
    calls.update(overrides)
    return SimpleNamespace(**calls)


@pytest.fixture
def fake_client(monkeypatch):
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    return client


async def test_share_private_prints_the_url_and_mints_a_key(fake_client, capsys):
    await _share_async("demo", public=False, consent=False, show=False)

    kwargs = fake_client.set_share.await_args.kwargs
    assert kwargs["access"] == "private"
    assert kwargs["consent"] is False
    assert kwargs["idempotency_key"]  # minted per invocation
    assert fake_client.set_share.await_args.args == ("demo",)

    out = capsys.readouterr().out
    assert "demo" in out
    assert "https://demo--gpu-box.nodes.test/" in out
    assert "(private)" in out


async def test_share_public_passes_consent_through_unchecked(fake_client, capsys):
    """No local pre-check: --public without --consent still reaches the daemon."""
    await _share_async("demo", public=True, consent=False, show=False)

    kwargs = fake_client.set_share.await_args.kwargs
    assert kwargs["access"] == "public"
    assert kwargs["consent"] is False


async def test_share_not_ready_prints_the_state_note(monkeypatch, capsys):
    client = _fake_client(
        set_share=AsyncMock(
            return_value={**_SHARE_OK, "state": "link_down", "url": None},
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=False, consent=False, show=False)

    out = capsys.readouterr().out
    assert "link_down" in out
    assert "tunnel is down" in out


async def test_share_not_entitled_note_names_beta_and_account_status(monkeypatch, capsys):
    client = _fake_client(
        set_share=AsyncMock(return_value={**_SHARE_OK, "access": "public", "state": "not_entitled"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=True, consent=True, show=False)

    out = capsys.readouterr().out
    assert "not_entitled" in out
    assert "free during the public beta" in out
    assert "Nerdit console" in " ".join(out.split())
    assert "Pro plan" not in out


async def test_share_ready_prints_no_state_note(fake_client, capsys):
    await _share_async("demo", public=False, consent=False, show=False)
    out = capsys.readouterr().out
    assert "tunnel is down" not in out
    assert "Pro plan" not in out


async def test_share_escapes_a_hostile_daemon_string(monkeypatch, capsys):
    """(P6 MarkupError lesson) The URL is composed from a service name and a
    cloud-supplied domain; neither is ours to trust as Rich markup."""
    client = _fake_client(
        set_share=AsyncMock(return_value={**_SHARE_OK, "url": "https://[red]evil--n.example/"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=False, consent=False, show=False)

    assert "[red]evil--n.example" in capsys.readouterr().out


async def test_share_show_reads_instead_of_writing(fake_client, capsys):
    await _share_async("demo", public=False, consent=False, show=True)

    fake_client.get_share.assert_awaited_once_with("demo")
    fake_client.set_share.assert_not_awaited()
    assert "https://demo--gpu-box.nodes.test/" in capsys.readouterr().out


async def test_share_show_on_an_unshared_service_exits_1(monkeypatch, capsys):
    response = httpx.Response(
        404,
        json={
            "code": "share.not_shared",
            "message": "This service is not shared.",
            "detail": "This service is not shared.",
        },
        request=httpx.Request("GET", "http://localhost:9321/api/services/demo/share"),
    )
    client = _fake_client(
        get_share=AsyncMock(
            side_effect=httpx.HTTPStatusError("404", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _share_async("demo", public=False, consent=False, show=True)
    assert excinfo.value.exit_code == 1
    assert "not shared" in capsys.readouterr().out


async def test_share_refusal_renders_the_daemon_hint(monkeypatch, capsys):
    response = httpx.Response(
        409,
        json={
            "code": "share.unprotected",
            "message": "Public means public.",
            "hint": "Add a [deploy].edge_auth block, or pass consent=true.",
            "detail": "Public means public.",
        },
        request=httpx.Request("PUT", "http://localhost:9321/api/services/demo/share"),
    )
    client = _fake_client(
        set_share=AsyncMock(
            side_effect=httpx.HTTPStatusError("409", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _share_async("demo", public=True, consent=False, show=False)
    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "Public means public." in out
    assert "consent=true" in out


async def test_unshare_reports_the_removal(fake_client, capsys):
    await _unshare_async("demo")

    fake_client.remove_share.assert_awaited_once()
    assert fake_client.remove_share.await_args.kwargs["idempotency_key"]
    assert "Unshared" in capsys.readouterr().out


async def test_unshare_of_an_unshared_service_is_a_no_op_line(monkeypatch, capsys):
    client = _fake_client(
        remove_share=AsyncMock(return_value={"service_name": "demo", "removed": False})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _unshare_async("demo")

    out = capsys.readouterr().out
    assert "was not shared" in out
    assert "Unshared" not in out


async def test_unshare_daemon_error_exits_1(monkeypatch, capsys):
    unreachable = httpx.ConnectError("down")
    unreachable.request = httpx.Request("DELETE", "http://localhost:9321/api/services/demo/share")
    client = _fake_client(remove_share=AsyncMock(side_effect=unreachable))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _unshare_async("demo")
    assert excinfo.value.exit_code == 1


# -- argv-level ---------------------------------------------------------------------

_runner = CliRunner()


def test_cli_share_argv_parses_public_and_consent(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["share", "demo", "--public", "--consent"])
    assert result.exit_code == 0, result.output
    kwargs = client.set_share.await_args.kwargs
    assert kwargs["access"] == "public"
    assert kwargs["consent"] is True


def test_cli_share_argv_show_takes_the_read_path(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["share", "demo", "--show"])
    assert result.exit_code == 0, result.output
    client.get_share.assert_awaited_once_with("demo")
    client.set_share.assert_not_awaited()


def test_cli_unshare_argv(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["unshare", "demo"])
    assert result.exit_code == 0, result.output
    client.remove_share.assert_awaited_once()


# -- P34: the origin line ---------------------------------------------------------


def _share_with_origin(**origin) -> dict:
    body = dict(_SHARE_OK)
    body["origin"] = {"status": "running", "answers": True, "hint": None, **origin}
    return body


async def test_share_show_prints_the_origin_hint_when_the_app_is_down(monkeypatch, capsys):
    """``state: ready`` + a dead app is the reported trap: "Shared: <url>" alone
    reads as "reachable" while the URL 404s with share.not_shared."""
    client = _fake_client(
        get_share=AsyncMock(
            return_value=_share_with_origin(
                status="failed",
                answers=False,
                hint="the service is failed; the URL returns share.not_shared "
                "until it runs — call diagnose_service",
            )
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=False, consent=False, show=True)

    out = capsys.readouterr().out
    assert "https://demo--gpu-box.nodes.test/" in out
    assert "origin" in out
    # Rendered from the DAEMON's hint — the CLI keeps no second copy.
    assert "diagnose_service" in out


async def test_share_prints_no_origin_line_when_the_app_answers(monkeypatch, capsys):
    client = _fake_client(set_share=AsyncMock(return_value=_share_with_origin()))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=False, consent=False, show=False)

    assert "origin" not in capsys.readouterr().out


async def test_share_tolerates_a_pre_p34_daemon_with_no_origin(monkeypatch, capsys):
    """The field is additive; its absence must print exactly the old output."""
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=False, consent=False, show=True)

    out = capsys.readouterr().out
    assert "https://demo--gpu-box.nodes.test/" in out
    assert "origin" not in out


async def test_share_escapes_a_hostile_origin_hint(monkeypatch, capsys):
    """The hint is server-derived text reaching ``console.print`` (P6 lesson)."""
    client = _fake_client(
        get_share=AsyncMock(
            return_value=_share_with_origin(answers=False, hint="[red]not markup[/red]")
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _share_async("demo", public=False, consent=False, show=True)

    assert "[red]not markup" in capsys.readouterr().out
