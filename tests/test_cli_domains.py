"""Test domain client requests, command output and Typer argument parsing.

Pin method/path encoding, bearer, idempotency key and JSON with MockTransport.
Send names and --acme unchanged for daemon validation; display the daemon's
normalized name. Escape all server-derived Rich text, including both certificate
state sinks.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.domains import _add_async, _list_async, _remove_async

_ADDED_OK = {
    "service_name": "demo",
    "domain": "app.example.com",
    "acme": False,
    "kind": "domain",
    "created_at": "2026-08-21T10:00:00Z",
    "url": "https://app.example.com/",
    "state": "ready",
    "cert_state": "internal",
    "cert_not_after": None,
    "created": True,
}


# -- client methods ---------------------------------------------------------------


async def test_list_domains_reads_without_a_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service_name": "demo", "domains": []})

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    assert await client.list_domains("demo") == {"service_name": "demo", "domains": []}
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/services/demo/domains"
    assert seen["auth"] == "Bearer t"
    assert seen["idem"] is None


async def test_add_domain_puts_the_domain_as_a_path_segment_with_a_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_ADDED_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.add_domain("demo", "app.example.com", idempotency_key="k1")

    assert result == _ADDED_OK
    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/services/demo/domains/app.example.com"
    assert seen["auth"] == "Bearer t"
    assert seen["idem"] == "k1"
    # ``acme`` is tri-state and OMITTED means "leave the row alone" — the client
    # must therefore send an empty body rather than re-state a default that is
    # not its to state (review round 1). Sending ``{"acme": false}`` here would
    # make every URL read-back a silent certificate downgrade.
    assert seen["body"] == {}


@pytest.mark.asyncio
async def test_add_domain_sends_acme_when_it_is_stated_either_way():
    """Stated values DO travel — both of them. ``False`` is a deliberate
    downgrade and must reach the daemon, not be optimised away as "the default"."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(httpx.Response(200, content=request.content).json())
        return httpx.Response(200, json=_ADDED_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.add_domain("demo", "app.example.com", acme=True, idempotency_key="k1")
    await client.add_domain("demo", "app.example.com", acme=False, idempotency_key="k2")

    assert bodies == [{"acme": True}, {"acme": False}]


async def test_add_domain_sends_acme_as_typed_without_a_local_pre_check():
    """Whether ACME is enabled is a DAEMON fact; the client never guesses it."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_ADDED_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.add_domain("demo", "app.example.com", acme=True, idempotency_key="k2")
    assert seen["body"] == {"acme": True}


async def test_add_domain_sends_a_hostile_name_as_one_encoded_segment():
    """A name with a slash must never become two path segments."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(200, json=_ADDED_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.add_domain("demo", "evil/../x", idempotency_key="k3")
    assert seen["raw_path"] == "/api/services/demo/domains/evil%2F..%2Fx"


async def test_add_domain_omits_the_header_without_a_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json=_ADDED_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    await client.add_domain("demo", "app.example.com")
    assert seen["idem"] is None


async def test_remove_domain_deletes_with_an_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(
            200, json={"service_name": "demo", "domain": "app.example.com", "removed": True}
        )

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.remove_domain("demo", "App.Example.COM.", idempotency_key="k4")

    assert result["removed"] is True
    assert seen["method"] == "DELETE"
    # Sent as typed — the daemon folds case and the trailing dot, not the client.
    assert seen["path"] == "/api/services/demo/domains/App.Example.COM."
    assert seen["idem"] == "k4"


async def test_add_domain_refusal_raises_http_status_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "domain.taken",
                "message": "Domain 'app.example.com' is already in use.",
                "hint": "Remove it from the service that holds it first.",
                "detail": "Domain 'app.example.com' is already in use.",
            },
        )

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await client.add_domain("demo", "app.example.com", idempotency_key="k5")
    assert excinfo.value.response.status_code == 409


# -- async command bodies ----------------------------------------------------------


def _fake_client(**overrides) -> SimpleNamespace:
    calls = {
        "list_domains": AsyncMock(
            return_value={"service_name": "demo", "domains": [dict(_ADDED_OK)]}
        ),
        "add_domain": AsyncMock(return_value=dict(_ADDED_OK)),
        "remove_domain": AsyncMock(
            return_value={"service_name": "demo", "domain": "app.example.com", "removed": True}
        ),
    }
    calls.update(overrides)
    return SimpleNamespace(**calls)


@pytest.fixture
def fake_client(monkeypatch):
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    return client


async def test_add_prints_the_url_and_the_dns_note(fake_client, capsys):
    await _add_async("demo", "app.example.com", acme=False)

    assert fake_client.add_domain.await_args.args == ("demo", "app.example.com")
    kwargs = fake_client.add_domain.await_args.kwargs
    assert kwargs["acme"] is False
    assert kwargs["idempotency_key"]  # minted per invocation

    out = capsys.readouterr().out
    assert "app.example.com" in out
    assert "demo" in out
    assert "https://app.example.com/" in out
    assert "Point DNS" in out
    assert "nerdit trust" in out


async def test_add_echoes_the_daemon_folded_name_not_argv(monkeypatch, capsys):
    client = _fake_client(add_domain=AsyncMock(return_value=dict(_ADDED_OK)))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "App.Example.COM.", acme=False)

    out = capsys.readouterr().out
    assert "app.example.com" in out
    assert "App.Example.COM." not in out


async def test_add_passes_acme_through_unchecked(fake_client):
    """No local pre-check: --acme reaches the daemon, which decides."""
    await _add_async("demo", "app.example.com", acme=True)
    assert fake_client.add_domain.await_args.kwargs["acme"] is True


async def test_add_withheld_prints_the_state_note(monkeypatch, capsys):
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "state": "withheld", "url": None})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=False)

    out = capsys.readouterr().out
    assert "withheld" in out
    assert "not routed yet" in out


async def test_add_ready_prints_no_state_note(fake_client, capsys):
    await _add_async("demo", "app.example.com", acme=False)
    assert "not routed yet" not in capsys.readouterr().out


async def test_add_pending_prints_the_cert_note(monkeypatch, capsys):
    """(P26 WP2) A stored ``acme`` row whose leaf is not on disk yet must say so
    at the moment of the add: until it is issued, clients still see the internal
    CA, and an operator who does not know that reads the green success line as
    "browsers are happy now"."""
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "pending"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    out = capsys.readouterr().out
    assert "pending" in out
    assert "has not been issued yet" in out


async def test_add_disabled_prints_the_cert_note(monkeypatch, capsys):
    """The ``disabled`` state can only be reached by a row bound while ACME was
    on, on a node that later turned it off — so the note names the fix
    (``proxy.acme``), not the symptom."""
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "disabled"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    out = capsys.readouterr().out
    assert "ACME is off on this node" in out
    assert "proxy.acme" in out


async def test_add_expired_prints_the_cert_note(monkeypatch, capsys):
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "expired"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    assert "has expired" in capsys.readouterr().out


async def test_add_expired_drops_the_trust_clause(monkeypatch, capsys):
    """(review round 2) ``expired`` is only ever an ``acme=1`` row on a node with
    ACME on — it is derived from an on-disk ACME leaf whose ``not_after`` has
    passed. The subject is therefore claimed by the acme-only automation policy,
    Caddy keeps serving the stale public leaf while renewal fails, and installing
    this node's internal root changes nothing a client sees. The clause was
    dropped for ``issued`` and ``pending`` in round 1 for the same reason and
    ``expired`` was simply missed.

    ``disabled`` is the state that keeps it, and does — an ``acme=1`` row on a
    node with ACME off IS an internal name (pinned below).
    """
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "expired"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    out = capsys.readouterr().out
    assert "Point DNS for app.example.com at this machine." in out
    assert "has expired" in out
    assert "Clients trust the internal CA" not in out


async def test_add_disabled_keeps_the_trust_clause(monkeypatch, capsys):
    """The other side of the same rule: with ACME off the name really is served
    by the internal CA, so dropping the trust pointer there would delete a true
    instruction."""
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "disabled"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    assert "Clients trust the internal CA" in capsys.readouterr().out


async def test_add_internal_prints_no_cert_note(fake_client, capsys):
    """``internal`` is the DEFAULT binding, not a defect — the DNS note already
    tells the operator about ``nerdit trust``, so a second yellow line would be
    noise on the ordinary path."""
    await _add_async("demo", "app.example.com", acme=False)

    out = capsys.readouterr().out
    assert "has not been issued yet" not in out
    assert "nerdit trust" in out


async def test_add_pending_drops_the_trust_clause(monkeypatch, capsys):
    """(review round 1) After ``--acme`` the trust step is what the operator is
    paying a public CA to remove, so the dim DNS line must not end by telling
    them to run ``nerdit trust`` — the cert note directly above already says
    what a client sees meanwhile. DNS is still theirs, so that half stays."""
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "pending"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    out = capsys.readouterr().out
    assert "Point DNS for app.example.com at this machine." in out
    assert "Clients trust the internal CA" not in out
    assert "has not been issued yet" in out
    # (review round 2) And the cert note itself no longer offers `nerdit trust`
    # as a workaround: a pending ACME subject serves NO leaf, measured against
    # real Caddy in tests/test_acme_smoke.py, so the internal root is not what
    # the client would see either.
    assert "nerdit trust does not help" in out


async def test_add_issued_prints_no_cert_note(monkeypatch, capsys):
    """Nothing to act on: a public certificate is the outcome the operator
    asked for."""
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "acme": True, "cert_state": "issued"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    out = capsys.readouterr().out
    assert "issued —" not in out
    # …and the WP1 trust sentence is simply false on an issued row.
    assert "Clients trust the internal CA" not in out
    assert "Point DNS" in out


async def test_add_tolerates_an_unknown_cert_state(monkeypatch, capsys):
    """A vocabulary this build does not know prints NO line rather than a wrong
    sentence — the ``_STATE_NOTES`` rule, so an older CLI against a newer daemon
    degrades quietly."""
    client = _fake_client(
        add_domain=AsyncMock(return_value={**_ADDED_OK, "cert_state": "renewing"})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "app.example.com", acme=True)

    out = capsys.readouterr().out
    assert "renewing —" not in out
    # An unknown token KEEPS the trust clause: dropping a true instruction is
    # the worse error of the two, so the unknown side falls on "say it".
    assert "Clients trust the internal CA" in out


async def test_add_escapes_a_hostile_domain_string(monkeypatch, capsys):
    """(P6 MarkupError lesson) The domain is operator input echoed back by the
    daemon; it is not ours to trust as Rich markup."""
    client = _fake_client(
        add_domain=AsyncMock(
            return_value={**_ADDED_OK, "domain": "[red]evil.example", "url": "https://[red]evil/"}
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _add_async("demo", "[red]evil.example", acme=False)

    out = capsys.readouterr().out
    assert "[red]evil.example" in out


async def test_add_refusal_renders_the_daemon_hint_and_exits_1(monkeypatch, capsys):
    response = httpx.Response(
        409,
        json={
            "code": "domain.acme_disabled",
            "message": "ACME is not enabled on this node.",
            "hint": (
                "Set [proxy.acme] enabled=true and email, restart the daemon, "
                "or add the domain without --acme."
            ),
            "detail": "ACME is not enabled on this node.",
        },
        request=httpx.Request(
            "PUT", "http://localhost:9321/api/services/demo/domains/app.example.com"
        ),
    )
    client = _fake_client(
        add_domain=AsyncMock(
            side_effect=httpx.HTTPStatusError("409", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _add_async("demo", "app.example.com", acme=True)
    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "ACME is not enabled on this node" in out
    assert "without --acme" in out


async def test_add_taken_hint_is_rendered(monkeypatch, capsys):
    response = httpx.Response(
        409,
        json={
            "code": "domain.taken",
            "message": "Domain 'app.example.com' is already in use.",
            "hint": "Remove it from the service that holds it first.",
            "detail": "Domain 'app.example.com' is already in use.",
        },
        request=httpx.Request(
            "PUT", "http://localhost:9321/api/services/demo/domains/app.example.com"
        ),
    )
    client = _fake_client(
        add_domain=AsyncMock(
            side_effect=httpx.HTTPStatusError("409", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _add_async("demo", "app.example.com", acme=False)
    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "already in use" in out
    assert "Remove it from the service" in out


async def test_remove_reports_the_removal(fake_client, capsys):
    await _remove_async("demo", "app.example.com")

    assert fake_client.remove_domain.await_args.args == ("demo", "app.example.com")
    assert fake_client.remove_domain.await_args.kwargs["idempotency_key"]
    out = capsys.readouterr().out
    assert "Removed" in out
    assert "app.example.com" in out


async def test_remove_of_an_unknown_domain_is_a_no_op_line(monkeypatch, capsys):
    client = _fake_client(
        remove_domain=AsyncMock(
            return_value={"service_name": "demo", "domain": "gone.example", "removed": False}
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _remove_async("demo", "gone.example")

    out = capsys.readouterr().out
    assert "nothing to do" in out
    assert "Removed" not in out


async def test_remove_daemon_error_exits_1(monkeypatch, capsys):
    unreachable = httpx.ConnectError("down")
    unreachable.request = httpx.Request(
        "DELETE", "http://localhost:9321/api/services/demo/domains/app.example.com"
    )
    client = _fake_client(remove_domain=AsyncMock(side_effect=unreachable))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _remove_async("demo", "app.example.com")
    assert excinfo.value.exit_code == 1


async def test_list_empty_says_so_plainly(monkeypatch, capsys):
    client = _fake_client(
        list_domains=AsyncMock(return_value={"service_name": "demo", "domains": []})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _list_async("demo")

    out = capsys.readouterr().out
    assert "has no direct domains" in out


async def test_list_renders_a_table(fake_client, capsys):
    await _list_async("demo")

    out = capsys.readouterr().out
    assert "app.example.com" in out
    assert "ready" in out


async def test_list_renders_the_cert_column(monkeypatch, capsys):
    """(P26 WP2) ``state`` and ``cert_state`` are independent axes, so the table
    shows both: a row can be routed and still serve a certificate a browser
    rejects, and one column could not say so."""
    client = _fake_client(
        list_domains=AsyncMock(
            return_value={
                "service_name": "demo",
                "domains": [{**_ADDED_OK, "acme": True, "cert_state": "issued"}],
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _list_async("demo")

    out = capsys.readouterr().out
    assert "cert" in out
    assert "issued" in out


async def test_list_escapes_a_hostile_cert_state(monkeypatch, capsys):
    """(P6 MarkupError lesson) ``cert_state`` is daemon-derived text reaching
    ``console.print``; a value carrying Rich markup must render, not raise."""
    client = _fake_client(
        list_domains=AsyncMock(
            return_value={
                "service_name": "demo",
                "domains": [{**_ADDED_OK, "cert_state": "[red]x"}],
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _list_async("demo")

    assert "[red]x" in capsys.readouterr().out


async def test_list_renders_a_row_with_no_cert_state_at_all(monkeypatch, capsys):
    """A pre-WP2 daemon omits the key entirely; the column shows the ``-``
    placeholder rather than the table blowing up."""
    row = {k: v for k, v in _ADDED_OK.items() if k != "cert_state"}
    client = _fake_client(
        list_domains=AsyncMock(return_value={"service_name": "demo", "domains": [row]})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _list_async("demo")

    assert "app.example.com" in capsys.readouterr().out


async def test_list_escapes_a_hostile_row(monkeypatch, capsys):
    client = _fake_client(
        list_domains=AsyncMock(
            return_value={
                "service_name": "demo",
                "domains": [{**_ADDED_OK, "domain": "[red]x.example"}],
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _list_async("demo")
    assert "x.example" in capsys.readouterr().out


async def test_list_daemon_error_exits_1(monkeypatch, capsys):
    unreachable = httpx.ConnectError("down")
    unreachable.request = httpx.Request("GET", "http://localhost:9321/api/services/demo/domains")
    client = _fake_client(list_domains=AsyncMock(side_effect=unreachable))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _list_async("demo")
    assert excinfo.value.exit_code == 1


# -- argv-level ---------------------------------------------------------------------

_runner = CliRunner()


def test_cli_domains_add_argv(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["domains", "add", "demo", "app.example.com"])
    assert result.exit_code == 0, result.output
    assert client.add_domain.await_args.args == ("demo", "app.example.com")
    # Tri-state (review round 1): NEITHER flag is "leave it alone", not "off" —
    # so a re-add to re-read the URL cannot downgrade an issued certificate.
    assert client.add_domain.await_args.kwargs["acme"] is None


def test_cli_domains_add_argv_parses_acme(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["domains", "add", "demo", "app.example.com", "--acme"])
    assert result.exit_code == 0, result.output
    assert client.add_domain.await_args.kwargs["acme"] is True


def test_cli_domains_add_argv_parses_no_acme(monkeypatch):
    """``--no-acme`` is the explicit way DOWN — the only way to turn an issued
    public certificate back into an internal-CA leaf from the CLI."""
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["domains", "add", "demo", "app.example.com", "--no-acme"])
    assert result.exit_code == 0, result.output
    assert client.add_domain.await_args.kwargs["acme"] is False


def test_cli_domains_remove_argv(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["domains", "remove", "demo", "app.example.com"])
    assert result.exit_code == 0, result.output
    client.remove_domain.assert_awaited_once()


def test_cli_domains_list_argv(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["domains", "list", "demo"])
    assert result.exit_code == 0, result.output
    client.list_domains.assert_awaited_once_with("demo")


def test_cli_domains_add_help_names_the_acme_requirement():
    """The flag is discoverable and honest: it says what the daemon needs, and
    which code comes back when it does not have it."""
    from nerdit.cli.app import app

    result = _runner.invoke(app, ["domains", "add", "--help"])
    assert result.exit_code == 0, result.output
    plain = " ".join(result.output.split())
    assert "ACME HTTP-01" in plain
    assert "domain.acme_disabled" in plain
    assert "later release" not in plain
    # Typer renders option help through Rich, so a bare ``[proxy.acme]`` is
    # eaten as a style tag and the operator reads "Requires .enabled". The help
    # string escapes it; this pins that it survives to the terminal.
    assert "[proxy.acme].enabled" in plain


def test_cli_domains_bare_shows_help():
    from nerdit.cli.app import app

    result = _runner.invoke(app, ["domains"])
    assert "list" in result.output
    assert "add" in result.output
    assert "remove" in result.output
