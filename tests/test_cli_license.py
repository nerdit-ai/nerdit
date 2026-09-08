"""CLI tests for ``nerdit license install|status|remove`` (P17d D-LIC5).

Same three-layer structure as ``test_cli_link.py``: the ``NerditClient``
methods over ``httpx.MockTransport`` (wire shape: path, bearer,
``Idempotency-Key``, exact JSON body), the async command bodies against a fake
client, and argv-level parsing through Typer's ``CliRunner``.

The two security-relevant assertions, repeated on every path that can print:

* **the blob never appears in output** — not on success, not on a daemon
  refusal, not when the argument was not a file (the case where the "path" the
  operator typed is most likely the blob itself);
* **the blob never rides argv** — there is no code path that accepts it as a
  positional value, which is what keeps it out of shell history and ``ps``.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.license import _install_async, _remove_async, _status_async

BLOB = "hdr.pyld.sig-this-is-the-secret-artifact"
CUSTOMER = "cus_p17d_golden"

_INSTALL_OK = {
    "lid": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
    "plan": "pro",
    "features": ["remote_link"],
    "state": "valid",
    "expires_at": "2027-01-01T00:00:00+00:00",
    "expires_in_s": 1000,
    "customer_id": CUSTOMER,
    "installed": True,
}


# -- client methods ---------------------------------------------------------------


async def test_install_license_posts_the_blob_with_an_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return httpx.Response(200, json=_INSTALL_OK)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.install_license(BLOB, idempotency_key="key-1")

    assert result == _INSTALL_OK
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/license"
    assert seen["auth"] == "Bearer t"
    assert seen["idem"] == "key-1"
    # The blob rides the JSON body and nothing else — never a query param, never
    # a header, never a URL segment.
    assert seen["body"] == {"blob": BLOB}


async def test_remove_license_deletes_with_an_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"removed": True})

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    result = await client.remove_license(idempotency_key="key-2")

    assert result == {"removed": True}
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/license"
    assert seen["idem"] == "key-2"


async def test_install_license_refusal_raises_http_status_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "license.invalid",
                "message": "The license did not verify (bad_signature).",
                "reason": "bad_signature",
            },
        )

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await client.install_license(BLOB)
    assert excinfo.value.response.status_code == 422


# -- async command bodies ----------------------------------------------------------


def _fake_client(**overrides) -> SimpleNamespace:
    calls = {
        "install_license": AsyncMock(return_value=dict(_INSTALL_OK)),
        "remove_license": AsyncMock(return_value={"removed": True}),
        "get_config": AsyncMock(
            return_value={"section": "license", "values": {"file": None}, "etag": "e"}
        ),
        "get_capabilities": AsyncMock(return_value={"license": {"installed": False}}),
    }
    calls.update(overrides)
    return SimpleNamespace(**calls)


@pytest.fixture
def fake_client(monkeypatch):
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    return client


@pytest.fixture
def license_file(tmp_path: Path) -> Path:
    path = tmp_path / "license.jws"
    path.write_text(f"{BLOB}\n")
    return path


async def test_install_from_a_file_sends_the_blob_and_prints_none_of_it(
    fake_client, license_file, capsys
):
    await _install_async(str(license_file))

    fake_client.install_license.assert_awaited_once()
    args, kwargs = fake_client.install_license.await_args
    assert args[0] == BLOB  # trailing newline stripped
    assert kwargs["idempotency_key"]  # minted per invocation

    out = capsys.readouterr().out
    assert "License installed" in out
    assert "pro" in out
    assert BLOB not in out
    assert "No restart needed" in out


async def test_install_reads_stdin_when_the_argument_is_a_dash(fake_client, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{BLOB}\n"))

    await _install_async("-")

    assert fake_client.install_license.await_args.args[0] == BLOB
    assert BLOB not in capsys.readouterr().out


async def test_install_reads_piped_stdin_when_no_argument_is_given(
    fake_client, monkeypatch, capsys
):
    """A pipe is a first-class input; a bare TTY is not (see the next test)."""
    monkeypatch.setattr("sys.stdin", io.StringIO(f"{BLOB}\n"))

    await _install_async(None)

    assert fake_client.install_license.await_args.args[0] == BLOB


async def test_install_with_no_argument_on_a_tty_prints_usage_instead_of_hanging(
    fake_client, monkeypatch, capsys
):
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True, read=lambda: ""))

    with pytest.raises(typer.Exit) as excinfo:
        await _install_async(None)

    assert excinfo.value.exit_code == 1
    fake_client.install_license.assert_not_awaited()
    assert "nerdit license install ./license.jws" in capsys.readouterr().out


async def test_install_refuses_a_blob_passed_as_the_argument_without_echoing_it(
    fake_client, capsys
):
    """The whole point of the FILE-or-stdin interface (D-LIC5).

    A blob on argv lands in shell history and ``ps``. There is no code path that
    accepts one — and the refusal does not print the value, because the value is
    exactly the thing that must not reach scrollback.
    """
    with pytest.raises(typer.Exit) as excinfo:
        await _install_async(BLOB)

    assert excinfo.value.exit_code == 1
    fake_client.install_license.assert_not_awaited()
    out = capsys.readouterr().out
    assert "No such license file" in out
    assert "never takes the blob as an argument" in out
    assert BLOB not in out


async def test_install_refuses_an_empty_file(fake_client, tmp_path, capsys):
    empty = tmp_path / "empty.jws"
    empty.write_text("\n")

    with pytest.raises(typer.Exit):
        await _install_async(str(empty))

    fake_client.install_license.assert_not_awaited()
    assert "empty" in capsys.readouterr().out


@pytest.mark.parametrize(
    "state,expected",
    [("expired_grace", "grace period"), ("expired", "not currently valid")],
)
async def test_install_warns_when_the_installed_license_is_not_valid(
    monkeypatch, license_file, capsys, state, expected
):
    """Temporal states install (D4) — the CLI is the one that says "but"."""
    client = _fake_client(
        install_license=AsyncMock(return_value={**_INSTALL_OK, "state": state}),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _install_async(str(license_file))

    out = capsys.readouterr().out
    assert expected in out
    assert BLOB not in out


async def test_install_daemon_refusal_exits_1_without_echoing_the_blob(
    monkeypatch, license_file, capsys
):
    response = httpx.Response(
        422,
        json={
            "code": "license.invalid",
            "message": "The license did not verify (bad_signature).",
            "hint": "Check that the whole one-line blob was pasted.",
            "reason": "bad_signature",
        },
        request=httpx.Request("POST", "http://localhost:9321/api/license"),
    )
    client = _fake_client(
        install_license=AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "refused", request=response.request, response=response
            )
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _install_async(str(license_file))

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "bad_signature" in out  # the machine token IS the actionable half
    assert BLOB not in out


# -- status: the two truths -------------------------------------------------------


async def test_status_on_an_unlicensed_daemon_reads_both_surfaces(fake_client, capsys):
    await _status_async()

    fake_client.get_config.assert_awaited_once_with("license")
    fake_client.get_capabilities.assert_awaited_once()
    out = capsys.readouterr().out
    assert "installed   = no" in out
    assert "<data_dir>/license.jws" in out
    assert "nerdit license install" in out


async def test_status_renders_the_live_block_and_the_persisted_path(monkeypatch, capsys):
    client = _fake_client(
        get_config=AsyncMock(
            return_value={"section": "license", "values": {"file": "/srv/lic.jws"}}
        ),
        get_capabilities=AsyncMock(
            return_value={
                "license": {
                    "installed": True,
                    "state": "valid",
                    "plan": "pro",
                    "features": ["remote_link"],
                    "lid": _INSTALL_OK["lid"],
                    "expires_at": "2027-01-01T00:00:00+00:00",
                    "expires_in_s": 1000,
                    "customer_id": CUSTOMER,
                }
            }
        ),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _status_async()

    out = capsys.readouterr().out
    assert "/srv/lic.jws" in out
    assert "state       = valid" in out
    assert "remote_link" in out
    assert "2027-01-01T00:00:00+00:00" in out
    assert CUSTOMER in out


async def test_status_omits_the_customer_line_when_the_daemon_omitted_the_field(
    monkeypatch, capsys
):
    """A non-admin caller gets no ``customer_id`` — print no placeholder either.

    A ``customer    = -`` line would read as "no customer id on this license",
    which is a different (and false) statement from "you are not allowed to see
    it".
    """
    client = _fake_client(
        get_capabilities=AsyncMock(
            return_value={
                "license": {
                    "installed": True,
                    "state": "valid",
                    "plan": "pro",
                    "features": ["remote_link"],
                    "lid": _INSTALL_OK["lid"],
                    "expires_at": "2027-01-01T00:00:00+00:00",
                    "expires_in_s": 1000,
                }
            }
        ),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _status_async()

    out = capsys.readouterr().out
    assert "customer" not in out
    assert "state       = valid" in out


@pytest.mark.parametrize(
    "state,expected",
    [("expired_grace", "grace period"), ("expired", "past the grace period")],
)
async def test_status_flags_the_temporal_states(monkeypatch, capsys, state, expected):
    client = _fake_client(
        get_capabilities=AsyncMock(
            return_value={
                "license": {
                    "installed": True,
                    "state": state,
                    "plan": "pro",
                    "features": ["remote_link"],
                    "lid": _INSTALL_OK["lid"],
                    "expires_at": "2026-01-01T00:00:00+00:00",
                    "expires_in_s": 0,
                }
            }
        ),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _status_async()
    assert expected in capsys.readouterr().out


async def test_status_on_an_invalid_license_prints_the_reason_token(monkeypatch, capsys):
    client = _fake_client(
        get_capabilities=AsyncMock(
            return_value={
                "license": {"installed": True, "state": "invalid", "reason": "unknown_kid"}
            }
        ),
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _status_async()

    out = capsys.readouterr().out
    assert "reason      = unknown_kid" in out
    assert "did not verify" in out


async def test_status_daemon_error_exits_1(monkeypatch):
    response = httpx.Response(
        403,
        json={"code": "forbidden", "message": "Admin role required."},
        request=httpx.Request("GET", "http://localhost:9321/api/capabilities"),
    )
    client = _fake_client(
        get_config=AsyncMock(
            side_effect=httpx.HTTPStatusError("nope", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _status_async()
    assert excinfo.value.exit_code == 1


# -- remove -----------------------------------------------------------------------


async def test_remove_decline_makes_zero_client_calls(monkeypatch, capsys):
    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)

    with pytest.raises(typer.Exit) as excinfo:
        await _remove_async(False)

    assert excinfo.value.exit_code == 1
    client.remove_license.assert_not_awaited()
    assert "Aborted" in capsys.readouterr().out


async def test_remove_yes_calls_the_route_with_a_minted_key(fake_client, capsys):
    await _remove_async(True)

    fake_client.remove_license.assert_awaited_once()
    assert fake_client.remove_license.await_args.kwargs["idempotency_key"]
    assert "License removed." in capsys.readouterr().out


async def test_remove_when_nothing_is_installed_says_so_honestly(monkeypatch, capsys):
    client = _fake_client(remove_license=AsyncMock(return_value={"removed": False}))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    await _remove_async(True)

    out = capsys.readouterr().out
    assert "nothing to remove" in out
    assert "License removed." not in out


async def test_remove_daemon_error_exits_1(monkeypatch):
    response = httpx.Response(
        403,
        json={"code": "forbidden", "message": "Admin role required."},
        request=httpx.Request("DELETE", "http://localhost:9321/api/license"),
    )
    client = _fake_client(
        remove_license=AsyncMock(
            side_effect=httpx.HTTPStatusError("nope", request=response.request, response=response)
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    with pytest.raises(typer.Exit) as excinfo:
        await _remove_async(True)
    assert excinfo.value.exit_code == 1


# -- argv-level ---------------------------------------------------------------------

_runner = CliRunner()


def test_cli_license_install_argv_takes_a_file(monkeypatch, license_file):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["license", "install", str(license_file)])

    assert result.exit_code == 0, result.output
    assert client.install_license.await_args.args[0] == BLOB
    assert BLOB not in result.output


def test_cli_license_status_argv(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    result = _runner.invoke(app, ["license", "status"])

    assert result.exit_code == 0, result.output
    client.get_capabilities.assert_awaited_once()


def test_cli_license_remove_argv_needs_yes_or_a_prompt(monkeypatch):
    from nerdit.cli.app import app

    client = _fake_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)

    assert _runner.invoke(app, ["license", "remove", "--yes"]).exit_code == 0
    client.remove_license.assert_awaited_once()

    # Declining at the prompt is exit 1 with no call (input "n" on stdin).
    declined = _runner.invoke(app, ["license", "remove"], input="n\n")
    assert declined.exit_code == 1
    assert client.remove_license.await_count == 1


def test_cli_license_group_is_registered_with_all_three_verbs():
    from nerdit.cli.app import app

    result = _runner.invoke(app, ["license", "--help"])

    assert result.exit_code == 0
    for verb in ("install", "status", "remove"):
        assert verb in result.output
