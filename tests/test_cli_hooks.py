"""`nerdit hooks`: wire shape of the three client calls and the CLI output."""

from __future__ import annotations

import json

import httpx
import pytest

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.hooks import _add_async, _list_async, _rm_async

SECRET = "whsec_never_echoed"  # noqa: S105 - a test literal
HOOK = {
    "name": "stripe",
    "sender": "stripe",
    "path": "/webhooks/stripe",
    "header": None,
    "prefix": None,
    "secret_ref": "${secrets.NERDIT_HOOK_STRIPE_SECRET}",
    "secret_key": "NERDIT_HOOK_STRIPE_SECRET",
    "url": "https://demo--box.nodes.example/.nerdit/hooks/abcdefghijklmnopqrst",
    "created_at": "2026-10-10T10:00:00Z",
}


def _wire(monkeypatch, handler) -> list[tuple[str, str, str | None, bytes]]:
    seen: list[tuple[str, str, str | None, bytes]] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.raw_path.decode(),
                request.headers.get("idempotency-key"),
                request.content,
            )
        )
        return handler(request)

    client = NerditClient("localhost", 9321, token="t", transport=httpx.MockTransport(record))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    return seen


async def test_hooks_list_renders_name_sender_path_url(monkeypatch, capsys):
    seen = _wire(
        monkeypatch,
        lambda r: httpx.Response(
            200, json={"service_name": "demo", "hooks": [{**HOOK, "path": "/[bold]x"}]}
        ),
    )
    await _list_async("demo")
    out = capsys.readouterr().out
    assert seen[0][:3] == ("GET", "/api/services/demo/hooks", None)
    assert "stripe" in out and "/[bold]x" in out  # server text escaped, not parsed as markup
    assert "/.nerdit/hooks/" in out


async def test_hooks_add_with_secret_prompt_posts_hook_then_the_secrets_route_and_never_echoes_it(
    monkeypatch, capsys
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/hooks"):
            return httpx.Response(200, json={**HOOK, "created": True})
        return httpx.Response(200, json={"service": "demo", "keys": [HOOK["secret_key"]]})

    seen = _wire(monkeypatch, handler)
    prompts: list[dict] = []

    def fake_prompt(text, **kw):  # noqa: ANN001, ANN003, ANN202
        prompts.append({"text": text, **kw})
        return SECRET

    monkeypatch.setattr("typer.prompt", fake_prompt)
    await _add_async("demo", "stripe", "stripe", "/webhooks/stripe", None, None, secret_prompt=True)

    (m1, p1, k1, b1), (m2, p2, k2, b2) = seen
    assert (m1, p1) == ("POST", "/api/services/demo/hooks") and k1
    assert json.loads(b1) == {"name": "stripe", "sender": "stripe", "path": "/webhooks/stripe"}
    assert SECRET.encode() not in b1
    assert (m2, p2) == ("POST", "/api/secrets/demo") and k2 and k2 != k1
    assert json.loads(b2) == {"values": {"NERDIT_HOOK_STRIPE_SECRET": SECRET}}
    assert prompts[0]["hide_input"] is True
    out = capsys.readouterr().out
    assert SECRET not in out
    assert HOOK["url"] in out


async def test_hooks_add_without_url_prints_the_link_hint(monkeypatch, capsys):
    _wire(monkeypatch, lambda r: httpx.Response(200, json={**HOOK, "url": None, "created": True}))
    await _add_async("demo", "stripe", "stripe", "/w", None, None, secret_prompt=False)
    out = capsys.readouterr().out
    assert "link this machine and share demo" in out
    assert "--prompt NERDIT_HOOK_STRIPE_SECRET" in " ".join(out.split())


async def test_hooks_rm_sends_an_idempotency_key(monkeypatch, capsys):
    seen = _wire(
        monkeypatch,
        lambda r: httpx.Response(200, json={"service_name": "demo", "name": "a", "removed": True}),
    )
    await _rm_async("demo", "a")
    method, path, key, _ = seen[0]
    assert (method, path) == ("DELETE", "/api/services/demo/hooks/a") and key
    assert "Removed webhook" in capsys.readouterr().out


@pytest.mark.parametrize("hook", ["..", "a/b"])
async def test_hook_name_cannot_escape_its_path_segment(monkeypatch, hook):
    seen = _wire(
        monkeypatch,
        lambda r: httpx.Response(200, json={"service_name": "demo", "name": "x", "removed": False}),
    )
    await _rm_async("demo", hook)
    assert seen[0][1].startswith("/api/services/demo/hooks/")
    assert seen[0][1].count("/") == 5
