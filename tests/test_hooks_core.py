"""Inbound webhook verification and the hook gate (core/hooks.py)."""

from __future__ import annotations

import hashlib
import hmac
import threading
from datetime import UTC, datetime, timedelta

import pytest

from nerdit.core import hooks as hooks_mod
from nerdit.core.hooks import (
    HookAdmission,
    HookGate,
    default_secret_ref,
    hook_url,
    new_hook_id,
    parse_received_at,
    secret_key,
    url_env_key,
    verify,
)
from nerdit.db.rows import Hook

GITHUB_SECRET = "It's a Secret to Everybody"
GITHUB_SIG = "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17"
SLACK_SECRET = "8f742231b10e8888abcd99yyyzzz85a5"
SLACK_TS = "1531420618"
SLACK_BODY = (
    b"token=xyzz0WbapA4vBCDEFasx0q6G&team_id=T1DC2JH3J&team_domain=testteamnow&"
    b"channel_id=G8PSS9T3V&channel_name=foobar&user_id=U2CERLKJA&user_name=roadrunner&"
    b"command=%2Fwebhook-collect&text=&response_url=https%3A%2F%2Fhooks.slack.com%2F"
    b"commands%2FT1DC2JH3J%2F397700885554%2F96rGlfmibIGlgcZRskXaIFfN&"
    b"trigger_id=398738663015.47445629121.803a0bc887a14d10d2c447fce8b6703c"
)
SLACK_SIG = "v0=a2114d57b48eac39b9ad189dd8316235a7b4a8d21a10bd27519666489c69b503"
SLACK_AT = datetime.fromtimestamp(int(SLACK_TS), UTC)

STRIPE_SECRET = "whsec_test"
STRIPE_BODY = b'{"id":"evt_1"}'


def _hex(secret: str, msg: bytes) -> str:
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def _stripe_header(ts: str, *v1: str, v0: str | None = None) -> dict[str, str]:
    parts = [f"t={ts}", *(f"v1={s}" for s in v1)]
    if v0:
        parts.append(f"v0={v0}")
    return {"stripe-signature": ",".join(parts)}


def _stripe_sig(ts: str, body: bytes = STRIPE_BODY) -> str:
    return _hex(STRIPE_SECRET, ts.encode() + b"." + body)


def _slack_headers(ts: str = SLACK_TS, sig: str = SLACK_SIG) -> dict[str, str]:
    return {"x-slack-request-timestamp": ts, "x-slack-signature": sig}


# --- ids and names ---


def test_new_hook_id_is_20_base32_chars_and_unique():
    ids = {new_hook_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(hooks_mod.HOOK_ID_RE.fullmatch(i) for i in ids)


def test_secret_and_url_env_keys_map_hyphens_to_underscores():
    assert secret_key("stripe-live") == "NERDIT_HOOK_STRIPE_LIVE_SECRET"
    assert url_env_key("stripe-live") == "NERDIT_HOOK_STRIPE_LIVE_URL"
    assert default_secret_ref("gh") == "${secrets.NERDIT_HOOK_GH_SECRET}"


def test_hook_url_composes_the_hosted_host_and_none_when_unlinked_or_label_too_long():
    hid = "a" * 20
    assert (
        hook_url("shop", hid, "abc", "nodes.nerdit.ai")
        == f"https://shop--abc.nodes.nerdit.ai/.nerdit/hooks/{hid}"
    )
    assert hook_url("shop", hid, None, "nodes.nerdit.ai") is None
    assert hook_url("shop", hid, "abc", None) is None
    assert hook_url("s" * 60, hid, "abc", "nodes.nerdit.ai") is None


# --- recipes ---


def test_github_signature_matches_the_published_vector():
    at = datetime.now(UTC)
    headers = {"x-hub-signature-256": GITHUB_SIG}
    assert verify("github", GITHUB_SECRET, headers, b"Hello, World!", at)
    assert not verify("github", "other", headers, b"Hello, World!", at)


def test_slack_signature_matches_the_published_vector():
    assert verify("slack", SLACK_SECRET, _slack_headers(), SLACK_BODY, SLACK_AT)


def test_stripe_signature_accepts_any_v1_and_rejects_v0_only():
    ts = "1700000000"
    at = datetime.fromtimestamp(int(ts), UTC)
    good = _stripe_sig(ts)
    assert verify("stripe", STRIPE_SECRET, _stripe_header(ts, "0" * 64, good), STRIPE_BODY, at)
    assert verify(
        "stripe", STRIPE_SECRET, {"stripe-signature": f" t={ts} , v1={good} "}, STRIPE_BODY, at
    )
    assert not verify("stripe", STRIPE_SECRET, _stripe_header(ts, v0=good), STRIPE_BODY, at)


def test_stripe_and_slack_window_is_measured_against_received_at_not_now():
    """A delivery replayed hours later still verifies against its Received-At."""
    ts = "1700000000"
    at = datetime.fromtimestamp(int(ts) + 299, UTC)
    assert verify("stripe", STRIPE_SECRET, _stripe_header(ts, _stripe_sig(ts)), STRIPE_BODY, at)
    assert verify(
        "slack", SLACK_SECRET, _slack_headers(), SLACK_BODY, SLACK_AT + timedelta(seconds=299)
    )


def test_stripe_and_slack_reject_a_timestamp_outside_300_seconds():
    ts = "1700000000"
    late = datetime.fromtimestamp(int(ts) + 301, UTC)
    early = datetime.fromtimestamp(int(ts) - 301, UTC)
    for at in (late, early):
        assert not verify(
            "stripe", STRIPE_SECRET, _stripe_header(ts, _stripe_sig(ts)), STRIPE_BODY, at
        )
    assert not verify(
        "slack", SLACK_SECRET, _slack_headers(), SLACK_BODY, SLACK_AT + timedelta(seconds=301)
    )


@pytest.mark.parametrize("ts", ["+1531420618", "1_531_420_618", " 1531420618", "١٥٣١٤٢٠٦١٨"])
def test_timestamps_must_be_plain_digits(ts):
    assert not verify("slack", SLACK_SECRET, _slack_headers(ts=ts), SLACK_BODY, SLACK_AT)
    stripe = _stripe_header(ts, _stripe_sig(ts))
    assert not verify("stripe", STRIPE_SECRET, stripe, STRIPE_BODY, SLACK_AT)


def test_hmac_recipe_uses_the_configured_header_and_prefix():
    body = b"payload"
    sig = _hex("k", body)
    at = datetime.now(UTC)
    assert verify(
        "hmac", "k", {"x-sig": "sha256=" + sig}, body, at, header="X-Sig", prefix="sha256="
    )
    assert verify("hmac", "k", {"x-sig": sig}, body, at, header="X-Sig")
    assert not verify("hmac", "k", {"x-sig": sig}, body, at, header="X-Sig", prefix="sha256=")
    assert not verify("hmac", "k", {"x-other": sig}, body, at, header="X-Sig")
    assert not verify("hmac", "k", {"x-sig": sig}, body, at)  # no header configured


def test_a_body_changed_by_one_byte_fails_every_recipe():
    ts = "1700000000"
    at = datetime.fromtimestamp(int(ts), UTC)
    stripe = _stripe_header(ts, _stripe_sig(ts))
    assert not verify("stripe", STRIPE_SECRET, stripe, STRIPE_BODY + b" ", at)
    gh = {"x-hub-signature-256": GITHUB_SIG}
    assert not verify("github", GITHUB_SECRET, gh, b"Hello, World?", at)
    assert not verify("slack", SLACK_SECRET, _slack_headers(), SLACK_BODY[:-1] + b"d", SLACK_AT)
    sig = _hex("k", b"payload")
    assert not verify("hmac", "k", {"x-sig": sig}, b"paylaod", at, header="x-sig")


@pytest.mark.parametrize(
    "sender,headers",
    [
        ("github", {}),
        ("github", {"x-hub-signature-256": "sha256=é" + "0" * 63}),
        ("stripe", {"stripe-signature": "garbage"}),
        ("stripe", {"stripe-signature": "t=1700000000,v1=é"}),
        ("slack", {"x-slack-request-timestamp": SLACK_TS}),
        ("slack", {"x-slack-request-timestamp": SLACK_TS, "x-slack-signature": "v0=ü"}),
        ("hmac", {"x-sig": "ß"}),
        ("bogus", {}),
    ],
)
def test_missing_malformed_or_non_ascii_signature_headers_are_rejected_not_raised(sender, headers):
    at = datetime.fromtimestamp(1700000000, UTC)
    assert verify(sender, "k", headers, b"x", at, header="x-sig") is False


def test_verify_uses_compare_digest(monkeypatch):
    calls: list[tuple[str, str]] = []
    real = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(hooks_mod.hmac, "compare_digest", spy)
    assert verify(
        "github",
        GITHUB_SECRET,
        {"x-hub-signature-256": GITHUB_SIG},
        b"Hello, World!",
        datetime.now(UTC),
    )
    assert calls == [(GITHUB_SIG, GITHUB_SIG)]


def test_parse_received_at_defaults_to_now_and_refuses_naive_garbage_stale_or_future():
    now = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    assert parse_received_at(None, now) == now
    assert parse_received_at("2026-10-10T11:00:00Z", now) == now - timedelta(hours=1)
    assert parse_received_at("2026-10-09T12:00:00+00:00", now) is not None  # 24 h old
    assert parse_received_at("2026-10-09T10:59:59Z", now) is None  # > 25 h old
    assert parse_received_at("2026-10-10T12:05:01Z", now) is None  # > 300 s ahead
    assert parse_received_at("2026-10-10T12:04:59Z", now) is not None
    assert parse_received_at("2026-10-10T11:00:00", now) is None  # naive
    assert parse_received_at("yesterday", now) is None


# --- gate ---

HOOK_ID = "abcdefghijklmnopqrst"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class _Queries:
    def __init__(self, hook: Hook | None, project_id: str | None = "p1") -> None:
        self.hook = hook
        self.project_id = project_id
        self.reads: list[tuple[str, str]] = []
        self.audits: list[dict] = []

    async def get_live_hook(self, hook_id, service_name):
        self.reads.append((hook_id, service_name))
        if self.hook is None or (hook_id, service_name) != (
            self.hook.hook_id,
            self.hook.service_name,
        ):
            return None
        return self.hook, self.project_id

    async def insert_audit_log(self, **kw):
        self.audits.append(kw)


def _hook(sender: str = "github", **kw) -> Hook:
    return Hook(
        hook_id=HOOK_ID,
        service_name="shop",
        name="gh",
        sender=sender,
        path="/webhooks/github",
        secret_ref=None if sender == "none" else default_secret_ref("gh"),
        created_at="2026-10-10 12:00:00",
        **kw,
    )


class _Resolver:
    def __init__(self, value: str | None = GITHUB_SECRET) -> None:
        self.value = value
        self.calls: list[tuple] = []
        self.threads: list[threading.Thread] = []

    def __call__(self, service, project_id, ref):
        self.calls.append((service, project_id, ref))
        self.threads.append(threading.current_thread())
        return self.value


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _gate(queries, resolver=None, mono=None) -> HookGate:
    return HookGate(
        queries,
        resolver or _Resolver(),
        clock=lambda: NOW,
        monotonic=mono or _Clock(),
    )


GOOD = [("X-Hub-Signature-256", GITHUB_SIG), ("x-nerdit-hook-received-at", "2026-10-10T11:59:00Z")]
BODY = b"Hello, World!"


async def test_gate_unknown_revoked_or_other_app_hook_is_404_without_secret_read():
    resolver = _Resolver()
    q = _Queries(_hook())
    gate = _gate(q, resolver)
    assert await gate.admit("b" * 20, "shop", GOOD, BODY) == 404
    assert await gate.admit(HOOK_ID, "other", GOOD, BODY) == 404
    assert await _gate(_Queries(None), resolver).admit(HOOK_ID, "shop", GOOD, BODY) == 404
    assert resolver.calls == []
    assert q.audits == []


@pytest.mark.parametrize("bad", ["short", "A" * 20, "abcdefghijklmnopqrs1", "a" * 21, ""])
async def test_gate_malformed_hook_id_is_404_without_a_database_read(bad):
    q = _Queries(_hook())
    assert await _gate(q).admit(bad, "shop", GOOD, BODY) == 404
    assert q.reads == []


async def test_gate_resolves_the_secret_off_the_event_loop():
    resolver = _Resolver()
    result = await _gate(_Queries(_hook()), resolver).admit(HOOK_ID, "shop", GOOD, BODY)
    assert isinstance(result, HookAdmission)
    assert resolver.calls == [("shop", "p1", "${secrets.NERDIT_HOOK_GH_SECRET}")]
    assert resolver.threads[0] is not threading.main_thread()
    assert result.name == "gh" and result.path == "/webhooks/github"
    assert result.received_at == "2026-10-10T11:59:00Z"  # the edge's header, not now


async def test_gate_bad_signature_is_401_and_audits_hook_rejected_without_body_or_id():
    q = _Queries(_hook())
    bad = [("x-hub-signature-256", "sha256=" + "0" * 64)]
    assert await _gate(q).admit(HOOK_ID, "shop", bad, BODY) == 401
    [row] = q.audits
    assert row["action"] == "hook.rejected"
    assert row["result"] == "denied"
    assert row["target_id"] == "shop"
    assert row["params_redacted"] == (
        '{"hook": "gh", "reason": "signature", "sender": "github", "service": "shop"}'
    )
    dumped = repr(row)
    for leak in (HOOK_ID, "Hello", GITHUB_SECRET, "0" * 64):
        assert leak not in dumped


async def test_gate_honors_the_edge_received_at_within_its_age_bound():
    q = _Queries(_hook())
    queued = [*GOOD[:1], ("x-nerdit-hook-received-at", "2026-10-09T13:00:00Z")]  # 23 h ago
    result = await _gate(q).admit(HOOK_ID, "shop", queued, BODY)
    assert isinstance(result, HookAdmission)
    assert result.received_at == "2026-10-09T13:00:00Z"


@pytest.mark.parametrize("bad", ["2026-10-09T10:00:00Z", "2026-10-10T12:06:00Z", "yesterday"])
async def test_gate_refuses_a_stale_future_or_garbage_received_at_with_401(bad):
    q = _Queries(_hook())
    headers = [*GOOD[:1], ("x-nerdit-hook-received-at", bad)]
    assert await _gate(q).admit(HOOK_ID, "shop", headers, BODY) == 401
    assert q.audits[0]["params_redacted"].count('"reason": "received_at"') == 1


@pytest.mark.parametrize("value", [None, ""])
async def test_gate_unresolvable_or_empty_secret_is_503_and_audits_secret_missing(value):
    q = _Queries(_hook())
    # An empty key would let anyone compute the HMAC: same 503 as a missing one.
    forged = [("x-hub-signature-256", "sha256=" + hmac.new(b"", BODY, hashlib.sha256).hexdigest())]
    assert await _gate(q, _Resolver(value)).admit(HOOK_ID, "shop", forged, BODY) == 503
    assert '"reason": "secret_missing"' in q.audits[0]["params_redacted"]


async def test_gate_rejected_audit_is_throttled_per_hook():
    q = _Queries(_hook())
    mono = _Clock()
    gate = _gate(q, mono=mono)
    bad = [("x-hub-signature-256", "sha256=" + "0" * 64)]
    for _ in range(3):
        assert await gate.admit(HOOK_ID, "shop", bad, BODY) == 401
    assert len(q.audits) == 1
    mono.t = 61
    assert await gate.admit(HOOK_ID, "shop", bad, BODY) == 401
    assert len(q.audits) == 2


async def test_gate_audit_failure_is_swallowed_without_traceback(caplog):
    q = _Queries(_hook())

    async def boom(**kw):
        raise RuntimeError("row " + HOOK_ID)

    q.insert_audit_log = boom  # type: ignore[method-assign]
    bad = [("x-hub-signature-256", "sha256=" + "0" * 64)]
    assert await _gate(q).admit(HOOK_ID, "shop", bad, BODY) == 401
    assert HOOK_ID not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


async def test_gate_none_sender_admits_without_a_secret():
    resolver = _Resolver()
    result = await _gate(_Queries(_hook("none")), resolver).admit(HOOK_ID, "shop", [], b"")
    assert isinstance(result, HookAdmission)
    assert resolver.calls == []
    assert result.received_at == "2026-10-10T12:00:00Z"  # missing header -> now


@pytest.mark.parametrize(
    "given,kept",
    [("evt_123-ABC", True), ("x" * 64, True), ("x" * 65, False), ("a b", False), ("", False)],
)
async def test_gate_keeps_a_sane_cloud_delivery_id_and_mints_one_otherwise(given, kept):
    headers = [("x-nerdit-hook-delivery", given)]
    result = await _gate(_Queries(_hook("none"))).admit(HOOK_ID, "shop", headers, b"")
    assert isinstance(result, HookAdmission)
    if kept:
        assert result.delivery == given
    else:
        assert result.delivery != given and len(result.delivery) == 16


async def test_gate_uses_the_first_header_value():
    headers = [GOOD[0], ("x-hub-signature-256", "sha256=" + "0" * 64), GOOD[1]]
    result = await _gate(_Queries(_hook())).admit(HOOK_ID, "shop", headers, BODY)
    assert isinstance(result, HookAdmission)
