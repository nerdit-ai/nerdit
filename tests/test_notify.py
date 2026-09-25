"""Tests for the outbound webhook dispatcher (P24c / WP8).

Every request goes through an ``httpx.MockTransport`` — no test touches the
network. Drains are driven by calling :meth:`WebhookDispatcher._drain_target`
directly, with the poll loop parked:
``_loop`` is a sleep loop whose only exit is cancellation, so running it would
buy nothing but flakiness. The two lifecycle tests at the bottom are the
exception — they are *about* the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import re
from pathlib import Path

import httpx
import pytest

import nerdit.core.notify as notify_mod
from nerdit.config.settings import NotificationsSettings, NotificationTarget
from nerdit.core.notify import WebhookDispatcher, _signature
from nerdit.core.secrets import SecretManager

TARGET_URL = "https://hook.example/nerdit"

# --- helpers ------------------------------------------------------------------


class _Recorder:
    """Records every request the transport sees; returns a fixed response."""

    def __init__(self, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}
        self.requests: list[httpx.Request] = []

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, headers=self.headers)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handler)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


class _PoisonStream(httpx.AsyncByteStream):
    """A response body that explodes if anyone iterates it; closing is fine."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):  # type: ignore[override]
        raise AssertionError("body was read")
        yield b""  # pragma: no cover — unreachable, keeps this an async generator

    async def aclose(self) -> None:
        self.closed = True


def _dispatcher(
    queries,
    tmp_path: Path,
    transport: httpx.AsyncBaseTransport,
    *,
    retry_max: int = 5,
    batch_max: int = 50,
    secrets: SecretManager | None = None,
) -> WebhookDispatcher:
    return WebhookDispatcher(
        queries,
        NotificationsSettings(enabled=True, retry_max=retry_max, batch_max=batch_max),
        secrets or SecretManager(tmp_path / "secrets"),
        version="0.0.test",
        instance_id="default",
        transport=transport,
    )


async def _park_loop(dispatcher: WebhookDispatcher) -> None:
    """Open the client but cancel the poll loop — drains are driven by hand."""
    await dispatcher.start()
    task = dispatcher._task
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        dispatcher._task = None


async def _drain(dispatcher: WebhookDispatcher, target: NotificationTarget) -> None:
    """One full drain pass against a parked dispatcher, client closed after."""
    await _park_loop(dispatcher)
    try:
        await dispatcher._drain_target(target)
    finally:
        await dispatcher.stop()


async def _seed_events(queries, count: int, *, type_: str = "service.healthy") -> list[int]:
    return [
        await queries.insert_event(type=type_, service_name="svc-a", kind="service")
        for _ in range(count)
    ]


def _target(**kwargs) -> NotificationTarget:
    return NotificationTarget(url=TARGET_URL, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    """Replace the retry backoff sleep with a recorder (never a real wait).

    Safe only because :func:`_drain` parks the poll loop — that loop's own
    ``asyncio.sleep`` would otherwise spin.
    """
    delays: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(notify_mod.asyncio, "sleep", _fake_sleep)
    return delays


# --- cursor semantics ---------------------------------------------------------


@pytest.mark.asyncio
async def test_2xx_delivers_the_batch_and_advances_the_cursor(queries, tmp_path):
    ids = await _seed_events(queries, 3)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target)

    assert len(recorder.requests) == 1
    body = recorder.bodies[0]
    assert body["event_ids"] == ids
    assert body["cursor"] == ids[-1]
    assert body["daemon"] == {"instance_id": "default", "version": "0.0.test"}
    assert [e["id"] for e in body["events"]] == ids
    assert recorder.requests[0].headers["X-Nerdit-Event-Count"] == "3"
    assert recorder.requests[0].headers["X-Nerdit-Delivery"]
    assert await queries.get_notification_cursor(tid) == ids[-1]


@pytest.mark.asyncio
async def test_new_target_seeds_at_the_feed_head_and_delivers_nothing(queries, tmp_path):
    """A newly configured target is NOT owed the backlog (D-BP-12)."""
    await _seed_events(queries, 4)

    target = _target()
    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target)

    assert recorder.requests == []
    assert await queries.get_notification_cursor(target.cursor_id()) == (
        await queries.last_event_id()
    )


@pytest.mark.asyncio
async def test_one_cursor_write_per_batch(queries, tmp_path, monkeypatch):
    """The cursor is written ONCE per delivered batch, never per event."""
    await _seed_events(queries, 3)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    writes: list[tuple[str, int]] = []
    original = queries.set_notification_cursor

    async def _counting(target_id: str, last_id: int) -> None:
        writes.append((target_id, last_id))
        await original(target_id, last_id)

    monkeypatch.setattr(queries, "set_notification_cursor", _counting)

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target)

    assert len(writes) == 1


@pytest.mark.asyncio
async def test_multiple_batches_drain_in_one_pass(queries, tmp_path):
    ids = await _seed_events(queries, 5)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport, batch_max=2), target)

    assert [len(b["event_ids"]) for b in recorder.bodies] == [2, 2, 1]
    assert await queries.get_notification_cursor(tid) == ids[-1]


# --- failure handling ---------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_exhaustion_advances_and_writes_one_audit_row(queries, tmp_path, no_sleep):
    ids = await _seed_events(queries, 2)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(500)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport, retry_max=2), target)

    assert len(recorder.requests) == 2
    # One backoff between the two attempts, jittered inside [base, base * 1.25).
    assert len(no_sleep) == 1
    assert 1.0 <= no_sleep[0] < 1.25
    assert await queries.get_notification_cursor(tid) == ids[-1]

    rows, _ = await queries.list_audit_log(limit=50)
    failures = [r for r in rows if r.action == "notification.delivery_failed"]
    assert len(failures) == 1
    # ``params_redacted`` is already parsed by ``_row_to_audit``.
    params = failures[0].params_redacted
    assert params == {
        "attempts": 2,
        "first_id": ids[0],
        "host": "hook.example",
        "last_id": ids[-1],
        "status_code": 500,
        "target_id": tid,
    }
    # The URL (which may carry a topic token) is nowhere in the row.
    assert TARGET_URL not in json.dumps(failures[0].model_dump(mode="json"))
    assert failures[0].principal_id == "system"


@pytest.mark.asyncio
async def test_exhaustion_stops_the_pass_after_one_batch(queries, tmp_path, no_sleep):
    """One failed batch per drain pass — bounded work against a dead target."""
    ids = await _seed_events(queries, 5)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(503)
    await _drain(
        _dispatcher(queries, tmp_path, recorder.transport, retry_max=1, batch_max=2), target
    )

    assert len(recorder.requests) == 1
    assert await queries.get_notification_cursor(tid) == ids[1]


@pytest.mark.asyncio
async def test_redirects_are_never_followed(queries, tmp_path, no_sleep):
    """A 302 would be an allow-set bypass — it is a plain delivery failure."""
    await _seed_events(queries, 1)
    target = _target()
    await queries.set_notification_cursor(target.cursor_id(), 0)

    recorder = _Recorder(302, headers={"Location": "http://169.254.169.254/latest"})
    await _drain(_dispatcher(queries, tmp_path, recorder.transport, retry_max=1), target)

    assert len(recorder.requests) == 1
    assert recorder.requests[0].url == httpx.URL(TARGET_URL)
    rows, _ = await queries.list_audit_log(limit=50)
    assert [r.action for r in rows if r.action == "notification.delivery_failed"] == [
        "notification.delivery_failed"
    ]


@pytest.mark.asyncio
async def test_response_body_is_never_read(queries, tmp_path):
    """Only the status code is consumed; the stream is closed, never iterated."""
    ids = await _seed_events(queries, 1)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    stream = _PoisonStream()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    await _drain(_dispatcher(queries, tmp_path, httpx.MockTransport(_handler)), target)

    assert await queries.get_notification_cursor(tid) == ids[-1]
    assert stream.closed is True


@pytest.mark.asyncio
async def test_transport_error_is_a_delivery_failure(queries, tmp_path, no_sleep):
    ids = await _seed_events(queries, 1)
    target = _target()
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    await _drain(_dispatcher(queries, tmp_path, httpx.MockTransport(_boom), retry_max=1), target)

    assert await queries.get_notification_cursor(tid) == ids[-1]
    rows, _ = await queries.list_audit_log(limit=50)
    failures = [r for r in rows if r.action == "notification.delivery_failed"]
    assert len(failures) == 1
    assert failures[0].params_redacted["status_code"] is None


# --- custody ------------------------------------------------------------------


def test_signature_fixed_vector():
    """The wire format is ``sha256=<hex>`` over the exact body bytes."""
    assert _signature("test-key", b'{"x":1}') == (
        "sha256=1fafc942f2943970f9db85c3f65cdf8c02dff5c5a4726b39e8cb9b28611fcdf1"
    )


@pytest.mark.asyncio
async def test_hmac_signature_verifies_end_to_end(queries, tmp_path):
    await _seed_events(queries, 2)
    target = _target(secret_ref="${secrets.shared.HOOK_HMAC}")
    await queries.set_notification_cursor(target.cursor_id(), 0)

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"HOOK_HMAC": "zqxjkw-ZQXJKW-hmac"})

    recorder = _Recorder(200)
    await _drain(
        _dispatcher(queries, tmp_path, recorder.transport, secrets=secrets),
        target,
    )

    request = recorder.requests[0]
    expected = hmac.new(b"zqxjkw-ZQXJKW-hmac", request.content, hashlib.sha256).hexdigest()
    assert request.headers["X-Nerdit-Signature"] == f"sha256={expected}"


@pytest.mark.asyncio
async def test_signature_v2_binds_timestamp_and_delivery(queries, tmp_path):
    """V2 signs `<ts>.<delivery>.<body>`; V1 stays the body-only HMAC."""
    await _seed_events(queries, 1)
    target = _target(secret_ref="${secrets.shared.HOOK_HMAC}")
    await queries.set_notification_cursor(target.cursor_id(), 0)
    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"HOOK_HMAC": "zqxjkw-ZQXJKW-hmac"})

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport, secrets=secrets), target)

    request = recorder.requests[0]
    key = b"zqxjkw-ZQXJKW-hmac"
    ts = request.headers["X-Nerdit-Timestamp"]
    assert ts.isdigit()
    delivery = request.headers["X-Nerdit-Delivery"]
    v1 = hmac.new(key, request.content, hashlib.sha256).hexdigest()
    v2 = hmac.new(key, f"{ts}.{delivery}.".encode() + request.content, hashlib.sha256).hexdigest()
    assert request.headers["X-Nerdit-Signature"] == f"sha256={v1}"
    assert request.headers["X-Nerdit-Signature-V2"] == f"sha256={v2}"


@pytest.mark.asyncio
async def test_auth_header_ref_is_sent_and_never_recorded(queries, tmp_path, caplog):
    sentinel = "Bearer zqxjkw-ZQXJKW-9"
    await _seed_events(queries, 1)
    target = _target(auth_header_ref="${secrets.shared.HOOK_TOKEN}")
    await queries.set_notification_cursor(target.cursor_id(), 0)

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"HOOK_TOKEN": sentinel})

    recorder = _Recorder(200)
    with caplog.at_level("DEBUG"):
        await _drain(
            _dispatcher(queries, tmp_path, recorder.transport, secrets=secrets),
            target,
        )

    assert recorder.requests[0].headers["Authorization"] == sentinel
    rows, _ = await queries.list_audit_log(limit=100)
    assert all(sentinel not in json.dumps(r.model_dump(mode="json")) for r in rows)
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_custom_auth_header_name_does_not_clobber_the_nerdit_headers(queries, tmp_path):
    await _seed_events(queries, 1)
    target = _target(
        secret_ref="${secrets.shared.HOOK_HMAC}",
        auth_header_ref="${secrets.shared.HOOK_TOKEN}",
        auth_header_name="X-Api-Key",
    )
    await queries.set_notification_cursor(target.cursor_id(), 0)

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"HOOK_TOKEN": "zqxjkw-key", "HOOK_HMAC": "zqxjkw-hmac"})

    recorder = _Recorder(200)
    await _drain(
        _dispatcher(queries, tmp_path, recorder.transport, secrets=secrets),
        target,
    )

    headers = recorder.requests[0].headers
    assert headers["X-Api-Key"] == "zqxjkw-key"
    assert headers["X-Nerdit-Signature"].startswith("sha256=")
    assert headers["X-Nerdit-Delivery"]
    assert headers["X-Nerdit-Event-Count"] == "1"


@pytest.mark.asyncio
async def test_missing_shared_key_is_a_per_target_delivery_failure(queries, tmp_path, no_sleep):
    """An unresolvable ref never reaches the wire and never raises."""
    ids = await _seed_events(queries, 1)
    target = _target(auth_header_ref="${secrets.shared.ABSENT}")
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(200)
    await _drain(
        _dispatcher(queries, tmp_path, recorder.transport, retry_max=2),
        target,
    )

    assert recorder.requests == []
    assert await queries.get_notification_cursor(tid) == ids[-1]
    rows, _ = await queries.list_audit_log(limit=50)
    failures = [r for r in rows if r.action == "notification.delivery_failed"]
    assert len(failures) == 1
    params = failures[0].params_redacted
    assert params["status_code"] is None
    assert params["host"] == "hook.example"


@pytest.mark.asyncio
async def test_unscoped_ref_resolves_to_nothing(queries, tmp_path, no_sleep):
    """A target ref is shared-scope-ONLY, even when it names a stored key.

    Validation rejects an unscoped ``${secrets.KEY}`` at config time, so this is
    the config-edited-underneath-us path (``model_construct`` bypasses the
    validators the same way). The dispatcher never consults a per-service scope,
    so an unscoped ref has nowhere to resolve — a same-named shared key does NOT
    satisfy it, and the target fails delivery like any missing key.
    """
    ids = await _seed_events(queries, 1)
    target = NotificationTarget.model_construct(
        url=TARGET_URL, auth_header_ref="${secrets.HOOK_TOKEN}"
    )
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"HOOK_TOKEN": "zqxjkw-secret"})

    recorder = _Recorder(200)
    await _drain(
        _dispatcher(queries, tmp_path, recorder.transport, retry_max=2, secrets=secrets),
        target,
    )

    assert recorder.requests == []
    assert await queries.get_notification_cursor(tid) == ids[-1]
    rows, _ = await queries.list_audit_log(limit=50)
    failures = [r for r in rows if r.action == "notification.delivery_failed"]
    assert len(failures) == 1
    assert "zqxjkw-secret" not in json.dumps(failures[0].model_dump(mode="json"))


@pytest.mark.asyncio
async def test_unencodable_auth_value_is_a_bounded_delivery_failure(
    queries, tmp_path, no_sleep, caplog
):
    """Building the request is part of the attempt, not a precondition of it.

    A shared secret holding a non-ASCII character makes ``build_request`` raise
    ``UnicodeEncodeError`` — whose ``args`` carry the WHOLE value. Outside the
    protected path that escapes ``_drain_target``, so the batch gets no audit
    row and no cursor advance (the feed wedges on it, every pass, forever) and
    the exception lands in ``_loop``'s ``logger.exception``. The audit row and
    the advanced cursor below are the proof it did not escape."""
    sentinel = "Bearer zqxjkw-ZQXJKW-café"
    ids = await _seed_events(queries, 1)
    target = _target(auth_header_ref="${secrets.shared.HOOK_TOKEN}")
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"HOOK_TOKEN": sentinel})

    recorder = _Recorder(200)
    with caplog.at_level("DEBUG"):
        await _drain(
            _dispatcher(queries, tmp_path, recorder.transport, retry_max=2, secrets=secrets),
            target,
        )

    assert recorder.requests == [], "nothing reached the wire"
    assert await queries.get_notification_cursor(tid) == ids[-1]
    rows, _ = await queries.list_audit_log(limit=50)
    failures = [r for r in rows if r.action == "notification.delivery_failed"]
    assert len(failures) == 1
    assert failures[0].params_redacted["status_code"] is None
    assert failures[0].params_redacted["host"] == "hook.example"
    assert sentinel not in json.dumps(failures[0].model_dump(mode="json"))
    # No log record carries the value — not the string, not an exception repr
    # (``UnicodeEncodeError.args`` embeds the offending object in full).
    assert sentinel not in caplog.text
    assert "zqxjkw-ZQXJKW" not in caplog.text


# --- filters + loop prevention ------------------------------------------------


@pytest.mark.asyncio
async def test_unmatched_batch_advances_without_a_post(queries, tmp_path):
    ids = await _seed_events(queries, 3, type_="service.healthy")
    target = _target(events=["service.failed"])
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target)

    assert recorder.requests == []
    assert await queries.get_notification_cursor(tid) == ids[-1]


@pytest.mark.asyncio
async def test_mixed_batch_posts_matches_but_advances_to_the_read_head(queries, tmp_path):
    """The cursor covers the whole READ batch, never the matched subset."""
    await _seed_events(queries, 1, type_="service.healthy")
    matched_id = await queries.insert_event(type="service.failed", service_name="svc-a")
    tail_id = await queries.insert_event(type="service.healthy", service_name="svc-a")
    target = _target(events=["service.failed"])
    tid = target.cursor_id()
    await queries.set_notification_cursor(tid, 0)

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target)

    assert len(recorder.requests) == 1
    assert recorder.bodies[0]["event_ids"] == [matched_id]
    assert await queries.get_notification_cursor(tid) == tail_id


@pytest.mark.asyncio
async def test_same_url_different_filters_get_independent_cursors_and_both_deliver(
    queries, tmp_path
):
    """One URL, two filters: two targets, two cursors, two deliveries.

    Keying the cursor on the URL alone would let whichever target drained first
    advance the shared row past the batch, and the second would silently
    deliver nothing.
    """
    ids = await _seed_events(queries, 2, type_="service.healthy")
    target_a = _target()
    target_b = _target(events=["service.healthy"])
    assert target_a.cursor_id() != target_b.cursor_id()
    await queries.set_notification_cursor(target_a.cursor_id(), 0)
    await queries.set_notification_cursor(target_b.cursor_id(), 0)

    recorder = _Recorder(200)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target_a)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport), target_b)

    assert [b["event_ids"] for b in recorder.bodies] == [ids, ids]
    assert await queries.get_notification_cursor(target_a.cursor_id()) == ids[-1]
    assert await queries.get_notification_cursor(target_b.cursor_id()) == ids[-1]


@pytest.mark.asyncio
async def test_dispatcher_never_emits_an_event(queries, tmp_path, no_sleep):
    """D-P24-6 loop prevention: a failed delivery writes audit, never a feed row."""
    await _seed_events(queries, 2)
    target = _target()
    await queries.set_notification_cursor(target.cursor_id(), 0)
    before = await queries.last_event_id()

    recorder = _Recorder(500)
    await _drain(_dispatcher(queries, tmp_path, recorder.transport, retry_max=2), target)

    assert await queries.last_event_id() == before


def test_notify_module_does_not_import_the_event_recorder():
    """The dispatcher must not be able to feed itself (D-P24-6)."""
    source = Path(notify_mod.__file__).read_text(encoding="utf-8")
    assert re.search(r"^\s*(from|import)\s+\S*eventlog", source, re.MULTILINE) is None


# --- lifecycle ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_cancels_the_loop(queries, tmp_path):
    dispatcher = _dispatcher(queries, tmp_path, _Recorder(200).transport)
    await dispatcher.start()
    assert dispatcher._task is not None
    await dispatcher.stop()
    await dispatcher.stop()
    assert dispatcher._task is None
    assert dispatcher._client is None


@pytest.mark.asyncio
async def test_loop_survives_a_failing_target(queries, tmp_path, monkeypatch):
    """One bad target must never take the shared drain loop down."""
    calls: list[str] = []

    async def _boom(target) -> None:
        calls.append(target.url)
        raise RuntimeError("nope")

    dispatcher = _dispatcher(queries, tmp_path, _Recorder(200).transport)
    dispatcher._settings = NotificationsSettings(
        enabled=True, targets=[_target()], poll_interval_s=300.0
    )
    monkeypatch.setattr(dispatcher, "_drain_target", _boom)

    await dispatcher.start()
    # Yield until the loop has run its single pass and parked on the sleep.
    for _ in range(5):
        await asyncio.sleep(0)
    assert dispatcher._task is not None
    assert not dispatcher._task.done()
    await dispatcher.stop()

    assert calls == [TARGET_URL]


@pytest.mark.asyncio
async def test_a_hanging_target_does_not_delay_another(queries, tmp_path):
    """Targets drain concurrently: a hung endpoint never blocks a healthy one."""
    never = asyncio.Event()
    fast = _Recorder(200)

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "hang.example":
            await never.wait()
        return fast._handler(request)

    hung = NotificationTarget(url="https://hang.example/x")
    ok = _target()
    for t in (hung, ok):
        await queries.set_notification_cursor(t.cursor_id(), 0)
    await _seed_events(queries, 1)

    dispatcher = _dispatcher(queries, tmp_path, httpx.MockTransport(_handler))
    dispatcher._settings = NotificationsSettings(
        enabled=True, targets=[hung, ok], poll_interval_s=300.0, timeout_s=30.0
    )
    await dispatcher.start()
    try:
        for _ in range(100):
            if fast.requests:
                break
            await asyncio.sleep(0.01)
        assert [r.url.host for r in fast.requests] == ["hook.example"]
    finally:
        await dispatcher.stop()


def test_cursor_id_is_canonical_and_url_free():
    """The cursor key covers the WHOLE delivery identity, and leaks no URL."""
    tid = _target().cursor_id()
    assert tid == _target().cursor_id()
    assert re.fullmatch(r"[0-9a-f]{16}", tid)
    assert TARGET_URL not in tid

    assert NotificationTarget(url="https://other.example/x").cursor_id() != tid
    assert _target(events=["service.healthy"]).cursor_id() != tid
    assert _target(secret_ref="${secrets.shared.HOOK_HMAC}").cursor_id() != tid
    assert _target(auth_header_ref="${secrets.shared.HOOK_TOKEN}").cursor_id() != tid
    assert _target(auth_header_name="X-Api-Key").cursor_id() != tid

    # The filter is a set, not a sequence: neither order nor repetition may
    # re-key the cursor — a repeated entry delivers exactly the same events,
    # and a distinct identity would let a disguised duplicate target past the
    # duplicate guard.
    assert (
        _target(events=["service.healthy", "service.deploy_failed"]).cursor_id()
        == _target(events=["service.deploy_failed", "service.healthy"]).cursor_id()
    )
    assert (
        _target(events=["service.healthy", "service.healthy"]).cursor_id()
        == _target(events=["service.healthy"]).cursor_id()
    )
