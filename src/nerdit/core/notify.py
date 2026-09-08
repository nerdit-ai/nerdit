"""Deliver durable events to webhook targets with bounded retries.

Read the events table, never the lossy bus. Per-target durable cursors advance
on 2xx or retry exhaustion so dead targets cannot wedge delivery. Consumers
recover missed events with `GET /events?since_id=<cursor>`. Changing URL, filter,
auth references, or header name gives a target a new cursor at the feed head.
Delivery failures create audit rows, never events that could feed themselves.

Resolve shared HMAC/auth secrets at send time and keep them request-local;
never persist, log, audit, or place values in argv. Read response status only,
never bodies, and reject redirects to preserve the target allowlist.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import random
import uuid
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from nerdit.config.settings import NotificationsSettings, NotificationTarget
from nerdit.core.bindings.secretref import walk_secret_ref
from nerdit.core.secrets import SHARED_SCOPE, SecretDecryptError, SecretManager
from nerdit.db.rows import Event

if TYPE_CHECKING:
    from nerdit.core.events import EventBus
    from nerdit.db.queries import Queries

logger = logging.getLogger(__name__)

#: Per-attempt backoff base, in seconds (D-P24-6: 1/2/4/8/16 s + jitter). The
#: last entry is reused for any attempt beyond the fifth.
_RETRY_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)


def _signature(key: str, raw: bytes) -> str:
    """`sha256=<hex>` HMAC over the EXACT bytes that go on the wire."""
    return "sha256=" + hmac.new(key.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def _build_payload(batch: list[Event], *, instance_id: str, version: str) -> bytes:
    """Serialize one delivery body; the returned bytes are what is signed and sent.

    `cursor` is the consumer's reconcile handle: a receiver that misses a
    delivery (or distrusts the body, which it should) replays everything after
    it with `GET /events?since_id=<cursor>`. The body is never re-serialized
    after this point — the HMAC is computed over exactly these bytes.
    """
    payload = {
        "cursor": batch[-1].id,
        "event_ids": [event.id for event in batch],
        "events": [event.model_dump(mode="json") for event in batch],
        "daemon": {"instance_id": instance_id, "version": version},
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class WebhookDispatcher:
    """Polls the durable feed and POSTs new events to every configured target.

    Lifecycle shape: built in the daemon lifespan (only when
    `[notifications].enabled` and at least one
    target is configured), `start` spawns the loop, `stop` cancels
    it and closes the client. Each target drains independently; a target that
    raises can never take the loop down.
    """

    def __init__(
        self,
        queries: Queries,
        settings: NotificationsSettings,
        secrets: SecretManager,
        *,
        bus: EventBus | None = None,
        version: str,
        instance_id: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._queries = queries
        self._settings = settings
        self._secrets = secrets
        # Reserved for the wake-up nudge (D-P24-6: the bus "may serve" as one).
        # v1 polls only — the interval poll is the fallback AND the guarantee.
        self._bus = bus
        self._version = version
        self._instance_id = instance_id
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Open the HTTP client and spawn the drain loop."""
        self._client = httpx.AsyncClient(
            # A 302 to an address outside the allow-set would defeat the whole
            # D-P24-7 validation, so redirects are refused at the client.
            follow_redirects=False,
            timeout=self._settings.timeout_s,
            transport=self._transport,
        )
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the loop and close the client; idempotent, never raises out."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

    # -- drain -----------------------------------------------------------------

    async def _loop(self) -> None:
        """Drain every target, then sleep. Cancellation is the only exit."""
        while True:
            for target in self._settings.targets:
                try:
                    await self._drain_target(target)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad target never kills the loop
                    logger.exception("Webhook drain failed for target %s", target.cursor_id())
            await asyncio.sleep(self._settings.poll_interval_s)

    async def _drain_target(self, target: NotificationTarget) -> None:
        """Read forward from the target's cursor and deliver, batch by batch.

        A first encounter **seeds** at the current feed head and delivers
        nothing: a newly configured target is not owed the backlog. After that,
        each pass reads at most `batch_max` rows at a time and stops on the
        first empty read — or after one exhausted batch, which bounds the work
        spent on a dead endpoint per pass.
        """
        tid = target.cursor_id()
        cursor = await self._queries.get_notification_cursor(tid)
        if cursor is None:
            await self._queries.set_notification_cursor(tid, await self._queries.last_event_id())
            return

        while True:
            batch, _ = await self._queries.list_events(
                since_id=cursor, limit=self._settings.batch_max
            )
            if not batch:
                return
            # The cursor covers the whole READ batch, never the matched subset:
            # a filtered-out event is delivered-by-omission, not pending.
            new_cursor = batch[-1].id
            matched = [e for e in batch if not target.events or e.type in target.events]
            if not matched:
                await self._queries.set_notification_cursor(tid, new_cursor)
                cursor = new_cursor
                continue

            status: int | None = None
            attempts = 0
            for attempt in range(1, self._settings.retry_max + 1):
                attempts = attempt
                status = await self._deliver(target, matched)
                if status is not None and 200 <= status < 300:
                    break
                if attempt < self._settings.retry_max:
                    base = _RETRY_BACKOFF_S[min(attempt - 1, len(_RETRY_BACKOFF_S) - 1)]
                    await asyncio.sleep(base + random.uniform(0, base / 4))
            else:
                await self._audit_failure(target, tid, matched, attempts=attempts, status=status)
                await self._queries.set_notification_cursor(tid, new_cursor)
                return

            await self._queries.set_notification_cursor(tid, new_cursor)
            cursor = new_cursor

    async def _audit_failure(
        self,
        target: NotificationTarget,
        tid: str,
        matched: list[Event],
        *,
        attempts: int,
        status: int | None,
    ) -> None:
        """One `notification.delivery_failed` audit row — never an event.

        Best-effort (the `CutoverManager._audit` posture): failing to record a
        failed delivery must not also fail the drain. The URL is reduced to its
        **host** — the path and query may carry a topic token.
        """
        try:
            await self._queries.insert_audit_log(
                action="notification.delivery_failed",
                result="error",
                principal_id="system",
                principal_role="system",
                target_type="notification",
                target_id=tid,
                params_redacted=json.dumps(
                    {
                        "target_id": tid,
                        "host": urlsplit(target.url).hostname,
                        "first_id": matched[0].id,
                        "last_id": matched[-1].id,
                        "attempts": attempts,
                        "status_code": status,
                    },
                    sort_keys=True,
                ),
            )
        except Exception:
            logger.warning("Failed to audit webhook delivery failure for %s", tid, exc_info=True)

    async def _deliver(self, target: NotificationTarget, batch: list[Event]) -> int | None:
        """POST one batch; returns the HTTP status, or `None` when none was reached.

        `None` covers both a transport error and a send-time custody failure
        (an unreadable shared scope, or a ref naming a key that is not there) —
        a missing key is a per-target *delivery* failure, never a daemon error.
        """
        if self._client is None:  # not started (or already stopped)
            return None
        raw = _build_payload(batch, instance_id=self._instance_id, version=self._version)
        headers = {
            "Content-Type": "application/json",
            "X-Nerdit-Delivery": str(uuid.uuid4()),
            "X-Nerdit-Event-Count": str(len(batch)),
        }
        if target.secret_ref is not None or target.auth_header_ref is not None:
            try:
                shared = self._secrets.load(SHARED_SCOPE)
            except SecretDecryptError:
                logger.warning("Webhook target %s: shared secrets unreadable", target.cursor_id())
                return None
            if target.secret_ref is not None:
                key = self._shared_value(shared, target.secret_ref)
                if key is None:
                    return None
                headers["X-Nerdit-Signature"] = _signature(key, raw)
            if target.auth_header_ref is not None:
                value = self._shared_value(shared, target.auth_header_ref)
                if value is None:
                    return None
                # Stored verbatim, including any scheme: the daemon never
                # concatenates one ("Bearer " is the operator's to write).
                headers[target.auth_header_name] = value

        # Building the request is part of the ATTEMPT, not a precondition of it:
        # header encoding happens here, so a resolved auth value that is not
        # ASCII raises `UnicodeEncodeError` — outside the protected block that
        # would escape to `_loop`'s `logger.exception` and print the value
        # (the exception's `args` carry the whole offending string). Caught
        # broadly, logged WITHOUT `exc_info` and without the exception itself,
        # it is one failed attempt like any other: retry exhaustion, cursor
        # advance, one value-free audit row.
        try:
            request = self._client.build_request("POST", target.url, content=raw, headers=headers)
        except Exception:  # noqa: BLE001 — a send-time custody failure, not a daemon error
            logger.warning(
                "Webhook target %s: request could not be built (bad header value?)",
                target.cursor_id(),
            )
            return None
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError:
            return None
        try:
            # The body is NEVER read: a receiver must not be able to
            # feed the daemon an unbounded response.
            return response.status_code
        finally:
            await response.aclose()

    @staticmethod
    def _shared_value(shared: dict[str, str], ref: str) -> str | None:
        """Resolve one `${secrets.shared.KEY}` ref against the shared scope.

        Shared-scope-ONLY by construction (WP9): the walk is handed no
        per-service loader, so a ref that does not name the shared scope has no
        scope left to read. Validation already guaranteed both the shared scope
        and the grammar, so an unscoped or unparseable ref here means the config
        was edited underneath us — treated exactly like a missing key. Never
        logs the value or the key.
        """
        return walk_secret_ref(ref, service_env=None, shared_env=lambda: shared).value


__all__ = ["WebhookDispatcher"]
