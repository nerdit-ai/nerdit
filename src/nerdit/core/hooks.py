"""Inbound webhooks: hook ids, env key names and signature verification.

Trust model:
    - The provider window (Stripe, Slack) runs against the edge's
      `X-Nerdit-Hook-Received-At`, so a delivery the edge queued and retried
      hours later still verifies. That header is trusted because the cloud
      edge strips every client copy of `X-Nerdit-Hook-*` and sets its own
      (deploy the cloud edge before this daemon ships: see exposure.md §8).
      `parse_received_at` bounds it (<= 25 h old, <= 5 min ahead) so the node
      enforces the 24 h promise itself. The mux drops these carriers from
      every other app stream, and every Caddy app route deletes them.
    - Every signature compare is constant-time (`hmac.compare_digest`).
    - The signing secret is resolved per verification, off the event loop,
      never cached and never logged. Hook ids, headers and bodies never reach
      a log line or an audit row.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from nerdit.core.link.hosted import hosted_host, hosted_label_fits

logger = logging.getLogger(__name__)

HOOK_ID_RE = re.compile(r"[a-z2-7]{20}")
#: Provider signature window (Stripe and Slack SDK default), against Received-At.
TOLERANCE_S = 300
#: W-I2 v6: the edge keeps a delivery at most 24 h; one hour of slack.
MAX_RECEIVED_AGE_S = 25 * 3600
HOOK_HEADER = "x-nerdit-hook"
DELIVERY_HEADER = "x-nerdit-hook-delivery"
RECEIVED_AT_HEADER = "x-nerdit-hook-received-at"

_TIMESTAMP_RE = re.compile(r"[0-9]{1,12}")  # not \d: int() also takes Unicode digits
_DELIVERY_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_REJECT_AUDIT_INTERVAL_S = 60.0


def new_hook_id() -> str:
    """A fresh 20-character base32 hook id (100 random bits)."""
    return base64.b32encode(secrets.token_bytes(13)).decode().lower()[:20]


def _env_stem(name: str) -> str:
    return "NERDIT_HOOK_" + name.upper().replace("-", "_")


def secret_key(name: str) -> str:
    """The S2 key holding a hook's signing secret: `NERDIT_HOOK_<NAME>_SECRET`."""
    return _env_stem(name) + "_SECRET"


def url_env_key(name: str) -> str:
    """The app env var carrying a hook's address: `NERDIT_HOOK_<NAME>_URL`."""
    return _env_stem(name) + "_URL"


def default_secret_ref(name: str) -> str:
    """The `${secrets.…}` reference a hook uses unless its nerdit.toml names one."""
    return "${secrets." + secret_key(name) + "}"


def hook_url(service: str, hook_id: str, slug: str | None, domain: str | None) -> str | None:
    """The hosted webhook address, or None when unlinked or the label is over 63 octets."""
    # ponytail: ignores an active <slug>.nerdit.app public address; the legacy
    # host resolves for the same job through the pinned alias (db/queries/shares.py).
    if not slug or not domain or not hosted_label_fits(service, slug):
        return None
    return f"https://{hosted_host(service, slug, domain)}/.nerdit/hooks/{hook_id}"


def parse_received_at(raw: str | None, now: datetime) -> datetime | None:
    """Parse the edge's Received-At; None means refuse (401).

    A missing header falls back to `now` (the stricter window). Naive,
    unparseable, older than `MAX_RECEIVED_AGE_S` or more than `TOLERANCE_S`
    in the future is refused, so the node enforces the 24 h promise itself.
    """
    if raw is None:
        return now
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    age = (now - parsed).total_seconds()
    if age > MAX_RECEIVED_AGE_S or age < -TOLERANCE_S:
        return None
    return parsed


def _hex(key: str, msg: bytes) -> str:
    return hmac.new(key.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _timestamp(raw: str | None, received_at: datetime) -> str | None:
    """The raw timestamp string when plain digits within the window, else None."""
    if raw is None or not _TIMESTAMP_RE.fullmatch(raw):
        return None
    if abs(int(raw) - received_at.timestamp()) > TOLERANCE_S:
        return None
    return raw


def _eq(given: str | None, expected: str) -> bool:
    return given is not None and hmac.compare_digest(given, expected)


def _verify_stripe(secret: str, headers: Mapping[str, str], body: bytes, at: datetime) -> bool:
    pairs = [item.strip().partition("=") for item in headers.get("stripe-signature", "").split(",")]
    ts = _timestamp(next((v for k, sep, v in pairs if sep and k == "t"), None), at)
    if ts is None:
        return False
    expected = _hex(secret, ts.encode() + b"." + body)
    # Check every v1 (no early exit) so the time does not depend on which matched.
    return any([_eq(v, expected) for k, sep, v in pairs if sep and k == "v1"])


def _verify_slack(secret: str, headers: Mapping[str, str], body: bytes, at: datetime) -> bool:
    ts = _timestamp(headers.get("x-slack-request-timestamp"), at)
    if ts is None:
        return False
    expected = "v0=" + _hex(secret, b"v0:" + ts.encode() + b":" + body)
    return _eq(headers.get("x-slack-signature"), expected)


def verify(  # noqa: PLR0913 - header/prefix are keyword-only recipe knobs
    sender: str,
    secret: str,
    headers: Mapping[str, str],
    body: bytes,
    received_at: datetime,
    *,
    header: str | None = None,
    prefix: str | None = None,
) -> bool:
    """Verify one delivery's signature over the raw body bytes; never raises.

    Args:
        headers: Lower-cased names, first value of each.
        received_at: The edge's receive time; timestamped recipes are checked
            against it, not the current time, so a redelivery still verifies.
        header, prefix: The generic `hmac` recipe's header and value prefix
            (lowercase hex HMAC-SHA256; base64 senders are out of V1).
    """
    try:
        if sender == "stripe":
            return _verify_stripe(secret, headers, body, received_at)
        if sender == "github":
            return _eq(headers.get("x-hub-signature-256"), "sha256=" + _hex(secret, body))
        if sender == "slack":
            return _verify_slack(secret, headers, body, received_at)
        if sender == "hmac" and header:
            return _eq(headers.get(header.lower()), (prefix or "") + _hex(secret, body))
        return sender == "none"
    except (TypeError, ValueError, UnicodeError):
        # compare_digest raises TypeError on a non-ASCII str: a refusal, not a crash.
        return False


@dataclass(frozen=True, slots=True)
class HookAdmission:
    """What the mux forwards to the app once a delivery is admitted."""

    name: str
    path: str
    delivery: str
    received_at: str  # RFC 3339, 'Z'


class HookGate:
    """Admit or refuse one hook stream: 404 unknown, 401 bad signature, 503 no secret."""

    def __init__(
        self,
        queries: Any,
        resolve_secret: Callable[[str, str | None, str], str | None],
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._queries = queries
        self._resolve_secret = resolve_secret
        self._clock = clock
        self._monotonic = monotonic
        # ponytail: per-process throttle; a counter column if an owner needs exact counts
        self._last_reject: dict[str, float] = {}

    async def admit(
        self,
        hook_id: str,
        service_name: str,
        headers: Sequence[tuple[str, str]],
        body: bytes,
    ) -> HookAdmission | int:
        """Return the admission, or the HTTP status to answer (404, 401 or 503)."""
        if not HOOK_ID_RE.fullmatch(hook_id):
            return 404
        found = await self._queries.get_live_hook(hook_id, service_name)
        if found is None:
            return 404
        hook, project_id = found
        lowered: dict[str, str] = {}
        for key, value in headers:
            lowered.setdefault(key.lower(), value)
        received_at = parse_received_at(lowered.get(RECEIVED_AT_HEADER), self._clock())
        if received_at is None:
            await self._reject(hook, "received_at")
            return 401
        if hook.sender != "none":
            # Off the loop: SecretManager I/O takes an RLock a key rotation or a
            # backup can hold for minutes; on the loop it would freeze the tunnel.
            secret = await asyncio.to_thread(
                self._resolve_secret, service_name, project_id, hook.secret_ref
            )
            if not secret:  # an empty key would make the HMAC public
                await self._reject(hook, "secret_missing")
                return 503
            ok = verify(
                hook.sender,
                secret,
                lowered,
                body,
                received_at,
                header=hook.header,
                prefix=hook.prefix,
            )
            if not ok:
                await self._reject(hook, "signature")
                return 401
        delivery = lowered.get(DELIVERY_HEADER) or ""
        if not _DELIVERY_RE.fullmatch(delivery):
            delivery = secrets.token_hex(8)
        stamp = received_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        return HookAdmission(hook.name, hook.path, delivery, stamp)

    async def _reject(self, hook: Any, reason: str) -> None:
        """Best-effort `hook.rejected` audit row, at most one per hook per minute."""
        now = self._monotonic()
        last = self._last_reject.get(hook.hook_id)
        if last is not None and now - last < _REJECT_AUDIT_INTERVAL_S:
            return
        self._last_reject[hook.hook_id] = now
        try:
            await self._queries.insert_audit_log(
                action="hook.rejected",
                result="denied",
                principal_id="system",
                principal_role="system",
                target_type="service",
                target_id=hook.service_name,
                params_redacted=json.dumps(
                    {
                        "service": hook.service_name,
                        "hook": hook.name,
                        "sender": hook.sender,
                        "reason": reason,
                    },
                    sort_keys=True,
                ),
            )
        except Exception:
            # No exc_info: the traceback could carry the row.
            logger.warning("Failed to audit a rejected webhook for %s", hook.service_name)
