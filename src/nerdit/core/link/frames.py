"""Parse and serialize the frozen `node-link/v1` wire contract.

Reject unknown frame fields, malformed base64, invalid HTTP headers, and
non-origin-relative request targets. Enforce the 100-header, 64 KiB header,
and 8192-query bounds. Serialize canonical frames without `None` fields.

`nerdit-cloud/docs/node-link-v1.md` defines the handshake, frame catalog,
capability payload, and stream-error vocabulary; fixtures in
`tests/data/node_link_v1/` and independent test models check conformance.
Production code must not import that test oracle.

`OpenStream` headers contain injected capabilities, so its repr is redacted.
Validation failures use `FrameError` without exposing credential material.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nerdit.core.link.capability import MintedCapability

#: The only protocol token this daemon speaks (spec §1; `relay/protocol.py`).
PROTOCOL_VERSION = "node-link/v1"

#: The only two `stream_error` codes we put on the wire — reference-daemon
#: parity (spec §4.3 "Uplink"; `dev/fake_daemon.py:797` and `:856`).
ERROR_CODE_AUTH = "node_authentication_failed"
ERROR_CODE_INTERNAL = "internal_error"

#: Envelope constants — outside `StreamLimits` and therefore never negotiated
#: in `hello_ack` (spec §8, "Envelope constants").
MAX_HEADER_COUNT = 100
MAX_HEADER_BYTES = 64 * 1024
MAX_QUERY_CHARS = 8192

#: RFC 7230 token set for a header name (spec §3, "Header syntax").
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

#: `method` matches `^[A-Z]{1,16}$` (spec §3, "Method").
_METHOD = re.compile(r"^[A-Z]{1,16}$")

Header: TypeAlias = tuple[str, str]
Headers: TypeAlias = tuple[Header, ...]


class FrameError(ValueError):
    """A frame could not be parsed or violated the envelope rules.

    One exception type for the whole layer on purpose: the receive loop's only
    two behaviours are "drop the frame" (mid-session) and "fail the handshake",
    and neither branches on *why* a frame was invalid.
    """


# ---------------------------------------------------------------------------
# Bodies, paths, headers — the shared envelope rules (spec §3)
# ---------------------------------------------------------------------------


def encode_body(data: bytes) -> str:
    """Return the standard-base64 form every body-carrying frame uses."""
    return base64.b64encode(data).decode("ascii")


def decode_body(body_b64: str) -> bytes:
    """Decode a frame body **strictly**; malformed input is a contract breach.

    `validate=True` matters: the permissive default silently discards
    non-alphabet characters, which would turn a corrupted or maliciously padded
    payload into a plausible-looking body instead of a rejected frame.
    """
    try:
        return base64.b64decode(body_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameError("body_b64 is not valid base64") from exc


def is_forwardable_path(path: str) -> bool:
    """Return whether a request target is origin-relative (spec §3).

    The relay never sends an absolute destination; a daemon that accepted one
    would be an open proxy.
    """
    if not path.startswith("/") or path.startswith("//"):
        return False
    if "://" in path or "\\" in path:
        return False
    return not any(character in path for character in ("\r", "\n", "\t", " "))


def _validate_headers(headers: Headers) -> Headers:
    """Apply the count / size / syntax bounds the relay itself applies."""
    if len(headers) > MAX_HEADER_COUNT:
        raise FrameError("too many headers")
    total = 0
    for name, value in headers:
        try:
            # HTTP header octets are latin-1 (RFC 7230 / PEP 3333), so this is
            # also *the* size accounting. A character above U+00FF cannot be a
            # legal header octet — and encoding it would raise
            # `UnicodeEncodeError`, which is not a `FrameError` and so
            # would escape the receive loop's `except FrameError` and tear down
            # the whole session over one nonconforming frame.
            total += len(name.encode("latin-1")) + len(value.encode("latin-1"))
        except UnicodeEncodeError as exc:
            # The *name* is safe to name; the value never is.
            raise FrameError(f"invalid header name or value: {name!r}") from exc
        control_char = any(
            (ord(character) < 32 and character != "\t") or ord(character) == 127
            for character in value
        )
        if not _HEADER_NAME.fullmatch(name) or control_char:
            # The *name* is safe to name; the value never is (a header value is
            # where the injected capability lives).
            raise FrameError(f"invalid header name or value: {name!r}")
    if total > MAX_HEADER_BYTES:
        raise FrameError("headers are too large")
    return headers


# ---------------------------------------------------------------------------
# Strict field readers — the `extra="forbid"` half of the contract
# ---------------------------------------------------------------------------


def _require_mapping(payload: object, what: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise FrameError(f"{what} must be a JSON object")
    for key in payload:
        if not isinstance(key, str):
            raise FrameError(f"{what} has a non-string key")
    return payload


def _check_keys(payload: Mapping[str, Any], allowed: frozenset[str], what: str) -> None:
    """Reject any key the model does not define (spec §3, "Unknown fields").

    A field *added* upstream must fail as loudly as a field removed — that is
    what makes this a contract check rather than a smoke test.
    """
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise FrameError(f"{what} carries unknown field(s): {', '.join(unknown)}")


def _str_field(payload: Mapping[str, Any], key: str, what: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise FrameError(f"{what}.{key} must be a string")
    return value


def _opt_str_field(payload: Mapping[str, Any], key: str, what: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise FrameError(f"{what}.{key} must be a string or absent")
    return value


def _int_field(payload: Mapping[str, Any], key: str, what: str) -> int:
    value = payload.get(key)
    # `bool` is an `int` subclass; `True` is not a body size.
    if not isinstance(value, int) or isinstance(value, bool):
        raise FrameError(f"{what}.{key} must be an integer")
    return value


def _float_field(payload: Mapping[str, Any], key: str, what: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrameError(f"{what}.{key} must be a number")
    result = float(value)
    # Python's JSON decoder accepts NaN/Infinity. NaN in particular slips
    # through every `<= 0` positivity check and `min()` clamp (all NaN
    # comparisons are False) and would land in asyncio timer arithmetic —
    # reject non-finite values as a parse failure, not a scheduling surprise.
    if not math.isfinite(result):
        raise FrameError(f"{what}.{key} must be a finite number")
    return result


def _headers_field(payload: Mapping[str, Any], what: str) -> Headers:
    raw = payload.get("headers", ())
    if raw is None:
        raise FrameError(f"{what}.headers must be a list of [name, value] pairs")
    if not isinstance(raw, (list, tuple)):
        raise FrameError(f"{what}.headers must be a list of [name, value] pairs")
    pairs: list[Header] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise FrameError(f"{what}.headers must be a list of [name, value] pairs")
        name, value = entry
        if not isinstance(name, str) or not isinstance(value, str):
            raise FrameError(f"{what}.headers must be a list of [name, value] pairs")
        pairs.append((name, value))
    return _validate_headers(tuple(pairs))


# ---------------------------------------------------------------------------
# Downlink frames — relay → daemon (spec §4.2)
# ---------------------------------------------------------------------------


#: Local sanity ceilings for the `hello_ack`-advertised limits table.
#:
#: The relay is trusted to *narrow* a limit, never to widen one past what this
#: daemon can safely do. `mux.py` accumulates a whole request body in memory
#: per stream, so `max_body_bytes` is a memory budget multiplied by
#: `max_concurrent_streams_per_node` — a relay (or a relay-shaped attacker on
#: a hijacked `[link].relay_url`) advertising `2**60` must not be able to
#: turn that into an OOM. Every value is therefore `min`-ed against these.
MAX_CONCURRENT_STREAMS_CEILING = 64
MAX_BODY_BYTES_CEILING = 64 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES_CEILING = 64 * 1024 * 1024
MAX_BANDWIDTH_BYTES_PER_SECOND_CEILING = 100 * 1024 * 1024
MAX_PENDING_FRAMES_CEILING = 1024
STREAM_TIMEOUT_CEILING_S = 24 * 60 * 60.0


def _bounded_int(payload: Mapping[str, Any], key: str, what: str, ceiling: int) -> int:
    """Read a positive integer limit, clamped down to `ceiling`."""
    value = _int_field(payload, key, what)
    if value <= 0:
        raise FrameError(f"{what}.{key} must be positive")
    return min(value, ceiling)


def _bounded_float(payload: Mapping[str, Any], key: str, what: str, ceiling: float) -> float:
    """Read a positive number limit, clamped down to `ceiling`."""
    value = _float_field(payload, key, what)
    if value <= 0:
        raise FrameError(f"{what}.{key} must be positive")
    return min(value, ceiling)


@dataclass(frozen=True, slots=True)
class StreamLimits:
    """Quotas the relay advertises in `hello_ack` (spec §8).

    No defaults, deliberately: a missing field is a parse failure rather than a
    silently-restored default, so a deployment that narrows a limit is honoured
    automatically.

    Obedience is **one-directional**, though, and that is the part worth
    stating: a relay may narrow any of these, and may never widen one past the
    local ceiling constants above. Non-positive values are rejected outright
    (a zero body budget or a zero timeout is not a narrowing, it is a broken
    table), and everything else is clamped. The daemon buffers a whole request
    body per stream, so trusting an advertised `max_body_bytes` verbatim
    would make the peer at the other end of the tunnel the author of this
    process's memory budget.

    `max_pending_frames` and `stream_idle_timeout_s` are parsed for wire
    parity (the table is frozen and every key is required) but not read
    locally. The relay enforces both, and the daemon bounds a stream by
    `mux.REQUEST_BODY_TIMEOUT_S` and `stream_absolute_timeout_s`.
    """

    max_concurrent_streams_per_node: int
    max_body_bytes: int
    max_response_body_bytes: int
    max_bandwidth_bytes_per_second: int
    max_pending_frames: int
    stream_idle_timeout_s: float
    stream_absolute_timeout_s: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> StreamLimits:
        """Parse `hello_ack.limits`; a missing, mistyped or absurd key is fatal."""
        what = "limits"
        mapping = _require_mapping(payload, what)
        _check_keys(mapping, _LIMITS_KEYS, what)
        return cls(
            max_concurrent_streams_per_node=_bounded_int(
                mapping,
                "max_concurrent_streams_per_node",
                what,
                MAX_CONCURRENT_STREAMS_CEILING,
            ),
            max_body_bytes=_bounded_int(mapping, "max_body_bytes", what, MAX_BODY_BYTES_CEILING),
            max_response_body_bytes=_bounded_int(
                mapping, "max_response_body_bytes", what, MAX_RESPONSE_BODY_BYTES_CEILING
            ),
            max_bandwidth_bytes_per_second=_bounded_int(
                mapping,
                "max_bandwidth_bytes_per_second",
                what,
                MAX_BANDWIDTH_BYTES_PER_SECOND_CEILING,
            ),
            max_pending_frames=_bounded_int(
                mapping, "max_pending_frames", what, MAX_PENDING_FRAMES_CEILING
            ),
            stream_idle_timeout_s=_bounded_float(
                mapping, "stream_idle_timeout_s", what, STREAM_TIMEOUT_CEILING_S
            ),
            stream_absolute_timeout_s=_bounded_float(
                mapping, "stream_absolute_timeout_s", what, STREAM_TIMEOUT_CEILING_S
            ),
        )


_LIMITS_KEYS = frozenset(
    {
        "max_concurrent_streams_per_node",
        "max_body_bytes",
        "max_response_body_bytes",
        "max_bandwidth_bytes_per_second",
        "max_pending_frames",
        "stream_idle_timeout_s",
        "stream_absolute_timeout_s",
    }
)


@dataclass(frozen=True, slots=True)
class NodeChallenge:
    """First relay frame: a fresh nonce that defeats signed-hello replay."""

    challenge: str
    relay_id: str


@dataclass(frozen=True, slots=True)
class HelloAck:
    """Relay acceptance: the heartbeat cadence and the limits to obey."""

    protocol: str
    node_id: str
    heartbeat_interval_s: float
    limits: StreamLimits


@dataclass(frozen=True, slots=True, repr=False)
class OpenStream:
    """One forwarded HTTP request the daemon must serve on its own listener.

    `role` is kept a plain `str` rather than a narrowed literal on
    purpose. The relay refuses to open a stream whose role differs from the
    connection's capability role (`relay/control.py:684`), so `"submitter"`
    is the only value the wire admits — but a *frame* that claimed otherwise
    must still parse, because the mux answers a bad role with a
    `stream_error` the requester can see. Rejecting it at parse time would
    silently drop the stream and leave the caller hanging.
    """

    stream_id: str
    node_id: str
    method: str
    path: str
    query: str
    headers: Headers
    body_b64: str | None
    role: str

    def header(self, name: str) -> str | None:
        """Return the first value of `name`, matched case-insensitively."""
        wanted = name.lower()
        for header_name, value in self.headers:
            if header_name.lower() == wanted:
                return value
        return None

    def body(self) -> bytes:
        """Decode the inline body; a bodiless frame yields `b""`."""
        if self.body_b64 is None:
            return b""
        return decode_body(self.body_b64)

    def __repr__(self) -> str:
        """Redacted by construction — headers carry the injected bearer.

        The relay strips any caller-supplied `authorization` and appends the
        connection's memory-only capability before forwarding (spec §4.2), so
        an `open_stream`'s headers are always credential-bearing. This repr
        is what keeps a stray `logger.debug("%s", frame)`, an f-string or a
        traceback from spilling it.
        """
        return (
            f"OpenStream(stream_id={self.stream_id!r}, method={self.method!r}, "
            f"path={self.path!r}, headers=<redacted>, body=<redacted>)"
        )

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class StreamBody:
    """Continuation of a request body too large for `open_stream`."""

    stream_id: str
    node_id: str
    body_b64: str

    def body(self) -> bytes:
        """Decode this chunk."""
        return decode_body(self.body_b64)


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """The request body is complete."""

    stream_id: str
    node_id: str


@dataclass(frozen=True, slots=True)
class CloseStream:
    """Abandon one stream; the daemon must stop producing for it."""

    stream_id: str
    node_id: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class ErrorFrame:
    """Connection-scoped failure (handshake, framing, authentication).

    `code` is a tolerant `str`, not an enum (oracle parity, spec §4.3): a
    newer relay may invent a code, and refusing to parse it would turn a
    diagnosable close into an unexplained one.
    """

    code: str
    message: str


StreamDownlink: TypeAlias = OpenStream | StreamBody | StreamEnd | CloseStream
Downlink: TypeAlias = NodeChallenge | HelloAck | StreamDownlink | ErrorFrame


def _parse_node_challenge(payload: Mapping[str, Any]) -> NodeChallenge:
    what = "node_challenge"
    _check_keys(payload, frozenset({"type", "protocol", "challenge", "relay_id"}), what)
    protocol = payload.get("protocol", PROTOCOL_VERSION)
    if protocol != PROTOCOL_VERSION:
        raise FrameError(f"{what}.protocol must be {PROTOCOL_VERSION!r}")
    return NodeChallenge(
        challenge=_str_field(payload, "challenge", what),
        relay_id=_str_field(payload, "relay_id", what),
    )


def _parse_hello_ack(payload: Mapping[str, Any]) -> HelloAck:
    what = "hello_ack"
    _check_keys(
        payload,
        frozenset({"type", "protocol", "node_id", "heartbeat_interval_s", "limits"}),
        what,
    )
    limits = payload.get("limits")
    if limits is None:
        raise FrameError(f"{what}.limits is required")
    return HelloAck(
        protocol=_str_field(payload, "protocol", what),
        node_id=_str_field(payload, "node_id", what),
        heartbeat_interval_s=_float_field(payload, "heartbeat_interval_s", what),
        limits=StreamLimits.from_payload(_require_mapping(limits, "limits")),
    )


def _parse_open_stream(payload: Mapping[str, Any]) -> OpenStream:
    what = "open_stream"
    _check_keys(
        payload,
        frozenset(
            {
                "type",
                "stream_id",
                "node_id",
                "method",
                "path",
                "query",
                "headers",
                "body_b64",
                "role",
            }
        ),
        what,
    )
    method = _str_field(payload, "method", what)
    if not _METHOD.fullmatch(method):
        raise FrameError(f"{what}.method is not a valid HTTP method token")
    path = _str_field(payload, "path", what)
    if not is_forwardable_path(path):
        raise FrameError(f"{what}.path must be an origin-relative request target")
    query = payload.get("query", "")
    if not isinstance(query, str):
        raise FrameError(f"{what}.query must be a string")
    if len(query) > MAX_QUERY_CHARS or any(ch in query for ch in ("\r", "\n", "#")):
        raise FrameError(f"{what}.query is not a valid origin query")
    body_b64 = _opt_str_field(payload, "body_b64", what)
    if body_b64 is not None:
        # Parsing the frame *is* the base64 check: a malformed payload fails
        # here rather than at the first later attempt to use the bytes.
        decode_body(body_b64)
    return OpenStream(
        stream_id=_str_field(payload, "stream_id", what),
        node_id=_str_field(payload, "node_id", what),
        method=method,
        path=path,
        query=query,
        headers=_headers_field(payload, what),
        body_b64=body_b64,
        role=_str_field(payload, "role", what),
    )


def _parse_stream_body(payload: Mapping[str, Any]) -> StreamBody:
    what = "stream_body"
    _check_keys(payload, frozenset({"type", "stream_id", "node_id", "body_b64"}), what)
    body_b64 = _str_field(payload, "body_b64", what)
    decode_body(body_b64)
    return StreamBody(
        stream_id=_str_field(payload, "stream_id", what),
        node_id=_str_field(payload, "node_id", what),
        body_b64=body_b64,
    )


def _parse_stream_end(payload: Mapping[str, Any]) -> StreamEnd:
    what = "stream_end"
    _check_keys(payload, frozenset({"type", "stream_id", "node_id"}), what)
    return StreamEnd(
        stream_id=_str_field(payload, "stream_id", what),
        node_id=_str_field(payload, "node_id", what),
    )


def _parse_close_stream(payload: Mapping[str, Any]) -> CloseStream:
    what = "close_stream"
    _check_keys(payload, frozenset({"type", "stream_id", "node_id", "reason"}), what)
    return CloseStream(
        stream_id=_str_field(payload, "stream_id", what),
        node_id=_str_field(payload, "node_id", what),
        reason=_opt_str_field(payload, "reason", what),
    )


def _parse_error(payload: Mapping[str, Any]) -> ErrorFrame:
    what = "error"
    _check_keys(payload, frozenset({"type", "code", "message"}), what)
    return ErrorFrame(
        code=_str_field(payload, "code", what),
        message=_str_field(payload, "message", what),
    )


_DOWNLINK_PARSERS: dict[str, Callable[[Mapping[str, Any]], Downlink]] = {
    "node_challenge": _parse_node_challenge,
    "hello_ack": _parse_hello_ack,
    "open_stream": _parse_open_stream,
    "stream_body": _parse_stream_body,
    "stream_end": _parse_stream_end,
    "close_stream": _parse_close_stream,
    "error": _parse_error,
}


def parse_downlink(raw: str) -> Downlink:
    """Parse one relay-to-daemon text frame.

    The union at spec §4.2 is exhaustive: an unknown `type` is a contract
    breach, not an extension point, and so is an unknown field on a known type.
    """
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise FrameError("frame is not valid JSON") from exc
    mapping = _require_mapping(payload, "frame")
    frame_type = mapping.get("type")
    if not isinstance(frame_type, str):
        raise FrameError("frame is missing its 'type' discriminator")
    parser = _DOWNLINK_PARSERS.get(frame_type)
    if parser is None:
        raise FrameError(f"unknown downlink frame type: {frame_type!r}")
    return parser(mapping)


# ---------------------------------------------------------------------------
# Uplink builders — daemon → relay (spec §4.1)
# ---------------------------------------------------------------------------


def _reject_none(value: object, path: str) -> None:
    if value is None:
        raise FrameError(f"frame field {path} is None; builders omit absent fields")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_none(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_none(item, f"{path}[{index}]")


def dump_frame(frame: Mapping[str, object]) -> str:
    """Serialize one uplink frame canonically (spec §3, "Canonical serialization").

    The relay's own `dump_frame` is `model_dump_json(exclude_none=True)`:
    a `None`-valued optional is **absent**, never `null`. Our builders omit
    those keys at construction, so this only has to *prove* it — a `None`
    that reached here would be a builder bug shipping a frame the relay's
    `extra="forbid"` models would refuse.
    """
    _reject_none(frame, "frame")
    return json.dumps(frame, separators=(",", ":"))


def hello_frame(
    *,
    node_id: str,
    node_name: str,
    daemon_version: str,
    uptime_s: int,
    capability: MintedCapability,
    proof: str,
) -> dict[str, object]:
    """Build the signed hello (spec §5 step 5).

    `expires_at` serializes with `datetime.isoformat` — the same call
    `nerdit.core.link.identity.proof_message` makes, so the signed bytes
    and the wire string cannot disagree. (The relay parses either the `Z` or
    the `+00:00` spelling; the proof binds the *datetime*, not the string.)
    """
    return {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "node_id": node_id,
        "node_name": node_name,
        "daemon_version": daemon_version,
        "uptime_s": uptime_s,
        "capability": {
            "token": capability.token,
            "expires_at": capability.expires_at.isoformat(),
            "role": "submitter",
        },
        "proof": proof,
    }


def heartbeat_frame(*, node_id: str, uptime_s: int) -> dict[str, object]:
    """Build one liveness beat.

    `workload_state` is optional on the wire and deliberately **omitted** in
    v1: it is free text about what the node is doing, the relay only throttles
    and forwards it, and nothing on the cloud side consumes it yet. Sending it
    would leak node activity for no consumer.
    """
    return {"type": "heartbeat", "node_id": node_id, "uptime_s": uptime_s}


def response_head_frame(
    *, stream_id: str, node_id: str, status: int, headers: Sequence[Header]
) -> dict[str, object]:
    """Build the status + headers of a locally served response."""
    return {
        "type": "stream_response_head",
        "stream_id": stream_id,
        "node_id": node_id,
        "status": status,
        "headers": [[name, value] for name, value in headers],
    }


def response_body_frame(*, stream_id: str, node_id: str, body: bytes) -> dict[str, object]:
    """Build one response body chunk (SSE is a sequence of these)."""
    return {
        "type": "stream_response_body",
        "stream_id": stream_id,
        "node_id": node_id,
        "body_b64": encode_body(body),
    }


def response_end_frame(*, stream_id: str, node_id: str) -> dict[str, object]:
    """Build the response terminator."""
    return {"type": "stream_response_end", "stream_id": stream_id, "node_id": node_id}


def stream_error_frame(
    *, stream_id: str, node_id: str, code: str, message: str
) -> dict[str, object]:
    """Build a stream-scoped failure (only the two §4.3 codes are emitted)."""
    return {
        "type": "stream_error",
        "stream_id": stream_id,
        "node_id": node_id,
        "code": code,
        "message": message,
    }


__all__ = [
    "ERROR_CODE_AUTH",
    "ERROR_CODE_INTERNAL",
    "MAX_BANDWIDTH_BYTES_PER_SECOND_CEILING",
    "MAX_BODY_BYTES_CEILING",
    "MAX_CONCURRENT_STREAMS_CEILING",
    "MAX_HEADER_BYTES",
    "MAX_HEADER_COUNT",
    "MAX_PENDING_FRAMES_CEILING",
    "MAX_QUERY_CHARS",
    "MAX_RESPONSE_BODY_BYTES_CEILING",
    "PROTOCOL_VERSION",
    "STREAM_TIMEOUT_CEILING_S",
    "CloseStream",
    "Downlink",
    "ErrorFrame",
    "FrameError",
    "Header",
    "Headers",
    "HelloAck",
    "NodeChallenge",
    "OpenStream",
    "StreamBody",
    "StreamDownlink",
    "StreamEnd",
    "StreamLimits",
    "decode_body",
    "dump_frame",
    "encode_body",
    "heartbeat_frame",
    "hello_frame",
    "is_forwardable_path",
    "parse_downlink",
    "response_body_frame",
    "response_end_frame",
    "response_head_frame",
    "stream_error_frame",
]
