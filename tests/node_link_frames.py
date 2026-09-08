"""Independently model node-link/v1 for conformance tests.

Source: cloud docs/node-link-v1.md at commit
d53507081821d62e99991e2ce9f675e80cdca93e; fixtures frozen 2026-08-05.
Derive models from the spec without importing producer or runtime models, so
agreement provides independent evidence. These models are test-only.

Names use the daemon's perspective: DaemonInbound comes from the relay;
DaemonOutbound goes to it. Every model forbids extra fields to detect added
fields as well as removed ones.
"""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

#: The only protocol token this daemon model set claims to speak.
PROTOCOL_VERSION = "node-link/v1"

#: Shared request/response header syntax and size bounds (spec §"headers").
MAX_HEADER_COUNT = 100
MAX_HEADER_BYTES = 64 * 1024
HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")

HeaderList: TypeAlias = list[tuple[str, str]]


def validate_header_list(headers: HeaderList) -> HeaderList:
    """Reject header lists the relay would refuse (count, size, syntax)."""
    if len(headers) > MAX_HEADER_COUNT:
        raise ValueError("too many headers")
    total = 0
    for name, value in headers:
        total += len(name.encode("latin-1")) + len(value.encode("latin-1"))
        control_char = any(
            (ord(character) < 32 and character != "\t") or ord(character) == 127
            for character in value
        )
        if not HEADER_NAME.fullmatch(name) or control_char:
            raise ValueError(f"invalid header name or value: {name!r}")
    if total > MAX_HEADER_BYTES:
        raise ValueError("headers are too large")
    return headers


def decode_body(body_b64: str) -> bytes:
    """Decode a frame body strictly; malformed base64 is a contract breach."""
    try:
        return base64.b64decode(body_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("body_b64 is not valid base64") from exc


def validate_body_b64(body_b64: str | None) -> str | None:
    """Field validator body: parsing a frame *is* the base64 check.

    Every body-bearing frame runs this, so a malformed payload fails at
    ``parse_inbound``/``parse_outbound`` rather than at the first later attempt
    to use the bytes. An absent body (``None``) and an empty body (``""``) are
    both legal on the wire — the fixtures carry bodiless ``open_stream`` frames
    — and are passed through untouched.
    """
    if body_b64 is None:
        return None
    decode_body(body_b64)
    return body_b64


def is_forwardable_path(path: str) -> bool:
    """Return whether a request target is origin-relative (spec §"open_stream").

    The relay never sends an absolute destination; a daemon that accepted one
    would be an open proxy.
    """
    if not path.startswith("/") or path.startswith("//"):
        return False
    if "://" in path or "\\" in path:
        return False
    return not any(character in path for character in ("\r", "\n", "\t", " "))


class NodeLinkFrame(BaseModel):
    """Base for every ``node-link/v1`` frame: unknown keys are a breach."""

    model_config = ConfigDict(extra="forbid")


class StreamLimits(NodeLinkFrame):
    """Quotas the relay advertises in ``hello_ack``.

    No defaults on purpose: the daemon must read what the relay actually sent,
    and the conformance test asserts the seven spec numbers itself rather than
    letting a model default paper over a missing field.
    """

    max_concurrent_streams_per_node: int
    max_body_bytes: int
    max_response_body_bytes: int
    max_bandwidth_bytes_per_second: int
    max_pending_frames: int
    stream_idle_timeout_s: float
    stream_absolute_timeout_s: float


class CapabilityPayload(NodeLinkFrame):
    """Model memory-only, submitter-only tunnel capabilities.

    Never log, persist or echo this material. Expiry must be timezone-aware;
    the daemon owns renewal, and the relay does not close before expiry.

    Require role explicitly with no default. Although the producer model allows
    admin, the relay and frozen fixtures accept only submitter. Missing and admin
    roles must fail instead of being silently repaired or accepted.
    """

    token: str
    expires_at: datetime
    role: Literal["submitter"]

    @field_validator("expires_at")
    @classmethod
    def require_aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("capability expiry must include a timezone")
        return value

    def __repr__(self) -> str:
        return f"CapabilityPayload(token=<redacted>, expires_at={self.expires_at!r})"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# relay -> daemon (DaemonInbound)
# ---------------------------------------------------------------------------


class NodeChallengeFrame(NodeLinkFrame):
    """First relay frame: a fresh nonce that defeats signed-hello replay."""

    type: Literal["node_challenge"] = "node_challenge"
    protocol: Literal["node-link/v1"] = "node-link/v1"
    challenge: str
    relay_id: str


class HelloAckFrame(NodeLinkFrame):
    """Relay acceptance: the heartbeat cadence and the limits to obey."""

    type: Literal["hello_ack"] = "hello_ack"
    protocol: str
    node_id: str
    heartbeat_interval_s: float
    limits: StreamLimits


class OpenStreamFrame(NodeLinkFrame):
    """One forwarded HTTP request the daemon must serve locally.

    ``role`` travels at the **top level of the frame**, not inside ``headers``:
    all six vendored exchange fixtures carry ``"role": "submitter"`` as a
    sibling of ``method``/``path``, and no fixture injects a role header. It is
    required and single-valued for the same reason as
    :class:`CapabilityPayload.role` — the relay refuses to open a stream whose
    role differs from the connection's capability role
    (``relay/control.py:684``), and that capability role can only ever be
    ``"submitter"`` (``relay/control.py:319``, P27 D-R2). A default would let a
    role-less frame parse as a submitter request it never claimed to be.
    """

    type: Literal["open_stream"] = "open_stream"
    stream_id: str
    node_id: str
    method: str = Field(pattern=r"^[A-Z]{1,16}$")
    path: str
    query: str = ""
    headers: HeaderList = Field(default_factory=list)
    body_b64: str | None = None
    role: Literal["submitter"]

    @field_validator("body_b64")
    @classmethod
    def check_body(cls, body_b64: str | None) -> str | None:
        return validate_body_b64(body_b64)

    @field_validator("path")
    @classmethod
    def reject_absolute_destination(cls, path: str) -> str:
        if not is_forwardable_path(path):
            raise ValueError("path must be an origin-relative request target")
        return path

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        if len(query) > 8192 or any(character in query for character in ("\r", "\n", "#")):
            raise ValueError("query is not a valid origin query")
        return query

    @field_validator("headers")
    @classmethod
    def check_headers(cls, headers: HeaderList) -> HeaderList:
        return validate_header_list(headers)


class StreamBodyFrame(NodeLinkFrame):
    """Continuation of a request body too large for ``open_stream``."""

    type: Literal["stream_body"] = "stream_body"
    stream_id: str
    node_id: str
    body_b64: str

    @field_validator("body_b64")
    @classmethod
    def check_body(cls, body_b64: str) -> str:
        decode_body(body_b64)
        return body_b64


class StreamEndFrame(NodeLinkFrame):
    """The request body is complete."""

    type: Literal["stream_end"] = "stream_end"
    stream_id: str
    node_id: str


class CloseStreamFrame(NodeLinkFrame):
    """Abandon one stream; the daemon must stop producing for it."""

    type: Literal["close_stream"] = "close_stream"
    stream_id: str
    node_id: str
    reason: str | None = None


class ErrorFrame(NodeLinkFrame):
    """Connection-scoped failure (handshake, framing, authentication).

    ``code`` is a plain ``str``, not an enum: the daemon must tolerate codes a
    newer relay invents. The conformance test asserts the specific codes the
    fixtures carry; the model refuses to over-constrain beyond the spec.
    """

    type: Literal["error"] = "error"
    code: str
    message: str


# ---------------------------------------------------------------------------
# daemon -> relay (DaemonOutbound)
# ---------------------------------------------------------------------------


class HelloFrame(NodeLinkFrame):
    """Daemon's signed hello, answering ``node_challenge``."""

    type: Literal["hello"] = "hello"
    protocol: str
    node_id: str
    node_name: str
    daemon_version: str
    uptime_s: int
    capability: CapabilityPayload
    proof: str = ""


class HeartbeatFrame(NodeLinkFrame):
    """Liveness beat at the cadence ``hello_ack`` asked for."""

    type: Literal["heartbeat"] = "heartbeat"
    node_id: str
    uptime_s: int
    workload_state: str | None = None


class StreamResponseHeadFrame(NodeLinkFrame):
    """Status and headers of the local response."""

    type: Literal["stream_response_head"] = "stream_response_head"
    stream_id: str
    node_id: str
    status: int = Field(ge=100, le=599)
    headers: HeaderList = Field(default_factory=list)

    @field_validator("headers")
    @classmethod
    def check_headers(cls, headers: HeaderList) -> HeaderList:
        return validate_header_list(headers)


class StreamResponseBodyFrame(NodeLinkFrame):
    """One response body chunk; SSE is a sequence of these."""

    type: Literal["stream_response_body"] = "stream_response_body"
    stream_id: str
    node_id: str
    body_b64: str

    @field_validator("body_b64")
    @classmethod
    def check_body(cls, body_b64: str) -> str:
        decode_body(body_b64)
        return body_b64


class StreamResponseEndFrame(NodeLinkFrame):
    """The response body is complete."""

    type: Literal["stream_response_end"] = "stream_response_end"
    stream_id: str
    node_id: str


class StreamErrorFrame(NodeLinkFrame):
    """Stream-scoped failure; ``code`` is tolerant for the reason above."""

    type: Literal["stream_error"] = "stream_error"
    stream_id: str
    node_id: str | None = None
    code: str
    message: str


DaemonInbound: TypeAlias = Annotated[
    NodeChallengeFrame
    | HelloAckFrame
    | OpenStreamFrame
    | StreamBodyFrame
    | StreamEndFrame
    | CloseStreamFrame
    | ErrorFrame,
    Field(discriminator="type"),
]

DaemonOutbound: TypeAlias = Annotated[
    HelloFrame
    | HeartbeatFrame
    | StreamResponseHeadFrame
    | StreamResponseBodyFrame
    | StreamResponseEndFrame
    | StreamErrorFrame,
    Field(discriminator="type"),
]

DAEMON_INBOUND_ADAPTER: TypeAdapter[DaemonInbound] = TypeAdapter(DaemonInbound)
DAEMON_OUTBOUND_ADAPTER: TypeAdapter[DaemonOutbound] = TypeAdapter(DaemonOutbound)


def parse_inbound(payload: dict[str, object]) -> DaemonInbound:
    """Validate one relay-to-daemon frame."""
    return DAEMON_INBOUND_ADAPTER.validate_python(payload)


def parse_outbound(payload: dict[str, object]) -> DaemonOutbound:
    """Validate one daemon-to-relay frame."""
    return DAEMON_OUTBOUND_ADAPTER.validate_python(payload)
