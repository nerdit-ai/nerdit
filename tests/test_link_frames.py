"""Compare runtime node-link frames with an independent conformance oracle.

Neither implementation imports the other. Validate downlink fixtures field by
field, runtime outbound builders against extra-forbid oracle models, and exchange
bytes against reference-daemon fixtures. Both models must reject the same invalid
serialization, base64, header/query/path/method bounds and unknown fields.
Keep fixtures-versus-oracle coverage in test_node_link_conformance.py separate.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from nerdit.core.link import frames as runtime
from nerdit.core.link.capability import MintedCapability
from tests import node_link_frames as oracle
from tests.link_fake_relay import FIXTURE_NODE_ID, load_fixture
from tests.node_link_files import EXCHANGE_FILES, INBOUND_FILES

#: Every downlink frame the fixture set carries, tagged with its origin.
DOWNLINK_FRAMES: list[tuple[str, dict[str, Any]]] = [
    *((name, load_fixture(name)) for name in INBOUND_FILES),
    *((name, load_fixture(name)["request"]) for name in EXCHANGE_FILES),
]

#: Every uplink frame of every exchange, tagged with its origin.
UPLINK_FRAMES: list[tuple[str, int, dict[str, Any]]] = [
    (name, index, frame)
    for name in EXCHANGE_FILES
    for index, frame in enumerate(load_fixture(name)["response"])
]


# ---------------------------------------------------------------------------
# constants agree with the oracle (and therefore with the spec)
# ---------------------------------------------------------------------------


def test_protocol_version_and_envelope_bounds_match_the_oracle() -> None:
    assert runtime.PROTOCOL_VERSION == oracle.PROTOCOL_VERSION == "node-link/v1"
    assert runtime.MAX_HEADER_COUNT == oracle.MAX_HEADER_COUNT == 100
    assert runtime.MAX_HEADER_BYTES == oracle.MAX_HEADER_BYTES == 64 * 1024
    assert runtime.MAX_QUERY_CHARS == 8192


def test_only_the_two_reference_stream_error_codes_exist() -> None:
    """Spec §4.3: the reference daemon emits exactly these two, and so do we."""
    assert runtime.ERROR_CODE_AUTH == "node_authentication_failed"
    assert runtime.ERROR_CODE_INTERNAL == "internal_error"


# ---------------------------------------------------------------------------
# 1. fixtures parse identically under runtime and oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "frame"), DOWNLINK_FRAMES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_every_fixture_downlink_frame_parses_under_the_runtime_parser(
    name: str, frame: dict[str, Any]
) -> None:
    parsed = runtime.parse_downlink(json.dumps(frame))
    assert parsed is not None, name


def test_node_challenge_agrees_with_the_oracle() -> None:
    raw = load_fixture("node_challenge.json")
    mine = runtime.parse_downlink(json.dumps(raw))
    theirs = oracle.parse_inbound(raw)
    assert isinstance(mine, runtime.NodeChallenge)
    assert isinstance(theirs, oracle.NodeChallengeFrame)
    assert (mine.challenge, mine.relay_id) == (theirs.challenge, theirs.relay_id)


def test_hello_ack_agrees_with_the_oracle_on_all_seven_limits() -> None:
    raw = load_fixture("hello_ack.json")
    mine = runtime.parse_downlink(json.dumps(raw))
    theirs = oracle.parse_inbound(raw)
    assert isinstance(mine, runtime.HelloAck)
    assert isinstance(theirs, oracle.HelloAckFrame)
    assert (mine.protocol, mine.node_id) == (theirs.protocol, theirs.node_id)
    assert mine.heartbeat_interval_s == theirs.heartbeat_interval_s
    # Field-by-field, not ``==`` on the objects: they are different classes, and
    # the point is that seven independently parsed numbers coincide.
    for field in (
        "max_concurrent_streams_per_node",
        "max_body_bytes",
        "max_response_body_bytes",
        "max_bandwidth_bytes_per_second",
        "max_pending_frames",
        "stream_idle_timeout_s",
        "stream_absolute_timeout_s",
    ):
        assert getattr(mine.limits, field) == getattr(theirs.limits, field), field


@pytest.mark.parametrize("name", EXCHANGE_FILES)
def test_exchange_open_stream_agrees_with_the_oracle(name: str) -> None:
    raw = load_fixture(name)["request"]
    mine = runtime.parse_downlink(json.dumps(raw))
    theirs = oracle.parse_inbound(raw)
    assert isinstance(mine, runtime.OpenStream)
    assert isinstance(theirs, oracle.OpenStreamFrame)
    assert mine.stream_id == theirs.stream_id
    assert mine.node_id == theirs.node_id
    assert mine.method == theirs.method
    assert mine.path == theirs.path
    assert mine.query == theirs.query
    assert mine.role == theirs.role
    assert list(mine.headers) == [tuple(pair) for pair in theirs.headers]
    assert mine.body_b64 == theirs.body_b64
    # And the decoded bytes, which is what the mux actually forwards.
    assert mine.body() == (oracle.decode_body(theirs.body_b64) if theirs.body_b64 else b"")


def test_stream_body_end_and_close_round_trip() -> None:
    """The three frames no fixture carries, checked against the oracle too."""
    body = {
        "type": "stream_body",
        "stream_id": "s1",
        "node_id": FIXTURE_NODE_ID,
        "body_b64": runtime.encode_body(b"chunk"),
    }
    end = {"type": "stream_end", "stream_id": "s1", "node_id": FIXTURE_NODE_ID}
    close = {
        "type": "close_stream",
        "stream_id": "s1",
        "node_id": FIXTURE_NODE_ID,
        "reason": "requester went away",
    }
    parsed_body = runtime.parse_downlink(json.dumps(body))
    assert isinstance(parsed_body, runtime.StreamBody)
    assert parsed_body.body() == b"chunk"
    assert isinstance(runtime.parse_downlink(json.dumps(end)), runtime.StreamEnd)
    parsed_close = runtime.parse_downlink(json.dumps(close))
    assert isinstance(parsed_close, runtime.CloseStream)
    assert parsed_close.reason == "requester went away"
    for frame in (body, end, close):
        assert oracle.parse_inbound(frame) is not None


def test_close_stream_reason_is_optional() -> None:
    parsed = runtime.parse_downlink(
        json.dumps({"type": "close_stream", "stream_id": "s1", "node_id": "n"})
    )
    assert isinstance(parsed, runtime.CloseStream)
    assert parsed.reason is None


def test_error_frame_code_is_tolerant() -> None:
    """A newer relay may invent a code; refusing it would hide the reason."""
    parsed = runtime.parse_downlink(
        json.dumps({"type": "error", "code": "a_code_from_2027", "message": "why"})
    )
    assert isinstance(parsed, runtime.ErrorFrame)
    assert parsed.code == "a_code_from_2027"
    assert oracle.parse_inbound({"type": "error", "code": "a_code_from_2027", "message": "why"})


# ---------------------------------------------------------------------------
# 2. runtime builders produce frames the oracle (and the relay) accept
# ---------------------------------------------------------------------------


def _capability() -> MintedCapability:
    return MintedCapability(
        token="fixture-capability-token-not-a-secret",
        expires_at=datetime(2026, 1, 1, 12, 10, tzinfo=UTC),
        role="submitter",
    )


def test_hello_builder_matches_the_frozen_hello_fixture() -> None:
    """The strongest single assertion in this file.

    ``hello.json`` is the frozen shape the relay parses. Building it from the
    runtime builder with the fixture's own inputs must reproduce it exactly —
    key for key, value for value — or our hello is not the hello the freeze
    pinned.
    """
    fixture = load_fixture("hello.json")
    built = runtime.hello_frame(
        node_id=fixture["node_id"],
        node_name=fixture["node_name"],
        daemon_version=fixture["daemon_version"],
        uptime_s=fixture["uptime_s"],
        capability=_capability(),
        proof=fixture["proof"],
    )
    # ``expires_at`` is the one legitimate spelling difference: the fixture
    # writes ``Z``, ``datetime.isoformat()`` writes ``+00:00``. The proof binds
    # the *datetime*, not the string (identity.proof_message), and the relay
    # parses both — so compare the parsed instants, and everything else exactly.
    assert built["capability"]["expires_at"] == "2026-01-01T12:10:00+00:00"
    assert datetime.fromisoformat(str(built["capability"]["expires_at"])) == datetime.fromisoformat(
        fixture["capability"]["expires_at"].replace("Z", "+00:00")
    )
    normalized = json.loads(json.dumps(built))
    normalized["capability"]["expires_at"] = fixture["capability"]["expires_at"]
    assert normalized == fixture


def test_heartbeat_builder_matches_the_fixture_minus_workload_state() -> None:
    """``workload_state`` is optional and deliberately omitted in v1."""
    fixture = load_fixture("heartbeat.json")
    built = runtime.heartbeat_frame(node_id=fixture["node_id"], uptime_s=fixture["uptime_s"])
    assert built == {k: v for k, v in fixture.items() if k != "workload_state"}
    assert "workload_state" not in built
    parsed = oracle.parse_outbound(built)
    assert isinstance(parsed, oracle.HeartbeatFrame)
    assert parsed.workload_state is None


@pytest.mark.parametrize("name", EXCHANGE_FILES)
def test_runtime_builders_reproduce_every_fixture_uplink_frame(name: str) -> None:
    """Replay each exchange's answer with our builders; expect byte agreement."""
    exchange = load_fixture(name)
    request = exchange["request"]
    built: list[dict[str, Any]] = []
    for frame in exchange["response"]:
        if frame["type"] == "stream_response_head":
            built.append(
                runtime.response_head_frame(
                    stream_id=frame["stream_id"],
                    node_id=frame["node_id"],
                    status=frame["status"],
                    headers=[tuple(pair) for pair in frame["headers"]],
                )
            )
        elif frame["type"] == "stream_response_body":
            built.append(
                runtime.response_body_frame(
                    stream_id=frame["stream_id"],
                    node_id=frame["node_id"],
                    body=base64.b64decode(frame["body_b64"], validate=True),
                )
            )
        else:
            built.append(
                runtime.response_end_frame(stream_id=frame["stream_id"], node_id=frame["node_id"])
            )
    assert json.loads(json.dumps(built)) == exchange["response"], name
    assert all(frame["stream_id"] == request["stream_id"] for frame in built)


@pytest.mark.parametrize(("name", "index", "frame"), UPLINK_FRAMES, ids=lambda v: str(v))
def test_every_fixture_uplink_frame_survives_the_runtime_serializer(
    name: str, index: int, frame: dict[str, Any]
) -> None:
    """``dump_frame`` is canonical: compact, and a parse round-trip is lossless."""
    dumped = runtime.dump_frame(frame)
    assert ", " not in dumped and '": ' not in dumped  # compact separators
    assert json.loads(dumped) == frame


def test_every_builder_output_validates_under_the_oracle() -> None:
    """The whole uplink catalog (spec §4.1), in one pass."""
    outbound = [
        runtime.hello_frame(
            node_id=FIXTURE_NODE_ID,
            node_name="Dev Node",
            daemon_version="0.1.0",
            uptime_s=4242,
            capability=_capability(),
            proof="fixture-ed25519-signature-not-a-secret",
        ),
        runtime.heartbeat_frame(node_id=FIXTURE_NODE_ID, uptime_s=4247),
        runtime.response_head_frame(
            stream_id="s1",
            node_id=FIXTURE_NODE_ID,
            status=200,
            headers=[("content-type", "application/json")],
        ),
        runtime.response_body_frame(stream_id="s1", node_id=FIXTURE_NODE_ID, body=b"{}"),
        runtime.response_end_frame(stream_id="s1", node_id=FIXTURE_NODE_ID),
        runtime.stream_error_frame(
            stream_id="s1",
            node_id=FIXTURE_NODE_ID,
            code=runtime.ERROR_CODE_AUTH,
            message="the tunnel capability is invalid",
        ),
    ]
    expected = [
        "hello",
        "heartbeat",
        "stream_response_head",
        "stream_response_body",
        "stream_response_end",
        "stream_error",
    ]
    assert [frame["type"] for frame in outbound] == expected
    for frame in outbound:
        parsed = oracle.parse_outbound(frame)
        assert parsed.type == frame["type"]
        # And it survives the canonical serializer (no ``None`` anywhere).
        assert json.loads(runtime.dump_frame(frame)) == frame


def test_response_body_frame_round_trips_arbitrary_bytes() -> None:
    payload = bytes(range(256))
    frame = runtime.response_body_frame(stream_id="s", node_id="n", body=payload)
    assert oracle.decode_body(str(frame["body_b64"])) == payload
    assert runtime.decode_body(str(frame["body_b64"])) == payload


# ---------------------------------------------------------------------------
# 3. canonical serialization + strict base64
# ---------------------------------------------------------------------------


def test_dump_frame_is_compact_and_refuses_none_values() -> None:
    """Spec §3: ``exclude_none`` parity — an absent optional is *absent*."""
    assert runtime.dump_frame({"a": 1, "b": "x"}) == '{"a":1,"b":"x"}'
    with pytest.raises(runtime.FrameError):
        runtime.dump_frame({"type": "heartbeat", "workload_state": None})


def test_dump_frame_rejects_a_nested_none() -> None:
    """The hello nests a capability object; a ``None`` in there is still a bug."""
    with pytest.raises(runtime.FrameError):
        runtime.dump_frame({"capability": {"token": "t", "expires_at": None}})
    with pytest.raises(runtime.FrameError):
        runtime.dump_frame({"headers": [["a", "b"], None]})


@pytest.mark.parametrize(
    "bad",
    [
        "not base64!",
        "YWJj*",  # non-alphabet character
        "YWJ",  # bad padding
        "  YWJj  ",  # whitespace is silently dropped by the permissive decoder
    ],
)
def test_decode_body_is_strict(bad: str) -> None:
    """``validate=True``: the permissive default would invent a plausible body."""
    with pytest.raises(runtime.FrameError):
        runtime.decode_body(bad)
    with pytest.raises(ValueError):
        oracle.decode_body(bad)


def test_encode_decode_round_trip() -> None:
    for payload in (b"", b"a", b"\x00\xff" * 100):
        assert runtime.decode_body(runtime.encode_body(payload)) == payload


# ---------------------------------------------------------------------------
# 4. the malformed-frame FrameError matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "raw"),
    [
        ("not json", "{"),
        ("not an object", "[1, 2]"),
        ("no discriminator", '{"stream_id": "s"}'),
        ("non-string discriminator", '{"type": 7}'),
        ("unknown type", '{"type": "open_tunnel"}'),
    ],
)
def test_parse_downlink_rejects_malformed_envelopes(label: str, raw: str) -> None:
    with pytest.raises(runtime.FrameError):
        runtime.parse_downlink(raw)


@pytest.mark.parametrize(
    ("label", "frame"),
    [
        (
            "unknown field",
            {"type": "stream_end", "stream_id": "s", "node_id": "n", "extra": 1},
        ),
        ("missing field", {"type": "stream_end", "stream_id": "s"}),
        ("mistyped field", {"type": "stream_end", "stream_id": 7, "node_id": "n"}),
        (
            "wrong protocol",
            {
                "type": "node_challenge",
                "protocol": "node-link/v2",
                "challenge": "c",
                "relay_id": "r",
            },
        ),
        (
            "missing limits",
            {
                "type": "hello_ack",
                "protocol": "node-link/v1",
                "node_id": "n",
                "heartbeat_interval_s": 5.0,
            },
        ),
        (
            "limits missing a key",
            {
                "type": "hello_ack",
                "protocol": "node-link/v1",
                "node_id": "n",
                "heartbeat_interval_s": 5.0,
                "limits": {"max_body_bytes": 1},
            },
        ),
        (
            "bad request body base64",
            {
                "type": "stream_body",
                "stream_id": "s",
                "node_id": "n",
                "body_b64": "not base64!",
            },
        ),
    ],
)
def test_parse_downlink_rejects_field_violations(label: str, frame: dict[str, Any]) -> None:
    with pytest.raises(runtime.FrameError):
        runtime.parse_downlink(json.dumps(frame))


# ---------------------------------------------------------------------------
# the limits table is obeyed DOWNWARD only
# ---------------------------------------------------------------------------


def _limits(**overrides: Any) -> dict[str, Any]:
    payload = dict(load_fixture("hello_ack.json")["limits"])
    payload.update(overrides)
    return payload


def test_a_relay_may_narrow_every_limit() -> None:
    """The obedience half: a deployment that tightens a quota is honoured."""
    limits = runtime.StreamLimits.from_payload(
        _limits(
            max_concurrent_streams_per_node=2,
            max_body_bytes=1024,
            max_response_body_bytes=2048,
            max_bandwidth_bytes_per_second=4096,
            max_pending_frames=8,
            stream_idle_timeout_s=1.5,
            stream_absolute_timeout_s=30.0,
        )
    )
    assert limits.max_concurrent_streams_per_node == 2
    assert limits.max_body_bytes == 1024
    assert limits.max_response_body_bytes == 2048
    assert limits.max_bandwidth_bytes_per_second == 4096
    assert limits.max_pending_frames == 8
    assert (limits.stream_idle_timeout_s, limits.stream_absolute_timeout_s) == (1.5, 30.0)


@pytest.mark.parametrize(
    ("field", "ceiling"),
    [
        ("max_concurrent_streams_per_node", runtime.MAX_CONCURRENT_STREAMS_CEILING),
        ("max_body_bytes", runtime.MAX_BODY_BYTES_CEILING),
        ("max_response_body_bytes", runtime.MAX_RESPONSE_BODY_BYTES_CEILING),
        ("max_bandwidth_bytes_per_second", runtime.MAX_BANDWIDTH_BYTES_PER_SECOND_CEILING),
        ("max_pending_frames", runtime.MAX_PENDING_FRAMES_CEILING),
    ],
)
def test_an_absurd_advertised_limit_is_clamped_to_the_local_ceiling(
    field: str, ceiling: int
) -> None:
    """A relay may narrow a limit; it may never widen one.

    ``mux.py`` buffers a whole request body in memory per stream, so an obeyed
    ``max_body_bytes = 2**60`` would make the peer at the far end of the tunnel
    the author of this process's memory budget.
    """
    limits = runtime.StreamLimits.from_payload(_limits(**{field: 2**60}))
    assert getattr(limits, field) == ceiling


@pytest.mark.parametrize(
    "field",
    [
        "max_concurrent_streams_per_node",
        "max_body_bytes",
        "max_response_body_bytes",
        "max_bandwidth_bytes_per_second",
        "max_pending_frames",
        "stream_idle_timeout_s",
        "stream_absolute_timeout_s",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_a_non_positive_advertised_limit_is_a_parse_failure(field: str, value: int) -> None:
    """Zero is not a narrowing — a zero body budget or a zero timeout is a
    broken table, and guessing a default for it would be worse than refusing."""
    with pytest.raises(runtime.FrameError, match="must be positive"):
        runtime.StreamLimits.from_payload(_limits(**{field: value}))


@pytest.mark.parametrize("field", ["stream_idle_timeout_s", "stream_absolute_timeout_s"])
def test_an_absurd_advertised_timeout_is_clamped_to_a_day(field: str) -> None:
    limits = runtime.StreamLimits.from_payload(_limits(**{field: 1e12}))
    assert getattr(limits, field) == runtime.STREAM_TIMEOUT_CEILING_S


def test_the_clamp_reaches_a_hello_ack_parsed_off_the_wire() -> None:
    """End to end: the ceiling is enforced where the frame actually arrives."""
    raw = load_fixture("hello_ack.json")
    raw["limits"] = _limits(max_body_bytes=2**60)
    frame = runtime.parse_downlink(json.dumps(raw))
    assert isinstance(frame, runtime.HelloAck)
    assert frame.limits.max_body_bytes == runtime.MAX_BODY_BYTES_CEILING


# ---------------------------------------------------------------------------
# header octets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "headers"),
    [
        ("value above U+00FF", [["x-snow", "sn☃wman"]]),
        ("name above U+00FF", [["x-sn☃w", "ok"]]),
        ("astral value", [["x-emoji", "\U0001f600"]]),
    ],
)
def test_a_non_latin1_header_is_a_frame_error_not_a_unicode_error(
    label: str, headers: list[list[str]]
) -> None:
    """Header octets are latin-1; anything above U+00FF cannot be one.

    The size accounting encodes them, so this used to raise
    ``UnicodeEncodeError`` — which is not a :class:`FrameError` and therefore
    escaped the mid-session receive loop's handler, tearing down the whole
    session (and every live stream on it) over one nonconforming frame.
    """
    with pytest.raises(runtime.FrameError, match="invalid header name or value"):
        runtime.parse_downlink(json.dumps(_open_stream(headers=headers)))


def test_a_rejected_header_never_names_its_value() -> None:
    """A header value is where the relay-injected capability lives."""
    secret = "Bearer sn☃wman-capability"  # noqa: S105 - a test literal
    with pytest.raises(runtime.FrameError) as excinfo:
        runtime.parse_downlink(json.dumps(_open_stream(headers=[["authorization", secret]])))
    assert secret not in str(excinfo.value)
    assert "snowman" not in str(excinfo.value)


def _open_stream(**overrides: Any) -> dict[str, Any]:
    frame = {
        "type": "open_stream",
        "stream_id": "s1",
        "node_id": FIXTURE_NODE_ID,
        "method": "GET",
        "path": "/api/status",
        "query": "",
        "headers": [["accept", "*/*"]],
        "role": "submitter",
    }
    frame.update(overrides)
    return frame


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("lowercase method", {"method": "get"}),
        ("overlong method", {"method": "A" * 17}),
        ("absolute destination", {"path": "http://evil.example/x"}),
        ("protocol-relative path", {"path": "//evil.example/x"}),
        ("relative path", {"path": "api/status"}),
        ("backslash path", {"path": "/a\\b"}),
        ("CR in path", {"path": "/a\rb"}),
        ("space in path", {"path": "/a b"}),
        ("fragment in query", {"query": "a=1#frag"}),
        ("newline in query", {"query": "a=1\nb=2"}),
        ("overlong query", {"query": "x" * 8193}),
        ("too many headers", {"headers": [["h", "v"]] * 101}),
        ("non-token header name", {"headers": [["bad header", "v"]]}),
        ("control char in header value", {"headers": [["x", "a\x00b"]]}),
        ("headers too large", {"headers": [["h", "v" * 700]] * 100}),
        ("headers not pairs", {"headers": [["only-one"]]}),
        ("bad inline body", {"body_b64": "%%%"}),
    ],
)
def test_open_stream_envelope_rules_reject_in_both_models(
    label: str, overrides: dict[str, Any]
) -> None:
    """Every §3 bound, refused by the runtime parser *and* by the oracle."""
    frame = _open_stream(**overrides)
    with pytest.raises(runtime.FrameError):
        runtime.parse_downlink(json.dumps(frame))
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        oracle.parse_inbound(frame)


def test_open_stream_accepts_the_maximum_legal_envelope() -> None:
    """The bounds are inclusive: exactly at the limit must still parse."""
    frame = _open_stream(headers=[["h", "v"]] * 100, query="x" * 8192)
    parsed = runtime.parse_downlink(json.dumps(frame))
    assert isinstance(parsed, runtime.OpenStream)
    assert len(parsed.headers) == 100
    assert oracle.parse_inbound(frame) is not None


def test_is_forwardable_path_agrees_with_the_oracle() -> None:
    for path in (
        "/",
        "/api/status",
        "/a%20b",
        "//evil",
        "http://evil/x",
        "/a\\b",
        "/a b",
        "/a\tb",
        "api",
        "",
    ):
        assert runtime.is_forwardable_path(path) == oracle.is_forwardable_path(path), path


# ---------------------------------------------------------------------------
# 5. redaction — the reason these are dataclasses and not dicts
# ---------------------------------------------------------------------------


def test_open_stream_repr_never_shows_headers_or_body() -> None:
    """§4.2: the relay injects ``Authorization: Bearer <capability>``.

    The repr is the structural guarantee that a stray ``logger.debug("%s",
    frame)``, an f-string or a traceback cannot spill it.
    """
    secret = "super-secret-capability"  # noqa: S105 - a literal, not a credential
    frame = runtime.parse_downlink(
        json.dumps(
            _open_stream(
                headers=[["authorization", f"Bearer {secret}"], ["x-nerdit-role", "submitter"]],
                body_b64=runtime.encode_body(b"private payload"),
            )
        )
    )
    for rendering in (repr(frame), str(frame), f"{frame}", f"{frame!r}"):
        assert secret not in rendering
        assert "Bearer" not in rendering
        assert "private payload" not in rendering
        assert "<redacted>" in rendering
    # The data itself is still reachable — redaction is about *rendering*.
    assert isinstance(frame, runtime.OpenStream)
    assert frame.header("Authorization") == f"Bearer {secret}"
    assert frame.body() == b"private payload"


def test_open_stream_header_lookup_is_case_insensitive_and_first_wins() -> None:
    frame = runtime.parse_downlink(
        json.dumps(_open_stream(headers=[["Accept", "first"], ["accept", "second"]]))
    )
    assert isinstance(frame, runtime.OpenStream)
    assert frame.header("ACCEPT") == "first"
    assert frame.header("missing") is None


def test_open_stream_role_stays_a_plain_string() -> None:
    """A bad role must PARSE so the mux can answer ``stream_error``.

    Rejecting it here would drop the stream silently and leave the requester
    hanging; the oracle narrows it to a literal precisely because it models the
    *wire*, not the daemon's error behaviour.
    """
    frame = runtime.parse_downlink(json.dumps(_open_stream(role="admin")))
    assert isinstance(frame, runtime.OpenStream)
    assert frame.role == "admin"
    with pytest.raises(Exception):  # noqa: B017 - the oracle narrows to submitter
        oracle.parse_inbound(_open_stream(role="admin"))


def test_bodiless_open_stream_decodes_to_empty_bytes() -> None:
    frame = runtime.parse_downlink(json.dumps(_open_stream()))
    assert isinstance(frame, runtime.OpenStream)
    assert frame.body_b64 is None
    assert frame.body() == b""


def test_empty_inline_body_is_distinguishable_from_an_absent_one() -> None:
    """Both are legal (§4.2) and mean different things to the mux."""
    frame = runtime.parse_downlink(json.dumps(_open_stream(method="POST", body_b64="")))
    assert isinstance(frame, runtime.OpenStream)
    assert frame.body_b64 == ""
    assert frame.body() == b""


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_heartbeat_is_a_parse_failure(bad: float) -> None:
    """PR #114 review: Python's JSON decoder accepts NaN/Infinity, and NaN
    slips through every positivity check and ``min()`` clamp (all comparisons
    False) straight into asyncio timer arithmetic. Non-finite numbers are a
    FrameError at the parse edge."""
    raw = load_fixture("hello_ack.json")
    payload = dict(raw)
    payload["heartbeat_interval_s"] = bad
    with pytest.raises(runtime.FrameError, match="finite"):
        runtime.parse_downlink(json.dumps(payload))


@pytest.mark.parametrize("field", ["stream_idle_timeout_s", "stream_absolute_timeout_s"])
def test_a_non_finite_timeout_limit_is_a_parse_failure(field: str) -> None:
    with pytest.raises(runtime.FrameError, match="finite"):
        runtime.StreamLimits.from_payload(_limits(**{field: float("nan")}))
