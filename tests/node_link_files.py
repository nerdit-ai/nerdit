"""Inventory of the vendored ``node-link/v1`` fixture files, by kind.

Shared between the protocol tests (``test_link_frames.py``), which exercise the
fixture JSONs, and the maintainer-side cross-repo conformance check, which pins
the vendored tree against its upstream. Split out so the protocol tests do not
import that check: its subject is the freeze machinery, which stays in the
private repository, while the fixture JSONs and the protocol tests are the wire
contract itself.
"""

#: The exchange fixtures: one ``open_stream`` plus the uplink frames answering it.
EXCHANGE_FILES = [
    "exchange_dashboard.json",
    "exchange_status.json",
    "exchange_events.json",
    "exchange_mcp.json",
    "exchange_mcp_tools_list.json",
    "exchange_mcp_tools_call.json",
]

#: Standalone lifecycle frames, by the direction that receives them.
INBOUND_FILES = ["node_challenge.json", "hello_ack.json"]
OUTBOUND_FILES = ["hello.json", "heartbeat.json"]

#: Phase 0 version-negotiation vectors (envelopes, not frames).
VERSION_FILES = ["handshake.json", "unsupported_version.json"]
