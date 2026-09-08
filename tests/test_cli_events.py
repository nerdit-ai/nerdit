"""Test event tables, forward pagination and streaming with a mocked client.

Map --type to comma-separated types and --since to since_id, never cursor.
Next-cursor hints must be directly reusable. Display feed.gap with retained_min_id
recovery guidance and end cleanly on feed.saturated. Escape hostile Rich markup
in tables and streams. Successful reads exit zero; unreachable daemons exit one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.app import app

runner = CliRunner()

# A value that is legal server output and illegal Rich markup.
MARKUP_BOMB = "[/x]"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Keep Rich from truncating table cells under a narrow test terminal."""
    monkeypatch.setenv("COLUMNS", "300")


EVENTS_BODY = {
    "items": [
        {
            "id": 12,
            "ts": "2026-08-07T03:12:44Z",
            "type": "service.failed",
            "kind": "service",
            "service_name": "my-app",
            "reason": "crash_loop",
            "build_version": 4,
            "data": {"restart_count": 3, "exit_code": 137},
        },
        {
            "id": 11,
            "ts": "2026-08-07T03:11:02Z",
            "type": "service.degraded",
            "kind": "service",
            "service_name": "my-app",
            "reason": None,
            "build_version": 4,
            "data": None,
        },
    ],
    "next_cursor": None,
}


def _client(**methods) -> AsyncMock:
    fake = AsyncMock()
    for name, value in methods.items():
        setattr(fake, name, AsyncMock(**value))
    return fake


def _streaming_client(frames: list[dict]) -> AsyncMock:
    """A client whose ``stream_events`` yields ``frames`` then completes."""
    fake = AsyncMock()

    async def _stream(*, last_event_id=None):
        fake.last_event_id = last_event_id
        for frame in frames:
            yield frame

    fake.stream_events = _stream
    return fake


# --- the bounded read ---------------------------------------------------------


def test_events_renders_the_feed():
    fake = _client(list_events={"return_value": EVENTS_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events"])
    assert result.exit_code == 0, result.output
    assert "service.failed" in result.output
    assert "my-app" in result.output
    assert "crash_loop" in result.output
    assert "exit_code=137" in result.output
    assert fake.list_events.await_args.kwargs == {
        "limit": 50,
        "types": None,
        "service": None,
        "since_id": None,
    }


def test_repeated_type_flags_collapse_into_the_comma_separated_filter():
    fake = _client(list_events={"return_value": EVENTS_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(
            app,
            [
                "events",
                "--type",
                "service.failed",
                "--type",
                "model.ready",
                "--service",
                "my-app",
                "--limit",
                "5",
            ],
        )
    assert result.exit_code == 0, result.output
    assert fake.list_events.await_args.kwargs == {
        "limit": 5,
        "types": "service.failed,model.ready",
        "service": "my-app",
        "since_id": None,
    }


def test_since_reaches_the_client_as_since_id_not_cursor():
    """The direction contract, at the CLI boundary.

    ``--since`` is the *forward* mode. A verb that sent it as ``cursor`` would
    hand a resuming script the events **older** than the id it already has —
    silently, and looking like success.
    """
    fake = _client(list_events={"return_value": dict(EVENTS_BODY, next_cursor="12")})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--since", "7"])
    assert result.exit_code == 0, result.output
    assert fake.list_events.await_args.kwargs["since_id"] == 7
    assert "cursor" not in fake.list_events.await_args.kwargs
    # And the hint is re-invocable verbatim, still in the forward mode.
    assert "nerdit events --since 12" in result.output


def test_empty_feed_says_so_instead_of_printing_a_bare_table():
    fake = _client(list_events={"return_value": {"items": [], "next_cursor": None}})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events"])
    assert result.exit_code == 0, result.output
    assert "No events recorded yet." in result.output


def test_unreachable_daemon_exits_one():
    fake = _client(list_events={"side_effect": httpx.ConnectError("refused")})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events"])
    assert result.exit_code == 1


def test_events_survives_markup_hostile_strings():
    body = {
        "items": [
            {
                "id": 1,
                "ts": MARKUP_BOMB,
                "type": MARKUP_BOMB,
                "service_name": MARKUP_BOMB,
                "reason": MARKUP_BOMB,
                "data": {"k": MARKUP_BOMB},
            }
        ],
        "next_cursor": None,
    }
    fake = _client(list_events={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events"])
    assert result.exit_code == 0, result.output
    assert MARKUP_BOMB in result.output


# --- --follow -----------------------------------------------------------------


def test_follow_renders_frames_and_passes_since_as_the_resume_cursor():
    fake = _streaming_client(
        [
            {},  # a heartbeat frame carries no type and must be skipped
            {
                "id": 8,
                "ts": "2026-08-07T03:12:44Z",
                "type": "service.failed",
                "service_name": "my-app",
                "reason": "crash_loop",
                "data": {"exit_code": 137},
            },
        ]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--follow", "--since", "7"])
    assert result.exit_code == 0, result.output
    assert fake.last_event_id == 7
    assert "service.failed" in result.output
    assert "exit_code=137" in result.output


def test_follow_renders_the_gap_frame_with_its_recovery_hint():
    """A ``feed.gap`` is the one frame that must never be quietly dropped."""
    fake = _streaming_client(
        [
            {"type": "feed.gap", "from": 2, "reason": "pruned", "retained_min_id": 5},
            {"id": 5, "type": "service.healthy", "service_name": "my-app"},
        ]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--follow"])
    assert result.exit_code == 0, result.output
    assert "feed.gap" in result.output
    assert "reason=pruned" in result.output
    assert "retained_min_id=5" in result.output
    # ``--since`` is EXCLUSIVE (id > N), so the hint is one BELOW the oldest
    # surviving row — printing 5 verbatim would skip row 5 itself.
    assert "nerdit events --since 4" in result.output
    assert "service.healthy" in result.output


def test_the_gap_recovery_hint_never_goes_negative_or_explodes():
    """``retained_min_id`` of 0 (empty feed) and a non-int both degrade safely."""
    from nerdit.cli.commands.events import _resume_hint

    assert _resume_hint(0) == "0"
    assert _resume_hint(1) == "0"
    assert _resume_hint(5) == "4"
    assert _resume_hint(None) == ""
    assert _resume_hint("odd") == "odd"


def test_follow_reports_a_saturated_daemon_and_stops():
    fake = _streaming_client(
        [
            {"type": "feed.saturated", "reason": "concurrency_limit", "limit": 32},
            {"id": 9, "type": "service.healthy", "service_name": "my-app"},
        ]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--follow"])
    assert result.exit_code == 0, result.output
    assert "feed.saturated" in result.output
    assert "32" in result.output
    # The stream ended at the saturation frame — nothing after it is rendered.
    assert "service.healthy" not in result.output


def test_follow_filters_client_side_but_never_hides_a_gap():
    fake = _streaming_client(
        [
            {"type": "feed.gap", "from": 1, "reason": "overflow", "retained_min_id": 1},
            {"id": 2, "type": "service.healthy", "service_name": "other"},
            {"id": 3, "type": "service.failed", "service_name": "my-app"},
        ]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--follow", "--service", "my-app"])
    assert result.exit_code == 0, result.output
    assert "feed.gap" in result.output  # never filtered away
    assert "service.failed" in result.output
    assert "other" not in result.output


def test_follow_survives_markup_hostile_strings():
    fake = _streaming_client(
        [{"id": 1, "ts": MARKUP_BOMB, "type": MARKUP_BOMB, "service_name": MARKUP_BOMB}]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--follow"])
    assert result.exit_code == 0, result.output
    assert MARKUP_BOMB in result.output


def test_follow_renders_a_stream_error():
    fake = AsyncMock()

    async def _stream(*, last_event_id=None):
        raise httpx.ConnectError("refused")
        yield {}  # pragma: no cover — unreachable, keeps this an async generator

    fake.stream_events = _stream
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["events", "--follow"])
    assert result.exit_code == 1
