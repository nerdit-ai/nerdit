"""Shared request limits and query-parameter normalization.

Route modules share these helpers without importing one another.
"""

from __future__ import annotations

from datetime import datetime, timezone

from nerdit.daemon.errors import NerditError

# Hard cap on a single service log request — BOTH branches, `tail` and the
# forward paged read — so a caller can't pull an unbounded log set into daemon
# memory (mirrors the jobs route). The forward branch applies it as the page
# size; a caller resumes past it with `since_id`.
_MAX_LOG_TAIL = 5000

# P13 WP3 — the `/wait` converge primitive.
# Clamp bound for the `timeout` query param (seconds).
_WAIT_TIMEOUT_MAX = 300

# Global concurrency cap. A saturated daemon resolves the request immediately
# as `timeout, waited_s=0` (agent-branchable) rather than holding unbounded
# 300 s connections — a cheap connection-exhaustion vector on a LAN daemon.
# Surfaced in `/capabilities.limits.wait_concurrency_max`.
WAIT_CONCURRENCY_MAX = 64

# --- /diagnose -----------------------------------------------------

# Hard cap on the log tail bundled into a single `/diagnose` response (§1.3).
_MAX_DIAGNOSE_TAIL = 200

# --- Durable event feed -------------------------------------------

# Hard cap on the number of rows one `/events/stream` resume replays before it
# gives up and tells the client the truth. A client that cannot be replayed
# exactly receives a synthetic `feed.gap` frame FIRST and is told to reconcile
# via `GET /events?since_id=…` — never a silently truncated replay.
MAX_SSE_REPLAY = 1000

#: Back-compat alias for the pre-`/capabilities` private name. Kept so the
#: route module and its tests keep importing what they always did; the public
#: spelling above is the one `/capabilities.limits` projects.
_MAX_SSE_REPLAY = MAX_SSE_REPLAY

# Global concurrency cap on `/events/stream`. Mirrors `WAIT_CONCURRENCY_MAX`
# and for the same reason: an SSE connection is held open indefinitely, so an
# uncapped stream route is a cheap connection-exhaustion vector on a LAN daemon.
# A saturated daemon emits one `feed.saturated` frame and closes, which is
# agent-branchable, rather than accepting a connection it cannot serve.
EVENTS_STREAM_CONCURRENCY_MAX = 32

# --- Audit & log queryability -------------------------------------

# NB: the `grep` needle clamp is NOT here. It lives at the one place that
# applies it (`db/queries/logs._MAX_GREP_LEN`), so a caller reaching the query
# layer directly is bounded by the same number the route is — two constants
# would only mean two chances to drift.

# Grammar for the `action_prefix` audit filter. Every action the daemon
# writes is built from lowercase words joined by `.` (`deploy.git_create`,
# `service.run`), so this is the full alphabet of the column. Belt and braces
# on top of the `GLOB` matching in the query layer: it keeps `*`/`?`/`[`
# — the characters GLOB *does* treat as wildcards — out of the argument
# entirely, so the prefix is literal at both layers.
_ACTION_PREFIX_MAX = 40

# Global concurrency cap on `/services/{ident}/logs/stream`. Its own object,
# never shared with `EVENTS_STREAM_CONCURRENCY_MAX`: the two streams are
# independent resources and one saturating the other would be a surprise.
LOGS_STREAM_CONCURRENCY_MAX = 32


# The format SQLite's `datetime('now')` writes — the shape `audit_log.ts`,
# `job_logs.timestamp` and `events.ts` are all stored in. Every producer of a
# value that will be COMPARED against one of those columns (the query-param
# normalizer below, the retention sweep cutoffs) formats through this one
# constant: a second literal is how two time filters silently start meaning two
# different things.
STORED_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def valid_ts_filter(value: str, field: str) -> str:
    """Normalize ISO-8601 input to SQLite's YYYY-MM-DD HH:MM:SS UTC format.

    Convert aware timestamps to UTC and treat naive ones as UTC. Reject invalid
    input instead of silently dropping a filter. Matching stored formatting is
    required because database comparisons are lexical.
    """
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise NerditError(
            400,
            "bad_request",
            f"Invalid {field} timestamp.",
            hint="Use ISO-8601, e.g. 2026-08-07T10:00:00Z or 2026-08-07.",
        ) from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.strftime(STORED_TS_FORMAT)


# --- One-off runs ------------------------------------------------------

# Hard cap on the `log_tail` a run response carries back (the route clamps the
# request's value to it). Mirrors `_MAX_DIAGNOSE_TAIL`: run stdout never
# reaches `job_logs` (D-P14-6), so this bound and the byte budget enforced at
# capture time in `core/` are the only things standing between a chatty
# command and an unbounded response body.
_MAX_RUN_LOG_TAIL = 200
