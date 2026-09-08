"""Unit tests for :func:`nerdit.daemon.audit.record_out_of_band` (Track B WP14).

``record_out_of_band`` is the single collapsed implementation behind the three
request-attributed out-of-band audit clones (``record_shared_referenced``,
``routes/services.py:_audit_purge``, ``routes/databases.py:_audit_credential_minted``).
Those call sites are already exercised end-to-end (unmodified) by
``test_audit.py`` and ``test_delete_purge.py``; this file pins the shared
primitive directly: the row shape it inserts, and that an ``insert_audit_log``
failure is swallowed (best-effort — a logging failure must never mask the
caller's own response).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from starlette.requests import Request

from nerdit.daemon.audit import record_out_of_band
from nerdit.daemon.auth import LEGACY_ADMIN


def _make_request(queries) -> Request:  # noqa: ANN001
    scope = {
        "type": "http",
        "method": "DELETE",
        "path": "/services/demo",
        "raw_path": b"/services/demo",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("test", 1234),
        "app": SimpleNamespace(state=SimpleNamespace(queries=queries)),
    }
    req = Request(scope)
    req.state.request_id = "rid-oob-1"
    req.state.principal = LEGACY_ADMIN
    return req


async def test_record_out_of_band_inserts_expected_row():
    queries = AsyncMock()
    queries.insert_audit_log = AsyncMock()
    req = _make_request(queries)

    await record_out_of_band(
        req,
        action="service.purge_data",
        target_type="service",
        target_id="demo",
        params={"key": "services/demo", "purged": True},
    )

    queries.insert_audit_log.assert_awaited_once()
    kwargs = queries.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "service.purge_data"
    assert kwargs["result"] == "ok"
    assert kwargs["target_type"] == "service"
    assert kwargs["target_id"] == "demo"
    assert kwargs["principal_id"] == LEGACY_ADMIN.token_id
    assert kwargs["principal_role"] == LEGACY_ADMIN.role.value
    assert json.loads(kwargs["params_redacted"]) == {"key": "services/demo", "purged": True}
    assert kwargs["request_id"] == "rid-oob-1"


async def test_record_out_of_band_swallows_insert_failure():
    queries = AsyncMock()
    queries.insert_audit_log = AsyncMock(side_effect=RuntimeError("db unavailable"))
    req = _make_request(queries)

    # Must not raise — a logging failure never masks the caller's own response.
    await record_out_of_band(
        req,
        action="database.credential_minted",
        target_type="database",
        target_id="demo-db",
        params={"service": "demo-db", "keys": ["POSTGRES_PASSWORD"]},
    )

    queries.insert_audit_log.assert_awaited_once()
