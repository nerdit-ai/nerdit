"""Route tests for the P13b WP6 System surface — ``/capabilities``,
``/doctor``, ``POST /daemon/restart``.

Same harness as ``test_routes_proxy_ca.py``: a hand-built FastAPI app with
mocked state on ``app.state`` (no real aiosqlite crossing event loops), the
real auth + error middleware, and — for the restart drain — an httpx
``ASGITransport`` so the fire-and-forget drain task runs on the test's own loop
(``os.kill`` and ``os.execv`` are ALWAYS mocked; the process is never signalled).
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import sqlite3
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.config.defaults import DEFAULT_DB_NAME
from nerdit.core.dns import MDNS_REASON_ZEROCONF_MISSING
from nerdit.core.license import (
    LICENSE_GRACE_S,
    LicenseClaims,
    LicenseState,
    LicenseVerdict,
)
from nerdit.core.proxy import ProxyState
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes import system as system_routes
from nerdit.daemon.routes.system import router as system_router
from nerdit.db.models import TokenRole

_ADMIN = {"Authorization": "Bearer admin-raw-token"}
_SUBMITTER = {"Authorization": "Bearer submitter-raw-token"}
_READONLY = {"Authorization": "Bearer readonly-raw-token"}


@pytest.fixture(autouse=True)
def _reset_restart_flag():
    """The restart flag is a module global — clear it around every test."""
    system_routes.reset_restart_requested()
    yield
    system_routes.reset_restart_requested()


def _token_row(
    role: TokenRole,
    *,
    expires_at: datetime | None = None,
    scope_services: list[str] | None = None,
) -> SimpleNamespace:
    # (P25) ``expires_at``/``scope_services`` are read by the auth middleware on
    # every scoped-token request, so the stand-in row must carry both.
    return SimpleNamespace(
        id=f"tok-{role.value}",
        name="ci",
        role=role,
        max_gpus=2,
        max_concurrent_jobs=4,
        expires_at=expires_at,
        scope_services=scope_services,
    )


def _queries_for(
    role: TokenRole | None = None,
    *,
    expires_at: datetime | None = None,
    scope_services: list[str] | None = None,
) -> SimpleNamespace:
    """A queries stand-in whose token lookup returns a scoped row of *role*.

    ``None`` ⇒ unknown token (403). The legacy ``admin-raw-token`` never reaches
    the lookup (it short-circuits to the LEGACY_ADMIN principal).
    """
    row = (
        _token_row(role, expires_at=expires_at, scope_services=scope_services)
        if role is not None
        else None
    )
    return SimpleNamespace(
        get_api_token_by_hash=AsyncMock(return_value=row),
        get_api_token_by_id=AsyncMock(return_value=row),
        touch_api_token=AsyncMock(),
        insert_audit_log=AsyncMock(),
    )


# --- GET /capabilities --------------------------------------------------------


def _caps_settings() -> SimpleNamespace:
    return SimpleNamespace(
        data_dir="/home/u/.nerdit",
        proxy=SimpleNamespace(
            enabled=True,
            mode="path",
            base_domain=None,
            hostname_override=None,
            scheme="https",
            https_port=443,
            public_port=None,
            dashboard_apex=True,
            mdns=True,
            admin_addr="localhost:2019",
        ),
        models=SimpleNamespace(default_backend="ollama", bridge_host="172.17.0.1"),
        git=SimpleNamespace(enabled=True, allowed_hosts=["github.com"]),
        daemon=SimpleNamespace(max_upload_bytes=524288000),
        services=SimpleNamespace(max_concurrent_builds=2, service_port_range="8100-8199"),
        containers=SimpleNamespace(
            drop_all_caps=True, no_new_privileges=True, read_only_rootfs=False
        ),
    )


def _caps_app(
    *,
    role: TokenRole | None,
    expires_at: datetime | None = None,
    scope_services: list[str] | None = None,
) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.settings = _caps_settings()
    app.state.started_at = datetime.now(UTC) - timedelta(seconds=12)
    app.state.hostname = "nerd-box.local"
    app.state.proxy_manager = SimpleNamespace(available=True)
    app.state.model_controller = SimpleNamespace(
        backends=["ollama", "vllm"],
        default_backend_name="ollama",
        bridge_host="172.17.0.1",
    )
    app.state.data_controller = SimpleNamespace(
        backends=["postgres", "redis"],
        default_backend_name="postgres",
        bridge_host="172.17.0.1",
    )
    app.state.shared_scope_blocked = False
    app.state.gpu_snapshot = {"count": 1, "schedulable": 1, "vendors": ["nvidia"]}
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)
    q = _queries_for(role, expires_at=expires_at, scope_services=scope_services)
    # Exposed so a test can prove ``/capabilities`` never reads a row (P25 WP3).
    app.state.queries = q
    return TestClient(app)


def test_capabilities_requires_auth():
    client = _caps_app(role=None)
    assert client.get("/api/capabilities").status_code == 401


def test_capabilities_admin_has_paths_and_admin_addr():
    client = _caps_app(role=None)  # legacy token → admin principal
    body = client.get("/api/capabilities", headers=_ADMIN).json()

    # Admin-only surfaces present.
    assert body["proxy"]["admin_addr"] == "localhost:2019"
    assert body["paths"] == {
        "data_dir": "/home/u/.nerdit",
        "db_path": "/home/u/.nerdit/nerdit.db",
    }
    # Shared fields.
    assert body["version"]
    assert body["uptime_s"] >= 0
    assert body["caller"]["role"] == "admin"
    assert body["proxy"]["url_shape"] == "https://nerd-box.local/<name>/"
    assert body["models"]["backends"] == ["ollama", "vllm"]
    # (P15) The managed-data plane block mirrors the models block. Both backends
    # are registered after the P15.5 registry proof (Postgres default + Redis).
    assert body["databases"] == {
        "backends": ["postgres", "redis"],
        "default_backend": "postgres",
        "bridge_host": "172.17.0.1",
    }
    assert body["buildpacks"] == ["dockerfile", "python", "node"]
    assert body["deploy"]["git_allowed_hosts"] == ["github.com"]
    assert body["deploy"]["build_settings"] == {
        "version": 1,
        "node_versions": ["22.23.2", "24.20.0"],
        "presets": ["node", "nextjs", "python", "dockerfile"],
        "fields": [
            "preset",
            "install",
            "build",
            "start",
            "node_version",
            "package_manager",
            "subdir",
            "public_env",
        ],
        "public_env": True,
        "secret_mounts": False,
    }
    assert body["limits"]["wait_concurrency_max"] == 64
    # (P24a) The stream budgets an agent needs BEFORE it opens the N+1st
    # follower and gets a ``*.saturated`` frame instead of a stream.
    assert body["limits"]["events_stream_concurrency_max"] == 32
    assert body["limits"]["logs_stream_concurrency_max"] == 32
    assert body["limits"]["events_stream_replay_max"] == 1000
    assert body["mcp"] == {"http_enabled": False}
    assert body["features"]["secrets_shared_scope"] is True
    # (PR1) Constant true on this build — lets a frontend distinguish a PR1+
    # daemon from a pre-PR1 one (FastAPI silently ignores unknown query params).
    assert body["features"]["audit_target_filter"] is True
    # (P24c) Enabled-flag only — a settings object without [notifications]
    # (this harness) reports the shipped-dark default.
    assert body["features"]["notifications"] is False
    # (P37) Constant true on this build, for the ``audit_target_filter`` reason:
    # an agent probing ``POST /databases/x/dump`` cannot tell "this daemon
    # predates dumps" from "no such database" (both are a plain 404), so the
    # capability has to be readable rather than discoverable.
    assert body["features"]["database_dumps"] is True


def test_capabilities_projects_the_container_sandbox():
    """(P33) An agent must be able to READ the hardening its image has to survive.

    The field failure this closes: an agent deployed a stock ``FROM nginx``
    static site, whose root entrypoint died on ``chown(...) Operation not
    permitted`` under ``cap_drop=["ALL"]``. Visible to every role — it names a
    policy, not a credential.
    """
    body = _caps_app(role=None).get("/api/capabilities", headers=_ADMIN).json()
    assert body["sandbox"] == {
        "drop_all_caps": True,
        "no_new_privileges": True,
        "read_only_rootfs": False,
    }


@pytest.mark.parametrize("headers, role", [(_SUBMITTER, "submitter"), (_READONLY, "readonly")])
def test_capabilities_sandbox_is_visible_to_every_role(headers, role):
    client = _caps_app(role=TokenRole.submitter if role == "submitter" else TokenRole.readonly)
    body = client.get("/api/capabilities", headers=headers).json()
    assert body["sandbox"]["drop_all_caps"] is True


def test_capabilities_sandbox_follows_settings():
    """A pure settings projection — an operator who loosens it sees the truth."""
    client = _caps_app(role=None)
    client.app.state.settings.containers = SimpleNamespace(
        drop_all_caps=False, no_new_privileges=False, read_only_rootfs=True
    )
    body = client.get("/api/capabilities", headers=_ADMIN).json()
    assert body["sandbox"] == {
        "drop_all_caps": False,
        "no_new_privileges": False,
        "read_only_rootfs": True,
    }


def test_capabilities_sandbox_defaults_without_a_containers_section():
    """A lightweight settings object still projects the shipped defaults."""
    client = _caps_app(role=None)
    del client.app.state.settings.containers
    body = client.get("/api/capabilities", headers=_ADMIN).json()
    assert body["sandbox"] == {
        "drop_all_caps": True,
        "no_new_privileges": True,
        "read_only_rootfs": False,
    }


def test_capabilities_notifications_flag_follows_settings():
    """The flag is a pure settings projection — and NEVER carries a target."""
    client = _caps_app(role=None)
    client.app.state.settings.notifications = SimpleNamespace(
        enabled=True, targets=[SimpleNamespace(url="https://hook.example/x")]
    )
    body = client.get("/api/capabilities", headers=_ADMIN).json()

    assert body["features"]["notifications"] is True
    assert "hook.example" not in json.dumps(body)


def test_capabilities_token_block_projects_the_principal():
    """(P25 D-P25-4 / D-P25-10) The agent's pre-expiry self-knowledge path."""
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    client = _caps_app(
        role=TokenRole.submitter, expires_at=expires_at, scope_services=["api", "worker"]
    )
    body = client.get("/api/capabilities", headers=_SUBMITTER).json()

    token = body["token"]
    assert token["role"] == "submitter"
    assert token["expires_at"].startswith(expires_at.isoformat()[:16])
    assert 3500 < token["expires_in_s"] <= 3600
    assert token["scope_services"] == ["api", "worker"]
    assert token["rotatable"] is True
    # The block is a projection, never a credential channel.
    assert set(token) == {"role", "expires_at", "expires_in_s", "scope_services", "rotatable"}


def test_capabilities_token_block_costs_zero_db_reads():
    """The route's docstring promises a pure ``app.state`` read — keep it true.

    The regression this pins: fetching the row here to enrich the block would
    make ``/capabilities`` do I/O on every call, on the hottest agent path.
    """
    client = _caps_app(role=TokenRole.submitter, expires_at=datetime.now(UTC) + timedelta(days=2))
    q = client.app.state.queries
    assert client.get("/api/capabilities", headers=_SUBMITTER).status_code == 200

    q.get_api_token_by_id.assert_not_awaited()
    # Exactly one lookup: the auth middleware resolving the bearer.
    assert q.get_api_token_by_hash.await_count == 1


def test_capabilities_token_block_for_a_sentinel_principal_is_not_rotatable():
    client = _caps_app(role=None)  # legacy global token → LEGACY_ADMIN
    token = client.get("/api/capabilities", headers=_ADMIN).json()["token"]

    assert token == {
        "role": "admin",
        "expires_at": None,
        "expires_in_s": None,
        "scope_services": None,
        "rotatable": False,
    }


@pytest.mark.parametrize(
    "headers,role",
    [(_SUBMITTER, TokenRole.submitter), (_READONLY, TokenRole.readonly)],
)
def test_capabilities_non_admin_omits_paths_and_admin_addr(headers, role):
    client = _caps_app(role=role)
    body = client.get("/api/capabilities", headers=headers).json()

    # Both admin-only surfaces are ABSENT (not nulled).
    assert "paths" not in body
    assert "admin_addr" not in body["proxy"]
    # But git allowed_hosts stays visible to every role.
    assert body["deploy"]["git_allowed_hosts"] == ["github.com"]
    assert body["caller"]["role"] == role.value
    assert body["caller"]["quotas"] == {"max_gpus": 2, "max_concurrent_jobs": 4}


# --- GET /doctor --------------------------------------------------------------


class _StubRuntimeMarker:
    """A stand-in for the real StubRuntime; the check uses isinstance so we
    patch the symbol the route imported instead of subclassing."""


def _doctor_app(*, auth_role: TokenRole | None = None, **state) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    settings = state.pop("settings", None) or _doctor_settings()
    app.state.settings = settings
    for key, value in _doctor_defaults(settings).items():
        setattr(app.state, key, value)
    for key, value in state.items():
        setattr(app.state, key, value)
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)
    # auth_role=None ⇒ authenticate via the legacy admin token; a scoped role
    # exercises the non-admin path (doctor details are role-agnostic).
    q = _queries_for(auth_role)
    return TestClient(app)


def _make_real_db_dir() -> str:
    """A temp dir holding a real, integrity-clean sqlite db.

    The ``_db`` check now opens a dedicated read-only connection against the file
    on disk (not the mocked shared connection), so the harness needs a real db
    for the check to pass ``quick_check``.
    """
    d = tempfile.mkdtemp(prefix="nerdit-doctor-db-")
    conn = sqlite3.connect(str(Path(d) / DEFAULT_DB_NAME))
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    return d


# Created once for the module — a valid db file the _db probe can quick_check.
_REAL_DB_DIR = _make_real_db_dir()


def _make_real_key_file() -> Path:
    """A real key file on disk for the default harness manager.

    The ``secrets_key`` probe classifies the key path from ``lstat`` mode bits
    (so a directory or a dangling symlink cannot masquerade as "absent"), which
    a stub answering only ``is_file()`` cannot satisfy.
    """
    path = Path(tempfile.mkdtemp(prefix="nerdit-doctor-key-")) / "secrets.key"
    path.write_text("ab" * 32 + "\n", encoding="utf-8")
    return path


_REAL_KEY_PATH = _make_real_key_file()


def _doctor_settings(tmp: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir=tmp or _REAL_DB_DIR,
        proxy=SimpleNamespace(mdns=False),
        git=SimpleNamespace(enabled=False),
    )


def _doctor_defaults(settings) -> dict:
    db_cursor = SimpleNamespace(fetchone=AsyncMock(return_value=("ok",)))
    db = SimpleNamespace(conn=SimpleNamespace(execute=AsyncMock(return_value=db_cursor)))
    return {
        "settings": settings,
        # (BUG-1) The docker row now reads BuildKit availability too; the
        # default harness runtime reports a healthy node.
        "runtime": SimpleNamespace(
            list_images=AsyncMock(return_value=[]),
            buildx_available=AsyncMock(return_value="present"),
        ),
        "gpu_snapshot": {"count": 1, "schedulable": 1, "vendors": ["nvidia"]},
        "proxy_manager": SimpleNamespace(state=ProxyState.available, available=True),
        "mdns_advertiser": SimpleNamespace(registered=False),
        "secret_manager": SimpleNamespace(
            key_path=_REAL_KEY_PATH,
            has_ciphertexts=lambda: True,
            self_check=lambda: True,
        ),
        "db": db,
    }


def _check(body: dict, name: str) -> dict:
    return next(c for c in body["checks"] if c["name"] == name)


def test_doctor_docker_stub_runtime_fails(monkeypatch):
    # Patch the isinstance target the route imported, then hand it a marker.
    monkeypatch.setattr(system_routes, "StubRuntime", _StubRuntimeMarker)
    client = _doctor_app(runtime=_StubRuntimeMarker())
    body = client.get("/api/doctor", headers=_ADMIN).json()
    docker = _check(body, "docker")
    assert docker["status"] == "fail"
    assert "stub runtime" in docker["detail"]
    assert body["status"] == "fail"  # worst-of


def _runtime_with_buildx(verdict: str) -> SimpleNamespace:
    return SimpleNamespace(
        list_images=AsyncMock(return_value=[]),
        buildx_available=AsyncMock(return_value=verdict),
    )


def test_doctor_docker_warns_when_buildx_is_missing():
    """(BUG-1) leg A: Docker up but no BuildKit builder ⇒ warn, never fail.

    ``warn``, not ``fail``: a node that only runs pre-built images is fully
    functional without buildx, so failing its health gate would be dishonest.
    The hard signal lives in ``check-deps`` and in the build's own
    PLATFORM_ERROR classification.
    """
    client = _doctor_app(runtime=_runtime_with_buildx("missing"))
    body = client.get("/api/doctor", headers=_ADMIN).json()
    docker = _check(body, "docker")
    assert docker["status"] == "warn"
    assert "buildx" in docker["detail"]
    assert "every image build will fail" in docker["detail"]
    assert body["status"] != "fail"


def test_doctor_docker_is_ok_when_buildx_is_present():
    """Leg B: the builder is there and the row says so."""
    client = _doctor_app(runtime=_runtime_with_buildx("present"))
    docker = _check(client.get("/api/doctor", headers=_ADMIN).json(), "docker")
    assert docker["status"] == "ok"
    assert "BuildKit builder available" in docker["detail"]


def test_doctor_docker_warns_when_the_daemon_has_no_docker_cli():
    """(Codex 3804646811) A daemon with no ``docker`` CLI on its PATH.

    Distinct from ``unknown``: this is a concluded, definite host fault — every
    image build on this node will fail — so it must not fall through to the
    fail-open ``ok``. Still ``warn`` and not ``fail``, for the same recorded
    reason as the buildx branch: a node serving only pre-built images is healthy.
    """
    client = _doctor_app(runtime=_runtime_with_buildx("no_cli"))
    body = client.get("/api/doctor", headers=_ADMIN).json()
    docker = _check(body, "docker")
    assert docker["status"] == "warn"
    assert "docker CLI" in docker["detail"]
    assert "every image build will fail" in docker["detail"]
    assert body["status"] != "fail"


def test_doctor_docker_stays_ok_when_the_buildx_probe_is_unknown():
    """An inconclusive probe never worsens an otherwise-healthy runtime row."""
    client = _doctor_app(runtime=_runtime_with_buildx("unknown"))
    docker = _check(client.get("/api/doctor", headers=_ADMIN).json(), "docker")
    assert docker["status"] == "ok"
    assert "unknown" in docker["detail"]


def test_doctor_docker_detail_carries_no_filesystem_path():
    """Doctor discipline: details are path-free for every role.

    The path-bearing guidance (``~/.docker/cli-plugins``, the system-wide
    plugin dirs) lives in ``check-deps`` remediation, a local operator surface.
    """
    for verdict in ("missing", "present", "unknown", "no_cli"):
        client = _doctor_app(runtime=_runtime_with_buildx(verdict))
        detail = _check(client.get("/api/doctor", headers=_ADMIN).json(), "docker")["detail"]
        assert "/" not in detail


def test_doctor_disk_thresholds(monkeypatch):
    # warn: < 10% free
    monkeypatch.setattr(
        system_routes.shutil, "disk_usage", lambda _p: SimpleNamespace(total=100, free=8)
    )
    body = _doctor_app().get("/api/doctor", headers=_ADMIN).json()
    assert _check(body, "disk")["status"] == "warn"

    # fail: < 5% free
    monkeypatch.setattr(
        system_routes.shutil, "disk_usage", lambda _p: SimpleNamespace(total=100, free=4)
    )
    body = _doctor_app().get("/api/doctor", headers=_ADMIN).json()
    assert _check(body, "disk")["status"] == "fail"

    # ok: plenty free
    monkeypatch.setattr(
        system_routes.shutil, "disk_usage", lambda _p: SimpleNamespace(total=100, free=80)
    )
    body = _doctor_app().get("/api/doctor", headers=_ADMIN).json()
    assert _check(body, "disk")["status"] == "ok"


def test_doctor_data_dir_perms_warns_on_group_world_access(tmp_path):
    """D-P14-3: /doctor warns when data_dir is group/world-accessible."""
    import os

    os.chmod(tmp_path, 0o755)  # group+world readable/executable
    settings = _doctor_settings(str(tmp_path))
    body = _doctor_app(settings=settings).get("/api/doctor", headers=_ADMIN).json()
    check = _check(body, "data_dir_perms")
    assert check["status"] == "warn"
    # Mode bits only — no path leaked into the detail.
    assert str(tmp_path) not in check["detail"]


def test_doctor_data_dir_perms_ok_when_private(tmp_path):
    """A 0o700 data_dir passes the perms check."""
    import os

    os.chmod(tmp_path, 0o700)
    settings = _doctor_settings(str(tmp_path))
    body = _doctor_app(settings=settings).get("/api/doctor", headers=_ADMIN).json()
    assert _check(body, "data_dir_perms")["status"] == "ok"


# --- GET /doctor: the acme_http_port row (P26 WP2 / S-W2-8) -------------------
#
# The one listener this daemon asks the OUTSIDE world to reach. Four verdicts,
# and the three failing ones are the difference between "a domain is stuck
# ``pending``" and knowing why. The bind probe is driven through a fake socket
# module so every errno branch is deterministic on every platform (a real
# privileged-port bind is root-dependent and a real EADDRINUSE needs a second
# process); one leg still binds for real to pin that the production code path
# works against the stdlib.


class _FakeSocket:
    """A stand-in for ``socket.socket`` that binds however the test says.

    Doubles as the module: the probe looks up ``socket.socket``/``AF_INET`` as
    module globals on the route module, so one object can serve as both and
    also record that the descriptor was closed on every path.
    """

    AF_INET = socket.AF_INET
    SOCK_STREAM = socket.SOCK_STREAM

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.bound: tuple[str, int] | None = None
        self.closed = False

    def socket(self, family, kind):  # noqa: ANN001 - mirrors the stdlib signature
        assert family is socket.AF_INET
        assert kind is socket.SOCK_STREAM
        return self

    def bind(self, address: tuple[str, int]) -> None:
        self.bound = address
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.closed = True


def _acme_settings(*, enabled: bool = True, http_port: int = 80) -> SimpleNamespace:
    settings = _doctor_settings()
    settings.proxy.acme = SimpleNamespace(enabled=enabled, http_port=http_port)
    return settings


def _acme_row(client: TestClient) -> dict:
    return _check(client.get("/api/doctor", headers=_ADMIN).json(), "acme_http_port")


def test_doctor_acme_is_skipped_when_the_section_is_absent():
    """A settings object predating [proxy.acme] reads as off, never raises.

    The ``[link]``/``[mcp]`` tolerance precedent — ``_doctor_settings`` builds
    exactly such a namespace, so this is the default harness path.
    """
    row = _acme_row(_doctor_app())
    assert row["status"] == "skipped"
    assert row["detail"] == "ACME disabled"


def test_doctor_acme_is_skipped_when_disabled():
    """An explicitly disabled section reads the same as an absent one.

    ``skipped`` is absent from ``_STATUS_RANK`` by design, so a node that never
    asked for ACME can never be pushed off ``ok`` by this row.
    """
    body = (
        _doctor_app(settings=_acme_settings(enabled=False))
        .get("/api/doctor", headers=_ADMIN)
        .json()
    )
    assert _check(body, "acme_http_port")["status"] == "skipped"
    assert "skipped" not in system_routes._STATUS_RANK


def test_doctor_acme_ok_when_the_proxy_is_available(monkeypatch):
    """An AVAILABLE proxy IS the bind fact — and the probe must not run.

    Caddy refuses to start when any configured listener fails to bind, so an
    available proxy with the ``:<http_port>`` server in its bootstrap has
    already proven the port. Probing here would collide with our own listener
    and report EADDRINUSE against ourselves — the fake socket is installed
    precisely to prove nothing touches it.
    """
    fake = _FakeSocket()
    monkeypatch.setattr(system_routes, "socket", fake)
    client = _doctor_app(
        settings=_acme_settings(http_port=8080),
        proxy_manager=SimpleNamespace(state=ProxyState.available, available=True),
    )

    row = _acme_row(client)
    assert row["status"] == "ok"
    assert row["detail"] == "http port 8080 bound by the embedded proxy"
    assert fake.bound is None


def test_doctor_acme_warns_when_the_port_is_free_but_the_proxy_is_not_up(monkeypatch):
    """Bindable + no proxy ⇒ the port is fine and the fault is the proxy's.

    ``warn``, not ``fail``: nothing about the ACME listener is broken, so the
    row points at the check that owns the real problem instead of raising a
    second alarm for the same fault.
    """
    fake = _FakeSocket()
    monkeypatch.setattr(system_routes, "socket", fake)
    client = _doctor_app(
        settings=_acme_settings(http_port=8080),
        proxy_manager=SimpleNamespace(state=ProxyState.backoff, available=False),
    )

    row = _acme_row(client)
    assert row["status"] == "warn"
    assert "http port 8080 is bindable" in row["detail"]
    assert "see the proxy check" in row["detail"]
    assert fake.bound == ("", 8080)
    assert fake.closed is True


def test_doctor_acme_fails_on_a_privileged_port_without_the_capability(monkeypatch):
    """The common ``:80`` failure, and the only one with two real fixes.

    ``PermissionError`` is what CPython raises for EACCES, so the branch is
    keyed on the exception type rather than the errno. The detail names both
    ways out — a capability on the binary or an unprivileged port — and is
    careful about WHOSE inability it just observed (review round 1): the probe
    runs in the daemon process and cannot see the caddy binary's file
    capabilities, so on a --user install where caddy already has
    CAP_NET_BIND_SERVICE it must not send the operator to re-grant it.
    """
    fake = _FakeSocket(PermissionError(errno.EACCES, "Permission denied"))
    monkeypatch.setattr(system_routes, "socket", fake)
    client = _doctor_app(
        settings=_acme_settings(http_port=80),
        proxy_manager=SimpleNamespace(state=ProxyState.no_binary, available=False),
    )

    row = _acme_row(client)
    assert row["status"] == "fail"
    assert "this daemon cannot bind http port 80" in row["detail"]
    assert "already has CAP_NET_BIND_SERVICE" in row["detail"]
    assert "see the proxy check" in row["detail"]
    assert "[proxy.acme].http_port" in row["detail"]
    assert fake.closed is True


def test_doctor_acme_fails_when_another_process_holds_the_port(monkeypatch):
    """EADDRINUSE with the proxy in BACKOFF means somebody else owns ``:80``."""
    fake = _FakeSocket(OSError(errno.EADDRINUSE, "Address already in use"))
    monkeypatch.setattr(system_routes, "socket", fake)
    client = _doctor_app(
        settings=_acme_settings(http_port=80),
        proxy_manager=SimpleNamespace(state=ProxyState.backoff, available=False),
    )

    row = _acme_row(client)
    assert row["status"] == "fail"
    assert row["detail"] == "http port 80 is held by another process"
    assert fake.closed is True


def test_doctor_acme_warns_while_the_proxy_is_still_starting(monkeypatch):
    """(review round 1) ``starting`` means Caddy is spawned and has very likely
    already bound the port. Probing then reports EADDRINUSE against OURSELVES and
    calls it "another process" — wrong, and a ``fail`` that drags the top-level
    doctor status down on every boot with ACME on. Say "not yet confirmed" and do
    not probe at all."""
    fake = _FakeSocket(OSError(errno.EADDRINUSE, "Address already in use"))
    monkeypatch.setattr(system_routes, "socket", fake)
    client = _doctor_app(
        settings=_acme_settings(http_port=80),
        proxy_manager=SimpleNamespace(state=ProxyState.starting, available=False),
    )

    row = _acme_row(client)
    assert row["status"] == "warn"
    assert "proxy is starting" in row["detail"]
    assert "another process" not in row["detail"]
    assert fake.bound is None


def test_doctor_acme_fails_when_the_adopted_proxy_carries_no_acme_listener():
    """(review round 1) An available proxy is the bind fact only for a Caddy THIS
    process spawned. One adopted from a pre-ACME run carries the ``nerdit``
    server (so adoption succeeds) and no ``nerdit-acme-http`` at all — the
    manager latches that, and the row must report it instead of inferring ``ok``
    from availability."""
    client = _doctor_app(
        settings=_acme_settings(http_port=80),
        proxy_manager=SimpleNamespace(
            state=ProxyState.available, available=True, acme_listener_live=False
        ),
    )

    row = _acme_row(client)
    assert row["status"] == "fail"
    assert "carries no ACME listener" in row["detail"]
    assert "restart the daemon" in row["detail"]


def test_doctor_acme_names_the_errno_on_any_other_bind_failure(monkeypatch):
    """An unmodelled OSError still yields a machine-readable token, not a shrug.

    ``_run_check``'s blanket handler would answer "check failed or timed out",
    which teaches an operator nothing and reads like a daemon bug.
    """
    fake = _FakeSocket(OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address"))
    monkeypatch.setattr(system_routes, "socket", fake)
    client = _doctor_app(
        settings=_acme_settings(http_port=8080),
        proxy_manager=SimpleNamespace(state=ProxyState.backoff, available=False),
    )

    row = _acme_row(client)
    assert row["status"] == "fail"
    assert "EADDRNOTAVAIL" in row["detail"]
    assert "http port 8080" in row["detail"]
    assert fake.closed is True


def test_doctor_acme_bind_probe_works_against_the_real_stdlib():
    """One leg with no fake: port 0 asks the kernel for any free port.

    Pins that the production path — ``socket``/``bind``/``close``, no
    ``SO_REUSEADDR`` — actually runs, so the fake-driven legs above are
    testing branches of code that works.
    """
    client = _doctor_app(
        settings=_acme_settings(http_port=0),
        proxy_manager=SimpleNamespace(state=ProxyState.backoff, available=False),
    )

    row = _acme_row(client)
    assert row["status"] == "warn"
    assert "http port 0 is bindable" in row["detail"]


@pytest.mark.parametrize(
    "error",
    [
        None,
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EADDRINUSE, "Address already in use"),
        OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address"),
    ],
)
def test_doctor_acme_details_carry_no_filesystem_path(monkeypatch, error):
    """Doctor discipline: a port number and a config KEY, never a path.

    Every branch is covered, including the two that never reach the probe —
    the detail is the same body for every role, so "no path" has to hold
    everywhere rather than on the branches a happy node takes.
    """
    monkeypatch.setattr(system_routes, "socket", _FakeSocket(error))
    for manager in (
        SimpleNamespace(state=ProxyState.available, available=True),
        SimpleNamespace(state=ProxyState.backoff, available=False),
    ):
        detail = _acme_row(
            _doctor_app(settings=_acme_settings(http_port=80), proxy_manager=manager)
        )["detail"]
        assert "/" not in detail

    assert "/" not in _acme_row(_doctor_app())["detail"]


def _materialize_key_shape(key_path: Path, shape: str) -> None:
    """Create *shape* at *key_path* — the six things a key path can actually be."""
    if shape == "absent":
        return
    if shape == "regular":
        key_path.write_text("ab" * 32 + "\n", encoding="utf-8")
    elif shape == "dir":
        key_path.mkdir()
    elif shape == "dangling":
        key_path.symlink_to(key_path.with_name("gone"))
    elif shape == "symlink_to_file":
        target = key_path.with_name("real.key")
        target.write_text("ab" * 32 + "\n", encoding="utf-8")
        key_path.symlink_to(target)
    elif shape == "fifo":
        os.mkfifo(key_path)
    else:  # pragma: no cover — a typo in a test, not a runtime path
        raise AssertionError(f"unknown key-path shape {shape!r}")


def _secrets_key_row(
    *,
    shape: str,
    ciphertexts: bool,
    round_trip: bool = True,
):
    """Drive the secrets_key check with a fake manager; report the row + probe calls.

    ``key_path`` is a REAL filesystem path (in a throwaway dir), because the
    check classifies it from ``lstat`` mode bits — a stub that only answers
    ``is_file()`` cannot express "a directory sits here". *shape* selects what
    is materialized: one of the shapes ``_materialize_key_shape`` knows.
    """
    calls = {"self_check_called": False}

    def _self_check():
        calls["self_check_called"] = True
        return round_trip

    with tempfile.TemporaryDirectory() as tmp:
        key_path = Path(tmp) / "secrets.key"
        _materialize_key_shape(key_path, shape)
        sm = SimpleNamespace(
            key_path=key_path,
            has_ciphertexts=lambda: ciphertexts,
            self_check=_self_check,
        )
        body = _doctor_app(secret_manager=sm).get("/api/doctor", headers=_ADMIN).json()
    return _check(body, "secrets_key"), calls, body


def test_doctor_secrets_key_fresh_install_is_not_a_failure():
    """No key + no ciphertexts is a healthy brand-new node, not a fault.

    The key is minted lazily on the first secret write, so a node that has
    never stored a secret legitimately has no key file. ``skipped`` is also
    the only status absent from ``_STATUS_RANK``, so the whole doctor stays
    ``ok`` — which is what a customer sees at the end of every install.
    """
    row, calls, body = _secrets_key_row(shape="absent", ciphertexts=False)
    assert row["status"] == "skipped"
    assert "created on the first secret write" in row["detail"]
    assert body["status"] != "fail"
    # ``skipped`` is absent from _STATUS_RANK, so the fresh-install row leaves
    # the top status exactly where the other eleven checks put it.
    _, _, healthy = _secrets_key_row(shape="regular", ciphertexts=True)
    assert body["status"] == healthy["status"]
    # A missing key file must short-circuit — self_check (which would touch the
    # key) is never called, so nothing can create it.
    assert calls["self_check_called"] is False


def test_doctor_secrets_key_missing_with_ciphertexts_still_fails():
    """No key + encrypted material at rest is unrecoverable loss — stays ``fail``."""
    row, calls, body = _secrets_key_row(shape="absent", ciphertexts=True)
    assert row["status"] == "fail"
    detail = row["detail"]
    assert "encrypted secrets exist" in detail
    assert "cannot be decrypted" in detail
    assert body["status"] == "fail"
    # Still never reaches a path that would create the key.
    assert calls["self_check_called"] is False


def test_doctor_secrets_key_present_and_usable_is_ok():
    row, calls, _ = _secrets_key_row(shape="regular", ciphertexts=True)
    assert row["status"] == "ok"
    assert row["detail"] == "secrets key present and usable"
    assert calls["self_check_called"] is True


def test_doctor_secrets_key_present_but_round_trip_fails():
    row, calls, body = _secrets_key_row(shape="regular", ciphertexts=True, round_trip=False)
    assert row["status"] == "fail"
    assert "round-trip" in row["detail"]
    assert body["status"] == "fail"
    assert calls["self_check_called"] is True


_POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt",
    reason="needs POSIX symlink/FIFO semantics at the key path",
)


@_POSIX_ONLY
@pytest.mark.parametrize("shape", ["dir", "dangling", "fifo"])
def test_doctor_secrets_key_non_regular_path_fails(shape):
    """Something-but-not-a-key at the key path is a fault, never ``skipped``.

    ``is_file()`` answers False for a directory, a dangling symlink and a FIFO
    exactly as it does for an absent path, so the lazy-creation branch used to
    call all three a healthy fresh install — on a node where the first secret
    write is guaranteed to fail (``O_EXCL`` raises EEXIST on every one of
    them). The empty-store case is the one that regressed, so pin it here.
    """
    row, calls, body = _secrets_key_row(shape=shape, ciphertexts=False)
    assert row["status"] == "fail"
    detail = row["detail"]
    assert "not a regular file" in detail
    assert "[security].secrets_key_file" in detail
    assert body["status"] == "fail"
    # Never opened: a FIFO reader blocks forever, and a wedged worker thread
    # outlives ``wait_for``'s 2 s budget (the timeout abandons it, nothing
    # unblocks the syscall). The classification is stat-only, so the row is
    # produced immediately.
    assert calls["self_check_called"] is False
    assert row["latency_ms"] < 2000


@_POSIX_ONLY
def test_doctor_secrets_key_non_regular_path_with_ciphertexts_says_both():
    """A bogus key path AND material at rest: the detail carries the worse half too."""
    row, calls, _ = _secrets_key_row(shape="dir", ciphertexts=True)
    assert row["status"] == "fail"
    detail = row["detail"]
    assert "not a regular file" in detail
    assert "cannot be decrypted" in detail
    assert calls["self_check_called"] is False


@_POSIX_ONLY
def test_doctor_secrets_key_symlink_to_a_real_key_is_ok():
    """A key file reached through a symlink is an ordinary deployment, not a fault."""
    row, calls, body = _secrets_key_row(shape="symlink_to_file", ciphertexts=True)
    assert row["status"] == "ok"
    assert calls["self_check_called"] is True
    assert body["status"] != "fail"


def test_key_path_shape_separates_absent_from_present_but_unusable(tmp_path):
    """``lstat`` (not ``exists``) is what makes a dangling symlink ``other``."""
    from nerdit.daemon.routes.system import _key_path_shape

    assert _key_path_shape(tmp_path / "nothing-here") == "absent"

    regular = tmp_path / "key"
    regular.write_text("ab", encoding="utf-8")
    assert _key_path_shape(regular) == "regular"

    if os.name != "nt":
        dangling = tmp_path / "dangling"
        dangling.symlink_to(tmp_path / "gone")
        # The trap this classification exists to remove: the entry IS there.
        assert dangling.exists() is False
        assert _key_path_shape(dangling) == "other"

    directory = tmp_path / "dir"
    directory.mkdir()
    assert _key_path_shape(directory) == "other"


def test_doctor_secrets_key_details_carry_no_filesystem_path(tmp_path):
    """Every branch's detail is path-free — the same body goes to every role."""
    from nerdit.core.secrets import SecretManager

    manager = SecretManager(tmp_path / "secrets")
    rows = [
        _secrets_key_row(shape="absent", ciphertexts=False)[0],
        _secrets_key_row(shape="absent", ciphertexts=True)[0],
        _secrets_key_row(shape="regular", ciphertexts=True)[0],
        _secrets_key_row(shape="regular", ciphertexts=True, round_trip=False)[0],
    ]
    if os.name != "nt":
        rows += [
            _secrets_key_row(shape="dir", ciphertexts=False)[0],
            _secrets_key_row(shape="dir", ciphertexts=True)[0],
            _secrets_key_row(shape="dangling", ciphertexts=False)[0],
            _secrets_key_row(shape="fifo", ciphertexts=False)[0],
        ]
    for row in rows:
        detail = row["detail"]
        assert "/" not in detail
        assert str(tmp_path) not in detail
        assert str(manager.key_path) not in detail


def test_doctor_secrets_key_on_a_real_fresh_manager(tmp_path):
    """End-to-end against a real SecretManager: no store dir at all ⇒ non-failing.

    A brand-new install has neither ``<data_dir>/secrets.key`` nor the
    ``<data_dir>/secrets/`` directory; the check must tolerate both being
    absent and must leave them absent.
    """
    from nerdit.core.secrets import SecretManager

    data_dir = tmp_path / "nerdit"
    data_dir.mkdir()
    manager = SecretManager(data_dir / "secrets")
    assert not manager.key_path.exists()
    assert not (data_dir / "secrets").exists()

    body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    row = _check(body, "secrets_key")
    assert row["status"] == "skipped"
    assert body["status"] != "fail"
    # The check created nothing.
    assert not manager.key_path.exists()
    assert not (data_dir / "secrets").exists()


def test_doctor_secrets_key_real_manager_with_ciphertext_and_no_key(tmp_path):
    """A real ``.enc`` at rest with the key gone is reported as data loss."""
    from nerdit.core.secrets import SecretManager

    data_dir = tmp_path / "nerdit"
    store = data_dir / "secrets"
    store.mkdir(parents=True)
    (store / "app.enc").write_text("{}", encoding="utf-8")
    manager = SecretManager(store)
    assert not manager.key_path.exists()

    body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    row = _check(body, "secrets_key")
    assert row["status"] == "fail"
    assert "cannot be decrypted" in row["detail"]
    assert not manager.key_path.exists()


def test_doctor_secrets_key_real_manager_with_a_directory_at_the_key_path(tmp_path):
    """The regression in the flesh: a bogus ``[security].secrets_key_file``.

    Point the key at a directory and leave the store empty — the state that
    used to report ``skipped`` ("all good, the key appears on first write")
    on a node where that first write raises ``FileExistsError``.
    """
    from nerdit.core.secrets import SecretManager

    data_dir = tmp_path / "nerdit"
    bogus = data_dir / "not-a-key"
    bogus.mkdir(parents=True)
    manager = SecretManager(data_dir / "secrets", key_path=bogus)

    body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    row = _check(body, "secrets_key")
    assert row["status"] == "fail"
    assert "not a regular file" in row["detail"]
    assert "/" not in row["detail"]
    # The very thing the check warns about: the first secret write cannot mint
    # a key there (``O_EXCL`` → EEXIST, then the read-back hits the directory).
    from nerdit.core.secrets import SecretDecryptError

    with pytest.raises(SecretDecryptError):
        manager.set("app", {"K": "v"})
    assert bogus.is_dir()


@pytest.mark.parametrize(
    "content,label",
    [("", "empty"), ("not-hex-at-all\n", "malformed"), ("ab\n", "too short")],
)
def test_doctor_secrets_key_unusable_key_file_is_diagnosable(tmp_path, content, label):
    """An unreadable/empty/malformed key file gets a real detail, not a generic one.

    All three raise ``SecretDecryptError`` out of ``self_check()``; left to
    ``_run_check``'s blanket handler they would surface as "check failed or
    timed out", which reads like a daemon bug and tells the operator nothing.
    """
    from nerdit.core.secrets import SecretManager

    data_dir = tmp_path / "nerdit"
    store = data_dir / "secrets"
    store.mkdir(parents=True)
    manager = SecretManager(store)
    manager.key_path.write_text(content, encoding="utf-8")

    body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    row = _check(body, "secrets_key")
    assert row["status"] == "fail", label
    detail = row["detail"]
    assert "could not be read as a key" in detail
    assert "check failed or timed out" not in detail
    assert "/" not in detail
    assert str(tmp_path) not in detail


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason="needs POSIX mode bits and a non-root euid (root ignores 0o000)",
)
def test_doctor_secrets_key_unreadable_key_file_is_diagnosable(tmp_path):
    """A 0o000 key file is a regular file — it reaches self_check and fails clean."""
    from nerdit.core.secrets import SecretManager

    store = tmp_path / "nerdit" / "secrets"
    store.mkdir(parents=True)
    manager = SecretManager(store)
    manager.set("app", {"API_KEY": "abc"})
    manager.key_path.chmod(0o000)
    try:
        body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    finally:
        manager.key_path.chmod(0o600)
    row = _check(body, "secrets_key")
    assert row["status"] == "fail"
    assert "could not be read as a key" in row["detail"]
    assert "/" not in row["detail"]


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason="needs POSIX mode bits and a non-root euid (root ignores 0o000)",
)
def test_doctor_secrets_key_unreadable_store_still_fails(tmp_path):
    """A store that exists but cannot be listed must not read as a fresh install.

    Hand-restoring a backup as root leaves ``<data_dir>/secrets/`` owned by root
    0700 while the daemon runs as the unit user, and the key can easily land
    somewhere else. Every secret injection then fails; doctor must say so
    (``fail``), not report the reassuring "nothing stored yet".
    """
    from nerdit.core.secrets import SecretManager

    store = tmp_path / "nerdit" / "secrets"
    store.mkdir(parents=True)
    manager = SecretManager(store)
    manager.set("app", {"API_KEY": "abc"})
    manager.key_path.unlink()
    store.chmod(0o000)
    try:
        body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    finally:
        store.chmod(0o700)
    row = _check(body, "secrets_key")
    assert row["status"] == "fail"
    assert "cannot be decrypted" in row["detail"]
    assert not manager.key_path.exists()


def test_doctor_secrets_key_ignores_temp_and_legacy_residue(tmp_path):
    """Staging residue and legacy plaintext are not encrypted material.

    ``app.enc.tmp-<pid>`` is crash residue from an interrupted write and
    ``app.json`` is pre-P8 plaintext that is recoverable without the key —
    neither may turn a healthy fresh install back into a ``fail``. Both are
    planted *after* construction so the constructor's own sweep/migration
    (which would mint a key) does not run over them.
    """
    from nerdit.core.secrets import SecretManager

    data_dir = tmp_path / "nerdit"
    store = data_dir / "secrets"
    store.mkdir(parents=True)
    manager = SecretManager(store)
    (store / "app.enc.tmp-123").write_text("{}", encoding="utf-8")
    (store / "app.json").write_text('{"K": "v"}', encoding="utf-8")
    assert manager.has_ciphertexts() is False

    body = _doctor_app(secret_manager=manager).get("/api/doctor", headers=_ADMIN).json()
    row = _check(body, "secrets_key")
    assert row["status"] == "skipped"
    assert body["status"] != "fail"
    assert not manager.key_path.exists()


@pytest.mark.parametrize(
    "headers,role",
    [(_ADMIN, None), (_SUBMITTER, TokenRole.submitter), (_READONLY, TokenRole.readonly)],
)
def test_doctor_restart_pending_detail_is_key_names_only(monkeypatch, headers, role):
    # A fresh load that differs from the boot settings in exactly one restart
    # key → the detail names the KEY, never the value, for every role.
    fresh = SimpleNamespace(proxy=SimpleNamespace(mode="subdomain"))
    monkeypatch.setattr(system_routes, "load_settings", lambda: fresh)
    monkeypatch.setattr(system_routes, "_RESTART_KEYS", {"proxy": frozenset({"mode"})})

    settings = _doctor_settings()
    settings.proxy = SimpleNamespace(mode="path", mdns=False)  # boot proxy.mode
    client = _doctor_app(auth_role=role, settings=settings)
    body = client.get("/api/doctor", headers=headers).json()
    pending = _check(body, "config_restart_pending")
    assert pending["status"] == "warn"
    assert pending["detail"] == "pending restart: proxy.mode"
    assert "subdomain" not in pending["detail"]  # the VALUE never leaks


def test_doctor_reports_boot_frozen_section_drift(monkeypatch):
    # Against the REAL _RESTART_KEYS: a TOML edit to a [containers] or [nerdit]
    # key used to leave the check reporting "no restart-required config drift"
    # while the running daemon was still on its boot value. [nerdit] also pins
    # the root-vs-submodel lookup: its keys live on NerditSettings itself.
    settings = _doctor_settings()
    settings.log_level = "info"
    settings.containers = SimpleNamespace(read_only_rootfs=False, drop_all_caps=True)
    fresh = SimpleNamespace(
        log_level="debug",
        data_dir=settings.data_dir,
        containers=SimpleNamespace(read_only_rootfs=True, drop_all_caps=True),
    )
    monkeypatch.setattr(system_routes, "load_settings", lambda: fresh)

    body = _doctor_app(settings=settings).get("/api/doctor", headers=_ADMIN).json()
    pending = _check(body, "config_restart_pending")
    assert pending["status"] == "warn"
    assert "containers.read_only_rootfs" in pending["detail"]
    assert "nerdit.log_level" in pending["detail"]
    assert "debug" not in pending["detail"]  # key names only, never values


def test_doctor_reports_no_drift_for_a_non_default_upload_dir(monkeypatch, tmp_path):
    """A normalized boot object vs a raw re-read is not drift.

    The lifespan appends ``[daemon].upload_dir`` to
    ``containers.allowed_mount_roots`` IN PLACE before the object becomes
    ``app.state.settings``. Comparing it against an un-normalized re-read
    reported a permanent, restart-proof ``containers.allowed_mount_roots``
    drift on every install whose upload dir is not already in the list —
    which trains the operator to ignore the row that flags a real
    ``auth_token`` change. Runs against the REAL ``_RESTART_KEYS``.
    """
    from nerdit.config.settings import load_settings as real_load_settings
    from nerdit.daemon.bootstrap import normalize_mount_roots

    config = tmp_path / "config.toml"
    config.write_text(f'[daemon]\nupload_dir = "{tmp_path / "uploads"}"\n', encoding="utf-8")

    boot = real_load_settings(config)
    normalize_mount_roots(boot)
    assert boot.daemon.upload_dir in boot.containers.allowed_mount_roots
    # The re-read the check performs is deliberately NOT normalized here — the
    # route has to do it, which is the whole point of the regression.
    monkeypatch.setattr(system_routes, "load_settings", lambda: real_load_settings(config))

    body = _doctor_app(settings=boot).get("/api/doctor", headers=_ADMIN).json()
    pending = _check(body, "config_restart_pending")
    assert pending["status"] == "ok", pending["detail"]
    assert "allowed_mount_roots" not in pending["detail"]


def test_doctor_still_reports_a_real_containers_change_beside_the_upload_dir(monkeypatch, tmp_path):
    """Positive control: normalizing must not swallow a genuine edit."""
    from nerdit.config.settings import load_settings as real_load_settings
    from nerdit.daemon.bootstrap import normalize_mount_roots

    config = tmp_path / "config.toml"
    config.write_text(f'[daemon]\nupload_dir = "{tmp_path / "uploads"}"\n', encoding="utf-8")
    boot = real_load_settings(config)
    normalize_mount_roots(boot)

    edited = tmp_path / "edited.toml"
    edited.write_text(
        f'[daemon]\nupload_dir = "{tmp_path / "uploads"}"\n'
        "[containers]\nread_only_rootfs = "
        f"{'false' if boot.containers.read_only_rootfs else 'true'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(system_routes, "load_settings", lambda: real_load_settings(edited))

    body = _doctor_app(settings=boot).get("/api/doctor", headers=_ADMIN).json()
    pending = _check(body, "config_restart_pending")
    assert pending["status"] == "warn"
    assert "containers.read_only_rootfs" in pending["detail"]
    assert "allowed_mount_roots" not in pending["detail"]


def test_doctor_db_quick_check_ok_on_real_file():
    # The _db probe reads a dedicated read-only connection off the real file
    # (not the shared aiosqlite conn); a clean db reports ok with a byte count.
    body = _doctor_app().get("/api/doctor", headers=_ADMIN).json()
    db = _check(body, "db")
    assert db["status"] == "ok"
    assert "quick_check ok" in db["detail"]


def test_doctor_proxy_no_binary_warns_with_install_hint():
    # Proxy enabled but the caddy binary is unresolved is a real misconfiguration
    # the operator asked for — warn (not skipped) with an actionable hint.
    pm = SimpleNamespace(state=ProxyState.no_binary, available=False)
    body = _doctor_app(proxy_manager=pm).get("/api/doctor", headers=_ADMIN).json()
    proxy = _check(body, "proxy")
    assert proxy["status"] == "warn"
    assert "caddy binary is missing" in proxy["detail"]
    assert "install caddy" in proxy["detail"]
    assert body["status"] == "warn"  # bubbles to worst-of


def test_doctor_proxy_disabled_stays_skipped():
    # A deliberately disabled proxy is not a problem — stays skipped, never warns.
    pm = SimpleNamespace(state=ProxyState.disabled, available=False)
    body = _doctor_app(proxy_manager=pm).get("/api/doctor", headers=_ADMIN).json()
    proxy = _check(body, "proxy")
    assert proxy["status"] == "skipped"
    assert proxy["detail"] == "proxy disabled"


def _mdns_settings(advertiser_reason=None):
    """Doctor settings with mDNS enabled + a matching advertiser stub."""
    settings = _doctor_settings()
    settings.proxy = SimpleNamespace(mdns=True)
    advertiser = SimpleNamespace(registered=False, reason=advertiser_reason)
    return settings, advertiser


def test_doctor_mdns_zeroconf_missing_warns_with_install_hint():
    settings, advertiser = _mdns_settings(MDNS_REASON_ZEROCONF_MISSING)
    body = (
        _doctor_app(settings=settings, mdns_advertiser=advertiser)
        .get("/api/doctor", headers=_ADMIN)
        .json()
    )
    mdns = _check(body, "mdns")
    assert mdns["status"] == "warn"
    assert "zeroconf package is missing" in mdns["detail"]
    assert "nerdit[mdns]" in mdns["detail"]


def test_doctor_mdns_disabled_stays_skipped():
    # Default _doctor_settings has proxy.mdns=False → skipped, unchanged.
    body = _doctor_app().get("/api/doctor", headers=_ADMIN).json()
    mdns = _check(body, "mdns")
    assert mdns["status"] == "skipped"
    assert mdns["detail"] == "mDNS advertising disabled"


def test_doctor_mdns_other_failure_warns_generically():
    # A non-zeroconf registration failure keeps the generic warn (unchanged).
    settings, advertiser = _mdns_settings(advertiser_reason="register_failed")
    body = (
        _doctor_app(settings=settings, mdns_advertiser=advertiser)
        .get("/api/doctor", headers=_ADMIN)
        .json()
    )
    mdns = _check(body, "mdns")
    assert mdns["status"] == "warn"
    assert mdns["detail"] == "mDNS enabled but not registered"


async def test_doctor_blocking_probe_is_preempted_within_budget(monkeypatch):
    # A probe whose sync body sleeps 3 s must NOT block the event loop: because
    # the blocking work runs via asyncio.to_thread, wait_for preempts it at the
    # 2 s budget (→ fail/timeout) while every other check still returns. On the
    # pre-fix code the sleep ran directly on the loop, so wait_for could not fire
    # and the whole doctor call took ~3 s with disk reported "ok".
    #
    # An httpx.AsyncClient over ASGITransport is used (not TestClient) so the
    # measured window is the gather itself — TestClient tears down the loop per
    # request and blocks on the background thread's executor shutdown, masking
    # the preemption in the timing.
    def _slow_disk(_p):
        time.sleep(3)
        return SimpleNamespace(total=100, free=80)

    monkeypatch.setattr(system_routes.shutil, "disk_usage", _slow_disk)

    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    settings = _doctor_settings()
    app.state.settings = settings
    for key, value in _doctor_defaults(settings).items():
        setattr(app.state, key, value)
    q = _queries_for(None)
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)

    transport = httpx.ASGITransport(app=app)
    start = time.monotonic()
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        body = (await client.get("/api/doctor", headers=_ADMIN)).json()
    elapsed = time.monotonic() - start

    disk = _check(body, "disk")
    assert disk["status"] == "fail"
    assert "timed out" in disk["detail"]
    # Other checks were not blocked — they returned their real status.
    assert _check(body, "gpu")["status"] == "ok"
    # The 2 s budget preempted the 3 s probe (proves to_thread concurrency).
    assert elapsed < 2.9


# --- the license surfaces (P17d D-LIC6) ---------------------------------------
#
# Two surfaces, one holder: ``/doctor``'s twelfth check and ``/capabilities``'
# ``license`` block are both pure ``app.state.license`` projections, and the
# *temporal* half is recomputed live from the holder's injected clock — so the
# whole six-state matrix is exercised with zero file I/O and zero sleeps.


def _license_holder(
    *,
    state: str | None = "valid",
    features: tuple[str, ...] = ("remote_link",),
    plan: str = "pro",
    reason: str | None = None,
) -> LicenseState:
    """A holder pinned at one of the six D-LIC2 states, on a frozen clock."""
    if state is None:
        return LicenseState(now=lambda: _LICENSE_NOW)
    if state == "invalid":
        return LicenseState(
            LicenseVerdict("invalid", reason=reason or "bad_signature"), now=lambda: _LICENSE_NOW
        )
    offsets = {
        "valid": timedelta(days=30),
        "expired_grace": -timedelta(days=2),
        "expired": -timedelta(seconds=LICENSE_GRACE_S + 86400),
    }
    claims = LicenseClaims(
        v=1,
        lid="0f1e2d3c4b5a69788796a5b4c3d2e1f0",
        iat=_LICENSE_NOW - timedelta(days=1),
        customer_id="cus_p17d_golden",
        plan=plan,
        features=features,
        expires_at=_LICENSE_NOW + offsets[state],
    )
    # The stored verdict's own state is irrelevant — the holder recomputes it
    # from the claims on every read, which is exactly the property under test.
    return LicenseState(LicenseVerdict("valid", claims=claims), now=lambda: _LICENSE_NOW)


_LICENSE_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def test_doctor_license_absent_is_skipped():
    """An unlicensed daemon is the ordinary free local product, not a problem.

    ``skipped`` never enters the worst-of ranking (``_STATUS_RANK``), so an
    unlicensed daemon's top status is whatever the other eleven checks say —
    pinned here by comparing against the very same run without a holder.
    """
    with_holder = (
        _doctor_app(license=_license_holder(state=None)).get("/api/doctor", headers=_ADMIN).json()
    )
    check = _check(with_holder, "license")

    assert check["status"] == "skipped"
    assert check["detail"] == "no license installed"
    without = _doctor_app().get("/api/doctor", headers=_ADMIN).json()
    assert with_holder["status"] == without["status"]


def test_doctor_license_is_skipped_when_the_holder_is_missing_entirely():
    """A daemon booted without the holder (an older app) must not fail here."""
    check = _check(_doctor_app().get("/api/doctor", headers=_ADMIN).json(), "license")

    assert check["status"] == "skipped"


def test_doctor_license_valid_reports_plan_and_days():
    client = _doctor_app(license=_license_holder(state="valid"))
    check = _check(client.get("/api/doctor", headers=_ADMIN).json(), "license")

    assert check["status"] == "ok"
    assert check["detail"] == "valid — plan pro, expires in 30d"


def test_doctor_license_in_grace_warns_with_both_day_counts():
    """WP-D1's "warn in doctor during grace" — with the renewal deadline."""
    body = (
        _doctor_app(license=_license_holder(state="expired_grace"))
        .get("/api/doctor", headers=_ADMIN)
        .json()
    )
    check = _check(body, "license")

    assert check["status"] == "warn"
    assert check["detail"] == "expired 2d ago — grace ends in 5d; renew"
    assert body["status"] == "warn"


def test_doctor_license_past_grace_fails():
    """WP-D1's "fail after" lands HERE — never on the tunnel (W-D11)."""
    body = (
        _doctor_app(license=_license_holder(state="expired"))
        .get("/api/doctor", headers=_ADMIN)
        .json()
    )
    check = _check(body, "license")

    assert check["status"] == "fail"
    assert check["detail"] == "expired — grace period over; renew and reinstall"
    assert body["status"] == "fail"


def test_doctor_license_invalid_fails_with_the_machine_reason_token():
    check = _check(
        _doctor_app(license=_license_holder(state="invalid", reason="unknown_kid"))
        .get("/api/doctor", headers=_ADMIN)
        .json(),
        "license",
    )

    assert check["status"] == "fail"
    assert "unknown_kid" in check["detail"]


def test_doctor_license_feature_gap_only_warns_when_the_link_is_enabled():
    """The gap is meaningless on a daemon with no tunnel configured."""
    holder = _license_holder(state="valid", plan="basic", features=("sso",))

    quiet = _check(_doctor_app(license=holder).get("/api/doctor", headers=_ADMIN).json(), "license")
    assert quiet["status"] == "ok"
    assert "remote_link" not in quiet["detail"]

    settings = _doctor_settings()
    settings.link = SimpleNamespace(enabled=True)
    loud = _check(
        _doctor_app(settings=settings, license=holder).get("/api/doctor", headers=_ADMIN).json(),
        "license",
    )
    assert loud["status"] == "warn"
    assert loud["detail"].endswith("; does not cover remote_link")
    # APPENDED, not substituted: the plan/expiry half of the truth survives.
    assert "plan basic" in loud["detail"]


def test_doctor_license_detail_carries_no_path_and_no_customer_id(tmp_path):
    """Doctor discipline: machine tokens and day counts only, for every role."""
    settings = _doctor_settings(str(tmp_path))
    settings.link = SimpleNamespace(enabled=True)
    body = _doctor_app(
        auth_role=TokenRole.readonly,
        settings=settings,
        license=_license_holder(state="expired_grace"),
    ).get("/api/doctor", headers=_READONLY)
    check = _check(body.json(), "license")

    assert str(tmp_path) not in check["detail"]
    assert "cus_p17d_golden" not in body.text
    assert "license.jws" not in check["detail"]


def test_capabilities_license_block_absent_says_installed_false():
    client = _caps_app(role=None)
    client.app.state.license = _license_holder(state=None)

    assert client.get("/api/capabilities", headers=_ADMIN).json()["license"] == {"installed": False}


def test_capabilities_without_a_license_holder_still_projects_the_block():
    """Tolerant like the mcp flag — absence means unlicensed."""
    body = _caps_app(role=None).get("/api/capabilities", headers=_ADMIN).json()

    assert body["license"] == {"installed": False}


def test_capabilities_license_block_projects_the_verified_claims():
    client = _caps_app(role=None)
    client.app.state.license = _license_holder(state="valid")

    block = client.get("/api/capabilities", headers=_ADMIN).json()["license"]

    assert block["installed"] is True
    assert block["state"] == "valid"
    assert block["plan"] == "pro"
    assert block["features"] == ["remote_link"]
    assert block["lid"] == "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
    assert block["expires_at"] == (_LICENSE_NOW + timedelta(days=30)).isoformat()
    # Rendered on the HOLDER's clock, so it agrees with ``state`` exactly.
    assert block["expires_in_s"] == 30 * 86400
    assert block["customer_id"] == "cus_p17d_golden"


@pytest.mark.parametrize(
    "headers,role",
    [(_SUBMITTER, TokenRole.submitter), (_READONLY, TokenRole.readonly)],
)
def test_capabilities_license_customer_id_is_omitted_for_non_admins(headers, role):
    """OMITTED, not nulled — the ``proxy.admin_addr``/``relay_host`` precedent."""
    client = _caps_app(role=role)
    client.app.state.license = _license_holder(state="valid")

    body = client.get("/api/capabilities", headers=headers).json()

    assert "customer_id" not in body["license"]
    # The rest of the block stays visible: a state name and a plan token are
    # not secrets, and an agent needs them to explain a refused feature.
    assert body["license"]["plan"] == "pro"
    assert "cus_p17d_golden" not in json.dumps(body)


def test_capabilities_license_invalid_projects_only_the_reason():
    """No claims survive an invalid verdict, so none are projected."""
    client = _caps_app(role=None)
    client.app.state.license = _license_holder(state="invalid", reason="crit_present")

    block = client.get("/api/capabilities", headers=_ADMIN).json()["license"]

    assert block == {"installed": True, "state": "invalid", "reason": "crit_present"}


def test_capabilities_license_temporal_state_is_recomputed_live():
    """No poller, no restart: the same holder answers grace once time moves.

    The clock is injected, so this pins the *mechanism* (state is derived on
    read, never frozen at boot) rather than waiting for a real week to pass.
    """
    moment = {"now": _LICENSE_NOW}
    claims = LicenseClaims(
        v=1,
        lid="0f1e2d3c4b5a69788796a5b4c3d2e1f0",
        iat=_LICENSE_NOW - timedelta(days=1),
        customer_id="cus_p17d_golden",
        plan="pro",
        features=("remote_link",),
        expires_at=_LICENSE_NOW + timedelta(days=1),
    )
    client = _caps_app(role=None)
    client.app.state.license = LicenseState(
        LicenseVerdict("valid", claims=claims), now=lambda: moment["now"]
    )

    assert client.get("/api/capabilities", headers=_ADMIN).json()["license"]["state"] == "valid"

    moment["now"] = _LICENSE_NOW + timedelta(days=2)
    assert (
        client.get("/api/capabilities", headers=_ADMIN).json()["license"]["state"]
        == "expired_grace"
    )


# --- POST /daemon/restart -----------------------------------------------------


def _restart_app(*, controller=None, order=None) -> tuple[FastAPI, SimpleNamespace]:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    audit_side_effect = None
    if order is not None:

        def audit_side_effect(**kw):  # noqa: F811 — bound only when ordering is tracked
            order.append(("audit", kw.get("action")))

    q = SimpleNamespace(
        get_api_token_by_hash=AsyncMock(return_value=_token_row(TokenRole.submitter)),
        touch_api_token=AsyncMock(),
        insert_audit_log=AsyncMock(side_effect=audit_side_effect),
    )
    app.state.queries = q
    app.state.service_controller = controller or SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 0,
        busy_cutovers=lambda: 0,
        # (P20 WP6) The drain kills still-active run containers at its deadline;
        # the default controller must carry the method or a deadline path in any
        # of these tests would AttributeError instead of re-execing.
        kill_transient_containers=AsyncMock(return_value=0),
        draining=False,
    )
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)
    return app, q


async def test_restart_closes_the_drain_gate_before_the_audit_await(monkeypatch):
    """The drain gate must close before the first await, not after it.

    ``POST /services/{ident}/run`` gates on ``controller.draining``, not on the
    restart flag. The counts reported in the 202 body and the audit row are
    taken in the route's await-free window — so if the gate were only closed
    after the awaited audit insert, a run arriving during that await would be
    admitted into an already-requested restart, be absent from both numbers,
    and at ``drain_timeout_s=0`` be killed immediately by the deadline path
    while the operator had just been told nothing was in flight.
    """
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    seen: dict[str, bool] = {}
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 0,
        busy_cutovers=lambda: 0,
        kill_transient_containers=AsyncMock(return_value=0),
        draining=False,
    )
    app, q = _restart_app(controller=controller)
    # The audit insert is the route's FIRST await — the window under test.
    q.insert_audit_log = AsyncMock(
        side_effect=lambda **kw: seen.setdefault("draining", controller.draining)
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/api/daemon/restart", headers={**_ADMIN, "Idempotency-Key": "k1"})

    assert resp.status_code == 202
    assert seen["draining"] is True, "a run could be admitted during the audit await"


def test_restart_requires_admin():
    app, _ = _restart_app()
    client = TestClient(app)
    # A submitter token passes the coarse mutating gate, then require_role(admin)
    # rejects it in-route.
    resp = client.post("/api/daemon/restart", headers={**_SUBMITTER, "Idempotency-Key": "k1"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_restart_missing_idempotency_key_400():
    app, _ = _restart_app()
    client = TestClient(app)
    resp = client.post("/api/daemon/restart", headers=_ADMIN)
    assert resp.status_code == 400
    assert resp.json()["code"] == "idempotency_key_required"


async def test_restart_double_returns_409(monkeypatch):
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    app, _ = _restart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first = await client.post(
            "/api/daemon/restart", headers={**_ADMIN, "Idempotency-Key": "k1"}
        )
        assert first.status_code == 202
        assert first.json() == {
            "restarting": True,
            "in_flight_builds": 0,
            "in_flight_runs": 0,
            # (P24b) The third in-flight kind: a cutover verify holds a green
            # container that must not be re-exec'd through.
            "in_flight_cutovers": 0,
            "drain_timeout_s": 60,
        }
        for _ in range(5):
            await asyncio.sleep(0)
        second = await client.post(
            "/api/daemon/restart", headers={**_ADMIN, "Idempotency-Key": "k2"}
        )
    assert second.status_code == 409
    assert second.json()["code"] == "daemon.restart_in_progress"


async def test_restart_audit_committed_before_sigterm_at_drain_zero(monkeypatch):
    order: list = []
    kill = MagicMock(side_effect=lambda *a: order.append(("sigterm", a)))
    monkeypatch.setattr(system_routes.os, "kill", kill)
    controller = SimpleNamespace(
        busy_builds=lambda: 0, busy_runs=lambda: 0, busy_cutovers=lambda: 0, draining=False
    )
    app, q = _restart_app(controller=controller, order=order)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/daemon/restart",
            headers={**_ADMIN, "Idempotency-Key": "k1"},
            json={"drain_timeout_s": 0},
        )
        assert resp.status_code == 202
        assert resp.json()["drain_timeout_s"] == 0
        # Let the fire-and-forget drain task run to the (mocked) SIGTERM.
        for _ in range(10):
            await asyncio.sleep(0)

    # The daemon.restart audit row landed, and it landed BEFORE the SIGTERM.
    assert kill.called
    assert order[0] == ("audit", "daemon.restart")
    assert order[-1][0] == "sigterm"
    assert controller.draining is True  # the drain gate was set


async def test_restart_reports_in_flight_builds(monkeypatch):
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    controller = SimpleNamespace(
        busy_builds=lambda: 3,
        busy_runs=lambda: 0,
        busy_cutovers=lambda: 0,
        kill_transient_containers=AsyncMock(return_value=0),
        draining=False,
    )
    app, _ = _restart_app(controller=controller)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/daemon/restart",
            headers={**_ADMIN, "Idempotency-Key": "k1"},
            json={"drain_timeout_s": 0},
        )
        for _ in range(5):
            await asyncio.sleep(0)
    assert resp.json()["in_flight_builds"] == 3


async def test_restart_reports_in_flight_runs(monkeypatch):
    # (P20 WP6) Rowless runs/releases get their own 202 field — the P13b
    # ``in_flight_builds`` name stays builds-only — and the same count lands in
    # the pre-SIGTERM audit row. Counts only: never a run id or its argv.
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 2,
        busy_cutovers=lambda: 0,
        kill_transient_containers=AsyncMock(return_value=2),
        draining=False,
    )
    app, q = _restart_app(controller=controller)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/daemon/restart",
            headers={**_ADMIN, "Idempotency-Key": "k1"},
            json={"drain_timeout_s": 0},
        )
        for _ in range(5):
            await asyncio.sleep(0)
    assert resp.json()["in_flight_runs"] == 2
    assert resp.json()["in_flight_builds"] == 0
    params = json.loads(q.insert_audit_log.await_args.kwargs["params_redacted"])
    assert params["in_flight_runs"] == 2
    assert params["in_flight_builds"] == 0


async def test_restart_audit_failure_resets_flag_and_allows_retry(monkeypatch):
    # C1: the audit insert is fallible. If it raises AFTER the flag is set, the
    # flag (and the drain gate) must be cleared so the daemon is not wedged into a
    # permanent 409 AND a re-exec on the next clean shutdown. A subsequent
    # well-formed restart must succeed (202), not a stuck 409.
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    controller = SimpleNamespace(
        busy_builds=lambda: 0, busy_runs=lambda: 0, busy_cutovers=lambda: 0, draining=False
    )
    app, q = _restart_app(controller=controller)
    q.insert_audit_log = AsyncMock(side_effect=RuntimeError("db down"))

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first = await client.post(
            "/api/daemon/restart", headers={**_ADMIN, "Idempotency-Key": "k1"}
        )
        # The failing audit propagates to a 5xx, and the flag/gate are cleared.
        assert first.status_code >= 500
        assert system_routes.is_restart_requested() is False
        assert controller.draining is False

        # A retry once the audit works again succeeds — not a permanent 409.
        q.insert_audit_log = AsyncMock()
        second = await client.post(
            "/api/daemon/restart", headers={**_ADMIN, "Idempotency-Key": "k2"}
        )
        for _ in range(5):
            await asyncio.sleep(0)
    assert second.status_code == 202


async def test_restart_retains_drain_task_handle(monkeypatch):
    # C2: the fire-and-forget drain task must be retained module-side so the loop
    # cannot GC it mid-drain. After a 202 the handle is set.
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    app, _ = _restart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/daemon/restart",
            headers={**_ADMIN, "Idempotency-Key": "k1"},
            json={"drain_timeout_s": 0},
        )
        assert resp.status_code == 202
        assert system_routes._drain_task is not None
        assert isinstance(system_routes._drain_task, asyncio.Task)
        for _ in range(5):
            await asyncio.sleep(0)


async def test_drain_waits_for_in_flight_runs_not_only_builds(monkeypatch):
    """(P20, security S9) The drain predicate must count runs AND releases.

    One-off runs and ``[deploy].release`` executions have no ``jobs`` row and
    are not builds, so a builds-only predicate re-execs straight through them
    and the re-exec kills a live migration the drain was supposed to wait out.
    The drain TIMEOUT bounds the wait, and at the deadline the bound is
    enforced rather than assumed — see
    ``test_drain_deadline_kills_active_run_containers``.

    The 202 body's ``in_flight_builds`` is deliberately NOT widened: the field
    name is part of the P13b response contract (pinned by
    ``test_restart_reports_in_flight_builds``), so it keeps reporting builds
    only and the run count rides its own ``in_flight_runs`` field (WP6).
    """
    kill = MagicMock()
    monkeypatch.setattr(system_routes.os, "kill", kill)
    monkeypatch.setattr(system_routes, "_uvicorn_server", None)
    monkeypatch.setattr(system_routes, "_DRAIN_POLL_SECONDS", 0.01)

    state = {"runs": 1}
    kill_runs = AsyncMock(return_value=0)
    controller = SimpleNamespace(
        busy_builds=lambda: 0,  # nothing building — only a run is in flight
        busy_runs=lambda: state["runs"],
        busy_cutovers=lambda: 0,
        kill_transient_containers=kill_runs,
        draining=False,
    )

    task = asyncio.create_task(system_routes._drain_and_restart(controller, 5))
    for _ in range(20):
        await asyncio.sleep(0.005)
    assert not kill.called, "the drain re-exec'd while a run/release container was still executing"

    state["runs"] = 0  # the run finishes → the drain completes
    await asyncio.wait_for(task, timeout=5)
    assert kill.called


async def test_restart_reports_in_flight_cutovers(monkeypatch):
    """(P24b) The third in-flight kind reaches both the 202 body and the audit row."""
    monkeypatch.setattr(system_routes.os, "kill", MagicMock())
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 0,
        busy_cutovers=lambda: 1,
        kill_transient_containers=AsyncMock(return_value=1),
        draining=False,
    )
    app, q = _restart_app(controller=controller)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/daemon/restart",
            headers={**_ADMIN, "Idempotency-Key": "k1"},
            json={"drain_timeout_s": 0},
        )
        for _ in range(5):
            await asyncio.sleep(0)
    assert resp.json()["in_flight_cutovers"] == 1
    params = json.loads(q.insert_audit_log.await_args.kwargs["params_redacted"])
    assert params["in_flight_cutovers"] == 1


async def test_drain_waits_for_and_kills_in_flight_cutovers(monkeypatch):
    """A cutover verify holds a green container: the drain must not re-exec past it.

    At the deadline the kill covers greens too (``kill_transient_containers``),
    which makes the verify's probe loop fail immediately rather than hang on its
    budget — blue keeps serving and the generation settles ``cutover_failed``.
    """
    order: list[str] = []
    kill = MagicMock(side_effect=lambda *a: order.append("sigterm"))
    monkeypatch.setattr(system_routes.os, "kill", kill)
    monkeypatch.setattr(system_routes, "_uvicorn_server", None)

    kill_transient = AsyncMock(side_effect=lambda: order.append("kill_transient") or 1)
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 0,
        busy_cutovers=lambda: 1,  # never drops — the deadline ends the drain
        kill_transient_containers=kill_transient,
        draining=False,
    )

    await system_routes._drain_and_restart(controller, 0)

    kill_transient.assert_awaited_once()
    assert order == ["kill_transient", "sigterm"]


async def test_drain_deadline_kills_active_run_containers(monkeypatch):
    """(P20 WP6) At the deadline the drain KILLS still-active run containers.

    Without this the bound was a lie: the drain gave up, uvicorn then waited
    for the live ``POST /run`` request (up to ``run_timeout_max_s``). The kill
    unblocks that request's ``wait()``, and it must happen BEFORE the
    should_exit/SIGTERM step, or the shutdown is already under way with the
    container still executing.
    """
    order: list[str] = []
    kill = MagicMock(side_effect=lambda *a: order.append("sigterm"))
    monkeypatch.setattr(system_routes.os, "kill", kill)
    monkeypatch.setattr(system_routes, "_uvicorn_server", None)

    kill_runs = AsyncMock(side_effect=lambda: order.append("kill_runs") or 1)
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 1,  # never drops — the deadline is what ends the drain
        busy_cutovers=lambda: 0,
        kill_transient_containers=kill_runs,
        draining=False,
    )

    await system_routes._drain_and_restart(controller, 0)  # zero drain budget

    kill_runs.assert_awaited_once()
    assert order == ["kill_runs", "sigterm"]


async def test_drain_restarts_even_when_the_deadline_kill_raises(monkeypatch):
    """A failed deadline kill must still restart the daemon.

    ``_drain_and_restart`` is a bare task, and by the time it runs the
    restart-requested flag and ``controller.draining`` are already set. If the
    kill — a fallible docker call; a dropped socket is enough — escaped this
    coroutine, the shutdown step would never run: the daemon would never
    restart, every later ``POST /daemon/restart`` would answer 409
    ``daemon.restart_in_progress`` forever, and ``draining`` would keep
    refusing every build. Degrading to "restart anyway" is the only safe
    failure mode.
    """
    kill = MagicMock()
    monkeypatch.setattr(system_routes.os, "kill", kill)
    monkeypatch.setattr(system_routes, "_uvicorn_server", None)

    kill_runs = AsyncMock(side_effect=RuntimeError("docker socket gone"))
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: 1,
        busy_cutovers=lambda: 0,
        kill_transient_containers=kill_runs,
        draining=False,
    )

    await system_routes._drain_and_restart(controller, 0)  # zero budget → deadline kill

    kill_runs.assert_awaited_once()
    assert kill.called, "the drain died before the shutdown step, wedging the daemon"


async def test_drain_no_deadline_kill_when_runs_finish_in_time(monkeypatch):
    """The mirror image: a run that finishes inside the budget is never killed."""
    kill = MagicMock()
    monkeypatch.setattr(system_routes.os, "kill", kill)
    monkeypatch.setattr(system_routes, "_uvicorn_server", None)
    monkeypatch.setattr(system_routes, "_DRAIN_POLL_SECONDS", 0.01)

    state = {"runs": 1}
    kill_runs = AsyncMock(return_value=0)
    controller = SimpleNamespace(
        busy_builds=lambda: 0,
        busy_runs=lambda: state["runs"],
        busy_cutovers=lambda: 0,
        kill_transient_containers=kill_runs,
        draining=False,
    )

    task = asyncio.create_task(system_routes._drain_and_restart(controller, 5))
    for _ in range(10):
        await asyncio.sleep(0.005)
    state["runs"] = 0  # the run finishes well inside the budget
    await asyncio.wait_for(task, timeout=5)

    kill_runs.assert_not_awaited()
    assert kill.called  # the drain still completed the shutdown


async def test_restart_stops_registered_server_without_sigterm(monkeypatch):
    # Uvicorn >=0.29 replays captured signals after run() returns, so a
    # SIGTERM-driven shutdown kills the process before main()'s re-exec branch
    # runs. When main() has registered the server, the drain must stop it via
    # should_exit and never signal the process (found live: the daemon shut
    # down cleanly on restart but exited 143 and never re-exec'd).
    kill = MagicMock()
    monkeypatch.setattr(system_routes.os, "kill", kill)
    fake_server = SimpleNamespace(should_exit=False)
    system_routes.set_uvicorn_server(fake_server)
    app, _ = _restart_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/daemon/restart",
            headers={**_ADMIN, "Idempotency-Key": "k1"},
            json={"drain_timeout_s": 0},
        )
        assert resp.status_code == 202
        for _ in range(10):
            await asyncio.sleep(0)
    assert fake_server.should_exit is True
    assert not kill.called


def test_uvicorn_config_sets_graceful_shutdown_timeout():
    """(P20 WP6) ``should_exit`` alone is not a bound.

    uvicorn waits for in-flight requests INDEFINITELY without
    ``timeout_graceful_shutdown``, so a live ``POST /run`` could hold the
    shutdown open for up to ``run_timeout_max_s`` after the drain had already
    given up. Imported lazily — importing the server module at collection time
    pulls the whole daemon assembly in.
    """
    from nerdit.config.settings import NerditSettings
    from nerdit.daemon import server as server_module

    settings = NerditSettings()
    config = server_module._build_uvicorn_config(FastAPI(), settings)

    assert config.timeout_graceful_shutdown == server_module.GRACEFUL_SHUTDOWN_TIMEOUT_S == 30
    assert config.host == settings.daemon.host
    assert config.port == settings.daemon.port
