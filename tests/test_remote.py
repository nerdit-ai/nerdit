"""Tests for v0.2 remote CLI features: auth, upload, ZIP, config."""

from __future__ import annotations

import io
import tempfile
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.cli.upload import check_upload_size, create_dir_zip, should_exclude
from nerdit.config.settings import generate_auth_token, get_client_config
from nerdit.daemon.errors import register_error_handlers
from nerdit.daemon.middleware import BearerAuthMiddleware
from nerdit.daemon.routes.auth import router as auth_router
from nerdit.daemon.routes.gpus import router as gpus_router
from nerdit.daemon.routes.health import router as health_router
from nerdit.db.models import Gpu, GpuStatus

# ---- Helpers ----


def _make_app(queries, monitor=None, settings=None, token="test-token"):
    """Create a test FastAPI app with auth middleware."""
    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(BearerAuthMiddleware, token=token)
    app.include_router(health_router)
    app.include_router(gpus_router)
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(gpus_router, prefix="/api")

    app.state.queries = queries or AsyncMock()
    app.state.monitor = monitor or MagicMock(get_metrics=MagicMock(return_value={}))

    if settings is None:
        settings = MagicMock()
        settings.daemon.auth_token = token
        settings.daemon.upload_dir = tempfile.mkdtemp()
        settings.daemon.max_upload_bytes = 500 * 1024 * 1024
    app.state.settings = settings
    return app


@pytest.fixture
def mock_queries():
    q = AsyncMock()
    q.list_gpus = AsyncMock(
        return_value=[
            Gpu(id="GPU-001", name="Test GPU", memory_mb=8192, status=GpuStatus.idle),
        ]
    )
    return q


# ---- TestBearerAuth ----


class TestBearerAuth:
    """Test authentication middleware."""

    def test_health_without_token(self, mock_queries):
        """Health endpoint is public, accessible without token."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_protected_route_without_token(self, mock_queries):
        """Authenticated endpoints require a token."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/gpus")
        assert resp.status_code == 401

    def test_wrong_token(self, mock_queries):
        """Wrong token returns 403."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/gpus", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_valid_token(self, mock_queries):
        """Correct token grants access."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/gpus", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200

    def test_no_token_configured_bypass(self, mock_queries):
        """When token is None (v0.1 compat), all requests pass through."""
        app = _make_app(mock_queries, token=None)
        client = TestClient(app)
        resp = client.get("/gpus")
        assert resp.status_code == 200

    def test_api_protected_route_without_token(self, mock_queries):
        """API endpoints are protected without token."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/api/gpus")
        assert resp.status_code == 401

    def test_api_protected_route_wrong_token(self, mock_queries):
        """API endpoints reject wrong tokens."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/api/gpus", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_api_protected_route_valid_token(self, mock_queries):
        """API endpoints accept correct tokens."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/api/gpus", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200

    def test_api_auth_check_without_token(self, mock_queries):
        """Auth check endpoint requires bearer token."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/api/auth/check")
        assert resp.status_code == 401

    def test_api_auth_check_wrong_token(self, mock_queries):
        """Auth check endpoint rejects invalid token."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/api/auth/check", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 403

    def test_api_auth_check_valid_token(self, mock_queries):
        """Auth check endpoint accepts valid token and returns ok payload."""
        app = _make_app(mock_queries, token="secret")
        client = TestClient(app)
        resp = client.get("/api/auth/check", headers={"Authorization": "Bearer secret"})
        assert resp.status_code == 200
        # The legacy global token maps to an admin principal (P8: role exposed
        # so the dashboard can gate write UI up front).
        assert resp.json() == {"ok": True, "role": "admin"}

    def test_shell_and_login_public_get_routes(self, mock_queries):
        """Dashboard shell routes stay reachable without auth token."""
        app = _make_app(mock_queries, token="secret")

        @app.get("/")
        async def shell_root():
            return {"ok": True}

        @app.get("/login")
        async def shell_login():
            return {"ok": True}

        client = TestClient(app)
        assert client.get("/").status_code == 200
        assert client.get("/login").status_code == 200


# ---- TestZipCreation ----


class TestZipCreation:
    """Test local ZIP creation logic."""

    def test_exclusion_patterns(self):
        assert should_exclude(".git/config") is True
        assert should_exclude("__pycache__/module.cpython-311.pyc") is True
        assert should_exclude("model.pyc") is True
        assert should_exclude("data/dataset.csv") is True
        assert should_exclude("train.py") is False
        assert should_exclude("utils/helpers.py") is False

    def test_create_zip_contents(self, tmp_path):
        # Create a small project directory
        (tmp_path / "main.py").write_text("print('main')")
        (tmp_path / "utils.py").write_text("print('utils')")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"bytecode")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")

        zip_bytes = create_dir_zip(tmp_path)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "main.py" in names
            assert "utils.py" in names
            # Excluded files should not be present
            assert not any(".git" in n for n in names)
            assert not any("__pycache__" in n for n in names)

    def test_check_upload_size_ok(self):
        check_upload_size(b"x" * 100, max_bytes=200)  # should not raise

    def test_check_upload_size_exceeded(self):
        with pytest.raises(ValueError, match="limit"):
            check_upload_size(b"x" * 300, max_bytes=200)


# ---- TestTokenGeneration ----


class TestTokenGeneration:
    """Test auth token generation."""

    def test_token_length(self):
        token = generate_auth_token()
        # secrets.token_urlsafe(32) produces 43 characters
        assert len(token) == 43

    def test_token_uniqueness(self):
        tokens = {generate_auth_token() for _ in range(100)}
        assert len(tokens) == 100  # all unique


# ---- TestClientConfig ----


class TestClientConfig:
    """Test client configuration resolution."""

    def test_localhost_fallback(self, tmp_path):
        """Without config, falls back to localhost with no token."""
        config_path = tmp_path / "config.toml"
        host, port, token = get_client_config(config_path)
        assert host == "127.0.0.1"
        assert port == 9321
        assert token is None

    def test_remote_config_resolution(self, tmp_path):
        """With [client] section, resolves to remote host."""
        import tomli_w

        config_path = tmp_path / "config.toml"
        config_data = {
            "client": {
                "remote_host": "192.168.1.100",
                "remote_port": 9321,
                "auth_token": "my-secret-token",
            }
        }
        with open(config_path, "wb") as f:
            tomli_w.dump(config_data, f)

        host, port, token = get_client_config(config_path)
        assert host == "192.168.1.100"
        assert port == 9321
        assert token == "my-secret-token"

    def test_local_config_with_daemon_token(self, tmp_path):
        """Without [client], daemon.auth_token is used."""
        import tomli_w

        config_path = tmp_path / "config.toml"
        config_data = {
            "daemon": {
                "host": "0.0.0.0",
                "port": 9321,
                "auth_token": "daemon-token",
            }
        }
        with open(config_path, "wb") as f:
            tomli_w.dump(config_data, f)

        host, port, token = get_client_config(config_path)
        assert host == "127.0.0.1"  # no remote_host → localhost
        assert token == "daemon-token"
