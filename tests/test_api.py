"""Tests for the FastAPI endpoints with mocked dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.gpus import router as gpus_router
from nerdit.daemon.routes.health import router as health_router
from nerdit.daemon.routes.images import router as images_router
from nerdit.db.models import Gpu, GpuStatus


def _make_app(queries, monitor=None, settings=None, runtime=None) -> FastAPI:
    """Create a test FastAPI app with mocked state."""
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(gpus_router)
    app.include_router(images_router)

    app.state.queries = queries
    app.state.monitor = monitor or MagicMock(get_metrics=MagicMock(return_value={}))
    if runtime is None:
        runtime = AsyncMock()
        runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime
    if settings is None:
        settings = MagicMock()
        settings.daemon.auth_token = None
    app.state.settings = settings
    # Mirror production's no-token path: the auth middleware attaches the LOCAL
    # admin principal so routes have a principal (current_principal now
    # fail-closes to ANONYMOUS/readonly when none is attached — I1).
    app.add_middleware(ScopedTokenAuthMiddleware, token=None)
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


@pytest.fixture
def client(mock_queries):
    app = _make_app(mock_queries)
    return TestClient(app, base_url="http://127.0.0.1")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["gpu_count"] == 1
    # The batch counter is gone with the batch HTTP surface (simplification WP2).
    assert "jobs_running" not in data


def test_health_excludes_offline_gpu_records(mock_queries):
    mock_queries.list_gpus = AsyncMock(
        return_value=[
            Gpu(
                id="gpu:amd:0",
                name="Missing AMD GPU",
                memory_mb=0,
                status=GpuStatus.offline,
            )
        ]
    )
    app = _make_app(mock_queries)

    response = TestClient(app).get("/health")

    assert response.json()["gpu_count"] == 0


def test_list_gpus(client):
    resp = client.get("/gpus")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "Test GPU"
    assert data[0]["vendor"] == "nvidia"
    assert data[0]["schedulable"] is True


def test_list_images_endpoint(mock_queries):
    runtime = AsyncMock()
    runtime.list_images = AsyncMock(return_value=["nerdit-runtime-rocm:0.1", "nerdit-runtime:0.1"])
    app = _make_app(mock_queries, runtime=runtime)
    client = TestClient(app, base_url="http://127.0.0.1")

    resp = client.get("/images")
    assert resp.status_code == 200
    assert resp.json() == ["nerdit-runtime-rocm:0.1", "nerdit-runtime:0.1"]
