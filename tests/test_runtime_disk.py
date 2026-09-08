"""Unit tests for the P14 WP-0 runtime additions (disk_usage, list_images_detailed)."""

from __future__ import annotations

import docker.errors
import pytest

from nerdit.core.runtime.docker import DockerRuntime
from nerdit.core.runtime.stub import StubRuntime


class _FakeImage:
    def __init__(self, tags, short_id, size, attrs=None):
        self.tags = tags
        self.short_id = short_id
        self.id = short_id
        self.attrs = {"Size": size, **(attrs or {})}


class _FakeImages:
    def __init__(self, images):
        self._images = images

    def list(self):
        return self._images


class _FakeClient:
    def __init__(self, images=None, df=None, df_error=None):
        self.images = _FakeImages(images or [])
        self._df = df
        self._df_error = df_error

    def df(self):
        if self._df_error is not None:
            raise self._df_error
        return self._df


@pytest.mark.asyncio
async def test_list_images_detailed_projection():
    client = _FakeClient(
        images=[
            _FakeImage(["nerdit-app/foo:1", "nerdit-app/foo:latest"], "sha256:abc", 1000),
            _FakeImage(["<none>:<none>"], "sha256:def", 500),
            _FakeImage([], "sha256:ghi", 300),
        ]
    )
    rt = DockerRuntime(client=client)
    out = await rt.list_images_detailed()
    # Two tagged entries; the <none> tag and the untagged image are skipped.
    assert {e["repo_tag"] for e in out} == {"nerdit-app/foo:1", "nerdit-app/foo:latest"}
    for e in out:
        assert e["id"] == "sha256:abc"
        assert e["size_bytes"] == 1000
        # No labels on the fake image ⇒ no recorded owner (Track 0.4).
        assert e["instance"] is None


@pytest.mark.asyncio
async def test_list_images_detailed_projects_the_instance_label():
    """The ``nerdit-instance`` label is surfaced as ``instance`` (Track 0.4).

    Both docker label shapes are accepted (``Config.Labels`` on an inspect
    payload, a flat ``Labels`` on a listing), and an image with labels but no
    ``nerdit-instance`` reads as unowned.
    """
    client = _FakeClient(
        images=[
            _FakeImage(
                ["nerdit-app/a:1"],
                "sha256:a",
                10,
                attrs={"Config": {"Labels": {"managed-by": "nerdit", "nerdit-instance": "beta"}}},
            ),
            _FakeImage(
                ["nerdit-app/b:1"], "sha256:b", 10, attrs={"Labels": {"nerdit-instance": "alpha"}}
            ),
            _FakeImage(["other/c:1"], "sha256:c", 10, attrs={"Config": {"Labels": {"foo": "bar"}}}),
        ]
    )
    rt = DockerRuntime(client=client)
    out = {e["repo_tag"]: e["instance"] for e in await rt.list_images_detailed()}
    assert out == {"nerdit-app/a:1": "beta", "nerdit-app/b:1": "alpha", "other/c:1": None}


@pytest.mark.asyncio
async def test_list_images_detailed_error_degrades_to_empty():
    class _Boom(_FakeImages):
        def list(self):
            raise RuntimeError("dockerd down")

    client = _FakeClient()
    client.images = _Boom([])
    rt = DockerRuntime(client=client)
    assert await rt.list_images_detailed() == []


@pytest.mark.asyncio
async def test_disk_usage_sums_report():
    df = {
        "Images": [{"Size": 100}, {"Size": 200}],
        "Containers": [{"SizeRw": 10}, {"SizeRw": 5}],
        "Volumes": [
            {"UsageData": {"Size": 40}},
            {"UsageData": {"Size": -1}},  # docker reports -1 when unknown → skipped
            {"Name": "novusage"},
        ],
        "BuildCache": [{"Size": 7}, {"Size": 3}],
    }
    rt = DockerRuntime(client=_FakeClient(df=df))
    out = await rt.disk_usage()
    assert out == {
        "images_bytes": 300,
        "containers_bytes": 15,
        "volumes_bytes": 40,
        "build_cache_bytes": 10,
    }


@pytest.mark.asyncio
async def test_disk_usage_apierror_is_none():
    err = docker.errors.APIError("boom")
    rt = DockerRuntime(client=_FakeClient(df_error=err))
    assert await rt.disk_usage() is None


@pytest.mark.asyncio
async def test_stub_runtime_new_methods():
    stub = StubRuntime()
    assert await stub.list_images_detailed() == []
    assert await stub.disk_usage() is None
