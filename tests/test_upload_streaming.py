"""``daemon/uploads.py:extract_upload`` — size caps, spooling, traversal, bombs.

Driven through a minimal route that does nothing but call ``extract_upload``.
The real ingress is ``POST /deploy`` (multipart ZIP), but that route's tail
needs the whole build/controller stack; this suite is about the shared
extraction guards, so it exercises them directly and stays independent of any
one ingress.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.testclient import TestClient

from nerdit.daemon.errors import register_error_handlers
from nerdit.daemon.uploads import extract_upload
from nerdit.utils.ids import generate_id


def _extract_router() -> APIRouter:
    r = APIRouter()

    @r.post("/extract", status_code=201)
    async def _extract(request: Request, archive: UploadFile) -> dict:
        settings = request.app.state.settings
        context_dir = await extract_upload(
            archive,
            max_bytes=settings["max_upload_bytes"],
            upload_root=Path(settings["upload_dir"]).expanduser(),
            dest_id=generate_id(),
        )
        return {"dir": str(context_dir)}

    return r


def _make_app(*, upload_dir: str | None = None, max_upload_bytes: int | None = None) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(_extract_router())
    app.state.settings = {
        "upload_dir": upload_dir or tempfile.mkdtemp(),
        "max_upload_bytes": max_upload_bytes or 500 * 1024 * 1024,
    }
    return app


def _make_zip(files: dict[str, str], compress: bool = False) -> bytes:
    buf = io.BytesIO()
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, "w", compression=mode) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _post(client: TestClient, zip_bytes: bytes):
    return client.post(
        "/extract",
        files={"archive": ("scripts.zip", zip_bytes, "application/zip")},
    )


class TestUploadStreaming:
    """Streaming upload and extraction-guard behaviors."""

    def test_small_upload_via_spool_memory_path(self):
        client = TestClient(_make_app())
        resp = _post(client, _make_zip({"train.py": "print('ok')"}))
        assert resp.status_code == 201
        assert (Path(resp.json()["dir"]) / "train.py").is_file()

    def test_upload_spilling_past_spool_threshold(self):
        """Archives past Starlette's in-memory spool (1 MiB) still extract."""
        client = TestClient(_make_app())
        large_payload = "x" * (2 * 1024 * 1024)
        zip_bytes = _make_zip({"train.py": "print('ok')", "big.txt": large_payload})

        resp = _post(client, zip_bytes)
        assert resp.status_code == 201

    def test_body_exceeding_cap_rejected_with_413(self):
        """When the streamed body exceeds max_upload_bytes, return 413 (not OOM)."""
        client = TestClient(_make_app(max_upload_bytes=512))

        zip_bytes = _make_zip({"train.py": "x" * 5000})
        assert len(zip_bytes) > 512

        resp = _post(client, zip_bytes)
        assert resp.status_code == 413
        body = resp.json()
        assert body["code"] == "payload_too_large"
        assert "too large" in body["detail"]

    def test_non_zip_body_rejected(self):
        client = TestClient(_make_app())
        resp = _post(client, b"not a zip file")
        assert resp.status_code == 400
        assert resp.json()["code"] == "deploy.invalid_zip"

    def test_upload_path_traversal_still_rejected(self, tmp_path):
        """Existing safety checks remain enforced after streaming refactor."""
        upload_root = str(tmp_path / "uploads")
        client = TestClient(_make_app(upload_dir=upload_root))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../escape.py", "print('bad')")
            zf.writestr("train.py", "print('ok')")

        resp = _post(client, buf.getvalue())
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "deploy.unsafe_path"
        assert "unsafe path" in body["detail"]
        # bug_005: a failed extraction leaves no partial tree behind.
        assert list(Path(upload_root).iterdir()) == []

    def test_upload_zip_bomb_still_rejected(self, tmp_path):
        """Zip-bomb detection (uncompressed size > 10x cap) still fires."""
        upload_root = str(tmp_path / "uploads")
        # 4 KB cap → 40 KB uncompressed max.
        client = TestClient(_make_app(upload_dir=upload_root, max_upload_bytes=4 * 1024))

        # Highly-compressible payload → large uncompressed, small compressed
        payload = "A" * (200 * 1024)
        zip_bytes = _make_zip({"train.py": payload}, compress=True)
        # The compressed archive must stay under the cap for the test to exercise
        # zip-bomb detection (vs. compressed-size rejection).
        assert len(zip_bytes) < 4 * 1024

        resp = _post(client, zip_bytes)
        assert resp.status_code == 413
        body = resp.json()
        assert body["code"] == "payload_too_large"
        assert "uncompressed" in body["detail"]
        assert list(Path(upload_root).iterdir()) == []

    def test_successful_path_leaves_extracted_tree(self, tmp_path):
        """Regression guard: the cleanup block must NOT fire on success."""
        upload_root = str(tmp_path / "uploads")
        client = TestClient(_make_app(upload_dir=upload_root))

        resp = _post(client, _make_zip({"train.py": "print('ok')"}))
        assert resp.status_code == 201

        subdirs = list(Path(upload_root).iterdir())
        assert len(subdirs) == 1
        assert (subdirs[0] / "train.py").is_file()


@pytest.mark.parametrize("bad", [b"", b"PK\x03\x04garbage"])
def test_malformed_archives_are_400(bad):
    client = TestClient(_make_app())
    resp = _post(client, bad)
    assert resp.status_code == 400
    assert resp.json()["code"] == "deploy.invalid_zip"


@pytest.mark.parametrize(
    ("name", "alias"),
    [
        ("app/config", "app/config"),
        ("app/config", "./app/config"),
        ("app/config", "app//config"),
        ("app/config", "app/./config"),
        ("nerdit.toml", "NERDIT.TOML"),
        ("apps/api/nerdit.toml", "apps/API/NERDIT.TOML"),
        ("apps/caf\u00e9/nerdit.toml", "apps/cafe\u0301/nerdit.toml"),
    ],
)
def test_duplicate_extraction_paths_are_rejected(name, alias, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, "first")
        zf.writestr(alias, "second")
    upload_root = tmp_path / "uploads"
    resp = _post(TestClient(_make_app(upload_dir=str(upload_root))), buf.getvalue())
    assert resp.status_code == 400
    assert resp.json()["code"] == "deploy.invalid_zip"
    assert list(upload_root.iterdir()) == []


def test_preflight_allows_unrelated_case_distinct_paths():
    from nerdit.daemon.uploads import _read_member

    buf = io.BytesIO(
        _make_zip({"nerdit.toml": "declaration", "README": "first", "readme": "second"})
    )
    assert _read_member(buf, "nerdit.toml") == b"declaration"


class TestValidateAndExtractOffLoop:
    """(P29 D12) The blocking half now runs in a worker thread.

    ``_validate_and_extract`` owns zip validation, the bomb sum, the per-member
    ``resolve()`` traversal walk and ``extractall`` — everything that used to
    pin the event loop for the length of a deploy. These pin its three error
    codes at the unit level and prove the loop stays responsive during a large
    extraction.
    """

    def test_rejects_non_zip(self, tmp_path):
        from nerdit.daemon.errors import NerditError
        from nerdit.daemon.uploads import _validate_and_extract

        tmp = io.BytesIO(b"not a zip file")
        with pytest.raises(NerditError) as exc:
            _validate_and_extract(tmp, tmp_path, 1024 * 1024)
        assert exc.value.code == "deploy.invalid_zip"

    def test_rejects_traversal_member(self, tmp_path):
        from nerdit.daemon.errors import NerditError
        from nerdit.daemon.uploads import _validate_and_extract

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../escape.py", "print('bad')")
        buf.seek(0)

        with pytest.raises(NerditError) as exc:
            _validate_and_extract(buf, tmp_path, 1024 * 1024)
        assert exc.value.code == "deploy.unsafe_path"

    def test_rejects_zip_bomb(self, tmp_path):
        from nerdit.daemon.errors import NerditError
        from nerdit.daemon.uploads import _validate_and_extract

        payload = "A" * (200 * 1024)
        buf = io.BytesIO(_make_zip({"train.py": payload}, compress=True))

        with pytest.raises(NerditError) as exc:
            _validate_and_extract(buf, tmp_path, 4 * 1024)
        assert exc.value.code == "payload_too_large"
        assert "uncompressed" in exc.value.message

    def test_extracts_on_the_happy_path(self, tmp_path):
        from nerdit.daemon.uploads import _validate_and_extract

        buf = io.BytesIO(_make_zip({"a/train.py": "print('ok')"}))
        _validate_and_extract(buf, tmp_path, 1024 * 1024)
        assert (tmp_path / "a" / "train.py").read_text() == "print('ok')"

    @pytest.mark.asyncio
    async def test_loop_stays_responsive_during_extraction(self, tmp_path):
        """A concurrent task makes progress while a large archive extracts.

        The point of the fix: before it, the whole extraction ran inline on the
        loop and this counter would be stuck at 0 when the extract returned.
        """
        import asyncio

        from fastapi import UploadFile

        from nerdit.daemon.uploads import extract_upload

        # ~1000 members: enough resolve() + write syscalls that a blocked loop
        # is unmistakable, small enough to stay fast.
        zip_bytes = _make_zip({f"pkg/mod_{i}.py": f"VALUE = {i}\n" * 40 for i in range(1000)})

        ticks = 0

        async def _tick() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.001)
                ticks += 1

        ticker = asyncio.create_task(_tick())
        try:
            upload = UploadFile(filename="app.zip", file=io.BytesIO(zip_bytes))
            out = await extract_upload(
                upload,
                max_bytes=50 * 1024 * 1024,
                upload_root=tmp_path,
                dest_id="ctx1",
            )
        finally:
            ticker.cancel()

        assert (out / "pkg" / "mod_999.py").is_file()
        assert ticks > 0, "the event loop never ran another task during the extraction"


@pytest.mark.parametrize("path", ["/api/deploy", "/api/projects/asso/apply"])
@pytest.mark.parametrize(
    ("declared", "status", "code"),
    [
        (str(512 + 1_048_576 + 1), 413, "payload_too_large"),
        (None, 411, "length_required"),
        ("100", 200, None),
    ],
)
@pytest.mark.asyncio
async def test_upload_routes_bounded_before_parsing(path, declared, status, code):
    """The multipart ingresses are refused on the declared length, unread."""
    import json

    from nerdit.daemon.bodylimit import BodyLimitMiddleware

    called: list[str] = []
    app = FastAPI()

    @app.post(path)
    async def _spy() -> dict:
        called.append(path)
        return {}

    app.add_middleware(BodyLimitMiddleware, max_upload_bytes=512)
    headers = [(b"content-type", b"multipart/form-data; boundary=x")]
    if declared is not None:
        headers.append((b"content-length", declared.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1),
        "server": ("localhost", 80),
    }
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await app(scope, receive, send)
    assert sent[0]["status"] == status
    if code is None:
        assert called == [path]
    else:
        assert called == []
        body = json.loads(b"".join(m.get("body", b"") for m in sent[1:]))
        assert body["code"] == code
