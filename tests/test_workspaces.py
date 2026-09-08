"""core/workspaces unit suite (P29 / WP1) — the guard family, the caps, the writer.

Pure: ``tmp_path`` is the ``data_dir``, no daemon, no FastAPI. The path/name
matrices are parametrized accept/reject pairs over the same helper so every
guard is falsified both ways — widening a rule means deleting an accept-set
member, which fails loudly.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import threading
import zipfile
from pathlib import Path

import pytest

from nerdit.config.defaults import (
    WORKSPACE_MAX_FILE_BYTES,
    WORKSPACE_MAX_FILES,
    WORKSPACE_MAX_TOTAL_BYTES,
)
from nerdit.core import workspaces
from nerdit.core.workspaces import (
    WORKSPACE_MAX_PATH_BYTES,
    WORKSPACE_MAX_SEGMENT_BYTES,
    WorkspaceError,
    ensure_workspace_dirs,
    list_files,
    read_file,
    read_meta,
    settled_to_thread,
    tree_root,
    validate_file_path,
    workspace_lock,
    workspace_root,
    write_files,
    zip_workspace,
)

NAME = "my-app"


def _write(tmp_path: Path, files: dict[str, str], delete: list[str] | None = None, owner="tok-1"):
    # ``existing_meta`` is the caller's read (the route's, under the lock): the
    # writer never re-reads the sidecar it was handed a decision about.
    return write_files(
        tmp_path,
        NAME,
        files,
        delete or [],
        owner_token_id=owner,
        existing_meta=read_meta(tmp_path, NAME),
    )


# --- path grammar ----------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "main.py",
        "src/app/util.py",
        ".gitignore",
        "static/index.html",
        "requirements.txt",
    ],
)
def test_validate_file_path_accepts_reasonable_relative_paths(path):
    validate_file_path(path)  # must not raise


@pytest.mark.parametrize(
    "path",
    [
        "",  # empty
        "/abs.py",  # absolute
        "..",  # bare parent
        "../x",  # leading traversal
        "a/../b",  # embedded traversal
        "a//b",  # doubled separator
        "/",  # root
        "a/",  # trailing separator
        "a/./b",  # embedded curdir
        ".",  # bare curdir
        "a\x00b",  # NUL byte
        "a\\b",  # backslash separator
        "C:\\x",  # windows drive + backslash
        "C:/x",  # windows drive + slash
    ],
)
def test_validate_file_path_rejects_the_traversal_matrix(path):
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.invalid_path"
    assert exc.value.status_code == 422


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        "node_modules/x/y.js",
        "__pycache__/m.pyc",
        "x.pyc",
        ".venv/bin/act",
        "data/db.sqlite",
    ],
)
def test_excluded_segments_rejected(path):
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.excluded_path"


@pytest.mark.parametrize("path", ["datafiles/x.txt", "gitthing/.keep", "pycache/x.py"])
def test_exclusion_near_misses_accepted(path):
    validate_file_path(path)  # widening ZIP_EXCLUDE_PATTERNS must fail here


@pytest.mark.parametrize("path", [".env", ".env.local", "config/.env", "a/.env.production"])
def test_secret_basenames_refused(path):
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.secret_file"
    assert "secrets" in (exc.value.hint or "")


@pytest.mark.parametrize("path", ["env", ".envrc", "environment.py", "dotenv.py"])
def test_secret_lookalikes_accepted(path):
    validate_file_path(path)


# --- content rules ---------------------------------------------------------


@pytest.mark.parametrize("content", ["a\x00b", "lone \ud800 surrogate"])
def test_nul_and_undecodable_content_refused(tmp_path, content):
    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"main.py": content})
    assert exc.value.code == "workspace.binary_content"
    assert "main.py" in exc.value.message
    assert not (tmp_path / "workspaces" / NAME / "meta.json").exists()


def test_utf8_unicode_content_roundtrips(tmp_path):
    body = "# héllo — 日本語 🚀\n"
    _write(tmp_path, {"main.py": body})
    assert read_file(tmp_path, NAME, "main.py") == body


# --- caps ------------------------------------------------------------------


def test_per_file_cap_boundary_both_ways(tmp_path):
    _write(tmp_path, {"big.txt": "a" * WORKSPACE_MAX_FILE_BYTES})
    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"big.txt": "a" * (WORKSPACE_MAX_FILE_BYTES + 1)})
    assert exc.value.code == "workspace.file_too_large"
    assert exc.value.extra["detail"]["limit"] == 262144
    assert exc.value.extra["limit"] == 262144


def test_total_cap_boundary_both_ways(tmp_path):
    chunk = "a" * WORKSPACE_MAX_FILE_BYTES
    full = {
        f"f{i}.txt": chunk for i in range(WORKSPACE_MAX_TOTAL_BYTES // WORKSPACE_MAX_FILE_BYTES)
    }
    summary = _write(tmp_path, full)
    assert summary["total_bytes"] == WORKSPACE_MAX_TOTAL_BYTES

    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"extra.txt": "x"})
    assert exc.value.code == "workspace.too_large"
    assert exc.value.extra["limit"] == WORKSPACE_MAX_TOTAL_BYTES

    # Post-state, not sum-of-request: shrinking one file makes room for another
    # in the SAME batch even though the request's own bytes exceed the headroom.
    summary = _write(tmp_path, {"f0.txt": "small", "extra.txt": "x"})
    assert summary["file_count"] == len(full) + 1
    assert summary["total_bytes"] < WORKSPACE_MAX_TOTAL_BYTES


def test_file_count_cap_boundary_both_ways(tmp_path):
    _write(tmp_path, {f"f{i}.txt": "x" for i in range(WORKSPACE_MAX_FILES)})
    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"one-more.txt": "x"})
    assert exc.value.code == "workspace.too_many_files"
    assert exc.value.extra["limit"] == 500
    assert exc.value.extra["detail"]["limit"] == 500
    # Delete-one/add-one keeps the post-state at exactly the cap.
    summary = _write(tmp_path, {"one-more.txt": "x"}, delete=["f0.txt"])
    assert summary["file_count"] == WORKSPACE_MAX_FILES


# --- batch atomicity -------------------------------------------------------


def test_one_bad_entry_writes_zero_files(tmp_path):
    with pytest.raises(WorkspaceError) as exc:
        _write(
            tmp_path,
            {"a.py": "1", "b.py": "2", "c.py": "3", "../escape.py": "4"},
        )
    assert exc.value.code == "workspace.invalid_path"
    assert not (tmp_path / "workspaces" / NAME / "tree").exists()
    assert not (tmp_path / "workspaces" / NAME / "meta.json").exists()


def test_atomic_replace_never_tears_existing_file(tmp_path, monkeypatch):
    _write(tmp_path, {"a.py": "v1", "b.py": "v1"})
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash between tmp and replace")
        return real_replace(src, dst)

    monkeypatch.setattr(workspaces.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        _write(tmp_path, {"a.py": "v2", "b.py": "v2"})
    monkeypatch.undo()

    # The file whose replace() blew up still reads its PREVIOUS content intact —
    # a torn/truncated file is structurally impossible, and the tmp sibling is
    # swept rather than left behind for the next batch to trip over.
    assert read_file(tmp_path, NAME, "a.py") == "v2"
    assert read_file(tmp_path, NAME, "b.py") == "v1"
    assert not list(tree_root(tmp_path, NAME).rglob("*.tmp-*"))


@pytest.mark.parametrize(
    ("existing", "batch", "delete"),
    [
        # (a) the write target is an existing DIRECTORY -> os.replace() raised
        #     IsADirectoryError from the apply pass.
        ({"src/a.py": "1"}, {"good.py": "g", "src": "x"}, []),
        # (b) the write path sits under an existing FILE -> the intermediate
        #     mkdir() raised FileExistsError.
        ({"a.py": "1"}, {"good.py": "g", "a.py/b.py": "x"}, []),
        # (c) same-batch prefix conflict: 'a' cannot be both a file and a
        #     directory, and the post-state simulation counted it as two files.
        ({}, {"good.py": "g", "a": "1", "a/b": "2"}, []),
        # (d) a write under a path the SAME batch deletes (R15, one level up):
        #     deletes run after writes, so the file was still on disk.
        ({"a.py": "1"}, {"good.py": "g", "a.py/b.py": "2"}, ["a.py"]),
    ],
    ids=["target-is-a-dir", "ancestor-is-a-file", "same-batch-prefix", "under-a-deleted-path"],
)
def test_structural_conflicts_reject_the_whole_batch(tmp_path, existing, batch, delete):
    """File/directory collisions are validation failures, never mid-apply OSErrors.

    ``good.py`` is the pin: it is first in insertion order, so it HAD already
    landed when the collision blew up — the assertion below is what falsifies
    ``_check_write_structure``, not the error code alone. The meta assertion
    carries the same invariant to the first-ever write: a batch that never
    validated must not leave a tree with no ``meta.json`` (a workspace that
    "does not exist" yet holds content).
    """
    if existing:
        _write(tmp_path, existing)
    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, batch, delete=delete)
    assert exc.value.code == "workspace.invalid_path"
    assert not (tmp_path / "workspaces" / NAME / "tree" / "good.py").exists()
    assert (tmp_path / "workspaces" / NAME / "meta.json").exists() is bool(existing)
    for path, content in existing.items():
        assert read_file(tmp_path, NAME, path) == content


def test_rename_file_to_package_is_two_batches(tmp_path):
    """The refusal above is a contract, not a dead end.

    Case (d)'s post-state is perfectly legitimate — it just needs two calls.
    This pins the semantics chosen over reordering deletes before writes: the
    tree reaches the same place with the all-or-nothing batch contract intact.
    """
    _write(tmp_path, {"a.py": "1"})
    _write(tmp_path, {}, delete=["a.py"])
    summary = _write(tmp_path, {"a.py/b.py": "2"})
    assert summary["file_count"] == 1
    assert read_file(tmp_path, NAME, "a.py/b.py") == "2"


def test_structural_near_misses_accepted(tmp_path):
    """The structural check must not swallow ordinary batches.

    ``srcx`` is a string-prefix neighbour of ``src`` but not a *path* prefix, and
    this batch writes both as files — a conflict test built on ``startswith``
    instead of on path segments rejects this and nothing else. The rest is the
    everyday shape: a new nested directory beside an overwrite of a file that
    already lives under an existing one.
    """
    _write(tmp_path, {"pkg/a.py": "1"})
    summary = _write(tmp_path, {"src": "s", "srcx": "x", "pkg/b/c.py": "2", "pkg/a.py": "3"})
    assert summary["written"] == 4
    assert read_file(tmp_path, NAME, "src") == "s"
    assert read_file(tmp_path, NAME, "srcx") == "x"
    assert read_file(tmp_path, NAME, "pkg/a.py") == "3"


# --- batch semantics -------------------------------------------------------


def test_delete_and_rewrite_batch_summary_counts(tmp_path):
    _write(tmp_path, {"a.py": "1", "b.py": "2", "c.py": "3"})
    summary = _write(tmp_path, {"a.py": "one", "d.py": "4"}, delete=["b.py", "c.py"])
    assert summary["written"] == 2
    assert summary["deleted"] == 2
    assert summary["file_count"] == 2
    assert [e["path"] for e in summary["files"]] == ["a.py", "d.py"]
    assert summary["files"][0]["sha256"] == hashlib.sha256(b"one").hexdigest()
    with pytest.raises(WorkspaceError):
        read_file(tmp_path, NAME, "b.py")


def test_delete_missing_path_is_noop_counted_zero(tmp_path):
    _write(tmp_path, {"a.py": "1"})
    summary = _write(tmp_path, {}, delete=["ghost.py", "ghost.py"])
    assert summary["deleted"] == 0
    assert summary["file_count"] == 1


def test_path_in_both_files_and_delete_rejected(tmp_path):
    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"a.py": "1"}, delete=["a.py"])
    assert exc.value.code == "workspace.invalid_path"


# --- symlink containment ---------------------------------------------------


def test_symlink_parent_component_rejected(tmp_path):
    _write(tmp_path, {"keep.py": "1"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (tree_root(tmp_path, NAME) / "lib").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"lib/x.py": "pwned"})
    assert exc.value.code == "workspace.invalid_path"
    assert list(outside.iterdir()) == []


def test_symlinked_parent_inside_the_tree_still_rejected(tmp_path):
    """Isolates the per-segment walk from the ``resolve()`` containment check.

    A ``lib -> real`` symlink stays *inside* the tree, so the resolve/
    is_relative_to backstop is satisfied — only the ancestor walk catches it.
    Deleting the walk fails here and nowhere else.
    """
    _write(tmp_path, {"real/a.py": "1"})
    tree = tree_root(tmp_path, NAME)
    (tree / "lib").symlink_to(tree / "real", target_is_directory=True)

    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"lib/x.py": "sneaky"})
    assert exc.value.code == "workspace.invalid_path"
    assert not (tree / "real" / "x.py").exists()


def test_symlinked_workspace_root_refused(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (tmp_path / "workspaces").mkdir()
    (tmp_path / "workspaces" / NAME).symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceError) as exc:
        workspace_root(tmp_path, NAME)
    assert exc.value.code == "workspace.invalid_path"


def test_symlinked_workspaces_dir_refused(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (tmp_path / "workspaces").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceError) as exc:
        workspace_root(tmp_path, NAME)
    assert exc.value.code == "workspace.invalid_path"


# --- meta.json -------------------------------------------------------------


def test_first_write_stamps_owner_second_preserves_it(tmp_path):
    _write(tmp_path, {"a.py": "1"}, owner="tok-owner")
    first = read_meta(tmp_path, NAME)
    assert first is not None
    assert first["owner_token_id"] == "tok-owner"
    assert first["created_at"] == first["last_written_at"]

    _write(tmp_path, {"b.py": "2"}, owner="tok-intruder")
    second = read_meta(tmp_path, NAME)
    assert second is not None
    assert second["owner_token_id"] == "tok-owner"
    assert second["created_at"] == first["created_at"]
    assert second["last_written_at"] >= first["last_written_at"]


def test_absent_meta_still_none(tmp_path):
    """(review round-1) Absent means absent — the first write must still work.

    The regression pin for the fail-closed change: only a *present but
    untrustworthy* sidecar raises; a workspace that was never written reads
    ``None`` and takes its first write exactly as before.
    """
    assert read_meta(tmp_path, NAME) is None
    ensure_workspace_dirs(tmp_path, NAME)  # dirs but no sidecar
    assert read_meta(tmp_path, NAME) is None
    _write(tmp_path, {"a.py": "1"}, owner="tok-owner")
    assert (read_meta(tmp_path, NAME) or {})["owner_token_id"] == "tok-owner"


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        '["a", "list"]',  # valid JSON, wrong shape
        '"a string"',
        "null",
        "",
    ],
)
def test_corrupt_meta_raises_meta_corrupt(tmp_path, raw):
    """A malformed sidecar is NOT an absent one (Codex 3803274894).

    Reading it as absence let the write route treat an existing tree as a first
    write, so any in-scope submitter could re-stamp ``owner_token_id`` and take
    over the source. Fail closed instead; the message is path-free.
    """
    _write(tmp_path, {"a.py": "1"}, owner="tok-owner")
    (workspace_root(tmp_path, NAME) / "meta.json").write_text(raw, encoding="utf-8")

    with pytest.raises(WorkspaceError) as exc:
        read_meta(tmp_path, NAME)
    assert exc.value.status_code == 500
    assert exc.value.code == "workspace.meta_corrupt"
    assert str(tmp_path) not in exc.value.message
    assert str(tmp_path) not in (exc.value.hint or "")


def test_undecodable_meta_raises(tmp_path):
    """Invalid UTF-8 bytes are a corruption too, not an absence."""
    _write(tmp_path, {"a.py": "1"})
    (workspace_root(tmp_path, NAME) / "meta.json").write_bytes(b"\xff\xfe{}")
    with pytest.raises(WorkspaceError) as exc:
        read_meta(tmp_path, NAME)
    assert exc.value.code == "workspace.meta_corrupt"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through mode 0o000")
def test_unreadable_meta_raises(tmp_path):
    """EACCES is the same hole as a decode error: an authz record we cannot read."""
    _write(tmp_path, {"a.py": "1"})
    meta_path = workspace_root(tmp_path, NAME) / "meta.json"
    meta_path.chmod(0o000)
    try:
        with pytest.raises(WorkspaceError) as exc:
            read_meta(tmp_path, NAME)
        assert exc.value.code == "workspace.meta_corrupt"
    finally:
        meta_path.chmod(0o600)


def test_write_files_uses_the_caller_read_meta(tmp_path):
    """(review round-1) One read, one decision: the writer never re-reads.

    The route reads the sidecar under the lock and makes the owner decision on
    it; ``write_files`` is handed that value, so the same state can never get a
    second interpretation mid-request.
    """
    _write(tmp_path, {"a.py": "1"}, owner="tok-owner")
    # A corrupt sidecar on disk does not even get looked at — the caller's read
    # is authoritative (and in production the caller already 500'd on it).
    (workspace_root(tmp_path, NAME) / "meta.json").write_text("{not json", encoding="utf-8")
    summary = write_files(
        tmp_path,
        NAME,
        {"b.py": "2"},
        [],
        owner_token_id="tok-second",
        existing_meta={"owner_token_id": "tok-owner", "created_at": "2020-01-01T00:00:00+00:00"},
    )
    assert summary["written"] == 1
    meta = read_meta(tmp_path, NAME) or {}
    assert meta["owner_token_id"] == "tok-owner"
    assert meta["created_at"] == "2020-01-01T00:00:00+00:00"


def test_meta_outside_tree_never_listed_never_zipped(tmp_path):
    _write(tmp_path, {"a.py": "1"})
    assert [e["path"] for e in list_files(tmp_path, NAME)["files"]] == ["a.py"]
    assert _namelist(zip_workspace(tree_root(tmp_path, NAME))) == ["a.py"]


# --- listing / reading -----------------------------------------------------


def test_list_files_sorted_with_sha256_mtime_totals(tmp_path):
    _write(tmp_path, {"z.py": "zz", "a/b.py": "b", "a.py": "aaa"})
    listing = list_files(tmp_path, NAME)
    assert [e["path"] for e in listing["files"]] == ["a.py", "a/b.py", "z.py"]
    assert listing["file_count"] == 3
    assert listing["total_bytes"] == 3 + 1 + 2
    assert listing["files"][0]["sha256"] == hashlib.sha256(b"aaa").hexdigest()
    assert listing["files"][0]["mtime"].endswith("+00:00")


def test_read_file_roundtrip_and_missing_404(tmp_path):
    _write(tmp_path, {"src/app.py": "print(1)\n"})
    assert read_file(tmp_path, NAME, "src/app.py") == "print(1)\n"
    with pytest.raises(WorkspaceError) as exc:
        read_file(tmp_path, NAME, "src/nope.py")
    assert exc.value.code == "workspace.not_found"
    assert exc.value.status_code == 404
    assert "src/nope.py" in exc.value.message


def test_read_file_rejects_traversal_and_symlink(tmp_path):
    _write(tmp_path, {"a.py": "1"})
    tree = tree_root(tmp_path, NAME)
    secret = tmp_path / "outside.txt"
    secret.write_text("classified", encoding="utf-8")
    (tree / "leak.txt").symlink_to(secret)

    with pytest.raises(WorkspaceError) as exc:
        read_file(tmp_path, NAME, "../../outside.txt")
    assert exc.value.code == "workspace.invalid_path"
    with pytest.raises(WorkspaceError) as exc:
        read_file(tmp_path, NAME, "leak.txt")
    assert exc.value.code == "workspace.invalid_path"


def test_read_file_refuses_out_of_band_binary(tmp_path):
    """A non-UTF-8 file dropped onto disk stays inside the WorkspaceError contract.

    The writer refuses binary at write time (D-P29-2), so this is reachable only
    by out-of-band tamper — the same scenario ``zip_workspace`` re-filters
    against. It must surface as a structured 422, never as a raw
    ``UnicodeDecodeError`` 500-ing through the route layer.
    """
    _write(tmp_path, {"a.py": "1"})
    (tree_root(tmp_path, NAME) / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(WorkspaceError) as exc:
        read_file(tmp_path, NAME, "blob.bin")
    assert exc.value.code == "workspace.binary_content"
    assert exc.value.status_code == 422
    assert "blob.bin" in exc.value.message


# --- the zip ---------------------------------------------------------------


def _namelist(blob: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return zf.namelist()


def test_zip_deterministic_bytes(tmp_path):
    _write(tmp_path, {"a.py": "1", "src/b.py": "2"})
    tree = tree_root(tmp_path, NAME)
    first = zip_workspace(tree)
    assert zip_workspace(tree) == first
    _write(tmp_path, {"a.py": "1"})  # same content, rewritten
    # A different mtime must not move a byte — the ZipInfo date_time is pinned,
    # which is what makes two snapshots of identical content identical archives.
    os.utime(tree / "a.py", (1_000_000_000, 1_000_000_000))
    os.utime(tree / "src" / "b.py", (1_600_000_000, 1_600_000_000))
    assert zip_workspace(tree) == first


def test_zip_reapplies_excludes_and_secret_names(tmp_path):
    _write(tmp_path, {"main.py": "ok"})
    tree = tree_root(tmp_path, NAME)
    # Dropped straight onto disk, bypassing the write-time guards.
    (tree / "node_modules").mkdir()
    (tree / "node_modules" / "x.js").write_text("junk", encoding="utf-8")
    (tree / ".env").write_text("SECRET=1", encoding="utf-8")
    assert _namelist(zip_workspace(tree)) == ["main.py"]


def test_zip_empty_tree_raises_workspace_empty(tmp_path):
    ensure_workspace_dirs(tmp_path, NAME)
    tree = tree_root(tmp_path, NAME)
    with pytest.raises(WorkspaceError) as exc:
        zip_workspace(tree)
    assert exc.value.code == "workspace.empty"

    (tree / ".env").write_text("SECRET=1", encoding="utf-8")
    with pytest.raises(WorkspaceError) as exc:
        zip_workspace(tree)
    assert exc.value.code == "workspace.empty"


def test_zip_skips_symlinked_files(tmp_path):
    _write(tmp_path, {"main.py": "ok"})
    tree = tree_root(tmp_path, NAME)
    secret = tmp_path / "outside.txt"
    secret.write_text("classified", encoding="utf-8")
    (tree / "leak.txt").symlink_to(secret)
    assert _namelist(zip_workspace(tree)) == ["main.py"]


# --- lock registry ---------------------------------------------------------


async def test_workspace_lock_registry():
    lock = workspace_lock("app-a")
    assert workspace_lock("app-a") is lock
    assert workspace_lock("app-b") is not lock
    assert not lock.locked()
    async with lock:
        assert workspace_lock("app-a").locked()
    assert not lock.locked()


async def test_settled_to_thread_waits_for_cancelled_worker():
    """(Codex 3803274892) A cancellation must not ABANDON the worker thread.

    With a bare ``asyncio.to_thread`` the await returns the moment the cancel
    lands and the surrounding ``async with workspace_lock`` unwinds — freeing
    the lock while the thread is still writing. Here the ``CancelledError``
    surfaces only AFTER the worker function returned.
    """
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _worker() -> str:
        started.set()
        release.wait(5)
        finished.set()
        return "done"

    task = asyncio.create_task(settled_to_thread(_worker))
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done(), "the await returned while the worker was still running"
    assert not finished.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_settled_to_thread_is_transparent_when_not_cancelled():
    """The happy path is plain ``to_thread``: same result, same exception."""
    assert await settled_to_thread(lambda a, b=0: a + b, 2, b=3) == 5

    def _boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        await settled_to_thread(_boom)


async def test_settled_to_thread_releases_the_lock_only_after_the_worker(tmp_path):
    """The invariant the helper exists for, asserted on the lock itself."""
    release = threading.Event()
    started = threading.Event()
    lock = workspace_lock("settle-app")

    async def _held() -> None:
        async with lock:
            await settled_to_thread(lambda: (started.set(), release.wait(5)))

    task = asyncio.create_task(_held())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert lock.locked(), "the lock was freed with the worker thread still running"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()


# --- perms and names -------------------------------------------------------


def test_dir_perms_enforced_on_every_write(tmp_path):
    _write(tmp_path, {"src/a.py": "1"})
    root = workspace_root(tmp_path, NAME)
    os.chmod(root, 0o755)
    _write(tmp_path, {"src/b.py": "2"})
    for level in (tmp_path / "workspaces", root, root / "tree", root / "tree" / "src"):
        assert oct(level.stat().st_mode & 0o777) == "0o700"
    assert oct((root / "tree" / "src" / "a.py").stat().st_mode & 0o777) == "0o600"
    assert oct((root / "meta.json").stat().st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("name", ["UPPER", "-x", "a_b", "a" * 64])
def test_workspace_name_grammar_rejected(tmp_path, name):
    with pytest.raises(WorkspaceError) as exc:
        workspace_root(tmp_path, name)
    assert exc.value.code == "workspace.invalid_path"


def test_workspace_name_grammar_accepted(tmp_path):
    assert workspace_root(tmp_path, "my-app").name == "my-app"


# --- C6: casefold collisions + case-insensitive name guards ------------------


@pytest.mark.parametrize(
    ("seed", "files", "delete"),
    [
        # The reported bug: on default macOS APFS the write lands on the
        # existing inode and the delete then unlinks the caller's new file.
        pytest.param({"app.py": "old\n"}, {"App.py": "new\n"}, ["app.py"], id="write-vs-delete"),
        # Two spellings of one path in the same batch.
        pytest.param({}, {"App.py": "a\n", "app.py": "b\n"}, [], id="write-vs-write"),
        # A write against an existing case variant that the batch does NOT delete.
        pytest.param({"app.py": "old\n"}, {"App.py": "new\n"}, [], id="write-vs-existing"),
        # The mkdir FileExistsError hole: a write's ANCESTOR vs an existing file.
        pytest.param({"A": "file\n"}, {"a/x.py": "y\n"}, [], id="ancestor-vs-existing-file"),
        # A directory already spelled the other way.
        pytest.param({"Src/a.py": "a\n"}, {"src/b.py": "b\n"}, [], id="ancestor-vs-ancestor"),
        # A delete naming an existing path only up to case.
        pytest.param({"App.py": "old\n"}, {}, ["app.py"], id="delete-vs-existing"),
        # A case rename in ONE batch: writes are applied before deletes, so a
        # case-sensitive host keeps the new file and a case-insensitive one
        # unlinks it. Two batches, exactly like the file→package rename.
        pytest.param({"app.py": "o\n"}, {"App.py": "n\n"}, ["app.py"], id="case-rename-one-batch"),
    ],
)
def test_casefold_collisions_reject_the_whole_batch(tmp_path, seed, files, delete):
    if seed:
        _write(tmp_path, seed)
    before = {e["path"]: e["sha256"] for e in list_files(tmp_path, NAME)["files"]}

    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, files, delete)
    assert exc.value.code == "workspace.invalid_path"
    assert exc.value.status_code == 422
    assert "case" in exc.value.message.lower()
    # All-or-nothing: not one byte of the refused batch landed.
    assert {e["path"]: e["sha256"] for e in list_files(tmp_path, NAME)["files"]} == before


@pytest.mark.parametrize(
    ("seed", "files", "delete"),
    [
        pytest.param({"app.py": "old\n"}, {"app.py": "new\n"}, [], id="exact-rewrite"),
        pytest.param({"app.py": "old\n"}, {}, ["app.py"], id="exact-delete"),
        pytest.param({}, {"a/x.py": "x\n", "a/y.py": "y\n"}, [], id="same-dir-two-files"),
    ],
)
def test_casefold_near_misses_accepted(tmp_path, seed, files, delete):
    if seed:
        _write(tmp_path, seed)
    _write(tmp_path, files, delete)  # must not raise


@pytest.mark.parametrize("path", [".ENV", ".Env.local", "src/.EnV.production"])
def test_secret_basename_guard_is_case_insensitive(path):
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.secret_file"


@pytest.mark.parametrize(
    "path", ["NODE_MODULES/x.js", "a/X.PYC", "__PYCACHE__/m.pyc", "Data/set.csv", ".GIT/config"]
)
def test_exclude_guard_is_case_insensitive(path):
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.excluded_path"


def test_zip_reapplies_the_casefolded_guards(tmp_path):
    """Belt-and-braces: out-of-band files in the variant spelling never deploy."""
    _write(tmp_path, {"main.py": "ok\n"})
    tree = tree_root(tmp_path, NAME)
    (tree / "NODE_MODULES").mkdir()
    (tree / "NODE_MODULES" / "x.js").write_text("nope\n", encoding="utf-8")
    (tree / ".ENV").write_text("K=v\n", encoding="utf-8")

    assert _namelist(zip_workspace(tree)) == ["main.py"]


# --- C7: *.tmp-* residue -----------------------------------------------------


@pytest.mark.parametrize("path", ["main.py.tmp-1234", "src/x.tmp-99", "a.TMP-7"])
def test_tmp_basenames_are_refused(path):
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.invalid_path"
    assert "temporary" in exc.value.message.lower()


def test_stale_tmp_residue_is_invisible_and_swept(tmp_path):
    """A hard crash between tmp create and ``os.replace`` used to leave residue
    inside ``tree/`` forever — counted against the caps, listed, and zipped."""
    _write(tmp_path, {"main.py": "ok\n"})
    tree = tree_root(tmp_path, NAME)
    stale = tree / "main.py.tmp-99999"
    stale.write_text("half a file", encoding="utf-8")
    meta_stale = tree.parent / "meta.json.tmp-99999"
    meta_stale.write_text("{", encoding="utf-8")

    listing = list_files(tmp_path, NAME)
    assert [e["path"] for e in listing["files"]] == ["main.py"]
    assert listing["total_bytes"] == len("ok\n")
    assert _namelist(zip_workspace(tree)) == ["main.py"]

    # The next write reclaims it (both the tree's and the sidecar's siblings).
    _write(tmp_path, {"other.py": "x\n"})
    assert not stale.exists()
    assert not meta_stale.exists()


# --- C8/C9: fsync batching and one hashing pass ------------------------------


def test_batch_fsyncs_each_dir_once_not_twice_per_file(tmp_path, monkeypatch):
    """~1000 serialized fsyncs inside the lock become ~writes + touched dirs."""
    _write(tmp_path, {"seed.py": "s\n"})
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])

    files = {f"pkg/f{i}.py": f"# {i}\n" for i in range(10)}
    files["top.py"] = "t\n"
    _write(tmp_path, files)

    # 11 file fsyncs + 1 meta.json + at most one per touched dir (pkg/, tree/)
    # and meta's own parent — never 2x per written file.
    assert len(calls) <= len(files) + 1 + 3
    assert len(calls) < 2 * (len(files) + 1)


def test_write_hashes_the_tree_exactly_once(tmp_path, monkeypatch):
    """The validation pass used to read+hash the whole tree, then the summary
    did it again: two full passes per PUT."""
    _write(tmp_path, {"a.py": "a\n", "b.py": "b\n"})
    seen: list[int] = []
    real = hashlib.sha256
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **kw: (seen.append(1), real(*a, **kw))[1])

    summary = _write(tmp_path, {"c.py": "c\n"})

    assert summary["file_count"] == 3
    assert len(seen) == 3, "expected exactly one hash per file in the post-state"


def test_summary_and_listing_shape_unchanged(tmp_path):
    """Pin: the C8/C9 rework is invisible to callers."""
    summary = _write(tmp_path, {"b.py": "bb\n", "a.py": "a\n"})
    assert summary["written"] == 2
    assert summary["deleted"] == 0
    assert summary["file_count"] == 2
    assert summary["total_bytes"] == 5
    assert [f["path"] for f in summary["files"]] == ["a.py", "b.py"]
    assert summary["files"][0]["sha256"] == hashlib.sha256(b"a\n").hexdigest()
    assert set(summary["files"][0]) == {"path", "size", "sha256"}
    listing = list_files(tmp_path, NAME)
    assert set(listing) == {"files", "file_count", "total_bytes"}
    assert set(listing["files"][0]) == {"path", "size", "sha256", "mtime"}


# --- C10: one exclusion predicate, two callers -------------------------------


@pytest.mark.parametrize(
    ("path", "excluded"),
    [
        (".git/config", True),
        ("__pycache__/module.cpython-311.pyc", True),
        ("model.pyc", True),
        ("data/dataset.csv", True),
        ("node_modules/x.js", True),
        (".venv/bin/python", True),
        ("train.py", False),
        ("utils/helpers.py", False),
        # Exact-case: the CLI ingress must NOT start excluding these.
        ("Data/set.csv", False),
        ("NODE_MODULES/x.js", False),
        ("a/X.PYC", False),
    ],
)
def test_cli_should_exclude_is_byte_identical(path, excluded):
    from nerdit.cli.upload import should_exclude

    assert should_exclude(path) is excluded


@pytest.mark.parametrize(
    ("path", "excluded"),
    [
        (".git/config", True),
        ("Data/set.csv", True),
        ("NODE_MODULES/x.js", True),
        ("a/X.PYC", True),
        ("train.py", False),
        ("database/schema.sql", False),
    ],
)
def test_core_is_excluded_casefolds(path, excluded):
    assert workspaces.is_excluded(path) is excluded


# --- path-length grammar (review round-2, Codex 3803596891) -----------------


@pytest.mark.parametrize(
    "path",
    [
        "a" * WORKSPACE_MAX_SEGMENT_BYTES,
        "src/" + "b" * WORKSPACE_MAX_SEGMENT_BYTES,
        "é" * (WORKSPACE_MAX_SEGMENT_BYTES // 2),  # 200 bytes exactly
        "/".join(["seg"] * 200),  # 799 bytes, under the whole-path bound
    ],
)
def test_path_length_accepts_the_boundary(path):
    validate_file_path(path)  # must not raise


@pytest.mark.parametrize(
    "path",
    [
        "a" * (WORKSPACE_MAX_SEGMENT_BYTES + 1),
        "src/" + "b" * (WORKSPACE_MAX_SEGMENT_BYTES + 1),
        "é" * (WORKSPACE_MAX_SEGMENT_BYTES // 2 + 1),  # 202 bytes: multibyte counts as bytes
    ],
)
def test_path_length_rejects_oversized_segments(path):
    """A 250-byte basename is legal for the filesystem but not once the daemon
    appends its ``.tmp-<pid>`` staging suffix — and the raw ``ENAMETOOLONG``
    landed in the APPLY pass, after earlier files of the batch had already
    landed and before the sidecar was ever stamped."""
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(path)
    assert exc.value.code == "workspace.invalid_path"
    assert exc.value.status_code == 422
    assert str(WORKSPACE_MAX_SEGMENT_BYTES) in exc.value.message
    assert "255" in (exc.value.hint or "")


def test_whole_path_boundary():
    """Deep paths hit the same ENAMETOOLONG class at the ancestor ``mkdir``."""
    at_limit = "/".join(["ab"] * 300)  # 899 bytes
    assert len(at_limit.encode()) == WORKSPACE_MAX_PATH_BYTES - 1
    validate_file_path(at_limit)
    validate_file_path(at_limit + "c")  # 900 bytes exactly
    with pytest.raises(WorkspaceError) as exc:
        validate_file_path(at_limit + "cd")  # 901
    assert exc.value.code == "workspace.invalid_path"
    assert str(WORKSPACE_MAX_PATH_BYTES) in exc.value.message


def test_oversized_segment_keeps_the_batch_all_or_nothing(tmp_path):
    """The reason this is a VALIDATION-pass rule: rejecting here writes nothing.

    Unfixed, ``ok.py`` landed and then ``os.open`` raised a bare ``OSError`` —
    a half-applied tree with no sidecar, invisible to every read (an unstamped
    workspace reads as absent) and an unstructured 500 to the caller.
    """
    with pytest.raises(WorkspaceError) as exc:
        _write(tmp_path, {"ok.py": "print(1)\n", "a" * 250: "boom\n"})
    assert exc.value.code == "workspace.invalid_path"
    assert not (tmp_path / "workspaces" / NAME / "tree").exists()
    assert not (tmp_path / "workspaces" / NAME / "meta.json").exists()
