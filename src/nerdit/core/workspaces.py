"""Manage agent-owned deployable workspace trees.

Files live under `<data_dir>/workspaces/<name>/tree/`; ownership metadata lives
in sibling `meta.json`, outside anything listed, read, zipped, or deployed.
A workspace exists exactly when its metadata exists.

Failures raise `WorkspaceError` for route-level translation. Blocking file
operations run off-thread. While holding `workspace_lock`, use
`settled_to_thread` so cancellation cannot release the lock before its worker
finishes writing.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import anyio

from nerdit.config.defaults import (
    WORKSPACE_MAX_FILE_BYTES,
    WORKSPACE_MAX_FILES,
    WORKSPACE_MAX_TOTAL_BYTES,
    ZIP_EXCLUDE_PATTERNS,
    matches_exclude_pattern,
)
from nerdit.utils.fs import atomic_write, fsync_dir
from nerdit.utils.names import DNS_LABEL_RE

_T = TypeVar("_T")

#: A Windows drive prefix (`C:\x` / `C:/x`) is absolute on its own platform
#: and would slip past `PurePosixPath.is_absolute()` — refuse it explicitly.
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")

#: The basename shape `atomic_write` stages (`<name>.tmp-<pid>`).
#: Reserved: refused at write time, and skipped by every read of the tree.
_TMP_GLOB = "*.tmp-*"

#: Path-length grammar. These are BOUNDS, not product caps: the caps trio
#: (files/bytes) lives in
#: `config/defaults.py`, but these two exist purely because the filesystem and
#: this module's own staging suffix say so, so they stay beside the suffix that
#: forces them.
#:
#: `.tmp-` (5 bytes) + a PID of at most 7 digits = 12 bytes of staging suffix,
#: so 200 + 12 = 212 stays comfortably under the 255-byte component limit APFS
#: and ext4 both enforce. Without this bound a legal 250-byte basename would
#: pass validation and then raise a raw `OSError(ENAMETOOLONG)` from `os.open`
#: in the APPLY pass — after earlier files in the same batch had already
#: landed, i.e. an unstructured 500 over a half-applied tree whose sidecar
#: (written last) never got stamped.
WORKSPACE_MAX_SEGMENT_BYTES = 200

#: The whole relative path, UTF-8 bytes. Headroom under the tightest `PATH_MAX`
#: we target (macOS, 1024) once the `<data_dir>/workspaces/<name>/tree/` prefix
#: is prepended — deep paths hit the same `ENAMETOOLONG` class at the ancestor
#: `mkdir` rather than at the staging `os.open`.
WORKSPACE_MAX_PATH_BYTES = 900

#: Hint reused by every secret-basename refusal (name only, never a content
#: heuristic: false positives break agent loops).
_SECRET_HINT = (
    "Secrets never belong in app files — use the secrets surface "
    "(`set_secret` / `nerdit secrets set`). Values written here transit the "
    "conversation in plaintext."
)

#: Per-workspace zip/write serialization. Daemon-lifetime, keyed by
#: an already-validated name; `setdefault` is safe because there is no await
#: between the lookup and the insert.
_WORKSPACE_LOCKS: dict[str, asyncio.Lock] = {}


class WorkspaceError(Exception):
    """A workspace failure mapped 1:1 to a structured error envelope by the route.

    Mirrors `nerdit.core.gitsource.GitSourceError`, plus `**extra`
    keywords (`detail`, `limit`) that ride into the envelope verbatim — the
    caps carry their number both in `detail` (REST callers) and as a
    top-level `limit` (which survives the MCP `_call` merge).
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        hint: str | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.hint = hint
        self.extra = extra


# --- naming and containment ------------------------------------------------


def validate_workspace_name(name: str) -> None:
    """Enforce the DNS-label grammar on a workspace name.

    Workspace names share the service-name grammar (a workspace deploys under
    its own name). A bad name is a bad *path* — the name is one path component
    of the workspace surface — so it reuses `workspace.invalid_path`.
    """
    if not DNS_LABEL_RE.fullmatch(name):
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"Invalid workspace name '{name}'.",
            hint="A workspace name is a DNS label: lowercase a-z0-9 and '-', 1-63 chars.",
        )


def workspace_root(data_dir: Path, name: str) -> Path:
    """Return `<data_dir>/workspaces/<name>`, re-validating name + containment.

    Twin of `nerdit.core.volumes.service_data_root`: the daemon-owned
    levels (`workspaces` and `workspaces/<name>`) must be real directories —
    `.resolve()` would otherwise adopt a symlink's target as the trusted base
    and silently relocate every subsequent write.
    """
    validate_workspace_name(name)
    unresolved_base = data_dir / "workspaces"
    if unresolved_base.is_symlink():
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            "The workspaces directory is a symlink; refusing to resolve it.",
        )
    if (unresolved_base / name).is_symlink():
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"Workspace '{name}' is a symlink; refusing to resolve it.",
        )
    base = unresolved_base.resolve()
    root = (base / name).resolve()
    if not root.is_relative_to(base):
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"Workspace '{name}' escapes the workspaces root.",
        )
    return root


def tree_root(data_dir: Path, name: str) -> Path:
    """Return `<data_dir>/workspaces/<name>/tree`.

    `tree/` is a parent component of *every* workspace file, so it gets the
    same symlink refusal the two levels above it get in `workspace_root` —
    otherwise the per-target ancestor walk would start from an already-relocated
    base.
    """
    tree = workspace_root(data_dir, name) / "tree"
    if tree.is_symlink():
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"The tree of workspace '{name}' is a symlink; refusing to resolve it.",
        )
    return tree


def _meta_path(data_dir: Path, name: str) -> Path:
    return workspace_root(data_dir, name) / "meta.json"


def ensure_workspace_dirs(data_dir: Path, name: str) -> Path:
    """Create `workspaces/<name>/tree` lazily and return the tree path.

    Perms are *enforced*, not assumed, on every call (the posture of
    `ServiceController._ensure_volume_dirs`): all three levels are held
    `0o700`. No `0o1777` leaf here — a workspace is never container-mounted,
    the docker build context is read by the daemon process itself.
    """
    tree = tree_root(data_dir, name)
    for level in (tree.parent.parent, tree.parent, tree):
        level.mkdir(parents=True, exist_ok=True)
        os.chmod(level, 0o700)
    return tree


# --- path grammar ----------------------------------------------------------


def is_excluded(rel_path: str) -> bool:
    """True when any path component matches `ZIP_EXCLUDE_PATTERNS`.

    The shared `nerdit.config.defaults.matches_exclude_pattern` predicate
    (the ZIP-ingress twin `nerdit.cli.upload.should_exclude` calls the same
    body; both import from `config`, so `core` still never imports `cli`)
    — folded to **case-insensitive** here, so `NODE_MODULES/x` and `a/X.PYC`
    are refused too. Enforcing the exclude set at *write* time is what keeps
    `list_app_files` equal to what deploys.
    """
    return matches_exclude_pattern(PurePosixPath(rel_path).parts, casefold=True)


def is_secret_basename(rel_path: str) -> bool:
    """True for `.env` / `.env.*` basenames (by name, never content).

    Case-insensitive on every platform: `.ENV` and `.Env.local` are the same
    file as `.env` on macOS/Windows, and an exact-case guard would have let
    them through on Linux too.
    """
    base = PurePosixPath(rel_path).name.casefold()
    return base == ".env" or base.startswith(".env.")


def is_tmp_basename(rel_path: str) -> bool:
    """True for the `<name>.tmp-<pid>` residue `atomic_write` stages.

    A crash between the tmp create and the `os.replace` leaves one behind. It
    is daemon residue, never a user file (the write path refuses the name), so
    it is excluded from listing, caps and the deploy zip in one place.
    """
    return fnmatch.fnmatchcase(PurePosixPath(rel_path).name.casefold(), _TMP_GLOB)


def _check_path_lengths(path: str) -> None:
    """Bound each segment and the whole path in UTF-8 BYTES.

    Bytes, not characters: the filesystem counts bytes, so a 101-character
    multibyte name can be a 202-byte component. Called from
    `validate_file_path` after the grammar checks, so `path` is already
    known relative, segment-clean and NUL-free.
    """
    for segment in path.split("/"):
        if len(segment.encode("utf-8")) > WORKSPACE_MAX_SEGMENT_BYTES:
            raise WorkspaceError(
                422,
                "workspace.invalid_path",
                f"Invalid file path '{path}': the path component '{segment}' exceeds "
                f"{WORKSPACE_MAX_SEGMENT_BYTES} bytes.",
                hint=(
                    "Filesystems cap one path component at 255 bytes and the daemon "
                    "appends a '.tmp-<pid>' staging suffix while writing, so each "
                    f"component is bounded at {WORKSPACE_MAX_SEGMENT_BYTES} UTF-8 bytes "
                    "(multibyte characters count for more than one)."
                ),
            )
    if len(path.encode("utf-8")) > WORKSPACE_MAX_PATH_BYTES:
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"Invalid file path '{path}': it exceeds {WORKSPACE_MAX_PATH_BYTES} bytes.",
            hint=(
                "The whole relative path is bounded so it still fits under the host's "
                f"PATH_MAX once the workspace prefix is prepended: "
                f"{WORKSPACE_MAX_PATH_BYTES} UTF-8 bytes."
            ),
        )


def validate_file_path(path: str) -> None:
    """Enforce the path guard family on one workspace-relative path.

    Grammar first (`workspace.invalid_path`) — including the byte-length
    bounds, which are grammar because the filesystem says so —
    then the exclude set (`workspace.excluded_path`), then the secret-basename
    refusal (`workspace.secret_file`). The raw string is inspected *before*
    `PurePosixPath` collapses it — `a//b` and `a/` are rejections, not
    normalizations.
    """
    if not path:
        raise WorkspaceError(422, "workspace.invalid_path", "A file path must not be empty.")
    bad: str | None = None
    if "\x00" in path:
        bad = "must not contain a NUL byte"
    elif "\\" in path:
        bad = "must not contain a backslash"
    elif _DRIVE_PREFIX_RE.match(path) or PurePosixPath(path).is_absolute():
        bad = "must be relative"
    else:
        segments = path.split("/")
        if any(seg == "" for seg in segments):
            bad = "must not contain empty segments (no leading, trailing or doubled '/')"
        elif any(seg in ("..", ".") for seg in segments):
            bad = "must not contain '.' or '..' components"
    if bad is not None:
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"Invalid file path '{path}': it {bad}.",
            hint="Paths are relative to the workspace root, e.g. 'src/app/main.py'.",
        )
    # Length is grammar too: a path the filesystem cannot hold
    # must be refused HERE, in the validation pass, so the all-or-nothing batch
    # contract holds by construction — the apply pass would otherwise raise a raw
    # ENAMETOOLONG with earlier files of the same batch already on disk.
    _check_path_lengths(path)
    if is_tmp_basename(path):
        # Refused BEFORE the exclusion below can hide it: a reserved basename
        # must never silently vanish from a listing the caller believes in.
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"File path '{path}' uses the reserved temporary-file name pattern '{_TMP_GLOB}'.",
            hint="That shape is the daemon's own atomic-write residue; pick another name.",
        )
    if is_excluded(path):
        raise WorkspaceError(
            422,
            "workspace.excluded_path",
            f"File path '{path}' matches an excluded pattern.",
            hint=(
                "These components never reach a build context (ZIP_EXCLUDE_PATTERNS): "
                + ", ".join(sorted(ZIP_EXCLUDE_PATTERNS))
                + "."
            ),
        )
    if is_secret_basename(path):
        raise WorkspaceError(
            422,
            "workspace.secret_file",
            f"File path '{path}' is a secret file (.env / .env.*).",
            hint=_SECRET_HINT,
        )


# --- meta.json -------------------------------------------------------------


def _meta_corrupt() -> WorkspaceError:
    """The one error for a sidecar that exists but cannot be trusted."""
    return WorkspaceError(
        500,
        "workspace.meta_corrupt",
        "The workspace's ownership sidecar (meta.json) is unreadable or malformed.",
        hint=(
            "An operator must repair or remove the workspace's meta.json on the daemon "
            "host (or delete the service with ?purge=workspace); the retention sweep "
            "will eventually reclaim an orphaned workspace."
        ),
    )


def read_meta(data_dir: Path, name: str) -> dict[str, Any] | None:
    """Read ownership metadata, returning None only when absent.

    Only FileNotFoundError/NotADirectoryError mean absence. Unreadable, malformed,
    or non-object metadata must fail closed: treating it as a first write would
    allow ownership takeover. Recovery is manual; the retention sweep owns its
    explicit fallback policy.

    Raises:
        WorkspaceError: Existing metadata cannot be trusted (`workspace.meta_corrupt`).
    """
    path = _meta_path(data_dir, name)
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, ValueError) as exc:
        raise _meta_corrupt() from exc
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _meta_corrupt() from exc
    if not isinstance(meta, dict):
        raise _meta_corrupt()
    return meta


def write_meta(data_dir: Path, name: str, meta: dict[str, Any]) -> None:
    """Write `meta.json` atomically at `0o600`."""
    ensure_workspace_dirs(data_dir, name)
    atomic_write(
        _meta_path(data_dir, name),
        json.dumps(meta, sort_keys=True).encode("utf-8"),
    )


# --- atomic write ----------------------------------------------------------


def _sweep_stale_tmp(tree: Path) -> None:
    """Unlink `*.tmp-*` residue left by a hard crash mid-atomic-write.

    Cleanup is otherwise exception-only, and the pre-write unlink matches just
    the *current* pid's tmp name — so a killed daemon leaves residue in `tree/`
    forever. Reclaimed here on the next write (cheap: ≤ 500-file trees), plus the
    workspace root's own `meta.json.tmp-*` siblings, which live one level up.
    Best-effort throughout: a write must never fail because a leftover would not
    unlink.
    """
    roots = [tree, tree.parent]
    for root in roots:
        if not root.is_dir():
            continue
        candidates = root.rglob(_TMP_GLOB) if root is tree else root.glob(_TMP_GLOB)
        for stale in candidates:
            with contextlib.suppress(OSError):
                if stale.is_symlink() or not stale.is_file():
                    continue
                stale.unlink(missing_ok=True)


# --- listing / reading -----------------------------------------------------


def _iter_tree_files(tree: Path) -> Iterator[Path]:
    """Yield the tree's regular files in POSIX-relative-path order, skipping symlinks.

    Sorted on the *arcname string*, not on `Path` — pathlib orders by parts
    tuple, which groups `a/b.py` before `a.py`; the listing and the zip must
    both be byte-stable across pathlib versions and platforms.

    `Path.rglob` does not descend into symlinked directories, so a symlinked
    component can neither be walked into nor (via the `is_symlink` skip)
    reported as a file.

    `*.tmp-*` residue from a crashed atomic write is skipped here, so listing,
    the post-state caps and the deploy zip all ignore it from one place — the
    write path refuses that basename, so nothing user-authored is ever hidden.
    """
    if not tree.is_dir():
        return
    found = [
        p
        for p in tree.rglob("*")
        if not p.is_symlink() and p.is_file() and not is_tmp_basename(p.name)
    ]
    yield from sorted(found, key=lambda p: p.relative_to(tree).as_posix())


def list_files(data_dir: Path, name: str) -> dict[str, Any]:
    """Return the workspace listing: per-file path/size/sha256/mtime + totals.

    Bounded by construction (`WORKSPACE_MAX_FILES`), so there is no
    pagination. An absent tree lists empty rather than raising.
    """
    tree = tree_root(data_dir, name)
    entries: list[dict[str, Any]] = []
    total = 0
    for path in _iter_tree_files(tree):
        data = path.read_bytes()
        total += len(data)
        entries.append(
            {
                "path": path.relative_to(tree).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            }
        )
    return {"files": entries, "file_count": len(entries), "total_bytes": total}


def _tree_sizes(tree: Path) -> dict[str, int]:
    """Return `{relative path: size}` for the tree — sizes only, no reads.

    The validation pass needs the post-state's *shape*, not its content, so it
    stats instead of calling `list_files` (a full read + sha256 of every
    file). One full hashing pass per request now happens exactly once, at the
    end of the apply pass, where the response summary is built.
    """
    return {p.relative_to(tree).as_posix(): p.stat().st_size for p in _iter_tree_files(tree)}


def read_file(data_dir: Path, name: str, path: str) -> str:
    """Return one file's UTF-8 text (≤ 256 KiB by construction)."""
    validate_file_path(path)
    tree = tree_root(data_dir, name)
    target = _checked_target(tree, path)
    if target.is_symlink() or not target.is_file():
        raise WorkspaceError(
            404,
            "workspace.not_found",
            f"File '{path}' does not exist in workspace '{name}'.",
            hint="List the workspace to see the files it holds.",
        )
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # Unreachable through the writer (it refuses non-text at write
        # time); reachable when a file is dropped onto disk out of band — the
        # same tamper copy_workspace's belt-and-braces re-filter defends against.
        raise WorkspaceError(
            422,
            "workspace.binary_content",
            f"File '{path}' in workspace '{name}' is not UTF-8 text.",
            hint="Workspace files are text only (D-P29-2); this one was written out of band.",
        ) from exc


def _checked_target(tree: Path, rel: str) -> Path:
    """Resolve `tree/rel` after walking every existing ancestor for symlinks.

    The multi-segment walk is what neither `gitsource`
    (lexical only) nor `volumes` (single level) provides: a symlink planted at
    *any* already-existing prefix directory relocates every write below it. The
    trailing `resolve()` containment check is defense-in-depth on top.
    """
    parts = PurePosixPath(rel).parts
    cursor = tree
    for segment in parts[:-1]:
        cursor = cursor / segment
        if cursor.is_symlink():
            raise WorkspaceError(
                422,
                "workspace.invalid_path",
                f"Path '{rel}' traverses a symlinked directory.",
                hint="Workspace files must live under the workspace tree; symlinks are refused.",
            )
    target = tree / PurePosixPath(rel)
    if not target.resolve().is_relative_to(tree.resolve()):
        raise WorkspaceError(
            422,
            "workspace.invalid_path",
            f"Path '{rel}' escapes the workspace tree.",
        )
    return target


# --- the batch writer ------------------------------------------------------


def _encode_batch(files: dict[str, str], delete_set: set[str]) -> dict[str, bytes]:
    """Validate every entry's path/type/content and return the encoded payloads."""
    encoded: dict[str, bytes] = {}
    for rel, content in files.items():
        validate_file_path(rel)
        if rel in delete_set:
            raise WorkspaceError(
                422,
                "workspace.invalid_path",
                f"Path '{rel}' appears in both 'files' and 'delete'.",
                hint="A single batch either writes a path or deletes it, never both.",
            )
        if not isinstance(content, str) or "\x00" in content:
            raise WorkspaceError(
                422,
                "workspace.binary_content",
                f"File '{rel}' is not NUL-free UTF-8 text.",
                hint="Workspace files are text only (D-P29-2); no base64 escape hatch in v1.",
            )
        try:
            blob = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise WorkspaceError(
                422,
                "workspace.binary_content",
                f"File '{rel}' is not encodable as UTF-8.",
                hint="Workspace files are text only (D-P29-2); no base64 escape hatch in v1.",
            ) from exc
        if len(blob) > WORKSPACE_MAX_FILE_BYTES:
            raise WorkspaceError(
                422,
                "workspace.file_too_large",
                f"File '{rel}' is {len(blob)} bytes; the per-file limit is "
                f"{WORKSPACE_MAX_FILE_BYTES} bytes.",
                hint="Split the file, or deploy from a repository with deploy_git.",
                detail={"path": rel, "size": len(blob), "limit": WORKSPACE_MAX_FILE_BYTES},
                limit=WORKSPACE_MAX_FILE_BYTES,
            )
        encoded[rel] = blob
    return encoded


def _check_post_state(
    existing: dict[str, int], encoded: dict[str, bytes], deletes: list[str]
) -> None:
    """Simulate the resulting workspace and enforce the count/total caps.

    Counted on the *post-state*, never on the sum of the request: rewriting a
    big file smaller and adding another can legitimately land under the cap.
    """
    sizes = dict(existing)
    for rel in deletes:
        sizes.pop(rel, None)
    for rel, blob in encoded.items():
        sizes[rel] = len(blob)
    if len(sizes) > WORKSPACE_MAX_FILES:
        raise WorkspaceError(
            422,
            "workspace.too_many_files",
            f"The workspace would hold {len(sizes)} files; the limit is {WORKSPACE_MAX_FILES}.",
            hint="Delete files in the same batch, or deploy from a repository with deploy_git.",
            detail={"file_count": len(sizes), "limit": WORKSPACE_MAX_FILES},
            limit=WORKSPACE_MAX_FILES,
        )
    total = sum(sizes.values())
    if total > WORKSPACE_MAX_TOTAL_BYTES:
        raise WorkspaceError(
            422,
            "workspace.too_large",
            f"The workspace would hold {total} bytes; the limit is "
            f"{WORKSPACE_MAX_TOTAL_BYTES} bytes.",
            hint="Delete files in the same batch, or deploy from a repository with deploy_git.",
            detail={"total_bytes": total, "limit": WORKSPACE_MAX_TOTAL_BYTES},
            limit=WORKSPACE_MAX_TOTAL_BYTES,
        )


def _path_and_ancestors(rel: str) -> Iterator[str]:
    """Yield `rel` and every ancestor-directory prefix of it, outermost first."""
    segments = rel.split("/")
    for i in range(1, len(segments) + 1):
        yield "/".join(segments[:i])


def _casefold_collision(first: str, second: str) -> WorkspaceError:
    """The one 422 for a pair of paths that differ only by case."""
    a, b = sorted((first, second))
    return WorkspaceError(
        422,
        "workspace.invalid_path",
        f"Paths '{a}' and '{b}' differ only by case.",
        hint=(
            "A workspace must apply identically everywhere: on a case-insensitive "
            "filesystem (macOS APFS, Windows) these name the same file, so the "
            "batch would not do what it reads like. Use one spelling per path — "
            "changing a path's case is two batches, delete in one and write the "
            "new spelling in the next."
        ),
    )


def _check_casefold_collisions(
    existing: dict[str, int], writes: dict[str, bytes], deletes: set[str]
) -> None:
    """Reject case-folded path collisions consistently on every filesystem.

    Include ancestor prefixes to catch file/directory conflicts before writes.
    Exact path matches are allowed. Case-only delete/write renames require two
    batches: writes precede deletes, so an insensitive filesystem could otherwise
    unlink the newly written file.
    """
    write_keys: dict[str, str] = {}
    for rel in writes:
        for key in _path_and_ancestors(rel):
            clash = write_keys.setdefault(key.casefold(), key)
            if clash != key:
                raise _casefold_collision(clash, key)

    existing_exact: dict[str, set[str]] = {}
    existing_keys: dict[str, set[str]] = {}
    for rel in existing:
        existing_exact.setdefault(rel.casefold(), set()).add(rel)
        for key in _path_and_ancestors(rel):
            existing_keys.setdefault(key.casefold(), set()).add(key)

    for folded, key in write_keys.items():
        for other in existing_keys.get(folded, ()):
            # A same-batch delete of the exact colliding path resolves it: the
            # tree the write lands in no longer holds that spelling.
            if other != key and other not in deletes:
                raise _casefold_collision(other, key)

    for rel in deletes:
        folded = rel.casefold()
        written = write_keys.get(folded)
        if written is not None and written != rel:
            raise _casefold_collision(written, rel)
        for other in existing_exact.get(folded, ()):
            if other != rel:
                raise _casefold_collision(other, rel)


def _check_write_structure(tree: Path, writes: list[str], delete_set: set[str]) -> None:
    """Refuse batches whose post-state is not a possible tree.

    The per-path grammar cannot see these four conflicts because they are
    *relations* — between a write and the existing tree, or between two writes of
    the same batch. Left unchecked they surface in the apply pass as a raw
    `OSError` (`IsADirectoryError` from `os.replace`, `FileExistsError`
    from the intermediate `mkdir`) *after* earlier files have already landed:
    a half-applied batch and an unstructured 500, both forbidden here.

    Write-under-a-path-this-batch-deletes is refused unconditionally rather than
    reordered: deletes run after writes, and the R15 both-lists precedent already
    says a batch never writes and removes the same path. Rename-a-file-to-a-
    package is therefore two batches, and the hint says so.
    """
    write_set = set(writes)
    for rel in writes:
        segments = rel.split("/")
        for i in range(1, len(segments)):
            ancestor = "/".join(segments[:i])
            if ancestor in delete_set:
                raise WorkspaceError(
                    422,
                    "workspace.invalid_path",
                    f"Path '{rel}' is written under '{ancestor}', which the same batch deletes.",
                    hint="Delete the file in one batch, then write under that name in the next.",
                )
            if ancestor in write_set:
                raise WorkspaceError(
                    422,
                    "workspace.invalid_path",
                    f"Path '{rel}' is written under '{ancestor}', "
                    "which the same batch writes as a file.",
                    hint="A path is either a file or a directory, never both.",
                )
            existing = tree / ancestor
            if existing.exists() and not existing.is_dir():
                raise WorkspaceError(
                    422,
                    "workspace.invalid_path",
                    f"Path '{rel}' is written under '{ancestor}', which is an existing file.",
                    hint="Delete the file in one batch, then write under that name in the next.",
                )
        if (tree / PurePosixPath(rel)).is_dir():
            raise WorkspaceError(
                422,
                "workspace.invalid_path",
                f"Path '{rel}' is an existing directory in the workspace.",
                hint="Delete the files it holds first, or write to a different path.",
            )


def _prune_empty_dirs(tree: Path, target: Path) -> None:
    """Best-effort removal of directories left empty by a delete, never `tree`."""
    cursor = target.parent
    while cursor != tree and cursor.is_relative_to(tree):
        try:
            cursor.rmdir()
        except OSError:
            return
        cursor = cursor.parent


def write_files(
    data_dir: Path,
    name: str,
    files: dict[str, str],
    delete: list[str],
    *,
    owner_token_id: str | None,
    existing_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate an entire write/delete batch before applying it.

    Check path grammar, exclusions, secret basenames, text encoding, size limits,
    symlinks, and file/directory conflicts against the tree and batch before any
    write. Caller-provided metadata must come from the same locked ownership check.

    Args:
        existing_meta: Metadata already authorized by the caller, or None for a
            workspace whose first write stamps ownership.

    Returns:
        Written/deleted paths, file count, total bytes, and per-file size/hash data.
    """
    # --- validation pass (zero filesystem writes) ---------------------------
    tree = tree_root(data_dir, name)
    deletes: list[str] = list(dict.fromkeys(delete))  # dedupe, order preserved
    for rel in deletes:
        validate_file_path(rel)
    encoded = _encode_batch(files, set(deletes))
    # Sizes only — the one full read+hash of the tree happens at the end.
    existing = _tree_sizes(tree)
    _check_post_state(existing, encoded, deletes)
    _check_casefold_collisions(existing, encoded, set(deletes))
    targets = {rel: _checked_target(tree, rel) for rel in (*encoded, *deletes)}
    _check_write_structure(tree, list(encoded), set(deletes))

    # --- apply pass ---------------------------------------------------------
    ensure_workspace_dirs(data_dir, name)
    _sweep_stale_tmp(tree)
    # Each write's parent, each delete's parent and every created ancestor are
    # fsynced ONCE after the loop rather than per file inside it.
    touched: set[Path] = set()
    written = 0
    for rel, blob in encoded.items():
        target = targets[rel]
        for parent in _ancestors(tree, target):
            parent.mkdir(exist_ok=True)
            os.chmod(parent, 0o700)
            touched.add(parent)
        # One fsync per touched directory at the end, not one per file: a
        # 500-file batch would otherwise hold the workspace lock for seconds.
        atomic_write(target, blob, sync_dir=False)
        touched.add(target.parent)
        written += 1
    removed = 0
    for rel in deletes:
        target = targets[rel]
        if target.is_file() and not target.is_symlink():
            target.unlink()
            removed += 1
            touched.add(target.parent)
            _prune_empty_dirs(tree, target)
    for directory in sorted(touched):
        # A directory pruned by a delete above is simply gone.
        with contextlib.suppress(FileNotFoundError):
            fsync_dir(directory)

    # --- meta (first write stamps the owner) --------------------------------
    now = datetime.now(UTC).isoformat()
    meta = (
        dict(existing_meta)
        if existing_meta is not None
        else {"owner_token_id": owner_token_id, "created_at": now}
    )
    meta["last_written_at"] = now
    write_meta(data_dir, name, meta)

    listing = list_files(data_dir, name)
    return {
        "written": written,
        "deleted": removed,
        "file_count": listing["file_count"],
        "total_bytes": listing["total_bytes"],
        "files": [
            {"path": e["path"], "size": e["size"], "sha256": e["sha256"]} for e in listing["files"]
        ],
    }


def _ancestors(tree: Path, target: Path) -> list[Path]:
    """Return `target`'s intermediate directories under `tree`, outermost first."""
    parts = target.parent.relative_to(tree).parts
    return [tree.joinpath(*parts[: i + 1]) for i in range(len(parts))]


# --- deploy snapshot -------------------------------------------------------


def _deployable_files(tree: Path) -> list[tuple[Path, str]]:
    """Return `(path, posix_rel)` for every file a deploy snapshot carries.

    The exclude set and the secret-basename refusal are re-applied here as
    belt-and-braces: write time is the enforcement point, this guards against
    out-of-band disk tamper. Symlinks are skipped by `_iter_tree_files`, and
    `meta.json` lives outside `tree/` so it structurally never appears.

    Raises:
        WorkspaceError: `workspace.empty` when nothing deployable remains.
    """
    files = [
        (path, rel)
        for path in _iter_tree_files(tree)
        if not (is_excluded(rel := path.relative_to(tree).as_posix()) or is_secret_basename(rel))
    ]
    if not files:
        raise WorkspaceError(
            422,
            "workspace.empty",
            "The workspace holds no deployable files.",
            hint="Write at least one file with write_app_files before deploying.",
        )
    return files


def copy_workspace(tree: Path, dest: Path) -> None:
    """Copy the deployable files into *dest*, the disposable deploy context.

    Destinations are `dest / relative_to(tree)` of regular non-symlink files,
    so containment holds by construction. The caller owns removing *dest* on
    failure.
    """
    for path, rel in _deployable_files(tree):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


# --- concurrency -----------------------------------------------------------


def workspace_lock(name: str) -> asyncio.Lock:
    """Return the daemon-lifetime lock for *name*.

    Callers must have validated *name* first — this is a bare dict key, not a
    path. `setdefault` is race-free here: there is no await between the lookup
    and the insert, so one asyncio tick can only ever create one lock per name.
    """
    return _WORKSPACE_LOCKS.setdefault(name, asyncio.Lock())


async def settled_to_thread(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Wait for a workspace worker to finish before propagating cancellation.

    Use for every thread operation touching a workspace tree under its lock,
    including reads. Ordinary `to_thread` cancellation releases the lock while
    its worker still runs, allowing retries to race the same staging filenames.

    Asyncio shielding protects the worker from raw task cancellation; AnyIO
    shielding prevents cancelled request scopes from repeatedly cancelling the
    settlement wait. A second raw Task.cancel during that wait remains a residual.
    Unrelated service-tree/image cleanup needs no workspace settlement wrapper.
    """
    task = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            # The worker's own outcome is the caller's no longer — the response
            # can never be sent — but it must SETTLE before the lock releases.
            with contextlib.suppress(BaseException):
                await task
        raise
