"""Tests for backup v3 — per-database logical dump tars (P37 WP3).

Direct-construction only (scratch ``data_dir`` trees, no TestClient/aiosqlite —
the standing sandbox caveat): the packing discipline of ``create_dump_backup``
(tar shape, member order, perms, manifest, sha256), the D-P37-7 refusals of a
container-written output (symlink / empty / unexpected extra entry), the
D-P37-10 extraction battery of ``extract_dump_tar``, the D-P37-2 boot sweep of
leftover staging dirs, the three-way glob disjointness against the v1
control-plane and v2 volume flavours, the phase-3e per-service keep-last, and
the ``/system/disk`` ``dumps`` bucket + ``dump_staging_bytes`` via
``_build_disk_report``.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import stat
import tarfile
from fnmatch import fnmatch
from pathlib import Path

import pytest

from nerdit.core.backup import (
    DUMP_MANIFEST_MAX_BYTES,
    DUMP_MANIFEST_NAME,
    DUMP_TAR_RE,
    BackupError,
    DumpManifest,
    DumpManifestError,
    DumpOutputError,
    create_dump_backup,
    extract_dump_tar,
    sweep_staging_orphans,
)
from nerdit.core.volumes import (
    VolumeSpecError,
    create_dump_staging_dir,
    dump_staging_dir,
    dump_staging_root,
)
from nerdit.daemon.routes.system import (
    _build_disk_report,
    _walk_backups,
    _walk_dumps,
    _walk_volume_backups,
)
from nerdit.daemon.server import _prune_backups, _prune_dump_backups, _prune_volume_backups
from nerdit.daemon.sweeps import _VOLUME_TAR_RE

_RUN_ID = "abcdef012345"
_DUMP_NAME = "dump.pgdump"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def _staging(data_dir: Path, run_id: str = _RUN_ID) -> Path:
    """Create the daemon-computed staging slot the sibling would have mounted."""
    staging = dump_staging_dir(data_dir, run_id)
    staging.mkdir(mode=0o700, parents=True, exist_ok=False)
    return staging


def _seed_output(
    staging: Path, content: bytes = b"PGDMP\x00payload", name: str = _DUMP_NAME
) -> Path:
    out = staging / name
    out.write_bytes(content)
    return out


def _pack(
    data_dir: Path,
    *,
    service: str = "pg",
    engine: str = "postgres",
    content: bytes = b"PGDMP\x00payload",
    run_id: str = _RUN_ID,
):
    """Seed a staging slot with one output file and pack it."""
    staging = _staging(data_dir, run_id)
    _seed_output(staging, content)
    return (
        create_dump_backup(
            data_dir,
            service=service,
            engine=engine,
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        ),
        staging,
    )


def _touch_tar(backups: Path, name: str, *, mtime: float | None = None) -> Path:
    backups.mkdir(parents=True, exist_ok=True)
    p = backups / name
    p.write_bytes(b"x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


# --------------------------------------------------------------------------- #
# staging grammar (core/volumes.py)
# --------------------------------------------------------------------------- #


def test_dump_staging_dir_grammar(tmp_path):
    assert dump_staging_root(tmp_path) == tmp_path / "dump-staging"
    assert dump_staging_dir(tmp_path, _RUN_ID) == tmp_path / "dump-staging" / _RUN_ID
    # Anything that is not a minted 12-char id is refused before it reaches the
    # filesystem — traversal included.
    for bad in ("", "..", "../etc", "short", "ABCDEF012345", "abcdef01234/x"):
        with pytest.raises(VolumeSpecError):
            dump_staging_dir(tmp_path, bad)


def test_dump_staging_dir_refuses_symlinked_root(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "dump-staging").symlink_to(elsewhere)
    with pytest.raises(VolumeSpecError):
        dump_staging_dir(tmp_path, _RUN_ID)


def test_create_dump_staging_dir_translates_filesystem_failures(tmp_path):
    # A regular file where the staging root belongs, and an unwritable data
    # dir: both are OSError at the syscall and must reach the callers as the
    # one class they map (VolumeSpecError), path-free per M3.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "dump-staging").write_bytes(b"x")
    with pytest.raises(VolumeSpecError) as exc:
        create_dump_staging_dir(blocked, _RUN_ID)
    assert "/" not in str(exc.value)

    if os.geteuid() == 0:  # root ignores the mode bits; the file case above stands
        return
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    os.chmod(readonly, 0o500)
    try:
        with pytest.raises(VolumeSpecError) as exc:
            create_dump_staging_dir(readonly, _RUN_ID)
        assert "/" not in str(exc.value)
    finally:
        os.chmod(readonly, 0o700)


# --------------------------------------------------------------------------- #
# tar shape / manifest / perms
# --------------------------------------------------------------------------- #


def test_dump_tar_shape_order_perms_and_manifest(tmp_path):
    content = b"PGDMP\x00" + b"a" * 512
    result, staging = _pack(tmp_path, content=content)

    # Name grammar + the service group the retention phase groups on.
    match = DUMP_TAR_RE.match(result.basename)
    assert match is not None
    assert match.group("service") == "pg"

    tar_path = Path(result.path)
    assert tar_path.parent == tmp_path / "backups"
    assert stat.S_IMODE(tar_path.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "backups").stat().st_mode) == 0o700
    assert result.size_bytes == tar_path.stat().st_size

    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        # Exactly two members, the payload first and the manifest LAST.
        assert [m.name for m in members] == [_DUMP_NAME, DUMP_MANIFEST_NAME]
        assert all(m.isreg() for m in members)
        assert {m.mode for m in members} == {0o600}
        assert members[0].size == len(content)
        assert tar.extractfile(members[0]).read() == content
        manifest = json.loads(tar.extractfile(members[1]).read().decode("utf-8"))

    assert manifest == {
        "version": 1,
        "kind": "dump",
        "service": "pg",
        "engine": "postgres",
        "format": "pg_custom",
        "image": "postgres:16",
        "created_at": manifest["created_at"],
        "file": _DUMP_NAME,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "tool": {"argv0": "pg_dump"},
    }
    assert result.sha256 == manifest["sha256"]
    # Staging is gone on the success path too (D-P37-2).
    assert not staging.exists()
    # …and no .tmp- leftover.
    assert list((tmp_path / "backups").glob(".tmp-*")) == []


def test_post_promotion_failure_removes_the_promoted_tar(tmp_path, monkeypatch):
    """A fault after the rename must leave nothing: the 500 says no outcome was produced."""

    def _boom(_path):
        raise OSError(errno.EIO, "eio")

    monkeypatch.setattr("nerdit.core.backup._fsync_dir", _boom)
    staging = _staging(tmp_path)
    _seed_output(staging)

    with pytest.raises(BackupError):
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )

    backups = tmp_path / "backups"
    assert list(backups.glob("nerdit-dump-*")) == []
    assert list(backups.glob(".tmp-*")) == []
    assert not staging.exists()


# --------------------------------------------------------------------------- #
# glob disjointness (three flavours, three walkers)
# --------------------------------------------------------------------------- #


def test_dump_glob_is_disjoint_from_both_older_flavours(tmp_path):
    result, _ = _pack(tmp_path)
    backups = tmp_path / "backups"
    v1 = "nerdit-backup-20260907T120000Z-abc123.tar.gz"
    v2 = "nerdit-volumes-pg-20260907T120000Z-abc123.tar.gz"

    # The dump name matches neither older glob/regex…
    assert not fnmatch(result.basename, "nerdit-backup-*.tar.gz")
    assert _VOLUME_TAR_RE.match(result.basename) is None
    # …and neither older name matches the dump regex.
    assert DUMP_TAR_RE.match(v1) is None
    assert DUMP_TAR_RE.match(v2) is None

    _touch_tar(backups, v1)
    _touch_tar(backups, v2)
    assert _walk_dumps(backups)["count"] == 1
    assert _walk_backups(backups)["count"] == 1
    assert _walk_volume_backups(backups)["count"] == 1


# --------------------------------------------------------------------------- #
# untrusted output refusals (D-P37-7) — every one leaves staging gone, no tar
# --------------------------------------------------------------------------- #


def _assert_nothing_left(tmp_path: Path, staging: Path) -> None:
    assert not staging.exists()
    backups = tmp_path / "backups"
    assert list(backups.glob("nerdit-dump-*")) == []
    assert list(backups.glob(".tmp-*")) == []


def test_symlinked_output_is_refused_and_never_followed(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"master key material")
    staging = _staging(tmp_path)
    (staging / _DUMP_NAME).symlink_to(secret)

    with pytest.raises(DumpOutputError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert exc.value.reason == "output_not_regular"
    _assert_nothing_left(tmp_path, staging)


def test_missing_output_is_output_not_regular(tmp_path):
    staging = _staging(tmp_path)
    with pytest.raises(DumpOutputError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert exc.value.reason == "output_not_regular"
    _assert_nothing_left(tmp_path, staging)


def test_zero_byte_output_is_empty_output(tmp_path):
    # ``pg_dump`` creates its output file BEFORE it connects (verified live,
    # §0), so a refused connection leaves exactly this.
    staging = _staging(tmp_path)
    _seed_output(staging, b"")
    with pytest.raises(DumpOutputError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert exc.value.reason == "empty_output"
    _assert_nothing_left(tmp_path, staging)


def test_extra_staging_entry_is_unexpected_output(tmp_path):
    staging = _staging(tmp_path)
    _seed_output(staging)
    (staging / "core.1234").write_bytes(b"whatever the image dropped")
    with pytest.raises(DumpOutputError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert exc.value.reason == "unexpected_output"
    _assert_nothing_left(tmp_path, staging)


def test_symlinked_staging_dir_is_refused_and_never_packed(tmp_path):
    """The staging path itself is lstat'd — a symlink is refused, not followed.

    ``dump_staging_dir`` + ``mkdir(exist_ok=False)`` make this unreachable from
    the controller; the pin exists because the failure it prevents is silent:
    packing through the link would tar the TARGET's contents and then, quite
    correctly, refuse to ``rmtree`` a non-directory — leaving the payload on
    disk with only a log line behind it.
    """
    real = tmp_path / "elsewhere"
    real.mkdir()
    _seed_output(real, b"PGDMP\x00not ours")
    staging = dump_staging_dir(tmp_path, _RUN_ID)
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.symlink_to(real, target_is_directory=True)

    with pytest.raises(DumpOutputError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert exc.value.reason == "output_not_regular"
    assert str(tmp_path) not in str(exc.value)
    # No tar, and the link's target is untouched (the ``finally`` refused it).
    assert list((tmp_path / "backups").glob("nerdit-dump-*")) == []
    assert (real / _DUMP_NAME).is_file()
    assert staging.is_symlink()


def test_unusable_backups_dir_is_a_typed_backup_error(tmp_path):
    # ``<data_dir>/backups`` as a regular file: the setup of the backups dir
    # lives INSIDE the typed boundary, so the caller sees BackupError (which
    # _pack_or_fail maps) and never a raw OSError, and staging is still gone.
    (tmp_path / "backups").write_bytes(b"x")
    staging = _staging(tmp_path)
    _seed_output(staging)
    with pytest.raises(BackupError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert str(tmp_path) not in str(exc.value)
    assert not staging.exists()


def test_refusal_messages_are_path_free(tmp_path):
    staging = _staging(tmp_path)
    _seed_output(staging, b"")
    with pytest.raises(DumpOutputError) as exc:
        create_dump_backup(
            tmp_path,
            service="pg",
            engine="postgres",
            image="postgres:16",
            staging=staging,
            backend_dump_filename=_DUMP_NAME,
            backend_dump_format="pg_custom",
            tool_argv0="pg_dump",
        )
    assert str(tmp_path) not in str(exc.value)


# --------------------------------------------------------------------------- #
# extract_dump_tar (D-P37-10)
# --------------------------------------------------------------------------- #


def _hand_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, mode="w:gz") as tar:
        for info, payload in members:
            tar.addfile(info, io.BytesIO(payload) if payload is not None else None)
    return path


def _reg(name: str, payload: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.REGTYPE
    info.mode = 0o600
    info.size = len(payload)
    return info, payload


def _manifest_bytes(content: bytes, **overrides) -> bytes:
    body = {
        "version": 1,
        "kind": "dump",
        "service": "pg",
        "engine": "postgres",
        "format": "pg_custom",
        "image": "postgres:16",
        "created_at": "2026-09-07T12:00:00+00:00",
        "file": _DUMP_NAME,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "tool": {"argv0": "pg_dump"},
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def test_extract_dump_tar_round_trip(tmp_path):
    content = b"PGDMP\x00round-trip"
    result, _ = _pack(tmp_path, content=content)

    target = tmp_path / "restore"
    target.mkdir(mode=0o700)
    manifest = extract_dump_tar(Path(result.path), target)

    assert isinstance(manifest, DumpManifest)
    assert manifest.service == "pg"
    assert manifest.engine == "postgres"
    assert manifest.file == _DUMP_NAME
    extracted = target / manifest.file
    assert extracted.read_bytes() == content
    assert stat.S_IMODE(extracted.stat().st_mode) == 0o600
    assert sorted(p.name for p in target.iterdir()) == [_DUMP_NAME]


def _extract_reason(tar_path: Path, target: Path) -> str:
    with pytest.raises(DumpManifestError) as exc:
        extract_dump_tar(tar_path, target)
    return exc.value.reason


def test_extract_refuses_extra_member(tmp_path):
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "extra.tar.gz",
        [
            _reg(_DUMP_NAME, content),
            _reg("hitchhiker.sh", b"#!/bin/sh\n"),
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content)),
        ],
    )
    target = tmp_path / "t1"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "members"
    assert list(target.iterdir()) == []


def test_extract_refuses_oversized_manifest(tmp_path):
    content = b"payload"
    fat = b"{" + b" " * (DUMP_MANIFEST_MAX_BYTES + 8) + b"}"
    tar_path = _hand_tar(
        tmp_path / "fat.tar.gz",
        [_reg(_DUMP_NAME, content), _reg(DUMP_MANIFEST_NAME, fat)],
    )
    target = tmp_path / "t2"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "size"
    assert list(target.iterdir()) == []


def test_extract_refuses_sha_mismatch_and_leaves_nothing(tmp_path):
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "bad-sha.tar.gz",
        [_reg(_DUMP_NAME, b"tampered"), _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content))],
    )
    target = tmp_path / "t3"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "sha256_mismatch"
    # The partially written payload is removed — a caller can never hand a
    # half-verified file to pg_restore.
    assert list(target.iterdir()) == []


def test_extract_refuses_traversal_member(tmp_path):
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "slip.tar.gz",
        [_reg("../escape.pgdump", content), _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content))],
    )
    target = tmp_path / "t4"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "unsafe_member"
    assert not (tmp_path / "escape.pgdump").exists()


def test_extract_refuses_symlink_member(tmp_path):
    content = b"payload"
    link = tarfile.TarInfo(name=_DUMP_NAME)
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    tar_path = _hand_tar(
        tmp_path / "link.tar.gz",
        [(link, None), _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content))],
    )
    target = tmp_path / "t5"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "unsafe_member"
    assert list(target.iterdir()) == []


def test_extract_refuses_unknown_manifest_key(tmp_path):
    # ``extra="forbid"``: an operator-supplied tar carrying an unknown key is a
    # refusal, never a tolerated future field.
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "schema.tar.gz",
        [
            _reg(_DUMP_NAME, content),
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content, restore_to="/etc")),
        ],
    )
    target = tmp_path / "t6"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "schema"


def test_extract_refuses_manifest_file_naming_a_path(tmp_path):
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "path.tar.gz",
        [
            _reg("../evil", content),
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content, file="../evil")),
        ],
    )
    target = tmp_path / "t7"
    target.mkdir()
    # The member pre-pass rejects it first; the manifest grammar is the second
    # wall (pinned directly below).
    assert _extract_reason(tar_path, target) == "unsafe_member"
    with pytest.raises(ValueError, match="plain dump member name"):
        DumpManifest.model_validate(json.loads(_manifest_bytes(content, file="../evil")))


def test_extract_refuses_two_manifest_members(tmp_path):
    """Two members, both named ``dump-manifest.json``: still a refusal.

    It clears ``len == 2`` and a plain ``in`` membership test, so before the
    ``count() == 1`` check the search for "the other member" found nothing and
    raised a bare ``StopIteration`` — a foreign exception class crossing a
    ``to_thread`` boundary that promises one machine ``reason`` for every
    failure (D-P37-10).
    """
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "twins.tar.gz",
        [
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content)),
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content)),
        ],
    )
    target = tmp_path / "t8"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "members"
    assert list(target.iterdir()) == []


def test_extract_refuses_truncated_archive(tmp_path):
    """A half-copied tar — the everyday damage case — is a typed refusal.

    The gzip layer raises ``EOFError``, which is not a ``tarfile.TarError``:
    before the total boundary it escaped ``extract_dump_tar`` untyped and
    reached the route as an unstructured 500 rather than the contracted 422.
    """
    result, _ = _pack(tmp_path, content=b"PGDMP\x00" + b"z" * 4096)
    raw = Path(result.path).read_bytes()
    truncated = tmp_path / "half.tar.gz"
    truncated.write_bytes(raw[: len(raw) // 2])

    target = tmp_path / "t9"
    target.mkdir()
    assert _extract_reason(truncated, target) == "members"
    assert list(target.iterdir()) == []


def test_extract_refuses_absent_paths_without_leaking_them(tmp_path):
    """Both paths are caller preconditions — and neither escapes as an OSError.

    A missing tar (the route 404s first) or a missing staging dir (the
    controller creates it) would otherwise surface as a ``FileNotFoundError``
    whose message embeds the absolute path, straight into a 500 envelope. M3
    says path-free; D-P37-10 says one machine reason.
    """
    target = tmp_path / "t10"
    target.mkdir()
    with pytest.raises(DumpManifestError) as absent_tar:
        extract_dump_tar(tmp_path / "no-such.tar.gz", target)
    assert absent_tar.value.reason == "members"
    assert str(tmp_path) not in str(absent_tar.value)

    result, _ = _pack(tmp_path)
    with pytest.raises(DumpManifestError) as absent_staging:
        extract_dump_tar(Path(result.path), tmp_path / "no-such-staging")
    assert absent_staging.value.reason == "members"
    assert str(tmp_path) not in str(absent_staging.value)


def test_extract_refuses_header_size_before_writing_anything(tmp_path):
    """A header claiming more bytes than the manifest records writes NOTHING.

    The two sizes must agree for the digest to have any chance of matching, so
    the disagreement is already the refusal — and checking it before ``os.open``
    means a tar whose header claims gigabytes never lands in the staging root
    at all. The restore route has no disk preflight of its own (§1.4), which is
    what makes this a bound rather than an optimisation.
    """
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "fat-member.tar.gz",
        [
            _reg(_DUMP_NAME, b"x" * 5000),
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content)),
        ],
    )
    target = tmp_path / "t11"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "sha256_mismatch"
    assert list(target.iterdir()) == []


def test_extract_refuses_same_size_tampered_payload(tmp_path):
    """The digest check still stands on its own when the sizes DO agree.

    The falsifier for the pre-check above: swapping the payload for one of the
    same length gets past every size comparison and is caught by the sha256
    alone, with the partial write removed.
    """
    content = b"payload"
    tar_path = _hand_tar(
        tmp_path / "swapped.tar.gz",
        [
            _reg(_DUMP_NAME, b"pay10ad"),
            _reg(DUMP_MANIFEST_NAME, _manifest_bytes(content)),
        ],
    )
    target = tmp_path / "t12"
    target.mkdir()
    assert _extract_reason(tar_path, target) == "sha256_mismatch"
    assert list(target.iterdir()) == []


# --------------------------------------------------------------------------- #
# boot sweep (D-P37-2)
# --------------------------------------------------------------------------- #


def test_sweep_staging_orphans_clears_both_roots(tmp_path):
    root = dump_staging_root(tmp_path)
    (root / "abcdef012345" / "nested").mkdir(parents=True)
    (root / "abcdef012345" / "nested" / "dump.pgdump").write_bytes(b"leftover")
    (root / "abcdef012346").mkdir()
    backups = tmp_path / "backups"
    (backups / ".staging-123-abcd").mkdir(parents=True)
    keeper = _touch_tar(backups, "nerdit-backup-20260907T120000Z-abc123.tar.gz")

    assert sweep_staging_orphans(tmp_path) == 3
    assert list(root.iterdir()) == []
    assert not (backups / ".staging-123-abcd").exists()
    # A real tar beside the staging dir is never swept.
    assert keeper.exists()
    # Idempotent: a second boot removes nothing.
    assert sweep_staging_orphans(tmp_path) == 0


def test_sweep_staging_orphans_clears_partial_tars_only_by_full_prefix(tmp_path):
    """A SIGKILL'd packer leaves ``backups/.tmp-nerdit-*``; nothing else sweeps it."""
    backups = tmp_path / "backups"
    partials = [
        _touch_tar(backups, ".tmp-nerdit-dump-db-20260907T120000Z-abc123.tar.gz"),
        _touch_tar(backups, ".tmp-nerdit-backup-20260907T120000Z-abc123.tar.gz"),
        _touch_tar(backups, ".tmp-nerdit-volumes-db-20260907T120000Z-abc123.tar.gz"),
    ]
    keeper = _touch_tar(backups, "nerdit-backup-20260907T120000Z-abc123.tar.gz")
    # Dot-prefixed but not minted by a packer: the prefix is the full
    # ``.tmp-nerdit-``, so an operator's own dotfile is never swept.
    stranger = _touch_tar(backups, ".DS_Store")
    other_tmp = _touch_tar(backups, ".tmp-something-else")

    assert sweep_staging_orphans(tmp_path) == 3
    assert not any(p.exists() for p in partials)
    assert keeper.exists()
    assert stranger.exists()
    assert other_tmp.exists()
    assert sweep_staging_orphans(tmp_path) == 0


def test_sweep_staging_orphans_refuses_a_symlinked_staging_root(tmp_path):
    """A symlinked root is refused, never swept through (the grammar owner refuses too)."""
    outside = tmp_path / "outside"
    (outside / "realdir" / "sub").mkdir(parents=True)
    (outside / "realdir" / "sub" / "keep.txt").write_bytes(b"data")
    (outside / "stray.txt").write_bytes(b"data")
    root = dump_staging_root(tmp_path)
    root.symlink_to(outside, target_is_directory=True)

    assert sweep_staging_orphans(tmp_path) == 0
    assert (outside / "realdir" / "sub" / "keep.txt").exists()
    assert (outside / "stray.txt").exists()
    # Refused, not repaired: the link is still there for the operator to see.
    assert root.is_symlink()


def test_sweep_staging_orphans_still_sweeps_through_a_symlinked_backups_root(tmp_path):
    """The asymmetry is deliberate: P14c has always followed a relocated ``backups/``."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".staging-123-abcd").mkdir()
    keeper = _touch_tar(elsewhere, "nerdit-backup-20260907T120000Z-abc123.tar.gz")
    (tmp_path / "backups").symlink_to(elsewhere, target_is_directory=True)

    assert sweep_staging_orphans(tmp_path) == 1
    assert not (elsewhere / ".staging-123-abcd").exists()
    assert keeper.exists()


def test_sweep_staging_orphans_never_follows_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keepme").write_bytes(b"data")
    root = dump_staging_root(tmp_path)
    root.mkdir(parents=True)
    (root / "abcdef012345").symlink_to(outside)

    assert sweep_staging_orphans(tmp_path) == 1
    assert list(root.iterdir()) == []
    assert (outside / "keepme").exists()  # the link was unlinked, not followed


def test_sweep_staging_orphans_tolerates_absent_roots(tmp_path):
    assert sweep_staging_orphans(tmp_path) == 0


# --------------------------------------------------------------------------- #
# retention phase 3e
# --------------------------------------------------------------------------- #


def test_prune_dump_backups_keeps_last_n_per_service(tmp_path):
    backups = tmp_path / "backups"
    names = []
    for i in range(7):
        names.append(
            _touch_tar(
                backups,
                f"nerdit-dump-pg-20260907T1200{i:02d}Z-abc12{i}.tar.gz",
                mtime=1_000_000 + i,
            )
        )
    cache = [
        _touch_tar(
            backups, f"nerdit-dump-cache-20260907T1200{i:02d}Z-def45{i}.tar.gz", mtime=2_000_000 + i
        )
        for i in range(2)
    ]
    v1 = _touch_tar(backups, "nerdit-backup-20260907T120000Z-abc123.tar.gz")
    v2 = _touch_tar(backups, "nerdit-volumes-pg-20260907T120000Z-abc123.tar.gz")

    # The shipped default is 5 per SERVICE: two ``pg`` tars go, ``cache`` (two,
    # under the keep) is untouched.
    assert _prune_dump_backups(tmp_path, keep_last=5) == 2
    assert [n.exists() for n in names] == [False, False, True, True, True, True, True]
    assert all(c.exists() for c in cache)
    # Neither older flavour is ever touched by this phase, nor this one by them.
    assert v1.exists() and v2.exists()
    assert _prune_backups(tmp_path, keep_last=1) == 0
    assert _prune_volume_backups(tmp_path, keep_last=1) == 0
    assert len(list(backups.glob("nerdit-dump-*.tar.gz"))) == 7


def test_prune_dump_backups_tolerates_absent_dir(tmp_path):
    assert _prune_dump_backups(tmp_path, keep_last=5) == 0


def test_dump_keep_last_default_is_five_and_zero_is_allowed():
    """D-P37-7's departure from P14c's zero default, pinned where it is read.

    5 per service, not 0: a dump is the flavour agents and P28 cron repeat, so
    accumulate-forever plus a loop is disk fill by design. ``0`` stays a legal
    value (= never prune, the operator's explicit choice); a negative one is
    not.
    """
    from pydantic import ValidationError as _ValidationError

    from nerdit.config.settings import RetentionSettings

    assert RetentionSettings().dump_keep_last == 5
    assert RetentionSettings(dump_keep_last=0).dump_keep_last == 0
    with pytest.raises(_ValidationError):
        RetentionSettings(dump_keep_last=-1)


def test_dump_keep_last_is_a_restart_key():
    """The sweep loop captures ``retention`` at startup, so the key is restart-keyed.

    Pinned because the config surface derives its ``restart_keys`` diff from
    this table: a key missing from it reports "no restart needed" for a change
    that in fact only takes effect on the next boot.
    """
    from nerdit.config.store import _RESTART_KEYS

    assert "dump_keep_last" in _RESTART_KEYS["retention"]


def test_retention_loop_is_started_for_dump_keep_last_alone():
    """The lifespan predicate must see phase 3e (the WP3-2 regression).

    Phase 3e was appended to the sweep but not to the condition that decides
    whether the sweep loop is created at all, so the legitimate "I archive
    nothing else" configuration — every other threshold zeroed, the default 5
    left on dump tars — spawned no task and accumulated dumps forever, which is
    exactly the disk fill the non-zero default exists to prevent.
    """
    from nerdit.config.settings import RetentionSettings
    from nerdit.daemon.sweeps import retention_sweep_enabled

    off = {
        "job_log_days": 0,
        "audit_days": 0,
        "backup_keep_last": 0,
        "volume_backup_keep_last": 0,
        "events_keep_last": 0,
        "workspace_orphan_days": 0,
        "dump_keep_last": 0,
    }
    assert retention_sweep_enabled(RetentionSettings(**off)) is False
    assert retention_sweep_enabled(RetentionSettings(**{**off, "dump_keep_last": 5})) is True
    # And the same class of omission for the phase before it (3d, P29).
    workspace_only = RetentionSettings(**{**off, "workspace_orphan_days": 30})
    assert retention_sweep_enabled(workspace_only) is True


@pytest.mark.asyncio
async def test_retention_sweep_runs_phase_3e_and_reports_the_count(db, queries, tmp_path):
    """End-to-end through ``_run_retention_sweep``: the phase runs and is counted.

    ``_prune_dump_backups`` is pinned in isolation above; this one pins the
    WIRING — that phase 3e is actually reached by the sweep, that its count
    rides the ``retention.sweep`` summary row under its own key, and that the
    two older tar flavours are still none of its business.
    """
    from nerdit.config.settings import RetentionSettings
    from nerdit.daemon.sweeps import _run_retention_sweep

    data_dir = tmp_path / "data"
    backups = data_dir / "backups"
    dumps = [
        _touch_tar(
            backups, f"nerdit-dump-pg-20260907T1200{i:02d}Z-abc12{i}.tar.gz", mtime=1_000 + i
        )
        for i in range(3)
    ]
    v1 = _touch_tar(backups, "nerdit-backup-20260907T120000Z-abc123.tar.gz")
    v2 = _touch_tar(backups, "nerdit-volumes-pg-20260907T120000Z-abc123.tar.gz")

    retention = RetentionSettings(
        job_log_days=0,
        audit_days=0,
        backup_keep_last=0,
        volume_backup_keep_last=0,
        events_keep_last=0,
        workspace_orphan_days=0,
        dump_keep_last=1,
    )
    await _run_retention_sweep(queries, retention, data_dir, None)

    assert [d.exists() for d in dumps] == [False, False, True]
    assert v1.exists() and v2.exists()

    cur = await db.conn.execute(
        "SELECT params_redacted FROM audit_log WHERE action = 'retention.sweep' ORDER BY id"
    )
    rows = [json.loads(r[0]) for r in await cur.fetchall()]
    assert rows[-1]["dump_backups_removed"] == 2

    # 0 = never: the phase is skipped outright. Nothing more is removed, and a
    # pass that pruned nothing writes no summary row at all — so the row count
    # standing still IS the assertion that phase 3e did nothing.
    await _run_retention_sweep(
        queries, retention.model_copy(update={"dump_keep_last": 0}), data_dir, None
    )
    assert dumps[2].exists()
    cur = await db.conn.execute("SELECT COUNT(*) FROM audit_log WHERE action = 'retention.sweep'")
    assert (await cur.fetchone())[0] == len(rows)


# --------------------------------------------------------------------------- #
# /system/disk bucket
# --------------------------------------------------------------------------- #


class _FakeRuntime:
    async def disk_usage(self):  # noqa: ANN201
        return {"images_bytes": 0, "containers_bytes": 0, "volumes_bytes": 0}

    async def list_images_detailed(self):  # noqa: ANN201
        return []


class _FakeQueries:
    async def list_workload_configs(self):  # noqa: ANN201
        return []


@pytest.mark.asyncio
async def test_disk_bucket_counts_dumps_and_staging_leftovers(tmp_path):
    result, _ = _pack(tmp_path)
    size = os.stat(result.path).st_size
    # A leftover staging dir (a crashed dump) must be visible, not invisible.
    leftover = dump_staging_root(tmp_path) / "abcdef012399"
    leftover.mkdir(parents=True)
    (leftover / "dump.pgdump").write_bytes(b"x" * 321)

    report = await _build_disk_report(_FakeRuntime(), _FakeQueries(), tmp_path, None)
    bucket = report["data_dir"]["dumps"]
    assert bucket["count"] == 1
    assert bucket["bytes"] == size
    assert report["data_dir"]["dump_staging_bytes"] == 321
    # Neither older bucket double-counts the dump tar.
    assert report["data_dir"]["backups"]["count"] == 0
    assert report["data_dir"]["volume_backups"]["count"] == 0
