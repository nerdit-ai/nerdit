"""Unit tests for the named-volume resolution seams (P14 WP-0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nerdit.core.volumes import (
    VolumeSpecError,
    make_tombstone_name,
    resolve_named_volumes,
    service_data_root,
    service_volumes,
    tombstone_service_name,
)


def test_service_data_root_shape(tmp_path: Path):
    root = service_data_root(tmp_path, "myapp")
    assert root == (tmp_path / "services" / "myapp").resolve()


@pytest.mark.parametrize("bad", ["", "Bad_Name", "under_score", "-lead", "a" * 64, "../etc"])
def test_service_data_root_rejects_bad_names(tmp_path: Path, bad: str):
    with pytest.raises(VolumeSpecError):
        service_data_root(tmp_path, bad)


def test_resolve_single_named_volume(tmp_path: Path):
    out = resolve_named_volumes("app", ["data:/data"], tmp_path)
    host = str((tmp_path / "services" / "app" / "data").resolve())
    assert out == {host: "/data"}


def test_resolve_multiple_volumes(tmp_path: Path):
    out = resolve_named_volumes("app", ["data:/data", "cache:/var/cache"], tmp_path)
    assert len(out) == 2
    assert set(out.values()) == {"/data", "/var/cache"}
    for host in out:
        assert Path(host).is_relative_to((tmp_path / "services" / "app").resolve())


def test_volname_data_allowed(tmp_path: Path):
    # ``data`` re-targets the implicit volume — must be accepted.
    out = resolve_named_volumes("app", ["data:/data"], tmp_path)
    assert out


@pytest.mark.parametrize(
    "spec",
    [
        "data",  # missing ':'
        ":/data",  # empty volname
        "Bad:/data",  # uppercase volname
        "under_score:/data",  # underscore volname
        ("x" * 33) + ":/data",  # volname too long (>32)
        "data:relative/path",  # non-absolute container path
        "data:/",  # forbidden root
        "data:/workspace",  # forbidden workspace
        "data:/proc",  # kernel-virtual root
        "data:/sys/kernel",  # under /sys
        "data:/dev/shm",  # under /dev
        "data:/data/../etc",  # non-normalized
        "data:/data:rw",  # extra ':' (options)
        "x://workspace",  # doubled leading slash — kernel collapses //x -> /x
        "x://",  # doubled slash collapsing to forbidden root
        "x://proc",  # doubled slash collapsing to a kernel-virtual root
    ],
)
def test_grammar_rejects(tmp_path: Path, spec: str):
    with pytest.raises(VolumeSpecError):
        resolve_named_volumes("app", [spec], tmp_path)


def test_traversal_in_volname_is_rejected(tmp_path: Path):
    # A '/' cannot pass the DNS-label volname regex, so a traversal never
    # reaches the is_relative_to assert — but assert it fails closed regardless.
    with pytest.raises(VolumeSpecError):
        resolve_named_volumes("app", ["../../etc:/x"], tmp_path)


def test_duplicate_names_rejected(tmp_path: Path):
    with pytest.raises(VolumeSpecError):
        resolve_named_volumes("app", ["data:/a", "data:/b"], tmp_path)


def test_duplicate_container_paths_rejected(tmp_path: Path):
    with pytest.raises(VolumeSpecError):
        resolve_named_volumes("app", ["one:/data", "two:/data"], tmp_path)


def test_max_volumes_enforced(tmp_path: Path):
    specs = [f"v{i}:/p{i}" for i in range(9)]
    with pytest.raises(VolumeSpecError):
        resolve_named_volumes("app", specs, tmp_path)
    # exactly 8 is allowed
    ok = resolve_named_volumes("app", [f"v{i}:/p{i}" for i in range(8)], tmp_path)
    assert len(ok) == 8


def test_forged_service_name_rejected_by_seam(tmp_path: Path):
    with pytest.raises(VolumeSpecError):
        resolve_named_volumes("../evil", ["data:/data"], tmp_path)


def test_service_volumes_empty(tmp_path: Path):
    assert service_volumes(tmp_path, "app", {}) == {}
    assert service_volumes(tmp_path, "app", {"volumes": []}) == {}
    assert service_volumes(tmp_path, "app", {"volumes": None}) == {}


def test_service_volumes_reads_cfg(tmp_path: Path):
    out = service_volumes(tmp_path, "app", {"volumes": ["data:/data"]})
    assert list(out.values()) == ["/data"]


def test_service_volumes_non_list_rejected(tmp_path: Path):
    with pytest.raises(VolumeSpecError):
        service_volumes(tmp_path, "app", {"volumes": "data:/data"})


def test_symlinked_services_dir_rejected(tmp_path: Path):
    # A symlink at <data_dir>/services must never become the trusted base:
    # .resolve() would adopt its target and relocate every mount (Codex P2).
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "services").symlink_to(target)
    with pytest.raises(VolumeSpecError, match="symlink"):
        service_data_root(tmp_path, "app")


def test_symlinked_service_root_rejected(tmp_path: Path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "app").symlink_to(target)
    with pytest.raises(VolumeSpecError, match="symlink"):
        service_data_root(tmp_path, "app")


def test_symlinked_volume_leaf_rejected(tmp_path: Path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    root = tmp_path / "services" / "app"
    root.mkdir(parents=True)
    (root / "data").symlink_to(target)
    with pytest.raises(VolumeSpecError, match="symlink"):
        resolve_named_volumes("app", ["data:/data"], tmp_path)


def test_nonexistent_dirs_still_resolve(tmp_path: Path):
    # The symlink checks must not require the dirs to exist yet — they are
    # created at launch, after resolution.
    out = resolve_named_volumes("app", ["data:/data"], tmp_path)
    assert str((tmp_path / "services" / "app" / "data").resolve()) in out


# --- C2 tombstone naming/parsing convention ----------------------------------


def test_tombstone_name_roundtrips():
    name = make_tombstone_name("my-db")
    assert name.startswith(".trash-my-db-")
    assert tombstone_service_name(name) == "my-db"


def test_tombstone_name_rejects_bad_service_name():
    with pytest.raises(VolumeSpecError):
        make_tombstone_name(".evil")  # a leading dot is not a DNS label


def test_tombstone_service_name_rejects_non_tombstones():
    # A plain service dir, an unparseable dotdir, and a bad nonce are all None.
    assert tombstone_service_name("pg") is None
    assert tombstone_service_name(".hidden") is None
    assert tombstone_service_name(".trash-pg") is None  # no nonce
    assert tombstone_service_name(".trash-pg-XYZ") is None  # nonce not hex
    assert tombstone_service_name(".trash-pg-abcd1234") == "pg"  # 8 hex ok


def test_tombstone_name_never_resolves_as_service_root(tmp_path: Path):
    # Defense in depth: a tombstone name must never be accepted by
    # service_data_root (the leading dot fails the DNS-label grammar).
    with pytest.raises(VolumeSpecError):
        service_data_root(tmp_path, make_tombstone_name("pg"))
