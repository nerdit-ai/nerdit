"""Pinned Node resolution accepts common ranges and rejects unsafe declarations."""

import pytest

from nerdit.core.node_runtime import DEFAULT_NODE_VERSION, resolve_node_version


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("*", "24.20.0"),
        (">=20", "24.20.0"),
        (">=22 <24", "22.23.2"),
        ("^22.0.0", "22.23.2"),
        ("~24.20", "24.20.0"),
        ("22.x", "22.23.2"),
        ("v24", "24.20.0"),
        ("22 || 24", "24.20.0"),
        ("22 - 24", "24.20.0"),
        (">22.22 <=22.23", "22.23.2"),
        (">= 22.0.0 < 24.0.0", "22.23.2"),
        ("=22.23.2", "22.23.2"),
        ("~22", "22.23.2"),
        ("<24", "22.23.2"),
        ("24.20.*", "24.20.0"),
        ("22.23.2 - 24.20.0", "24.20.0"),
    ],
)
def test_supported_ranges(tmp_path, requirement, expected):
    assert resolve_node_version(tmp_path, {"engines": {"node": requirement}}) == expected


def test_default_and_intersection(tmp_path):
    assert resolve_node_version(tmp_path, {}) == DEFAULT_NODE_VERSION
    (tmp_path / ".nvmrc").write_text("v22\n")
    (tmp_path / ".node-version").write_text("22.23\n")
    assert resolve_node_version(tmp_path, {"engines": {"node": ">=22"}}) == "22.23.2"
    (tmp_path / ".node-version").write_text("24")
    with pytest.raises(ValueError, match="Dockerfile"):
        resolve_node_version(tmp_path, {})


@pytest.mark.parametrize(
    "requirement",
    [
        "20",
        "22.1.0",
        "^0.0",
        "~22.0",
        ">24",
        "<22",
        "^25",
        "",
        " ",
        "22 ||",
        "|| 22",
        "22.x.2",
        "022",
        "22.0.0-beta.1",
        "lts/*",
        "node",
        ">=22, <25",
        "24; curl SECRET",
        "$(SECRET)",
        "24\nRUN SECRET",
        "24\x00",
        "1" * 513,
        None,
        22,
        {},
        ["22"],
        ">*",
    ],
)
def test_invalid_or_unsatisfied_ranges_are_safe(tmp_path, requirement):
    with pytest.raises(ValueError, match="Dockerfile") as error:
        resolve_node_version(tmp_path, {"engines": {"node": requirement}})
    assert "SECRET" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("value", [None, "22", []])
def test_invalid_engines(tmp_path, value):
    with pytest.raises(ValueError, match="Dockerfile"):
        resolve_node_version(tmp_path, {"engines": value})


@pytest.mark.parametrize("contents", [b"lts/*", b"24\nRUN SECRET", b"24" * 300, b"\xff"])
def test_version_file_rejects_unsupported_or_unsafe_values(tmp_path, contents):
    (tmp_path / ".nvmrc").write_bytes(contents)
    with pytest.raises(ValueError, match="Dockerfile") as error:
        resolve_node_version(tmp_path, {})
    assert "SECRET" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


def test_symlink_and_directory_not_followed(tmp_path):
    path = tmp_path / ".nvmrc"
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="Dockerfile"):
        resolve_node_version(tmp_path, {})
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="Dockerfile"):
        resolve_node_version(tmp_path, {})


@pytest.mark.parametrize("requirement", ["24 || SECRET", "20 || SECRET", "24 || 22.x.2"])
def test_every_or_branch_is_validated(tmp_path, requirement):
    with pytest.raises(ValueError, match="Dockerfile") as error:
        resolve_node_version(tmp_path, {"engines": {"node": requirement}})
    assert "SECRET" not in str(error.value)


def test_engine_and_file_conflict(tmp_path):
    (tmp_path / ".node-version").write_text("22")
    with pytest.raises(ValueError, match="Dockerfile"):
        resolve_node_version(tmp_path, {"engines": {"node": "^24"}})
