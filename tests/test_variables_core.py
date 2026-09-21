"""Unit tests for the merged variable reader (P40c / D-P40-9)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nerdit.core.secrets import SecretManager, project_storage_name
from nerdit.core.variables import load_scoped, plain_keys
from nerdit.db.rows import VariableFlag

_PRJ = "prj_abcdefghij234567"


@pytest.fixture
def mgr(tmp_path) -> SecretManager:
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(project_storage_name(_PRJ), {"SHARED_KEY": "project", "ONLY_PROJECT": "p"})
    mgr.set("asso", {"SHARED_KEY": "service", "ONLY_SERVICE": "s"})
    return mgr


def test_service_scope_wins_over_project_scope(mgr):
    env, winners = load_scoped(mgr, "asso", _PRJ)
    assert env == {"SHARED_KEY": "service", "ONLY_PROJECT": "p", "ONLY_SERVICE": "s"}
    assert winners == {
        "SHARED_KEY": "service",
        "ONLY_PROJECT": "project",
        "ONLY_SERVICE": "service",
    }


def test_include_project_false_never_reads_the_project_file(mgr, monkeypatch):
    loaded: list[str] = []
    real = mgr.load
    monkeypatch.setattr(mgr, "load", lambda name: loaded.append(name) or real(name))
    env, winners = load_scoped(mgr, "asso", _PRJ, include_project=False)
    assert env == {"SHARED_KEY": "service", "ONLY_SERVICE": "s"}
    assert set(winners.values()) == {"service"}
    assert loaded == ["asso"]  # the gate is structural: the file is not even opened


def test_include_service_false_reads_only_the_project_scope(mgr):
    env, winners = load_scoped(mgr, "asso", _PRJ, include_service=False)
    assert env == {"SHARED_KEY": "project", "ONLY_PROJECT": "p"}
    assert set(winners.values()) == {"project"}


def test_both_gates_closed_is_empty(mgr):
    assert load_scoped(mgr, "asso", _PRJ, include_service=False, include_project=False) == ({}, {})


def test_none_project_degrades_to_service_only(mgr):
    env, winners = load_scoped(mgr, "asso", None)
    assert env == {"SHARED_KEY": "service", "ONLY_SERVICE": "s"}
    assert set(winners.values()) == {"service"}


def test_none_label_skips_the_service_scope(mgr):
    assert load_scoped(mgr, None, _PRJ)[0] == {"SHARED_KEY": "project", "ONLY_PROJECT": "p"}


def test_missing_files_are_empty_not_errors(tmp_path):
    empty = SecretManager(tmp_path / "secrets")
    assert load_scoped(empty, "asso", _PRJ) == ({}, {})


def test_shared_scope_is_never_merged(mgr):
    mgr.set("_shared", {"MACHINE": "m"})
    assert "MACHINE" not in load_scoped(mgr, "asso", _PRJ)[0]


async def test_service_wins_regardless_of_flag_and_plain_keys_is_per_scope(mgr):
    """A key plain at project scope and secret at service scope resolves to the service."""
    queries = AsyncMock()
    queries.list_variable_flags = AsyncMock(
        return_value=[
            VariableFlag(key="SHARED_KEY", service="", plain=True),
            VariableFlag(key="ONLY_PROJECT", service="", plain=False),
            VariableFlag(key="SHARED_KEY", service="web", plain=False),
            VariableFlag(key="ONLY_SERVICE", service="web", plain=True),
        ]
    )
    assert await plain_keys(queries, _PRJ) == {"SHARED_KEY"}
    assert await plain_keys(queries, _PRJ, "web") == {"ONLY_SERVICE"}
    assert await plain_keys(queries, _PRJ, "api") == set()
    env, winners = load_scoped(mgr, "asso", _PRJ)
    assert (env["SHARED_KEY"], winners["SHARED_KEY"]) == ("service", "service")


# --- the seven load sites stay replaced (AST pin) -----------------------------

# `(file, enclosing function)` pairs that legitimately read ONE scope file, and why.
_SINGLE_FILE_READS = {
    # A `kind=database` row's minted credential lives under its own label and
    # database rows carry no project (`project_id` NULL): nothing to merge.
    ("core/services.py", "_managed_password"),
    # `load_scope` handed to the `[db.*]` resolvers: it reads the BOUND
    # database's label (its minted credential), never the app's own scope.
    ("core/services.py", "_resolve_launch_env"),
    ("daemon/routes/service_diagnose.py", "_classify_bindings"),
}
# Receivers whose `.load` is not a `SecretManager` read. The secrets route and
# `daemon/secret_scope.py` are absent on purpose: they address one scope file
# through `list_keys`/`set`/`delete*` and never call `load`.
_FOREIGN_RECEIVERS = {"json", "tomllib", "tomli", "_admin"}


def _bare_secret_loads() -> list[tuple[str, str, int]]:
    import ast
    from pathlib import Path

    import nerdit

    root = Path(nerdit.__file__).parent
    hits: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in ("core/variables.py", "core/secrets.py") or rel.startswith("daemon/web/"):
            continue
        tree = ast.parse(path.read_text())
        # The machine scope is ref-only and never merged (D-P40-9): `load(SHARED_SCOPE)` stays.
        shared = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "SHARED_SCOPE"
        }

        def visit(node: ast.AST, func: str, rel: str = rel, shared: set[int] = shared) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                func = node.name if func == "<module>" else func
            if isinstance(node, ast.Attribute) and node.attr == "load" and id(node) not in shared:
                receiver = node.value
                name = getattr(receiver, "attr", getattr(receiver, "id", ""))
                if name not in _FOREIGN_RECEIVERS:
                    hits.append((rel, func, node.lineno))
            for child in ast.iter_child_nodes(node):
                visit(child, func)

        visit(tree, "<module>")
    return hits


def test_no_bare_secret_load_survives_outside_the_merged_reader():
    """Every per-label read goes through `load_scoped`, or is allow-listed above with a reason."""
    hits = _bare_secret_loads()
    stray = [hit for hit in hits if (hit[0], hit[1]) not in _SINGLE_FILE_READS]
    assert stray == []
    # The allow-list stays honest: an entry whose site disappeared must be deleted.
    assert {(rel, func) for rel, func, _ in hits} == _SINGLE_FILE_READS
    # `_resolve_launch_env` is allow-listed for the `load_scope` hand-off ONLY.
    assert [func for _, func, _ in hits].count("_resolve_launch_env") == 1
