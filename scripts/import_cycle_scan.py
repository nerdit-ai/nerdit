#!/usr/bin/env python3
"""Find import cycles in src/nerdit, including deferred function-body imports.

Resolve absolute and package-relative imports, excluding TYPE_CHECKING guards.
Report strongly connected components larger than one; exit zero when none exist.
"""

import ast
import sys
from pathlib import Path

SRC = Path("src/nerdit")
mods: dict[str, Path] = {}
pkgs: set[str] = set()
for p in SRC.rglob("*.py"):
    rel = p.relative_to("src").with_suffix("")
    name = ".".join(rel.parts)
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
        pkgs.add(name)
    mods[name] = p


def _is_type_checking(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
    )


def targets(node: ast.stmt, mod_name: str) -> list[str]:
    if isinstance(node, ast.Import):
        out = [a.name for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            # Package-relative: level 1 is the importing module's own package
            # (the package itself for an __init__), each extra level one parent
            # up — then ``node.module`` (absent for ``from . import x``).
            parts = mod_name.split(".")
            drop = node.level - 1 if mod_name in pkgs else node.level
            if drop >= len(parts):
                return []
            base = ".".join(parts[: len(parts) - drop])
            prefix = f"{base}.{node.module}" if node.module else base
        elif node.module:
            prefix = node.module
        else:
            return []
        out = [prefix] + [f"{prefix}.{a.name}" for a in node.names]
    else:
        out = []
    return [m for m in out if m.startswith("nerdit")]


edges: dict[str, set[str]] = {m: set() for m in mods}
deferred: set[tuple[str, str]] = set()
for name, path in mods.items():
    tree = ast.parse(path.read_text())

    def visit(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if _is_type_checking(child):
                continue  # typing-only imports create no runtime edge
            f = in_func or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for tgt in targets(child, name):  # noqa: B023 — visit() runs synchronously per iteration
                    t = tgt
                    while t and t not in mods:
                        t = t.rpartition(".")[0]
                    if t and t != name:  # noqa: B023 — visit() runs synchronously per iteration
                        edges[name].add(t)  # noqa: B023
                        if f:
                            deferred.add((name, t))  # noqa: B023
            visit(child, f)

    visit(tree, False)

sys.setrecursionlimit(100000)
counter = [0]
stack: list[str] = []
low: dict[str, int] = {}
idx: dict[str, int] = {}
on: dict[str, bool] = {}
sccs: list[list[str]] = []


def connect(v: str) -> None:
    idx[v] = low[v] = counter[0]
    counter[0] += 1
    stack.append(v)
    on[v] = True
    for w in edges.get(v, ()):
        if w not in idx:
            connect(w)
            low[v] = min(low[v], low[w])
        elif on.get(w):
            low[v] = min(low[v], idx[w])
    if low[v] == idx[v]:
        scc = []
        while True:
            w = stack.pop()
            on[w] = False
            scc.append(w)
            if w == v:
                break
        if len(scc) > 1:
            sccs.append(scc)


for v in list(mods):
    if v not in idx:
        connect(v)

for scc in sccs:
    print("CYCLE:", " <-> ".join(sorted(scc)))
    for a in scc:
        for b in edges[a]:
            if b in scc:
                kind = "deferred" if (a, b) in deferred else "module-level"
                print(f"   {a} -> {b} [{kind}]")
if not sccs:
    print("0 cycles (including deferred imports).")
sys.exit(1 if sccs else 0)
