"""Route + helper tests for the P14b WP-A2 disk/GC surface —
``GET /system/disk`` and ``POST /system/gc``.

Harness mirrors ``test_routes_system.py``: a hand-built FastAPI app with mocked
state on ``app.state`` (no real aiosqlite crossing event loops), the real auth +
audit + error middleware. A ``FakeRuntime`` supplies exactly the new methods the
disk/GC paths call; ``StubRuntime`` drives the docker-unavailable 503.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core.runtime.stub import StubRuntime
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.imagegc import _orphan_app_repos, _protected_image_refs
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes import system as system_routes
from nerdit.daemon.routes.system import router as system_router
from nerdit.db.models import TokenRole

_ADMIN = {"Authorization": "Bearer admin-raw-token"}
_READONLY = {"Authorization": "Bearer readonly-raw-token"}


# --- fakes --------------------------------------------------------------------


class FakeRuntime:
    """Minimal runtime exposing only the disk/GC methods (reconcile never runs).

    ``list_images_detailed`` entries default to ``instance="default"`` — i.e.
    "built by this daemon", the shape a post-Track-0.4 build produces — so a test
    only spells the label out when instance ownership is what it is about. Pass
    ``instance: None`` explicitly for a pre-label (unlabelled) image.
    """

    def __init__(self, detailed, df=None, refuse=()):  # noqa: ANN001
        self._detailed = [{"instance": "default", **dict(e)} for e in detailed]
        self._df = (
            df
            if df is not None
            else {
                "images_bytes": 100,
                "containers_bytes": 10,
                "volumes_bytes": 20,
                "build_cache_bytes": 30,
            }
        )
        self.refuse = set(refuse)
        self.removed: list[str] = []

    async def disk_usage(self):  # noqa: ANN201
        return dict(self._df) if self._df is not None else None

    async def list_images_detailed(self):  # noqa: ANN201
        return [dict(e) for e in self._detailed]

    async def remove_image(self, tag, force=False):  # noqa: ANN001
        self.removed.append(tag)
        if tag not in self.refuse:
            self._detailed = [e for e in self._detailed if e["repo_tag"] != tag]

    async def list_managed_containers(self):  # noqa: ANN201
        return []


def _token_row(role: TokenRole) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"tok-{role.value}",
        name="ci",
        role=role,
        max_gpus=2,
        max_concurrent_jobs=4,
        # P25 WP1: the middleware reads both on every scoped-token request.
        expires_at=None,
        scope_services=None,
    )


def _seq(values):  # noqa: ANN001
    """Return an async callable yielding each ``values`` entry, repeating the last."""
    vals = [list(v) for v in values]
    box = {"i": 0}

    async def _call(*_a, **_k):  # noqa: ANN002, ANN003
        i = min(box["i"], len(vals) - 1)
        box["i"] += 1
        return list(vals[i])

    return _call


def _queries(role, *, workloads=None, workloads_seq=None):  # noqa: ANN001
    row = _token_row(role) if role is not None else None
    q = SimpleNamespace(
        get_api_token_by_hash=AsyncMock(return_value=row),
        touch_api_token=AsyncMock(),
        insert_audit_log=AsyncMock(),
    )
    if workloads_seq is not None:
        q.list_workload_configs = _seq(workloads_seq)
    else:
        q.list_workload_configs = AsyncMock(return_value=list(workloads or []))
    return q


def _app(  # noqa: ANN001
    *, runtime, queries, data_dir, keep=0, archive_dir="", instance_id="default"
) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.settings = SimpleNamespace(
        data_dir=str(data_dir),
        # ``audit_archive_dir`` is what the disk route resolves the effective
        # archive dir from (F14); "" ⇒ the <data_dir>/archive default.
        retention=SimpleNamespace(backup_keep_last=keep, audit_archive_dir=archive_dir),
        # Track 0.4: the image-GC ownership scope.
        daemon=SimpleNamespace(instance_id=instance_id),
    )
    app.state.runtime = runtime
    app.state.queries = queries
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(
        ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: queries
    )
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def _svc_row(name, *, image_repo=None, image=None, previous_image=None, phase=None, kind="service"):  # noqa: ANN001
    cfg: dict = {}
    if image_repo:
        cfg["image_repo"] = image_repo
    if image:
        cfg["image"] = image
    if previous_image:
        cfg["previous_image"] = previous_image
    if phase:
        cfg["last_deploy"] = {"phase": phase}
    return {
        "id": f"job-{name}",
        "kind": kind,
        "service_name": name,
        "status": "running",
        "config": cfg,
    }


def _seed_dir(root: Path, size: int = 16) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "blob.bin").write_bytes(b"x" * size)


# --- _protected_image_refs (pure) --------------------------------------------


def test_protected_refs_spares_repo_tags_and_model_images():
    rows = [
        _svc_row(
            "a", image_repo="nerdit-app/a", image="nerdit-app/a:2", previous_image="nerdit-app/a:1"
        ),
        _svc_row("m", image="ollama/ollama:latest", kind="model"),
    ]
    tags = [
        "nerdit-app/a:1",
        "nerdit-app/a:2",
        "nerdit-app/a:3",
        "ollama/ollama:latest",
        "nerdit-app/b:1",
    ]
    protected = _protected_image_refs(rows, tags)
    # Every existing tag of a live repo is protected (keep-last-N older versions
    # too), plus the model image via its row's config['image'].
    assert protected == {
        "nerdit-app/a:1",
        "nerdit-app/a:2",
        "nerdit-app/a:3",
        "ollama/ollama:latest",
    }
    # nerdit-app/b has no live row → not protected (a reclaim candidate).
    assert "nerdit-app/b:1" not in protected


# --- GET /system/disk ---------------------------------------------------------


def test_disk_composition_and_orphans(tmp_path):
    detailed = [
        {"repo_tag": "nerdit-app/live:1", "id": "a", "size_bytes": 50},
        {"repo_tag": "nerdit-app/orphan:1", "id": "b", "size_bytes": 70},
        {"repo_tag": "ollama/ollama:latest", "id": "c", "size_bytes": 200},
    ]
    rows = [
        _svc_row("live", image_repo="nerdit-app/live", image="nerdit-app/live:1"),
        _svc_row("m", image="ollama/ollama:latest", kind="model"),
    ]
    _seed_dir(tmp_path / "services" / "live")
    _seed_dir(tmp_path / "services" / "ghost")
    client = _app(
        runtime=FakeRuntime(detailed), queries=_queries(None, workloads=rows), data_dir=tmp_path
    )

    body = client.get("/api/system/disk", headers=_ADMIN).json()
    assert body["docker"]["images_bytes"] == 100
    assert set(body["images"]["by_repo"]) == {
        "nerdit-app/live",
        "nerdit-app/orphan",
        "ollama/ollama",
    }
    assert body["orphan_images"] == ["nerdit-app/orphan"]
    assert body["orphan_data_dirs"] == ["ghost"]  # live has a row; ghost does not
    names = {s["name"] for s in body["data_dir"]["services"]}
    assert names == {"live", "ghost"}
    for svc in body["data_dir"]["services"]:
        assert svc["bytes"] > 0
    assert body["data_dir"]["backups"] == {"bytes": 0, "count": 0}  # absent dir
    assert body["warnings"] == []


def test_disk_report_includes_workspaces_names_and_bytes(tmp_path):
    """(P29) The agent workspace trees get the same names-and-bytes treatment.

    A workspace is daemon-owned and never container-mounted, so nothing else in
    the report would ever account for it — without this leg the bytes are simply
    invisible.
    """
    _seed_dir(tmp_path / "workspaces" / "alpha" / "tree", size=32)
    (tmp_path / "workspaces" / "alpha" / "meta.json").write_bytes(b"{}")
    _seed_dir(tmp_path / "workspaces" / "beta" / "tree", size=8)
    _seed_dir(tmp_path / "services" / "alpha")
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)

    body = client.get("/api/system/disk", headers=_ADMIN).json()["data_dir"]
    assert [w["name"] for w in body["workspaces"]] == ["alpha", "beta"]  # sorted
    assert all(w["bytes"] > 0 for w in body["workspaces"])
    assert body["workspaces_total_bytes"] == sum(w["bytes"] for w in body["workspaces"])
    # Disjoint buckets: the services walk never counts a workspace and vice versa.
    assert [s["name"] for s in body["services"]] == ["alpha"]
    assert body["workspaces_total_bytes"] != body["services_total_bytes"]


def test_disk_report_absent_workspaces_dir_is_empty_not_an_error(tmp_path):
    """The root is created lazily at the first write — absent ⇒ [] / 0, never 500."""
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)

    body = client.get("/api/system/disk", headers=_ADMIN)
    assert body.status_code == 200
    assert body.json()["data_dir"]["workspaces"] == []
    assert body.json()["data_dir"]["workspaces_total_bytes"] == 0


def test_disk_requires_auth(tmp_path):
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)
    assert client.get("/api/system/disk").status_code == 401


def test_disk_body_has_no_absolute_paths(tmp_path):
    _seed_dir(tmp_path / "services" / "svc")
    _seed_dir(tmp_path / "models" / "ollama")
    detailed = [{"repo_tag": "nerdit-app/svc:1", "id": "a", "size_bytes": 5}]
    rows = [_svc_row("svc", image_repo="nerdit-app/svc", image="nerdit-app/svc:1")]
    client = _app(
        runtime=FakeRuntime(detailed), queries=_queries(None, workloads=rows), data_dir=tmp_path
    )
    text = json.dumps(client.get("/api/system/disk", headers=_ADMIN).json())
    # The data_dir absolute path must not leak, and no JSON string value may be
    # an absolute path (repo/name tokens never start with '/').
    assert str(tmp_path) not in text
    assert re.search(r'"/[^"]', text) is None


def test_disk_backups_accounting(tmp_path):
    bdir = tmp_path / "backups"
    bdir.mkdir(parents=True)
    (bdir / "nerdit-backup-2026-07-13.tar.gz").write_bytes(b"z" * 42)
    (bdir / "unrelated.txt").write_bytes(b"nope")
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)
    backups = client.get("/api/system/disk", headers=_ADMIN).json()["data_dir"]["backups"]
    assert backups == {"bytes": 42, "count": 1}


def test_walk_backups_counts_real_backup_tar(tmp_path):
    """D1 naming contract: a real ``create_backup`` tar is counted by
    ``_walk_backups`` (the disk-report walker) under the locked glob."""
    import asyncio

    from nerdit.core.backup import create_backup
    from nerdit.core.secrets import SecretManager
    from nerdit.daemon.routes.system import _walk_backups
    from nerdit.db.database import Database

    async def _produce():  # noqa: ANN202
        db = Database(":memory:")
        await db.connect()
        await db.init_schema()
        sm = SecretManager(tmp_path / "secrets")
        sm.set("app", {"K": "v"})
        try:
            return await create_backup(db=db, secret_manager=sm, data_dir=tmp_path)
        finally:
            await db.close()

    result = asyncio.run(_produce())
    walked = _walk_backups(tmp_path / "backups")
    assert walked["count"] == 1
    assert walked["bytes"] == (tmp_path / "backups" / result.basename).stat().st_size


def test_disk_walks_configured_archive_dir(tmp_path):
    """A custom [retention].audit_archive_dir is the dir that gets accounted (F14).

    Pre-F14 the route walked <data_dir>/archive unconditionally, so an operator
    with a custom location saw the (stale/absent) default while the real archive
    grew unwatched.
    """
    custom = tmp_path / "custom-archive"
    _seed_dir(custom, size=64)
    _seed_dir(tmp_path / "archive", size=999)  # the default dir is a decoy
    client = _app(
        runtime=FakeRuntime([]),
        queries=_queries(None),
        data_dir=tmp_path,
        archive_dir=str(custom),
    )

    body = client.get("/api/system/disk", headers=_ADMIN).json()
    assert body["data_dir"]["archive_bytes"] == 64  # the custom dir, not the decoy
    assert body["warnings"] == []
    # The resolved path never enters the body (no-absolute-paths posture).
    assert str(custom) not in json.dumps(body)


def test_disk_archive_defaults_to_data_dir_archive(tmp_path):
    _seed_dir(tmp_path / "archive", size=33)
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)
    body = client.get("/api/system/disk", headers=_ADMIN).json()
    assert body["data_dir"]["archive_bytes"] == 33


def test_disk_archive_disabled_reports_zero_without_error(tmp_path):
    """A guard-rejected archive dir (archiving DISABLED) ⇒ 0 bytes, no error.

    ``0`` and not ``null``: null is the walk's timed-out/failed value, and the two
    must stay distinguishable.
    """
    evil = tmp_path / "services" / "evil"  # under <data_dir>/services ⇒ rejected
    _seed_dir(evil, size=128)
    client = _app(
        runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path, archive_dir=str(evil)
    )

    resp = client.get("/api/system/disk", headers=_ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_dir"]["archive_bytes"] == 0
    assert body["warnings"] == []


def test_disk_readonly_allowed(tmp_path):
    client = _app(runtime=FakeRuntime([]), queries=_queries(TokenRole.readonly), data_dir=tmp_path)
    assert client.get("/api/system/disk", headers=_READONLY).status_code == 200


# --- POST /system/gc ----------------------------------------------------------


def test_gc_requires_admin(tmp_path):
    client = _app(runtime=FakeRuntime([]), queries=_queries(TokenRole.readonly), data_dir=tmp_path)
    # readonly is coarse-gated on mutating methods before the route.
    assert client.post("/api/system/gc", headers=_READONLY).status_code == 403


def test_gc_docker_unavailable_503(tmp_path):
    client = _app(runtime=StubRuntime(), queries=_queries(None), data_dir=tmp_path)
    resp = client.post("/api/system/gc", headers=_ADMIN)
    assert resp.status_code == 503
    assert resp.json()["code"] == "system.docker_unavailable"


def test_gc_docker_unavailable_503_on_dry_run(tmp_path):
    client = _app(runtime=StubRuntime(), queries=_queries(None), data_dir=tmp_path)
    resp = client.post("/api/system/gc?dry_run=true", headers=_ADMIN)
    assert resp.status_code == 503


def test_gc_concurrent_409(tmp_path, monkeypatch):
    monkeypatch.setattr(system_routes._gc_lock, "locked", lambda: True)
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)
    resp = client.post("/api/system/gc", headers=_ADMIN)
    assert resp.status_code == 409
    assert resp.json()["code"] == "system.gc_in_progress"


def test_gc_dry_run_enumerates_without_removing(tmp_path):
    detailed = [{"repo_tag": "nerdit-app/orphan:1", "id": "b", "size_bytes": 70}]
    q = _queries(None, workloads=[])
    runtime = FakeRuntime(detailed)
    client = _app(runtime=runtime, queries=q, data_dir=tmp_path)

    body = client.post("/api/system/gc?dry_run=true", headers=_ADMIN).json()
    assert body["dry_run"] is True
    assert body["images"]["removed"] == ["nerdit-app/orphan"]
    assert body["images"]["reclaimed_bytes_estimate"] == 70
    assert runtime.removed == []  # zero writes
    # The dry-run audit action is overridden to system.gc_plan.
    assert q.insert_audit_log.await_args.kwargs["action"] == "system.gc_plan"


def test_gc_real_removes_orphan_and_audits(tmp_path):
    detailed = [
        {"repo_tag": "nerdit-app/orphan:1", "id": "b", "size_bytes": 70},
        {"repo_tag": "nerdit-app/live:1", "id": "a", "size_bytes": 50},
    ]
    rows = [_svc_row("live", image_repo="nerdit-app/live", image="nerdit-app/live:1")]
    q = _queries(None, workloads=rows)
    runtime = FakeRuntime(detailed)
    client = _app(runtime=runtime, queries=q, data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"]["removed"] == ["nerdit-app/orphan"]
    assert body["images"]["skipped"] == []
    assert runtime.removed == ["nerdit-app/orphan:1"]  # live repo untouched
    assert q.insert_audit_log.await_args.kwargs["action"] == "system.gc"


def test_gc_reports_skipped_when_removal_refused(tmp_path):
    detailed = [{"repo_tag": "nerdit-app/orphan:1", "id": "b", "size_bytes": 70}]
    # remove_image is a no-op for this tag → re-list still shows it → skipped.
    runtime = FakeRuntime(detailed, refuse={"nerdit-app/orphan:1"})
    client = _app(runtime=runtime, queries=_queries(None, workloads=[]), data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"]["removed"] == []
    assert body["images"]["skipped"] == [{"repo": "nerdit-app/orphan", "reason": "in_use_or_error"}]


def test_gc_toctou_image_skips_repo_that_regains_a_row(tmp_path):
    detailed = [{"repo_tag": "nerdit-app/orphan:1", "id": "b", "size_bytes": 70}]
    runtime = FakeRuntime(detailed)
    # First read: no row (orphan). Re-check read: a row now references the repo.
    q = _queries(
        None,
        workloads_seq=[
            [],
            [_svc_row("orphan", image_repo="nerdit-app/orphan", image="nerdit-app/orphan:1")],
        ],
    )
    client = _app(runtime=runtime, queries=q, data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"]["removed"] == []
    assert body["images"]["skipped"] == [{"repo": "nerdit-app/orphan", "reason": "in_use_or_error"}]
    assert runtime.removed == []  # the re-check aborted before any removal


# --- Track 0.4: instance-scoped image reclaim ---------------------------------


def _img(repo_tag, instance, size=10):  # noqa: ANN001, ANN202
    return {"repo_tag": repo_tag, "id": repo_tag, "size_bytes": size, "instance": instance}


def test_orphan_repos_are_scoped_to_the_owning_instance():
    """Ownership is judged per repo over ALL its tags (pure predicate).

    Own-labelled ⇒ eligible; foreign-labelled ⇒ never; unlabelled (pre-fix)
    ⇒ skipped, mirroring the container sweep's treatment of unlabelled
    containers; a repo mixing our label with a foreign OR an unlabelled tag
    ⇒ refused wholesale (reclaim removes every unprotected tag of an eligible
    repo, so one own tag must never license removing an unlabelled sibling).
    """
    detailed = [
        _img("nerdit-app/mine:1", "alpha"),
        _img("nerdit-app/theirs:1", "beta"),
        _img("nerdit-app/legacy:1", None),
        _img("nerdit-app/mixed:1", "alpha"),
        _img("nerdit-app/mixed:2", "beta"),
        _img("nerdit-app/mine-legacy:1", None),
        _img("nerdit-app/mine-legacy:2", "alpha"),
        _img("ollama/ollama:latest", None),  # not a deploy repo → never in scope
    ]
    assert _orphan_app_repos(detailed, set(), set(), "alpha") == ["nerdit-app/mine"]
    # The other daemon sees the mirror image of the same host.
    assert _orphan_app_repos(detailed, set(), set(), "beta") == ["nerdit-app/theirs"]


def test_orphan_repos_ownership_counts_protected_tags_too():
    """A foreign tag blocks its repo even when that tag is live/protected.

    Ownership is decided BEFORE the orphan filter drops live and protected tags,
    so a repo whose only foreign tag is protected is still refused.
    """
    detailed = [_img("nerdit-app/shared:1", "beta"), _img("nerdit-app/shared:2", "alpha")]
    assert _orphan_app_repos(detailed, set(), {"nerdit-app/shared:1"}, "alpha") == []


def test_disk_orphan_images_hides_foreign_and_unlabelled(tmp_path):
    detailed = [
        _img("nerdit-app/mine:1", "default", size=70),
        _img("nerdit-app/theirs:1", "sibling", size=70),
        _img("nerdit-app/legacy:1", None, size=70),
    ]
    client = _app(
        runtime=FakeRuntime(detailed), queries=_queries(None, workloads=[]), data_dir=tmp_path
    )
    body = client.get("/api/system/disk", headers=_ADMIN).json()
    # All three repos are still attributed in the size report (nothing hidden
    # from accounting) — only the reclaim candidate list is instance-scoped.
    assert set(body["images"]["by_repo"]) == {
        "nerdit-app/mine",
        "nerdit-app/theirs",
        "nerdit-app/legacy",
    }
    assert body["orphan_images"] == ["nerdit-app/mine"]


def test_gc_never_removes_a_sibling_daemons_image(tmp_path):
    """The PR #81 regression class: a co-located daemon's images are untouchable.

    The sibling's rows live in the sibling's DB, so its repo looks orphan through
    ours — the instance label is the only thing standing between it and removal.
    """
    detailed = [_img("nerdit-app/theirs:1", "sibling", size=70)]
    runtime = FakeRuntime(detailed)
    client = _app(runtime=runtime, queries=_queries(None, workloads=[]), data_dir=tmp_path)

    plan = client.post("/api/system/gc?dry_run=true", headers=_ADMIN).json()
    assert plan["images"]["removed"] == []
    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"] == {"removed": [], "skipped": [], "reclaimed_bytes_estimate": 0}
    assert runtime.removed == []


def test_gc_leaves_unlabelled_pre_fix_images_alone(tmp_path):
    """Unlabelled images are NOT assumed to be ours (documented posture).

    Mirrors ``DockerRuntime.list_own_managed_containers``, which excludes
    unlabelled pre-upgrade containers from the sweep's kill set: on a destructive
    path, no evidence of ownership must never mean "mine".
    """
    detailed = [_img("nerdit-app/legacy:1", None, size=70)]
    runtime = FakeRuntime(detailed)
    client = _app(runtime=runtime, queries=_queries(None, workloads=[]), data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"]["removed"] == []
    assert body["images"]["skipped"] == []  # not a candidate at all, not a failure
    assert runtime.removed == []


def test_gc_refuses_a_repo_mixing_an_own_tag_with_an_unlabelled_one(tmp_path):
    """PR #105 Codex P1: one own tag never licenses removing an unlabelled one.

    Reclaim removes EVERY unprotected tag of an eligible repo, so a repo whose
    ``foo:1`` predates the label while ``foo:2`` is ours must be refused
    wholesale — ``foo:1`` may belong to a pre-upgrade co-located daemon. Dry run
    and real run agree (the plan is the same predicate).
    """
    detailed = [
        _img("nerdit-app/foo:1", None, size=70),
        _img("nerdit-app/foo:2", "default", size=70),
    ]
    runtime = FakeRuntime(detailed)
    client = _app(runtime=runtime, queries=_queries(None, workloads=[]), data_dir=tmp_path)

    empty = {"removed": [], "skipped": [], "reclaimed_bytes_estimate": 0}
    plan = client.post("/api/system/gc?dry_run=true", headers=_ADMIN).json()
    assert plan["images"] == empty
    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"] == empty
    assert runtime.removed == []


def test_gc_removes_own_images_under_a_custom_instance_id(tmp_path):
    detailed = [_img("nerdit-app/mine:1", "beta", size=70), _img("nerdit-app/theirs:1", "alpha")]
    runtime = FakeRuntime(detailed)
    client = _app(
        runtime=runtime,
        queries=_queries(None, workloads=[]),
        data_dir=tmp_path,
        instance_id="beta",
    )

    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["images"]["removed"] == ["nerdit-app/mine"]
    assert runtime.removed == ["nerdit-app/mine:1"]


def test_gc_toctou_data_skips_dir_that_regains_a_row(tmp_path):
    _seed_dir(tmp_path / "services" / "ghost")
    q = _queries(None, workloads_seq=[[], [_svc_row("ghost")]])
    client = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN, json={"include_orphan_data": True}).json()
    assert body["orphan_data"]["removed"] == []
    assert body["orphan_data"]["skipped"] == [{"name": "ghost", "reason": "now_referenced"}]
    # The data dir survived — no irreversible loss.
    assert (tmp_path / "services" / "ghost").is_dir()


def test_gc_orphan_data_foreign_dir_isolated(tmp_path):
    _seed_dir(tmp_path / "services" / "good")
    _seed_dir(tmp_path / "services" / "Bad_Name")  # not a DNS label → VolumeSpecError
    q = _queries(None, workloads=[])
    client = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN, json={"include_orphan_data": True}).json()
    assert body["orphan_data"]["removed"] == ["good"]
    assert body["orphan_data"]["skipped"] == [{"name": "Bad_Name", "reason": "VolumeSpecError"}]
    # gc completed (200) despite the foreign entry; the good dir was reclaimed.
    assert not (tmp_path / "services" / "good").exists()
    assert (tmp_path / "services" / "Bad_Name").is_dir()


def test_gc_orphan_data_never_reclaims_live_row_tombstone(tmp_path):
    """A crash-window ``.trash-<name>-<nonce>`` whose base row is still live must
    never be GC-reclaimed — that is the exact data loss the C2 startup-restore
    sweep exists to prevent (WP16 PR2 — moved alongside the predicate family).
    """
    _seed_dir(tmp_path / "services" / ".trash-alpha-deadbeef")
    _seed_dir(tmp_path / "services" / ".trash-ghost-cafebabe")
    _seed_dir(tmp_path / "services" / "gone")
    q = _queries(None, workloads=[_svc_row("alpha")])
    client = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)

    # dry_run: the plan must never list the live row's tombstone.
    plan = client.post(
        "/api/system/gc?dry_run=true", headers=_ADMIN, json={"include_orphan_data": True}
    ).json()
    assert ".trash-alpha-deadbeef" not in plan["orphan_data"]["removed"]
    assert sorted(plan["orphan_data"]["removed"]) == [".trash-ghost-cafebabe", "gone"]

    # real run: the live row's tombstone is untouched; the two true orphans go.
    body = client.post("/api/system/gc", headers=_ADMIN, json={"include_orphan_data": True}).json()
    assert ".trash-alpha-deadbeef" not in body["orphan_data"]["removed"]
    assert ".trash-alpha-deadbeef" not in [s["name"] for s in body["orphan_data"]["skipped"]]
    assert sorted(body["orphan_data"]["removed"]) == [".trash-ghost-cafebabe", "gone"]
    assert (tmp_path / "services" / ".trash-alpha-deadbeef").is_dir()
    assert not (tmp_path / "services" / ".trash-ghost-cafebabe").exists()
    assert not (tmp_path / "services" / "gone").exists()


def test_gc_orphan_data_off_by_default(tmp_path):
    _seed_dir(tmp_path / "services" / "ghost")
    q = _queries(None, workloads=[])
    client = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)

    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["orphan_data"] == {"enabled": False, "removed": [], "skipped": []}
    assert (tmp_path / "services" / "ghost").is_dir()  # never touched


def test_gc_dry_run_enumerates_orphan_data(tmp_path):
    _seed_dir(tmp_path / "services" / "ghost")
    q = _queries(None, workloads=[])
    client = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)

    body = client.post(
        "/api/system/gc?dry_run=true", headers=_ADMIN, json={"include_orphan_data": True}
    ).json()
    assert body["orphan_data"]["removed"] == ["ghost"]
    assert (tmp_path / "services" / "ghost").is_dir()  # dry run wrote nothing


def test_gc_reports_backups_over_keep(tmp_path):
    bdir = tmp_path / "backups"
    bdir.mkdir(parents=True)
    for i in range(4):
        (bdir / f"nerdit-backup-{i}.tar.gz").write_bytes(b"z")
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path, keep=2)
    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["reports"]["backups_over_keep"] == 2
    assert body["reports"]["build_cache_bytes"] == 30


def test_gc_reports_no_warnings_on_fast_walks(tmp_path):
    _seed_dir(tmp_path / "models" / "ollama", size=7)
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)
    body = client.post("/api/system/gc", headers=_ADMIN).json()
    assert body["warnings"] == []
    assert body["reports"]["weights"] == {"ollama": 7, "huggingface": 0}


def test_gc_report_walks_overrun_yields_null_and_warning(tmp_path, monkeypatch):
    """The GC report walks must be bounded + detached, like /system/disk's (F8).

    Pre-F8 they ran on the loop's DEFAULT executor via ``asyncio.to_thread``: a
    slow weights walk blocked the gc for its whole duration, and an abandoned one
    is joined by ``asyncio.run`` teardown (unboundedly under uvloop), wedging a
    /daemon/restart that lands mid-gc. Now: budget overrun ⇒ the gc still
    completes, the sizes come back null with a ``scan_timeout`` warning, and the
    overrunning walker is a detached daemon thread.
    """
    release = threading.Event()

    def _blocking_du(_path):  # noqa: ANN001, ANN202
        release.wait(timeout=30)
        return 1

    monkeypatch.setattr(system_routes, "_DISK_SCAN_BUDGET_S", 0.2)
    monkeypatch.setattr(system_routes, "du_bytes", _blocking_du)
    client = _app(runtime=FakeRuntime([]), queries=_queries(None), data_dir=tmp_path)

    try:
        resp = client.post("/api/system/gc", headers=_ADMIN)
        assert resp.status_code == 200  # the gc completed; it was never blocked
        body = resp.json()
        assert body["reports"]["weights"] == {"ollama": None, "huggingface": None}
        assert "scan_timeout" in body["warnings"]
        walkers = [t for t in threading.enumerate() if t.name == "nerdit-du-walk"]
        assert walkers, "the overrunning walk should still be alive (detached)"
        assert all(t.daemon for t in walkers), "walk threads must be daemon threads"
    finally:
        release.set()  # let the walkers finish so they never leak past the test


# --- _bounded_walks detachment (the /daemon/restart wedge regression) ---------


async def test_bounded_walks_overrun_detaches_daemon_thread():
    """An overrunning walk must be abandoned on a DAEMON thread, never the
    default executor: asyncio.run teardown joins the default executor, which
    under uvloop blocks unboundedly and wedges the /daemon/restart re-exec
    (live-run leg 8). A daemon thread is joined by nothing."""
    import threading

    from nerdit.daemon.routes.system import _bounded_walks

    release = threading.Event()

    def slow_walk():
        release.wait(timeout=30)
        return {"bytes": 1}

    def fast_walk():
        return {"bytes": 2}

    results, timed_out = await _bounded_walks({"slow": slow_walk, "fast": fast_walk}, budget=0.2)
    try:
        assert timed_out is True
        assert results["slow"] is None
        assert results["fast"] == {"bytes": 2}
        walkers = [t for t in threading.enumerate() if t.name == "nerdit-du-walk"]
        assert walkers, "overrunning walk thread should still be alive (detached)"
        assert all(t.daemon for t in walkers), "walk threads must be daemon threads"
    finally:
        release.set()  # let the walker finish so it never leaks past the test


async def test_bounded_walks_walk_exception_is_contained():
    from nerdit.daemon.routes.system import _bounded_walks

    def boom():
        raise OSError("disk on fire")

    results, timed_out = await _bounded_walks({"bad": boom}, budget=5.0)
    assert results["bad"] is None
    assert timed_out is True
