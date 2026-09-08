"""Shared image and orphan-data predicates for `/system/disk` and `/system/gc`.

Keep this module independent of routes to avoid import cycles.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nerdit.config.defaults import APP_IMAGE_REPO_PREFIX
from nerdit.core.volumes import VolumeSpecError, service_data_root, tombstone_service_name

# A repo whose row is mid-deploy must never have its images reclaimed — the tags
# are about to be (re)built onto it. Mirrors `config['last_deploy']['phase']`.
_IN_PROGRESS_PHASES = frozenset({"queued", "building", "launching"})

#: Fallback ownership scope, identical to `[daemon].instance_id`'s own default
#: — a host that never configured an instance id is the single-daemon case.
DEFAULT_INSTANCE_ID = "default"


def _repo_of(repo_tag: str) -> str:
    """Return the repo portion of a `repo:tag` string (`""` when empty)."""
    if not repo_tag or ":" not in repo_tag:
        return repo_tag or ""
    return repo_tag.rsplit(":", 1)[0]


def _protected_image_refs(rows: list[dict[str, Any]], image_tags: Iterable[str]) -> set[str]:
    """Protect live workloads' current/rollback tags and every tag in their image_repo.

    A repository referenced by any live row is wholly protected, including older
    versions and model images.
    """
    protected: set[str] = set()
    repos: set[str] = set()
    for row in rows:
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            continue
        for key in ("image", "previous_image"):
            tag = cfg.get(key)
            if isinstance(tag, str) and tag:
                protected.add(tag)
        repo = cfg.get("image_repo")
        if isinstance(repo, str) and repo:
            repos.add(repo)
    for repo_tag in image_tags:
        repo = _repo_of(repo_tag)
        if repo and repo in repos:
            protected.add(repo_tag)
    return protected


def _live_repos(rows: list[dict[str, Any]]) -> set[str]:
    """Every `config['image_repo']` referenced by a live service/model row."""
    repos: set[str] = set()
    for row in rows:
        cfg = row.get("config")
        if isinstance(cfg, dict):
            repo = cfg.get("image_repo")
            if isinstance(repo, str) and repo:
                repos.add(repo)
    return repos


def _live_service_names(rows: list[dict[str, Any]]) -> set[str]:
    """Every non-null `service_name` across the live service/model rows."""
    return {name for row in rows if isinstance((name := row.get("service_name")), str) and name}


def _orphan_data_dir_names(dir_names: list[str], live_names: set[str]) -> list[str]:
    """Names under `<data_dir>/services` with no live row (reclaim candidates).

    A plain service dir is orphan when its own name has no live row. A C2
    tombstone (`.trash-<name>-<nonce>`) is orphan only when its BASE service
    has no live row — a tombstone whose row still exists is a crash-window
    artifact the startup sweep will rename back, and must never be GC-reclaimed
    (that would be the very data loss C2 closes). An unparseable dotdir keeps the
    plain rule (its literal name is never a live service name ⇒ orphan).
    """
    orphans: list[str] = []
    for name in dir_names:
        base = tombstone_service_name(name)
        key = base if base is not None else name
        if key not in live_names:
            orphans.append(name)
    return sorted(orphans)


def _orphan_data_dir_path(data_dir: Path, name: str) -> Path:
    """Resolve a reclaimable orphan dir name to its host path (tombstone-aware).

    A plain service name goes through `service_data_root` (full grammar +
    symlink re-validation). A `.trash-<base>-<nonce>` tombstone cannot (its
    dotted name fails the DNS-label grammar), so it is resolved as a direct child
    of `<data_dir>/services` with the same symlink reject + `is_relative_to`
    assertion, fail-closed on the first violation.
    """
    if tombstone_service_name(name) is None:
        return service_data_root(data_dir, name)
    services_root = data_dir / "services"
    if services_root.is_symlink():
        raise VolumeSpecError("services dir is a symlink; refusing to resolve orphan")
    if (services_root / name).is_symlink():
        raise VolumeSpecError(f"orphan tombstone is a symlink: {name!r}")
    base = services_root.resolve()
    root = (base / name).resolve()
    if not root.is_relative_to(base):
        raise VolumeSpecError(f"orphan tombstone escapes services root: {name!r}")
    return root


def _entry_instance(entry: dict) -> str | None:
    """The owning `nerdit-instance` of a `list_images_detailed` entry.

    `None` when the key is missing, null or empty — an image with no recorded
    owner (a non-nerdit image, or a `nerdit-app/*` image built before the
    label shipped).
    """
    value = entry.get("instance")
    return value if isinstance(value, str) and value else None


def _repo_is_ours(instances: set[str | None], instance_id: str) -> bool:
    """Return true only for a nonempty repo whose every tag has this instance label.

    Any foreign or unlabelled tag blocks the entire repo: missing ownership evidence
    is never permission to delete. Mixed or pre-upgrade repos remain visible in disk
    inventory but require explicit manual removal rather than automatic GC.
    """
    return instance_id in instances and all(inst == instance_id for inst in instances)


def _orphan_app_repos(
    detailed: list[dict], live_repos: set[str], protected: set[str], instance_id: str
) -> list[str]:
    """`nerdit-app/*` repos present in images with no live row (reclaim scope).

    A repo is orphan when it carries the deploy prefix, no live workload row
    references it, at least one of its tags is not in *protected* (a repo all of
    whose tags are protected has nothing reclaimable), **and** it is provably
    owned by the daemon whose `[daemon].instance_id` is *instance_id* (see
    `_repo_is_ours` — ownership is judged over ALL of the repo's tags, the
    live and protected ones included). Report-only for the disk route; the
    reclaim candidate list for gc.
    """
    prefix = f"{APP_IMAGE_REPO_PREFIX}/"
    owners: dict[str, set[str | None]] = {}
    orphan: set[str] = set()
    for entry in detailed:
        repo_tag = entry.get("repo_tag") or ""
        repo = _repo_of(repo_tag)
        if not repo.startswith(prefix):
            continue
        owners.setdefault(repo, set()).add(_entry_instance(entry))
        if repo in live_repos or repo_tag in protected:
            continue
        orphan.add(repo)
    return sorted(repo for repo in orphan if _repo_is_ours(owners[repo], instance_id))


def _repo_tags(detailed: list[dict], repo: str, protected: set[str]) -> list[str]:
    """The reclaimable tags of *repo* (prefix `<repo>:`, minus *protected*)."""
    prefix = f"{repo}:"
    return [
        rt
        for entry in detailed
        if (rt := entry.get("repo_tag") or "").startswith(prefix) and rt not in protected
    ]


def _repo_size_estimate(detailed: list[dict], tags: Iterable[str]) -> int:
    """Sum the on-disk size of *tags*, deduped by image id (shared layers).

    An estimate only: `list_images_detailed` reports each image's full `Size`
    per tag, so tags sharing layers are counted once (by id) but layers shared
    with a surviving image are still attributed here.
    """
    tagset = set(tags)
    seen: set[str] = set()
    total = 0
    for entry in detailed:
        rt = entry.get("repo_tag") or ""
        if rt not in tagset:
            continue
        img_id = entry.get("id") or rt
        if img_id not in seen:
            seen.add(img_id)
            total += int(entry.get("size_bytes") or 0)
    return total


def _attribute_images(detailed: list[dict]) -> tuple[dict[str, int], int]:
    """Per-repo size attribution + a de-duplicated grand total.

    Deduped by image id: within a repo an id is summed once, and the grand total
    counts each id once globally, so tags sharing an image never double-count.
    Attribution only — the docker `df` aggregate is the authoritative total.
    """
    by_repo: dict[str, int] = {}
    seen_per_repo: dict[str, set[str]] = {}
    total_ids: set[str] = set()
    total = 0
    for entry in detailed:
        repo_tag = entry.get("repo_tag") or ""
        repo = _repo_of(repo_tag)
        if not repo:
            continue
        img_id = entry.get("id") or repo_tag
        size = int(entry.get("size_bytes") or 0)
        ids = seen_per_repo.setdefault(repo, set())
        if img_id not in ids:
            ids.add(img_id)
            by_repo[repo] = by_repo.get(repo, 0) + size
        if img_id not in total_ids:
            total_ids.add(img_id)
            total += size
    return by_repo, total


def _repo_in_progress(rows: list[dict[str, Any]], repo: str) -> bool:
    """True iff a fresh row for *repo* is mid-deploy (must not have images pruned)."""
    for row in rows:
        cfg = row.get("config")
        if not isinstance(cfg, dict) or cfg.get("image_repo") != repo:
            continue
        last_deploy = cfg.get("last_deploy")
        if isinstance(last_deploy, dict) and last_deploy.get("phase") in _IN_PROGRESS_PHASES:
            return True
    return False
