"""Write guarded deploy-generation transitions and durable settlement events.

`stamp_last_deploy` rereads config before updating `last_deploy` and emits its
terminal event. `record_deploy_settled` serves callers that write terminal
phases themselves. Neither helper imports a controller.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from nerdit.core import eventlog
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.remediation_settle import settle_remediation_code

if TYPE_CHECKING:
    from nerdit.db.queries import Queries


async def stamp_last_deploy(
    queries: Queries,
    job_id: str,
    *,
    only_from: tuple[str, ...] | None = None,
    require_version_match: bool = False,
    require_ready_flag: str | None = None,
    expect_version: int | None = None,
    **fields: object,
) -> bool:
    """Advance a deploy phase using freshly read config and explicit transition guards.

    Rows without `last_deploy` are unchanged. Refresh `updated_at` only on a write.

    Args:
        only_from: Allowed current phases, if constrained.
        require_version_match: Require build and deploy versions to agree.
        require_ready_flag: Config flag that must be truthy, such as model_pulled
            or db_ready; None skips the gate.
        expect_version: Generation to settle, protecting newer redeploys.

    Returns:
        True only when written, allowing callers to emit edge-triggered events.
    """
    row = await queries.get_job(job_id)
    if row is None:
        return False
    cfg = parse_job_config(row)
    ld = cfg.get("last_deploy")
    if not isinstance(ld, dict):
        return False
    if only_from is not None and ld.get("phase") not in only_from:
        return False
    if require_version_match and cfg.get("build_version") != ld.get("version"):
        return False
    if require_ready_flag is not None and not cfg.get(require_ready_flag):
        return False
    if expect_version is not None and ld.get("version") != expect_version:
        return False
    new_ld = dict(ld)
    new_ld.update(fields)
    new_ld["updated_at"] = datetime.now(UTC).isoformat()
    # Settle-time remediation: the crash forensics are already on
    # the fresh blob (`_persist_forensics` runs before the crash-loop stamp),
    # so the generation that settles `failed` carries the rule id — the code
    # only, never the detail text or the log tail — for the view and the
    # `service.deploy_failed` event. Stamped once, on the failed transition;
    # a later re-stamp of an already-failed generation leaves it alone.
    #
    # **service rows only.** The settle classifier mirrors just
    # rules 1b/4b/5 of the LOCKED table; the daemon's Rule 1 (fresh `gpu_oom`
    # on a `kind=model` row → `model.gpu_oom`) OUTRANKS 1b there, so a vLLM
    # model dying of CUDA OOM (`gpu_oom` AND `oom_killed`) would otherwise be
    # stamped `raise_memory_limit` here while `/diagnose` returns
    # `model.gpu_oom` — two contradictory codes, the event one being what the
    # cloud reads. Models are diagnosed via `/diagnose` and the cloud status
    # story (D-GH-5) is git-sourced *service* deploys, so `null` on model
    # (and database) rows is correct and simplest — no need to mirror Rule 1.
    if (
        new_ld.get("phase") == "failed"
        and ld.get("phase") != "failed"
        and getattr(row.kind, "value", row.kind) == "service"
    ):
        new_ld["remediation_code"] = settle_remediation_code(cfg, new_ld)
    if not await queries.patch_job_config(job_id, {"last_deploy": new_ld}):
        return False

    # Durable settle event — STRICTLY after the real write, past
    # every no-op guard. Only a `service` row that settled to a terminal
    # deploy phase counts: `kind=model`/`database` readiness also funnels
    # through here (require_ready_flag) and must NOT be reported as a deploy.
    # Emits only on the FIRST terminal transition of a generation: a later
    # `healthy` → `failed` re-stamp (restarts exhausted after a successful
    # deploy) is a runtime crash, not a second deploy outcome, and must not
    # double-count the generation. NOT a kind dispatch: model and
    # database readiness funnels through here too, but their own controllers
    # already emit `model.ready`/`database.ready` — and rows created by
    # POST /models / POST /databases carry no `last_deploy`, so this tail
    # never runs for them anyway.
    if getattr(row.kind, "value", row.kind) == "service" and ld.get("phase") not in (
        "healthy",
        "failed",
    ):
        await record_deploy_settled(getattr(row, "service_name", None), new_ld)
    return True


async def record_deploy_settled(service_name: str | None, new_ld: dict) -> None:
    """Emit the first terminal deploy event, including phase-writer bypass branches.

    Use machine-shaped reason/version/error-class values only, never free-text
    errors or container output. Include settled repo/ref/SHA provenance and optional
    remediation code so consumers need not infer it from mutable service state.
    """
    phase = new_ld.get("phase")
    if phase not in ("healthy", "failed"):
        return
    recorder = eventlog.get_recorder()
    if recorder is None:
        return
    error_class = new_ld.get("error_class")
    version = new_ld.get("version")
    reason = new_ld.get("reason")
    repo, ref, sha, remediation_code = (
        value if isinstance(value, str) else None
        for value in (
            new_ld.get("repo"),
            new_ld.get("ref"),
            new_ld.get("sha"),
            new_ld.get("remediation_code"),
        )
    )
    await recorder.record(
        "service.deploy_succeeded" if phase == "healthy" else "service.deploy_failed",
        kind="service",
        service_name=service_name,
        reason=reason if isinstance(reason, str) else None,
        build_version=version if isinstance(version, int) else None,
        data={
            "repo": repo,
            "ref": ref,
            "sha": sha,
            "remediation_code": remediation_code,
            "error_class": error_class if isinstance(error_class, str) else None,
        },
    )
