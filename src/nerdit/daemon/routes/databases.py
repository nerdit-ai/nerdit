"""Create and read managed databases; lifecycle actions use /services.

Creation is authorized, idempotent and audited. The controller pulls backend
images asynchronously while rows remain building, then reconciles them.

The daemon mints passwords as scoped secrets under backend-native env names.
It injects them at database/app launch but never returns them in API responses,
audits, logs, diagnosis or dry-run diffs. Responses expose key names only.
The backend entrypoint initializes the database without a daemon SQL driver.

This surface also creates and lists dumps and restores databases. Lifecycle
writes remain under /services. Mounted under /api only.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import logging
import os
import secrets
import shutil
import stat
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Query, Request

from nerdit.core.backup import (
    DUMP_TAR_GLOB,
    DUMP_TAR_RE,
    BackupError,
    DumpBackupResult,
    DumpManifestError,
    DumpOutputError,
    _rmtree_staging,
    create_dump_backup,
    extract_dump_tar,
)
from nerdit.core.data.backend import DataBackend
from nerdit.core.eventlog import get_recorder
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.core.services import DumpError, DumpResult, RunPreconditionError
from nerdit.core.volumes import (
    VolumeSpecError,
    create_dump_staging_dir,
    dump_staging_root,
    service_data_root,
)
from nerdit.core.workspaces import settled_to_thread
from nerdit.daemon.audit import audit_params, record_out_of_band
from nerdit.daemon.auth import require_owner_or_admin, require_role, require_service_scope
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes._resources import (
    authorize_create,
    new_workload_row,
    reject_reserved_name,
    reserve_or_conflict,
    resolve_backend_or_422,
)
from nerdit.daemon.schemas.databases import (
    DatabaseDumpItem,
    DatabaseDumpListResponse,
    DatabaseDumpRequest,
    DatabaseDumpResponse,
    DatabaseRestoreRequest,
    DatabaseRestoreResponse,
)
from nerdit.daemon.service_purge import _dep_entry, _iter_db_dependents
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.models import (
    DatabaseCreateRequest,
    DatabaseListPage,
    DatabaseResponse,
    Job,
    JobKind,
    JobStatus,
    TokenRole,
)
from nerdit.db.queries._base import mark_request_side_effect
from nerdit.utils.disk import du_bytes
from nerdit.utils.ids import generate_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_idempotency_key(request: Request, what: str) -> None:
    """In-route ``Idempotency-Key`` gate (the ``routes/share.py`` wording).

    A local copy rather than an import: ``routes.share`` is a sibling route
    module, and a route importing another route module is the edge the
    import-cycle scanner exists to prevent. Four lines is the cheaper coupling.
    """
    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            f"A {what} requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )


# --- Helpers -----------------------------------------------------------------


async def _database_response(request: Request, job: Job) -> DatabaseResponse:
    """Map a `kind=database` Job to the dedicated database-shaped projection.

    `endpoint` is the backend's **password-free** `host:port` display string
    (D7) — never a DSN. The host shown is the bridge-advertise host so it matches
    what a bound app is wired to (minus the credential).
    """
    queries = request.app.state.queries
    controller = getattr(request.app.state, "data_controller", None)
    cfg = parse_job_config(job)
    name = job.service_name or job.name or job.id

    endpoint_str: str | None = None
    endpoint = await queries.get_service_endpoint(name)
    if endpoint is not None and controller is not None:
        backend = controller.backend_for(cfg)
        endpoint_str = backend.public_endpoint(controller.bridge_host, endpoint.host_port)

    return DatabaseResponse(
        id=job.id,
        name=name,
        backend=cfg.get("backend"),
        status=job.status,
        desired_state=job.desired_state,
        db_ready=bool(cfg.get("db_ready", False)),
        endpoint=endpoint_str,
        created_at=job.created_at,
    )


async def _audit_credential_minted(request: Request, service_name: str, keys: list[str]) -> None:
    """Record `database.credential_minted` (key NAMES only — never a value).

    A second row on top of the middleware's `database.create` row. Thin
    wrapper over `nerdit.daemon.audit.record_out_of_band`.
    """
    await record_out_of_band(
        request,
        action="database.credential_minted",
        target_type="database",
        target_id=service_name,
        params={"service": service_name, "keys": keys},
    )


# --- Write (desired-state; controller converges) ------------------------------


@router.post(
    "/databases", response_model=DatabaseResponse, status_code=201, operation_id="create_database"
)
async def create_database(request: Request, body: DatabaseCreateRequest) -> DatabaseResponse:
    """Provision a managed database: register a `kind=database` desired-state workload.

    Writes a stable row with `desired_state='running'` and status `building`;
    the controller pulls the backend image off-tick, allocates the port and
    launches the server on a later tick. The image is deliberately **not**
    required to exist locally (unlike `POST /services`). Authorized to
    `submitter`/`admin`, idempotent, audited (`database.create`).
    """
    authorize_create(request, body)
    queries = request.app.state.queries
    settings = request.app.state.settings
    controller = getattr(request.app.state, "data_controller", None)
    secret_manager = getattr(request.app.state, "secret_manager", None)
    if controller is None or secret_manager is None:
        raise NerditError(
            503,
            "db.unavailable",
            "The managed-data plane is not available on this daemon.",
            hint="Ensure the daemon is configured with a data backend.",
        )

    # Resolve the data backend. An explicit unknown name is a 422, not a
    # silent default (mirror of the model write path). The backend supplies the
    # name prefix, image, container port and health path so the row is
    # backend-shaped from creation.
    backend = resolve_backend_or_422(
        controller.get_backend,
        body.backend,
        code="db.unknown_backend",
        message=f"Unknown data backend '{body.backend}'.",
        hint=f"Supported backends: {', '.join(controller.backends)}.",
    )

    service_name = body.name or backend.name_prefix
    reject_reserved_name(service_name)
    # (P25 D-P25-3 leg b) Checked on the DERIVED row name (the backend prefix
    # when `--name` is omitted), before the row write and the credential mint.
    require_service_scope(request, service_name)

    config = {
        "backend": backend.name,
        "image": backend.image,
        "port": backend.container_port,
        # PGDATA rides the shipped P14a named-volume machinery — the create
        # route stamps the backend's spec so the reconcile launch materializes
        # the leaf dir under <data_dir>/services/<name>/ (§1.3).
        "volumes": [backend.volume_spec],
    }

    job = new_workload_row(
        request,
        kind=JobKind.database,
        service_name=service_name,
        gpu_count=0,  # databases take no GPU parameter at all (decided point (a))
        # TCP probe — the server answers no HTTP; the start period covers
        # initdb/WAL recovery.
        health_check={"type": "tcp", "start_period_s": settings.databases.start_period_s},
        config=config,
    )

    # The row-write is the ownership-authorization point (the P11.5
    # template-secrets precedent): mint the credential only AFTER it succeeds so
    # a name clash never leaves an orphaned secret.
    job = await reserve_or_conflict(
        queries,
        job,
        service_name=service_name,
        name_taken_hint="Check `nerdit db list`, or pass a different --name.",
    )

    # Mint the credential (the security-review core, D-B).
    #
    # This value is minted server-side and stored write-only under the backend's
    # image-native env name so the ordinary per-service secret scope delivers it
    # at launch (decided point (b)). It is NEVER returned by any response body,
    # audit row, log line, diagnose payload, or dry-run diff — names only. If
    # this action ever begins riding a value on a response body, it MUST be added
    # to NO_BODY_CACHE_ACTIONS (daemon/idempotency.py) so idempotent replays
    # never re-serve the plaintext credential (the LOCAL_MODEL_API_KEY trap,
    # core/models/binding.py, retargeted).
    password = secrets.token_hex(32)
    try:
        # SecretManager.set takes the rotation RLock, which a rotate_key worker
        # thread can hold for a full re-encrypt; run it off the event loop so an
        # in-flight rotation cannot stall the whole daemon (§2 WP2).
        await asyncio.to_thread(
            secret_manager.set, service_name, {backend.minted_secret_key: password}
        )
    except Exception as exc:  # noqa: BLE001 — surface as an actionable structured error
        # The row already exists but has no usable credential: it will never
        # reach db_ready. No compensating row rollback (the P11.5 posture) — the
        # operator sets the secret and restarts, or deletes and recreates.
        logger.warning("Failed to mint credential for database %s: %s", service_name, exc)
        raise NerditError(
            500,
            "db.credential_mint_failed",
            "The database was created but its credential could not be stored.",
            hint=(
                f"Set it manually (nerdit secrets set {service_name} "
                f"{backend.minted_secret_key}=...) and restart the database, "
                "or delete and recreate it."
            ),
        ) from exc
    await _audit_credential_minted(request, service_name, [backend.minted_secret_key])

    return await _database_response(request, job)


# --- Bounded read --------------------------------------------------------------


@router.get("/databases", response_model=DatabaseListPage, operation_id="list_databases")
async def list_databases(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
) -> DatabaseListPage:
    """Cursor-paginated, database-shaped list of `kind=database` rows (newest first).

    Read is intentionally open to any authenticated principal (readonly and up)
    — it carries no credential (`endpoint` is password-free by construction).
    """
    queries = request.app.state.queries
    try:
        jobs, next_cursor = await queries.list_services(
            cursor=cursor, limit=limit, kinds=(JobKind.database,)
        )
    except ValueError as exc:
        raise NerditError(400, "bad_request", str(exc)) from exc
    items = [await _database_response(request, job) for job in jobs]
    return DatabaseListPage(items=items, next_cursor=next_cursor)


# --- (P37) Managed-database dumps ---------------------------------------------
#
# Three operations nested under ``/databases`` (D-P37-8): the controller runs
# the sibling, ``core.backup`` owns the tar; this module owns the D-P37-10
# precondition ladder, the §1.4 error registry, audit, events and custody
# (metadata out, never the tar, never a path). Dump/list are owner-or-admin +
# scope; restore is admin-only. No ``_backup_lock``: the per-row slot and
# ``[services].max_concurrent_dumps`` bound dumps instead (D-P37-9).

#: Headroom over the volume's size in the pre-flight estimate (D-P37-10) — a
#: heuristic; the post-tool re-check uses the size actually produced.
_DUMP_DISK_MARGIN_BYTES = 64 * 1024 * 1024


def _dump_data_dir(request: Request) -> Path:
    """The daemon's data dir — where ``backups/`` and ``dump-staging/`` live."""
    return Path(request.app.state.settings.data_dir).expanduser()


def _dump_not_a_database(name: str, kind: JobKind) -> NerditError:
    """422 for a dump/restore aimed at a service or model row. Deliberately distinct
    from the frozen ``run.not_supported`` pair in ``routes/service_run.py`` (P20).
    """
    return NerditError(
        422,
        "dump.not_a_database",
        f"'{name}' is a {kind.value}, not a managed database.",
        hint="Dumps are managed-database only; back a service's volumes up with "
        "`nerdit backup --volume` instead.",
    )


def _dump_preflight(request: Request, name: str, job: Job, *, timeout_s: int) -> DataBackend:
    """Refuse every dump/restore decidable from row + daemon state alone (D-P37-10).

    Locked order, before any slot or container: (1) ``kind == database`` (422
    ``dump.not_a_database``); (2) data plane present (503 ``db.unavailable``);
    (3) run subsystem present and not draining (409
    ``daemon.restart_in_progress``); (4) row ``running`` AND ``config['db_ready']``
    (409 ``dump.database_not_ready`` — status alone can dial a userland-proxy port
    whose listener is gone); (5) ``timeout_s`` within
    ``[services].dump_timeout_max_s`` (422 ``dump.timeout_too_large``, never
    clamped). The disk pre-flight runs afterwards, on a worker thread. Returns the
    row's :class:`DataBackend`.
    """
    if job.kind is not JobKind.database:
        raise _dump_not_a_database(name, job.kind)

    data_controller = getattr(request.app.state, "data_controller", None)
    if data_controller is None:
        raise NerditError(
            503,
            "db.unavailable",
            "The managed-data plane is not available on this daemon.",
            hint="Ensure the daemon is configured with a data backend.",
        )
    controller = getattr(request.app.state, "service_controller", None)
    if controller is None:
        # Not reachable through the real app (``server.py`` always attaches the
        # controller); a 503 is the honest answer if it ever is, since the
        # subsystem that would execute the sibling does not exist.
        raise NerditError(
            503,
            "db.unavailable",
            "The dump subsystem is unavailable on this daemon.",
            hint="Check `nerdit doctor` and the daemon log.",
        )
    if controller.draining:
        raise NerditError(
            409,
            "daemon.restart_in_progress",
            "The daemon is draining for a restart; no new dumps are accepted.",
            hint="Wait for the daemon to come back up, then retry.",
        )

    cfg = parse_job_config(job)
    if job.status is not JobStatus.running or not cfg.get("db_ready"):
        raise NerditError(
            409,
            "dump.database_not_ready",
            f"Database '{name}' is not ready to be dumped.",
            hint="Wait for it to report db_ready (`nerdit db list`), or start it first.",
        )

    max_timeout = request.app.state.settings.services.dump_timeout_max_s
    if timeout_s > max_timeout:
        raise NerditError(
            422,
            "dump.timeout_too_large",
            f"timeout_s={timeout_s} exceeds the server cap of {max_timeout}s.",
            hint=f"Lower timeout_s, or raise [services].dump_timeout_max_s "
            f"(currently {max_timeout}) on the daemon.",
        )
    return data_controller.backend_for(cfg)


async def _require_endpoint(request: Request, name: str) -> None:
    """Refuse a dump of a row with no published port — the awaiting half of check 4,
    sharing ``dump.database_not_ready`` because "cannot be dialled" is one fact to
    the caller.
    """
    endpoint = await request.app.state.queries.get_service_endpoint(name)
    if endpoint is None or not endpoint.live_port:
        raise NerditError(
            409,
            "dump.database_not_ready",
            f"Database '{name}' has no published port to dump from.",
            hint="Wait for it to report db_ready (`nerdit db list`), or start it first.",
        )


def _free_bytes_for(root: Path) -> int:
    """Free bytes on the filesystem that will hold *root*, creating nothing: climbs
    to the nearest existing ancestor (same filesystem in every daemon layout).
    """
    probe = root
    while not probe.exists():
        parent = probe.parent
        if parent == probe:  # reached the filesystem root; nothing left to climb
            break
        probe = parent
    return shutil.disk_usage(probe).free


async def _disk_preflight(data_dir: Path, service_name: str) -> None:
    """Refuse a dump the daemon probably cannot fit (409 ``dump.insufficient_disk``).

    Both halves run on a worker thread (D-P37-10): ``du_bytes`` is a full walk of
    the volume. An estimate, stated as one in the hint; the exact number is
    re-checked after the tool exits, before packing.
    """

    def _probe() -> tuple[int, int]:
        volume = service_data_root(data_dir, service_name)
        return du_bytes(volume) + _DUMP_DISK_MARGIN_BYTES, _free_bytes_for(
            dump_staging_root(data_dir)
        )

    required, free = await asyncio.to_thread(_probe)
    if free < required:
        raise NerditError(
            409,
            "dump.insufficient_disk",
            "There is not enough free disk to stage this dump.",
            hint="The requirement is an estimate from the volume size plus a fixed "
            "margin; free space under the daemon's data directory, or prune old "
            "dumps (`nerdit gc`, [retention].dump_keep_last).",
            detail={"required_bytes": required, "free_bytes": free},
        )


async def _recheck_disk(data_dir: Path, output_path: str) -> None:
    """Second, cheap disk check against the size the tool actually wrote.

    The pack phase briefly holds the dump AND the tar, so "the dump fitted" does
    not imply "the tar will". This is the only estimate-free number in the
    feature, which is why it is worth a second ``stat`` — and it is the same
    409 the pre-flight raises, because the caller's remedy is identical.
    """

    def _probe() -> tuple[int, int]:
        try:
            size = os.stat(output_path).st_size
        except OSError:
            # The packer's own ``lstat``/NOFOLLOW checks own this failure and
            # give it a machine reason; refusing here on a stat error would
            # answer "no disk" for "no file".
            return 0, 0
        return size + _DUMP_DISK_MARGIN_BYTES, _free_bytes_for(dump_staging_root(data_dir))

    required, free = await asyncio.to_thread(_probe)
    if required and free < required:
        raise NerditError(
            409,
            "dump.insufficient_disk",
            "There is not enough free disk to pack this dump.",
            hint="The dump itself was captured but the tar does not fit; free space "
            "under the daemon's data directory and retry.",
            detail={"required_bytes": required, "free_bytes": free},
        )


def _dump_precondition_error(name: str, exc: RunPreconditionError, *, cap: int) -> NerditError:
    """Map a controller-side :class:`RunPreconditionError` to its §1.4 envelope.
    ``run_in_progress`` covers a run, a release or another dump on the row (one
    slot namespace, D-P37-9); an unknown reason falls back to the same 409.
    """
    if exc.reason == "too_many_dumps":
        return NerditError(
            409,
            "dump.too_many_in_flight",
            f"The daemon is already running {cap} database dump(s) or restore(s).",
            hint=f"Wait for one to finish, or raise [services].max_concurrent_dumps "
            f"(currently {cap}).",
        )
    return NerditError(
        409,
        "dump.in_progress",
        f"Database '{name}' already has a dump, restore or run in flight.",
        hint="Dumps are single-flight per database — this is a per-row lock, not the "
        "daemon-wide backup lock behind `backup.in_progress`; wait and retry.",
    )


#: Hard cap on ``GET /databases/{name}/dumps`` (the D-P29-5 "module constant,
#: not a knob" pattern). ``[retention].dump_keep_last`` defaults to 5 and keeps
#: the directory small, but ``0`` is a supported "never prune", so retention
#: cannot be what bounds a read — this is.
DUMP_LIST_MAX = 500


#: The reason for "no outcome was observed" (D-P37-11): a runtime that never
#: started the sibling, or a daemon that failed to keep what it produced.
_DUMP_REASON_NO_OUTCOME = "interrupted"


def _dump_failed_error(code: str, reason: str, tail: list[str], *, noun: str) -> NerditError:
    """Build the 500 for a failed dump/restore (D-P37-11). ``hint`` is the scrubbed
    tail's last line only; the whole tail lives in ``config['last_dump']`` and
    surfaces on ``/diagnose``. The audit row and the event carry neither.
    """
    last = next((line for line in reversed(tail) if line.strip()), None)
    return NerditError(
        500,
        code,
        f"The database {noun} failed ({reason}).",
        hint=last or "Check the daemon log and `nerdit diagnose` for the captured output.",
        reason=reason,
    )


async def _emit_dump_event(
    event_type: str, *, service_name: str, reason: str | None = None, data: dict | None = None
) -> None:
    """Record one durable ``database.dump_*`` / ``database.restore_*`` row. ``data``
    is machine-shaped, never a log line (D-P37-11): failures carry ``reason``
    only, successes the basename and byte count.
    """
    recorder = get_recorder()
    if recorder is not None:
        await recorder.record(
            event_type, kind="database", service_name=service_name, reason=reason, data=data
        )


async def _pack_or_fail(
    data_dir: Path,
    result: DumpResult,
    *,
    service_name: str,
    image: str,
    backend: DataBackend,
) -> DumpBackupResult:
    """Pack a captured dump into its tar, or fail loudly leaving nothing (D-P37-7/11).

    ``create_dump_backup`` owns the checks, the manifest, the atomic write and the
    staging removal; this adds the disk re-check and maps its failures onto the
    §1.4 registry. ``tool_argv0`` is the program NAME only (D-P37-4), derived from
    a throwaway argv that dials nothing.
    """
    # ``DumpResult.output_path`` is ``None`` only for a RESTORE, and this helper
    # is reached from the dump route alone — narrowed here rather than at the
    # call site so the invariant is stated where it is relied on.
    output_path = result.output_path
    assert output_path is not None
    staging = Path(output_path).parent
    # What the packer published, if it published anything: filled by the worker
    # itself so a CANCELLED await can still see (and remove) a tar the awaiting
    # side never received.
    published: list[DumpBackupResult] = []

    def _pack_now() -> DumpBackupResult:
        packed = create_dump_backup(
            data_dir,
            service=service_name,
            engine=backend.name,
            image=image,
            staging=staging,
            backend_dump_filename=backend.dump_filename,
            backend_dump_format=backend.dump_format,
            tool_argv0=backend.dump_argv("h", 0, "o")[0],
        )
        published.append(packed)
        return packed

    try:
        await _recheck_disk(data_dir, output_path)
        # ``settled_to_thread`` (the P29 rule): a cancelled bare ``to_thread`` would
        # let the packer publish a tar after the slot was released (D-P37-9).
        return await settled_to_thread(_pack_now)
    except DumpOutputError as exc:
        # The sibling exited 0 and produced something the packer refuses — a
        # planted symlink, a zero-byte file, an extra entry. ``create_dump_backup``
        # removed the staging dir on its way out, so nothing is left behind.
        await _emit_dump_event("database.dump_failed", service_name=service_name, reason=exc.reason)
        raise _dump_failed_error("dump.failed", exc.reason, result.log_tail, noun="dump") from exc
    except BackupError as exc:
        # Daemon I/O failed while packing: a path-free daemon message becomes the
        # hint; no outcome was produced.
        logger.warning("Packing the dump of %s failed", service_name, exc_info=True)
        await _emit_dump_event(
            "database.dump_failed", service_name=service_name, reason=_DUMP_REASON_NO_OUTCOME
        )
        raise NerditError(
            500,
            "dump.failed",
            "The dump was captured but could not be packed.",
            hint=str(exc),
            reason=_DUMP_REASON_NO_OUTCOME,
        ) from exc
    except NerditError as exc:
        # The post-capture disk refusal has no reason of its own: ``interrupted``
        # on the feed, as ``_pack_reason`` stamps on the row (D-P37-11). The two
        # clauses above are never re-caught here, so this cannot double-emit.
        await _emit_dump_event(
            "database.dump_failed", service_name=service_name, reason=_pack_reason(exc)
        )
        await asyncio.to_thread(_rmtree_staging, staging)
        raise
    except asyncio.CancelledError:
        # The abandoned packer settled above, so a tar it published belongs to a
        # run the controller is about to record ``interrupted``: a failed dump
        # leaves nothing (D-P37-11). Unlinked without an await — this task is
        # already cancelled.
        for packed in published:
            with contextlib.suppress(OSError):
                os.unlink(packed.path)
        await asyncio.to_thread(_rmtree_staging, staging)
        raise
    except BaseException:
        # Anything else leaves behind the staging dir ``create_dump_backup``
        # would have removed. ``_rmtree_staging`` is missing-tolerant, so this
        # is a no-op on the paths where the packer already ran its ``finally``.
        await asyncio.to_thread(_rmtree_staging, staging)
        raise


@router.post(
    "/databases/{name}/dump",
    response_model=DatabaseDumpResponse,
    operation_id="create_database_dump",
)
async def create_database_dump(
    request: Request, name: str, body: DatabaseDumpRequest
) -> DatabaseDumpResponse:
    """Capture a logical, application-consistent dump of a managed database (P37).

    Bounded-synchronous (``timeout_s``, capped by ``[services].dump_timeout_max_s``):
    a rowless sibling from the database's own image runs ``pg_dump
    --format=custom`` / ``redis-cli --rdb`` over TCP, and the daemon packs the
    artifact with a manifest into ``<data_dir>/backups/nerdit-dump-*.tar.gz``.
    The tar is never served over HTTP; the response is metadata. Retention:
    ``[retention].dump_keep_last`` (default 5).

    Owner-or-admin + scope, mandatory in-route ``Idempotency-Key``, single-flight
    per database, audited ``database.dump``, events ``database.dump_succeeded`` /
    ``database.dump_failed``.
    """
    # Stamped BEFORE the first refusal so an authz denial is still audited
    # against the database the caller aimed at (the ``routes/domains.py``
    # posture). Extended, never rebuilt, as the facts become known.
    audit: dict[str, object] = {"service": name, "dump": None, "engine": None, "bytes": None}
    request.state.audit_params = audit_params(audit)

    require_role(request, TokenRole.submitter, TokenRole.admin)
    _require_idempotency_key(request, "database dump")

    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    # Owner-or-admin AND the P25 scope binding in one call (``require_owner_or_admin``
    # runs both legs of the D-P25-3 tree); the explicit name-scope check beside it
    # is what §1.4 asks for and what keeps the gate correct if the row ever loses
    # its owner column.
    require_owner_or_admin(request, job)
    service_name = job.service_name or name
    require_service_scope(request, service_name)

    backend = _dump_preflight(request, name, job, timeout_s=body.timeout_s)
    audit["engine"] = backend.name
    request.state.audit_params = audit_params(audit)
    await _require_endpoint(request, service_name)

    data_dir = _dump_data_dir(request)
    await _disk_preflight(data_dir, service_name)

    # Read once, here: the controller re-validates it (a row with no image
    # raises ``data_plane_unavailable`` before any container), so by the time the
    # packer wants it for the manifest it is known to be a non-empty string.
    image = str(parse_job_config(job).get("image") or "")
    service_controller = request.app.state.service_controller
    run_id = generate_id()
    started = time.monotonic()

    packed: DumpBackupResult | None = None

    async def _pack(captured: DumpResult) -> None:
        """Pack while the controller still holds the slot (D-P37-9): released before the
        pack, DELETE, a second dump and the drain would all see the row idle across
        the disk-heaviest half of the work.
        """
        nonlocal packed
        packed = await _pack_or_fail(
            data_dir, captured, service_name=service_name, image=image, backend=backend
        )

    # (P22) The durable effect is a file staged by a worker thread, so there is
    # no serialized DB writer to flip the claim marker; a cancellation landing
    # mid-dump must pin the claim ``interrupted`` rather than free the key for a
    # retry that would start a second sibling. Immediately before the container.
    mark_request_side_effect()
    try:
        await service_controller.dump_database(
            job, run_id=run_id, timeout_s=body.timeout_s, on_captured=_pack
        )
    except RunPreconditionError as exc:
        raise _dump_precondition_error(
            service_name,
            exc,
            cap=request.app.state.settings.services.max_concurrent_dumps,
        ) from exc
    except DumpError as exc:
        await _emit_dump_event("database.dump_failed", service_name=service_name, reason=exc.reason)
        raise _dump_failed_error("dump.failed", exc.reason, exc.log_tail, noun="dump") from exc
    except ContainerRuntimeError as exc:
        # The sibling never started: the same 500 ``dump.failed`` (§1.4's locked
        # registry), path-free — the runtime's message carries host paths.
        logger.warning("Dump of database %s failed in the runtime", service_name, exc_info=True)
        await _emit_dump_event(
            "database.dump_failed", service_name=service_name, reason=_DUMP_REASON_NO_OUTCOME
        )
        raise NerditError(
            500,
            "dump.failed",
            "The container runtime could not execute the dump.",
            hint="Check `nerdit doctor` (docker) and retry.",
            reason=_DUMP_REASON_NO_OUTCOME,
        ) from exc
    except NerditError as exc:
        # A packer refusal raised from inside the callback: the controller's
        # ``finally`` already stamped the "no outcome" floor; re-stamp the packer's
        # own token so ``/diagnose`` does not render a failure as a success
        # (D-P37-11).
        await _stamp_dump_reason(queries, job.id, _pack_reason(exc), run_id=run_id)
        raise
    # ``on_captured`` runs on exactly the path that returns, so a return with no
    # tar is a programming error, not a state a caller can reach.
    assert packed is not None

    duration_s = round(time.monotonic() - started, 3)
    audit.update({"dump": packed.basename, "bytes": packed.size_bytes})
    request.state.audit_params = audit_params(audit)
    await _stamp_dump_basename(queries, job.id, packed.basename, run_id=run_id)
    await _emit_dump_event(
        "database.dump_succeeded",
        service_name=service_name,
        data={"dump": packed.basename, "bytes": packed.size_bytes},
    )
    return DatabaseDumpResponse(
        service_name=service_name,
        dump=packed.basename,
        size_bytes=packed.size_bytes,
        sha256=packed.sha256,
        engine=packed.engine,
        format=packed.manifest.format,
        # The MANIFEST's timestamp, never a fresh clock read: the tar and the
        # response must agree about when this dump was taken, and the value that
        # survives is the one inside the archive.
        created_at=datetime.fromisoformat(packed.manifest.created_at),
        duration_s=duration_s,
    )


def _pack_reason(exc: NerditError) -> str:
    """The reason token a packer refusal leaves on the row (D-P37-11): the envelope's
    ``reason`` extra, or ``interrupted`` for the one refusal without a token (the
    post-capture 409 ``dump.insufficient_disk``).
    """
    reason = exc.extra.get("reason")
    return reason if isinstance(reason, str) and reason else _DUMP_REASON_NO_OUTCOME


async def _patch_last_dump(
    queries, job_id: str, *, field: str, value: str, run_id: str, what: str
) -> None:
    """Best-effort single-key patch of ``config['last_dump']`` (D-P37-11).

    One guarded ``json_set`` (``patch_last_dump_field``): applied only while the
    blob still carries ``run_id``, so a later dump or restore on the row is never
    overwritten. A failure here never masks the outcome the caller already has.
    """
    try:
        applied = await queries.patch_last_dump_field(
            job_id, field=field, value=value, expect_run_id=run_id
        )
        if not applied:
            logger.warning(
                "Did not record the %s on row %s: last_dump belongs to another run", what, job_id
            )
    except Exception:  # noqa: BLE001 — bookkeeping must not mask the outcome
        logger.warning("Could not record the %s on row %s", what, job_id, exc_info=True)


async def _stamp_dump_basename(queries, job_id: str, basename: str, *, run_id: str) -> None:
    """Fill ``last_dump.dump`` once the tar exists; the controller stamps ``None``
    in its ``finally`` because the packer has not returned yet (D-P37-11)."""
    await _patch_last_dump(
        queries, job_id, field="dump", value=basename, run_id=run_id, what="dump basename"
    )


async def _stamp_dump_reason(queries, job_id: str, reason: str, *, run_id: str) -> None:
    """Correct ``last_dump.reason`` after a packer refusal so ``/diagnose`` does not
    render the run as a success that produced nothing (D-P37-11). ``exit_code``
    stays the tool's real ``0``."""
    await _patch_last_dump(
        queries, job_id, field="reason", value=reason, run_id=run_id, what="dump failure reason"
    )


@router.get(
    "/databases/{name}/dumps",
    response_model=DatabaseDumpListResponse,
    operation_id="list_database_dumps",
)
async def list_database_dumps(request: Request, name: str) -> DatabaseDumpListResponse:
    """List the dump tars this daemon holds for one managed database (P37).

    Owner-or-admin + scope (the P29 workspace-read precedent: enumerating dumps is
    knowing something about the data). Names, sizes, mtimes only — archives are
    never opened. Newest first, capped at :data:`DUMP_LIST_MAX` with ``truncated``
    (bounded by construction, not by retention: ``dump_keep_last = 0`` is
    supported). No cursor.
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    require_owner_or_admin(request, job)
    service_name = job.service_name or name
    require_service_scope(request, service_name)
    if job.kind is not JobKind.database:
        raise _dump_not_a_database(name, job.kind)

    backups = _dump_data_dir(request) / "backups"

    def _scan() -> list[DatabaseDumpItem]:
        items: list[DatabaseDumpItem] = []
        try:
            entries = list(os.scandir(backups))
        except OSError:
            # No backups dir yet (nothing has ever been dumped or backed up) is
            # an empty list, never a 404: "this database has no dumps" is the
            # true answer and the caller's next step is the same either way.
            return items
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if not fnmatch.fnmatch(entry.name, DUMP_TAR_GLOB):
                    continue
                match = DUMP_TAR_RE.match(entry.name)
                # The service segment of the NAME is the filter: the tar is
                # never opened, so the manifest's own ``service`` (provenance,
                # and possibly a different database after a clone-by-restore)
                # is not consulted here.
                if match is None or match.group("service") != service_name:
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue  # deleted between scan and stat — a retention sweep race
            items.append(
                DatabaseDumpItem(
                    dump=entry.name,
                    size_bytes=st.st_size,
                    created_at=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
                )
            )
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items

    items = await asyncio.to_thread(_scan)
    return DatabaseDumpListResponse(
        service_name=service_name,
        dumps=items[:DUMP_LIST_MAX],
        truncated=len(items) > DUMP_LIST_MAX,
    )


#: "This container is up right now" — ``degraded`` included, since Decision #2
#: leaves a degraded container running with its connection pool open.
_LIVE_DEPENDENT_STATUSES = frozenset({JobStatus.running.value, JobStatus.degraded.value})


def _restore_dependents(
    rows: list[dict], service_name: str, *, exclude_id: str
) -> list[dict[str, str | None]]:
    """``[db.*]`` dependents of *service_name* whose container is UP (D-P37-6):
    ``service_purge._iter_db_dependents`` narrowed to ``running`` or ``degraded``
    rows. A stopped app blocks a delete (its next launch breaks) but not a
    restore; a degraded one still holds locks and connections, so it counts.
    """
    return [
        _dep_entry(row, binding)
        for row, binding in _iter_db_dependents(rows, service_name, exclude_id=exclude_id)
        if row.get("status") in _LIVE_DEPENDENT_STATUSES
    ]


def _check_manifest_matches_row(manifest, backend, service_name: str) -> None:
    """Refuse a dump that contradicts this row's engine (D-P37-10), all 422 and
    before any container: manifest ``engine`` == the backend key, ``file`` == the
    backend's dump filename, ``format`` == ``DataBackend.dump_format``. Bytes are
    not judged (sha256 verifies content against the manifest, the engine's tool
    refuses the rest). ``manifest.service`` is deliberately NOT checked: a
    cross-name restore is how a clone is made.
    """
    if manifest.engine != backend.name:
        raise NerditError(
            422,
            "restore.engine_mismatch",
            f"That dump is a {manifest.engine} dump; '{service_name}' is {backend.name}.",
            hint="Restore it into a database of the same engine, or create one.",
            detail={"expected": backend.name, "found": manifest.engine},
            reason="engine_mismatch",
        )
    if manifest.file != backend.dump_filename or manifest.format != backend.dump_format:
        raise NerditError(
            422,
            "restore.manifest_invalid",
            "The dump archive's manifest disagrees with its own engine.",
            hint="Restore a tar this daemon wrote; a re-packed or edited archive is refused.",
            detail={"reason": "schema"},
            reason="schema",
        )


async def _stage_or_fail(data_dir: Path, run_id: str, *, service_name: str) -> Path:
    """Create this restore's private staging slot, or refuse path-free.
    ``create_dump_staging_dir`` (D-P37-2) raises only
    :class:`~nerdit.core.volumes.VolumeSpecError`; that is a tampering or I/O
    signal, so it becomes a path-free 500 with the "no outcome" token and the
    detail stays in the daemon log (M3).
    """
    try:
        return await asyncio.to_thread(create_dump_staging_dir, data_dir, run_id)
    except VolumeSpecError as exc:
        logger.error("Refusing to stage a restore for %s: %s", service_name, exc)
        raise NerditError(
            500,
            "restore.failed",
            "The daemon could not prepare a staging directory for the restore.",
            hint="Check the daemon log; the dump-staging directory is daemon-owned "
            "and must not be a symlink.",
            reason=_DUMP_REASON_NO_OUTCOME,
        ) from exc


def _refuse_if_in_use(rows: list[dict], service_name: str, *, exclude_id: str) -> None:
    """409 ``restore.in_use`` when a LIVE app is bound to this database (D-P37-6)."""
    dependents = _restore_dependents(rows, service_name, exclude_id=exclude_id)
    if not dependents:
        return
    raise NerditError(
        409,
        "restore.in_use",
        f"Database '{service_name}' is in use by {len(dependents)} live service(s).",
        hint="Stop the bound app(s) first, or pass --force — a live client holds the "
        "locks a Postgres restore needs, and sees a Redis restore as a dropped "
        "connection.",
        detail={"dependents": dependents},
        # Top-level too, matching the ``resource.in_use`` shape the delete guard
        # already emits and the CLI already knows how to render.
        dependents=dependents,
    )


async def _restore_with_slot(  # noqa: PLR0913 - the route's own locals, all keyword
    request: Request,
    job: Job,
    body: DatabaseRestoreRequest,
    *,
    backend: DataBackend,
    service_name: str,
    tar_path: Path,
    data_dir: Path,
    run_id: str,
    force: bool,
) -> None:
    """Extract, validate and run the restore under a slot the caller holds.

    Raises the route's envelopes: 422 ``restore.manifest_invalid``, 422
    ``restore.engine_mismatch``, 409 ``restore.in_use``, 500 ``restore.failed``.
    The staging dir is removed on every path.
    """
    queries = request.app.state.queries
    service_controller = request.app.state.service_controller
    staging = await _stage_or_fail(data_dir, run_id, service_name=service_name)
    try:
        try:
            manifest = await asyncio.to_thread(extract_dump_tar, tar_path, staging)
        except DumpManifestError as exc:
            raise NerditError(
                422,
                "restore.manifest_invalid",
                "The dump archive failed validation.",
                hint="Restore a tar this daemon wrote; a re-packed or edited archive "
                "is refused, and so is one whose payload does not match its manifest.",
                detail={"reason": exc.reason},
                # Top-level too: the MCP mapper drops ``detail`` but copies extras
                # through (the ``domain.invalid`` precedent).
                reason=exc.reason,
            ) from exc

        _check_manifest_matches_row(manifest, backend, service_name)

        if not force:
            rows = await queries.list_workload_configs()
            _refuse_if_in_use(rows, service_name, exclude_id=job.id)

        mark_request_side_effect()
        await service_controller.restore_database(
            job, run_id=run_id, staging=staging, timeout_s=body.timeout_s, slot_held=True
        )
    except DumpError as exc:
        await _emit_dump_event(
            "database.restore_failed", service_name=service_name, reason=exc.reason
        )
        raise _dump_failed_error(
            "restore.failed", exc.reason, exc.log_tail, noun="restore"
        ) from exc
    except ContainerRuntimeError as exc:
        # One code, one reason token — see the dump route's twin above.
        logger.warning("Restore of database %s failed in the runtime", service_name, exc_info=True)
        await _emit_dump_event(
            "database.restore_failed", service_name=service_name, reason=_DUMP_REASON_NO_OUTCOME
        )
        raise NerditError(
            500,
            "restore.failed",
            "The container runtime could not execute the restore.",
            hint="Check `nerdit doctor` (docker) and retry.",
            reason=_DUMP_REASON_NO_OUTCOME,
        ) from exc
    finally:
        # ``restore_database`` removes staging only once it has entered its try
        # block; every earlier refusal would leak it. Missing-tolerant.
        await asyncio.to_thread(_rmtree_staging, staging)


@router.post(
    "/databases/{name}/restore",
    response_model=DatabaseRestoreResponse,
    operation_id="restore_database_dump",
)
async def restore_database_dump(
    request: Request,
    name: str,
    body: DatabaseRestoreRequest,
    force: bool = Query(False, description="Restore even while running apps are bound to it"),
) -> DatabaseRestoreResponse:
    """Restore a managed database from one of this daemon's dump tars (P37).

    Destructive, admin-only (D-P37-8). Postgres: ``pg_restore --clean --if-exists
    --single-transaction --exit-on-error`` into the running database — one
    transaction, objects created since the dump survive, no CASCADE. Redis: the
    RDB is installed as the multi-part-AOF base under a brief stop/start; the
    previous ``appendonlydir`` is kept as ``appendonlydir.pre-restore-<stamp>``.

    A live (``running`` or ``degraded``) bound app refuses with ``409
    restore.in_use`` naming the dependents; ``?force=true`` proceeds. Cross-name
    restore is allowed (the target is the path parameter; ``manifest.service`` is
    provenance). The row's dump slot is held from before extraction to after the
    controller returns (D-P37-9).

    Mandatory in-route ``Idempotency-Key``, audited ``database.restore``, events
    ``database.restore_succeeded`` / ``database.restore_failed``.
    """
    audit: dict[str, object] = {
        "service": name,
        "dump": body.dump,
        "engine": None,
        "force": force,
    }
    request.state.audit_params = audit_params(audit)

    require_role(request, TokenRole.admin)
    _require_idempotency_key(request, "database restore")

    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    service_name = job.service_name or name

    backend = _dump_preflight(request, name, job, timeout_s=body.timeout_s)
    audit["engine"] = backend.name
    request.state.audit_params = audit_params(audit)
    await _require_endpoint(request, service_name)

    data_dir = _dump_data_dir(request)
    # ``body.dump`` cleared the basename grammar during request validation
    # (``DUMP_BASENAME_PATTERN``), so this join is structurally incapable of
    # escaping ``backups/`` — no traversal check is needed here because no
    # traversal-shaped value can reach it.
    tar_path = data_dir / "backups" / body.dump
    if not await asyncio.to_thread(_is_regular_file, tar_path):
        raise NerditError(
            404,
            "restore.not_found",
            f"No dump named '{body.dump}'.",
            hint=f"List this database's dumps with `nerdit db dumps {service_name}`.",
        )

    service_controller = request.app.state.service_controller
    run_id = generate_id()
    started = time.monotonic()

    # Claim the row's dump slot BEFORE the archive is touched (D-P37-9, §1.4):
    # a second restore, dump or run is refused before it can extract, and
    # ``DELETE /services/{name}`` sees the row busy for the whole operation.
    try:
        service_controller.reserve_dump_slot(job.id, run_id)
    except RunPreconditionError as exc:
        raise _dump_precondition_error(
            service_name,
            exc,
            cap=request.app.state.settings.services.max_concurrent_dumps,
        ) from exc
    try:
        await _restore_with_slot(
            request,
            job,
            body,
            backend=backend,
            service_name=service_name,
            tar_path=tar_path,
            data_dir=data_dir,
            run_id=run_id,
            force=force,
        )
    finally:
        service_controller.release_dump_slot(job.id, run_id)

    duration_s = round(time.monotonic() - started, 3)
    await _emit_dump_event(
        "database.restore_succeeded", service_name=service_name, data={"dump": body.dump}
    )
    return DatabaseRestoreResponse(
        service_name=service_name,
        dump=body.dump,
        engine=backend.name,
        duration_s=duration_s,
    )


def _is_regular_file(path: Path) -> bool:
    """``lstat``-based "this basename names a regular file under backups/".

    ``lstat``, never ``exists()``: a symlink planted in the backups directory
    must not be followed into a file the daemon would then hand to
    ``tarfile.open``. Run on a worker thread with the extraction, so a slow or
    hung filesystem never stalls the loop.
    """
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False
