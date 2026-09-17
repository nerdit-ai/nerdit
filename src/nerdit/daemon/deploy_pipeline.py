"""Shared deployment pipeline for ZIP, Git, template and workspace inputs.

Ingress routes prepare a build context; this module validates config, detects
the buildpack, writes the service row and shapes the response. Failed requests
remove their context tree. Imports stay below routes to avoid cycles.

Redeploy ordering matters: source metadata follows build fields, queued state
follows both, and data-volume setup follows the environment merge.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import shutil
import stat
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from fastapi import Request
from pydantic import ValidationError

from nerdit.config.app_config import (
    AI_SHAPE_HINT,
    require_provisioned_databases,
    require_served_models,
    validate_ai_section,
    validate_db_section,
    validate_deploy_fields,
)
from nerdit.config.build import BuildSettings
from nerdit.config.defaults import ZIP_EXCLUDE_PATTERNS, app_image_repo, app_image_tag
from nerdit.config.project import (
    _DISPLAY_UNSAFE_RE,
    _DNS_LABEL_RE,
    PROJECT_CONFIG_NAME,
    DeployConfig,
    shared_secret_keys,
    shared_secret_keys_for,
)
from nerdit.config.redaction import redact_url_userinfo
from nerdit.core.bindings.secretref import walk_secret_ref
from nerdit.core.builder import (
    GENERATED_DOCKERFILE_NAME,
    BuildpackNotSupported,
    BuildPlan,
    detect,
)
from nerdit.core.eventlog import get_recorder, record_job_event
from nerdit.core.gitsource import (
    _AUTH_SIGNATURES,
    GITHUB_INSTALLATION_REF,
    GitSourceError,
    clone_source,
    git_source_meta,
    github_repo_slug,
    installation_token_allowed_for_host,
    validate_ref,
    validate_repo_url,
    validate_subdir,
)
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.secrets import SHARED_SCOPE, SecretDecryptError
from nerdit.daemon.audit import audit_params, record_shared_referenced
from nerdit.daemon.auth import (
    Principal,
    QuotaExceeded,
    current_principal,
    require_owner_or_admin,
)
from nerdit.daemon.errors import NerditError
from nerdit.daemon.views.hosted import load_hosted_context
from nerdit.daemon.views.service import (
    _cutover_in_progress_error,
    _run_in_progress_error,
    _service_response,
)
from nerdit.db.models import Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries, ServiceNameTaken
from nerdit.utils.ids import generate_id

logger = logging.getLogger(__name__)

# Cap on how much of an unknown `[deploy]` key set reaches a daemon log line
# . TOML keys are app-author-controlled strings, so both their count and
# their content are bounded before interpolation.
_UNKNOWN_KEY_LOG_LIMIT = 10
_UNKNOWN_KEY_LOG_MAX_CHARS = 64


def _warn_unknown_deploy_keys(section: dict, app_name: str) -> list[str]:
    """Warn about unknown deploy keys without rejecting newer app configurations.

    Sanitize control and bidi characters and bound key counts/lengths before
    logging untrusted TOML keys. Return the same sanitized names for response
    hints, with a truncation marker when more keys were omitted.
    """
    unknown = sorted(key for key in section if key not in DeployConfig.model_fields)
    if not unknown:
        return []
    shown = [
        _DISPLAY_UNSAFE_RE.sub("?", str(key))[:_UNKNOWN_KEY_LOG_MAX_CHARS]
        for key in unknown[:_UNKNOWN_KEY_LOG_LIMIT]
    ]
    suffix = f" (+{len(unknown) - len(shown)} more)" if len(unknown) > len(shown) else ""
    logger.warning(
        "[deploy] in %s for app '%s' declares key(s) this daemon does not "
        "understand and will IGNORE: %s%s",
        PROJECT_CONFIG_NAME,
        app_name,
        ", ".join(shown),
        suffix,
    )
    # The truncation marker rides the RETURNED list as a trailing pseudo-entry
    # so the response hint and the dry-run plan warning truncate as LOUDLY as
    # the log line does — past the cap the reader must not read a silently
    # shortened list as the whole set. The log line above renders from `shown`
    # and stays byte-unchanged.
    return [*shown, suffix.strip()] if suffix else shown


# (Agent-DX) The one definition of each advisory wording. Both the deploy-result
# `hints` channel and the `?dry_run` plan `warnings` render THESE strings,
# so the plan and the response can never drift apart.
_UNKNOWN_KEYS_HINT = (
    "[deploy] key(s) this daemon does not understand were IGNORED: {keys}. "
    "Check them against the [deploy] schema (there is no `build` key — run the "
    "build in your Dockerfile)."
)
# (P26 WP0 / L1) The hint named the trap but not the way out: an operator who
# reads "set your base path" has no idea the daemon has a mode that removes the
# requirement entirely. Naming `subdomain` here is the whole of WP0 —
# `docs/guide/proxy.md` §7 has documented that mode since P3.5.
_PATH_MODE_HINT = (
    "This node serves apps in 'path' proxy mode: the app is reachable at "
    "/{name}/, so a frontend build must set its base path to '/{name}/' "
    "(e.g. Vite `base`, CRA `homepage`) or its assets will 404. Subdomain "
    "mode ([proxy].mode = 'subdomain', docs/guide/proxy.md §7) serves the app "
    "at the root of its own hostname and needs no base path."
)

# (P34, field failure 2026-08-23) The deploy verbs return the instant the row is
# written — the build and the first launch happen off-tick — so a 201 saying
# `status: building` is the NORMAL success shape, not a finished deploy. The
# agent that hit this read `hints: []` beside it and concluded the deploy had
# landed; the crash-loop was only discoverable by knowing to call
# `get_events`/`diagnose_service` unprompted.
#
# Two channels close that loop, carrying deliberately the SAME sentence so a
# reader that parses only one of them still learns the whole fact:
#
# * `next_step` — a machine-shaped `{tool, args, why}` an agent executes
#   without parsing prose. It always names `wait_for_service`: that is the one
#   call which blocks until the outcome is KNOWN.
# * a `hints` entry carrying the same `why`, which is what makes "hints is
#   never empty on an in-flight deploy" true. Both pre-existing hints are
#   conditional (unknown keys / path mode AND a live proxy), so the empty list
#   was the common case on exactly the nodes a remote agent works on.
_ASYNC_DEPLOY_WHY = (
    "deploy is asynchronous; this returns when the service is healthy or "
    "failed and includes the diagnosis on failure"
)
# (D) The failure-side companion: a deploy landing on a row that is ALREADY
# failed (a redeploy over a crash-looping app) must not answer with the generic
# async sentence alone. Ordered FIRST for the same reason the unknown-key hint
# is — it is about state the caller already owns. The interpolated `detail` is
# a machine token from the row (a JobStatus value or a `last_deploy.reason`),
# never app-authored text and never an error message, so nothing app-controlled
# reaches the hints channel through it.
_FAILED_STATE_HINT = (
    "The previous generation of '{name}' is in a failed state ({detail}). "
    "Call diagnose_service('{name}') for the remediation code, crash forensics "
    "and log tail if this deploy settles the same way."
)


def _next_step(name: str) -> dict[str, object]:
    """The machine-shaped "what now" every in-flight deploy response carries.

    Structured, not prose: an agent reads `tool`/`args` and executes them.
    `why` is the same string the companion hint renders, so a client that
    surfaces only one of the two channels still states the whole fact.
    """
    return {
        "tool": "wait_for_service",
        "args": {"name": name},
        "why": _ASYNC_DEPLOY_WHY,
    }


def _unknown_keys_message(unknown_deploy_keys: list[str]) -> str:
    """Render the ignored-`[deploy]`-keys advisory. Key NAMES only, never values.

    The names arrive already control-char-sanitized and bounded (count and
    length) by `_warn_unknown_deploy_keys`; nothing here widens either
    bound, and a value from the TOML never reaches this string.
    """
    return _UNKNOWN_KEYS_HINT.format(keys=", ".join(unknown_deploy_keys))


def _read_project_toml(
    context_dir: Path,
) -> tuple[dict, tomllib.TOMLDecodeError | UnicodeDecodeError | None]:
    """Read once; defer parse errors until after deploy/buildpack validation.

    The read side of the H1 class: a git clone materializes a committed
    `nerdit.toml -> ../../secrets.key` verbatim, and `is_file()`/`read_text()`
    both follow it — the daemon would read a file of the submitter's choosing
    and then render its decode failure into a `422 deploy.invalid_ai` message
    carrying one byte of the target and its offset. `lstat`, so a symlink is
    refused by name instead. Anything else non-regular (a directory, a fifo)
    stays "absent", exactly as `is_file()` reported it.

    Raises:
        NerditError: 422 when the entry exists and is a symlink.
    """
    path = context_dir / PROJECT_CONFIG_NAME
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return {}, None
    if stat.S_ISLNK(mode):
        raise _unsafe_source_file(PROJECT_CONFIG_NAME)
    if not stat.S_ISREG(mode):
        return {}, None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")), None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        return {}, exc


def _parse_deploy_defaults(data: dict, fallback_name: str) -> tuple[dict, list[str]]:
    """Validate parsed deploy defaults and return them with unknown-key warnings.

    Explicit request fields override these defaults; fallback_name is the service
    identity even when the file declares another name. Missing config returns
    ({}, []).
    """
    section = data.get("deploy")
    if section is None:
        return {}, []
    if not isinstance(section, dict):
        raise NerditError(
            422,
            "deploy.invalid",
            f"[deploy] in {PROJECT_CONFIG_NAME} must be a table.",
            hint=_DEPLOY_SHAPE_HINT,
        )
    try:
        # The form name wins over a [deploy].name in the ZIP (it is the service
        # identity and the only name the route consumes), so validate with it
        # last — a stale ZIP name must not 422 an otherwise valid deploy.
        DeployConfig(**{**section, "name": fallback_name})
    except (ValidationError, TypeError) as exc:
        msg = (
            exc.errors()[0].get("msg", "validation error")
            if isinstance(exc, ValidationError)
            else str(exc)
        )
        raise NerditError(
            422,
            "deploy.invalid",
            f"Invalid [deploy] section in {PROJECT_CONFIG_NAME}: {msg}",
            hint=_DEPLOY_SHAPE_HINT,
        ) from exc
    # Non-fatal visibility for keys this daemon silently drops — into the
    # daemon log, and (Agent-DX) back to the caller via the returned names.
    unknown = _warn_unknown_deploy_keys(section, fallback_name)
    return section, unknown


# The [deploy] shape hint returned with a `deploy.invalid` envelope when the
# uploaded app's own nerdit.toml [deploy] section fails to parse.
#
# SYNC OBLIGATION: this hint, the `[deploy]` key list in the MCP deploy tool
# docstrings (`mcp/tools/deploy.py` `deploy`/`deploy_git`,
# `mcp/tools/workspaces.py` `deploy_app`) and `DeployConfig.model_fields`
# describe the same schema and must agree. Both halves are guarded:
# `tests/test_mcp.py::test_deploy_tool_descriptions_list_every_deploy_config_field`
# for the docstrings, and
# `tests/test_deploy_route.py::test_deploy_shape_hint_names_every_deploy_config_field`
# for the hint below — a key added to the schema and not named here would leave
# the 422 recommending a schema the daemon no longer has. The hint's guard is
# FORWARD-only (every field is named) because, unlike the docstrings' `key`
# markup, this is prose: the reverse direction cannot distinguish a key name
# from an ordinary word.
_DEPLOY_SHAPE_HINT = (
    "[deploy] takes name (DNS label), port (1-65535), gpus (>= 0), start, "
    "health, health_type ('http'|'tcp'), memory_limit (e.g. '512m'), "
    "cpu_limit (> 0), volumes (['data:/data'] — at most 7 declared, or 8 when "
    "one of them is named 'data', since the implicit data volume counts "
    "towards the limit of 8; DNS-label names, "
    "absolute container paths), release (a single-line pre-swap command, "
    "<=4096 chars), cutover (false disarms the zero-downtime swap), "
    "auto_deploy (true opts into push-triggered redeploy) and edge_auth "
    "({user, password = '${secrets.KEY}'}); "
    "build_settings (install/build/start/node_version/package_manager/subdir/"
    "public_env); "
    "explicit deploy fields override these values."
)


def reject_non_service_row(existing: Job | None, name: str) -> None:
    """Reject a name owned by a model or database before deploy can overwrite it.

    Kind is publicly readable, so this 409 guard precedes ownership checks.
    """
    if existing is not None and existing.kind is not JobKind.service:
        noun = "database" if existing.kind is JobKind.database else "served model"
        # A database row cannot be removed with a bare `nerdit services rm`
        # (that 409s `db.delete_requires_purge` — the CLI purge default is
        # `secrets`); it must name `--purge data` (which also removes the
        # minted credential). A served model has no such coupled-lifecycle gate.
        rm_hint = (
            f"remove the database first with `nerdit services rm {name} --purge data`"
            if existing.kind is JobKind.database
            else "remove the served model first with `nerdit services rm`"
        )
        raise NerditError(
            409,
            "deploy.kind_mismatch",
            f"'{name}' is a {noun}, not a deployable app.",
            hint=f"Pick another name, or {rm_hint}.",
        )


def _validate_request_name(name: str) -> None:
    """Validate request names before TOML parsing so errors identify the right input."""
    if not _DNS_LABEL_RE.match(name):
        raise NerditError(
            422,
            "deploy.invalid",
            f"Invalid service name '{name}' in the request 'name' field: must be a "
            "DNS label (lowercase letters, digits and '-', 1-63 chars, starting "
            "and ending with a letter or digit).",
            hint="Fix the request's 'name' field — this error is not about the app's nerdit.toml.",
        )


def _env_diff(prior: dict[str, str], req: dict[str, str | None] | None) -> dict:
    """Names-only diff of a redeploy's env request against the prior env.

    `req` carries the incoming env map (a `None` value is a null-delete). The
    result lists KEY NAMES only — never values (D-B) — plus `kept` as an int
    count of the prior keys the request leaves untouched.
    """
    req = req or {}
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    for k, v in req.items():
        if v is None:
            if k in prior:
                removed.append(k)  # null-delete of an existing key
        elif k not in prior:
            added.append(k)
        elif prior.get(k) != v:
            changed.append(k)
    kept = len(prior) - len(removed) - len(changed)
    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "changed": sorted(changed),
        "kept": kept,
    }


# Container path of the implicit per-app data volume (D-P14-4).
IMPLICIT_DATA_VOLNAME = "data"
IMPLICIT_DATA_PATH = "/data"


def _resolve_data_volume(volumes: list[str] | None) -> tuple[list[str], str]:
    """Return the volume list with an implicit `data:/data` + the data path.

    Every `/deploy`-created app gets the implicit `data:/data` volume unless
    it already declares a volume named `data` (which re-targets the implicit
    volume to a different container path). Idempotent: an already-present `data`
    volume is left untouched, so a redeploy never doubles it (D-P14-4).
    """
    result = list(volumes or [])
    for spec in result:
        if not isinstance(spec, str):
            continue
        volname, sep, container_path = spec.partition(":")
        if volname == IMPLICIT_DATA_VOLNAME and sep:
            return result, container_path or IMPLICIT_DATA_PATH
    result.append(f"{IMPLICIT_DATA_VOLNAME}:{IMPLICIT_DATA_PATH}")
    return result, IMPLICIT_DATA_PATH


def _ensure_data_volume(config: dict, prev_data_path: str | None = None) -> None:
    """Ensure the implicit data volume and NERDIT_DATA_DIR in the config, in place.

    Applies only to deployment, not POST /services. Preserve explicit env overrides.
    On retargeted redeploy, move NERDIT_DATA_DIR only if it equals prev_data_path,
    identifying the previously generated value rather than a user override.
    """
    volumes, data_path = _resolve_data_volume(config.get("volumes"))
    config["volumes"] = volumes
    env = config.setdefault("env", {})
    if isinstance(env, dict):
        current = env.get("NERDIT_DATA_DIR")
        if current is None or (prev_data_path is not None and current == prev_data_path):
            env["NERDIT_DATA_DIR"] = data_path


def _stamp_queued(config: dict, *, version: int, action: str) -> None:
    """Start a queued deploy generation and clear previous crash forensics.

    Capture immutable source repo/ref/sha after source metadata is updated. Rollback
    and nongit deployments leave these null because no source was cloned for this
    generation. Remediation starts null and is set at settlement.
    """
    now = datetime.now(UTC).isoformat()
    source = config.get("source")
    git = (
        source
        if action != "rollback" and isinstance(source, dict) and source.get("type") == "git"
        else {}
    )
    config["last_deploy"] = {
        "version": version,
        "action": action,
        "phase": "queued",
        "image": config.get("image"),
        "started_at": now,
        "updated_at": now,
        "reason": None,
        "error_class": None,
        "error_message": None,
        "repo": git.get("repo"),
        "ref": git.get("ref"),
        "sha": git.get("commit_sha"),
        "remediation_code": None,
    }
    for key in ("last_exit_code", "oom_killed", "last_crash_at", "gpu_oom", "priv_denied"):
        config.pop(key, None)


def _dockerignore_patterns() -> list[str]:
    """Translate `ZIP_EXCLUDE_PATTERNS` into any-depth `.dockerignore` rules.

    The ZIP path (`cli/upload.should_exclude`) matches an excluded name at *any*
    path depth (it tests every path component). Bare `.dockerignore` patterns are
    root-anchored, so a nested `node_modules` / `__pycache__` (e.g. a vendored
    subpackage) would still ship in the build context. Prefixing each pattern with
    `**/` restores the ZIP's any-depth semantics.
    """
    out: list[str] = []
    for pattern in sorted(ZIP_EXCLUDE_PATTERNS):
        # Directory patterns keep their trailing slash; `**/` anchors at any depth.
        out.append(f"**/{pattern}")
    return out


def _unsafe_generated_file(filename: str) -> NerditError:
    """The refusal both generated-file writes share. Names the entry, never a path."""
    return NerditError(
        422,
        "deploy.unsafe_generated_file",
        f"The build context has a non-regular file at '{filename}'; the daemon "
        "generates that file and refuses to write through it.",
        hint=f"Remove '{filename}' (a symlink or special file) from the source tree.",
    )


def _audit_safe_ai(ai_spec: dict) -> dict:
    """An `[ai.*]` spec with every `base_url` credential stripped, for the audit row.

    The persisted spec is unchanged — the launch path needs the URL as written.
    This is the rendering that rides into `audit_log.params_redacted` and the
    admin `audit.*` bus frame, where a userinfo password must never appear.
    """
    out: dict = {}
    for name, spec in ai_spec.items():
        if isinstance(spec, dict) and "base_url" in spec:
            out[name] = {**spec, "base_url": redact_url_userinfo(spec["base_url"])}
        else:
            out[name] = spec
    return out


def _unsafe_source_file(filename: str) -> NerditError:
    """The refusal for a non-regular file the daemon READS out of the context.

    Twin of `_unsafe_generated_file` for the read side: the daemon does not
    generate this one, it parses it, and following a committed symlink means
    parsing a file the submitter never uploaded.
    """
    return NerditError(
        422,
        "deploy.unsafe_source_file",
        f"The build context has a symlink at '{filename}'; the daemon reads that "
        "file and refuses to follow a link out of the tree.",
        hint=f"Commit '{filename}' as an ordinary file.",
    )


def _existing_regular_file(target: Path) -> bool:
    """Report whether *target* is an ordinary file, refusing anything else.

    `lstat`, never `exists()`: the build context is caller-supplied and a git
    clone materializes committed symlinks verbatim, so a `.dockerignore`
    symlink pointing outside the tree must not read as "absent" (dangling) or
    as "a user's file" (resolvable).

    Raises:
        NerditError: 422 when the path exists and is not a regular file.
    """
    try:
        mode = os.lstat(target).st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise _unsafe_generated_file(target.name)
    return True


def _write_generated_file(target: Path, text: str) -> None:
    """Write a daemon-generated file into the build context, never through a link.

    `O_NOFOLLOW` is the enforcement; the `_existing_regular_file` pre-check only
    buys the clear error message. Without it a committed
    `Dockerfile.nerdit -> ../../secrets.key` makes a submitter deploy an
    arbitrary-file overwrite, and a pre-check alone would lose the race against
    a symlink created after it.
    """
    _existing_regular_file(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
    try:
        fd = os.open(target, flags, 0o644)
    except OSError as exc:
        # ELOOP (Linux/macOS) / EMLINK (some BSDs): the final component became a
        # symlink between the check and the open.
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise _unsafe_generated_file(target.name) from exc
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_generated_dockerignore(context_dir: Path) -> None:
    """Add buildpack exclusions without overwriting a user's .dockerignore.

    Match ZIP exclusions at every depth. Exclude Dockerfile.nerdit from COPY context;
    BuildKit reads the selected Dockerfile separately.
    """
    target = context_dir / ".dockerignore"
    if _existing_regular_file(target):
        return  # respect a user-provided .dockerignore
    patterns = _dockerignore_patterns() + [GENERATED_DOCKERFILE_NAME]
    header = "# Generated by Nerdit (P13) — regenerated on deploy.\n"
    _write_generated_file(target, header + "\n".join(patterns) + "\n")


def _build_health_blob(path: str | None, health_type: str | None) -> dict | None:
    """Build a health specification from probe kind and path.

    TCP yields only type=tcp. Explicit HTTP defaults its path to /. Without a kind,
    return a path-only HTTP blob when a path exists, otherwise None.
    """
    if health_type == "tcp":
        return {"type": "tcp"}
    if health_type == "http":
        return {"path": path or "/"}
    if path:
        return {"path": path}
    return None


@dataclass
class Effective:
    """The resolved effective deploy fields for one `_finalize_deploy` call.

    Bundles the validated `DeployConfig` (name/port/gpus/start/health/
    memory_limit/cpu_limit/volumes/release) with the three scalar fields the
    later phases consume directly: the VALIDATED gpus count (`DeployConfig` coerces
    lax inputs, e.g. a quoted `"2"` in a ZIP's `nerdit.toml`, to int — the
    raw dict value would reach the quota arithmetic as a str), the resolved
    start command, and the resolved health path + probe kind (used for the
    health blob and the API-clobber diff).
    """

    deploy_cfg: DeployConfig
    eff_gpus: int
    eff_start: str | None
    eff_health: str | None
    eff_health_type: str | None


def resolve_effective_fields(
    *,
    name: str,
    port: int | None,
    gpus: int | None,
    start: str | None,
    health: str | None,
    zip_deploy: dict,
    existing: Job | None,
    prev_cfg: dict,
) -> Effective:
    """Precedence merge: form field > ZIP [deploy] value > prior row value > buildpack
    default.

    [deploy] from the app's own nerdit.toml supplies server-side DEFAULTS (the
    CLI reads the file client-side; the dashboard cannot). Precedence: explicit
    form field > ZIP [deploy] value > prior row value (redeploy) > buildpack
    default. Only keys the section actually declares count — the section is
    schema-validated (fail-loud 422) but read as a raw dict so a [deploy] table
    without `gpus` never zeroes a redeploy's GPUs.
    """
    eff_port = port if port is not None else zip_deploy.get("port")
    if eff_port is None and existing is not None:
        eff_port = prev_cfg.get("port")
    eff_start = start if start is not None else zip_deploy.get("start")
    eff_health = health if health is not None else zip_deploy.get("health")
    # (P14 WP-C1) Probe kind is TOML-only in v1 (no form/body field); the ZIP
    # [deploy].health_type value wins, else the prior row's blob type carries
    # forward on a SILENT redeploy so a tcp probe survives. But if THIS request
    # supplies an explicit health path (form field or [deploy].health), the
    # redeploy is not silent about the probe: an explicit path is a request for
    # an http probe, so the prior tcp type must NOT be carried forward and
    # discard it (contract review MINOR-1).
    explicit_health_path = health is not None or "health" in zip_deploy
    eff_health_type = zip_deploy.get("health_type")
    if eff_health_type is None and existing is not None and not explicit_health_path:
        eff_health_type = (existing.health_check or {}).get("type")
    # resource caps have no form/body field (nerdit.toml + config
    # API only). Precedence = ZIP [deploy] value > prior row value > absent.
    eff_memory_limit = zip_deploy.get("memory_limit")
    if eff_memory_limit is None and existing is not None:
        eff_memory_limit = prev_cfg.get("memory_limit")
    eff_cpu_limit = zip_deploy.get("cpu_limit")
    if eff_cpu_limit is None and existing is not None:
        eff_cpu_limit = prev_cfg.get("cpu_limit")
    # P14 WP-A1: named volumes are a list (no form field) — the ZIP [deploy]
    # value wins, else the prior row's list carries forward on a redeploy.
    eff_volumes = zip_deploy.get("volumes")
    if eff_volumes is None and existing is not None:
        eff_volumes = prev_cfg.get("volumes")
    # Fold the implicit `data:/data` in BEFORE validation so the validated list
    # is the list that gets persisted and re-validated at launch: appending it
    # afterwards let an app declaring the cap's worth of volumes persist one
    # over it and fail `core.volumes.resolve_named_volumes` forever. Idempotent
    # (a declared `data` volume re-targets rather than doubling), so the
    # effective declared budget is MAX_VOLUMES - 1, or MAX_VOLUMES when one of
    # them is named `data`. `_ensure_data_volume` still runs downstream for the
    # NERDIT_DATA_DIR half and re-folds a no-op here.
    if isinstance(eff_volumes, list):
        eff_volumes = _resolve_data_volume(eff_volumes)[0]
    # The release command has no form/body field either (nerdit.toml +
    # config API only, like the resource caps): the ZIP/repo [deploy].release
    # wins, else the prior row's value carries forward on a redeploy so a
    # migration gate survives a deploy from a source that stays silent.
    eff_release = zip_deploy.get("release")
    if eff_release is None and existing is not None:
        eff_release = prev_cfg.get("release")
    # `cutover` / `auto_deploy` are next-deploy keys with no form
    # field either — literal twins of `release`: the source's [deploy] value
    # wins, else the prior row's value carries forward, so an opt-out survives a
    # deploy from a source that stays silent. Tri-state: `None` means "not
    # declared" (the daemon default posture), which is why the carry-forward
    # test is `is None` and not falsiness — `False` is a real opt-out.
    eff_cutover = zip_deploy.get("cutover")
    if eff_cutover is None and existing is not None:
        eff_cutover = prev_cfg.get("cutover")
    eff_auto_deploy = zip_deploy.get("auto_deploy")
    if eff_auto_deploy is None and existing is not None:
        eff_auto_deploy = prev_cfg.get("auto_deploy")
    # (P25, D-P25-5) `edge_auth` is the eighth no-form-field [deploy] key and
    # a literal twin of `release`: the source's [deploy] table wins, else the
    # prior row's persisted blob carries forward. The carry-forward is the
    # SECURITY-relevant half here — a redeploy from a source that stays silent
    # (a ZIP with no nerdit.toml, a `nerdit dev` push) must never unprotect a
    # published app. Disarming is explicit only:
    # `nerdit config app set <app> deploy edge_auth=null`.
    eff_edge_auth = zip_deploy.get("edge_auth")
    if eff_edge_auth is None and existing is not None:
        eff_edge_auth = prev_cfg.get("edge_auth")
    if gpus is not None:
        eff_gpus = gpus
    elif "gpus" in zip_deploy:
        eff_gpus = zip_deploy["gpus"]
    elif existing is not None:
        eff_gpus = existing.gpu_count
    else:
        eff_gpus = 0

    deploy_cfg = validate_deploy_fields(
        name,
        eff_port,
        eff_gpus,
        eff_start,
        eff_health,
        eff_memory_limit,
        eff_cpu_limit,
        eff_volumes,
        release=eff_release,
        cutover=eff_cutover,
        auto_deploy=eff_auto_deploy,
        edge_auth=eff_edge_auth,
    )
    # Use the VALIDATED gpus value downstream: DeployConfig coerces lax
    # inputs (e.g. a quoted "2" in the ZIP's nerdit.toml) to int, while the
    # raw dict value would reach the quota arithmetic as a str -> TypeError.
    eff_gpus = deploy_cfg.gpus

    return Effective(
        deploy_cfg=deploy_cfg,
        eff_gpus=eff_gpus,
        eff_start=eff_start,
        eff_health=eff_health,
        eff_health_type=eff_health_type,
    )


def assemble_build_fields(
    *,
    name: str,
    plan: BuildPlan,
    context_dir: Path,
    context_root: Path | None,
    deploy_cfg: DeployConfig,
) -> tuple[dict, int]:
    """Assemble build fields and resolve the container port without filesystem writes.

    Prefer the buildpack port, then the Dockerfile fallback. Persist release without
    arming release_pending: that crash marker belongs immediately before execution,
    not declaration, including when an existing image bypasses the build.
    """
    container_port = plan.port or 8000
    build_fields = {
        "build_context_dir": str(context_dir),
        "dockerfile_name": plan.dockerfile_name,
        "port": container_port,
        "image_repo": app_image_repo(name),
        # Persist the buildpack (dockerfile/node/python) so a later reader
        # has it without re-detecting. Additive config key, no schema change.
        "buildpack": plan.language,
    }
    # Ingress-owned clone ROOT (subsumes a nested subdir context): the
    # controller must remove the WHOLE tree, not just the built subdir.
    if context_root is not None:
        build_fields["build_context_root"] = str(context_root)
    # Dockerfile passthrough: honor --start/[deploy].start as the container
    # command (the generated buildpacks — Node and Python — bake their CMD
    # into the Dockerfile, so they must NOT get an override here). Persisted
    # for both fresh + redeploy
    # via `build_fields` so the controller's _build_command picks it up.
    if plan.dockerfile_text is None and plan.start_command:
        build_fields["command"] = plan.start_command

    # persist the resource caps top-level (the launch reader uses
    # cfg.get("memory_limit")/cfg.get("cpu_limit")). eff_* already carries a
    # prior row value forward on redeploy, so setting only when present both
    # applies a newly-declared value and preserves a prior one — there is no
    # deploy-time null-delete (no form field; config API owns deletion).
    if deploy_cfg.memory_limit is not None:
        build_fields["memory_limit"] = deploy_cfg.memory_limit
    if deploy_cfg.cpu_limit is not None:
        build_fields["cpu_limit"] = deploy_cfg.cpu_limit
    # P14 WP-A1: persist the user-declared named volumes (the validated spec
    # list). `_ensure_data_volume` folds in the implicit `data:/data`
    # after the config blob is assembled, so this carries only the explicit
    # set. Absent ⇒ the implicit volume is still added below.
    if deploy_cfg.volumes is not None:
        build_fields["volumes"] = deploy_cfg.volumes
    # Persist the pre-swap release command (same "set when present"
    # idiom as the resource caps — `eff_release` already carried a prior row
    # value forward, and the config API owns deletion). The `release_pending`
    # crash marker is NOT armed here — see the docstring.
    if deploy_cfg.release is not None:
        build_fields["release"] = deploy_cfg.release
    # Same "set when present" idiom for the two tri-state next-deploy
    # booleans. `is not None` (never truthiness): `cutover = false` is the
    # opt-out and must persist, and `eff_*` already carried a prior row value
    # forward, so an absent key preserves rather than disarms. The config API
    # owns deletion (`deploy cutover=null`).
    if deploy_cfg.cutover is not None:
        build_fields["cutover"] = deploy_cfg.cutover
    if deploy_cfg.auto_deploy is not None:
        build_fields["auto_deploy"] = deploy_cfg.auto_deploy
    # (P25, D-P25-5) Persist the edge-auth declaration top-level under its own
    # literal key: the desired-route query and `core/launch.py` both read
    # `cfg['edge_auth']` verbatim. `model_dump()` because `build_fields`
    # is `json.dumps`'d into `jobs.config` — a pydantic model is not JSON
    # serializable there. The dumped dict holds the `${secrets.KEY}`
    # REFERENCE, never a resolved password (the daemon resolves + bcrypt-hashes
    # at route-build time). Same "set when present" idiom as the rest of the
    # family: `eff_edge_auth` already carried a prior blob forward, and the
    # config API owns deletion (`deploy edge_auth=null`).
    if deploy_cfg.edge_auth is not None:
        build_fields["edge_auth"] = deploy_cfg.edge_auth.model_dump()

    return build_fields, container_port


def classify_binding_action(
    spec: dict | None,
    prior_map: dict | None,
    prior_api: bool,
    existing: Job | None,
) -> tuple[str, list[str]]:
    """The 5-way set/replace/preserve/remove/none classification (P13 WP9 / P15 D-A).

    Shared by both `[ai.*]` and `[db.*]` (a literal mirror — D-A): the
    source declares a spec ⇒ `set` (fresh) / `replace` (redeploy over a
    prior spec); the source is silent AND the prior spec was API-authored
    (`prior_api`) ⇒ `preserve` the carried spec across every subsequent
    silent redeploy; the source is silent with a non-API-authored prior spec ⇒
    `remove` it; otherwise ⇒ `none`.
    """
    if spec:
        action = "replace" if (existing and prior_map) else "set"
        diff_bindings = sorted(spec.keys())
    elif existing and prior_api and prior_map:
        action = "preserve"
        diff_bindings = sorted(prior_map.keys())
    elif existing and prior_map:
        action = "remove"
        diff_bindings = sorted(prior_map.keys())
    else:
        action = "none"
        diff_bindings = []
    return action, diff_bindings


def compute_overwrote_api_config(
    *,
    ai_spec: dict | None,
    db_spec: dict | None,
    prior_ai_api: bool,
    prior_db_api: bool,
    prior_config_api: bool,
    zip_deploy: dict,
    gpus: int | None,
    start: str | None,
    health: str | None,
    effective: Effective,
    existing: Job | None,
    prev_cfg: dict,
) -> bool:
    """Narrowed clobber flag: fires iff the API actually authored what is overwritten.

    Fires for a declared [ai]/[db] over an API-set spec, OR an explicit +
    changed deploy field over an API-authored config.
    """

    def _explicit_changed(form_val, zip_key: str, new_eff, prior_val) -> bool:  # noqa: ANN001
        explicit = form_val is not None or zip_key in zip_deploy
        return bool(explicit and new_eff != prior_val)

    # The API persists a start only under `command` (the app-config deploy
    # PUT). A buildpack row bakes its CMD into the generated Dockerfile and
    # never persists `command`, so a declared [deploy].start there has NO
    # API-authored start to clobber — comparing eff_start against the missing
    # `command` baseline (None) would spuriously flag every identical
    # redeploy. Only count a start change when a `command` baseline exists.
    prior_command = prev_cfg.get("command")
    ai_overwrite = bool(ai_spec) and prior_ai_api
    db_overwrite = bool(db_spec) and prior_db_api
    deploy_overwrite = prior_config_api and (
        _explicit_changed(
            gpus, "gpus", effective.eff_gpus, existing.gpu_count if existing else None
        )
        or (
            prior_command is not None
            and _explicit_changed(start, "start", effective.eff_start, prior_command)
        )
        or _explicit_changed(
            health,
            "health",
            effective.eff_health,
            (existing.health_check or {}).get("path") if existing else None,
        )
        or _explicit_changed(
            None,
            "memory_limit",
            effective.deploy_cfg.memory_limit,
            prev_cfg.get("memory_limit"),
        )
        or _explicit_changed(
            None, "cpu_limit", effective.deploy_cfg.cpu_limit, prev_cfg.get("cpu_limit")
        )
        # release is API-writable (the app-config deploy PUT persists it
        # under the same literal key), so a source that declares its own
        # [deploy].release over an API-authored one is a visible clobber too.
        or _explicit_changed(None, "release", effective.deploy_cfg.release, prev_cfg.get("release"))
        # (P24b, PR #108 review) cutover/auto_deploy are literal twins of
        # release on the same PUT surface — same visibility rule.
        or _explicit_changed(None, "cutover", effective.deploy_cfg.cutover, prev_cfg.get("cutover"))
        or _explicit_changed(
            None, "auto_deploy", effective.deploy_cfg.auto_deploy, prev_cfg.get("auto_deploy")
        )
        # `edge_auth` is API-writable on the same PUT surface, so the
        # same visibility rule applies — and it matters more here than for the
        # rest of the family: a source silently replacing an API-set credential
        # declaration changes who can reach the app. Compared as the dumped
        # dict, which is what the persisted blob holds.
        or _explicit_changed(
            None,
            "edge_auth",
            effective.deploy_cfg.edge_auth.model_dump()
            if effective.deploy_cfg.edge_auth is not None
            else None,
            prev_cfg.get("edge_auth"),
        )
    )
    return ai_overwrite or db_overwrite or deploy_overwrite


def build_plan_body(
    *,
    name: str,
    existing: Job | None,
    plan: BuildPlan,
    container_port: int,
    effective: Effective,
    prev_cfg: dict,
    env: dict[str, str | None] | None,
    ai_action: str,
    ai_bindings: list[str],
    db_action: str,
    db_bindings: list[str],
    overwrote_api_config: bool,
    unknown_deploy_keys: list[str],
) -> dict:
    """Build the dry-run plan after all validators and authorization gates pass.

    Persist nothing; the caller cleans up the context. Include unknown deploy keys
    in the existing warnings channel used by real deployment responses.
    """
    warnings: list[str] = []
    if ai_action == "preserve":
        warnings.append("source declares no [ai]; API-set [ai.*] will be preserved")
    if db_action == "preserve":
        warnings.append("source declares no [db]; API-set [db.*] will be preserved")
    if unknown_deploy_keys:
        warnings.append(_unknown_keys_message(unknown_deploy_keys))
    return {
        "dry_run": True,
        "action": "redeploy" if existing else "create",
        "name": name,
        "buildpack": plan.language,
        "effective": {
            "port": container_port,
            "gpus": effective.eff_gpus,
            "start": plan.start_command,
            "health": effective.eff_health,
            "memory_limit": effective.deploy_cfg.memory_limit,
            "cpu_limit": effective.deploy_cfg.cpu_limit,
            # P14 WP-A1: names + container paths only (never a host
            # path); includes the implicit data:/data retrofit.
            "volumes": _resolve_data_volume(effective.deploy_cfg.volumes)[0],
            # the pre-swap release command this deploy WOULD run
            # (verbatim — it is app-authored config, not a resolved value).
            "release": effective.deploy_cfg.release,
            # the edge-auth this deploy WOULD apply, as
            # `{user, password: "${secrets.KEY}"}`. Value-free BY
            # CONSTRUCTION, not by a filter: `EdgeAuthConfig` rejects a
            # literal password, so what survives validation is a reference —
            # the same names-only discipline `_env_diff` gets structurally
            # (it emits key names, never values). There is no masking pass to
            # route this through, and masking the ref would hide the one thing
            # the operator needs to see before committing to the deploy.
            "edge_auth": (
                effective.deploy_cfg.edge_auth.model_dump()
                if effective.deploy_cfg.edge_auth is not None
                else None
            ),
        },
        "env_diff": _env_diff(dict(prev_cfg.get("env") or {}), env),
        "ai_diff": {"action": ai_action, "bindings": ai_bindings},
        "db_diff": {"action": db_action, "bindings": db_bindings},
        "overwrote_api_config": overwrote_api_config,
        "warnings": warnings,
    }


async def _remove_superseded_context(prev_cfg: dict, build_fields: dict) -> None:
    """Remove a redeploy predecessor's now-unreferenced build context.

    A successful `write_redeploy` overwrites `build_context_dir`/`build_context_root`
    in the row, so the generation it supersedes is no longer named anywhere. The
    controller's builder only cleans the CURRENT row's context (the
    `build_context_root or build_context_dir` it reads back), so a generation
    superseded BEFORE its build ran — two redeploys that both commit with
    sequential versions, neither hitting the 409 — would leak its extracted tree
    under `upload_dir` forever. The §7 live run reproduced exactly this: two
    concurrent ZIP redeploys, both 201, one cutover, and the loser's tree left
    behind with its generated Dockerfile.

    Safe unconditionally: a service container serves from its IMAGE, never the
    context dir, so removing a superseded (or already-built, or image-reused)
    context never touches a running generation, and aborting an in-flight build
    of an already-superseded version only saves work. Best-effort and tolerant —
    the builder's own `cleanup_context` contract — because a leftover tree is a
    disk leak, never a reason to fail a committed deploy.
    """
    prev_root = prev_cfg.get("build_context_root") or prev_cfg.get("build_context_dir")
    if not prev_root:
        return
    new_root = build_fields.get("build_context_root") or build_fields.get("build_context_dir")
    # Ids are unique per extraction/clone, so prev != new always; guard anyway so
    # a degenerate equal-path config can never delete the generation just written.
    if prev_root == new_root:
        return
    try:
        await asyncio.to_thread(shutil.rmtree, prev_root)
    except FileNotFoundError:
        pass  # already gone: built-and-cleaned, or a concurrent build's cleanup
    except OSError:
        logger.warning(
            "Failed to remove superseded build context %s (leaked dir)",
            prev_root,
            exc_info=True,
        )


async def write_redeploy(
    request: Request,
    queries: Queries,
    *,
    name: str,
    existing: Job,
    prev_cfg: dict,
    build_fields: dict,
    effective: Effective,
    env: dict[str, str | None] | None,
    vendor: str | None,
    ai_spec: dict | None,
    db_spec: dict | None,
    prior_ai_api: bool,
    prior_db_api: bool,
    overwrote_api_config: bool,
    source_meta: dict,
) -> Job:
    """Merge build fields into prior config while preserving deployment write order.

    Use max_version for a fresh tag even after rollback lowers build_version. Keep
    source metadata, queued generation and data-volume updates in their stated order.

    Commit through a compare-and-swap on that same max_version, so an overlapping
    deploy that already took the version makes this one a 409 rather than a
    silently discarded write.

    Raises:
        NerditError: 409 `deploy.concurrent_redeploy` when the version was taken.
    """
    expect_max_version = int(prev_cfg.get("max_version", prev_cfg.get("build_version", 0)))
    next_ver = expect_max_version + 1
    config: dict = dict(prev_cfg)
    config.update(build_fields)
    # A stale command from the previous deploy must not survive a
    # buildpack switch (e.g. Dockerfile→Node) or a dropped --start: only
    # keep a command the CURRENT plan supplied.
    if "command" not in build_fields:
        config.pop("command", None)
    # Same for the clone-root marker: a ZIP redeploy over a prior
    # git/template deploy must not carry forward the old (already
    # rmtree'd) root, or the controller would clean up the gone path
    # and leak the new ZIP context. Only keep a root the CURRENT
    # ingress stamped.
    if "build_context_root" not in build_fields:
        config.pop("build_context_root", None)
    # And the cutover crash marker: it names the PREVIOUS generation's
    # verify, which this write supersedes. It is inert for the bumped version
    # (layer 4 matches on `version == build_version`) but not harmless: a
    # later rollback lowers `build_version` back onto an already-used number,
    # and a marker carried this far would then re-match and settle a generation
    # that ran no cutover at all. The in-flight 409 above means anything still
    # here belongs to an abandoned or already-settled verify.
    config.pop("cutover_pending", None)
    config["image"] = app_image_tag(name, next_ver)
    config["build_version"] = next_ver
    config["max_version"] = next_ver
    if prev_cfg.get("image"):
        config["previous_image"] = prev_cfg["image"]  # rollback target
    # env: request keys merge/replace over the previous env; a null value
    # deletes the key; else keep the previous env untouched.
    if env:
        merged = dict(prev_cfg.get("env") or {})
        merged.update(env)
        for ekey, eval_ in env.items():
            if eval_ is None:
                merged.pop(ekey, None)
        config["env"] = merged
    # vendor: an explicit value overrides; else the prev value is kept
    # (already carried over by the dict(prev_cfg) copy above).
    if vendor:
        config["vendor"] = vendor
    # [ai.*]: the source declares [ai] ⇒ wholesale REPLACE +
    # drop the API-provenance marker; the source is silent AND the spec
    # was API-authored (`ai_source == "api"`) ⇒ PRESERVE the carried
    # spec + marker across every subsequent silent redeploy (the
    # dict(prev_cfg) copy already carried both — do nothing); else the
    # source is silent with no marker ⇒ drop any carried spec (unchanged).
    if ai_spec:
        config["ai"] = ai_spec
        config.pop("ai_source", None)
    elif prior_ai_api:
        pass  # preserve: copy already carried config['ai'] + config['ai_source']
    else:
        config.pop("ai", None)
    # [db.*]: identical preserve rule keyed on `db_source`.
    if db_spec:
        config["db"] = db_spec
        config.pop("db_source", None)
    elif prior_db_api:
        pass  # preserve: copy already carried config['db'] + config['db_source']
    else:
        config.pop("db", None)
    # stamp writer provenance. A redeploy over API-set config is
    # last-writer-wins; the clobber is made VISIBLE (response flag +
    # audit param) only when the API actually authored what was
    # overwritten (P13 WP9 narrowed rule, computed above).
    config["config_source"] = "deploy"
    config["config_revision"] = int(prev_cfg.get("config_revision", 0)) + 1
    if overwrote_api_config:
        request.state.audit_params["overwrote_api_config"] = True
    # Provenance stamp: assigned AFTER the dict(prev_cfg) copy +
    # config.update(build_fields) so a redeploy REPLACES the carried-over
    # source wholesale (e.g. a ZIP redeploy over a git deploy → zip).
    config["source"] = source_meta
    # Seed the new deploy generation's phase object (also clears
    # the previous generation's crash forensics carried by dict(prev_cfg)).
    _stamp_queued(config, version=next_ver, action="redeploy")
    # (P14 WP-A1) Retrofit the implicit data volume + NERDIT_DATA_DIR —
    # applies to pre-P14 rows on this redeploy (D-P14-4). Runs AFTER the
    # env merge so a user's explicit NERDIT_DATA_DIR still wins. The prior
    # implicit data path lets the wedge re-target its own persisted value
    # (never leaving a stale NERDIT_DATA_DIR when the mount moves).
    prev_data_path = _resolve_data_volume(prev_cfg.get("volumes"))[1]
    _ensure_data_volume(config, prev_data_path=prev_data_path)

    # Resolve the dedicated columns too (they don't live in the config
    # blob): --health/--gpus must actually take effect on a redeploy.
    resolved_health = (
        _build_health_blob(effective.eff_health, effective.eff_health_type) or existing.health_check
    )
    # Redeploy: reuse the row (stable id + endpoint). `restarting` keeps
    # the old container serving until the controller builds + swaps. The
    # GPU quota is charged to the service OWNER (not the actor), so an
    # admin redeploying does not launder another token's quota.
    try:
        committed = await queries.update_service_config_guarded(
            existing.id,
            json.dumps(config),
            expect_max_version=expect_max_version,
            expect_cutover_pending=prev_cfg.get("cutover_pending") is not None,
            status=JobStatus.restarting,
            desired_state="running",
            gpu_count=effective.eff_gpus,
            health_check=resolved_health,
            token_id=existing.submitted_by_token,
        )
    except QuotaExceeded as exc:
        raise exc.to_error() from exc
    if not committed:
        # Another deploy allocated this version — or a cutover armed its marker
        # under us — between the config read far upstream and this write.
        # Refusing is the only honest answer: the blob above names an image tag
        # the winner already claimed, so committing it would hand this caller a
        # 201 for code that is never built and silently discard one of the two
        # deploys. The caller's extracted tree is removed by
        # `_finalize_deploy`'s failure path.
        raise NerditError(
            409,
            "deploy.concurrent_redeploy",
            f"Another deploy of '{name}' committed while this one was being prepared.",
            hint="Wait for the in-flight deploy or cutover to settle, then deploy again.",
        )
    # This write overwrote build_context_dir/build_context_root, orphaning the
    # generation it superseded. Remove that generation's context so a redeploy
    # landing before its predecessor's build ran does not leak the extracted tree
    # (the CAS above prevents a duplicate version; this prevents the tree leak on
    # the sequential-version path the CAS lets through).
    await _remove_superseded_context(prev_cfg, build_fields)
    return await queries.get_service_by_name(name) or existing


async def write_fresh(
    request: Request,
    queries: Queries,
    *,
    name: str,
    build_fields: dict,
    effective: Effective,
    env: dict[str, str | None] | None,
    vendor: str | None,
    ai_spec: dict | None,
    db_spec: dict | None,
    source_meta: dict,
    principal: Principal,
    owner_token_id: str | None = None,
) -> Job:
    """Fresh deploy: assemble a brand-new config blob + reserve the row.

    `owner_token_id` overrides who the fresh row belongs to. Default `None`
    keeps every existing ingress byte-for-byte (the row is the acting
    principal's); the workspace ingress passes the sidecar owner so an admin's
    fresh deploy does not orphan somebody else's workspace. Quota follows the
    same value (`reserve_service_for_token` charges `submitted_by_token`),
    which is the point: an admin cannot launder a submitter's quota.
    """
    config = {
        "image": app_image_tag(name, 1),
        "build_version": 1,
        "max_version": 1,
        "config_source": "deploy",
        "config_revision": 1,
        "source": source_meta,  # deploy provenance
        **build_fields,
    }
    # env: drop null-valued keys — a delete has nothing to
    # act on for a fresh deploy, so it never lands in the blob.
    if env:
        cleaned_env = {k: v for k, v in env.items() if v is not None}
        if cleaned_env:
            config["env"] = cleaned_env
    if vendor:
        config["vendor"] = vendor
    if ai_spec:
        config["ai"] = ai_spec
    if db_spec:
        config["db"] = db_spec
    # Seed the phase object for the first deploy generation.
    _stamp_queued(config, version=1, action="create")
    # (P14 WP-A1) Every fresh deploy gets the implicit data volume +
    # NERDIT_DATA_DIR (D-P14-4).
    _ensure_data_volume(config)
    job = Job(
        kind=JobKind.service,
        service_name=name,
        name=name,
        gpu_count=effective.eff_gpus,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        health_check=_build_health_blob(effective.eff_health, effective.eff_health_type),
        config=json.dumps(config),
        submitted_by_token=owner_token_id if owner_token_id is not None else principal.token_id,
        idempotency_key=request.headers.get("Idempotency-Key"),
    )
    try:
        return await queries.reserve_service_for_token(job)
    except ServiceNameTaken as exc:  # lost a create race
        raise NerditError(
            409,
            "service.name_taken",
            f"A service named '{name}' already exists.",
            hint="Delete it first with `nerdit services rm`.",
        ) from exc
    except QuotaExceeded as exc:
        raise exc.to_error() from exc


def _deploy_summary(body: dict) -> dict:
    """Copy the four summary fields from the assembled response without recomputing.

    Fresh deployments have no endpoint yet: public_url is null and status reflects
    pre-launch state.
    """
    endpoint = body.get("endpoint") or {}
    return {
        "app": body.get("name"),
        "status": body.get("status"),
        "version": body.get("build_version"),
        "public_url": endpoint.get("public_url"),
    }


def _deploy_hints(
    request: Request,
    name: str,
    unknown_deploy_keys: list[str],
    *,
    failure_detail: str | None = None,
) -> list[str]:
    """Build ordered success advisories, never an error channel.

    Failure detail, when present, leads; ignored input precedes routing advice, and
    an async-progress sentence closes every list. Use only sanitized, bounded key
    names and validated service names, never values. Base-path advice requires live
    proxy availability and path mode, not an endpoint that fresh deploys lack.
    """
    hints: list[str] = []
    if failure_detail:
        hints.append(_FAILED_STATE_HINT.format(name=name, detail=failure_detail))
    if unknown_deploy_keys:
        hints.append(_unknown_keys_message(unknown_deploy_keys))
    settings = getattr(request.app.state, "settings", None)
    proxy = getattr(settings, "proxy", None) if settings else None
    proxy_mgr = getattr(request.app.state, "proxy_manager", None)
    if getattr(proxy, "mode", None) == "path" and bool(getattr(proxy_mgr, "available", False)):
        hints.append(_PATH_MODE_HINT.format(name=name))
    hints.append(_ASYNC_DEPLOY_WHY)
    return hints


def _failure_detail(job: Job) -> str | None:
    """A machine token naming why the row reads as failed, or `None` if it does not.

    Row columns first (the terminal-status fact), then the phase object's
    `reason` — the same precedence `/diagnose` uses. Only vocabulary the
    daemon itself writes is ever returned: a `JobStatus` value or a
    `last_deploy.reason` token, never `error_message` (app-authored text
    must not reach the hints channel).
    """
    status = getattr(job, "status", None)
    if status is JobStatus.failed:
        return JobStatus.failed.value
    cfg = parse_job_config(job)
    ld = cfg.get("last_deploy")
    if isinstance(ld, dict) and ld.get("phase") == "failed":
        reason = ld.get("reason")
        return str(reason) if isinstance(reason, str) and reason.isascii() else "failed"
    return None


def _request_build_settings(requested: dict | None, start: str | None) -> dict | None:
    if start is None:
        return requested
    return {"start": start, **(requested or {})}


def _merge_build_settings(
    repository: dict | None, saved: dict | None, requested: dict | None
) -> tuple[BuildSettings, dict, dict]:
    """Keep only explicit request overrides; null resets a key to repository defaults."""
    try:
        repo = BuildSettings.model_validate(repository or {}).model_dump(exclude_none=True)
        patch = BuildSettings.model_validate(requested or {}).model_dump(exclude_unset=True)
        # Reset/replacement may remove a previously accepted unsafe command, but
        # malformed saved shapes and unknown fields must still be rejected.
        if saved is not None and not isinstance(saved, dict):
            BuildSettings.model_validate(saved)
        overrides = dict(saved or {})
        for key, value in patch.items():
            if value is None:
                overrides.pop(key, None)
            else:
                overrides[key] = value
        overrides = BuildSettings.model_validate(overrides).model_dump(exclude_none=True)
    except ValidationError:
        raise NerditError(
            422,
            "deploy.invalid_build_settings",
            "Invalid build_settings. Use supported commands, runtime, manager, relative subdir "
            "and public_env values; secret inputs and secret references are not supported.",
        ) from None
    origins = {key: "config" for key in repo}
    origins.update({key: "request" if key in patch else "saved" for key in overrides})
    return BuildSettings.model_validate({**repo, **overrides}), overrides, origins


def _build_context(context: Path, subdir: str | None) -> Path:
    """Select a directory inside the ingress-owned tree, never a host path."""
    if not subdir:
        return context
    selected = (context / subdir).resolve()
    if not selected.is_relative_to(context.resolve()) or not selected.is_dir():
        raise NerditError(
            422,
            "deploy.invalid_build_settings",
            "build_settings.subdir must select an existing directory inside the project.",
        )
    return selected


def _build_preview(
    plan: BuildPlan,
    settings: BuildSettings,
    origins: dict,
    source: dict,
    *,
    legacy_start: str | None = None,
    request_start: str | None = None,
) -> dict:
    if settings.start is None and legacy_start:
        origins = {**origins, "start": "request" if request_start is not None else "config"}
    if plan.language == "dockerfile":
        origins = {
            **origins,
            **{
                key: "dockerfile"
                for key in ("preset", "install", "build", "node_version", "package_manager")
            },
        }
    root = "/".join(part for part in (source.get("subdir"), settings.subdir) if part) or "."
    return {
        "preset": settings.preset,
        "framework": plan.framework or plan.language,
        "node_version": plan.node_version,
        "package_manager": plan.package_manager,
        "install": plan.install_command,
        "build": False
        if settings.build is False and plan.language != "dockerfile"
        else plan.build_command,
        "start": plan.effective_start_command or plan.start_command,
        "subdir": root,
        "commit_sha": source.get("commit_sha"),
        # Deliberately in clear: these values are compiled into public build
        # output (a browser bundle), and secret references are refused by
        # `BuildSettings`, so masking them would hide the only thing an
        # operator can check before the deploy.
        "public_env": dict(settings.public_env or {}),
        "sources": {key: origins.get(key, "detected") for key in BuildSettings.model_fields},
        "warnings": list(plan.warnings),
    }


async def _finalize_deploy(
    request: Request,
    context_dir: Path,
    *,
    name: str,
    port: int | None,
    gpus: int | None,
    start: str | None,
    health: str | None,
    env: dict[str, str | None] | None,
    vendor: str | None,
    source_meta: dict,
    context_root: Path | None = None,
    owner_token_id: str | None = None,
    build_settings: dict | None = None,
    dry_run: bool = False,
) -> dict:
    """Validate a prepared build context and create or update its service row.

    Callers must authorize, validate syntax and reserved names, and set audit
    parameters before preparing the context. Re-read and re-authorize existing
    rows here: another request may create the name during extraction or cloning.

    Replace config['source'] with source_meta on every deployment. context_root,
    when supplied, owns the entire extracted/cloned tree and is the cleanup target;
    otherwise clean up context_dir. Persist that root for subsequent build cleanup.

    owner_token_id overrides ownership only for fresh rows (workspace deployments
    retain their sidecar owner). Redeploys always preserve the recorded owner.
    """
    principal = current_principal(request)
    queries = request.app.state.queries

    context_root = context_root or context_dir

    # Everything past extraction must clean up the context dir on failure —
    # including the row re-read below (a DB error must not leak the tree).
    try:
        # Belt-and-braces: runs BEFORE _parse_deploy_defaults so a bad form
        # name never mis-attributes to the app's nerdit.toml (PROBE-11).
        _validate_request_name(name)
        existing = await queries.get_service_by_name(name)
        # A name owned by a live kind=model/database row is not a redeploy
        # target — the create-race security boundary (ingresses are fast paths).
        reject_non_service_row(existing, name)
        # Re-authorize on the SAME read the fresh-vs-redeploy branch uses: a
        # name created during the extraction/clone window must not slip onto
        # the redeploy path without an owner check (the caller's pre-ingress
        # gate is only a resource-saving optimization, not the security one).
        if existing is not None:
            require_owner_or_admin(request, existing)
        prev_cfg = parse_job_config(existing) if existing else {}
        # (P34 / D) Captured BEFORE the redeploy write overwrites the phase
        # object: "the generation you are replacing was failing" is exactly the
        # context a caller redeploying a crash-loop needs, and one line later it
        # is gone. `None` on a fresh deploy — there is no history to report.
        prev_failure = _failure_detail(existing) if existing is not None else None

        project, project_error = _read_project_toml(context_dir)
        zip_deploy, unknown_deploy_keys = _parse_deploy_defaults(project, name)
        # Legacy explicit start is also a request override; nested membership wins,
        # including null reset. Clients must not promote repository defaults here.
        build_settings = _request_build_settings(build_settings, start)
        start = None if build_settings and "start" in build_settings else start
        settings, overrides, origins = _merge_build_settings(
            zip_deploy.get("build_settings"), prev_cfg.get("build_overrides"), build_settings
        )
        selected = _build_context(context_dir, settings.subdir)
        if selected != context_dir:
            root_settings = zip_deploy.get("build_settings") or {}
            context_dir = selected
            nested, nested_error = _read_project_toml(context_dir)
            project_error = project_error or nested_error
            # Preserve root auth, bindings and deployment defaults. The selected
            # app may explicitly replace sections or override deploy keys.
            nested_deploy, _ = _parse_deploy_defaults(nested, name)
            project = {**project, **nested, "deploy": {**zip_deploy, **nested_deploy}}
            zip_deploy, unknown_deploy_keys = _parse_deploy_defaults(project, name)
            # Root selection happens once; nested config cannot recursively escape it.
            repository = {**root_settings, **(zip_deploy.get("build_settings") or {})}
            repository["subdir"] = settings.subdir
            settings, overrides, origins = _merge_build_settings(
                repository, prev_cfg.get("build_overrides"), build_settings
            )
        effective = resolve_effective_fields(
            name=name,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            zip_deploy=zip_deploy,
            existing=existing,
            prev_cfg=prev_cfg,
        )
        deploy_cfg = effective.deploy_cfg
        deploy_cfg.build_settings = settings

        # Detect a buildpack and materialize the generated Dockerfile (if any).
        try:
            plan = detect(context_dir, deploy_cfg)
        except BuildpackNotSupported as exc:
            raise NerditError(
                400,
                "deploy.no_buildpack",
                str(exc),
                hint=(
                    "Choose a compatible preset or update the repository configuration. "
                    'To clear a saved preset override, send build_settings={"preset": null}; '
                    "this restores repository/default detection."
                    if settings.preset is not None
                    else "Add a Dockerfile, a package.json (Node), or a "
                    "requirements.txt / pyproject.toml (Python)."
                ),
            ) from exc

        # P5 (S8): parse+gate [ai.*] server-side on a served-model row; only
        # the SPEC persists (config['ai']) — resolved to OPENAI_* at launch.
        if project_error is not None:
            raise NerditError(
                422,
                "deploy.invalid_ai",
                f"The app's {PROJECT_CONFIG_NAME} is not valid TOML: {project_error}",
                hint=AI_SHAPE_HINT,
            ) from project_error
        ai_bindings = (
            validate_ai_section(project["ai"], source=PROJECT_CONFIG_NAME)
            if "ai" in project
            else None
        )
        if ai_bindings:
            await require_served_models(queries, ai_bindings)
        ai_spec = (
            {bname: b.model_dump(exclude_none=True) for bname, b in ai_bindings.items()}
            if ai_bindings
            else None
        )
        if ai_spec:
            # audit_params masks api_key leaves — even secret refs, verbatim —
            # but leaf-name masking cannot see a credential embedded IN a value,
            # and `validate_ai_section` runs no userinfo check on `base_url`.
            # A `https://user:pw@host/v1` therefore reached the audit row and
            # the admin `audit.*` frame: the M7 hazard, on the other binding.
            request.state.audit_params["ai"] = audit_params(_audit_safe_ai(ai_spec))

        # identical [db.*] gate (D-A mirror) on a provisioned database
        # row; only the SPEC persists — the credential DSN resolves at launch.
        db_bindings = (
            validate_db_section(project["db"], source=PROJECT_CONFIG_NAME)
            if "db" in project
            else None
        )
        if db_bindings:
            await require_provisioned_databases(
                queries, db_bindings, request, previous=prev_cfg.get("db")
            )
        db_spec = (
            {bname: b.model_dump(exclude_none=True) for bname, b in db_bindings.items()}
            if db_bindings
            else None
        )
        if db_spec:
            # audit_params masks password leaves — a resolved DSN never a param.
            request.state.audit_params["db"] = audit_params(db_spec)

        build_fields, container_port = assemble_build_fields(
            name=name,
            plan=plan,
            context_dir=context_dir,
            context_root=context_root,
            deploy_cfg=deploy_cfg,
        )

        preview = _build_preview(
            plan, settings, origins, source_meta, legacy_start=deploy_cfg.start, request_start=start
        )
        build_fields["build_overrides"] = overrides
        build_fields["build_plan"] = preview
        # Written on EVERY deploy, empty map included: `write_redeploy` carries
        # unlisted build keys forward, so an omitted key would resurrect a map
        # the caller just removed.
        build_fields["public_env"] = dict(settings.public_env or {})

        effective.eff_start = plan.start_command

        # classify the [ai.*] preserve action + the NARROWED
        # overwrote_api_config flag once, up front — the write branches below
        # and the dry_run preview both consume it (prev_cfg={} for a fresh
        # deploy folds every prior-config check to False).
        prior_ai = prev_cfg.get("ai") if isinstance(prev_cfg.get("ai"), dict) else None
        prior_ai_api = prev_cfg.get("ai_source") == "api"
        prior_config_api = prev_cfg.get("config_source") == "api"
        ai_action, ai_diff_bindings = classify_binding_action(
            ai_spec, prior_ai, prior_ai_api, existing
        )

        # P15 mirror (D-A) for [db.*], keyed on the db_source == "api" marker.
        prior_db = prev_cfg.get("db") if isinstance(prev_cfg.get("db"), dict) else None
        prior_db_api = prev_cfg.get("db_source") == "api"
        db_action, db_diff_bindings = classify_binding_action(
            db_spec, prior_db, prior_db_api, existing
        )

        overwrote_api_config = compute_overwrote_api_config(
            ai_spec=ai_spec,
            db_spec=db_spec,
            prior_ai_api=prior_ai_api,
            prior_db_api=prior_db_api,
            prior_config_api=prior_config_api,
            zip_deploy=zip_deploy,
            gpus=gpus,
            start=settings.start or ("" if build_settings and "start" in build_settings else start),
            health=health,
            effective=effective,
            existing=existing,
            prev_cfg=prev_cfg,
        )

        if dry_run:
            # Every validator has run but nothing is persisted; the route
            # wraps this in a 200 (the decorator's 201 is for the real path).
            body = build_plan_body(
                name=name,
                existing=existing,
                plan=plan,
                container_port=container_port,
                effective=effective,
                prev_cfg=prev_cfg,
                env=env,
                ai_action=ai_action,
                ai_bindings=ai_diff_bindings,
                db_action=db_action,
                db_bindings=db_diff_bindings,
                overwrote_api_config=overwrote_api_config,
                unknown_deploy_keys=unknown_deploy_keys,
            )
            body["build"] = preview
            body["warnings"].extend(preview["warnings"])
            cleanup_root = context_root or context_dir
            await asyncio.to_thread(shutil.rmtree, cleanup_root, ignore_errors=True)
            return body

        # Materialize the generated Dockerfile — AFTER the dry-run return, which
        # promises zero writes, and before the row is written so the off-tick
        # build finds it.
        if plan.dockerfile_text is not None:
            _write_generated_file(context_dir / plan.dockerfile_name, plan.dockerfile_text)
            # mirror the ZIP-upload excludes into a generated
            # .dockerignore (buildpack-generated Dockerfiles only; never
            # overwrites a user-authored file).
            _write_generated_dockerignore(context_dir)

        if existing:
            # A plain redeploy landing mid-cutover is the same
            # hazard the rollback route already refuses one layer up: the write
            # below bumps `build_version`, so the in-flight verify's own
            # CAS-guarded arm/pop/settle all miss, its marker is orphaned and
            # its green keeps running as a rowless container behind the new
            # generation. The dry-run branch returned above, so this is
            # writes-path only.
            controller = getattr(request.app.state, "service_controller", None)
            if controller is not None and controller.has_active_cutover(existing.id):
                raise _cutover_in_progress_error(name, "deploy over it")
            job = await write_redeploy(
                request,
                queries,
                name=name,
                existing=existing,
                prev_cfg=prev_cfg,
                build_fields=build_fields,
                effective=effective,
                env=env,
                vendor=vendor,
                ai_spec=ai_spec,
                db_spec=db_spec,
                prior_ai_api=prior_ai_api,
                prior_db_api=prior_db_api,
                overwrote_api_config=overwrote_api_config,
                source_meta=source_meta,
            )
        else:
            job = await write_fresh(
                request,
                queries,
                name=name,
                build_fields=build_fields,
                effective=effective,
                env=env,
                vendor=vendor,
                ai_spec=ai_spec,
                db_spec=db_spec,
                source_meta=source_meta,
                principal=principal,
                owner_token_id=owner_token_id,
            )

        # One `service.deploy_started` per generation, at the only
        # seam every ingress (ZIP / git / template store) shares — after the row
        # is written, so `build_version` is the generation just stamped by
        # `_stamp_queued`. A dry run returned above and never reaches here.
        # `get_recorder()` — the same object installed at the same startup
        # moment, and the ONE access idiom every emit site outside a controller
        # uses (`core/deploy_state.py` already did). `None` outside a running
        # daemon makes the emit inert exactly as before.
        await record_job_event(
            get_recorder(),
            "service.deploy_started",
            job,
            data={"action": "redeploy" if existing else "create"},
        )

        # write-time shared-ref visibility (secret.shared_referenced),
        # audited to the REQUESTING principal — the admin's before-launch
        # signal, on top of the launch-time secret.shared_resolved row.
        shared_keys = sorted(
            set(shared_secret_keys(ai_spec)) | set(shared_secret_keys_for(db_spec, ("password",)))
        )
        if shared_keys:
            await record_shared_referenced(request, name, shared_keys)

        endpoint = await queries.get_service_endpoint(name)
        gpu_ids = await queries.get_job_gpus(job.id)
        hosted = await load_hosted_context(request)
        resp = _service_response(request, job, gpu_ids, endpoint, hosted=hosted)
        body = resp.model_dump(mode="json")
        return {
            **body,
            "build": preview,
            # Additive P7 field: agents detect an API-config clobber on redeploy.
            "overwrote_api_config": overwrote_api_config,
            # (Agent-DX) The at-a-glance block + the ordered advisory channel,
            # added at the ONE tail every ingress shares (ZIP / git / redeploy /
            # template / workspace), so all five carry them from one edit.
            # `summary` is PROJECTED from `body` — never recomputed.
            "summary": _deploy_summary(body),
            "hints": _deploy_hints(request, name, unknown_deploy_keys, failure_detail=prev_failure),
            # The structured half of the same statement. Machine-shaped so
            # an agent branches on it instead of parsing `hints` prose.
            "next_step": _next_step(name),
        }
    except BaseException:
        # Cleanup on any failure past extraction: the wider clone root when
        # the ingress owns one, else the context dir itself (ZIP path).
        cleanup_root = context_root or context_dir
        await asyncio.to_thread(shutil.rmtree, cleanup_root, ignore_errors=True)
        raise


# --- P24b WP6: redeploy from the row's recorded git source --------------------
#
# `POST /deploy/{name}/redeploy` (and, in P24c, the GitWatch poller) carry no
# deploy coordinates at all: every input is read back off `config['source']`,
# which was stamped by the original `POST /deploy/git`. That makes the guard
# set below load-bearing rather than redundant — the recorded URL/ref/subdir are
# re-validated against the CURRENT `[git].allowed_hosts` so an allowlist
# narrowed since the original deploy bites on the redeploy too.


def _source_credential_error(name: str, detail: str) -> NerditError:
    """The one `deploy.no_source_credential` shape (D-P24-9 / D-P24-14)."""
    return NerditError(
        409,
        "deploy.no_source_credential",
        detail,
        hint=(
            # `--name <name>`, never a positional: the CLI rejects a
            # positional path alongside --repo, so the obvious
            # `nerdit deploy <name> --repo ...` reads as a folder deploy and
            # errors out before it reaches the daemon.
            f"Re-run `nerdit deploy --name {name} --repo <url> "
            "--token-ref '${secrets.KEY}'` once to record the reference."
        ),
    )


#: What a submitter can do about a private repository: nothing alone. Its
#: role can neither install the GitHub App nor write the shared scope, so the
#: hint names the person who can and the exact reference to pass afterwards
#: (audit A22). Admin wording is untouched: the poller and CLI key off status
#: and code; neither depends on a subscription.
_SUBMITTER_PRIVATE_REPO_HINT = (
    "this token cannot install the Nerdit GitHub App or write a shared secret — "
    "ask the node owner to install the app on this repository, or to run "
    "`nerdit secrets set --shared --prompt GITHUB_TOKEN`, then pass "
    "token_ref '${secrets.shared.GITHUB_TOKEN}'"
)

#: The clone-failure twin keeps the stock "check the URL" action (a typo and a
#: private repository fail identically) and names no host and no key: the
#: repository may live on any `[git].allowed_hosts` entry, so a GitHub-named
#: secret must not be steered at a GitLab or Gitea server.
_SUBMITTER_CLONE_HINT = (
    "repository not found or private — verify the URL is correct and the repo is "
    "public; for a private repository this token cannot write a shared secret, so "
    "ask the node owner to run `nerdit secrets set --shared --prompt KEY`, then pass "
    "token_ref '${secrets.shared.KEY}'"
)


def github_token_absent_error(*, role: TokenRole | None = None) -> NerditError:
    """Build the explicit-action GitHub-token absence error.

    GitWatch treats this exact 422/code pair as quiet backoff. The caller's role
    changes only the hint; status, code and detail remain stable. Public-share
    eligibility does not determine repository authorization.
    """
    if role is TokenRole.submitter:
        hint = _SUBMITTER_PRIVATE_REPO_HINT
    else:
        hint = (
            "link this node and install the Nerdit GitHub App, or pass a `${secrets.*}` token_ref"
        )
    return NerditError(
        422,
        "deploy.github_token_absent",
        "token_ref '${github.installation}' resolves to no installation token for this repository.",
        hint=hint,
    )


def private_repo_hint(exc: GitSourceError, *, role: TokenRole, had_token: bool) -> str | None:
    """The clone-failure hint a caller can act on, by role.

    An unauthenticated clone that hit an auth challenge is a private (or
    missing) repository. The stock hint tells the caller to store a token;
    a submitter cannot, so it gets the owner-facing, host-neutral sentence.
    """
    if role is TokenRole.submitter and not had_token and _is_auth_failure(exc):
        return _SUBMITTER_CLONE_HINT
    return exc.hint


def resolve_github_installation_token(app: Any, repo_url: str) -> str | None:
    """Resolve a live installation token by the repository's owner/name.

    Missing link state, unmatched repos, expiry and non-GitHub hosts return None.
    Enforce github.com here where credentials leave custody; alternate configured
    hosts require the explicit NERDIT_DEV_GITHUB_CLONE_BASE test guard. Raw tokens
    stay local and reach only GIT_ASKPASS.
    """
    manager = getattr(app.state, "link_manager", None)
    if manager is None:
        return None
    settings = getattr(app.state, "settings", None)
    github_host = settings.git.github_host if settings is not None else "github.com"
    if not installation_token_allowed_for_host(github_host):
        return None
    slug = github_repo_slug(repo_url, github_host)
    if slug is None:
        return None
    token = manager.github_token_for_repo(slug)
    return token if isinstance(token, str) and token else None


async def _resolve_source_token(
    request: Request, secrets: Any, name: str, token_ref: str, *, repo_url: str
) -> str:
    """Resolve a recorded `token_ref` to its raw value for a re-clone.

    `${github.installation}` never touches the secret store:
    it resolves through the link manager's mirror by repo, and its absence is
    the loud `422 deploy.github_token_absent` (D-GH-9 — the poller maps
    that one code back to a quiet backoff).

    The redeploy path always runs AFTER the owner-or-admin gate on an existing
    row, so the per-service scope is provably the caller's — the fresh-deploy
    `service_owned` carve-out in `routes/deploy.py::_resolve_token_ref` has
    nothing to protect here. Precedence is the launch-path one: per-service
    scope first, then shared. The raw value lives only as a local — never in
    audit params, config, logs, argv, or the response.
    """
    if token_ref == GITHUB_INSTALLATION_REF:
        token = resolve_github_installation_token(request.app, repo_url)
        if token is None:
            raise github_token_absent_error(role=current_principal(request).role)
        return token
    if secrets is None:  # pragma: no cover - always wired in the daemon
        raise NerditError(500, "internal", "Secret manager is not configured.")
    try:
        # A grammar miss short-circuits inside the walk before either scope is
        # loaded, so an unusable reference is still reported ahead of any
        # decrypt failure.
        res = walk_secret_ref(
            token_ref,
            service_env=lambda: secrets.load(name),
            shared_env=lambda: secrets.load(SHARED_SCOPE),
        )
    except SecretDecryptError as exc:
        raise NerditError(500, "secret.decrypt_failed", str(exc)) from exc
    if not res.matched:
        # A persisted reference that no longer parses is unusable, and there is
        # no caller input to correct — same remediation as a missing one.
        raise _source_credential_error(
            name, f"Service '{name}' records an unusable source credential reference."
        )
    if res.value is not None:
        if res.source == "shared":
            assert res.key is not None  # a resolved value implies a parsed key
            await record_shared_referenced(request, name, [res.key])
        return res.value
    raise _source_credential_error(
        name,
        f"The recorded source credential for '{name}' no longer resolves to a stored secret.",
    )


def _is_auth_failure(exc: GitSourceError) -> bool:
    """Whether a failed clone looks like a private/missing repo auth challenge."""
    if exc.code != "deploy.git_clone_failed":
        return False
    lowered = exc.message.lower()
    return any(sig in lowered for sig in _AUTH_SIGNATURES)


def _read_git_source(job: Job, name: str) -> dict:
    """Return the row's `config['source']` when it is a usable git provenance.

    The single redeploy-source refusal for BOTH callers: the
    `POST /deploy/{name}/redeploy` route (which calls it as a pure gate,
    discarding the result) and the GitWatch poller. Everything it raises is a
    `409 deploy.no_source`.
    """
    source = parse_job_config(job).get("source")
    if not isinstance(source, dict) or source.get("type") != "git":
        # A workspace app has a source, just not a re-clonable one:
        # its files live in the daemon's own workspace tree, so the redeploy verb
        # for it is POST /api/workspaces/{name}/deploy.
        if isinstance(source, dict) and source.get("type") == "workspace":
            raise NerditError(
                409,
                "deploy.no_source",
                f"Service '{name}' deploys from its workspace.",
                hint=(
                    "This app deploys from its workspace — call deploy_app "
                    f"(POST /api/workspaces/{name}/deploy)."
                ),
            )
        raise NerditError(
            409,
            "deploy.no_source",
            f"Service '{name}' was not deployed from a git source.",
            hint="Redeploy is git-only; re-upload with `nerdit deploy <path>` instead.",
        )
    repo_url = source.get("repo_url")
    if not isinstance(repo_url, str) or not repo_url:
        raise NerditError(
            409,
            "deploy.no_source",
            f"Service '{name}' records a git source with no repository URL.",
            hint="Re-run `nerdit deploy --repo <url>` once to record the source.",
        )
    return source


#: The identity an unattended GitWatch redeploy runs as (P24c / WP10). Admin so
#: `require_owner_or_admin` passes on any owner's row — the poller acts for the
#: daemon, not for whoever last deployed the service — and `token_id="system"`
#: so every row it writes is attributable, matching `principal='system'` on the
#: controller-authored audit rows.
_GITWATCH_PRINCIPAL = Principal(token_id="system", name="gitwatch", role=TokenRole.admin)


class _SystemRequest:
    """Supply GitWatch's system principal, app state and empty headers to redeploy.

    Discard request audit_params; the poller writes its own deploy.auto_redeploy row.
    No HTTP request means no Idempotency-Key or request ID.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.state = SimpleNamespace(
            principal=_GITWATCH_PRINCIPAL, audit_params={}, request_id=None
        )
        self.headers: dict[str, str] = {}


async def redeploy_from_source(
    *,
    request_or_none: Request | None,
    queries: Queries,
    settings: Any,
    secrets: Any,
    job: Job,
    principal: str,
    dry_run: bool = False,
    app: Any | None = None,
) -> dict:
    """Re-clone a service's recorded git source and run the shared deploy tail.

    The one code path behind `POST /deploy/{name}/redeploy` and
    the GitWatch poller. Reads `config['source']` off a FRESH row read,
    re-validates the recorded URL/ref/subdir against the current
    `[git].allowed_hosts`, resolves any recorded `token_ref` through the
    `SecretManager`, clones, and hands the context to
    `_finalize_deploy`. No deploy coordinates are accepted from the
    caller: `port`/`gpus`/`start`/`health`/`env`/`vendor` all come
    from the re-cloned repo's `nerdit.toml` layered over the row's carried
    config, exactly as a `POST /deploy/git` redeploy with an empty body does.

    `ref` is the recorded *branch or tag* (D-P24-9): re-cloning it is what
    picks up the new HEAD. Pinned-SHA redeploy is deliberately out of v1.
    """
    if request_or_none is None:
        if app is None:
            raise RuntimeError(
                "redeploy_from_source needs a Request, or an app to build the "
                "system-principal shim from."
            )
        request: Request = cast(Request, _SystemRequest(app))
    else:
        request = request_or_none
    name = job.service_name or job.name or ""
    if not name:  # pragma: no cover - a service row always carries a name
        raise NerditError(500, "internal", "The service row carries no name.")
    # Re-read the row AND bind to its identity (PR #108 review): the caller's
    # authorization ran against `job`. If the service was deleted since, a
    # stale-object fallback would let `_finalize_deploy` silently recreate
    # it; if the name was re-created by another owner, adopting the
    # replacement row would resolve THAT owner's recorded source credential
    # under this caller's authorization. Same-name-different-id is therefore a
    # hard stop, not a fallback.
    fresh = await queries.get_service_by_name(name)
    if fresh is None or fresh.id != job.id:
        raise NerditError(
            404,
            "not_found",
            f"Service '{name}' was deleted or replaced while the redeploy was being prepared.",
            hint="Re-run the redeploy against the current service.",
        )
    # (P24c review) The run/release guard belongs to the PRIMITIVE, not to the
    # route: the GitWatch poller drives this function in-process, and a P20 run
    # keeps the row `running` — squarely inside the poller's candidate set. A
    # push landing during a live release would otherwise bump the generation the
    # release owns, exactly the race the rollback route is the template for
    # (D-P24-9). The HTTP route's identical check simply fires earlier, before
    # the clone budget; the poller turns this 409 into a backoff.
    controller = getattr(request.app.state, "service_controller", None)
    if controller is not None and controller.has_active_run(fresh.id):
        raise _run_in_progress_error(name, "redeploy")
    source = _read_git_source(fresh, name)
    repo_url = str(source["repo_url"])
    ref = source.get("ref")
    subdir = source.get("subdir")
    token_ref = source.get("token_ref")
    logger.info(
        "Redeploy from recorded source for '%s' (principal=%s, dry_run=%s)",
        name,
        principal,
        dry_run,
    )

    # Re-validate against the CURRENT allowlist: a host removed from
    # [git].allowed_hosts since the original deploy must stop redeploying.
    try:
        validate_repo_url(repo_url, settings.git.allowed_hosts)
        validate_ref(ref)
        validate_subdir(subdir)
    except GitSourceError as exc:
        raise NerditError(exc.status_code, exc.code, exc.message, hint=exc.hint) from exc

    token = None
    if token_ref is not None:
        token = await _resolve_source_token(
            request, secrets, name, str(token_ref), repo_url=repo_url
        )

    dest_dir = Path(settings.daemon.upload_dir).expanduser() / generate_id()
    try:
        info = await clone_source(
            repo_url,
            ref=ref,
            subdir=subdir,
            dest_dir=dest_dir,
            token=token,
            timeout_s=settings.git.clone_timeout_s,
            max_bytes=settings.git.max_clone_bytes,
            allowed_hosts=settings.git.allowed_hosts,
        )
    except GitSourceError as exc:
        # A row deployed before P24b records no `token_ref` (D-P24-14), so a
        # private repo's unauthenticated re-clone comes back as an auth
        # challenge. Report the missing *reference*, not the raw git failure —
        # the remediation is to record it once, not to fix the URL.
        if token is None and _is_auth_failure(exc):
            raise _source_credential_error(
                name,
                f"The source repository for '{name}' needs a credential, "
                "but none is recorded on the service.",
            ) from exc
        raise NerditError(exc.status_code, exc.code, exc.message, hint=exc.hint) from exc

    # Carry the reference NAME forward so the next redeploy is still unattended
    # (D-P24-14) — never a raw token, which exists only as a local above.
    source_meta = git_source_meta(info, repo_url, subdir=subdir, token_ref=token_ref)

    # Identity re-check AFTER the clone too — the clone window is the long one
    # (bounded only by [git].clone_timeout_s). Without it, a service deleted
    # mid-clone is silently recreated by _finalize_deploy's fresh-create
    # branch, and a same-name replacement row is adopted by its redeploy tail.
    current = await queries.get_service_by_name(name)
    if current is None or current.id != job.id:
        await asyncio.to_thread(shutil.rmtree, dest_dir, ignore_errors=True)
        raise NerditError(
            404,
            "not_found",
            f"Service '{name}' was deleted or replaced while its source was being cloned.",
            hint="Re-run the redeploy against the current service.",
        )
    # …and the run/release guard AFTER it too. The pre-clone check only proves
    # the row was quiet when the clone started; a `nerdit services run` or a
    # release registering inside that window would otherwise have its image
    # swapped out from under it by the finalize below — the very race the
    # pre-clone check exists to prevent, just moved later.
    if controller is not None and controller.has_active_run(current.id):
        await asyncio.to_thread(shutil.rmtree, dest_dir, ignore_errors=True)
        raise _run_in_progress_error(name, "redeploy")

    return await _finalize_deploy(
        request,
        info.context_dir,
        name=name,
        port=None,
        gpus=None,
        start=None,
        health=None,
        env=None,
        vendor=None,
        source_meta=source_meta,
        context_root=dest_dir,
        dry_run=dry_run,
    )
