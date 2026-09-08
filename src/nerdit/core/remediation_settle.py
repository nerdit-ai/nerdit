"""Classify failed generations for public settlement records.

Emit machine remediation codes only: free-text detail and log tails stay in
`/diagnose`. The daemon classifier imports these helpers, never the reverse;
core must not import daemon modules.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Literals mirrored from `nerdit.daemon.remediation.RemediationCode`.
SETTLE_RAISE_MEMORY_LIMIT = "raise_memory_limit"
SETTLE_IMAGE_NEEDS_PRIVILEGES = "image_needs_privileges"
SETTLE_FIX_START_COMMAND = "fix_start_command"


def parse_iso(value: object) -> datetime | None:
    """Parse an ISO-8601 string to a tz-aware datetime, or `None` on any error."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def forensics_fresh(forensics: dict, last_deploy: dict | None) -> bool:
    """True when the persisted crash forensics belong to the current deploy generation.

    A crash whose `last_crash_at` predates the current `last_deploy.started_at`
    is a *stale* generation's crash (the queued stamp pops the forensics keys, so
    this is belt-and-braces) and must not drive a forensics-keyed rule. Absent a
    `last_deploy` boundary (pre-P13 rows) a present crash is treated as fresh.
    """
    crash_at = parse_iso(forensics.get("last_crash_at"))
    if crash_at is None:
        return False
    started_at = parse_iso((last_deploy or {}).get("started_at"))
    if started_at is None:
        return True
    return crash_at >= started_at


def forensics_from_config(cfg: dict) -> dict:
    """Read the persisted crash-forensics bundle out of a config blob.

    Names/values are numeric/bool/timestamp only (`last_exit_code` /
    `oom_killed` / `gpu_oom` / `priv_denied` / `last_crash_at`), coerced
    defensively so a malformed blob degrades to "no forensics". The diagnose
    route and the settle-time stamp (`settle_remediation_code`) read the
    same bundle through this one reader.
    """
    raw_exit = cfg.get("last_exit_code")
    raw_crash = cfg.get("last_crash_at")
    return {
        "last_exit_code": raw_exit if isinstance(raw_exit, int) else None,
        "oom_killed": bool(cfg.get("oom_killed")),
        "gpu_oom": bool(cfg.get("gpu_oom")),
        "priv_denied": bool(cfg.get("priv_denied")),
        "last_crash_at": raw_crash if isinstance(raw_crash, str) else None,
    }


def settle_remediation_code(cfg: dict, last_deploy: dict) -> str | None:
    """The remediation code to stamp on a generation settling `failed`.

    Runs rule 1b (cgroup OOM) and the crash-loop rules 4b/5 of the LOCKED table
    against the fresh config blob, in the table's ranking. `None` when no rule
    fires — a build failure, a release/cutover failure, or a crash outside the
    user-error exit band — so a reader never mistakes "no rule" for a verdict.
    """
    forensics = forensics_from_config(cfg)
    fresh = forensics_fresh(forensics, last_deploy)
    if not fresh:
        return None
    if forensics.get("oom_killed"):
        return SETTLE_RAISE_MEMORY_LIMIT
    exit_code = forensics.get("last_exit_code")
    crash_loop = last_deploy.get("reason") == "crash_loop"
    if not (crash_loop and isinstance(exit_code, int) and 1 <= exit_code <= 126):
        return None
    if forensics.get("priv_denied"):
        return SETTLE_IMAGE_NEEDS_PRIVILEGES
    return SETTLE_FIX_START_COMMAND
