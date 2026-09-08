"""Apply role-based mount restrictions alongside unconditional runtime guards.

Non-admin workloads may mount only daemon-managed roots. The runtime's denylist
still applies to every role, including admins.
"""

from __future__ import annotations

from pathlib import Path

from nerdit.core.runtime.protocol import SandboxViolationError
from nerdit.db.enums import TokenRole
from nerdit.db.queries import Queries
from nerdit.db.rows import Job


async def resolve_role(queries: Queries, job: Job) -> TokenRole:
    """Resolve the role of the token that submitted *job* for sandbox tiering.

    `submitted_by_token is None` is the legacy global token / local bypass →
    `admin`. A token id that no longer resolves (revoked + purged) is treated
    as a non-admin `submitter` so it keeps the tighter sandbox (fail closed).
    """
    if job.submitted_by_token is None:
        return TokenRole.admin
    token = await queries.get_api_token_by_id(job.submitted_by_token)
    return token.role if token is not None else TokenRole.submitter


def enforce_mount_allowlist(volumes: dict[str, str], allowed_roots: list[str]) -> None:
    """Reject any host mount outside the Tier-B allowlist (non-admin only).

    Only the daemon-managed roots (upload + cache dirs) are allowed. Caller must
    invoke this only for non-admin roles. Paths are symlink-resolved and an
    ancestor match is required. Raises
    `SandboxViolationError` (`reason='mount_not_allowlisted'`) on the
    first offending mount.
    """
    if not volumes:
        return
    allowed: list[Path] = []
    for root in allowed_roots:
        try:
            allowed.append(Path(root).expanduser().resolve())
        except (OSError, RuntimeError):
            allowed.append(Path(root).expanduser())
    for host in volumes:
        try:
            hp = Path(host).expanduser().resolve()
        except (OSError, RuntimeError):
            hp = Path(host).expanduser()
        if not any(hp == root or root in hp.parents for root in allowed):
            raise SandboxViolationError(
                f"Mount {host!r} is outside the allowed roots for non-admin "
                f"services (allowed: {[str(r) for r in allowed]}).",
                reason="mount_not_allowlisted",
                path=host,
            )
