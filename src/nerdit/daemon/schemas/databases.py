"""Request/response schemas for database creation, reads, dumps and restores."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.enums import JobStatus
from nerdit.utils.names import DNS_LABEL_PATTERN


class DatabaseCreateRequest(StrictRequestModel):
    """Request to provision a managed database.

    Backend defaults to databases.default_backend; name defaults to the backend's
    prefix and follows service DNS-label rules. Databases accept no GPU parameter.
    The daemon generates credentials server-side.
    """

    backend: str | None = Field(
        default=None,
        description="Data backend ('postgres' | 'redis'); None uses the daemon default",
    )
    name: str | None = Field(
        default=None,
        pattern=DNS_LABEL_PATTERN,
        description="Optional service-name override (default: the backend name prefix)",
    )


class DatabaseResponse(BaseModel):
    """Database projection, also visible through GET /services.

    Endpoint is a password-free host:port, never a DSN. db_ready becomes true only
    after the wire-protocol probe succeeds, not merely when the container starts.
    """

    id: str = Field(description="Unique 12-character row identifier")
    name: str = Field(description="Stable service name (identity)")
    backend: str | None = Field(default=None, description="Data backend (e.g. 'postgres')")
    status: JobStatus = Field(description="Current lifecycle state")
    desired_state: str | None = Field(
        default=None, description="Reconciler target ('running' | 'stopped')"
    )
    db_ready: bool = Field(
        default=False, description="Whether the database is accepting connections"
    )
    endpoint: str | None = Field(
        default=None,
        description="Password-free host:port display string; null until the host port is "
        "published. Never a DSN — the credential-bearing DSN is injected into bound apps only.",
    )
    created_at: datetime = Field(description="Timestamp when the database workload was created")


class DatabaseListPage(BaseModel):
    """Cursor-paginated page of databases for the bounded `GET /databases` read."""

    items: list[DatabaseResponse]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page; null when the list is exhausted",
    )


# --- Dumps (P37, §1.4) --------------------------------------------------------

#: Default ``timeout_s`` for a dump (D-P37-8). Fifteen minutes: a logical dump
#: streams the whole dataset through the server, and the operator who needs more
#: raises it per request up to ``[services].dump_timeout_max_s``.
DUMP_DEFAULT_TIMEOUT_S = 900

#: Default ``timeout_s`` for a restore (D-P37-8): shorter than the dump's, since
#: a Postgres restore competes for locks with every bound app.
RESTORE_DEFAULT_TIMEOUT_S = 600

#: The dump basename grammar (D-P37-10), a field pattern so a malformed value is
#: refused before any filesystem access. Stricter than ``core.backup.DUMP_TAR_RE``:
#: the service segment is DNS-label only, so the value cannot name a path.
DUMP_BASENAME_PATTERN = r"^nerdit-dump-[a-z0-9-]{1,63}-\d{8}T\d{6}Z-[0-9a-f]{6}\.tar\.gz$"


class DatabaseDumpRequest(StrictRequestModel):
    """Request body for ``POST /databases/{name}/dump`` (P37, D-P37-8). The server
    cap ``[services].dump_timeout_max_s`` binds: over it is ``422
    dump.timeout_too_large``, never a silent clamp.
    """

    timeout_s: int = Field(
        default=DUMP_DEFAULT_TIMEOUT_S,
        ge=1,
        description="Seconds the dump may run before it is killed "
        "(server cap: [services].dump_timeout_max_s)",
    )


class DatabaseRestoreRequest(StrictRequestModel):
    """Request body for ``POST /databases/{name}/restore`` (P37, D-P37-10). ``dump``
    is a basename (the pattern makes a path unrepresentable); cross-name restore
    is allowed and is how a clone is made.
    """

    dump: str = Field(
        pattern=DUMP_BASENAME_PATTERN,
        description="Basename of a dump tar under the daemon's backups directory "
        "(as returned by POST /databases/{name}/dump or GET /databases/{name}/dumps)",
    )
    timeout_s: int = Field(
        default=RESTORE_DEFAULT_TIMEOUT_S,
        ge=1,
        description="Seconds the restore may run before it is killed "
        "(server cap: [services].dump_timeout_max_s)",
    )


class DatabaseDumpResponse(BaseModel):
    """Response for ``POST /databases/{name}/dump`` (P37, §1.4): metadata only, the
    tar is never served over HTTP and ``path`` is deliberately absent (M3).
    """

    service_name: str = Field(description="Stable name of the database that was dumped")
    dump: str = Field(description="Basename of the dump tar under the daemon's backups directory")
    size_bytes: int = Field(description="Size of the packed tar on disk")
    sha256: str = Field(description="sha256 of the dump member inside the tar (not of the tar)")
    engine: str = Field(description="Data backend the dump came from ('postgres' | 'redis')")
    format: str = Field(description="Dump format ('pg_custom' | 'rdb')")
    created_at: datetime = Field(description="When the dump tar was written")
    duration_s: float = Field(description="Wall-clock seconds the dump tool ran")


class DatabaseDumpItem(BaseModel):
    """One row of ``GET /databases/{name}/dumps``.

    ``created_at`` is the tar's mtime: the listing never OPENS a tar, so it
    cannot report the manifest's own ``created_at`` without paying a gzip read
    per file — and the two agree to the second on every tar this daemon wrote.
    """

    dump: str = Field(description="Basename of the dump tar")
    size_bytes: int = Field(description="Size of the tar on disk")
    created_at: datetime = Field(description="Tar mtime (the listing never opens the archive)")


class DatabaseDumpListResponse(BaseModel):
    """Response for ``GET /databases/{name}/dumps`` — newest first, capped.

    ``truncated`` is what keeps this a bounded read (Invariant #3) without a
    cursor: ``[retention].dump_keep_last`` normally holds the directory to five
    tars per database, but ``0`` is a supported "never prune", so the cap is a
    module constant in the route rather than a consequence of a config key.
    """

    service_name: str = Field(description="Stable name of the database the dumps belong to")
    dumps: list[DatabaseDumpItem] = Field(
        default_factory=list, description="Dump tars for this database, newest first"
    )
    truncated: bool = Field(
        default=False,
        description="True when older dumps exist beyond the page cap (the newest are returned)",
    )


class DatabaseRestoreResponse(BaseModel):
    """Response for ``POST /databases/{name}/restore`` (P37, §1.4)."""

    service_name: str = Field(description="Stable name of the database that was restored into")
    dump: str = Field(description="Basename of the dump tar that was restored")
    engine: str = Field(description="Data backend that was restored ('postgres' | 'redis')")
    duration_s: float = Field(description="Wall-clock seconds the restore took")
