"""Response schema for the `/audit` surface."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.db.rows import AuditLogEntry


class AuditLogPage(BaseModel):
    """Cursor-paginated page of audit log entries."""

    items: list[AuditLogEntry]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page; null when the log is exhausted",
    )
