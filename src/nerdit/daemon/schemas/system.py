"""Response schemas for `POST /system/backup` and its volume twin."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BackupResponse(BaseModel):
    """Response for `POST /system/backup` (P14c WP-B3).

    Admin-only; the `path` is the absolute on-box location of the tar (never
    served over HTTP). `contains_master_key` is always true — the tar carries
    the secrets master key, so custody transfers with the file. The `hint`
    spells out the custody obligation (copy off-box + delete the local file).
    """

    backup: str = Field(description="Tar filename (basename only)")
    path: str = Field(description="Absolute on-box path of the tar (admin-only; never served)")
    size_bytes: int = Field(description="Size of the packed tar in bytes")
    kid: str = Field(description="Active secrets key id (8-hex) captured in the tar")
    created_at: str = Field(description="ISO-8601 UTC timestamp the backup was staged")
    contains_master_key: Literal[True] = Field(
        default=True, description="Always true — the tar holds the secrets master key"
    )
    hint: str = Field(description="Key-custody obligation for the operator")


class VolumeBackupResponse(BaseModel):
    """Response for `POST /system/backup/volumes` (P15 WP7 — backup v2).

    Admin-only; the `path` is the absolute on-box location of the per-database
    volume tar (never served over HTTP). Unlike the control-plane
    `BackupResponse`, `contains_master_key` is always **false** — the
    volume tar carries the database data and the SCRAM password verifiers but
    NEVER the secrets master key or `.enc` ciphertexts. The `hint` spells out
    the custody note.
    """

    service: str = Field(description="The managed database the volumes were captured from")
    backend: str | None = Field(default=None, description="Data backend (e.g. 'postgres')")
    backup: str = Field(description="Tar filename (basename only)")
    path: str = Field(description="Absolute on-box path of the tar (admin-only; never served)")
    size_bytes: int = Field(description="Size of the packed tar in bytes")
    created_at: str = Field(description="ISO-8601 UTC timestamp the backup was staged")
    contains_master_key: Literal[False] = Field(
        default=False,
        description="Always false — the volume tar holds no secrets master key",
    )
    hint: str = Field(description="Custody note for the operator")
