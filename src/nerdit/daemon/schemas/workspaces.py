"""Schemas for text-only agent workspace writes and deployments.

Files map relative paths to UTF-8 content. Deploy options match the other
ingresses, excluding git coordinates, and feed the shared deployment pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.daemon.schemas._base import StrictRequestModel

# --- requests ---------------------------------------------------------------


class WorkspaceWriteRequest(StrictRequestModel):
    """Request body for `PUT /workspaces/{name}/files` (batch write + delete).

    All-or-nothing (D-P29-1): the daemon validates every entry — path grammar,
    exclude set, secret basenames, text-only, and the three caps — before the
    first byte lands, so a rejected batch leaves the workspace untouched.
    """

    files: dict[str, str] = Field(
        default_factory=dict,
        description="Workspace-relative path -> UTF-8 text content",
    )
    delete: list[str] = Field(
        default_factory=list,
        description="Workspace-relative paths to remove (missing paths are a no-op)",
    )


class WorkspaceDeployRequest(StrictRequestModel):
    """Request body for `POST /workspaces/{name}/deploy` (all fields optional).

    Mirrors the ZIP/git ingress option set; the workspace name IS the service
    name, so there is no `name` field to disagree with the path.
    """

    port: int | None = Field(default=None, description="Container port to publish")
    gpus: int | None = Field(default=None, description="GPUs the service needs (0 = none)")
    start: str | None = Field(default=None, description="Start command override")
    health: str | None = Field(default=None, description="HTTP health-check path")
    env: dict[str, str | None] | None = Field(
        default=None, description="Environment variables (null value deletes on redeploy)"
    )
    vendor: str | None = Field(default=None, description="Required GPU vendor")


# --- responses --------------------------------------------------------------


class WorkspaceFileEntry(BaseModel):
    """One file in a workspace listing or write summary.

    `mtime` is omitted in write summaries (the post-state listing a write
    returns is about *what landed*, not when).
    """

    path: str
    size: int
    sha256: str
    mtime: str | None = None


class WorkspaceWriteResponse(BaseModel):
    """Post-state summary of one batch write."""

    written: int
    deleted: int
    file_count: int
    total_bytes: int
    files: list[WorkspaceFileEntry]


class WorkspaceMetaView(BaseModel):
    """The daemon-owned `meta.json` sidecar, projected (D-P29-6)."""

    owner_token_id: str | None = None
    created_at: str | None = None
    last_written_at: str | None = None


class WorkspaceListResponse(BaseModel):
    """Full workspace listing (bounded by `WORKSPACE_MAX_FILES`, no paging)."""

    name: str
    files: list[WorkspaceFileEntry]
    file_count: int
    total_bytes: int
    meta: WorkspaceMetaView
