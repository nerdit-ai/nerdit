"""Response schemas for the purge/delete outcome of `DELETE /services`."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PurgeImages(BaseModel):
    """Image-purge outcome for `DELETE /services` (P14b WP-B1)."""

    removed: list[str] = Field(
        default_factory=list, description="Image tags actually removed (re-list verified)"
    )
    skipped: list[dict[str, str]] = Field(
        default_factory=list, description="Tags still present after removal ({tag, reason})"
    )


class PurgeReport(BaseModel):
    """Best-effort purge outcome bundled into `DELETE /services` (P14b WP-B1).

    Each field is `None` when its target was **not requested** in `?purge`
    (so an agent can distinguish "not asked for" from a False outcome). `secrets`,
    `data` and `workspace` carry a bool outcome; `images` carries a
    removed/skipped split (always `None` for `kind=model` rows — model images
    are never purged here).
    """

    secrets: bool | None = Field(default=None, description="Secrets file removed (None=not asked)")
    data: bool | None = Field(default=None, description="Data dir removed (None=not asked)")
    images: PurgeImages | None = Field(
        default=None, description="Image purge split (None=not asked / model row)"
    )
    # The agent workspace tree + its meta.json sidecar.
    workspace: bool | None = Field(
        default=None, description="Workspace tree removed (None=not asked)"
    )


class ServiceDeletedResponse(BaseModel):
    """Response for `DELETE /services/{ident}` — legacy wire shape + P14b purge."""

    id: str = Field(description="Row id of the deleted service")
    name: str | None = Field(default=None, description="Stable service name of the deleted row")
    deleted: bool = Field(
        description="Always true on success (errors use the NerditError envelope)"
    )
    purged: PurgeReport | None = Field(
        default=None,
        description="Best-effort purge outcome (P14b WP-B1; None when nothing was purged)",
    )
