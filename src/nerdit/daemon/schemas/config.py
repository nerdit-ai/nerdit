"""Response schemas for the config-as-API surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nerdit.daemon.schemas._base import StrictRequestModel


class ConfigDiagnostic(BaseModel):
    """A single validation diagnostic for a config write."""

    loc: list[str] = Field(default_factory=list, description="Path to the offending field")
    message: str = Field(description="Human-readable diagnostic message")
    type: str | None = Field(default=None, description="Machine-readable error type")


class ConfigDiffEntry(BaseModel):
    """One changed configuration key, with secrets redacted."""

    key: str = Field(description="Dotted path of the changed key")
    old: object | None = Field(default=None, description="Previous value (redacted if secret)")
    new: object | None = Field(default=None, description="New value (redacted if secret)")
    section: str = Field(default="", description="Config section the key belongs to")
    op: Literal["add", "change", "delete"] = Field(
        default="change", description="Whether the key was added, changed, or deleted"
    )
    requires_restart: bool = Field(
        default=False, description="True if this key only takes effect after a daemon restart"
    )
    secret: bool = Field(default=False, description="True if the values were redacted as secrets")


class ConfigView(BaseModel):
    """A read view of a config section (secrets redacted)."""

    section: str = Field(description="Section name (e.g., 'daemon', 'scheduler')")
    values: dict[str, object] = Field(
        default_factory=dict, description="Section key/value pairs, secrets redacted"
    )
    etag: str | None = Field(default=None, description="SHA-256 of the underlying bytes")


class ConfigWriteResponse(BaseModel):
    """Result of a config write (or dry-run)."""

    applied: bool = Field(description="True if persisted; False for a dry-run")
    diff: list[ConfigDiffEntry] = Field(default_factory=list, description="Changed keys (redacted)")
    diagnostics: list[ConfigDiagnostic] = Field(
        default_factory=list, description="Validation diagnostics (empty on success)"
    )
    requires_restart: bool = Field(
        default=False, description="True if the change needs a daemon restart to take effect"
    )


class ConfigApplyRequest(StrictRequestModel):
    """Declarative multi-section daemon config apply.

    Desired state for the *mentioned* sections only — sections absent from the
    document are left untouched (never a whole-file replace).

    Strict at the top level only: `sections` is an open dict by contract, and
    the per-key validation inside it is the config store's (D-P22-3).
    """

    sections: dict[str, dict[str, object]] = Field(
        description="Section name -> partial key/value document (null deletes a key)"
    )


class AppConfigView(BaseModel):
    """Read view of a deployed app's daemon-persisted config.

    Projects exactly what the daemon stores for a service — the `[deploy]`
    fields, the `[ai.*]` binding *spec* (secret refs only, never values),
    and secret/env key names. `source`/`revision` make redeploy-vs-API
    precedence visible (last writer wins, clobbers detectable).
    """

    service_name: str = Field(description="The service this config belongs to")
    deploy: dict[str, object] = Field(
        default_factory=dict,
        description="[deploy] fields (name, port, gpus, start, health, memory_limit, cpu_limit)",
    )
    ai: dict[str, dict[str, object]] = Field(
        default_factory=dict, description="[ai.*] binding spec (api_key is a ${secrets.X} ref)"
    )
    db: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="[db.*] binding spec (P15; password is a ${secrets.X} ref, url has no creds)",
    )
    env_keys: list[str] = Field(
        default_factory=list, description="Env var names (values never returned)"
    )
    source: Literal["deploy", "api"] = Field(
        default="deploy", description="Who last wrote this config"
    )
    revision: int = Field(default=0, description="Monotonic revision, bumped by every writer")
    etag: str | None = Field(default=None, description="SHA-256 of the canonical view JSON")


class AppConfigWriteResponse(BaseModel):
    """Result of a per-app config section write (or dry-run)."""

    applied: bool = Field(description="True if persisted; False for a dry-run")
    requires_restart: bool = Field(
        default=False, description="True if the change only takes effect after a service restart"
    )
    restarted: bool = Field(
        default=False, description="True if ?restart=true triggered a service restart"
    )
    view: AppConfigView = Field(description="The resulting (or would-be) config view")


class ConfigApplyResponse(BaseModel):
    """Result of a declarative config apply (or dry-run)."""

    applied: bool = Field(description="True if persisted; False for a dry-run")
    changed: bool = Field(
        default=False, description="False when the document was already in effect (no-op)"
    )
    etag: str | None = Field(default=None, description="ETag of the resulting config")
    requires_restart: bool = Field(
        default=False, description="True if any changed key needs a daemon restart"
    )
    restart_keys: list[str] = Field(
        default_factory=list, description="Dotted keys whose change needs a daemon restart"
    )
    diff: list[ConfigDiffEntry] = Field(default_factory=list, description="Changed keys (redacted)")
    diagnostics: list[ConfigDiagnostic] = Field(
        default_factory=list, description="Validation diagnostics (empty on success)"
    )
