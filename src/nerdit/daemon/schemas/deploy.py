"""Request schemas for git deploy and the app-template store (Part B)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from nerdit.config.build import BuildSettings
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.utils.names import DNS_LABEL_PATTERN


class ServiceTarget(StrictRequestModel):
    """The service a git-sourced deploy creates or redeploys: a bare `name`, or a project seat.

    `project` + `service` (deploy_into_project_v1) join an existing project as
    `(project, production, service)`; the row's label is then composed by the
    label rule (`<service>--<project>`, or the project name for `web`) and
    `name` may only repeat it. Without them `name` is required and the legacy
    mapping applies (the label is its own project's `web`).
    """

    name: str | None = Field(
        default=None,
        pattern=DNS_LABEL_PATTERN,
        description="DNS-label service name (stable identity, lowercase, 1-63 chars)",
    )
    project: str | None = Field(
        default=None, description="Existing project to join (name or prj_ id); needs `service`"
    )
    service: str | None = Field(
        default=None, description="Service name inside `project` (production environment)"
    )

    @model_validator(mode="after")
    def _name_or_seat(self) -> "ServiceTarget":
        if (self.project is None) != (self.service is None):
            raise ValueError("project and service must be given together.")
        if self.project is None and self.name is None:
            raise ValueError("name is required unless project and service are given.")
        return self


# --- Deploy from a git repository (P11.5 / Part A) ---


class GitDeployRequest(ServiceTarget):
    """Request body for `POST /deploy/git` (JSON — no multipart upload).

    The second deploy ingress: instead of a ZIP the daemon shallow-clones
    `repo_url` (https-only, host-allowlisted) into upload space and hands the
    checkout to the shared deploy tail. `token_ref` is a **secret reference**
    (`${secrets.KEY}` / `${secrets.shared.KEY}`) resolved server-side, so a
    private-repo credential never reaches the request body, audit, or the
    idempotency cache. All other fields mirror the ZIP route's form fields.
    """

    repo_url: str = Field(description="https:// clone URL on an allowed host")
    ref: str | None = Field(
        default=None, description="Branch or tag to clone (default branch if omitted)"
    )
    subdir: str | None = Field(
        default=None, description="Subdirectory to deploy (monorepo support)"
    )
    port: int | None = Field(default=None, description="Container port to publish")
    gpus: int | None = Field(default=None, description="GPUs the service needs (0 = none)")
    start: str | None = Field(default=None, description="Start command override")
    health: str | None = Field(default=None, description="HTTP health-check path")
    env: dict[str, str | None] | None = Field(
        default=None, description="Environment variables (null value deletes on redeploy)"
    )
    build_settings: BuildSettings | None = None
    vendor: str | None = Field(default=None, description="Required GPU vendor")
    token_ref: str | None = Field(
        default=None,
        description=(
            "Private-repo token as a ${secrets.KEY} reference, or the literal "
            "${github.installation} (cloud-pushed GitHub App token, resolved by repo)"
        ),
    )


# --- App template store ---


class AppTemplateEnvVar(BaseModel):
    """A single env/secret input declared by an app template."""

    name: str
    description: str = ""
    required: bool = False
    secret: bool = False


class AppTemplateDeployDefaults(BaseModel):
    """Deploy defaults an app template suggests (overridable at deploy time)."""

    port: int | None = None
    gpus: int | None = None
    start: str | None = None
    health: str | None = None


class AppTemplate(BaseModel):
    """A built-in app template deployable from a git repository."""

    id: str
    name: str
    description: str
    icon: str
    category: str
    repo_url: str
    ref: str | None = None
    subdir: str | None = None
    deploy_defaults: AppTemplateDeployDefaults = Field(default_factory=AppTemplateDeployDefaults)
    env_schema: list[AppTemplateEnvVar] = Field(default_factory=list)
    ai_hint: str | None = None


class TemplateDeployRequest(ServiceTarget):
    """Deploy an app template into a new (or existing) service."""

    env: dict[str, str | None] | None = None
    secrets: dict[str, str] | None = None
    port: int | None = None
    gpus: int | None = None
    start: str | None = None
    health: str | None = None
    build_settings: BuildSettings | None = None
    vendor: str | None = None
