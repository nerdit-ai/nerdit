"""nerdit.toml project configuration — declarative job description."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nerdit.config.build import BuildSettings

PROJECT_CONFIG_NAME = "nerdit.toml"

# DNS-label rule for [deploy].name — same regex as ServiceCreateRequest.name
# (db/models.py): lowercase, 1-63 chars, no leading/trailing '-'.
_DNS_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

# [deploy].memory_limit grammar — docker ``mem_limit``: a bare byte
# count or a number with an optional b/k/m/g suffix (case-insensitive).
_MEMORY_LIMIT_RE = re.compile(r"^\d+(\.\d+)?[bkmg]?$", re.IGNORECASE)

# Reject release control characters to prevent hidden statements and log forging.
# Keep this grammar separate from the broader advisory-display sanitization.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

#: Advisory text also strips Unicode separators and bidi controls from app keys.
#: Keep separate from release validation so display hardening does not change
#: the accepted command grammar.
_DISPLAY_UNSAFE_RE = re.compile(
    r"[\x00-\x1f\x7f-\x9f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]"
)

# Basic-auth user: 1–64 printable ASCII characters, no colon, at least one nonspace.
# RFC 7617 uses colon as the separator; controls would leak into config and logs.
# Ingress and persisted-config checks share this grammar.
_EDGE_AUTH_USER_RE = re.compile(r"^(?=[\x20-\x7e]*[\x21-\x39\x3b-\x7e])[\x20-\x39\x3b-\x7e]{1,64}$")

# Frozen [ai.*] grammar (P5, Invariant #1) — do not loosen without a
# contract-freeze review (tests/test_ai_binding_schema.py pins these).
# Binding names map to NERDIT_AI_<NAME>_* env vars.
AI_BINDING_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# api_key must be a SecretManager reference — literal keys never live in TOML.
# Two groups (P8): (scope | None, KEY). The only scope is the literal 'shared'
# — ``${secrets.shared.KEY}`` resolves the per-service KEY first (override
# escape hatch), then the shared store; unscoped refs stay per-service-only.
SECRET_REF_RE = re.compile(r"^\$\{secrets\.(?:(shared)\.)?([A-Z][A-Z0-9_]*)\}$")

# [db.*] binding name grammar (P15, D-A) — its OWN constant, cloned from
# AI_BINDING_NAME_RE (never repointed) so the frozen [ai.*] grammar and the new
# [db.*] grammar evolve independently. Binding names map to NERDIT_DB_<NAME>_URL.
DB_BINDING_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# External-DSN schemes a [db.<name>] url may use (P15). Postgres and Redis only;
# a credential-bearing scheme like ``postgresql+asyncpg`` or an unrelated scheme
# is rejected so the injected env is always a plain, driver-agnostic DSN.
_DB_URL_SCHEMES = frozenset({"postgresql", "redis", "rediss"})

# Match userinfo through the last @ before the path, as urlsplit does, including
# raw whitespace and @ inside passwords. Redact rejected URLs before diagnostics.
_URL_USERINFO_RE = re.compile(r"//[^/]*@")


def _redact_db_url(value: str) -> str:
    """A credential-free rendering of a `[db] url` for error messages.

    Strips any `user:password@` userinfo and any query string (which may carry
    a `?password=`/`?sslpassword=`), keeping the scheme/host/path so the
    message stays actionable without ever echoing the credential it rejects.
    Regex-based so it is safe even on a URL that failed to parse.
    """
    masked = _URL_USERINFO_RE.sub("//", value)
    return masked.split("?", 1)[0]


# Parse-time volume grammar; launch revalidates untrusted persisted specs.
# DNS-label names cannot contain host paths; the daemon computes them.
# Keep this definition independent of core.volumes to avoid an import cycle.
_VOLUME_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$")
_FORBIDDEN_VOLUME_PATHS = frozenset({"/", "/workspace"})
_FORBIDDEN_VOLUME_PREFIXES = ("/proc", "/sys", "/dev")
MAX_VOLUMES = 8


def validate_volume_specs(specs: list) -> list[str]:
    """Validate volume names and container paths without resolving host paths.

    Require DNS-label names, normalized absolute paths outside reserved mounts,
    unique names/paths and at most MAX_VOLUMES. Launch-time resolution revalidates.

    Raises:
        ValueError: The first invalid specification.
    """
    if len(specs) > MAX_VOLUMES:
        raise ValueError(f"at most {MAX_VOLUMES} volumes allowed, got {len(specs)}")
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for spec in specs:
        if not isinstance(spec, str):
            raise ValueError(f"volume spec must be a string: {spec!r}")
        volname, sep, container_path = spec.partition(":")
        if not sep:
            raise ValueError(f"volume spec must be '<name>:<path>': {spec!r}")
        if ":" in container_path:
            raise ValueError(f"volume spec has an extra ':': {spec!r}")
        if not _VOLUME_NAME_RE.fullmatch(volname):
            raise ValueError(
                f"invalid volume name {volname!r}: 1-32 lowercase letters, digits or '-'"
            )
        if not container_path.startswith("/"):
            raise ValueError(f"volume container path must be absolute: {container_path!r}")
        # POSIX normpath preserves an exactly-two-slash prefix (``//x`` stays
        # ``//x``), which would slip past the forbidden-path checks below while
        # the kernel collapses it to ``/x``. Reject a doubled leading slash.
        if container_path.startswith("//"):
            raise ValueError(
                f"volume container path has a doubled leading slash: {container_path!r}"
            )
        if os.path.normpath(container_path) != container_path:
            raise ValueError(f"volume container path must be normalized: {container_path!r}")
        if container_path in _FORBIDDEN_VOLUME_PATHS:
            raise ValueError(f"volume container path not allowed: {container_path!r}")
        for prefix in _FORBIDDEN_VOLUME_PREFIXES:
            if container_path == prefix or container_path.startswith(prefix + "/"):
                raise ValueError(f"volume container path not allowed: {container_path!r}")
        if volname in seen_names:
            raise ValueError(f"duplicate volume name: {volname!r}")
        if container_path in seen_paths:
            raise ValueError(f"duplicate volume container path: {container_path!r}")
        seen_names.add(volname)
        seen_paths.add(container_path)
    return specs


def shared_secret_keys_for(specs: dict | None, fields: tuple[str, ...]) -> list[str]:
    """Return sorted unique shared-secret keys referenced by the selected fields.

    Scan API-key fields for AI bindings or password fields for DB bindings.
    Ignore non-dict persisted bindings. Launch and write-time audit paths share
    this scanner.
    """
    keys: set[str] = set()
    for spec in (specs or {}).values():
        if not isinstance(spec, dict):
            continue
        for field in fields:
            match = SECRET_REF_RE.match(str(spec.get(field) or ""))
            if match is not None and match.group(1) == "shared":
                keys.add(match.group(2))
    return sorted(keys)


def shared_secret_keys(ai_specs: dict | None) -> list[str]:
    """Return sorted unique shared-secret keys referenced by AI api_key fields."""
    return shared_secret_keys_for(ai_specs, ("api_key",))


class EdgeAuthConfig(BaseModel):
    """HTTP basic auth enforced at the Caddy edge.

    Password must reference a service or shared secret; literals are rejected.
    The daemon resolves and bcrypt-hashes it at route build time, never storing
    plaintext in config or logs. Reject unknown fields and suppress input values
    in validation errors; custom validator messages must also remain value-free.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    user: str = Field(min_length=1, max_length=64)
    password: str

    @field_validator("user")
    @classmethod
    def _check_user_grammar(cls, value: str) -> str:
        # ``fullmatch``: ``$`` also matches before a trailing newline, so
        # ``match`` would accept "admin\n" — the one control character the
        # anchored grammar admits — and the ingress would diverge from the
        # ``load_edge_auth`` backstop, which fullmatches (review-upheld P25
        # finding). Same rule on the password ref below.
        if not _EDGE_AUTH_USER_RE.fullmatch(value):
            # Never echo the value: a control character echoed back through the
            # 422 envelope / job_logs / diagnose is a log-forging vector (the
            # ``[deploy].release`` precedent), and the ':' rule is RFC 7617's —
            # the Basic credential is ``user:password``, so a ':' in the user
            # would silently change which password the browser sends.
            raise ValueError(
                "Invalid [deploy] edge_auth user: must be 1-64 printable ASCII "
                "characters and must not contain ':' (RFC 7617 separates the "
                "user and the password with a colon)."
            )
        return value

    @field_validator("password")
    @classmethod
    def _check_password_is_secret_ref(cls, value: str) -> str:
        if not SECRET_REF_RE.fullmatch(value):
            # Never echo the rejected value: a literal password is exactly what
            # this rejects, and the message rides the same 422/job_logs/diagnose
            # surfaces as every other secret-ref validator (finding #6).
            raise ValueError(
                "Invalid [deploy] edge_auth password: must be a secret reference "
                "like '${secrets.APP_PW}' or '${secrets.shared.APP_PW}' "
                "(uppercase key, set via 'nerdit secrets set') — literal "
                "passwords are not allowed in nerdit.toml."
            )
        return value


class DeployConfig(BaseModel):
    """Deployment defaults from the deploy section.

    Services default to zero GPUs. Unset port, start and health permit buildpack
    ports, image commands and liveness-only supervision. Names must be DNS labels;
    explicit ports must be valid TCP ports. AI bindings are parsed separately.

    Unknown keys warn rather than fail. Suppress validation input values here as
    well as in EdgeAuthConfig: nested models do not protect outer error rendering.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    name: str
    port: int | None = Field(default=None, ge=1, le=65535)
    gpus: int = Field(default=0, ge=0)
    start: str | None = None
    build_settings: BuildSettings | None = None
    health: str | None = None
    # Health-probe kind: "tcp" supervises a non-HTTP server by a bare
    # connect to the published port; "http"/absent keeps the HTTP-GET probe. v1
    # is TOML/deploy-route only (not app-config-PUT-mutable — the API path for a
    # tcp probe is ``POST /services`` with a full ``HealthCheck.type``).
    health_type: Literal["http", "tcp"] | None = None
    # Container resource caps — per-app only (``nerdit.toml`` + config
    # API), applied at every launch (``services.py`` reads these top-level config
    # keys). ``memory_limit`` follows docker's ``mem_limit`` grammar (a bare byte
    # count or a number with a b/k/m/g suffix); ``cpu_limit`` is a fractional
    # core count. Absent ⇒ the ``[containers]`` daemon defaults apply.
    memory_limit: str | None = None
    cpu_limit: float | None = Field(default=None, gt=0)
    # Named volumes (P14 WP-A1): each ``<volname>:<container_path>`` is a
    # daemon-computed leaf under ``<data_dir>/services/<name>`` bind-mounted at
    # the declared container path. A user never supplies a host path (the volname
    # grammar forbids it). Re-validated at every launch by ``core/volumes.py``.
    volumes: list[str] | None = None
    # Run once in the candidate image before swapping; failure restores the old image,
    # never data. Migrations must be idempotent and compatible with the previous image.
    # Unlike start's exec form, release uses /bin/sh -c and requires a shell.
    release: str | None = Field(default=None, max_length=4096)
    # None uses daemon cutover eligibility; False keeps the same-port swap.
    # This next-deploy setting carries forward when omitted from TOML;
    # clear an opt-out explicitly with config app set ... deploy cutover=null.
    cutover: bool | None = None
    # Push-triggered redeploy opt-in (P24b schema, consumed by the P24c
    # GitWatch/webhook plane — WP10). Declared here so the key validates and
    # persists from the day the cutover ships; nothing reads it yet. Same
    # carry-forward semantics as ``cutover``.
    auto_deploy: bool | None = None
    # Resolve and hash the secret at the Caddy edge; apply changes on the next tick.
    # Like other non-form deployment fields, auth carries forward when omitted.
    # Only an explicit edge_auth=null removes protection; no container restart needed.
    edge_auth: EdgeAuthConfig | None = None

    @field_validator("name")
    @classmethod
    def _check_name_is_dns_label(cls, value: str) -> str:
        if not _DNS_LABEL_RE.match(value):
            raise ValueError(
                f"Invalid [deploy] name '{value}': must be a DNS label "
                "(lowercase letters, digits and '-', 1-63 chars, "
                "starting and ending with a letter or digit)."
            )
        return value

    @field_validator("memory_limit")
    @classmethod
    def _check_memory_limit_format(cls, value: str | None) -> str | None:
        if value is not None and not _MEMORY_LIMIT_RE.match(value):
            raise ValueError(
                f"Invalid [deploy] memory_limit '{value}': expected a byte count "
                "with an optional b/k/m/g suffix (e.g. '512m', '2g', '1073741824')."
            )
        return value

    @field_validator("volumes")
    @classmethod
    def _check_volumes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        return validate_volume_specs(value)

    @field_validator("release")
    @classmethod
    def _check_release_command(cls, value: str | None) -> str | None:
        """Reject blank, multiline or control-character release commands before shell wrapping."""
        if value is None:
            return value
        if not value.strip():
            raise ValueError(
                "Invalid [deploy] release: must be a non-empty command "
                "(drop the key entirely to run no release step)."
            )
        if _CONTROL_CHAR_RE.search(value):
            # Never echo the value back: the control characters are exactly what
            # makes it unsafe to interpolate into a message that rides the 422
            # envelope, job_logs, /diagnose and the audit trail.
            raise ValueError(
                "Invalid [deploy] release: control characters (newlines, tabs "
                "and NUL included) are not allowed — chain commands on one "
                "line with ';' or '&&'."
            )
        return value


class AiBindingConfig(BaseModel):
    """An AI binding resolved into OpenAI-compatible environment variables at launch.

    Ollama bindings name a served model and forbid base_url/api_key; the daemon
    composes the endpoint and placeholder key. API bindings require base_url and
    a service/shared secret reference for api_key, never a literal credential.
    Inject OPENAI_BASE_URL/OPENAI_API_KEY and NERDIT_AI_<NAME>_* variables.
    Reject unknown fields and hide validation inputs to prevent credential leaks.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: Literal["ollama", "api"]
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: str | None = None

    @field_validator("api_key")
    @classmethod
    def _check_api_key_is_secret_ref(cls, value: str | None) -> str | None:
        if value is not None and not SECRET_REF_RE.match(value):
            # Never echo the rejected value: a literal API key is exactly what
            # this rejects, and ``hide_input_in_errors`` does NOT reach a
            # validator's own message — it rides ``msg`` into the 422 envelope,
            # job_logs and diagnose. The diagnostic lives in
            # the ``loc`` (``ai.<name>.api_key``), which pydantic keeps.
            raise ValueError(
                "Invalid [ai] api_key: must be a secret reference "
                "like '${secrets.OPENAI_KEY}' or '${secrets.shared.OPENAI_KEY}' "
                "(uppercase key, set via 'nerdit secrets set') — literal API "
                "keys are not allowed in nerdit.toml."
            )
        return value

    @model_validator(mode="after")
    def _check_provider_fields(self) -> "AiBindingConfig":
        if self.provider == "api":
            if self.base_url is None:
                raise ValueError("[ai] provider 'api' requires 'base_url'.")
            if self.api_key is None:
                raise ValueError(
                    "[ai] provider 'api' requires 'api_key' (a '${secrets.KEY}' reference)."
                )
        else:  # provider == "ollama"
            if self.base_url is not None:
                raise ValueError(
                    "[ai] provider 'ollama' forbids 'base_url': "
                    "the daemon composes the local model endpoint."
                )
            if self.api_key is not None:
                raise ValueError(
                    "[ai] provider 'ollama' forbids 'api_key': "
                    "the daemon injects a placeholder key."
                )
        return self


class DbBindingConfig(BaseModel):
    """A database binding resolved into environment URLs at every launch.

    Managed bindings require a database service name and forbid url/password;
    the daemon composes the DSN. External bindings require a credential-free
    postgresql/redis/rediss URL, without query strings, and a service/shared
    password reference inserted at launch. Literal passwords are rejected.

    Inject NERDIT_DB_<NAME>_URL and DATABASE_URL/REDIS_URL for the default binding.
    Reject unknown fields and hide validation inputs to prevent credential leaks.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: Literal["managed", "external"]
    database: str | None = None
    url: str | None = None
    password: str | None = None

    @field_validator("url")
    @classmethod
    def _check_url_shape(cls, value: str | None) -> str | None:
        if value is None:
            return value
        # Redact BEFORE any interpolation: the message flows into a 422 envelope,
        # the persisted-spec BindingNotReady string, job_logs and diagnose, and a
        # userinfo password would otherwise ride all of them (finding #6).
        safe = _redact_db_url(value)
        try:
            parts = urlsplit(value)
        except ValueError as exc:
            raise ValueError(f"Invalid [db] url '{safe}': could not be parsed.") from exc
        if parts.scheme not in _DB_URL_SCHEMES:
            raise ValueError(
                f"Invalid [db] url '{safe}': scheme must be one of "
                "postgresql://, redis://, rediss:// — got "
                f"'{parts.scheme or '(none)'}'."
            )
        if parts.password is not None:
            raise ValueError(
                f"Invalid [db] url '{safe}': it must not embed a password — "
                "set password = '${secrets.KEY}' instead."
            )
        if parts.query:
            raise ValueError(
                f"Invalid [db] url '{safe}': a query string is not allowed "
                "(a credential-bearing '?password='/'?sslpassword=' would persist "
                "as a literal)."
            )
        return value

    @field_validator("password")
    @classmethod
    def _check_password_is_secret_ref(cls, value: str | None) -> str | None:
        if value is not None and not SECRET_REF_RE.match(value):
            # Never echo the rejected value: a literal password is exactly what
            # this rejects, and the message rides the same 422/job_logs/diagnose
            # surfaces as the url shape error (finding #6).
            raise ValueError(
                "Invalid [db] password: must be a secret reference "
                "like '${secrets.DB_PASSWORD}' or '${secrets.shared.DB_PASSWORD}' "
                "(uppercase key, set via 'nerdit secrets set') — literal "
                "passwords are not allowed in nerdit.toml."
            )
        return value

    @model_validator(mode="after")
    def _check_provider_fields(self) -> "DbBindingConfig":
        if self.provider == "managed":
            if self.database is None:
                raise ValueError(
                    "[db] provider 'managed' requires 'database' (the managed "
                    "database row's name, e.g. 'pg')."
                )
            if self.url is not None:
                raise ValueError(
                    "[db] provider 'managed' forbids 'url': the daemon composes "
                    "the endpoint from the managed database's own row."
                )
            if self.password is not None:
                raise ValueError(
                    "[db] provider 'managed' forbids 'password': the daemon "
                    "injects the minted credential."
                )
        else:  # provider == "external"
            if self.url is None:
                raise ValueError("[db] provider 'external' requires 'url'.")
            if self.password is None:
                raise ValueError(
                    "[db] provider 'external' requires 'password' (a '${secrets.KEY}' reference)."
                )
            if self.database is not None:
                raise ValueError(
                    "[db] provider 'external' forbids 'database': the target is "
                    "the external 'url', not a managed row."
                )
        return self


class ProjectConfig(BaseModel):
    """Parsed project configuration; unknown sections, including legacy batch tables, are
    ignored.
    """

    deploy: DeployConfig | None = None
    ai: dict[str, AiBindingConfig] | None = None
    db: dict[str, DbBindingConfig] | None = None

    @field_validator("ai")
    @classmethod
    def _check_ai_binding_names(
        cls, value: dict[str, AiBindingConfig] | None
    ) -> dict[str, AiBindingConfig] | None:
        if value is not None:
            for name in value:
                if not AI_BINDING_NAME_RE.match(name):
                    raise ValueError(
                        f"Invalid [ai] binding name '{name}': must be a lowercase "
                        "letter followed by up to 31 lowercase letters, digits or "
                        "'_' (it maps to NERDIT_AI_<NAME>_* env vars)."
                    )
        return value

    @field_validator("db")
    @classmethod
    def _check_db_binding_names(
        cls, value: dict[str, DbBindingConfig] | None
    ) -> dict[str, DbBindingConfig] | None:
        if value is not None:
            for name in value:
                if not DB_BINDING_NAME_RE.match(name):
                    raise ValueError(
                        f"Invalid [db] binding name '{name}': must be a lowercase "
                        "letter followed by up to 31 lowercase letters, digits or "
                        "'_' (it maps to NERDIT_DB_<NAME>_URL env vars)."
                    )
        return value


def find_project_config(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for nerdit.toml."""
    current = (start or Path.cwd()).resolve()
    while True:
        candidate = current / PROJECT_CONFIG_NAME
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def parse_ai_bindings(raw: object) -> dict[str, AiBindingConfig]:
    """Parse AI binding tables through the shared schema.

    Raises:
        ValueError: A binding has an invalid shape or fails schema validation.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            "[ai] must be a table of [ai.<name>] binding tables, e.g. "
            "[ai.default] with 'provider' and 'model' keys."
        )
    bindings: dict[str, AiBindingConfig] = {}
    for name, table in raw.items():
        if not isinstance(table, dict):
            raise ValueError(
                f"[ai.{name}] must be a table with 'provider' and 'model' keys, "
                f"got {type(table).__name__}."
            )
        bindings[name] = AiBindingConfig(**table)
    return bindings


def parse_db_bindings(raw: object) -> dict[str, DbBindingConfig]:
    """Parse database binding tables through the shared schema.

    Raises:
        ValueError: A binding has an invalid shape or fails schema validation.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            "[db] must be a table of [db.<name>] binding tables, e.g. "
            "[db.default] with a 'provider' key."
        )
    bindings: dict[str, DbBindingConfig] = {}
    for name, table in raw.items():
        if not isinstance(table, dict):
            raise ValueError(
                f"[db.{name}] must be a table with a 'provider' key, got {type(table).__name__}."
            )
        bindings[name] = DbBindingConfig(**table)
    return bindings


def load_project_config(path: Path | None = None) -> ProjectConfig | None:
    """Load and parse a nerdit.toml file.

    If *path* is None, searches upward from cwd.  Returns None if no file found.
    """
    if path is None:
        path = find_project_config()
    if path is None or not path.is_file():
        return None
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return ProjectConfig(
        deploy=DeployConfig(**data["deploy"]) if "deploy" in data else None,
        ai=parse_ai_bindings(data["ai"]) if "ai" in data else None,
        db=parse_db_bindings(data["db"]) if "db" in data else None,
    )
