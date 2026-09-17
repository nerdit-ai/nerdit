"""Validate and atomically persist daemon config while preserving unknown sections.

Keep the raw TOML dict: settings models are lossy. Validate touched sections,
write and fsync a sibling temporary file, then replace the target. Redact views
and diffs; reject auth_token writes in favor of the token API.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, ValidationError

from nerdit.config.defaults import DEFAULT_DATA_DIR, DEFAULT_LOG_LEVEL
from nerdit.config.redaction import is_secret_key, redact_section, redact_value
from nerdit.config.settings import (
    ClientSettings,
    ContainerSettings,
    DaemonSettings,
    DatabasesSettings,
    GitSettings,
    LicenseSettings,
    LinkSettings,
    McpSettings,
    ModelsSettings,
    MonitorSettings,
    NotificationsSettings,
    PostHogSettings,
    ProxySettings,
    RetentionSettings,
    SecuritySettings,
    ServicesSettings,
)
from nerdit.db.models import ConfigDiagnostic, ConfigDiffEntry


class ConfigError(Exception):
    """A config-store error mapped to a structured envelope by the route.

    `code` is the machine-readable error code (`config.unknown_section`,
    `config.invalid`, `config.forbidden`); `status_code` the HTTP status;
    `diagnostics` carries per-field validation detail for `config.invalid`.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        diagnostics: list[ConfigDiagnostic] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.hint = hint
        self.diagnostics = diagnostics or []


class _NerditTopModel(BaseModel):
    """The top-level `[nerdit]` section (carved from `NerditSettings`)."""

    data_dir: str = str(DEFAULT_DATA_DIR)
    log_level: Literal["debug", "info", "warning", "error", "critical"] = DEFAULT_LOG_LEVEL


# Section name → Pydantic model. Mirrors the mapping in ``load_settings`` so the
# API validates exactly what the daemon consumes. Daemon config only (P1).
_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "nerdit": _NerditTopModel,
    "daemon": DaemonSettings,
    "containers": ContainerSettings,
    "monitor": MonitorSettings,
    "security": SecuritySettings,
    "services": ServicesSettings,
    "proxy": ProxySettings,
    "models": ModelsSettings,
    "databases": DatabasesSettings,
    "git": GitSettings,
    "notifications": NotificationsSettings,
    "mcp": McpSettings,
    "link": LinkSettings,
    "license": LicenseSettings,
    "retention": RetentionSettings,
    "posthog": PostHogSettings,
    "client": ClientSettings,
}

# Keys whose change only takes effect after a daemon restart (no live reload in
# P1). ``auth_token`` is here for completeness but its write is refused upstream.
# ``service_port_range`` is bound once at startup by the ServiceController, so a
# change needs a restart to take effect.
_RESTART_KEYS: dict[str, frozenset[str]] = {
    # The top-level ``[nerdit]`` section. ``data_dir`` is resolved once in the
    # lifespan (database, secrets, volumes, workspaces, backups all hang off the
    # boot value) and ``log_level`` is consumed once by ``setup_logging``, so
    # both are restart-required. Note the keys live on the ``NerditSettings``
    # root, not on a ``settings.nerdit`` attribute — the doctor drift check
    # resolves that via ``restart_section_holder``.
    "nerdit": frozenset({"data_dir", "log_level"}),
    # ``upload_dir``/``max_upload_bytes`` are read off the BOOT
    # ``app.state.settings`` by every ingress (deploy, workspaces, templates,
    # ``/capabilities``), and ``upload_dir`` is additionally folded into
    # ``containers.allowed_mount_roots`` once in the lifespan; ``pid_file`` is
    # where the running daemon's pid was written, so moving it only takes
    # effect for the next start. All boot-frozen, like the four above.
    "daemon": frozenset(
        {
            "host",
            "port",
            "auth_token",
            "instance_id",
            "upload_dir",
            "max_upload_bytes",
            "pid_file",
        }
    ),
    # ``[containers]`` is captured once at boot in two places — ``DockerRuntime``
    # (``daemon/server.py``) and ``ServiceController`` (``daemon/bootstrap.py``),
    # whose copy ``core/launch.py`` reads at every launch — so a config PUT only
    # rewrites TOML until the daemon restarts. That includes the sandbox
    # hardening trio, where an untruthful ``requires_restart: false`` would read
    # as "hardening is live" when it is not. Whole-section frozenset.
    "containers": frozenset(
        {
            "default_image",
            "default_memory_limit",
            "default_cpu_limit",
            "allowed_mount_roots",
            "denied_mount_paths",
            "drop_all_caps",
            "no_new_privileges",
            "read_only_rootfs",
        }
    ),
    # The P20 trio is read off the settings captured at startup (the run route
    # off ``app.state.settings``, the release hook off the controller's copy),
    # so a config PUT only rewrites TOML until the daemon restarts. The P24b
    # cutover trio is snapshotted by ``CutoverManager`` at controller
    # construction for exactly the same reason.
    "services": frozenset(
        {
            "service_port_range",
            "max_concurrent_builds",
            "loop_interval",
            "run_timeout_max_s",
            "release_timeout_s",
            "max_concurrent_runs",
            "cutover_grace_s",
            "cutover_verify_timeout_s",
            "cutover_repoint_timeout_s",
            # (P37) The dump pair is read off the same boot snapshot: the route
            # reads ``dump_timeout_max_s`` from ``app.state.settings`` and the
            # ServiceController snapshots ``max_concurrent_dumps`` at
            # construction, exactly like the P20 run trio above.
            "dump_timeout_max_s",
            "max_concurrent_dumps",
            # The restart-policy pair is snapshotted by ``ServiceController``
            # at construction (``core/services.py``) exactly like the trio
            # above, so a PUT only rewrites TOML until the daemon restarts.
            "service_max_restarts",
            "restart_window_seconds",
        }
    ),
    # ``enable_amd`` is read once by GPU discovery in create_app(); the probe
    # budgets are captured by the ``ResourceMonitor`` at construction and
    # ``zml_smi_path`` by the boot-time discovery call, so the whole section is
    # restart-required.
    "monitor": frozenset(
        {
            "interval_seconds",
            "gpu_temp_warning",
            "gpu_temp_critical",
            "zml_smi_path",
            "enable_amd",
        }
    ),
    # (P25) The whole ``[security]`` section is bound once at startup: the
    # idempotency middleware and the SecretManager capture their setting at
    # construction, and ``token_default_ttl_s`` is read off
    # ``app.state.settings`` (the boot snapshot). Declaring all three keys keeps
    # the ``requires_restart`` answer truthful for the section rather than
    # truthful for one key and silently wrong for the others.
    "security": frozenset({"require_idempotency_key", "secrets_key_file", "token_default_ttl_s"}),
    # Proxy/controller settings are captured at startup, including URL shape.
    # Mark every key restart-required so config writes report their actual effect.
    "proxy": frozenset(
        {
            "enabled",
            "mode",
            "admin_addr",
            "https_port",
            "public_port",
            "scheme",
            "base_domain",
            "hostname_override",
            "caddy_binary",
            "mdns",
            "mdns_address",
            "dashboard_apex",
            # P25 D-P25-9: the SAN registry is read by ``_tls_subjects`` off the
            # ``ProxySettings`` captured at ProxyManager construction, and a
            # self-spawned Caddy only picks the subjects up at launch — live,
            # no-restart TLS sync is deliberately deferred.
            "extra_hostnames",
            # The whole ``[proxy.acme]`` block is one key here. The
            # diff engine compares the nested dict as a unit, so a change to any
            # sub-key reads ``requires_restart=True`` — which is the truth:
            # ``ProxyManager`` captures ``ProxySettings`` at construction and
            # the ``:80`` server is written into the bootstrap config Caddy is
            # spawned with. There is no live reload of listeners or issuers.
            "acme",
        }
    ),
    # Every ``[models]`` field follows the ``[proxy]`` classification: the daemon
    # loads settings once at startup and the P5 controllers capture
    # ``ModelsSettings`` at construction, so even the per-launch-read fields
    # (``bridge_host``, ``ollama_image``) only pick up a TOML change after a
    # restart. Flagging them keeps the CLI/API warning accurate.
    "models": frozenset(
        {
            "ollama_image",
            "bridge_host",
            "pull_timeout_s",
            "start_period_s",
            "default_backend",
            "vllm_image",
            "vllm_shm_size",
            "vllm_extra_args",
        }
    ),
    # Every ``[git]`` field follows the ``[proxy]``/``[models]`` classification:
    # settings load once at startup and the deploy routes read them per request
    # off the captured ``NerditSettings``, so a config PUT only rewrites TOML and
    # a restart is needed to pick the change up. Flagging them keeps the CLI/API
    # warning accurate.
    "git": frozenset(
        {
            "enabled",
            "allowed_hosts",
            "clone_timeout_s",
            "max_clone_bytes",
            "watch_interval_s",
            "github_clone_base_url",
        }
    ),
    # ``[notifications]``: the ``WebhookDispatcher`` is built once in
    # the lifespan and captures ``NotificationsSettings`` (targets included), so
    # a config PUT only rewrites TOML until the daemon restarts. Whole-section
    # frozenset, the ``[git]``/``[retention]`` precedent.
    "notifications": frozenset(
        {
            "enabled",
            "targets",
            "timeout_s",
            "retry_max",
            "batch_max",
            "poll_interval_s",
        }
    ),
    # Every ``[databases]`` field follows the ``[models]`` classification: the
    # daemon loads settings once at startup and the P15 ``DataController``
    # captures ``DatabasesSettings`` at construction (backends carry the pinned
    # images + probe budgets), so a config PUT only rewrites TOML and a restart
    # is needed to pick the change up. Whole-section frozenset.
    "databases": frozenset(
        {
            "default_backend",
            "postgres_image",
            "redis_image",
            "start_period_s",
            "ready_timeout_s",
        }
    ),
    # ``[mcp]`` follows the ``[git]`` classification: the mount + hard-error
    # run once in create_app(), so the flag only takes effect on restart.
    "mcp": frozenset({"http_enabled", "max_body_bytes"}),
    # ``[link]`` (P27 WP-C3): the WP-C1 tunnel client will capture
    # ``LinkSettings`` once at startup and the identity key path is resolved at
    # boot, so a config PUT only rewrites TOML — all keys are restart-required
    # in v1 per the WP-C3 checklist. Whole-section frozenset, the
    # ``[git]``/``[mcp]`` precedent.
    "link": frozenset(
        {
            "enabled",
            "relay_url",
            "key_file",
            "capability_ttl_s",
            "renew_margin_s",
            # WP-C1: the claim result (written by WP-C2's ``nerdit link``,
            # nulled by ``nerdit unlink``). The manager captures both at boot,
            # so a re-link takes effect on restart like the rest of [link].
            "node_id",
            "slug",
            # (P26 D-P26-H5) The hosted base domain the claim / ``nerdit link
            # refresh`` learns. Bound at boot with the rest of [link] — the
            # app-stream resolver captures it when the tunnel manager is built —
            # so writing it is honestly restart-required (S2).
            "nodes_base_domain",
        }
    ),
    # The license path is restart-keyed; API installs refresh its contents.
    # Callers must not assume boot-frozen license state.
    "license": frozenset({"file"}),
    # ``[retention]`` (P14b): the sweep loop captures the interval + day/count
    # thresholds at startup and ``setup_logging`` reads the rotation knobs once
    # per process, so every field only takes effect on restart. Whole-section
    # frozenset, the ``[git]``/``[mcp]`` precedent.
    "retention": frozenset(
        {
            "job_log_days",
            "audit_days",
            "audit_archive_dir",
            "sweep_interval_seconds",
            "daemon_log_max_bytes",
            "daemon_log_backups",
            "container_log_max_size",
            "container_log_max_file",
            "backup_keep_last",
            "volume_backup_keep_last",
            "dump_keep_last",
            "events_keep_last",
            "workspace_orphan_days",
        }
    ),
    # ``[posthog]`` (feat/posthog): the dashboard reads the project key/host once
    # per ``/cluster/info`` request off the captured ``NerditSettings``, and the
    # daemon loads settings once at startup, so a config PUT only rewrites TOML —
    # a restart (and a dashboard reload) is needed to pick the change up. Whole-
    # section frozenset, the ``[git]``/``[mcp]``/``[retention]`` precedent.
    "posthog": frozenset({"enabled", "project_key", "host"}),
    # ``[client]`` is deliberately absent and must stay absent: its only
    # consumer is ``get_client_config()``, re-read on every CLI invocation, so
    # ``requires_restart: false`` is already the truthful answer.
}


def restart_section_holder(settings: Any, section: str) -> Any:
    """Return the object carrying a config section's keys on a settings tree.

    Every section is a sub-model attribute except ``[nerdit]``, whose keys
    (``data_dir``, ``log_level``) live on the ``NerditSettings`` root. Callers
    comparing boot settings against a fresh load need that distinction, or a
    ``[nerdit]`` drift silently compares ``None`` to ``None``.
    """
    return settings if section == "nerdit" else getattr(settings, section, None)


# Leaf keys that may never be written through the config API (rotate via
# ``/api/tokens``). Section-qualified so only the real secret slots are blocked.
_FORBIDDEN_WRITE_KEYS: frozenset[str] = frozenset({"auth_token"})


@dataclass
class StagedConfig:
    """A validated, not-yet-written config change.

    `new_raw` is the full config dict with the target section replaced; the
    route persists it via `ConfigStore.commit` only after the dry-run /
    ETag checks pass.
    """

    section: str
    new_raw: dict[str, Any]
    diff: list[ConfigDiffEntry] = field(default_factory=list)
    requires_restart: bool = False
    restart_keys: list[str] = field(default_factory=list)


@dataclass
class StagedApply:
    """A validated multi-section apply, not yet written.

    `new_raw` is the full config dict with every mentioned section merged in
    (sections absent from the document untouched). `changed` is False when
    the staged document serializes identically to the current file — the
    idempotent no-op case.
    """

    new_raw: dict[str, Any]
    diff: list[ConfigDiffEntry] = field(default_factory=list)
    requires_restart: bool = False
    restart_keys: list[str] = field(default_factory=list)
    changed: bool = False


def _strip_none(values: dict[str, Any]) -> dict[str, Any]:
    """Drop `None` values: TOML has no null and `tomli_w` rejects `None`."""
    return {k: v for k, v in values.items() if v is not None}


def _unknown_key_hint(section: str, model: type[BaseModel], unknown: list[str]) -> str:
    """Point unknown dotted subtable keys to a whole-table write rather than scalar config set."""
    valid = f"Valid keys: {', '.join(sorted(model.model_fields))}."
    nested = sorted(
        {key.split(".", 1)[0] for key in unknown if "." in key} & set(model.model_fields)
    )
    if not nested:
        return valid
    parent = nested[0]
    return (
        f"{valid} '{parent}' is a sub-table: `nerdit config set` addresses scalars only. "
        f"Send the table as one value instead — PUT /api/config/daemon/{section} "
        f'{{"{parent}": {{...}}}} — or `nerdit config apply` a TOML file with a '
        f"[{section}.{parent}] block. Keys you omit inside the table keep their stored values."
    )


def _both_tables(stored: dict[str, Any], key: str, value: Any) -> bool:
    """Whether `key` is a sub-TABLE on both sides — the deep-merge predicate.

    Deliberately narrow: only a dict written over a stored dict merges. A body
    `None` clears, a scalar replaces, and a list replaces whole.
    """
    return isinstance(value, dict) and isinstance(stored.get(key), dict)


def _diagnostics(exc: ValidationError) -> list[ConfigDiagnostic]:
    """Project a Pydantic `ValidationError` onto the structured diagnostics list."""
    return [
        ConfigDiagnostic(
            loc=[str(part) for part in err.get("loc", ())],
            message=err.get("msg", "invalid value"),
            type=err.get("type"),
        )
        for err in exc.errors()
    ]


class ConfigStore:
    """Read/validate/write the daemon TOML config as a structured artifact."""

    def __init__(self, config_path: Path) -> None:
        self._path = Path(config_path)

    # -- introspection ---------------------------------------------------------

    @staticmethod
    def known_sections() -> list[str]:
        """Return the writable/readable daemon config sections."""
        return list(_SECTION_MODELS)

    def _require_section(self, section: str) -> type[BaseModel]:
        model = _SECTION_MODELS.get(section)
        if model is None:
            raise ConfigError(
                404,
                "config.unknown_section",
                f"Unknown config section '{section}'.",
                hint=f"Known sections: {', '.join(_SECTION_MODELS)}.",
            )
        return model

    # -- raw artifact ----------------------------------------------------------

    def load_raw(self) -> dict[str, Any]:
        """Return the full parsed TOML dict (empty when the file is absent)."""
        if not self._path.exists():
            return {}
        with open(self._path, "rb") as handle:
            return tomllib.load(handle)

    @staticmethod
    def _serialize(raw: dict[str, Any]) -> bytes:
        """Deterministically serialize a config dict to TOML bytes."""
        return tomli_w.dumps(raw).encode("utf-8")

    def current_etag(self) -> str:
        """Hash canonical parsed TOML so formatting changes do not affect concurrency checks."""
        return hashlib.sha256(self._serialize(self.load_raw())).hexdigest()

    # -- effective values ------------------------------------------------------

    def effective_section(self, section: str) -> dict[str, Any]:
        """Return a section's effective values (stored values over defaults)."""
        return self._effective_from_raw(section, self.load_raw())

    def _effective_from_raw(self, section: str, raw: dict[str, Any]) -> dict[str, Any]:
        """Effective values for `section` computed against a given raw dict."""
        model = self._require_section(section)
        stored = raw.get(section, {})
        if not isinstance(stored, dict):
            stored = {}
        return model(**stored).model_dump()

    def view_section(self, section: str) -> dict[str, Any]:
        """Return a section's effective values with secrets redacted."""
        return redact_section(self.effective_section(section))

    # -- staging / writing -----------------------------------------------------

    def stage(
        self, section: str, body: dict[str, Any], base_raw: dict[str, Any] | None = None
    ) -> StagedConfig:
        """Validate a section and compute its redacted diff without writing.

        Args:
            base_raw: Snapshot to stage against; defaults to the current file.
                Multi-section apply passes its accumulated snapshot here.

        Raises:
            ConfigError: Unknown section, forbidden auth_token write or invalid fields
                with structured diagnostics.
        """
        model = self._require_section(section)

        forbidden = sorted(k for k in body if k.lower() in _FORBIDDEN_WRITE_KEYS)
        if forbidden:
            raise ConfigError(
                403,
                "config.forbidden",
                f"Writing {', '.join(forbidden)} via the config API is not allowed.",
                hint="Rotate authentication tokens with `nerdit token` / POST /api/tokens.",
            )

        # Reject unknown keys rather than silently dropping them: Pydantic's
        # default ``extra='ignore'`` would let a typo (`prot` for `port`) validate
        # and then be dropped at persist time, so the write looks successful but
        # is a no-op. Surface it as ``config.invalid`` with per-key diagnostics.
        unknown = sorted(set(body) - set(model.model_fields))
        if unknown:
            raise ConfigError(
                422,
                "config.invalid",
                f"Unknown key(s) for section '{section}': {', '.join(unknown)}.",
                hint=_unknown_key_hint(section, model, unknown),
                diagnostics=[
                    ConfigDiagnostic(loc=[key], message="unknown field", type="extra_forbidden")
                    for key in unknown
                ],
            )

        raw = self.load_raw() if base_raw is None else base_raw
        stored_section = raw.get(section, {})
        if not isinstance(stored_section, dict):
            stored_section = {}

        # Merge dict-over-dict sub-tables one level deep, preserving omitted siblings.
        # Null clears, scalars replace and lists replace whole; partial ACME updates
        # must not reset enabled/email or downgrade existing certificates.
        body = {
            key: (
                {**stored_section[key], **value}
                if _both_tables(stored_section, key, value)
                else value
            )
            for key, value in body.items()
        }

        # Validate against stored siblings, not defaults, so partial cross-field updates
        # see the intended document. Explicit None clears nullable values; invalid nulls
        # fail. Unknown stored keys are tolerated rather than blocking unrelated writes.
        try:
            validated = model(**{**stored_section, **body})
        except ValidationError as exc:
            raise ConfigError(
                422,
                "config.invalid",
                f"Invalid configuration for section '{section}'.",
                diagnostics=_diagnostics(exc),
            ) from exc

        dumped = validated.model_dump()
        # Persist only the keys the caller actually supplied, with validated
        # (coerced) values and ``None`` stripped — preserving the rest of the
        # file untouched.
        persisted_section = _strip_none({k: dumped[k] for k in body if k in dumped})
        # Keys the caller explicitly set to null (e.g. ``base_domain=null``): TOML
        # has no null, so we DELETE them from the section — reverting to the model
        # default — instead of silently dropping the ``None`` and leaving the old
        # value in place. Without this, a nullable field could be set but never
        # cleared through the API. Only nullable fields reach here as ``None``
        # (non-nullable ones fail validation above).
        cleared = {k for k in body if k in dumped and dumped[k] is None}

        # The section EXACTLY as it will land on disk: the stored keys with the
        # supplied ones written over them, minus the cleared (null'd) ones.
        # ``new_raw[section]`` below is this same dict — one construction, used
        # both to validate and to persist, so the two can never drift.
        merged = {**stored_section, **persisted_section}
        for key in cleared:
            merged.pop(key, None)

        # Revalidate the effective merged section: null deletion restores model defaults,
        # which can change cross-field validity. Unknown stored keys remain on disk but
        # are ignored by validation; unknown input keys already failed above.
        # Discard this model dump: persist only the supplied changes, not new defaults.
        try:
            model(**merged)
        except ValidationError as exc:
            raise ConfigError(
                422,
                "config.invalid",
                f"Invalid configuration for section '{section}' after merging with "
                "the stored values.",
                hint=(
                    "The result of the merge violates a cross-field rule; supply the "
                    "sibling key(s) in the same request."
                ),
                diagnostics=_diagnostics(exc),
            ) from exc

        old_values = self._effective_from_raw(section, raw)
        restart_key_set = _RESTART_KEYS.get(section, frozenset())
        # A cleared (null'd) key reverts to the model default: the diff must
        # show the *effective* resulting value, not a literal ``None``.
        new_values = dict(dumped)
        if cleared:
            defaults = model().model_dump()
            for key in cleared:
                new_values[key] = defaults.get(key)
        diff = self._diff(
            section,
            old_values,
            new_values,
            set(body),
            stored_keys=set(stored_section),
            cleared=cleared,
            restart_key_set=restart_key_set,
        )

        new_raw = dict(raw)
        # Preserve omitted siblings, especially write-blocked auth_token.
        # Replacing the whole section could erase auth and enable anonymous admin access.
        new_raw[section] = merged

        restart_keys = sorted(entry.key for entry in diff if entry.requires_restart)

        return StagedConfig(
            section=section,
            new_raw=new_raw,
            diff=diff,
            requires_restart=bool(restart_keys),
            restart_keys=restart_keys,
        )

    def stage_many(self, sections: dict[str, dict[str, Any]]) -> StagedApply:
        """Stage sections in sorted order against one accumulated snapshot, without writing.

        Aggregate all invalid-section diagnostics under section-qualified locations
        into one 422; use unknown_section when only unknown sections fail. Forbidden
        auth_token writes fail immediately with 403. Validation is all-or-nothing.
        """
        base_raw = self.load_raw()
        current = dict(base_raw)
        diff: list[ConfigDiffEntry] = []
        restart_keys: list[str] = []
        diagnostics: list[ConfigDiagnostic] = []
        only_unknown_sections = True
        for name in sorted(sections):
            body = sections[name]
            try:
                staged = self.stage(name, body, base_raw=current)
            except ConfigError as exc:
                if exc.status_code == 403:
                    raise
                if exc.code == "config.unknown_section":
                    diagnostics.append(
                        ConfigDiagnostic(loc=[name], message=exc.message, type="unknown_section")
                    )
                else:
                    only_unknown_sections = False
                    diagnostics.extend(
                        ConfigDiagnostic(loc=[name, *d.loc], message=d.message, type=d.type)
                        for d in exc.diagnostics
                    )
                continue
            current = staged.new_raw
            diff.extend(staged.diff)
            restart_keys.extend(staged.restart_keys)
        if diagnostics:
            code = "config.unknown_section" if only_unknown_sections else "config.invalid"
            raise ConfigError(
                422,
                code,
                "Invalid configuration document.",
                hint="Diagnostics cover every invalid section; nothing was applied.",
                diagnostics=diagnostics,
            )
        changed = self._serialize(current) != self._serialize(base_raw)
        restart_keys = sorted(set(restart_keys))
        return StagedApply(
            new_raw=current,
            diff=diff,
            requires_restart=bool(restart_keys),
            restart_keys=restart_keys,
            changed=changed,
        )

    @staticmethod
    def _diff(
        section: str,
        old_values: dict[str, Any],
        new_values: dict[str, Any],
        supplied: set[str],
        *,
        stored_keys: set[str] | None = None,
        cleared: set[str] | None = None,
        restart_key_set: frozenset[str] = frozenset(),
    ) -> list[ConfigDiffEntry]:
        """Build the redacted diff for the keys the caller supplied."""
        stored_keys = stored_keys or set()
        cleared = cleared or set()
        entries: list[ConfigDiffEntry] = []
        for key in supplied:
            if key not in new_values:
                continue
            old = old_values.get(key)
            new = new_values.get(key)
            if old == new:
                continue
            op: Literal["add", "change", "delete"]
            if key in cleared:
                op = "delete"
            elif key in stored_keys:
                op = "change"
            else:
                op = "add"
            entries.append(
                ConfigDiffEntry(
                    key=f"{section}.{key}",
                    old=redact_value(key, old),
                    new=redact_value(key, new),
                    section=section,
                    op=op,
                    requires_restart=key in restart_key_set,
                    secret=is_secret_key(key),
                )
            )
        entries.sort(key=lambda entry: entry.key)
        return entries

    def commit(self, staged: StagedConfig | StagedApply) -> str:
        """Atomically persist a staged change; return the new ETag.

        Writes to a temp file in the target directory, `fsync`s it, then
        `os.replace`s it over the config file (atomic on POSIX). When the
        staged raw is identical to the current file the write is skipped
        entirely (no-op short-circuit) and the unchanged ETag is returned.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize(staged.new_raw)
        if payload == self._serialize(self.load_raw()):
            return hashlib.sha256(payload).hexdigest()
        tmp_path = self._path.with_name(f"{self._path.name}.tmp-{os.getpid()}")
        # Create the temp with 0600: ``os.replace`` adopts the temp inode's
        # permissions, so the config file (which holds ``auth_token`` in plain
        # text) must never be left world-readable by the umask default.
        if tmp_path.exists():
            tmp_path.unlink()
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self._path)
        return hashlib.sha256(payload).hexdigest()
