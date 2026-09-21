"""Pydantic Settings for Nerdit configuration."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import secrets
import tomllib
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
    model_validator,
)

from nerdit.config.defaults import (
    DEFAULT_ALLOWED_MOUNT_ROOTS,
    DEFAULT_CADDY_BINARY,
    DEFAULT_DATA_DIR,
    DEFAULT_DATABASES_BACKEND,
    DEFAULT_DB_READY_TIMEOUT_S,
    DEFAULT_DB_START_PERIOD_S,
    DEFAULT_DENIED_MOUNT_PATHS,
    DEFAULT_DUMP_TIMEOUT_MAX_S,
    DEFAULT_GIT_ALLOWED_HOSTS,
    DEFAULT_GIT_CLONE_TIMEOUT_S,
    DEFAULT_GPU_TEMP_CRITICAL,
    DEFAULT_GPU_TEMP_WARNING,
    DEFAULT_HOST,
    DEFAULT_IMAGE,
    DEFAULT_LINK_CAPABILITY_TTL_S,
    DEFAULT_LINK_RENEW_MARGIN_S,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENT_DUMPS,
    DEFAULT_MAX_CONCURRENT_RUNS,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_MODEL_PULL_TIMEOUT_S,
    DEFAULT_MODEL_START_PERIOD_S,
    DEFAULT_MODELS_BACKEND,
    DEFAULT_MODELS_BRIDGE_HOST,
    DEFAULT_MONITOR_INTERVAL,
    DEFAULT_OLLAMA_IMAGE,
    DEFAULT_PORT,
    DEFAULT_POSTGRES_IMAGE,
    DEFAULT_PROXY_ADMIN_ADDR,
    DEFAULT_PROXY_HTTPS_PORT,
    DEFAULT_REDIS_IMAGE,
    DEFAULT_RELEASE_TIMEOUT_S,
    DEFAULT_RESTART_WINDOW_SECONDS,
    DEFAULT_RUN_TIMEOUT_MAX_S,
    DEFAULT_SERVICE_MAX_RESTARTS,
    DEFAULT_SERVICE_PORT_RANGE,
    DEFAULT_UPLOAD_DIR,
    DEFAULT_VLLM_IMAGE,
    DEFAULT_VLLM_SHM_SIZE,
    default_bridge_binding,
)
from nerdit.config.project import _DNS_LABEL_RE, SECRET_REF_RE


class DaemonSettings(BaseModel):
    """Settings for the nerditd HTTP server."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    pid_file: str = "~/.nerdit/nerditd.pid"
    auth_token: str | None = None
    upload_dir: str = str(DEFAULT_UPLOAD_DIR)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    # Labels this daemon's containers (``nerdit-instance=<id>``) so multiple
    # daemons can share one Docker host without their zombie sweeps colliding —
    # each sweep only lists/kills containers carrying its own instance_id.
    instance_id: str = "default"

    @field_validator("auth_token")
    @classmethod
    def _check_auth_token(cls, value: str | None) -> str | None:
        # Compared against a bearer decoded from the Authorization header,
        # which is ASCII by every generator we ship (`secrets.token_urlsafe`).
        # A hand-edited non-ASCII token would authenticate nothing — the header
        # arrives latin-1-decoded, so its bytes never round-trip — while
        # costing a confusing 403 on every request. Fail at load instead.
        if value is not None and not value.isascii():
            raise ValueError(
                "auth_token must be ASCII (it is compared against an "
                "Authorization header, which carries no other charset). "
                "Edit [daemon].auth_token in the daemon config by hand: every "
                "CLI verb loads this file, so no shipped command can repair it "
                "('nerdit token' only displays the configured value, and "
                "'nerdit init --auth-token-only' leaves any non-empty token "
                "alone — clear the line first to have a fresh one minted)."
            )
        return value

    @field_validator("instance_id")
    @classmethod
    def _check_instance_id(cls, value: str) -> str:
        # Interpolated verbatim into a docker label filter
        # (``nerdit-instance=<id>``); a comma/'='/space would produce a
        # malformed multi-value filter and silently break sweep isolation, so
        # constrain it to a safe, non-empty token that fails fast at load.
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(
                f"instance_id '{value}' must be non-empty and match [A-Za-z0-9_.-]+ "
                "(it becomes a docker label value)."
            )
        return value


class ContainerSettings(BaseModel):
    """Settings for container runtime behavior."""

    default_image: str = DEFAULT_IMAGE
    # Default container resource limits applied to workloads when the workload
    # does not override them. ``None`` means "no limit" (Docker default).
    default_memory_limit: str | None = None
    default_cpu_limit: float | None = None
    # --- Sandbox hardening (P1 / S5) — policy data only ---
    # Tier-B allowlist: roots under which non-admin workloads may mount host
    # paths (the daemon's own upload/cache dirs; a caller's own workspace is
    # never auto-appended).
    allowed_mount_roots: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_MOUNT_ROOTS)
    )
    # Tier-A denylist: host paths no caller (admin included) may ever mount.
    denied_mount_paths: list[str] = Field(default_factory=lambda: list(DEFAULT_DENIED_MOUNT_PATHS))
    # Capability / privilege hardening applied to every launched container.
    drop_all_caps: bool = True
    no_new_privileges: bool = True
    # Read-only rootfs is opt-in: app images may write to ``/`` (Open Q7).
    read_only_rootfs: bool = False


class MonitorSettings(BaseModel):
    """Settings for the GPU resource monitor."""

    interval_seconds: int = DEFAULT_MONITOR_INTERVAL
    gpu_temp_warning: int = DEFAULT_GPU_TEMP_WARNING
    gpu_temp_critical: int = DEFAULT_GPU_TEMP_CRITICAL
    zml_smi_path: str = "zml-smi"
    # Gates whether discovered AMD GPUs are marked schedulable (allocatable to
    # workloads) or stay inventory/metrics only. Read once at startup by GPU
    # discovery, so it is restart-required.
    enable_amd: bool = False


class SecuritySettings(BaseModel):
    """Security policy, including optional mandatory idempotency keys.

    Keyless writes are allowed by default. An unset secrets_key_file resolves to
    <data_dir>/secrets.key, outside the ciphertext directory for separate custody.
    """

    require_idempotency_key: bool = False
    secrets_key_file: str | None = None
    # (P25 D-P25-1) Default TTL applied to NEW tokens whose create request omits
    # ``expires_in_s``. ``None`` = tokens never expire unless asked to (zero
    # breakage; expiry is opt-in per token, and an operator can make a finite TTL
    # the site policy with one config write). An explicit ``expires_in_s: null``
    # in the body is "no expiry", NOT "use this default". Read per-request off
    # the settings captured at boot ⇒ restart-required.
    token_default_ttl_s: int | None = Field(default=None, ge=60, le=31_536_000)


class ServicesSettings(BaseModel):
    """Service ports, restart budgets, one-off run limits and cutover deadlines.

    Validate the inclusive loopback port range before allocation, excluding the
    default daemon port. Run/release, cutover and database dump/restore limits
    are captured at startup and require restart to change.
    """

    service_port_range: str = DEFAULT_SERVICE_PORT_RANGE
    service_max_restarts: int = DEFAULT_SERVICE_MAX_RESTARTS
    restart_window_seconds: int = DEFAULT_RESTART_WINDOW_SECONDS
    # Global build-concurrency cap: the ServiceController holds one
    # asyncio.Semaphore of this size across all app builds. Restart-required.
    max_concurrent_builds: int = Field(default=2, ge=1)
    # Sole WorkloadManager reconcile-tick interval, in seconds. Bound at startup
    # (the loop captures it once), so it is restart-required.
    loop_interval: float = 2.0
    # Server cap on a run request's ``timeout_s``. The run route rejects a
    # larger value with 422 ``run.timeout_too_large`` — the timeout is enforced
    # server-side, never merely honored from the client.
    run_timeout_max_s: int = Field(default=DEFAULT_RUN_TIMEOUT_MAX_S, ge=1)
    # Wall-clock bound on one ``[deploy].release`` execution. Past it the
    # release container is killed and the generation settles ``failed``.
    release_timeout_s: int = Field(default=DEFAULT_RELEASE_TIMEOUT_S, ge=1)
    # Daemon-wide cap on route-initiated runs; a run over the cap
    # gets 409 ``run.too_many_in_flight``. Releases are exempt so unrelated runs
    # cannot fail a deploy — and, deliberately, they are bounded by no other cap
    # either: ``max_concurrent_builds`` is already released by the time a
    # release container starts.
    max_concurrent_runs: int = Field(default=DEFAULT_MAX_CONCURRENT_RUNS, ge=1)
    # (P37 / D-P37-8) Server cap on a dump/restore request's ``timeout_s``, the
    # exact ``run_timeout_max_s`` mirror: the route rejects a larger value with
    # 422 ``dump.timeout_too_large`` naming this key. Floor 60 s — a dump that
    # cannot be given a minute is a misconfiguration, not a tight budget.
    dump_timeout_max_s: int = Field(default=DEFAULT_DUMP_TIMEOUT_MAX_S, ge=60)
    # (P37 / D-P37-9) Daemon-wide cap on in-flight dump/restore siblings (409
    # ``dump.too_many_in_flight``); a SEPARATE pool from ``max_concurrent_runs``.
    max_concurrent_dumps: int = Field(default=DEFAULT_MAX_CONCURRENT_DUMPS, ge=1)
    # The health-gated cutover trio. Read off the settings the
    # ServiceController captured at startup (CutoverManager snapshots them at
    # construction), so all three are restart-required.
    #
    # ``cutover_grace_s`` — how long a green with NO health spec must simply
    # stay ``running`` to count as verified.
    cutover_grace_s: int = Field(default=10, ge=1, le=300)
    # Total budget for one verify (launch → first 2xx / grace), after which the
    # green is destroyed and the generation settles ``cutover_failed``.
    cutover_verify_timeout_s: int = Field(default=90, ge=5, le=900)
    # Budget for the WP5.4 f2 dial READ-BACK (~3 proxy reconcile ticks).
    # Exceeded ⇒ the (g') settle: the pointer is unwound and blue, which never
    # stopped serving, is never destroyed.
    cutover_repoint_timeout_s: int = Field(default=15, ge=1, le=120)

    @field_validator("service_port_range")
    @classmethod
    def _check_port_range(cls, value: str) -> str:
        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid service_port_range '{value}'. Expected 'lo-hi' (e.g. '9400-9499')."
            )
        try:
            lo, hi = int(parts[0].strip()), int(parts[1].strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid service_port_range '{value}'. Expected integer bounds 'lo-hi'."
            ) from exc
        if not (1 <= lo <= 65535 and 1 <= hi <= 65535):
            raise ValueError(f"service_port_range '{value}' bounds must be within 1-65535.")
        if lo > hi:
            raise ValueError(f"service_port_range '{value}' has lo > hi.")
        if lo <= DEFAULT_PORT <= hi:
            raise ValueError(
                f"service_port_range '{value}' must not contain the daemon port {DEFAULT_PORT}."
            )
        return value


#: Let's Encrypt production — the default CA for ``[proxy.acme].directory``.
DEFAULT_ACME_DIRECTORY = "https://acme-v02.api.letsencrypt.org/directory"
#: Let's Encrypt staging — untrusted leaves, but effectively no rate limits.
#: Named here (rather than spelled out in prose) so the docs, the tests and an
#: operator's config all quote **one** string.
LE_STAGING_DIRECTORY = "https://acme-staging-v02.api.letsencrypt.org/directory"

#: A bare mailbox: no display name, no ``mailto:``, no whitespace. Deliberately
#: coarse — the CA is the real authority on the address; this only refuses the
#: shapes Caddy would hand to the CA verbatim and get a 400 for.
_ACME_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ProxyAcmeSettings(BaseModel):
    """Optional public certificates through ACME HTTP-01; disabled by default.

    When enabled, Caddy adds an HTTP challenge/redirect listener and ACME policies
    for flagged domains. Account keys and certificates live under data_dir/caddy.
    All settings require restart; unknown nested keys are rejected.

    ca_root_file supplies trust for a private directory endpoint. Validate its
    absolute path, but leave existence checks to Caddy on the daemon host.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # The CA account contact — REQUIRED when enabled (see ``_check_contact``).
    # It reaches the CA and nothing else: never an audit body, an event, a
    # remediation detail, a doctor detail or ``/proxy/status``.
    email: str | None = None
    directory: str = DEFAULT_ACME_DIRECTORY
    # The HTTP-01 / redirect listener. Wildcard-bound by necessity — the CA
    # must reach it from the internet — which is why it carries no route to any
    # upstream. 80 is privileged: a per-user install needs ``setcap`` on the
    # caddy binary or a port above 1023 (the ``acme_http_port`` doctor row says
    # which).
    http_port: int = 80
    # 308 → https on that listener. Off ⇒ it answers 404 to everything but the
    # solver path.
    http_redirect: bool = True
    # Absolute path to a PEM bundle trusted for ``directory`` (private CA).
    ca_root_file: str | None = None

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Drop nested None leaves before TOML serialization; TOML has no null value."""
        return {key: value for key, value in handler(self).items() if value is not None}

    @field_validator("email")
    @classmethod
    def _check_email(cls, value: str | None) -> str | None:
        """Validate and strip a bare mailbox without echoing it in API diagnostics."""
        if value is None:
            return None
        stripped = value.strip()
        # The regex alone would accept "mailto:ops@example.com" ("mailto:ops"
        # is a legal local part to it). Caddy adds the ``mailto:`` prefix itself
        # when it builds the account contact, so a hand-written one yields
        # ``mailto:mailto:…`` and a CA 400 — refuse it by name.
        if stripped.lower().startswith("mailto:") or not _ACME_EMAIL_RE.match(stripped):
            raise ValueError(
                "email must be a bare address like 'ops@example.com' — no display "
                "name, no 'mailto:' prefix, no whitespace"
            )
        return stripped

    @field_validator("directory")
    @classmethod
    def _check_directory(cls, value: str) -> str:
        """Require an absolute HTTPS URL, allowing HTTP only on loopback; never echo credentials."""
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(
                "directory must be an absolute URL to the CA's ACME directory "
                "(e.g. 'https://acme-v02.api.letsencrypt.org/directory')"
            )
        if parts.username or parts.password:
            raise ValueError(
                "directory must not embed credentials (userinfo) — an ACME "
                "directory is a public endpoint"
            )
        try:
            # Validate parsed ports and explicitly reject zero; urlsplit accepts :0.
            # Missing ports remain valid. Reject here rather than leave issuance stuck pending.
            port = parts.port
        except ValueError:
            raise ValueError("directory port must be an integer in 1-65535") from None
        if port is not None and not (1 <= port <= 65535):
            raise ValueError("directory port must be an integer in 1-65535")
        if parts.scheme == "http" and parts.hostname.strip("[]").lower() not in LOOPBACK_HOSTS:
            raise ValueError(
                "directory must use https — plain http is accepted only for a "
                "loopback host (localhost / 127.0.0.1 / ::1), the local-CA test carve-out"
            )
        return value

    @field_validator("http_port")
    @classmethod
    def _check_http_port(cls, value: int) -> int:
        if not (1 <= value <= 65535):
            raise ValueError(f"http_port '{value}' must be within 1-65535.")
        return value

    @field_validator("ca_root_file")
    @classmethod
    def _check_ca_root_file(cls, value: str | None) -> str | None:
        """Require an absolute path without stat: Caddy resolves relative paths in its own
        directory.
        """
        if value is None:
            return None
        if not value.strip():
            raise ValueError("ca_root_file must not be empty — omit the key instead.")
        if not Path(value).is_absolute():
            raise ValueError(f"ca_root_file '{value}' must be an absolute path to a PEM bundle.")
        return value

    @model_validator(mode="after")
    def _check_contact(self) -> ProxyAcmeSettings:
        """Require contact information for enabled ACME accounts so recovery and expiry notices
        work.
        """
        if self.enabled and not self.email:
            raise ValueError(
                "email is required when acme is enabled — the CA records it as "
                "the account contact for expiry and revocation notices"
            )
        return self


class ProxySettings(BaseModel):
    """Optional Caddy URL layer, configured through its admin API.

    Disabled or unavailable proxies leave services on loopback. Mode selects path
    or subdomain routing; all settings are captured at startup and require restart.
    Advertised URLs use HTTPS. hostname_override supplies a LAN name/IP and CA SAN,
    otherwise the machine hostname is used.

    Port 443 needs bind privileges: system units provide CAP_NET_BIND_SERVICE;
    user units and manually started daemons need setcap or an unprivileged port.
    Bind failures enter respawn backoff and report unavailable.

    Enabled ACME adds a public HTTP listener for challenges and optional redirects.
    It requires the proxy enabled and an HTTP port distinct from the HTTPS port.
    """

    enabled: bool = False
    mode: Literal["path", "subdomain"] = "path"
    admin_addr: str = DEFAULT_PROXY_ADMIN_ADDR
    https_port: int = DEFAULT_PROXY_HTTPS_PORT
    # Port ADVERTISED in public URLs when it differs from the bound one —
    # i.e. an external reverse proxy fronts the embedded Caddy (public 443 →
    # loopback ``https_port``). ``None`` (default) advertises ``https_port``.
    # Advertise-only: the embedded Caddy still binds ``https_port``.
    public_port: int | None = None
    # https only — the bootstrap is a TLS-only listener (see class docstring).
    scheme: Literal["https"] = "https"
    base_domain: str | None = None
    hostname_override: str | None = None
    caddy_binary: str = DEFAULT_CADDY_BINARY
    # Advertise the daemon's name over mDNS so ``<name>.local`` resolves from
    # other LAN machines. Fresh installs enable it; the fallback remains off
    # for existing configurations without this key. Path mode
    # only — subdomain-mode service names (``<service>.<base>``) are
    # multi-label and standard mDNS resolvers will not answer them.
    mdns: bool = False
    # Explicit IPv4 to advertise; ``None`` = autodetect the primary LAN IP.
    mdns_address: str | None = None
    # P9.5: serve the dashboard/API at the proxy apex (``https://<host>/``) via a
    # lowest-priority catch-all route to the daemon, so one HTTPS name covers
    # both the dashboard and every service. Opt-in (default ``False``) so
    # existing proxy users are unaffected — the apex keeps 404-ing until set.
    # Path mode only: subdomain mode Host-matches per service and leaves the
    # apex routeless by design (the flag is a no-op there).
    dashboard_apex: bool = False
    # Extra names/IPs receive internal-CA certificates, not generated URLs.
    # Changes require restart. In subdomain mode they reach the apex only;
    # service Host matchers still use base_domain.
    extra_hostnames: list[str] = Field(default_factory=list)
    # Public certificates via ACME HTTP-01. A nested model
    # rather than flat ``acme_*`` keys so the TOML reads ``[proxy.acme]`` and so
    # the whole block is one restart key in ``config/store.py``.
    acme: ProxyAcmeSettings = Field(default_factory=ProxyAcmeSettings)

    @field_validator("https_port", "public_port")
    @classmethod
    def _check_https_port(cls, value: int | None, info: ValidationInfo) -> int | None:
        if value is not None and not (1 <= value <= 65535):
            raise ValueError(f"{info.field_name} '{value}' must be within 1-65535.")
        return value

    @field_validator("admin_addr")
    @classmethod
    def _check_admin_loopback(cls, value: str) -> str:
        """Restrict the full-control Caddy admin API to localhost, 127.0.0.1 or ::1."""
        host, sep, port = value.rpartition(":")
        if not sep or not host:
            raise ValueError(
                f"admin_addr '{value}' must be 'host:port' on loopback (e.g. 'localhost:2019')."
            )
        if not port.isdigit() or not (1 <= int(port) <= 65535):
            raise ValueError(f"admin_addr '{value}' has an invalid port.")
        normalized = host.strip("[]").lower()  # strip IPv6 brackets
        if normalized not in LOOPBACK_HOSTS:
            raise ValueError(
                f"admin_addr host '{host}' must be loopback "
                "(localhost / 127.0.0.1 / ::1) — the Caddy admin API must not be LAN-exposed."
            )
        return value

    @field_validator("base_domain")
    @classmethod
    def _check_base_domain(cls, value: str | None) -> str | None:
        """Validate the subdomain-mode wildcard base (D5, P3.5)."""
        if value is None:
            return None
        _validate_dns_name(value, field_name="base_domain")
        return value

    @field_validator("hostname_override")
    @classmethod
    def _check_hostname_override(cls, value: str | None) -> str | None:
        """Validate the LAN-resolvable hostname/IP override (D5, P3.5)."""
        if value is None:
            return None
        _validate_dns_name(value, field_name="hostname_override")
        return value

    @field_validator("extra_hostnames")
    @classmethod
    def _check_extra_hostnames(cls, value: list[str]) -> list[str]:
        """Validate lowercase certificate SAN names/IPs without silent normalization.

        Reject schemes, ports, paths, whitespace and explicit wildcards; the wildcard
        subject derives from base_domain. Limit input to 32 entries before deduplicating
        in order.
        """
        if len(value) > _MAX_EXTRA_HOSTNAMES:
            raise ValueError(
                f"extra_hostnames has {len(value)} entries; at most "
                f"{_MAX_EXTRA_HOSTNAMES} are allowed."
            )
        for entry in value:
            if not entry:
                raise ValueError("extra_hostnames must not contain an empty entry.")
            if "*" in entry:
                raise ValueError(
                    f"extra_hostnames entry '{entry}' must not be a wildcard — the "
                    "'*.<base>' subject is derived from [proxy].base_domain in subdomain "
                    "mode and is never listed here."
                )
            _validate_dns_name(entry, field_name="extra_hostnames")
        return list(dict.fromkeys(value))

    @field_validator("mdns_address")
    @classmethod
    def _check_mdns_address(cls, value: str | None) -> str | None:
        """Require a routable IPv4 literal — a loopback advertisement would
        publish an address other LAN machines cannot reach."""
        if value is None:
            return None
        try:
            addr = ipaddress.IPv4Address(value)
        except ValueError as exc:
            raise ValueError(
                f"mdns_address '{value}' must be an IPv4 literal (e.g. '192.168.1.20')."
            ) from exc
        if addr.is_loopback or addr.is_unspecified:
            raise ValueError(
                f"mdns_address '{value}' must be a LAN-reachable address, not loopback/unspecified."
            )
        return value

    @model_validator(mode="after")
    def _check_acme(self) -> ProxySettings:
        """When ACME is enabled, require the proxy and distinct HTTP/HTTPS ports.

        Disabled ACME settings are inert. Reject collisions before Caddy can fail its
        entire listener startup.
        """
        if not self.acme.enabled:
            return self
        if not self.enabled:
            raise ValueError(
                "[proxy.acme].enabled requires [proxy].enabled — ACME issues "
                "certificates for the embedded proxy's own listener"
            )
        if self.acme.http_port == self.https_port:
            raise ValueError(
                f"[proxy.acme].http_port {self.acme.http_port} must differ from "
                f"[proxy].https_port — one listener cannot serve both roles"
            )
        # An ACME/admin port clash also prevents Caddy startup.
        # The separate daemon-port collision is checked by doctor, outside this model.
        admin_port = self.admin_addr.rpartition(":")[2]
        if admin_port.isdigit() and self.acme.http_port == int(admin_port):
            raise ValueError(
                f"[proxy.acme].http_port {self.acme.http_port} must differ from the port "
                f"in [proxy].admin_addr — Caddy cannot bind one port twice"
            )
        return self


# P25 D-P25-9: a sanity bound on the SAN registry — every entry lands in one
# internal-CA leaf, and a runaway list is an operator mistake, not a use case.
#: The three loopback authorities, shared by every "only on this box" gate:
#: the Caddy admin-addr validator here and the link claim's http-for-loopback
#: carve-out (``daemon/schemas/link.py``, which additionally accepts RFC 6761
#: ``*.localhost`` names on top of this set).
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_MAX_EXTRA_HOSTNAMES = 32


def _is_ip_literal(value: str) -> bool:
    """Detect IP literals so suffix fields can reject addresses allowed by proxy hostname rules."""
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _validate_dns_name(value: str, *, field_name: str) -> None:
    """Require a bare lowercase DNS name; IPv4 literals are also accepted.

    Reject schemes, ports, wildcards and whitespace rather than silently normalize.
    """
    if value != value.strip() or any(ch.isspace() for ch in value):
        raise ValueError(f"{field_name} '{value}' must not contain whitespace.")
    if "://" in value:
        raise ValueError(f"{field_name} '{value}' must be a bare hostname, not a URL (no scheme).")
    if ":" in value:
        raise ValueError(f"{field_name} '{value}' must not include a port; use a bare hostname.")
    if value != value.lower():
        raise ValueError(f"{field_name} '{value}' must be lowercase.")
    if value.startswith(".") or value.endswith("."):
        raise ValueError(f"{field_name} '{value}' must not have a leading or trailing dot.")
    if len(value) > 253:
        raise ValueError(f"{field_name} '{value}' exceeds the 253-character DNS name limit.")
    for label in value.split("."):
        if not _DNS_LABEL_RE.match(label):
            raise ValueError(
                f"{field_name} '{value}' has an invalid label '{label}' — DNS labels are "
                "lowercase alphanumeric with internal hyphens only, 1-63 chars, no wildcards "
                "(e.g. 'apps.lan.local')."
            )


class ModelsSettings(BaseModel):
    """Model images, readiness limits and app-container bridge reachability.

    Auto bridge uses 172.17.0.1 for bind/advertise on Linux; macOS adds no bind and
    advertises host.docker.internal. Explicit hosts serve both roles. Pull timeout
    bounds weight downloads; start period allows health-check grace.
    All settings are captured at startup and require restart.
    """

    ollama_image: str = DEFAULT_OLLAMA_IMAGE
    bridge_host: str = DEFAULT_MODELS_BRIDGE_HOST
    pull_timeout_s: int = Field(default=DEFAULT_MODEL_PULL_TIMEOUT_S, ge=1)
    start_period_s: int = Field(default=DEFAULT_MODEL_START_PERIOD_S, ge=0)
    # (P11) Backend selection. ``default_backend`` serves a model when the serve
    # request omits ``--backend``. vLLM knobs mirror the Ollama ones.
    default_backend: str = DEFAULT_MODELS_BACKEND
    vllm_image: str = DEFAULT_VLLM_IMAGE
    vllm_shm_size: str = DEFAULT_VLLM_SHM_SIZE
    vllm_extra_args: list[str] = Field(default_factory=list)

    @field_validator("default_backend")
    @classmethod
    def _check_default_backend(cls, value: str) -> str:
        """Only the built-in backend names are selectable as the default."""
        allowed = {"ollama", "vllm"}
        stripped = value.strip()
        if stripped not in allowed:
            raise ValueError(
                f"default_backend '{value}' is not a known backend "
                f"(one of {', '.join(sorted(allowed))})."
            )
        return stripped

    @field_validator("bridge_host")
    @classmethod
    def _check_bridge_host(cls, value: str) -> str:
        """Reject empty or all-interfaces bridge hosts.

        The bridge host becomes an extra port *bind* address: `0.0.0.0`
        (or `::`) would expose model endpoints on the LAN, defeating the
        loopback-only MVP posture. `"auto"` is the platform-resolved default.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("bridge_host must not be empty.")
        if stripped in {"0.0.0.0", "::", "[::]"}:
            raise ValueError(
                f"bridge_host '{value}' would bind model ports on all interfaces "
                "(LAN-exposed); use the docker bridge gateway IP instead."
            )
        return stripped

    @property
    def bridge_bind_ip(self) -> str | None:
        """Extra host IP model ports are bound on (`None` => loopback only).

        `"auto"` resolves per platform (Linux: `172.17.0.1`; macOS: no
        extra bind — Docker Desktop's VM proxy reaches loopback-published
        ports). An explicit `bridge_host` is used verbatim.
        """
        if self.bridge_host == "auto":
            return default_bridge_binding()[0]
        return self.bridge_host

    @property
    def bridge_advertise_host(self) -> str:
        """Host advertised to app containers in resolved `[ai.*]` base URLs.

        `"auto"` resolves per platform (Linux: `172.17.0.1`; macOS:
        `host.docker.internal`). An explicit `bridge_host` is used verbatim.
        """
        if self.bridge_host == "auto":
            return default_bridge_binding()[1]
        return self.bridge_host


class DatabasesSettings(BaseModel):
    """Managed database backends, images and readiness limits.

    Default backend must be built-in. start_period_s is startup grace;
    ready_timeout_s bounds each Postgres SSLRequest or Redis PING probe.
    Reuse models.bridge_host for reachability. All settings require restart.
    """

    default_backend: str = DEFAULT_DATABASES_BACKEND
    postgres_image: str = DEFAULT_POSTGRES_IMAGE
    redis_image: str = DEFAULT_REDIS_IMAGE
    start_period_s: int = Field(default=DEFAULT_DB_START_PERIOD_S, ge=0)
    ready_timeout_s: int = Field(default=DEFAULT_DB_READY_TIMEOUT_S, ge=1)

    @field_validator("default_backend")
    @classmethod
    def _check_default_backend(cls, value: str) -> str:
        """Only the built-in backend names are selectable as the default."""
        allowed = {"postgres", "redis"}
        stripped = value.strip()
        if stripped not in allowed:
            raise ValueError(
                f"default_backend '{value}' is not a known backend "
                f"(one of {', '.join(sorted(allowed))})."
            )
        return stripped


class GitSettings(BaseModel):
    """Git deployment policy, captured at startup and requiring restart.

    Enabled gates Git deploys and templates. Deny unlisted egress hosts to prevent
    access to arbitrary internal hosts. clone_timeout_s caps subprocess duration;
    max_clone_bytes bounds the checkout, matching ZIP upload limits by default.
    """

    enabled: bool = True
    allowed_hosts: list[str] = Field(default_factory=lambda: list(DEFAULT_GIT_ALLOWED_HOSTS))
    clone_timeout_s: int = Field(default=DEFAULT_GIT_CLONE_TIMEOUT_S, ge=1)
    max_clone_bytes: int = Field(default=DEFAULT_MAX_UPLOAD_BYTES, ge=1)
    # How often the GitWatch poller re-resolves the remote HEAD of an
    # ``auto_deploy`` app. Floored at 15 s so a misconfiguration cannot turn the
    # poller into a remote hammer.
    watch_interval_s: int = Field(default=60, ge=15)
    # Origin for GitHub clone URLs and cloud owner/repo nudges; restart-required.
    # Installation-token resolution enforces github.com unless the explicit dev
    # environment guard allows another origin. Config loading checks syntax only,
    # so CLI reads work without weakening the credential-use boundary.
    github_clone_base_url: str = "https://github.com"

    @field_validator("github_clone_base_url")
    @classmethod
    def _check_github_clone_base_url(cls, value: str) -> str:
        """Validate an HTTPS origin without userinfo, path, query or fragment.

        Host trust is enforced when installation tokens are resolved, not while loading
        settings. Non-github.com origins cannot receive those tokens unless the explicit
        NERDIT_DEV_GITHUB_CLONE_BASE environment guard is nonempty. This keeps ordinary
        CLI config reads working while credential use fails closed at deployment.
        """
        stripped = value.strip().rstrip("/")
        parts = urlsplit(stripped)
        if parts.scheme != "https":
            raise ValueError("github_clone_base_url must be an https origin.")
        if not parts.hostname:
            raise ValueError("github_clone_base_url must include a host.")
        if parts.username or parts.password:
            raise ValueError("github_clone_base_url must not embed credentials.")
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ValueError("github_clone_base_url must be a bare origin — no path.")
        return stripped

    @property
    def github_host(self) -> str:
        """The lower-case host of `github_clone_base_url` (the D-GH-3 pin)."""
        host = urlsplit(self.github_clone_base_url).hostname
        assert host is not None  # guaranteed by the validator
        return host.lower()

    @field_validator("allowed_hosts")
    @classmethod
    def _normalize_allowed_hosts(cls, value: list[str]) -> list[str]:
        """Strip + lowercase entries; reject empties.

        Hosts are matched case-insensitively against the URL hostname, so they
        are stored lowercase. A blank entry would be a silent no-op slot, so it
        is rejected rather than dropped.
        """
        normalized: list[str] = []
        for entry in value:
            stripped = entry.strip()
            if not stripped:
                raise ValueError("allowed_hosts must not contain empty entries.")
            normalized.append(stripped.lower())
        return normalized


# D-P24-7 (countersigned): the ``allow_http`` allow-set is an EXPLICIT membership
# table — never ``ipaddress.is_private``, which admits 169.254.169.254 (cloud
# metadata), 0.0.0.0, 192.0.0.1, CGNAT and the fe80::/10 + fc00::/7 blocks.
# Widening this tuple must always be a visible diff.
_ALLOWED_HTTP_NETS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)

# Header-name grammar for ``auth_header_name`` (RFC 7230 token, narrowed).
_HEADER_NAME_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class NotificationTarget(BaseModel):
    """A webhook destination with optional shared-secret authentication.

    URLs are persisted and exposed in config reads/backups. Userinfo is rejected;
    query tokens are discouraged but not blocked. Use auth_header_ref for secrets
    resolved at send time. Reject unknown nested keys to catch unsigned or
    unfiltered target typos.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    # Empty ⇒ every durable event type is delivered.
    events: list[str] = Field(default_factory=list)
    # ``${secrets.shared.KEY}`` — HMAC key for the X-Nerdit-Signature header.
    secret_ref: str | None = None
    # ``${secrets.shared.KEY}`` — sent verbatim as ``auth_header_name``.
    auth_header_ref: str | None = None
    auth_header_name: str = "Authorization"
    # D-P24-7 carve-out (countersigned): plain http, IP literals only.
    allow_http: bool = False

    @model_serializer(mode="wrap")
    def _drop_unset_refs(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Drop nested None references before TOML serialization; TOML has no null value."""
        return {key: value for key, value in handler(self).items() if value is not None}

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        """Absolute http(s) URL with a host and no embedded credentials.

        Plain `http` is NOT refused here — the scheme × `allow_http` rule is
        the model validator's (a field-level reject would shadow it).
        """
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("url must be an absolute http(s) URL with a host")
        if parts.username or parts.password:
            raise ValueError(
                "url must not embed credentials (userinfo); "
                "authenticate with auth_header_ref instead"
            )
        try:
            # Force urlsplit to validate nonnumeric and out-of-range ports here.
            _ = parts.port
        except ValueError:
            raise ValueError("url port must be an integer in 0-65535") from None
        return value

    @field_validator("secret_ref", "auth_header_ref")
    @classmethod
    def _check_secret_ref(cls, value: str | None) -> str | None:
        """Only `${secrets.shared.KEY}` references — never a literal value."""
        if value is None:
            return value
        match = SECRET_REF_RE.match(value)
        if match is None:
            raise ValueError("must be a ${secrets.shared.KEY} reference")
        if match.group(1) != "shared":
            raise ValueError(
                "must name the shared scope (${secrets.shared.KEY}) — a "
                "per-service secret has no meaning for a daemon-level target"
            )
        return value

    @field_validator("auth_header_name")
    @classmethod
    def _check_auth_header_name(cls, value: str) -> str:
        """A plain header token, outside the reserved `X-Nerdit-` namespace."""
        if not _HEADER_NAME_RE.match(value):
            raise ValueError("auth_header_name must match ^[A-Za-z0-9-]{1,64}$")
        if value.lower().startswith("x-nerdit-"):
            raise ValueError("the X-Nerdit- header namespace is reserved")
        return value

    @field_validator("events")
    @classmethod
    def _check_events(cls, value: list[str]) -> list[str]:
        """Every entry must name a declared durable event type."""
        # Deferred import: ``core.eventlog`` pulls in ``core.jobconfig`` at
        # runtime, so a module-level config→core edge could grow into a cycle.
        from nerdit.core.eventlog import EVENT_TYPES

        for entry in value:
            if entry not in EVENT_TYPES:
                raise ValueError(f"unknown event type '{entry}' — see the /events type vocabulary")
        return value

    @model_validator(mode="after")
    def _check_scheme(self) -> NotificationTarget:
        """https always; http only under `allow_http` for an allow-set IP literal."""
        parts = urlsplit(self.url)
        if parts.scheme == "https":
            return self
        if not self.allow_http:
            raise ValueError(
                "url scheme must be https (or http with allow_http for a "
                "loopback or RFC1918 IP literal)"
            )
        try:
            addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(
                parts.hostname or ""
            )
        except ValueError:
            raise ValueError(
                "allow_http requires an IP-literal host inside the loopback/"
                "RFC1918 allow-set — a hostname re-resolves (DNS rebinding) and "
                "is refused even with allow_http"
            ) from None
        # Normalize IPv4-mapped IPv6 (::ffff:a.b.c.d) BEFORE the membership test
        # so a mapped metadata address cannot sneak past an IPv6 branch.
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if not any(addr in net for net in _ALLOWED_HTTP_NETS):
            raise ValueError(
                f"http target address {addr} is outside the allowed loopback/"
                "RFC1918 set (127.0.0.0/8, ::1, 10.0.0.0/8, 172.16.0.0/12, "
                "192.168.0.0/16)"
            )
        return self

    def cursor_id(self) -> str:
        """Return the 16-hex identity used for this target's delivery cursor.

        Hash URL, sorted unique events, secret/auth reference names and header name;
        never secret values. allow_http is implied by the URL and excluded. Editing
        identity reseeds at the feed head without replaying backlog; old cursor rows
        remain as small orphans.
        """
        canonical = json.dumps(
            {
                "auth_header_name": self.auth_header_name,
                "auth_header_ref": self.auth_header_ref,
                "events": sorted(set(self.events)),
                "secret_ref": self.secret_ref,
                "url": self.url,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class NotificationsSettings(BaseModel):
    """Webhook delivery, disabled with no targets by default; all changes require restart."""

    enabled: bool = False
    targets: list[NotificationTarget] = Field(default_factory=list)
    timeout_s: float = Field(default=10.0, ge=1.0, le=30.0)
    retry_max: int = Field(default=5, ge=1, le=10)
    batch_max: int = Field(default=50, ge=1, le=200)
    poll_interval_s: float = Field(default=5.0, ge=1.0, le=300.0)

    @model_validator(mode="after")
    def _check_duplicate_targets(self) -> NotificationsSettings:
        """Reject identical delivery identities, which would share a cursor and starve one
        target.
        """
        seen: dict[str, int] = {}
        for idx, target in enumerate(self.targets):
            tid = target.cursor_id()
            if tid in seen:
                raise ValueError(
                    f"targets[{seen[tid]}] and targets[{idx}] are identical "
                    "(same url, events filter, auth refs and header name) — "
                    "duplicates would share one delivery cursor; remove one or "
                    "differentiate them"
                )
            seen[tid] = idx
        return self


class McpSettings(BaseModel):
    """Settings for the streamable-HTTP MCP transport (P13c §3).

    ``http_enabled`` mounts the FastMCP streamable-HTTP app at ``/api/mcp``
    on the daemon (default off ⇒ 404, zero new surface). Requires
    ``[daemon].auth_token`` — the daemon refuses to start otherwise. Like
    ``[git]``, the field is bound at startup, so changes are flagged
    restart-required in ``config/store.py``.

    ``max_body_bytes`` (P29) is the transport's request-body cap, bound once
    into ``_TransportGuard`` at mount time in ``build_http_app`` — restart-
    required for the same reason. The default is **deliberately unchanged**
    from the pre-P29 bare constant (``MAX_MCP_BODY_BYTES``, which stays as the
    constructor default): P29 §0.1 found the model's own per-call output, not
    this cap, to be the binding constraint, so this is an operator escape
    hatch, not a raise.
    """

    http_enabled: bool = False
    max_body_bytes: int = Field(default=1_048_576, ge=65_536, le=16_777_216)


class LinkSettings(BaseModel):
    """Outbound relay settings, disabled by default; enabling requires a relay URL.

    The tunnel opens no inbound port and is structurally limited to submitter;
    admin access cannot be enabled through config. All settings require restart.

    Unset key_file uses <data_dir>/link/node.key. Identity code enforces a 0600
    key and a 0700 daemon-created parent; custom parents are warned about, not
    chmodded. Null deletion restores the data-dir-relative default.
    """

    enabled: bool = False
    # No default relay: the daemon dials out, so an operator must name one.
    # Empty is allowed while disabled (the model validator ties the two).
    relay_url: str = ""
    # ``None`` ⇒ ``<data_dir>/link/node.key``; when set, must be absolute.
    key_file: str | None = None
    # Hard-capped at the ADR-W2 relay-side ≤15-min accepted look-ahead: a
    # capability the relay would refuse at hello cannot be configured.
    capability_ttl_s: int = Field(default=DEFAULT_LINK_CAPABILITY_TTL_S, ge=60, le=900)
    # Renewal fires at ``expires_at - renew_margin_s`` (D-R8).
    renew_margin_s: int = Field(default=DEFAULT_LINK_RENEW_MARGIN_S, ge=10)
    # Claim writes identity; unlink clears it. Startup requires enabled, relay_url
    # and node_id. Enabled-but-unlinked is valid: warn and leave the tunnel stopped.
    # Identity changes require restart.
    node_id: str | None = None
    slug: str | None = None
    # (P26 D-P26-H5) The cloud's hosted base domain (``nodes.nerdit.ai`` in
    # production), written by the claim from the claim response and by ``nerdit
    # link refresh`` — never guessed, so a daemon that has not learned it simply
    # cannot advertise a hosted URL. Hosted names are one label under it:
    # ``https://<app>--<slug>.<nodes_base_domain>/``. ``None`` on a node linked
    # before P26 until it refreshes. Restart-required like the rest of [link].
    nodes_base_domain: str | None = None

    @field_validator("relay_url")
    @classmethod
    def _check_relay_url(cls, value: str) -> str:
        """Empty ⇒ unset; otherwise a clean `wss`/`https` URL with a host.

        Whitespace anywhere is rejected rather than stripped: a URL that is
        silently repaired here would differ from the one the operator reads back
        out of `GET /config/daemon`, and an embedded space is always a paste
        accident.
        """
        if not value:
            return value
        if value != value.strip() or any(ch.isspace() for ch in value):
            raise ValueError("relay_url must not contain whitespace.")
        parts = urlsplit(value)
        if parts.scheme not in ("wss", "https"):
            # No interpolation: a credential pasted AS the scheme
            # (sk-…://host) must not ride back in the message.
            raise ValueError(
                "relay_url scheme is not supported — use 'wss' "
                "(the tunnel dial) or 'https' (a relay fronted by an HTTP/2 "
                "endpoint, D-R3)."
            )
        if not parts.hostname:
            raise ValueError("relay_url must include a host.")
        # ``urlsplit`` defers port parsing to attribute ACCESS — a URL like
        # ``wss://relay:notaport/link`` splits cleanly and only blows up when
        # someone finally reads ``.port`` (the WebSocket dial, at the next
        # restart, terminally). Force the parse here so an unusable port is a
        # 422 at claim/config time, not a wedged manager later.
        try:
            port = parts.port
        except ValueError as exc:
            raise ValueError("relay_url port must be a number in 1-65535.") from exc
        if port == 0:
            raise ValueError("relay_url port must be a number in 1-65535.")
        if parts.username or parts.password:
            raise ValueError(
                "relay_url must not embed credentials (userinfo) — the daemon "
                "authenticates via the link handshake, never the URL."
            )
        if parts.fragment:
            raise ValueError("relay_url must not carry a fragment.")
        if parts.query:
            # A ``?token=…`` is credentials by another name, and unlike
            # userinfo it would round-trip verbatim through GET
            # /config/daemon, write diffs, the config.apply audit row and the
            # persisted TOML ([link] has no SECRET_LEAF_KEYS leaf).
            raise ValueError(
                "relay_url must not carry a query string — the daemon "
                "authenticates via the link handshake, never the URL."
            )
        return value

    @field_validator("node_id", "slug")
    @classmethod
    def _check_claim_identity(cls, value: str | None) -> str | None:
        """Require nonempty whitespace-free identity, or None; never silently alter signed wire
        values.
        """
        if value is None:
            return value
        if not value:
            raise ValueError("must not be empty when set (omit the key, or set it to null).")
        if value != value.strip() or any(ch.isspace() for ch in value):
            raise ValueError("must not contain whitespace.")
        return value

    @field_validator("nodes_base_domain")
    @classmethod
    def _check_nodes_base_domain(cls, value: str | None) -> str | None:
        """Require a bare lowercase DNS suffix, or None.

        Reuse proxy hostname rules but reject wildcards and IP literals: neither can
        host computed app subdomains. Empty strings are invalid; null deletion is the
        explicit way to forget the domain.
        """
        if value is None:
            return value
        if not value:
            raise ValueError("must not be empty when set (omit the key, or set it to null).")
        if "*" in value:
            raise ValueError("must not contain a wildcard — it is a concrete DNS name.")
        if _is_ip_literal(value):
            raise ValueError("must be a DNS name, not an IP address.")
        _validate_dns_name(value, field_name="nodes_base_domain")
        return value

    @field_validator("key_file")
    @classmethod
    def _check_key_file(cls, value: str | None) -> str | None:
        """Unset ⇒ the data_dir default; otherwise a non-empty absolute path.

        Only the structural check lives here — the `0o600` file / `0o700`
        parent permissions are enforced by `core/link/identity.py`, which is
        the only writer of the key.
        """
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "key_file must not be empty when set (empty/unset => <data_dir>/link/node.key)."
            )
        try:
            expanded = Path(stripped).expanduser()
        except RuntimeError as exc:
            # ``~missinguser/...``: pathlib raises RuntimeError, which Pydantic
            # does NOT wrap into ValidationError — it would escape the config
            # route as a 500 instead of the structured 422. Surface it as an
            # ordinary invalid-value.
            raise ValueError(f"key_file '{value}' has an unresolvable '~user' component.") from exc
        if not expanded.is_absolute():
            raise ValueError(
                f"key_file '{value}' must be an absolute path when set "
                "(empty/unset => <data_dir>/link/node.key)."
            )
        return stripped

    @model_validator(mode="after")
    def _check_link(self) -> LinkSettings:
        """Cross-field rules: a usable relay, and a renewal that can fire.

        `enabled` without a `relay_url` would boot a tunnel client with
        nowhere to dial; `renew_margin_s >= capability_ttl_s` would schedule
        the D-R8 renewal at or before the moment the capability was minted.
        """
        if self.enabled and not self.relay_url:
            raise ValueError(
                "[link].enabled requires relay_url to be set (the daemon dials "
                "out; there is no default relay)."
            )
        if self.renew_margin_s >= self.capability_ttl_s:
            raise ValueError(
                f"renew_margin_s ({self.renew_margin_s}) must be strictly less "
                f"than capability_ttl_s ({self.capability_ttl_s}) — renewal at "
                "expires_at - renew_margin_s (D-R8) must fire after minting."
            )
        return self


class LicenseSettings(BaseModel):
    """Path to the optional product license, separate from repository licensing.

    Unset file uses <data_dir>/license.jws; absence means unlicensed local use.
    Grace and trusted keys are fixed policy. License code enforces safe reads,
    0600 creation and atomic installation.

    Path changes require restart; API installation refreshes live content
    immediately. Onboarding does not install a product license.
    """

    # ``None`` ⇒ ``<data_dir>/license.jws``; when set, must be absolute.
    file: str | None = None

    @field_validator("file")
    @classmethod
    def _check_file(cls, value: str | None) -> str | None:
        """Validate an optional absolute license path without reading or exposing file contents."""
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "file must not be empty when set (empty/unset => <data_dir>/license.jws)."
            )
        if any(ord(char) < 32 or ord(char) == 127 for char in stripped):
            # Reject controls before os.open can raise ValueError rather than OSError.
            # Never echo the offending value into diagnostics or logs.
            raise ValueError("file must not contain control characters.")
        try:
            expanded = Path(stripped).expanduser()
        except RuntimeError as exc:
            # ``~missinguser/...``: pathlib raises RuntimeError, which Pydantic
            # does NOT wrap into ValidationError — it would escape the config
            # route as a 500 instead of the structured 422.
            raise ValueError(f"file '{value}' has an unresolvable '~user' component.") from exc
        if not expanded.is_absolute():
            raise ValueError(
                f"file '{value}' must be an absolute path when set "
                "(empty/unset => <data_dir>/license.jws)."
            )
        return stripped


class RetentionSettings(BaseModel):
    """Retention and log-rotation policy, captured at startup and requiring restart.

    Covers job logs, archive-first audit pruning, backups, workspace orphans and
    file/container log caps. Backup pruning defaults off; enabled audit pruning
    retains at least seven days.
    """

    # Prunes ``job_logs`` rows older than this, for every workload kind.
    # 0 = never prune.
    job_log_days: int = Field(default=14, ge=0)
    # Archive-first (D-H): audit rows are only pruned after being exported to a
    # fsynced JSONL.gz archive. 0 = never; floored at 7 when nonzero.
    audit_days: int = 90
    # Where audit archives are written; ``""`` ⇒ ``<data_dir>/archive``. When set,
    # must be absolute (model-level). The data_dir-relative guards (not under the
    # secrets dir, not under ``<data_dir>/services``) are enforced where the sweep
    # consumes this setting — they need the data_dir the model cannot see.
    audit_archive_dir: str = ""
    sweep_interval_seconds: int = Field(default=3600, ge=60)
    # 0 = no file rotation (daemon logs stay on stderr → boot.log only). When
    # >0, ``daemon_log_backups`` must be >= 1 (model validator below): the stdlib
    # handler never rotates with 0 backups.
    daemon_log_max_bytes: int = Field(default=10_485_760, ge=0)
    daemon_log_backups: int = Field(default=3, ge=0)
    # Docker json-file per-container log size cap (e.g. "10m"); ``\\d+[kmg]``.
    # (F7) ``""`` is the explicit opt-out: no ``log_config`` is set on the
    # container at all, so it inherits the Docker daemon's default log driver.
    container_log_max_size: str = "10m"
    container_log_max_file: int = Field(default=3, ge=1)
    # When >0, the sweep deletes the oldest backup tars beyond N. 0 = keep all.
    backup_keep_last: int = Field(default=0, ge=0)
    # (P15) Per-service keep-last for the P15 backup-v2 volume tars
    # (``nerdit-volumes-<service>-*.tar.gz``). Default 0 = never swept: these
    # tars carry database data + SCRAM verifiers and accumulate forever until
    # configured (stated honestly in the CLI hint + docs, the P14c M4 rule).
    volume_backup_keep_last: int = Field(default=0, ge=0)
    # (P37 / D-P37-7) Per-service keep-last for dump tars. Default 5, unlike
    # ``backup_keep_last``: dumps are the flavour agents and cron repeat, so
    # accumulate-forever plus a loop is disk fill. 0 = never sweep.
    dump_keep_last: int = Field(default=5, ge=0)
    # (P24a / D-P24-2) Keep-last-N for the durable ``events`` feed. A **real**
    # bound, not the ``backup_keep_last`` 0-default: the feed is chatty and
    # carries no keys, so leaving it unbounded would be the M4 honesty rule read
    # backwards. Pruned by a plain chunked delete — deliberately NOT the
    # archive-first D-H sweep the audit table gets. 0 = never prune.
    events_keep_last: int = Field(default=10000, ge=0)
    # (P29 / D-P29-8) Age in days after which an ORPHAN agent workspace (a
    # ``<data_dir>/workspaces/<name>`` whose name matches no service row) is
    # removed. A workspace whose service is live is never swept regardless of
    # age. 0 = never sweep.
    workspace_orphan_days: int = Field(default=30, ge=0)

    @field_validator("audit_days")
    @classmethod
    def _check_audit_days(cls, value: int) -> int:
        """0 disables pruning; any nonzero value is floored at 7.

        A short nonzero window would prune audit rows almost as fast as they are
        written, defeating the tamper-evidence trail; the floor keeps a
        meaningful history while still bounding growth.
        """
        if value < 0:
            raise ValueError("audit_days must be >= 0 (0 = never prune).")
        if value != 0 and value < 7:
            raise ValueError("audit_days must be 0 (never prune) or >= 7.")
        return value

    @field_validator("audit_archive_dir")
    @classmethod
    def _check_audit_archive_dir(cls, value: str) -> str:
        """Empty ⇒ default (`<data_dir>/archive`); otherwise absolute.

        Only the structural absolute-path check lives here: the data_dir-relative
        guards (reject under the secrets dir / under `<data_dir>/services`) need
        the daemon's `data_dir` and are enforced by the sweep consumer.
        """
        stripped = value.strip()
        if not stripped:
            return ""
        if not Path(stripped).is_absolute():
            raise ValueError(
                f"audit_archive_dir '{value}' must be an absolute path when set "
                "(empty ⇒ <data_dir>/archive)."
            )
        return stripped

    @field_validator("container_log_max_size")
    @classmethod
    def _check_container_log_max_size(cls, value: str) -> str:
        """Require a Docker size (integer plus k/m/g), or empty to inherit its log driver.

        A cap forces json-file and overrides custom drivers. Empty skips log_config
        entirely, trading the per-container cap for the daemon's driver. Default: 10m.
        """
        stripped = value.strip()
        if not stripped:
            return ""
        if not re.fullmatch(r"\d+[kmg]", stripped):
            raise ValueError(
                f"container_log_max_size '{value}' must be an integer followed by "
                "'k', 'm', or 'g' (e.g. '10m'), or '' to disable the cap and keep "
                "the Docker daemon's default log driver."
            )
        return stripped

    @model_validator(mode="after")
    def _check_daemon_log_rotation(self) -> RetentionSettings:
        """Require at least one backup when file-log rotation is enabled.

        RotatingFileHandler with backupCount=0 never truncates or renames, so an apparent
        cap would allow unbounded growth. max_bytes=0 disables file logging entirely.
        """
        if self.daemon_log_max_bytes > 0 and self.daemon_log_backups < 1:
            raise ValueError(
                "daemon_log_backups must be >= 1 when daemon_log_max_bytes > 0 "
                "(RotatingFileHandler never rotates with 0 backups, so nerditd.log "
                "would grow without bound); set daemon_log_max_bytes = 0 to disable "
                "file logging instead."
            )
        return self


class PostHogSettings(BaseModel):
    """Dashboard product analytics with a publishable client project key.

    The project key is safe to expose through cluster info and must not be named
    api_key, which redaction would mask. Never store personal server-side keys here.
    All changes require restart.
    """

    # Master switch. Analytics initialize in the browser only when this is true
    # AND ``project_key`` is set. Off by default (opt-in): no deployment phones
    # home until an operator sets BOTH ``[posthog].enabled = true`` and a
    # ``project_key``. Nothing is emitted otherwise.
    enabled: bool = False
    # Publishable PostHog *project* key (``phc_…``). Unset by default, so a
    # ``null``/absent key leaves analytics inert — the config store's null-delete
    # reverts to this ``None`` default (no bundled key to fall back onto), so
    # clearing the key through the API durably turns analytics off. An operator
    # sets it to their project's key (with ``enabled = true``) to opt in.
    project_key: str | None = None
    # PostHog ingest host. US cloud by default; EU is https://eu.i.posthog.com,
    # or a self-hosted origin.
    host: str = "https://us.i.posthog.com"


class ClientSettings(BaseModel):
    """Settings for remote daemon connection (CLI side)."""

    remote_host: str | None = None
    remote_port: int = DEFAULT_PORT
    auth_token: str | None = None


class NerditSettings(BaseModel):
    """Root settings model loaded from `~/.nerdit/config.toml`."""

    data_dir: str = str(DEFAULT_DATA_DIR)
    log_level: Literal["debug", "info", "warning", "error", "critical"] = DEFAULT_LOG_LEVEL
    daemon: DaemonSettings = DaemonSettings()
    containers: ContainerSettings = ContainerSettings()
    monitor: MonitorSettings = MonitorSettings()
    security: SecuritySettings = SecuritySettings()
    services: ServicesSettings = ServicesSettings()
    proxy: ProxySettings = ProxySettings()
    models: ModelsSettings = ModelsSettings()
    databases: DatabasesSettings = DatabasesSettings()
    git: GitSettings = GitSettings()
    notifications: NotificationsSettings = NotificationsSettings()
    mcp: McpSettings = McpSettings()
    link: LinkSettings = LinkSettings()
    license: LicenseSettings = LicenseSettings()
    retention: RetentionSettings = RetentionSettings()
    posthog: PostHogSettings = PostHogSettings()
    client: ClientSettings = ClientSettings()


def generate_auth_token() -> str:
    """Generate a secure auth token (43 chars, URL-safe base64)."""
    return secrets.token_urlsafe(32)


def get_client_config(
    config_path: Path | None = None,
) -> tuple[str, int, str | None]:
    """Return host, port and token for the remote connection or local daemon.

    Local dialing uses loopback and daemon.port; daemon.host is a bind address,
    which may be 0.0.0.0 and is not a valid local destination policy.
    """
    settings = load_settings(config_path)
    if settings.client.remote_host:
        return (
            settings.client.remote_host,
            settings.client.remote_port,
            settings.client.auth_token,
        )
    return (DEFAULT_HOST, settings.daemon.port, settings.daemon.auth_token)


def load_settings(config_path: Path | None = None) -> NerditSettings:
    """Load settings from TOML config file, falling back to defaults."""
    if config_path is None:
        config_path = Path("~/.nerdit/config.toml").expanduser()

    if not config_path.exists():
        return NerditSettings()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Map top-level [nerdit] section
    nerdit_data = data.get("nerdit", {})
    settings_dict: dict = {}
    if "data_dir" in nerdit_data:
        settings_dict["data_dir"] = nerdit_data["data_dir"]
    if "log_level" in nerdit_data:
        settings_dict["log_level"] = nerdit_data["log_level"]

    # Map subsections. Explicit allow-list, so a section this build does not
    # know is parsed and ignored rather than refused — the upgrade contract for
    # a fielded config.toml that still carries a removed section (``[telemetry]``,
    # deleted before the first public release).
    if "daemon" in data:
        settings_dict["daemon"] = data["daemon"]
    if "containers" in data:
        settings_dict["containers"] = data["containers"]
    if "monitor" in data:
        settings_dict["monitor"] = data["monitor"]
    if "security" in data:
        settings_dict["security"] = data["security"]
    if "services" in data:
        settings_dict["services"] = data["services"]
    if "proxy" in data:
        settings_dict["proxy"] = data["proxy"]
    if "models" in data:
        settings_dict["models"] = data["models"]
    if "databases" in data:
        settings_dict["databases"] = data["databases"]
    if "git" in data:
        settings_dict["git"] = data["git"]
    if "notifications" in data:
        settings_dict["notifications"] = data["notifications"]
    if "mcp" in data:
        settings_dict["mcp"] = data["mcp"]
    if "link" in data:
        settings_dict["link"] = data["link"]
    if "license" in data:
        settings_dict["license"] = data["license"]
    if "retention" in data:
        settings_dict["retention"] = data["retention"]
    if "posthog" in data:
        settings_dict["posthog"] = data["posthog"]
    if "client" in data:
        settings_dict["client"] = data["client"]

    return NerditSettings(**settings_dict)
