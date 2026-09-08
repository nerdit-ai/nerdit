"""Converge database-backed service routes and report proxy state.

`supervisor` owns Caddy process lifecycle; `tls` owns certificate policy.
Reconcile is authoritative; inline register/deregister calls are best-effort
and never raise. Disabled or unavailable Caddy leaves services on loopback.
Settings are captured at construction, so mode changes require restart.

Route specs carry their own shape: custom domains always use Host matchers.
Each service owns its default route plus `nerdit-route-<svc>@<domain>` entries;
pruning and ordering derive ownership from these IDs without caching domains.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from nerdit.config.defaults import DEFAULT_HOST, DEFAULT_PORT
from nerdit.config.project import SECRET_REF_RE
from nerdit.utils.certs import ca_fingerprint
from nerdit.utils.install_layout import resolve_caddy_binary

from . import edgeauth
from .admin import CaddyAdmin
from .certs import CertStatus, acme_cert_path, acme_storage_key, inspect_cert
from .edgeauth import EdgeAuthInvalid, EdgeAuthMaterial, EdgeAuthSpec, load_edge_auth
from .routing import (
    _APEX_ID,
    LiveRoute,
    RouteSpec,
    _domain_route_id,
    _route_id,
    _split_route_id,
    auth_fingerprint,
    generate_route,
    ordering_violations,
)
from .supervisor import CaddySupervisorMixin
from .tls import CaddyTlsMixin, partition_domains

if TYPE_CHECKING:
    from nerdit.config.settings import ProxySettings
    from nerdit.db.queries import Queries
    from nerdit.db.rows import ServiceDomain

logger = logging.getLogger(__name__)

#: Host of Let's Encrypt's PRODUCTION ACME directory. Compared as a HOST rather
#: than as the whole URL so a trailing slash or an `/acme/directory` variant
#: still trips the rate-limit warning in `ProxyManager.start`.
_LE_PRODUCTION_HOST = "acme-v02.api.letsencrypt.org"


class EdgeAuthUnresolved(Exception):  # noqa: N818 — a retry condition, not an error
    """A well-formed `edge_auth` declaration whose secret does not resolve.

    Private to the proxy plane: every caller routes it into the same
    fail-closed path as `nerdit.core.proxy.edgeauth.EdgeAuthInvalid` —
    the service is excluded from the reconcile desired set (so the prune loop
    removes any live route, **including one that was serving openly**) and the
    inline `register` fast-path refuses. Carries the secret KEY NAME so the
    log line and `/diagnose` can tell the operator what to set; the value is
    never read into it.
    """

    def __init__(self, service_name: str, key: str) -> None:
        self.service_name = service_name
        self.key = key
        super().__init__(
            f"[deploy] edge_auth secret '{key}' is not set — run: "
            f"nerdit secrets set {service_name} {key}=..."
        )


def _ref_key_name(ref: str) -> str:
    """The KEY name inside a `${secrets[.shared].KEY}` reference (never the value).

    `EdgeAuthSpec.password_ref` always matched the grammar (`load_edge_auth`
    re-checks it), so the fallback is defensive only — and it deliberately
    yields a placeholder rather than echoing an ungrammatical ref into a log.
    """
    match = SECRET_REF_RE.fullmatch(ref)
    return match.group(2) if match else "?"


def _desired_fingerprint(spec: RouteSpec) -> str | None:
    """The fingerprint the live route must carry for *spec*.

    `None` ⇔ the spec carries no auth, i.e. the route must carry no
    `authentication` handler — the comparison is symmetric with
    `nerdit.core.proxy.routing._extract_auth_fingerprint`'s `None`, and
    ONE function computes both sides so the two can never disagree.
    """
    if spec.auth is None:
        return None
    return auth_fingerprint(spec.auth.user, spec.auth.bcrypt_hash)


def _spec_domain(spec: RouteSpec) -> str | None:
    """The custom domain a spec serves, or `None` for a default route.

    One reader for the audit `domain` parameter, so a subdomain-mode DEFAULT
    route — which also carries a `host` — can never be mislabelled as a
    custom domain.
    """
    return spec.host if spec.kind == "domain" else None


class ProxyState(str, Enum):
    """The observable lifecycle state of the embedded Caddy (P13b, 1.8).

    A first-class projection of `ProxyManager`'s internal flags for the
    read-only `GET /proxy/status` surface, replacing the log-only
    `_foreign_warned` latch as the *queryable* signal (that latch is kept, but
    solely for log de-duplication).
    """

    disabled = "disabled"  # not settings.enabled
    no_binary = "no_binary"  # enabled, binary resolution failed
    starting = "starting"  # enabled, no successful tick yet
    backoff = "backoff"  # respawn attempts > 0, waiting
    foreign_conflict = "foreign_conflict"  # :2019 answered by a non-nerdit process
    available = "available"  # up and ours


class ProxyManager(CaddySupervisorMixin, CaddyTlsMixin):
    """Owns the embedded Caddy process and reconciles service routes.

    Process lifecycle (spawn/kill/respawn-backoff/adopt — the §1.3 patch
    points) lives on `nerdit.core.proxy.supervisor.CaddySupervisorMixin`;
    TLS subjects/bootstrap-config/adoption-sync live on
    `nerdit.core.proxy.tls.CaddyTlsMixin`; this class owns route
    shaping and the reconcile loop, and is the sole place state is
    initialised (both mixins declare none).
    """

    def __init__(
        self,
        queries: Queries,
        settings: ProxySettings,
        *,
        hostname: str,
        data_dir: Path,
        audit: bool = True,
        admin: CaddyAdmin | None = None,
        dashboard_upstream: str = f"{DEFAULT_HOST}:{DEFAULT_PORT}",
        secret_resolver: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self._queries = queries
        self._settings = settings
        self._hostname = hostname
        # P9.5: pre-resolved `host:port` the apex catch-all reverse-proxies to
        # (the daemon serving the dashboard + API). The host/port policy lives in
        # `server.py`; this module only dials what it is handed.
        self._dashboard_upstream = dashboard_upstream
        self._data_dir = Path(data_dir).expanduser()
        self._audit = audit
        self._admin = admin or CaddyAdmin(settings.admin_addr)
        # Resolve the binary once; `enabled` re-uses it so the gate is cheap.
        # A frozen install prefers the Caddy bundled in its own
        # tarball — the one tested combination for that release — but ONLY while
        # `[proxy].caddy_binary` is untouched; an explicit value always wins,
        # and a source checkout resolves through `PATH` exactly as before.
        self._binary = resolve_caddy_binary(settings.caddy_binary)
        self._pid_file = self._data_dir / "caddy.pid"
        self._log_file = self._data_dir / "caddy.log"
        self._bootstrap_file = self._data_dir / "caddy-bootstrap.json"
        self._storage_root = self._data_dir / "caddy"
        self._proc: subprocess.Popen[bytes] | None = None
        self._available = False
        # TLS-on-adoption sync guard: True once the adopted Caddy's TLS
        # subjects are known-current. Reset wherever `_available` drops so a
        # re-adoption re-checks; without it the _ensure_alive adopt branch (every
        # reconcile tick, ~5s) would re-verify TLS per tick.
        self._tls_synced = False
        # Digest of the `apps.tls` subtree the latch above was
        # set for. `_tls_synced` alone answered "has TLS been pushed at least
        # once"; with custom domains the DESIRED subtree changes whenever a
        # domain is added or removed, so the latch has to be keyed on content or
        # a new domain would never get a certificate.
        self._tls_hash: str | None = None
        # Tri-state latch for the `nerdit-acme-http`
        # listener: `None` = not yet determined for the current availability
        # episode, `True`/`False` = the server block was/was not found on the
        # live Caddy. `[proxy.acme]` is bound into the SPAWN-time bootstrap, so
        # "the proxy is available" only implies "the :80 listener is bound" for a
        # Caddy this process spawned with the current config. An ADOPTED one —
        # the daemon was SIGKILLed after ACME was enabled and its Caddy outlived
        # it — carries the `nerdit` server (so adoption succeeds) but no
        # `nerdit-acme-http`, and inferring the bind from availability made
        # `/proxy/status` and the doctor row both claim a listener that does
        # not exist. Latched, so the extra admin read is once per episode, never
        # per tick.
        self._acme_listener_live: bool | None = None
        # The custom-domain route ids
        # this manager last CONFIRMED live in Caddy — desired this tick AND
        # present in the live table. It is the only fact that distinguishes a
        # bound domain that answers from one that is merely stored: a row exists
        # from the moment of the PUT, while its Host route lands on the next
        # reconcile tick, and never at all while `_converge_tls` fails. Read
        # by `views/hosted.py` through `live_domain_route_ids`; empty is
        # the fail-closed value, so every path that ends a Caddy process's
        # lifetime clears it (`_invalidate_convergence`).
        self._live_domain_ids: frozenset[str] = frozenset()
        # Log-once latch for the "domain routes withheld because the
        # TLS subtree has not converged" condition. Same shape as
        # `_foreign_warned`: the condition is re-evaluated every ~5 s tick and
        # a persistent one must not write a line per tick.
        self._tls_withheld_warned = False
        # Respawn backoff state (self-heal when Caddy dies mid-run).
        self._spawn_attempts = 0
        self._last_spawn_at = 0.0
        # Log the "foreign process owns the admin port" condition once, not every tick.
        self._foreign_warned = False
        # Queryable counterpart of `_foreign_warned`: the admin port is
        # answered by a foreign (non-nerdit) Caddy. Set/cleared alongside the
        # ownership decisions in `start()`/`_ensure_alive()`; projected by
        # `state` as `foreign_conflict`. Unlike `_foreign_warned` (a
        # one-shot log latch) this tracks the live condition, so it is cleared
        # whenever the proxy is adopted, respawned, or turned off.
        self._conflict = False
        # `(service_name, ref) -> plaintext | None` — a CALLABLE,
        # never the `SecretManager`: the proxy resolves the one reference a row
        # declares and has no business enumerating the secret store. `None`
        # (unwired) makes every edge-auth declaration unresolvable, i.e. fail
        # closed, which is the correct posture for a daemon that somehow booted
        # without the wiring.
        self._secret_resolver = secret_resolver
        # bcrypt is ~100 ms BY DESIGN, and the reconcile loop
        # rebuilds every route object every tick, so the hash is memoized on
        # `(service_name, user, sha256(plaintext))`. The username is part of
        # the key because the cached material EMBEDS it: keying on the digest
        # alone would return stale material — and therefore an unchanged
        # fingerprint — when only `edge_auth.user` changes, and Caddy would
        # keep accepting the obsolete username until the next restart.
        self._auth_cache: dict[tuple[str, str, str], EdgeAuthMaterial] = {}
        # Services whose edge auth currently cannot be materialized. A set, not
        # a counter: the failure is logged once per service per transition, not
        # once per reconcile tick, so a permanent misconfiguration does not spam
        # a line every five seconds.
        self._edge_auth_failed: set[str] = set()

    def set_secret_resolver(self, resolver: Callable[[str, str], str | None]) -> None:
        """Wire the edge-auth secret resolver after construction (P25 §3.4.5).

        The daemon lifespan builds the `ProxyManager` before the
        `SecretManager` (the resolver closes over the latter), so the wiring
        is a late setter rather than a constructor argument — reordering boot is
        a much bigger blast radius than one setter, and the proxy is not started
        until after both exist.
        """
        self._secret_resolver = resolver

    # -- gates ----------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """`True` when the proxy is configured on AND the binary is resolvable."""
        return bool(self._settings.enabled and self._binary)

    @property
    def available(self) -> bool:
        """`True` when Caddy is currently answering (set by start/reconcile)."""
        return self._available

    @property
    def acme_listener_live(self) -> bool | None:
        """Report the live ACME HTTP-listener state without optimistic assumptions.

        Returns:
            None when disabled; False while unavailable, absent, or not yet confirmed;
            True when the available proxy's live config contains its HTTP server.
            Inconclusive probes retry on the next tick.
        """
        if not self._settings.acme.enabled:
            return None
        if not self._available:
            return False
        return bool(self._acme_listener_live)

    @property
    def mode(self) -> str:
        return self._settings.mode

    # -- read accessors (P13b, observability) ---------------------------------

    @property
    def state(self) -> ProxyState:
        """The current `ProxyState`, derived from the internal flags.

        Precedence (1.8): `disabled` (proxy off) → `no_binary` (on but the
        `caddy` binary is unresolved) → `foreign_conflict` (the admin port is
        owned by a non-nerdit process) → `available` (up and ours) → `backoff`
        (a respawn has been attempted, waiting out the delay) → `starting` (on,
        no successful tick yet).
        """
        if not self._settings.enabled:
            return ProxyState.disabled
        if self._binary is None:
            return ProxyState.no_binary
        if self._conflict:
            return ProxyState.foreign_conflict
        if self._available:
            return ProxyState.available
        if self._spawn_attempts > 0:
            return ProxyState.backoff
        return ProxyState.starting

    @property
    def respawn_attempts(self) -> int:
        """How many times the reconcile loop has respawned Caddy (backoff counter)."""
        return self._spawn_attempts

    @property
    def last_spawn_ago_s(self) -> float | None:
        """Seconds since the last respawn attempt, or `None` if never spawned.

        `_last_spawn_at` is a `time.monotonic` reading, so it is projected
        as a *relative* age computed at read time — never as a wall-clock stamp.
        """
        if self._last_spawn_at == 0.0:
            return None
        return time.monotonic() - self._last_spawn_at

    @property
    def next_retry_in_s(self) -> float | None:
        """Seconds until the next respawn is eligible, or `None`.

        `None` when the proxy is available (no retry pending) or has never
        spawned; otherwise the remaining slice of the current
        `_backoff_seconds` window, floored at 0.
        """
        if self._available or self._last_spawn_at == 0.0:
            return None
        elapsed = time.monotonic() - self._last_spawn_at
        return max(0.0, self._backoff_seconds() - elapsed)

    async def status_snapshot(self, domains: Sequence[ServiceDomain] = ()) -> dict[str, Any]:
        """Build read-only proxy status from typed live-state queries.

        Unreadable tables produce null counts, never zero. Skip admin reads while
        unavailable; derive each domain's live flag from the same snapshot, using None
        when unknown. Certificate facts come from storage without extra admin I/O.
        The route adds mDNS status separately.

        Disabled ACME config values are None; listener status reflects live state.
        Expose the directory URL for staging diagnosis, never the account email.
        """
        state = self.state
        available = self._available
        acme_cfg = self._settings.acme

        pem = await self.ca_root_pem()
        fingerprint: str | None = None
        if pem is not None:
            try:
                fingerprint = ca_fingerprint(pem)
            except ValueError:
                fingerprint = None

        apex: dict[str, Any] = {
            "enabled": self._apex_enabled,
            "present": None,
            "is_last": None,
        }
        routes_count: int | None = None
        live_map: dict[str, LiveRoute] | None = None
        if state in (ProxyState.disabled, ProxyState.no_binary) or not available:
            live_table = "disabled"
        else:
            live = await self._admin.live_routes()
            if live is None:
                live_table = "unreadable"
            else:
                live_table = "readable"
                routes_count = len(live)
                live_map = live
            apex_state = await self._admin.apex_state()
            if apex_state is not None:
                present, is_last, _dial, _count = apex_state
                apex["present"] = present
                apex["is_last"] = is_last

        return {
            "state": state.value,
            "enabled": self._settings.enabled,
            "available": available,
            "mode": self._settings.mode,
            "base_domain": self._settings.base_domain,
            "hostname": self._settings.hostname_override or self._hostname,
            "scheme": self._settings.scheme,
            "https_port": self._settings.https_port,
            "tls": {"subjects": self._tls_subjects(), "synced": self._tls_synced},
            "ca": {"fingerprint": fingerprint, "present": pem is not None},
            "apex": apex,
            "respawn": {
                "attempts": self.respawn_attempts,
                "last_spawn_ago_s": self.last_spawn_ago_s,
                "next_retry_in_s": self.next_retry_in_s,
            },
            "routes": {"count": routes_count, "live_table": live_table},
            "acme": {
                "enabled": acme_cfg.enabled,
                "directory": acme_cfg.directory if acme_cfg.enabled else None,
                "http_port": acme_cfg.http_port if acme_cfg.enabled else None,
                "http_redirect": acme_cfg.http_redirect if acme_cfg.enabled else None,
                "listening": self.acme_listener_live,
            },
            "domains": [
                {
                    "domain": row.domain,
                    "service_name": row.service_name,
                    "acme": row.acme,
                    "live": (
                        None
                        if live_map is None
                        else _domain_route_id(row.service_name, row.domain) in live_map
                    ),
                    "cert_state": self.cert_status(row).state,
                }
                for row in domains
            ],
        }

    def cert_status(self, row: ServiceDomain) -> CertStatus:
        """Read a domain's certificate state from its exact Caddy storage path.

        Return internal for non-ACME rows, disabled for ACME requests when ACME is off
        (still internally served), or issued/expired/pending from the public leaf.
        No admin call or TLS handshake is required.
        """
        if not row.acme:
            return CertStatus("internal")
        acme = self._settings.acme
        if not acme.enabled:
            return CertStatus("disabled")
        return inspect_cert(acme_cert_path(self._storage_root, acme.directory, row.domain))

    def cert_states(self, rows: Sequence[ServiceDomain]) -> dict[str, CertStatus]:
        """`cert_status` for a batch, keyed by domain.

        The shape the exposure projections load once per request
        (`views/hosted.py`) so a list surface stays pure: one pass over the
        rows the caller already read, no per-row call back into the manager.
        """
        return {row.domain: self.cert_status(row) for row in rows}

    async def live_routes(self) -> dict[str, LiveRoute] | None:
        """Typed passthrough to `CaddyAdmin.live_routes` for the `GET /routes` read.

        Keeps the `CaddyAdmin` instance private to the manager (Invariant #2:
        observability reads go through the typed wrapper, never raw admin JSON).
        `None`-tolerant exactly like the wrapper — an unreadable live set is a
        distinct signal from "no routes", which the route projects as a tri-state
        `live_table`.
        """
        return await self._admin.live_routes()

    async def ca_root_pem(self) -> str | None:
        """The internal-CA root certificate PEM, or `None` when absent.

        Prefers the live admin API (which lazily provisions the CA), falling
        back to Caddy's file storage — the CA persists across restarts, so the
        disk copy also serves the proxy-temporarily-down case. `None` means
        the proxy has never issued a CA here (e.g. `[proxy].enabled=false`
        and no prior run); the route layer maps that to a structured 404.
        """
        if self._available:
            pem = await self._admin.get_ca()
            if pem is not None:
                return pem
        root_crt = self._storage_root / "pki" / "authorities" / "local" / "root.crt"
        try:
            return root_crt.read_text()
        except OSError:
            return None

    @property
    def _expected_shape(self) -> str:
        """The matcher shape a DEFAULT route must carry in the current mode.

        Compared against `LiveRoute` `shape` by `register` /
        `reconcile`: a live route whose dial matches but whose
        matcher is the OTHER mode's shape is drift and must be upserted. It feeds `build_route`,
        which STAMPS the answer onto the
        spec; the drift comparisons then read `spec.shape`, never this
        property. Custom-domain routes are Host-shaped in either mode, so a
        mode-derived expectation would read every one of them as drift and
        rewrite it every tick.
        """
        return "host" if self._settings.mode == "subdomain" else "path"

    @property
    def withheld_services(self) -> frozenset[str]:
        """Services whose route is currently WITHHELD by the fail-closed path.

        (P26 S-W8) A snapshot of the D-P25-8 latch, read by
        `views/hosted.py` to decide whether a custom domain is `ready` or
        `withheld`: the app is deployed and the proxy is up, but its declared
        `edge_auth` cannot be materialized, so neither its default route nor
        any of its domain routes are live. Frozen so no caller can mutate the
        manager's own set through it.
        """
        return frozenset(self._edge_auth_failed)

    @property
    def live_domain_route_ids(self) -> frozenset[str]:
        """Return custom-domain IDs last confirmed live, without I/O.

        Reconcile intersects desired and live routes; register/deregister update the
        set and convergence invalidation clears it. Proxy availability or resolved auth
        alone cannot prove a newly added or TLS-withheld domain is ready.
        """
        return self._live_domain_ids

    @property
    def _apex_enabled(self) -> bool:
        """`True` when the dashboard-apex route should exist.

        The apex serves the dashboard/API at `https://<host>/` iff the proxy is
        on, the opt-in `[proxy].dashboard_apex` flag is set, AND the proxy is in
        `path` mode. Subdomain mode Host-matches per service and leaves the apex
        routeless by design, so the flag is a no-op there (Decision 1); when it is
        off, `_reconcile_apex` prunes any apex route a mode flip left behind.
        """
        return self.enabled and self._settings.dashboard_apex and self._settings.mode == "path"

    # -- route shaping --------------------------------------------------------

    def build_route(
        self, service_name: str, host_port: int, edge_auth: EdgeAuthSpec | None = None
    ) -> RouteSpec:
        """Build a route spec for the service's active upstream and resolved edge auth.

        Memoize bcrypt hashes; never fall back to an unprotected route on failure.
        Read paths needing only an ID must use `_route_id` to avoid resolving secrets.

        Raises:
            EdgeAuthUnresolved: The password reference cannot resolve.
            EdgeAuthInvalid: Persisted auth or password material is unusable.
        """
        subdomain = self._settings.mode == "subdomain"
        return RouteSpec(
            service_name=service_name,
            host_port=host_port,
            route=generate_route(service_name, mode=self._settings.mode),
            caddy_id=_route_id(service_name),
            auth=None if edge_auth is None else self._resolve_auth(service_name, edge_auth),
            # `mode` enters HERE and is recorded on the spec; from
            # this point the emitter and the drift classifier read the spec's
            # own shape, never the manager's mode. That is what lets a
            # Host-shaped domain route coexist with a path-shaped default.
            kind="default",
            shape="host" if subdomain else "path",
            host=(
                f"{service_name}.{self._settings.base_domain or self._hostname}"
                if subdomain
                else None
            ),
        )

    def build_domain_route(
        self,
        service_name: str,
        host_port: int,
        domain: str,
        *,
        auth: EdgeAuthMaterial | None,
    ) -> RouteSpec:
        """Resolve the `RouteSpec` for ONE custom domain (P26 D-P26-2).

        Host-shaped in **both** proxy modes — the domain is served at its own
        root, which is the whole point — and carrying the service's already
        resolved *auth* material rather than re-resolving it: bcrypt is ~100 ms
        by design, and a service's domain routes must present exactly the same
        credential as its default route (S-W5). `route` is `""` because a
        domain route has no persisted projection; `service_endpoints.route`
        describes the default route only.
        """
        return RouteSpec(
            service_name=service_name,
            host_port=host_port,
            route="",
            caddy_id=_domain_route_id(service_name, domain),
            auth=auth,
            kind="domain",
            shape="host",
            host=domain,
        )

    def _domain_specs(self, spec: RouteSpec, domains: Sequence[str]) -> list[RouteSpec]:
        """One domain spec per name, reusing *spec*'s resolved auth material.

        Takes the service's DEFAULT spec so the credential is resolved exactly
        once per service per tick (S-W5) and so a service whose auth could not be
        materialized never reaches here at all — it failed before the default
        spec existed, and its domain routes are withheld with it (fail closed,
        D-P25-8).
        """
        return [
            self.build_domain_route(spec.service_name, spec.host_port, domain, auth=spec.auth)
            for domain in domains
        ]

    def _resolve_auth(self, service_name: str, spec: EdgeAuthSpec) -> EdgeAuthMaterial:
        """Resolve + hash one edge-auth declaration into emittable material.

        The plaintext lives in a local for the length of one hash: it is never
        persisted, never written to Caddy (only the bcrypt hash is), never
        logged and never audited.
        """
        key = _ref_key_name(spec.password_ref)
        resolver = self._secret_resolver
        if resolver is None:
            raise EdgeAuthUnresolved(service_name, key)
        try:
            plaintext = resolver(service_name, spec.password_ref)
        except Exception as exc:  # noqa: BLE001 — injected code; any failure is "unresolved"
            # The resolver owns its own error translation, so anything reaching
            # here is unexpected — and an unexpected failure must fail CLOSED,
            # not abort the reconcile tick. The exception TYPE is logged, never
            # its message or args: a secret-store error can quote a value, and
            # this line would carry it into the daemon log.
            logger.debug(
                "[proxy] edge auth resolver raised %s for %s", type(exc).__name__, service_name
            )
            raise EdgeAuthUnresolved(service_name, key) from None
        if not plaintext:
            raise EdgeAuthUnresolved(service_name, key)
        cache_key = (
            service_name,
            spec.user,
            hashlib.sha256(plaintext.encode("utf-8")).hexdigest(),
        )
        material = self._auth_cache.get(cache_key)
        if material is None:
            # `edgeauth.hash_password`, through the module (never imported by
            # value), so tests can patch it: bcrypt salts per call, so its output
            # is never byte-stable and a golden needs a deterministic stand-in.
            material = EdgeAuthMaterial(
                user=spec.user, bcrypt_hash=edgeauth.hash_password(plaintext)
            )
            self._auth_cache[cache_key] = material
        return material

    def _evict_auth_cache(self, service_name: str) -> None:
        """Drop every memoized credential for a service.

        First-element match: the user/digest legs of the key are not known at
        eviction time — and must not have to be, since eviction happens exactly
        when the route goes away.
        """
        for key in [k for k in self._auth_cache if k[0] == service_name]:
            del self._auth_cache[key]

    def _note_edge_auth_failure(self, service_name: str, exc: Exception) -> None:
        """Log a withheld route ONCE per service per transition.

        The exception messages name a secret KEY or a set of FIELD names — never
        a value — by construction, so this line is safe to write verbatim.
        """
        if service_name in self._edge_auth_failed:
            return
        self._edge_auth_failed.add(service_name)
        logger.warning(
            "[proxy] edge auth for service %s cannot be materialized; the route is "
            "withheld (the app stays reachable on loopback): %s",
            service_name,
            exc,
        )

    def _caddy_route_obj(self, spec: RouteSpec) -> dict[str, Any]:
        """Emit handlers from the route spec's shape, not the manager's mode.

        Path routes strip their matched prefix; Host routes preserve paths, and custom
        domains are terminal. Place authentication before rewrite or reverse proxy
        inside the matched route.
        """
        dial = f"127.0.0.1:{spec.host_port}"
        proxy_handler: dict[str, Any] = {
            "handler": "reverse_proxy",
            "upstreams": [{"dial": dial}],
        }
        # The shape is the one `caddy adapt` emits for a `basic_auth` block
        # (captured against Caddy v2.11.4), MINUS its `hash_cache: {}` key:
        # that module is optional and Caddy provisions the default cache when
        # the key is absent, and since the admin API returns config as stored,
        # omitting it keeps our read-back symmetric with what we wrote (the
        # drift classifier is tolerant of it either way).
        auth_handlers: list[dict[str, Any]] = (
            []
            if spec.auth is None
            else [
                {
                    "handler": "authentication",
                    "providers": {
                        "http_basic": {
                            "hash": {"algorithm": "bcrypt"},
                            "accounts": [
                                {
                                    "username": spec.auth.user,
                                    "password": spec.auth.bcrypt_hash,
                                }
                            ],
                        }
                    },
                }
            ]
        )
        if spec.shape == "host":
            # `spec.host` is stamped by `build_route`/`build_domain_route`;
            # the fallback keeps a hand-built spec (tests, older call sites)
            # meaning exactly what it did before WP1.
            host = spec.host or (
                f"{spec.service_name}.{self._settings.base_domain or self._hostname}"
            )
            obj: dict[str, Any] = {
                "@id": spec.caddy_id,
                "match": [{"host": [host]}],
                "handle": [*auth_handlers, proxy_handler],
            }
            if spec.kind == "domain":
                # Terminal: a matched custom domain STOPS route
                # evaluation, so no later path route can also handle the
                # request. Only the domain routes carry it — adding it to the
                # subdomain-mode default would change a shape pinned by goldens
                # for no gain (a Host default already matches nothing else).
                obj["terminal"] = True
            return obj
        return {
            "@id": spec.caddy_id,
            "match": [{"path": [spec.route, f"{spec.route}/*"]}],
            "handle": [
                {
                    "handler": "subroute",
                    "routes": [
                        {
                            "handle": [
                                *auth_handlers,
                                {"handler": "rewrite", "strip_path_prefix": spec.route},
                                proxy_handler,
                            ]
                        }
                    ],
                }
            ],
        }

    def _apex_route_obj(self) -> dict[str, Any]:
        """Build the dashboard-apex catch-all route JSON.

        A route object with **no** `match` key matches every request, so this
        is a catch-all; it reverse-proxies verbatim (no `strip_path_prefix`) to
        `_dashboard_upstream` — the daemon — so `/`, `/assets/*`,
        `/api/*` and the SSE streams all reach it unchanged over HTTPS. Its
        `@id` is `_APEX_ID` (outside `_ID_PREFIX`), keeping it clear of
        the service-route machinery. It MUST be the last element of the routes
        array (see `_reconcile_apex`) or it would shadow every service.
        """
        return {
            "@id": _APEX_ID,
            "handle": [
                {"handler": "reverse_proxy", "upstreams": [{"dial": self._dashboard_upstream}]}
            ],
        }

    async def start(self) -> None:
        """Adopt a live nerdit-owned Caddy or spawn one, then load the bootstrap.

        Safe to call when disabled (logs and no-ops). Never raises into the
        daemon lifespan: a failure leaves `available=False` and the reconcile
        loop retries.
        """
        if not self.enabled:
            self._available = False
            self._invalidate_convergence()
            self._conflict = False
            # Reap a leftover nerdit-owned Caddy from a previous (enabled) run that
            # outlived an ungraceful daemon crash. Our pid file is written only by
            # _spawn, so a live pid there is ours — without this it would keep
            # serving the old routes on the LAN while the new config reports the
            # proxy disabled.
            if self._pid_alive():
                logger.info("[proxy] disabled — reaping leftover Caddy from a prior run")
                await self._kill()
            if self._settings.enabled and not self._binary:
                logger.warning(
                    "[proxy] enabled but '%s' binary not found on PATH — "
                    "services stay on loopback (no URLs). Install Caddy to enable.",
                    self._settings.caddy_binary,
                )
            return
        if self._settings.acme.enabled:
            # (S-W2-15) One line, at the one moment the posture is decided. The
            # HOST of the directory, never the whole URL and never the account
            # email: an operator reading a shared log should learn which CA this
            # node talks to without the log becoming a place credentials or PII
            # accumulate.
            directory_host = urlsplit(self._settings.acme.directory).hostname or "?"
            logger.info(
                "[proxy] ACME enabled: directory host %s, http port %d",
                directory_host,
                self._settings.acme.http_port,
            )
            if directory_host == _LE_PRODUCTION_HOST:
                # (review round 1) A staging GUARD, not a permanent nag. The
                # default directory IS production, so every correctly configured
                # node used to log this at WARNING on every single boot —
                # including after the operator did validate against staging and
                # has the leaves on disk to prove it. A warning that fires on the
                # correct steady state teaches operators to ignore warnings. Once
                # this node has issued anything from this directory, the same
                # sentence is an `info`: it is context, not a caution.
                issued_before = (
                    self._storage_root
                    / "certificates"
                    / acme_storage_key(self._settings.acme.directory)
                ).is_dir()
                logger.log(
                    logging.INFO if issued_before else logging.WARNING,
                    "[proxy] ACME points at the Let's Encrypt PRODUCTION directory "
                    "— rate limits apply (5 duplicate certificates per week)%s",
                    ""
                    if issued_before
                    else "; validate with the staging directory first "
                    "(https://acme-staging-v02.api.letsencrypt.org/directory)",
                )
        # P3.5: a SINGLE-LABEL wildcard base (`*.localhost`, `*.box`) is
        # rejected by every strict TLS client — RFC 6125 (enforced by OpenSSL,
        # so curl/Python/browsers) requires at least two labels after the
        # `*`. Caddy ≤2.7 masked this by minting an exact-name leaf per
        # Host-matched route; ≥2.8 serves the wildcard itself and handshakes
        # fail verification. Routing still works (Host matching is
        # TLS-agnostic), so warn rather than reject.
        if self._settings.mode == "subdomain":
            base = self._settings.base_domain or self._hostname
            if "." not in base:
                logger.warning(
                    "[proxy] subdomain mode with single-label base '%s': the "
                    "*.%s wildcard cert violates RFC 6125's two-label rule and "
                    "strict TLS clients (curl, browsers, Python) will refuse "
                    "it. Use a multi-label [proxy].base_domain such as "
                    "'dev.localhost' or 'apps.lan' (see docs/guide/proxy.md).",
                    base,
                    base,
                )
        # D5: subdomain mode with no base_domain is legal (the hostname
        # fallback is tested behavior) but almost always a misconfiguration —
        # warn loudly instead of failing startup on existing TOML.
        if self._settings.mode == "subdomain" and not self._settings.base_domain:
            if self._settings.hostname_override:
                logger.warning(
                    "[proxy] mode=subdomain with no [proxy].base_domain — URLs and "
                    "Host matchers fall back to *.%s (from hostname_override); "
                    "wildcard DNS for that name must resolve to this host "
                    "(see docs/guide/proxy.md).",
                    self._hostname,
                )
            else:
                logger.error(
                    "[proxy] mode=subdomain with no [proxy].base_domain and no "
                    "[proxy].hostname_override — URLs and Host matchers fall back "
                    "to *.%s (the machine hostname), which almost never resolves "
                    "on a LAN. Set [proxy].base_domain (see docs/guide/proxy.md).",
                    self._hostname,
                )
        try:
            if await self._admin.ping():
                # Something already owns the admin port.
                if await self._admin.has_nerdit_server():
                    self._available = True
                    self._conflict = False
                    await self._sync_acme_listener()
                    await self._converge_tls()
                    logger.info("[proxy] adopted a running nerdit Caddy (skipping load)")
                    return
                if self._pid_alive():
                    # Our process, but the nerdit-server probe came back negative.
                    # That probe also fails on a transient error, and `load()`
                    # would reset the whole config to empty routes — so re-check
                    # once and only (re)load if the server is *definitively* absent,
                    # never wiping live routes over a single blip.
                    if not await self._admin.has_nerdit_server():
                        await self._admin.load(self._bootstrap_config())
                    self._available = True
                    self._conflict = False
                    await self._sync_acme_listener()
                    # On the recheck-success (adopt) path the live TLS config was
                    # never pushed; on the load() path this is a compare-and-skip.
                    await self._converge_tls()
                    return
                logger.error(
                    "[proxy] admin API %s is busy but not nerdit-owned; "
                    "refusing to load (would clobber a foreign config). Proxy disabled.",
                    self._settings.admin_addr,
                )
                self._available = False
                self._invalidate_convergence()
                self._conflict = True
                return
            # Nothing there → spawn with the bootstrap config loaded at launch.
            self._spawn()
            if await self._wait_ready():
                # Defensive: the config is loaded at launch, but if the nerdit
                # server is somehow absent, push it via the admin API.
                if not await self._admin.has_nerdit_server():
                    await self._admin.load(self._bootstrap_config())
                self._available = True
                self._spawn_attempts = 0
                self._conflict = False
                await self._sync_acme_listener()
                logger.info("[proxy] Caddy ready on :%d", self._settings.https_port)
            else:
                self._available = False
                self._invalidate_convergence()
                # macOS allows unprivileged low-port binds, so the likely cause
                # there is a port conflict, not missing capabilities (setcap is
                # Linux-only anyway).
                #
                # The Linux hint names the binary WE resolved, never
                # `$(which caddy)`: on a frozen release install the bundled
                # Caddy is not on PATH at all, so that substitution expanded to
                # the empty string and the suggested command was a silent no-op
                # against the very binary that needed the capability. A P30
                # system unit already carries AmbientCapabilities=
                # CAP_NET_BIND_SERVICE, so this path is reached mainly by a
                # --user unit or a hand-started daemon — for which setcap on the
                # resolved path is the real fix.
                port = self._settings.https_port
                if sys.platform == "darwin":
                    port_hint = f"is another process already listening on :{port}?"
                elif self._binary:
                    port_hint = (
                        f"binding :{port} may need "
                        f"`sudo setcap cap_net_bind_service=+ep {self._binary}`"
                    )
                else:
                    port_hint = (
                        f"binding :{port} may need CAP_NET_BIND_SERVICE, "
                        "or an unprivileged [proxy].https_port"
                    )
                logger.error(
                    "[proxy] Caddy did not become ready (check %s; %s)",
                    self._log_file,
                    port_hint,
                )
        except Exception:
            self._available = False
            self._invalidate_convergence()
            logger.exception("[proxy] start failed; services stay on loopback")

    # -- audit ---------------------------------------------------------------

    async def _audit_route(
        self, action: str, service_name: str, *, domain: str | None = None
    ) -> None:
        """Record a `system` audit row for a route change (route is non-secret).

        *domain* names the custom domain when the row concerns one of
        a service's domain routes rather than its default. It rides
        `params_redacted`; `target_id` stays the SERVICE, so every route row
        for an app still groups under one target. A domain is public DNS the
        operator chose, never a bearer capability, so writing it verbatim is
        safe — the same reasoning as the `domain.added` event (S-W9).
        """
        if not self._audit:
            return
        params: dict[str, str] = {"service_name": service_name}
        if domain is not None:
            params["domain"] = domain
        try:
            await self._queries.insert_audit_log(
                action=action,
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="service",
                target_id=service_name,
                params_redacted=json.dumps(params),
            )
        except Exception:
            logger.warning("[proxy] failed to audit %s for %s", action, service_name, exc_info=True)

    async def _audit_apex(self, action: str) -> None:
        """Record a `system` audit row for the dashboard-apex route.

        Dedicated actions (`proxy.apex_registered` / `proxy.apex_deregistered`)
        rather than the per-service `proxy.route_*` names, so the apex is
        distinguishable in the audit log. The upstream dial is non-secret.
        """
        if not self._audit:
            return
        try:
            await self._queries.insert_audit_log(
                action=action,
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="proxy",
                target_id=_APEX_ID,
                params_redacted=json.dumps({"upstream": self._dashboard_upstream}),
            )
        except Exception:
            logger.warning("[proxy] failed to audit %s for apex", action, exc_info=True)

    # -- best-effort inline fast-path ----------------------------------------

    async def register(
        self,
        service_name: str,
        host_port: int,
        edge_auth: object = None,
        *,
        with_domains: bool = False,
        domains: Sequence[str] | None = None,
    ) -> None:
        """Refresh routes best-effort, auditing only actual Caddy changes.

        Always pass raw persisted edge auth; invalid or unresolved auth deregisters the
        service instead of creating an open route. Reconcile remains authoritative.

        Args:
            with_domains: Include the whole service route set for cutover. Ordinary
                launch updates only the default route; domain writes require TLS
                convergence first.
            domains: Optional shared snapshot of domain names. Cutover must write and
                verify the same set, avoiding races with concurrent domain deletion.
                None reads the names here.
        """
        if not self.enabled or not self._available:
            return
        new_dial = f"127.0.0.1:{host_port}"
        try:
            try:
                spec = self.build_route(service_name, host_port, load_edge_auth(edge_auth))
            except (EdgeAuthInvalid, EdgeAuthUnresolved) as exc:
                self._note_edge_auth_failure(service_name, exc)
                await self.deregister(service_name)
                return
            self._edge_auth_failed.discard(service_name)
            specs = [spec]
            if with_domains:
                # The same gate reconcile applies: a domain Host route
                # may only be written once the live `apps.tls` is known to
                # carry the catch-all internal-issuer policy. On the normal path
                # the hash latch has held since boot, so this is zero admin I/O;
                # when it has not, the cutover commits the DEFAULT route only and
                # the domains follow on the first tick that converges TLS.
                if await self._converge_tls():
                    if domains is None:
                        rows = await self._queries.get_service_domains(service_name)
                        names = [row.domain for row in rows]
                    else:
                        names = list(domains)
                    specs.extend(self._domain_specs(spec, names))
                else:
                    logger.warning(
                        "[proxy] %s: TLS subtree not converged — committing the default "
                        "route only; custom domains follow on a later reconcile tick",
                        service_name,
                    )
            live = await self._admin.live_routes()
            if live is None:
                return  # couldn't read state — let the reconcile loop register
            wrote_default = False
            # (S-W8) Domain ids this call leaves live — the ones already current
            # plus the ones it successfully upserts. A per-route upsert failure
            # below `continue`s, so the id never enters the set and the surface
            # keeps reporting that domain `withheld` until a reconcile tick
            # actually lands it.
            live_domains: set[str] = set()
            for one in specs:
                current = live.get(one.caddy_id)
                if (
                    current is not None
                    and current.dial == new_dial
                    and current.shape == one.shape
                    and current.count == 1
                    and current.auth_fingerprint == _desired_fingerprint(one)
                ):
                    # Already current: upstream, matcher shape, no duplicates AND
                    # the same edge-auth state (D-P25-7 — without that term an auth
                    # change on an otherwise-current route is invisible here).
                    if one.kind == "domain":
                        live_domains.add(one.caddy_id)
                    continue
                try:
                    await self._admin.upsert_route(
                        self._caddy_route_obj(one), prepend=one.kind == "domain"
                    )
                    await self._audit_route(
                        "proxy.route_registered", service_name, domain=_spec_domain(one)
                    )
                except Exception:
                    # One failed route must not withhold the others: the default
                    # going live still beats nothing, and reconcile converges the
                    # rest next tick.
                    logger.warning(
                        "[proxy] register failed for %s (%s)",
                        service_name,
                        one.caddy_id,
                        exc_info=True,
                    )
                    continue
                wrote_default = wrote_default or one.kind == "default"
                if one.kind == "domain":
                    live_domains.add(one.caddy_id)
            # P9.5: an absent-id upsert (a genuinely new service) appends the
            # route to the END of the array — AFTER the apex catch-all — which
            # would shadow the just-registered service (the apex has no matcher
            # and Caddy takes the first terminal match) until the next reconcile
            # tick. Re-anchor the apex last NOW so the inline fast-path keeps its
            # "URL live within this tick" guarantee instead of serving the
            # dashboard at the service's URL for ~5s. Steady state (every route
            # already current) never reaches here, so this is zero extra work on
            # the common no-op path. Domain routes are PREPENDED and so never
            # land below the apex — only a default write re-anchors it.
            if live_domains:
                self._live_domain_ids |= frozenset(live_domains)
            if wrote_default and self._apex_enabled:
                await self._reconcile_apex()
        except Exception:
            logger.warning("[proxy] register failed for %s", service_name, exc_info=True)

    async def deregister(self, service_name: str) -> None:
        """Remove a service's route now (best-effort; never raises). Also drops the service's
        memoized credential material and its
        withheld-route latch: a route torn down is a state transition, so the
        next failure is logged once more rather than staying silent forever. Removes EVERY route
        the service owns — its default plus one per
        custom domain — derived from the live table by splitting the composite
        `@id`. When the live table is unreadable only the default id is
        deleted; the prune loop removes the rest on the next reconcile tick.
        """
        # Eviction runs even when the proxy is off/unavailable: the cache is
        # process state, not Caddy state, and a service whose route we cannot
        # delete right now must still not keep a hash for a credential it may
        # no longer own.
        self._evict_auth_cache(service_name)
        self._edge_auth_failed.discard(service_name)
        # (S-W8) Same reasoning as the cache eviction: this runs even when the
        # proxy is off, because the snapshot is process state. A service whose
        # routes we are tearing down (or cannot reach to tear down) must stop
        # reporting its domains `ready` immediately, not one tick later.
        self._live_domain_ids = frozenset(
            rid for rid in self._live_domain_ids if _split_route_id(rid)[0] != service_name
        )
        if not self.enabled or not self._available:
            return
        try:
            default_id = _route_id(service_name)
            ids = [default_id]
            live = await self._admin.live_routes()
            if live is not None:
                # LIVE-derived, not DB-derived: a service being torn down may
                # already have lost its `service_domains` rows (the delete
                # cascade runs inside the same request), so asking the table
                # would leave the very routes this call exists to remove. The
                # composite `@id` grammar is the whole lookup.
                ids.extend(
                    sorted(
                        rid
                        for rid in live
                        if rid != default_id and _split_route_id(rid)[0] == service_name
                    )
                )
            for rid in ids:
                removed = await self._admin.delete_route(rid)
                if removed:
                    await self._audit_route(
                        "proxy.route_deregistered", service_name, domain=_split_route_id(rid)[1]
                    )
        except Exception:
            logger.warning("[proxy] deregister failed for %s", service_name, exc_info=True)

    # -- reconcile (source of truth) -----------------------------------------

    async def reconcile(self) -> None:
        """Converge Caddy routes to database state without steady-state writes.

        Detect missing routes, wrong dial/shape/auth, domain ordering behind defaults,
        and stale default route strings. Converge TLS first, then upsert, prune,
        restore domain ordering, and reconcile the dashboard apex. Withhold domain
        routes when TLS convergence fails to prevent unintended public issuance.

        Normal domain additions become live within one tick. A cold auth cache after
        restart generates new salted hashes and one corrective write per authed service;
        otherwise converged ticks are read-only.
        """
        if not self.enabled:
            return
        if not await self._ensure_alive():
            return
        try:
            desired = await self._queries.list_active_service_routes()
            domain_rows = await self._queries.list_service_domains()
            live = await self._admin.live_routes()
        except Exception:
            logger.warning("[proxy] reconcile: could not read state", exc_info=True)
            return
        if live is None:
            # Couldn't read Caddy's live routes — skip this tick rather than treat
            # everything as missing (which would re-push + re-audit every route).
            logger.warning("[proxy] reconcile: live routes unreadable; skipping tick")
            return

        domains_by_service: dict[str, list[str]] = {}
        for row in domain_rows:
            domains_by_service.setdefault(row.service_name, []).append(row.domain)

        # BEFORE any route write: the certificate must exist before the
        # Host route that will present it. The rows were just read, so this costs
        # no extra query, and the hash latch makes a converged tick free.
        #
        # The return value is a GATE, not a log line: until the live `apps.tls`
        # is known to carry the trailing catch-all policy, a Host route landing
        # in front of an ADOPTED pre-WP1 Caddy would hand the name to that
        # Caddy's default (public ACME) issuers. So a failed convergence
        # withholds every domain spec this tick — the same fail-closed shape the
        # edge-auth `withheld` set uses below, and the prune loop tears down
        # any domain route already live, which is the safe direction: no Host
        # route without a certificate policy that pins it to the internal CA.
        #
        # The partition — every name, and the subset that gets a
        # PUBLIC certificate — is computed from the rows just read, in the one
        # place the "acme=1 AND [proxy.acme].enabled" rule lives. Still one read
        # of `service_domains` per tick.
        all_names, acme_names = partition_domains(
            domain_rows, acme_enabled=self._settings.acme.enabled
        )
        tls_ok = await self._converge_tls(all_names, acme_names)
        if not tls_ok and domain_rows:
            if not self._tls_withheld_warned:
                self._tls_withheld_warned = True
                logger.warning(
                    "[proxy] TLS subtree not converged — withholding %d custom-domain "
                    "route(s) until it is (a Host route must never precede its "
                    "internal-issuer policy)",
                    len(domain_rows),
                )
        else:
            self._tls_withheld_warned = False

        desired_ids: set[str] = set()
        # Every desired spec by `@id` — the ordering pass needs to re-emit the
        # exact object it is re-anchoring, and looking it up here is what keeps
        # that pass from touching an id the prune loop already removed.
        specs_by_id: dict[str, RouteSpec] = {}
        # Services whose edge auth could not be materialized THIS tick. They are
        # deliberately absent from `desired_ids` (so the prune loop tears their
        # route down), and the prune loop must not clear their withheld latch —
        # that would re-log the same permanent misconfiguration every tick.
        withheld: set[str] = set()
        # (S-W8, Codex round 2 #3835632983) Ids whose convergence this tick
        # FAILED — a raised upsert, or a re-anchor that deleted the route and
        # could not put it back. They are still `desired`, and a failed upsert
        # leaves the STALE object in Caddy, so presence alone would advertise
        # them as live. Excluded from `_live_domain_ids` regardless of what
        # the post-write read shows.
        unconverged: set[str] = set()
        # The dial each desired id must carry, recorded where it is computed so
        # the convergence check below needs no second pass over `desired`.
        desired_dial_by_id: dict[str, str] = {}
        wrote = False
        for entry in desired:
            try:
                # Route building can now FAIL, and a failure must
                # skip THIS entry without aborting the tick — hence the build
                # sits in its own try ahead of the write block below.
                spec = self.build_route(
                    entry.service_name, entry.host_port, load_edge_auth(entry.edge_auth)
                )
            except (EdgeAuthInvalid, EdgeAuthUnresolved) as exc:
                # Fail closed: NOT added to `desired_ids`, so the prune loop
                # deletes any live route for this service — including one that
                # was serving UNAUTHENTICATED a moment ago (the "owner declared
                # edge_auth but hasn't set the secret yet" transition, which is
                # exactly when serve-it-anyway would leak the app). (P26 S-W5)
                # The service's DOMAIN routes are withheld with it: they are
                # built from this spec and it never existed.
                withheld.add(entry.service_name)
                self._note_edge_auth_failure(entry.service_name, exc)
                continue
            self._edge_auth_failed.discard(entry.service_name)
            new_dial = f"127.0.0.1:{entry.host_port}"
            specs = [spec]
            if tls_ok:
                specs.extend(
                    self._domain_specs(spec, domains_by_service.get(entry.service_name, []))
                )
            for one in specs:
                desired_ids.add(one.caddy_id)
                specs_by_id[one.caddy_id] = one
                desired_dial_by_id[one.caddy_id] = new_dial
                current = live.get(one.caddy_id)
                desired_fp = _desired_fingerprint(one)
                # Route-string drift is a property of the DEFAULT route only —
                # a domain route has no persisted projection.
                route_changed = one.kind == "default" and entry.route != one.route
                needs_upsert = (
                    current is None
                    or current.dial != new_dial
                    or current.shape != one.shape
                    # The auth dimension: a stripped, added or obsolete
                    # `authentication` handler is drift exactly like a moved dial.
                    or current.auth_fingerprint != desired_fp
                    or route_changed
                )
                try:
                    if current is not None and current.count > 1:
                        # Self-heal duplicates left by the pre-PATCH upsert (or an
                        # append race): the survivor is the id-map target whose
                        # dial/shape `current` already reflects, so the drift
                        # decision below stays valid. No audit row — infrastructure
                        # convergence, same rationale as the TLS sync.
                        removed = await self._admin.dedupe_route(one.caddy_id)
                        if removed:
                            wrote = True
                            logger.warning(
                                "[proxy] removed %d duplicate route object(s) for %s",
                                removed,
                                one.caddy_id,
                            )
                    if needs_upsert:
                        await self._admin.upsert_route(
                            self._caddy_route_obj(one), prepend=one.kind == "domain"
                        )
                        await self._audit_route(
                            "proxy.route_registered",
                            entry.service_name,
                            domain=_spec_domain(one),
                        )
                        wrote = True
                    # DB write LAST (P3.5 D3 write-order hardening): a failed upsert
                    # leaves the OLD route string in the row, so `route_changed`
                    # re-fires and the upsert naturally retries next tick. Writing
                    # the DB first would converge the row while Caddy still serves
                    # the stale object — permanent staleness once the dial matches.
                    if route_changed:
                        await self._queries.set_endpoint_route(entry.service_name, one.route)
                except Exception:
                    # (S-W8) The route may still be PRESENT in Caddy — with its
                    # OLD dial / matcher / auth handler. Present is not
                    # converged, so mark it and keep it out of the live set.
                    unconverged.add(one.caddy_id)
                    logger.warning(
                        "[proxy] reconcile: upsert failed for %s",
                        one.caddy_id,
                        exc_info=True,
                    )

        for rid, live_route in live.items():
            if rid in desired_ids:
                continue
            # The composite grammar is the only lookup: the service half
            # keys the caches, the domain half labels the audit row.
            service_name, domain = _split_route_id(rid)
            try:
                if live_route.count > 1:
                    # delete_route only removes the id-map target; drop the
                    # stale twins first so the orphan is fully gone this tick.
                    await self._admin.dedupe_route(rid)
                removed = await self._admin.delete_route(rid)
                if removed:
                    wrote = True
                    await self._audit_route("proxy.route_deregistered", service_name, domain=domain)
                    # The credential material goes with the route. The
                    # withheld latch does NOT when the prune is the fail-closed
                    # teardown itself — clearing it there would re-log the same
                    # misconfiguration on every tick.
                    #
                    # Scoped to the DEFAULT id: a pruned DOMAIN route is
                    # usually just `nerdit domains remove` on a service whose
                    # other routes stay live, and evicting there would re-bcrypt
                    # the same credential with a fresh salt next tick — the
                    # fingerprint would no longer match the live handler, so
                    # reconcile would PATCH (and re-audit) the default route and
                    # every surviving domain route of an app that did not
                    # change. A wholesale teardown still evicts: its default id
                    # is pruned in this same loop, and `deregister` evicts up
                    # front regardless.
                    if domain is None:
                        self._evict_auth_cache(service_name)
                    if service_name not in withheld:
                        self._edge_auth_failed.discard(service_name)
            except Exception:
                logger.warning("[proxy] reconcile: prune failed for %s", rid, exc_info=True)

        # The fifth dimension. Indexes read before the loops above are
        # stale the moment anything was written, so re-read only in that case —
        # a converged tick still issues zero extra requests. A violating id is
        # deleted and re-inserted at index 0; `specs_by_id` guards against
        # re-anchoring something the prune loop just removed.
        live_after = await self._admin.live_routes() if wrote else live
        # (S-W8, Codex round 1) The tick's verdict on which custom domains
        # actually answer: DESIRED this tick (so a TLS-withheld tick, an
        # edge-auth-withheld service and a domain removed from the table all
        # drop out) AND present in Caddy (so a route the upsert loop failed to
        # write is not advertised). Derived from the post-write read when there
        # was one, else from the read the tick opened with — nothing was written
        # in that case, so it is still accurate.
        observed = live_after if live_after is not None else live
        if live_after is not None:
            for rid in ordering_violations(live_after):
                offender = specs_by_id.get(rid)
                if offender is None:
                    continue
                try:
                    await self._admin.delete_route(rid)
                    await self._admin.insert_route_first(self._caddy_route_obj(offender))
                except Exception:
                    # The delete may have landed and the insert not: the route
                    # can be GONE from Caddy while `observed` — read before
                    # this loop — still lists it. Never advertise it as live.
                    unconverged.add(rid)
                    logger.warning("[proxy] reconcile: re-anchor failed for %s", rid, exc_info=True)
                    continue
                # No audit row: the route's existence and content did not change,
                # only its position — infrastructure convergence, the same
                # rationale as `dedupe_route`.
                logger.warning("[proxy] re-anchored Host route %s ahead of the path routes", rid)

        # (S-W8, Codex round 1; tightened round 2 #3835632983) The tick's verdict
        # on which custom domains actually answer: DESIRED this tick (so a
        # TLS-withheld tick, an edge-auth-withheld service and a domain removed
        # from the table all drop out) AND OBSERVED CONVERGED — present in Caddy
        # dialling the port we want, with the matcher shape we want and the auth
        # handler we want. Presence alone is not convergence: a corrective PATCH
        # that raised leaves the STALE object in place (old dial after a
        # container respawn, or obsolete credentials after an `edge_auth`
        # change), and `observed` would list it either way — the pre-write
        # snapshot when nothing else wrote, a fresh read showing the same stale
        # object when something did. `unconverged` additionally covers the
        # route a failed re-anchor deleted and could not re-insert, which is why
        # this is computed AFTER the ordering pass, not before it.
        #
        # Still zero extra admin requests on a converged tick: this reads the
        # snapshot the tick already holds.
        self._live_domain_ids = frozenset(
            rid
            for rid, live_route in observed.items()
            if rid in desired_ids
            and rid not in unconverged
            and specs_by_id[rid].kind == "domain"
            and live_route.dial == desired_dial_by_id[rid]
            and live_route.shape == specs_by_id[rid].shape
            and live_route.auth_fingerprint == _desired_fingerprint(specs_by_id[rid])
        )

        # Dashboard apex LAST: converge it only after every service route
        # is in place, so a service appended this tick (which lands after the
        # apex) is re-anchored below the apex's re-append.
        await self._reconcile_apex()

    async def _reconcile_apex(self) -> None:
        """Keep the enabled dashboard apex unique, correctly dialed, and last.

        Run after service upserts/prunes so the catch-all cannot shadow new routes.
        Collapse duplicates, then delete/append on drift; remove it when disabled.
        Skip unreadable state rather than treating it as absent and rewriting.
        """
        try:
            state = await self._admin.apex_state()
        except Exception:
            logger.warning("[proxy] reconcile: apex state unreadable; skipping", exc_info=True)
            return
        if state is None:
            return  # transient read failure — skip, don't treat as absent
        present, is_last, dial, count = state
        try:
            if not self._apex_enabled:
                if present:
                    # Collapse any duplicate twins first (delete_route only
                    # removes the id-map target), so the apex is fully gone.
                    if count > 1:
                        await self._admin.dedupe_route(_APEX_ID)
                    removed = await self._admin.delete_route(_APEX_ID)
                    if removed:
                        await self._audit_apex("proxy.apex_deregistered")
                return
            if present and count == 1 and is_last and dial == self._dashboard_upstream:
                return  # converged — single apex, last, correct dial → zero writes
            if count > 1:
                # Concurrent appends (the reconcile tick and the inline
                # register() fast-path both read the apex as absent) left
                # duplicate catch-alls; an earlier twin shadows every service
                # ordered after it even when the last copy looks converged.
                # Collapse the twins, then re-anchor a single apex LAST below.
                await self._admin.dedupe_route(_APEX_ID)
            if present:
                # Not last (a new service appended below), dial drifted, or we
                # just deduped → drop the survivor and re-append so exactly one
                # apex lands last again.
                await self._admin.delete_route(_APEX_ID)
            await self._admin.append_route(self._apex_route_obj())
            await self._audit_apex("proxy.apex_registered")
        except Exception:
            logger.warning("[proxy] reconcile: apex converge failed", exc_info=True)
