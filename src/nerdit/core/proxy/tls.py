"""Build and converge Caddy TLS policy before writing service routes.

`ProxyManager` initializes mixin state. Desired-policy hashes suppress redundant
writes, but adoption, config changes, and custom domains require convergence.
A trailing catch-all internal issuer prevents accidental public issuance.
Public ACME policies require explicitly opted-in domains and enabled ACME
settings; they precede the catch-all. Disable Caddy config persistence to avoid
saving edge-auth hashes outside the daemon data directory.

When ACME is enabled, the HTTP listener serves Caddy's built-in HTTP-01 handler
before routes, followed by an optional HTTPS redirect and explicit 404 fallback.
The listener and ACME policies are absent when disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nerdit.config.settings import ProxySettings
    from nerdit.db.queries import Queries
    from nerdit.db.rows import ServiceDomain

    from .admin import CaddyAdmin

logger = logging.getLogger(__name__)

#: Name of the Caddy HTTP server block that carries the ACME HTTP-01 listener
#:. One constant, two readers: `CaddyTlsMixin._bootstrap_config`
#: writes it, and the supervisor asks the live admin API whether an ADOPTED
#: Caddy actually carries it (review round 1) — a string literal in both places
#: is exactly the drift that would make "listening" a lie.
ACME_HTTP_SERVER = "nerdit-acme-http"


def partition_domains(
    rows: Iterable[ServiceDomain], *, acme_enabled: bool
) -> tuple[list[str], list[str]]:
    """Split `service_domains` rows into (every name, the ACME-issued names).

    Both lists are sorted and deduped: they are hashed into the convergence key
    (`CaddyTlsMixin._tls_hash_of`), so an unstable order would be read as
    drift and rewrite the subtree every tick.

    The single place the rule "a public certificate needs BOTH the row's own
    `acme=1` flag AND `[proxy.acme].enabled`" is written down. A row flagged
    `acme=1` on a node that later turned ACME off is therefore an ordinary
    internal-CA name here — it keeps its Host route and its internal leaf, and
    the operator sees the discrepancy as `cert_state='disabled'` rather than
    as a name that silently stopped being served.
    """
    names: set[str] = set()
    acme: set[str] = set()
    for row in rows:
        names.add(row.domain)
        if acme_enabled and row.acme:
            acme.add(row.domain)
    return sorted(names), sorted(acme)


class CaddyTlsMixin:
    """Owns TLS subject derivation, the bootstrap config, and per-tick convergence."""

    # -- state provided by ProxyManager.__init__ (composed, never set here) --
    _settings: ProxySettings
    _hostname: str
    _storage_root: Path
    _admin: CaddyAdmin
    _queries: Queries
    _tls_synced: bool
    _tls_hash: str | None

    def _tls_subjects(self) -> list[str]:
        """Return stable, deduplicated internal certificate subjects.

        Keep the primary hostname first, then extra hostnames; subdomain mode also adds
        `*.<base_domain or hostname>`. Alternates gain certificates, not generated URLs.
        Preserve order because the TLS convergence hash is order-sensitive.
        """
        subject = self._settings.hostname_override or self._hostname
        subjects = [subject, *self._settings.extra_hostnames]
        if self._settings.mode == "subdomain":
            subjects.append(f"*.{self._settings.base_domain or self._hostname}")
        return list(dict.fromkeys(subjects))

    def _acme_issuer(self) -> dict[str, Any]:
        """Build an HTTP-01-only ACME issuer for the configured directory and port.

        Use `ca` for the directory URL and an unprefixed contact email. Disable TLS-ALPN
        and always set the HTTP alternate port. Optional trusted-root file paths affect
        Caddy's directory client and allow private CAs.
        """
        acme = self._settings.acme
        issuer: dict[str, Any] = {
            "module": "acme",
            "ca": acme.directory,
            "email": acme.email,
            "challenges": {
                "tls-alpn": {"disabled": True},
                "http": {"alternate_port": acme.http_port},
            },
        }
        if acme.ca_root_file:
            issuer["trusted_roots_pem_files"] = [acme.ca_root_file]
        return issuer

    def _acme_http_server(self) -> dict[str, Any]:
        """Build the public HTTP-01 listener with only static fallback handlers.

        Caddy handles challenges before routes. Host-independent redirect/404 handlers
        need no domain convergence and expose no upstream. Redirect to the advertised
        HTTPS port when it differs from 443; `{http.request.host}` strips input ports.
        """
        advertised = self._settings.public_port or self._settings.https_port
        port_suffix = "" if advertised == 443 else f":{advertised}"
        routes: list[dict[str, Any]] = []
        if self._settings.acme.http_redirect:
            routes.append(
                {
                    "handle": [
                        {
                            "handler": "static_response",
                            "status_code": 308,
                            "headers": {
                                "Location": [
                                    "https://{http.request.host}"
                                    f"{port_suffix}"
                                    "{http.request.uri}"
                                ]
                            },
                        }
                    ],
                    "terminal": True,
                }
            )
        # ALWAYS last and always present: a Caddy server whose route table
        # matches nothing answers `200 OK` with an empty body (measured), so
        # "else 404" is an explicit route, not a default.
        routes.append({"handle": [{"handler": "static_response", "status_code": 404}]})
        return {
            "listen": [f":{self._settings.acme.http_port}"],
            "routes": routes,
            # Caddy treats this as the HTTP-role listener and applies no
            # automatic HTTPS to it; the flag says so out loud, and matches the
            # main server.
            "automatic_https": {"disable_redirects": True},
        }

    def _tls_desired_app(
        self, domains: Sequence[str], acme_domains: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Build ordered TLS automation policies for node and custom-domain names.

        Automate stable node subjects followed by sorted, deduplicated domains before
        writing Host routes. ACME names remain automated; their policy precedes the
        internal-subject and catch-all internal policies only when ACME is enabled.
        The final subjectless internal policy prevents route-triggered public issuance.

        ACME policies have one issuer. The catch-all is not a fallback for names matched
        earlier: pending/failed issuance means TLS fails, including after switching an
        existing internal name to ACME. Do not add an internal fallback issuer: route
        reloads cancel ACME attempts, allowing a fast internal certificate to suppress
        public issuance until renewal.
        """
        subjects = self._tls_subjects()
        known = set(subjects)
        acme_names = sorted(set(acme_domains)) if self._settings.acme.enabled else []
        policies: list[dict[str, Any]] = []
        if acme_names:
            policies.append({"subjects": acme_names, "issuers": [self._acme_issuer()]})
        policies.append({"subjects": subjects, "issuers": [{"module": "internal"}]})
        # The catch-all, LAST and subject-less (F4/F6). Deleting it would
        # re-open public ACME as the fallback issuer for any Host-derived name
        # — see the module docstring.
        policies.append({"issuers": [{"module": "internal"}]})
        return {
            "certificates": {"automate": [*subjects, *sorted({d for d in domains} - known)]},
            "automation": {"policies": policies},
        }

    @staticmethod
    def _tls_hash_of(app: dict[str, Any]) -> str:
        """Canonical-JSON digest of a `tls` subtree — the convergence key.

        `sort_keys` makes the digest independent of key order (Caddy returns
        objects in its own order) while list order stays significant, which is
        correct: the policy list is ordered and the catch-all MUST be last, so a
        reordered live policy list is genuine drift, not noise.
        """
        return hashlib.sha256(
            json.dumps(app, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _bootstrap_config(self) -> dict[str, Any]:
        """Build initial Caddy listeners, TLS policy, and storage configuration.

        Disabled ACME emits only the TLS listener. Enabled ACME adds its HTTP listener
        at process spawn; ACME settings require restart rather than per-tick reload.
        """
        http_app: dict[str, Any] = {
            # Mark our listen port as THE HTTPS port so Caddy applies TLS
            # automation to it. Without this, automation only covers the
            # default 443 and any other port (e.g. an 8443 fallback or a
            # test's ephemeral port) is served as plain HTTP.
            "https_port": self._settings.https_port,
            "servers": {
                "nerdit": {
                    "listen": [f":{self._settings.https_port}"],
                    "routes": [],
                    # This server never redirects. With [proxy.acme] off there
                    # is no HTTP listener at all — nothing binds the privileged
                    # :80 just to redirect, and apps are reached directly over
                    # HTTPS. With it on, the redirect belongs to the separate
                    # `nerdit-acme-http` server below, which exists because
                    # the CA has to reach the HTTP-01 solver on that port.
                    "automatic_https": {"disable_redirects": True},
                }
            },
        }
        if self._settings.acme.enabled:
            # `apps.http.http_port` is what tells Caddy which listener plays
            # the HTTP role — that is the listener whose solver answers the
            # challenge. `_acme_issuer` still names the port explicitly rather
            # than relying on this value propagating into the solver's default.
            http_app["http_port"] = self._settings.acme.http_port
            http_app["servers"][ACME_HTTP_SERVER] = self._acme_http_server()
        return {
            "admin": {
                "listen": self._settings.admin_addr,
                # Never autosave the live config. Caddy's
                # autosave file lives outside the daemon's `data_dir` and the
                # live config carries bcrypt-hashed edge-auth material, so
                # persistence here is credential sprawl for no benefit: the
                # daemon reloads this bootstrap on every spawn and reconcile is
                # the source of truth for routes. Applies to a Caddy WE spawn;
                # an ADOPTED pre-WP1 process keeps autosaving until its next
                # respawn (reconfiguring `/config/admin` restarts the admin
                # listener mid-tick — not worth it). Documented in proxy.md.
                "config": {"persist": False},
            },
            "storage": {"module": "file_system", "root": str(self._storage_root)},
            "apps": {
                # Don't let Caddy try to sudo-install its root CA into the system
                # trust store at startup (wrong for a headless daemon — it would
                # prompt / fail). Operators run `caddy trust` or copy root.crt; see
                # docs/guide/proxy.md.
                "pki": {"certificate_authorities": {"local": {"install_trust": False}}},
                "http": http_app,
                # Proactively obtain the cert(s). This is REQUIRED for
                # path-based routing: our routes match on path, not Host, so
                # Caddy has no Host matcher to infer the managed name from —
                # without an explicit `automate` list it never provisions a
                # cert and every TLS handshake fails with an internal-error
                # alert. The automation policies say HOW (internal CA, always).
                #
                # Subdomain mode adds a `*.<base>` wildcard subject:
                # one internal-CA wildcard cert covers every routed
                # `<service>.<base>` name, so route churn never touches TLS
                # config. The wildcard in the POLICY subjects is load-bearing
                # on its own: Caddy also auto-manages each Host-matched route
                # name individually, and those auto-derived names get INTERNAL
                # issuance because they match a policy — which since WP1 is
                # guaranteed for EVERY name by the trailing catch-all (F4/F6).
                #
                # Custom domains are absent here on purpose: the DB is not
                # readable at bootstrap time (this config is also passed as
                # `--config` at spawn). The first reconcile tick converges
                # them through `_converge_tls`.
                "tls": self._tls_desired_app(()),
            },
        }

    # -- TLS convergence -------------------------------------------------------

    async def _load_domain_partition(self) -> tuple[list[str], list[str]]:
        """`(every name, the ACME names)` from the table — or two empty lists.

        Total by construction: no queries object (route-shaping-only managers
        and several test stubs), a stub without the method, or any DB failure
        all answer `([], [])`. That is the safe direction — a tick that cannot
        read the table converges TLS to the node's own names and simply does not
        add the domains yet; the route loop reads the same table and would fail
        the same way, so the two never disagree in a way that lands a Host route
        without a certificate. It is also the fail-closed direction for ACME: an
        unreadable table withholds the public-issuer policy rather than
        inventing one.
        """
        # `getattr` rather than a call: several managers are built with no
        # queries at all (pure route-shaping) and several test/smoke stubs
        # implement only the handful of methods the proxy actually uses.
        lister = getattr(self._queries, "list_service_domains", None)
        if lister is None:
            return [], []
        try:
            rows = await lister()
        except Exception:  # noqa: BLE001 — an unreadable table is "no domains yet"
            logger.debug("[proxy] could not read service_domains for TLS", exc_info=True)
            return [], []
        return partition_domains(rows, acme_enabled=self._settings.acme.enabled)

    async def _load_domain_names(self) -> list[str]:
        """Every custom-domain name, sorted — the names half of the partition."""
        names, _acme = await self._load_domain_partition()
        return names

    async def _converge_tls(
        self,
        domains: Sequence[str] | None = None,
        acme_domains: Sequence[str] | None = None,
    ) -> bool:
        """Converge the whole TLS policy before callers write domain routes.

        Cache the desired subtree hash to avoid admin I/O when converged. Unreadable
        state retries later without being treated as drift. Errors are contained and
        produce no mutation audit record.

        A failed convergence must withhold Host routes: an adopted Caddy without the
        internal catch-all could otherwise initiate unwanted public ACME issuance.
        Re-derive domain/ACME partitions when omitted; an explicit domain list without
        an ACME subset permits no public issuance for that call.

        Returns:
            Whether the live subtree is known to carry the desired policy.
        """
        try:
            if domains is None:
                names, acme_names = await self._load_domain_partition()
            else:
                names = list(domains)
                acme_names = list(acme_domains or ())
            desired = self._tls_desired_app(names, acme_names)
            digest = self._tls_hash_of(desired)
            if self._tls_synced and self._tls_hash == digest:
                return True
            live = await self._admin.get_tls_config()
            if live is None:
                # Unreadable — leave the latch unset and retry on a later tick.
                # False, not True: we do NOT know the catch-all is in place.
                return False
            if self._tls_hash_of(live) == digest:
                self._tls_synced = True
                self._tls_hash = digest
                return True
            await self._admin.set_tls_config(desired)
            self._tls_synced = True
            self._tls_hash = digest
            logger.info(
                "[proxy] TLS subtree converged: subjects %s + %d custom domain(s), %d via ACME",
                self._tls_subjects(),
                len(names),
                len(acme_names),
            )
            return True
        except Exception:
            logger.warning("[proxy] TLS convergence failed; will retry", exc_info=True)
            return False
