"""Build route specs and classify Caddy drift without I/O.

Route specs capture mode when built; custom domains use Host matchers in either
mode. Emitters follow each spec's shape rather than current settings. Reuse the
edge-auth fingerprint function so desired and live handlers compare equally.
This leaf must not import the admin client or manager.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .edgeauth import EdgeAuthMaterial, auth_fingerprint

# Caddy route objects we own carry this `@id` prefix so reconcile can tell
# nerdit-managed routes apart from anything an operator added by hand.
_ID_PREFIX = "nerdit-route-"

# The dashboard apex route's `@id`. Deliberately NOT under
# `_ID_PREFIX`: the service reconcile/prune loop and `CaddyAdmin.live_routes`
# both filter on that prefix, so the apex is invisible to them and is managed
# solely by `ProxyManager._reconcile_apex`. It also cannot collide with a
# service literally named `apex` (whose id would be `nerdit-route-apex`).
_APEX_ID = "nerdit-apex"

# Ports that need no `:port` suffix in a public URL for their scheme.
_DEFAULT_SCHEME_PORTS = {"https": 443, "http": 80}


#: Separator between the service half and the domain half of a domain route's
#: `@id` (P26 D-P26-2). `@` is not a legal character in a service NAME
#: (`core/secrets.py`'s `_DNS_LABEL_RE`) nor in a DNS name, so
#: `nerdit-route-<svc>@<dom>` splits unambiguously on the FIRST `@` and a
#: default id can never be mistaken for a domain one.
_DOMAIN_SEP = "@"


def _route_id(service_name: str) -> str:
    """Return the stable Caddy `@id` for a service's default route."""
    return f"{_ID_PREFIX}{service_name}"


def _domain_route_id(service_name: str, domain: str) -> str:
    """Return the stable Caddy `@id` for one custom-domain route (P26 D-P26-2).

    Composite by design: every `@id`-keyed site in the proxy (prune, `GET
    /routes`, deregister, the cutover's all-or-nothing repoint) learns the
    whole SET of a service's ids from this one grammar, with no second lookup
    and no state of its own.
    """
    return f"{_ID_PREFIX}{service_name}{_DOMAIN_SEP}{domain}"


def _split_route_id(route_id: str) -> tuple[str, str | None]:
    """Split a nerdit-owned `@id` into `(service_name, domain | None)`.

    `None` for the domain half means "the service's default route". The
    partition is on the FIRST `@`: a service name cannot contain one, so the
    service half is exact, and everything after it is the domain verbatim
    (which may itself contain no `@` either — the grammar is unambiguous in
    both directions).

    Callers pass ids already filtered on `_ID_PREFIX` (the apex id is
    deliberately outside it); a foreign id would simply yield a nonsense
    service half, never an exception.
    """
    body = route_id[len(_ID_PREFIX) :]
    svc, sep, dom = body.partition(_DOMAIN_SEP)
    return svc, (dom if sep else None)


def ordering_violations(live: Mapping[str, LiveRoute]) -> list[str]:
    """Domain-route ids that sit AFTER the first default route (P26 S-W4).

    Ordering is a **drift dimension**, not a one-shot placement: Caddy serves
    the first matching route, so in path mode a default route (matching
    `/<other-service>/*`) placed ahead of a custom-domain route would shadow
    it for that path — the request would reach the wrong app under the
    operator's own domain. Domain routes are therefore held ahead of every
    default one, and this classifier is what reconcile compares live state
    against; the ids it returns are deleted and re-inserted at index 0.

    Pure, and total over any live table: with no default routes present there
    is nothing a domain route can shadow, so the answer is `[]`. Ordered by
    live array index so the corrective pass is deterministic.

    In subdomain mode the defaults are Host-matched too, and validation
    guarantees a custom domain never falls under `base_domain` — so no
    default can match a custom domain's Host and the ordering is
    correctness-neutral there. The same single rule still applies (one rule, no
    mode branch); at worst it re-anchors routes that did not need it, once.
    """
    entries = [(rid, route) for rid, route in live.items() if rid.startswith(_ID_PREFIX)]
    defaults = [route.index for rid, route in entries if _split_route_id(rid)[1] is None]
    if not defaults:
        return []
    first_default = min(defaults)
    offenders = [
        (route.index, rid)
        for rid, route in entries
        if _split_route_id(rid)[1] is not None and route.index > first_default
    ]
    return [rid for _, rid in sorted(offenders)]


# --------------------------------------------------------------------------- #
# Mode-aware seams (Invariant #2). These three route-shaping functions plus a  #
# fourth, TLS-only site — `ProxyManager._tls_subjects` (feeding              #
# `_bootstrap_config`) — are the ONLY code that knows about `mode`; P3.5   #
# (subdomain) flips them, nothing else.                                        #
# --------------------------------------------------------------------------- #


def generate_route(service_name: str, *, mode: str = "path") -> str:
    """Return the route DATA persisted in `service_endpoints.route`.

    * `mode='path'`      → `"/<service_name>"` (the path prefix Caddy strips)
    * `mode='subdomain'` → `""` (P3.5: identity moves to the Host header)
    """
    if mode == "subdomain":
        return ""
    return f"/{service_name}"


def public_url_for(
    service_name: str,
    route: str | None,
    *,
    mode: str,
    hostname: str,
    base_domain: str | None,
    scheme: str,
    https_port: int,
    public_port: int | None = None,
) -> str | None:
    """Return the URL shown by `GET /services`, or `None` when not routed.

    `route is None` means the service holds no proxy route (proxy off, or not
    yet registered) → `None`. The URL is *regenerated* from
    `(service_name, hostname, mode)` rather than parsed from the stored route
    string, so a `mode` flip needs no row migration.

    `public_port` overrides the port ADVERTISED in the URL (not the bind):
    when an external proxy fronts the embedded Caddy (public 443 → loopback
    `https_port`), the reachable port differs from the bound one. `None`
    keeps the historical behaviour (advertise `https_port`).
    """
    if route is None:
        return None
    advertised = https_port if public_port is None else public_port
    suffix = "" if advertised == _DEFAULT_SCHEME_PORTS.get(scheme) else f":{advertised}"
    if mode == "subdomain":
        host = f"{service_name}.{base_domain or hostname}"
        return f"{scheme}://{host}{suffix}/"
    return f"{scheme}://{hostname}{suffix}/{service_name}/"


def domain_url_for(
    domain: str, *, scheme: str, https_port: int, public_port: int | None = None
) -> str:
    """Return the public URL of a custom domain (P26 D-P26-14).

    Not mode-aware, and deliberately so: a custom domain is served at its own
    ROOT in both proxy modes (the Host matcher carries the identity), which is
    the whole ergonomic win over a path prefix. The port suffix follows
    `public_url_for`'s rule verbatim — `public_port` wins when set,
    otherwise the bound `https_port`, and the scheme's default port is
    suppressed.

    Always returns a URL: unlike `public_url_for` there is no
    "not routed" input to answer `None` for — whether the domain currently
    answers is the caller's `state` field, not a property of the string.
    """
    advertised = https_port if public_port is None else public_port
    suffix = "" if advertised == _DEFAULT_SCHEME_PORTS.get(scheme) else f":{advertised}"
    return f"{scheme}://{domain}{suffix}/"


@dataclass(frozen=True)
class RouteSpec:
    """A resolved route: the service, its live upstream port, and Caddy `@id`."""

    service_name: str
    host_port: int
    route: str  # persisted projection (generate_route output)
    caddy_id: str
    auth: EdgeAuthMaterial | None = None
    """(P25 D-P25-8) Resolved edge-auth credential material, or ``None``.

    ``None`` means the route serves openly. Never a *declaration*: by the time
    a spec exists the ``${secrets.…}`` reference has been resolved and hashed,
    so :meth:`ProxyManager._caddy_route_obj` only ever renders material it can
    actually emit — a declaration that could not be materialized never reaches
    a spec at all (it fails closed upstream).
    """

    kind: Literal["default", "domain"] = "default"
    """(P26 D-P26-2) Which of a service's routes this is.

    A service has exactly one ``"default"`` spec (the path prefix or the
    ``<service>.<base>`` Host) and one ``"domain"`` spec per custom domain.
    Reconcile reads it to decide the upsert leg (domain routes are *prepended*,
    defaults appended) and to skip the route-string comparison, which is a
    property of the default route only.
    """

    shape: Literal["host", "path"] = "path"
    """The matcher shape this spec expects to be emitted and found live.

    Per-spec rather than per-mode (P26 WP1): a domain route is Host-shaped in
    BOTH proxy modes, so the drift classifier must compare a live route against
    the shape ITS OWN spec asks for, never against the manager's mode-derived
    ``_expected_shape``. Defaults to ``"path"`` so every pre-WP1 keyword
    construction keeps its meaning unchanged.
    """

    host: str | None = None
    """The Host matcher value when :attr:`shape` is ``"host"``, else ``None``."""


@dataclass(frozen=True)
class LiveRoute:
    """A nerdit-owned route as read live from Caddy.

    `dial` is the upstream the route currently proxies to (`None` when no
    `reverse_proxy` handler could be found). `shape` classifies the matcher:
    `"host"` (subdomain mode's Host matcher) or `"path"`. Reconcile compares
    BOTH to the desired state — dial-only comparison misses a stale matcher
    left behind by a mode flip when the upstream port happens to match.

    `count` is how many live route objects carry this `@id` (normally 1).
    Duplicates arise from the pre-PATCH `POST /id` upsert (which appended on
    Caddy 2.6.2) or an append race; `dial`/`shape` reflect the LAST
    occurrence — the one the id map resolves to — and reconcile self-heals the
    stale twins via `CaddyAdmin.dedupe_route`.

    A plain `dataclasses.dataclass` (Track B W22.M), not a
    `typing.NamedTuple`: the NamedTuple form shadowed `tuple.count`
    with this class's own `count` *field*, which mypy correctly flagged as an
    `[assignment]` incompatibility (the base `tuple.count` is a *method*).
    Nothing in this codebase used the tuple protocol (indexing, unpacking, or
    positional construction beyond the keyword sites in
    `CaddyAdmin.live_routes` and the tests) — equality/repr semantics are
    unchanged by the switch, so the mypy override is deleted rather than kept.
    """

    dial: str | None
    shape: str
    count: int = 1
    auth_fingerprint: str | None = None
    """(P25 D-P25-7) Digest of the live ``authentication`` handler, or ``None``.

    ``None`` means the live route carries no auth handler; :data:`_AUTH_UNPARSABLE`
    means it carries one whose shape is not what we emit. Reconcile compares this
    against the DESIRED fingerprint — without that dimension, an added or removed
    auth handler is invisible to the drift classifier and the steady state either
    silently strips the handler or never applies it.
    """

    index: int = -1
    """(P26 S-W4) Position of this id in Caddy's live route array.

    Of the LAST occurrence, for the same reason ``dial``/``shape`` are: that is
    the object the id map resolves to. Feeds :func:`ordering_violations`, the
    fifth drift dimension — a domain route that has drifted BEHIND a default
    one is silently shadowed, which no dial/shape/auth comparison can see. The
    ``-1`` default keeps every hand-built test construction valid; it reads as
    "position unknown", which the ordering classifier treats as ahead of
    everything (it can only ever be compared, never dereferenced).
    """


# A live `authentication` handler we cannot read as our own emission. It must
# read as DRIFT (forcing one corrective upsert), never as "no auth" — the whole
# point of the auth dimension is that an unrecognised edge-auth state converges
# to ours rather than being trusted. A constant string (not `None`,
# not an exception) keeps the comparison total.
_AUTH_UNPARSABLE = "unparsable"


def _extract_auth_fingerprint(route_obj: dict[str, Any]) -> str | None:
    """Digest the live `authentication` handler of a route object.

    Sibling of `_extract_dial`: the same recursive walk, tolerant of both
    emitted shapes (path mode nests the handler inside a `subroute`, subdomain
    mode keeps it at the top level) and of unknown sibling keys — Caddy's own
    Caddyfile adapter emits an extra `hash_cache` key we deliberately do not
    write, and a newer Caddy may add more.

    Tri-state, matching `nerdit.core.proxy.edgeauth.load_edge_auth`'s
    posture: no handler → `None`; exactly one `http_basic` account carrying
    a `username` and a `password` → `auth_fingerprint` over the two;
    ANY other shape (another provider, zero or several accounts, a missing key)
    → `_AUTH_UNPARSABLE`, i.e. drift.
    """

    def walk(node: Any) -> dict[str, Any] | None:
        if isinstance(node, dict):
            if node.get("handler") == "authentication":
                return node
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found is not None:
                    return found
        return None

    handler = walk(route_obj)
    if handler is None:
        return None
    providers = handler.get("providers")
    if not isinstance(providers, dict):
        return _AUTH_UNPARSABLE
    # Exactly the provider we emit, and only it: an unknown provider is an auth
    # state we cannot reproduce, so it is drift rather than something to adopt.
    if set(providers) != {"http_basic"}:
        return _AUTH_UNPARSABLE
    http_basic = providers.get("http_basic")
    if not isinstance(http_basic, dict):
        return _AUTH_UNPARSABLE
    accounts = http_basic.get("accounts")
    if not isinstance(accounts, list) or len(accounts) != 1 or not isinstance(accounts[0], dict):
        return _AUTH_UNPARSABLE
    user = accounts[0].get("username")
    password = accounts[0].get("password")
    if not isinstance(user, str) or not isinstance(password, str) or not user or not password:
        return _AUTH_UNPARSABLE
    return auth_fingerprint(user, password)


def _extract_dial(route_obj: dict[str, Any]) -> str | None:
    """Pull the first `reverse_proxy` upstream `dial` out of a route object.

    Tolerant of both shapes we emit (path mode wraps the proxy in a `subroute`;
    subdomain mode proxies directly), and of Caddy's config normalization, by
    recursively scanning `handle` lists for a `reverse_proxy` handler.
    """

    def walk(node: Any) -> str | None:
        if isinstance(node, dict):
            if node.get("handler") == "reverse_proxy":
                upstreams = node.get("upstreams") or []
                if upstreams and isinstance(upstreams[0], dict):
                    dial = upstreams[0].get("dial")
                    if isinstance(dial, str):
                        return dial
            for value in node.values():
                found = walk(value)
                if found:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found:
                    return found
        return None

    return walk(route_obj)


def _route_shape(route_obj: dict[str, Any]) -> str:
    """Classify a live route object's matcher shape (sibling of `_extract_dial`).

    `"host"` when any matcher set carries a `host` key (the subdomain-mode
    shape `ProxyManager._caddy_route_obj` emits), `"path"` otherwise.
    Tolerant of a missing/malformed `match` — that reads as `"path"`, which
    at worst forces one corrective upsert.
    """
    matchers = route_obj.get("match")
    if isinstance(matchers, list):
        for matcher in matchers:
            if isinstance(matcher, dict) and "host" in matcher:
                return "host"
    return "path"
