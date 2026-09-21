"""Record mutations as redacted audit rows and EventBus events.

Middleware order is Auth → Audit → Idempotency. Auth records its own denials;
Audit records idempotent replays with null params because no route ran.
Actions come from method and templated path, with `/api` stripped, never from
`scope['route']`, which is unavailable on replays and across middleware tasks.

Never read request bodies here. Routes supply redacted `audit_params`.
`request.state.audit_skip` suppresses successful no-op records only; denials
and errors are always recorded.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from nerdit.daemon.auth import current_principal
from nerdit.daemon.errors import request_id_of

logger = logging.getLogger(__name__)

# Methods that mutate state — the only ones the audit middleware acts on.
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Collection segments whose following path segment is a resource id. Used only
# to template the *fallback* action for routes not in the explicit map below, so
# unknown mutations still log with bounded cardinality.
_ID_COLLECTIONS = {
    "tokens",
    "gpus",
    "audit",
    "services",
    "models",
    "config",
    "secrets",
    "deploy",
    "app-templates",
    # (P40b) `/projects/<name>` — the D-P40-10 collection.
    "projects",
    # (P40c) `/projects/<name>/variables/<key>`.
    "variables",
}

# Explicit route → (action, target_type, id-group) map. Ordered; first match
# wins. `id_group` is the regex capture index for the target id (or `None`).
# Patterns match the path **after** the optional `/api` prefix is stripped, so
# bare-root and `/api` mounts collapse to one action.
_ROUTE_RULES: list[tuple[str, re.Pattern[str], str, str | None, int | None]] = [
    ("POST", re.compile(r"^/tokens$"), "token.create", "token", None),
    # Kept adjacent to token.create for readability. There is no
    # parametric POST /tokens/{id} rule to be shadowed by (and the revoke rule
    # below is DELETE-only), so first-match ordering is not load-bearing here —
    # but a future POST /tokens/{id} MUST go below this literal.
    ("POST", re.compile(r"^/tokens/self/rotate$"), "token.rotate", "token", None),
    ("DELETE", re.compile(r"^/tokens/([^/]+)$"), "token.revoke", "token", 1),
    ("PUT", re.compile(r"^/config/daemon/([^/]+)$"), "config.update", "config", 1),
    ("POST", re.compile(r"^/config/daemon/apply$"), "config.apply", "config", None),
    ("PUT", re.compile(r"^/config/apps/([^/]+)/([^/]+)$"), "config.app_update", "service", 1),
    ("POST", re.compile(r"^/deploy$"), "deploy.create", "service", None),
    # Literal /deploy/git must sit ABOVE the /deploy/{name}/rollback pattern
    # (first match wins) — adjacency, no actual overlap ("git" is not "*/rollback").
    ("POST", re.compile(r"^/deploy/git$"), "deploy.git_create", "service", None),
    ("POST", re.compile(r"^/deploy/([^/]+)/rollback$"), "deploy.rollback", "service", 1),
    ("POST", re.compile(r"^/deploy/([^/]+)/redeploy$"), "deploy.redeploy", "service", 1),
    # The workspace surface. The name rides the path, so no
    # `_TARGET_STAMP_ALLOWED` entry is needed. Omitting these rules would also
    # silently disable the idempotency body hash for both routes: the
    # space-bearing fallback action is excluded by `_hashable_body`.
    ("PUT", re.compile(r"^/workspaces/([^/]+)/files$"), "workspace.write", "workspace", 1),
    (
        "POST",
        re.compile(r"^/workspaces/([^/]+)/deploy$"),
        "deploy.workspace_create",
        "service",
        1,
    ),
    # Literal rotate-key rule must sit ABOVE the generic secret.set pattern
    # (first match wins), else rotation is audited as set(service="rotate-key").
    ("POST", re.compile(r"^/secrets/rotate-key$"), "secret.rotate_key", "secret", None),
    ("POST", re.compile(r"^/secrets/([^/]+)$"), "secret.set", "secret", 1),
    ("DELETE", re.compile(r"^/secrets/([^/]+)/([^/]+)$"), "secret.delete", "secret", 1),
    ("DELETE", re.compile(r"^/secrets/([^/]+)$"), "secret.delete", "secret", 1),
    ("POST", re.compile(r"^/app-templates/([^/]+)/deploy$"), "template.deploy", "template", 1),
    ("POST", re.compile(r"^/models$"), "model.serve", "model", None),
    # (P37) The dump pair. Both are anchored on a sub-path of
    # ``/databases/<name>``, so neither can be shadowed by (nor shadow) the
    # anchored ``^/databases$`` create rule below — they sit ABOVE it anyway,
    # because first-match ordering is the property a future non-anchored
    # ``/databases`` prefix rule would silently break. Group 1 is the DATABASE,
    # so a reader filtering by target sees one database's whole dump history in
    # one place; the tar basename rides ``audit_params`` beside the engine and
    # the byte count. ``GET …/dumps`` is a read and is never audited (the
    # middleware only acts on mutations).
    ("POST", re.compile(r"^/databases/([^/]+)/dump$"), "database.dump", "database", 1),
    ("POST", re.compile(r"^/databases/([^/]+)/restore$"), "database.restore", "database", 1),
    ("POST", re.compile(r"^/databases$"), "database.create", "database", None),
    ("POST", re.compile(r"^/services$"), "service.create", "service", None),
    ("POST", re.compile(r"^/services/([^/]+)/stop$"), "service.stop", "service", 1),
    ("POST", re.compile(r"^/services/([^/]+)/restart$"), "service.restart", "service", 1),
    ("POST", re.compile(r"^/services/([^/]+)/run$"), "service.run", "service", 1),
    ("DELETE", re.compile(r"^/services/([^/]+)$"), "service.delete", "service", 1),
    # (P26 WP-H) The hosted-share pair. The service name rides the path (group
    # 1), so no `_TARGET_STAMP_ALLOWED` entry is needed. Both patterns are
    # anchored on `/share`, so neither can be shadowed by (nor shadow) the
    # generic `/services/([^/]+)$` delete rule above — they are kept adjacent
    # to the other service rules for readability, not for ordering. `GET` is a
    # read and is never audited (the middleware only acts on mutations).
    ("PUT", re.compile(r"^/services/([^/]+)/share$"), "share.set", "service", 1),
    ("DELETE", re.compile(r"^/services/([^/]+)/share$"), "share.removed", "service", 1),
    # The custom-domain pair. Group 1 is the SERVICE — the target of
    # the mutation is the app, and the domain itself rides `audit_params`
    # (`{service, domain, acme}`), so a reader filtering by target sees an
    # app's whole exposure history in one place. Both patterns are anchored on
    # `/domains/<name>`, so neither shadows (nor is shadowed by) the generic
    # `/services/([^/]+)$` delete rule above (a path containing `/domains/`
    # cannot match it). `GET` is a read and is never audited (the middleware
    # only acts on mutations).
    #
    # The domain group is `.+` — NOT `[^/]+` — because both routes are
    # declared `{domain:path}` (WP1 F1) and the middleware sees the DECODED
    # path, so a pasted `https://host/<secret>` arrives with real slashes. A
    # slash-bearing paste must still hit the mapped rule: otherwise
    # `derive_action` falls through to the `f"{method} {path}"` fallback
    # and the raw string — a refused, possibly secret-shaped value — lands in
    # `audit_log.action` and on the admin `audit.*` event stream verbatim.
    ("PUT", re.compile(r"^/services/([^/]+)/domains/(.+)$"), "domain.added", "service", 1),
    (
        "DELETE",
        re.compile(r"^/services/([^/]+)/domains/(.+)$"),
        "domain.removed",
        "service",
        1,
    ),
    ("POST", re.compile(r"^/daemon/restart$"), "daemon.restart", "daemon", None),
    ("POST", re.compile(r"^/system/gc$"), "system.gc", "system", None),
    # Literal /system/backup/volumes must sit ABOVE /system/backup so it is not
    # shadowed (both are anchored, so no true overlap — kept adjacent for clarity).
    ("POST", re.compile(r"^/system/backup/volumes$"), "system.backup_volume", "system", None),
    ("POST", re.compile(r"^/system/backup$"), "system.backup", "system", None),
    # The claim/unlink pair. Audit action names are past-tense verbs, matching
    # the `link.connected`/`link.disconnected` durable event family — these
    # two are config acts, so the audit row is their only record (no new event
    # type).
    ("POST", re.compile(r"^/link/claim$"), "link.created", "link", None),
    # Name both device actions so link.* filtering shows the full flow. Approved polls
    # override audit_action to link.created; derive_action remains path-only for
    # idempotency. Request bodies contain only public config or a grantless session,
    # so body hashing is safe. User codes occur only in responses; device_code stays
    # in process memory.
    (
        "POST",
        re.compile(r"^/link/device/poll$"),
        "link.device_poll",
        "link",
        None,
    ),
    ("POST", re.compile(r"^/link/device$"), "link.device_started", "link", None),
    # (P26 WP-H) The hosted-metadata re-read. A config act like the two above,
    # so the audit row is its only record — and, like them, a literal path whose
    # node id the route stamps (see `_TARGET_STAMP_ALLOWED`).
    ("POST", re.compile(r"^/link/refresh$"), "link.refreshed", "link", None),
    # The cloud's entitlement push. Recorded CHANGE-only: the
    # route sets `request.state.audit_skip` on an unchanged or out-of-order
    # push, so a 300 s re-assert loop does not write a row every five minutes.
    # Denials are always recorded (the skip is honoured for 2xx only), which is
    # the half that matters — a 403 here means something tried to write the
    # mirror from outside the tunnel. Like the three rules above it, a literal
    # path whose node id the route stamps (see `_TARGET_STAMP_ALLOWED`).
    ("PUT", re.compile(r"^/link/entitlement$"), "link.entitlement", "link", None),
    # The cloud's GitHub installation-token push. Change-only like
    # the entitlement push; the row's params are the installation id, the
    # expiry and the repo COUNT — never the token, never the repo names.
    ("PUT", re.compile(r"^/link/github-token$"), "link.github_token", "link", None),
    # The cloud's push-to-deploy nudge. Always recorded (it is a
    # real act, unlike the re-asserted mirrors above); the row's params are
    # the canonical repo, the ref, the pushed sha and the matched service
    # names — public facts about a public push, no credential anywhere near.
    ("POST", re.compile(r"^/link/git-nudge$"), "gitwatch.nudge", "link", None),
    ("DELETE", re.compile(r"^/link$"), "link.revoked", "link", None),
    # The product-license install pair. Both paths are literals —
    # the license id lives in the VERIFIED claims, never in the path — so the
    # route stamps the target (see `_TARGET_STAMP_ALLOWED` below).
    ("POST", re.compile(r"^/license$"), "license.install", "license", None),
    ("DELETE", re.compile(r"^/license$"), "license.remove", "license", None),
    # (P40b / D-P40-10) The project pair. The create's name rides the body, so
    # the route stamps it (see `_TARGET_STAMP_ALLOWED`); the delete's rides
    # the path. Bodies carry names only — no `NO_BODY_HASH_ACTIONS` entry.
    ("POST", re.compile(r"^/projects$"), "project.create", "project", None),
    ("DELETE", re.compile(r"^/projects/([^/]+)$"), "project.delete", "project", 1),
    # (P40d / D-P40-10) The declaration apply. Multipart, so never body-hashed
    # (the `deploy.create` treatment); `token_ref` is a reference name, so no
    # `NO_BODY_HASH_ACTIONS` entry. Params are hand-built by the route.
    ("POST", re.compile(r"^/projects/([^/]+)/apply$"), "project.apply", "project", 1),
    # (P40c / D-P40-10) The variable pair; the target is the project. The set's
    # body is a value map, so `variable.set` is in `NO_BODY_HASH_ACTIONS` and
    # the route hand-builds its params (names only, never `environment`).
    ("PUT", re.compile(r"^/projects/([^/]+)/variables$"), "variable.set", "project", 1),
    (
        "DELETE",
        re.compile(r"^/projects/([^/]+)/variables/([^/]+)$"),
        "variable.unset",
        "project",
        1,
    ),
]

# Actions whose target the route may stamp via `request.state.audit_target`
# (the resource name lives in the request body, so the path rules above leave
# target_id null). Maps action -> the target_type the stamp asserts. The stamp
# only ever FILLS a null target; the one sanctioned rewrite is template.deploy,
# whose path-derived target is the *template* id — it is re-pointed to the
# deployed service (the template id stays in the event params).
_TARGET_STAMP_ALLOWED: dict[str, str] = {
    # The rotated token's id lives on the principal, not in the
    # path (`/tokens/self/rotate` is a literal), so the route stamps it —
    # without this entry the stamp is silently dropped and the rotate is audited
    # with a null target, which is exactly the attribution the trade rests on.
    "token.rotate": "token",
    "deploy.create": "service",
    "deploy.plan": "service",
    "deploy.git_create": "service",
    "deploy.git_plan": "service",
    "template.deploy": "service",
    # (P27 WP-C2) Both link paths are literals — the node id lives in the cloud's
    # claim RESPONSE (and, on unlink, in the stored config), never in the path —
    # so the route stamps it, the `token.rotate` precedent. Without these
    # entries the stamp is silently dropped and the row carries a null target.
    "link.created": "link",
    "link.revoked": "link",
    # (P26 WP-H) Same shape: `/link/refresh` is a literal path and the node id
    # comes out of the stored `[link]` section, so the route stamps it.
    "link.refreshed": "link",
    # Same shape again: `/link/entitlement` is a literal path and the
    # node id comes off the live `LinkStatus`, so the route stamps it.
    "link.entitlement": "link",
    # And again for the token push and the nudge.
    "link.github_token": "link",
    "gitwatch.nudge": "link",
    # The license id (`lid`) is the customer-free correlation
    # handle the row is worth having — it comes out of the verified claims, not
    # out of the path, so the route stamps it.
    "license.install": "license",
    "license.remove": "license",
    # (P40b) The project name lives in the create body (the `model.serve`
    # shape), so the route stamps it.
    "project.create": "project",
}

# Redaction denylist — any key whose lowercase name is here is masked. Secret
# values must never reach the audit row or the event bus.
_REDACT_KEYS = {
    "authorization",
    # (P27 WP-C2) The link code is a secret in transit. The claim route builds
    # its audit params by hand and the code is never a member — this entry is
    # defense in depth, so a future mistake masks instead of leaking.
    "code",
    # The license blob. Like `code`, this is depth only — the
    # install route builds its params by hand from the verified claims and the
    # blob is never a member — so a future mistake masks instead of leaking.
    "blob",
    "token",
    "token_ref",
    "api_key",
    "auth_token",
    "password",
    "secret",
    "secrets",
    # (P40c / D-P40-10) The variable set body is `{values: {K: v}}`. Depth only:
    # the route hand-builds its params and a value is never a member.
    "value",
    "values",
}
# Keys whose value is a free-form map of *user-chosen* names → secret-bearing
# values (e.g. a service's injected `env`). Name-based matching cannot catch
# user keys like `OPENAI_API_KEY`/`DB_PASSWORD`, so the whole map is treated
# as opaque: keys are kept (auditability — which vars were set) but every value
# is masked. `env` is P2's only secret channel until P4's SecretManager.
_REDACT_MAP_KEYS = {
    "env",
    "environment",
}
_REDACTED = "***"


def _strip_api_prefix(path: str) -> str:
    """Drop a leading `/api` so dual-mounted routes share one action."""
    if path == "/api":
        return "/"
    if path.startswith("/api/"):
        return path[4:]
    return path


# The MCP transport mount (P13c §3). Boundary-exact on purpose: exactly the
# mount root or a slash-separated descendant, never a bare startswith that
# would also match a future /api/mcpX route.
_MCP_MOUNT = "/api/mcp"


def is_mcp_path(path: str) -> bool:
    """True iff `path` is the MCP transport mount or lives under it."""
    return path == _MCP_MOUNT or path.startswith(_MCP_MOUNT + "/")


def _templated_path(path: str) -> str:
    """Replace resource-id segments with `{id}` for bounded-cardinality actions.

    Used only for the fallback action of routes not in `_ROUTE_RULES`. Anything after a
    `domains` segment collapses to a single
    `{domain}` and the walk stops: the domain is declared `{domain:path}`,
    so the remainder can be a whole pasted URL — a slash-split `{id}` would
    still leave the tail (the secret-shaped part) in the action. Templating the
    WHOLE remainder is defence in depth for any future `{...:path}` mutation
    under `/domains/`; the mapped rules above already keep the two shipped
    domain routes off this path entirely.
    """
    out: list[str] = []
    prev: str | None = None
    for seg in path.split("/"):
        if prev == "domains" and seg:
            out.append("{domain}")
            break
        if prev in _ID_COLLECTIONS and seg:
            out.append("{id}")
        else:
            out.append(seg)
        if seg:
            prev = seg
    return "/".join(out)


def derive_action(method: str, path: str) -> tuple[str, str | None, str | None]:
    """Map an HTTP method + path to `(action, target_type, target_id)`.

    Matches against `_ROUTE_RULES` (after stripping `/api`); unmapped
    mutations fall back to `f"{method} {templated_path}"` with no target. This
    is deliberately independent of `scope['route']` so it is identical for
    bare-root and `/api` mounts and works on idempotency replays.
    """
    p = _strip_api_prefix(path)
    for rule_method, pattern, action, target_type, id_group in _ROUTE_RULES:
        if rule_method != method:
            continue
        match = pattern.match(p)
        if match is None:
            continue
        target_id = match.group(id_group) if id_group is not None else None
        return action, target_type, target_id
    return f"{method} {_templated_path(p)}", None, None


def _mask_map_values(value: Any) -> Any:
    """Mask every value of an opaque secret-bearing map, keeping its keys."""
    if isinstance(value, dict):
        return {key: _REDACTED for key in value}
    return _REDACTED  # non-dict env (shouldn't happen) → fully masked


def redact(value: Any) -> Any:
    """Recursively mask values whose key is in `_REDACT_KEYS`, and mask
    every value of an opaque secret map (`_REDACT_MAP_KEYS`, e.g. `env`).
    """
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, val in value.items():
            lk = key.lower() if isinstance(key, str) else key
            if lk in _REDACT_KEYS:
                out[key] = _REDACTED
            elif lk in _REDACT_MAP_KEYS:
                out[key] = _mask_map_values(val)
            else:
                out[key] = redact(val)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def audit_params(model: Any) -> dict[str, Any]:
    """Build a redacted params dict from a pydantic model (or plain dict).

    Routes call this and assign the result to `request.state.audit_params`;
    the middleware serializes it into the `params_redacted` column. Secret
    keys are masked here so a secret value never reaches the audit store.
    """
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        data = model.model_dump(mode="json")
    elif isinstance(model, dict):
        data = model
    else:
        data = {"value": model}
    return redact(data)


async def record_out_of_band(
    request: Request,
    *,
    action: str,
    target_type: str,
    target_id: str | None,
    params: dict[str, Any],
) -> None:
    """Insert one out-of-band audit row attributed to the requester.

    A best-effort *second* row on top of the middleware's own row for the same
    request (e.g. `deploy.create` / `config.app_update` / `service.delete`
    / `database.create`) — routes call this directly when they need a durable
    signal that does not fit the one-row-per-request shape the middleware
    provides (a purge, a minted credential, a shared-secret reference). A
    logging failure here must never mask the caller's own response.
    """
    principal = current_principal(request)
    queries = request.app.state.queries
    try:
        await queries.insert_audit_log(
            action=action,
            result="ok",
            principal_id=principal.token_id,
            principal_role=principal.role.value,
            target_type=target_type,
            target_id=target_id,
            params_redacted=json.dumps(params),
            request_id=request_id_of(request),
        )
    except Exception:
        logger.warning("Failed to record %s for %s", action, target_id, exc_info=True)


# The PLR0913 below is the audit ROW's own shape (action/target pair, result
# status, principal pair, correlation id) plus the two sinks it writes to —
# every one of them a distinct column, so bundling them would only move the
# list. Suppressed inline rather than per-file (the module is otherwise clean).
async def record_denial(  # noqa: PLR0913
    *,
    queries: Any,
    bus: Any,
    action: str,
    target_type: str | None,
    target_id: str | None,
    status_code: int,
    principal_id: str | None,
    principal_role: str | None,
    request_id: str | None,
) -> None:
    """Append a `result='denied'` audit row and mirror it on the `audit.*` bus.

    The one writer for "a request was refused before any route ran", shared by
    `ScopedTokenAuthMiddleware._deny` and by the MCP transport's own pre-body
    refusals — those short-circuit inside the tool body, so the inner loopback
    hop that used to produce this row never happens, and without it a readonly
    or out-of-scope caller hammering a write tool leaves no trace at all.

    Best effort in both halves: an audit failure must never mask the denial it
    is recording. Carries no request body and no token material.
    """
    if queries is not None:
        try:
            await queries.insert_audit_log(
                action=action,
                target_type=target_type,
                target_id=target_id,
                result="denied",
                principal_id=principal_id,
                principal_role=principal_role,
                status_code=status_code,
                request_id=request_id,
            )
        except Exception:
            logger.warning("Failed to record auth denial audit row", exc_info=True)

    if bus is not None:
        try:
            bus.publish(
                {
                    "type": f"audit.{action}",
                    "ts": datetime.now(UTC).isoformat(),
                    "action": action,
                    "result": "denied",
                    "status_code": status_code,
                    "principal_id": principal_id,
                    "principal_role": principal_role,
                    "target_type": target_type,
                    "target_id": target_id,
                    "params": None,
                    "request_id": request_id,
                }
            )
        except Exception:
            logger.warning("Failed to publish auth denial event", exc_info=True)


async def record_shared_referenced(request: Request, service: str, keys: list[str]) -> None:
    """Insert one `secret.shared_referenced` row attributed to the requester.

    Fired by the deploy / app-config write paths when a parsed `[ai.*]` spec
    carries `${secrets.shared.KEY}` refs — the admin's *before-launch* signal
    that a principal pointed an app at shared secrets (the compensating control
    for the shared-scope confidentiality boundary). Thin wrapper over
    `record_out_of_band`.
    """
    await record_out_of_band(
        request,
        action="secret.shared_referenced",
        target_type="secret",
        target_id="shared",
        params={"service": service, "keys": keys},
    )


class AuditMiddleware(BaseHTTPMiddleware):
    """Record every mutating request that reaches a route."""

    def __init__(
        self,
        app,  # noqa: ANN001
        get_queries: Callable[[], Any | None] | None = None,
        get_event_bus: Callable[[], Any | None] | None = None,
    ) -> None:
        super().__init__(app)
        self._get_queries = get_queries
        self._get_event_bus = get_event_bus

    def _queries(self):  # noqa: ANN202
        return self._get_queries() if self._get_queries is not None else None

    def _event_bus(self):  # noqa: ANN202
        return self._get_event_bus() if self._get_event_bus is not None else None

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        """Pass safe methods through; record mutations (incl. failures/replays)."""
        if request.method not in _MUTATING_METHODS:
            return await call_next(request)

        # (P13c §3) MCP framing is excluded from recording: every MCP message is
        # a POST and would flood the log with noise, while the inner loopback
        # hops are audited normally with the REAL caller's principal (better
        # fidelity than an opaque POST /api/mcp row). Auth denials on /api/mcp
        # are still recorded — the auth middleware's own denial path, not here.
        if is_mcp_path(request.url.path):
            return await call_next(request)

        try:
            response = await call_next(request)
        except Exception:
            # The route (or an inner middleware) raised before producing a
            # response — record the attempt as an error, then re-raise so the
            # exception handlers still build the envelope.
            await self._record(request, status_code=500, result="error")
            raise

        # A route may declare that a SUCCESSFUL request changed nothing
        # worth a row — today only the cloud's entitlement push, which re-asserts
        # the same value every few minutes and would otherwise bury the log.
        # Deliberately honoured for 2xx/3xx ONLY: a denial or an error is never
        # a route's to suppress, so a future misuse of the flag cannot hide a
        # 403. The flag carries no payload, so it can never leak one.
        if getattr(request.state, "audit_skip", False) and response.status_code < 400:
            return response

        if response.headers.get("Idempotent-Replay") == "true":
            result = "replay"
        elif response.status_code in (401, 403):
            # Route-level require_role/require_owner_or_admin denials join the
            # auth-middleware coarse-gate under the documented `denied` label
            # (routes/audit.py: ok/error/denied/replay) — a `denied`-filtered
            # query must not silently miss a route-level role/owner denial.
            result = "denied"
        elif response.status_code >= 400:
            result = "error"
        else:
            result = "ok"
        await self._record(request, status_code=response.status_code, result=result)
        return response

    async def _record(self, request: Request, *, status_code: int, result: str) -> None:
        """Persist one audit row and publish a redacted `audit.*` event.

        Best-effort: a logging failure must never mask the original response.
        On a replay the route never ran, so `audit_params` is unset and the
        params column is null (intended).
        """
        action, target_type, target_id = derive_action(request.method, request.url.path)
        # A route may override the derived action for a request the path+method
        # rules cannot distinguish (e.g. `POST /deploy?dry_run=true` audits as
        # `deploy.plan`). `derive_action` stays path-only (IdempotencyMiddleware
        # also calls it); the override is applied ONLY here and to the event type.
        override = getattr(request.state, "audit_action", None)
        if override:
            action = override
        # A route may stamp the target it resolved when the path rules leave it
        # null (the resource name is in the body — see `_TARGET_STAMP_ALLOWED`).
        # This bypasses `redact` by design, so the stamp must only ever carry a
        # bare, already-validated resource NAME (never a secret-bearing value).
        # Applied AFTER the action override so the allowlist keys on the FINAL
        # action (incl. the dry-run `deploy.plan`/`deploy.git_plan` overrides).
        stamp = getattr(request.state, "audit_target", None)
        if (
            stamp
            and action in _TARGET_STAMP_ALLOWED
            and (target_id is None or action == "template.deploy")
        ):
            target_type = _TARGET_STAMP_ALLOWED[action]
            target_id = stamp
        principal = current_principal(request)
        params = getattr(request.state, "audit_params", None)
        params_json = json.dumps(params) if params is not None else None
        rid = request_id_of(request)
        idem_key = request.headers.get("Idempotency-Key")

        queries = self._queries()
        if queries is not None:
            try:
                await queries.insert_audit_log(
                    action=action,
                    result=result,
                    principal_id=principal.token_id,
                    principal_role=principal.role.value,
                    target_type=target_type,
                    target_id=target_id,
                    params_redacted=params_json,
                    status_code=status_code,
                    request_id=rid,
                    idempotency_key=idem_key,
                )
            except Exception:
                logger.warning("Failed to record audit row for %s", action, exc_info=True)

        bus = self._event_bus()
        if bus is not None:
            try:
                bus.publish(
                    {
                        "type": f"audit.{action}",
                        "ts": datetime.now(UTC).isoformat(),
                        "action": action,
                        "result": result,
                        "status_code": status_code,
                        "principal_id": principal.token_id,
                        "principal_role": principal.role.value,
                        "target_type": target_type,
                        "target_id": target_id,
                        "params": params,
                        "request_id": rid,
                    }
                )
            except Exception:
                logger.warning("Failed to publish audit event for %s", action, exc_info=True)
