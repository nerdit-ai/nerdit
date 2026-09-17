"""Link nodes and accept cloud updates through daemon-owned custody paths.

Claim, unlink and refresh are admin-only, require Idempotency-Key and share the
link mutation lock. A tunnel principal is submitter and cannot change custody.
Claim loads the Ed25519 identity, exchanges a grant, then stages and commits
config without an intervening await. Audit node IDs, never grants or raw cloud
responses. Stable daemon errors wrap cloud failures without reflecting secrets.

API URL is per-invocation and never persisted. Resolve relay URL before cloud
contact and again at commit, preserving supplied or stored values; never consume
a grant for a node that cannot start. Claim enables linking unless opted out and
reports restart requirements. Existing identity refuses claim with 409; cloud
idempotency supports recovery only after local state is lost.

Unlink immediately drops the live tunnel, clears node ID/slug and disables link,
but retains operator relay URL/key path. Already-unlinked is a 200 no-op.
Refresh reads public hosted metadata under the same lock and commit rules;
changed determines whether restart is needed. Older clouds without metadata
return 409 link.refresh_unsupported.

Console codes, pre-auth keys and device approval share one identity commit tail.
Device flow never holds the lock while waiting for a human. Its authorizing
160-bit device_code stays in process memory; the CLI receives only a grantless
session selector so concurrent terminals cannot poll another flow.
These routes have no license entitlement gate, including when link is disabled.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Request

import nerdit
from nerdit.config.settings import _is_ip_literal, _validate_dns_name
from nerdit.config.store import ConfigError, ConfigStore
from nerdit.core.link.identity import (
    LinkIdentityError,
    load_or_create_identity,
    resolve_key_file,
)
from nerdit.core.link.manager import ENTITLEMENT_MAX_FUTURE_SKEW_S
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import (
    CLOUD_CONTROL_ENTITLEMENT,
    CLOUD_CONTROL_GIT_NUDGE,
    CLOUD_CONTROL_GITHUB_TOKEN,
    require_cloud_principal,
    require_role,
)
from nerdit.daemon.errors import NerditError

# The config-write plumbing is routes/config.py's, not a copy: same store
# handle, same ConfigError → envelope mapping, one source of truth for both.
from nerdit.daemon.routes.config import _as_nerdit_error, _store
from nerdit.daemon.schemas.link import (
    EntitlementPushRequest,
    EntitlementPushView,
    GithubTokenPushRequest,
    GithubTokenPushView,
    GitNudgeRequest,
    GitNudgeView,
    LinkClaimRequest,
    LinkClaimView,
    LinkDevicePollRequest,
    LinkDevicePollView,
    LinkDeviceStartRequest,
    LinkDeviceStartView,
    LinkRefreshRequest,
    LinkRefreshView,
    LinkUnlinkView,
    _is_loopback,
)
from nerdit.db.models import TokenRole
from nerdit.db.queries._base import mark_request_side_effect

logger = logging.getLogger(__name__)

router = APIRouter()

#: Serializes the claim against unlink (and a second claim). Both routes are a
#: read → awaited file/network work → commit sequence over the same identity
#: file and `[link]` section, so without this an unlink overlapping a claim's
#: cloud hop observes "not linked", deletes the key the claim just created, and
#: lets the claim commit a cloud identity whose private key no longer exists —
#: on the next restart the daemon mints a FRESH key and relay authentication
#: fails terminally. Claim and unlink are rare interactive
#: admin operations; a module lock is proportionate, and it also collapses two
#: concurrent claims into claim-then-409 instead of a double cloud exchange.
_LINK_MUTATION_LOCK = asyncio.Lock()

#: What a cloud refusal `code` may look like before it is reflected to the
#: caller as `cloud_code`: the frozen `link_code_*` vocabulary and this
#: module's own `http_<status>` fallback are all lowercase machine tokens. An
#: operator-supplied `--api-url` endpoint is UNTRUSTED — a nonconforming one
#: could echo the submitted link code (or anything else) inside
#: `{"error": {"code": …}}`, and reflecting that verbatim would violate the
#: secret-value response contract. Anything outside this shape degrades to the
#: fallback.
_CLOUD_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

#: How long the daemon waits on the cloud claim hop. Short on purpose: a claim
#: is an interactive operator action, and the CLI is parked on it.
_CLAIM_TIMEOUT_S = 10.0

#: The wall-clock ceiling on the WHOLE exchange. The httpx scalar above is a
#: per-read inactivity budget, not a total one — a byte every nine seconds
#: satisfies it forever — and the exchange runs holding the mutation lock.
_CLAIM_TOTAL_TIMEOUT_S = 30.0

#: The frozen slug shape: a DNS label (slugs become `<slug>.node.<domain>`
#: gateway hostnames cloud-side, so anything wider could not be routed anyway).
#: The slug is also the second half of every hosted app name —
#: `<app>--<slug>.<nodes_base_domain>`, still one label — so the same grammar
#: carries both.
_SLUG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")

#: Operator guidance per cloud refusal code. The cloud's `link_code_*`
#: vocabulary is frozen (`nerdit-cloud` `errors.py`); anything unmapped
#: falls back to `_CLAIM_HINT_FALLBACK`, so a new cloud code degrades to a
#: generic hint rather than to a KeyError.
_CLAIM_HINTS: dict[str, str] = {
    "link_code_invalid": "No code matches — re-check the paste, or generate a new one.",
    "link_code_expired": "The code has expired — generate a new one and claim it promptly.",
    "link_code_consumed": ("Already claimed by another node identity — generate a new code."),
    "link_code_attempts_exceeded": (
        "Too many failed attempts burned this code — generate a new one."
    ),
    # The pre-auth key vocabulary. The cloud collapses EVERY key death
    # — unknown, malformed, revoked, expired, exhausted, race-lost — into one
    # `link_key_invalid` (a no-oracle rule), so the daemon offers one
    # message for all of them: distinguishing deaths the cloud deliberately
    # merged would reopen the oracle client-side, on this machine, for whoever
    # holds a former key.
    "link_key_invalid": (
        "The link key is not recognised (it may be revoked, expired, or exhausted) — "
        "mint a new one in the console."
    ),
    # The credential this daemon presented already names a node owned by a
    # different account. Never a silent transfer: a machine
    # changes hands by being unlinked first.
    "node_already_linked": (
        "This machine's credential already belongs to another account's node — "
        "unlink it there first."
    ),
    "plan_limit_exceeded": (
        "The account is at its node limit — remove a node in the console, then retry."
    ),
    # NOT a plan question, and not transitional.
    # The cloud runs the same account-standing assertion on mint
    # and on approval, so this code arrives here with exactly one
    # meaning: the account is suspended or being deleted (the cloud's own text
    # is "this account is not active"). Linking itself is free, and the
    # node-count refusal is `plan_limit_exceeded` above, so sending the reader
    # to a plan page would be the same false remedy the doctor and CLI copy just
    # dropped. It still says nothing about THIS daemon's runtime: no route,
    # deploy, proxy or model is gated on it.
    "entitlement_required": (
        "This Nerdit account is not active (suspended, or being deleted); "
        "restore it with support, then link again."
    ),
}
_CLAIM_HINT_FALLBACK = "Generate a fresh link code and run 'nerdit link' again."

#: The canonical grouping the cloud mints a `user_code` in — three groups of
#: four Crockford symbols. Pinned so a nonconforming `--api-url` endpoint
#: cannot hand the terminal an arbitrary string to print in the place a human
#: reads a code from. The alphabet is Crockford's (no `I`, `L`, `O` or
#: `U`), restated rather than imported because this tree vendors no cloud code.
_USER_CODE_RE = re.compile(r"[0-9A-HJKMNP-TV-Z]{4}(?:-[0-9A-HJKMNP-TV-Z]{4}){2}")

#: The device code the cloud mints: 32 Crockford symbols, 160 bits. Bounded on
#: the way in because it is stored on the pending slot and re-sent on every
#: poll — a "code" that is really a paragraph would be re-POSTed for ten
#: minutes. The shape is checked, not the exact length, so a cloud that widens
#: the credential does not brick a fielded daemon.
_DEVICE_CODE_MAX_LENGTH = 128

#: Sanity bounds on the two cadence integers the mint advertises. The cloud owns
#: the cadence (that is why it travels rather than being a daemon constant), but
#: a value outside these is not a cadence — it is a nonconforming endpoint
#: parking this terminal on a poll loop that never ends or never sleeps.
_DEVICE_MAX_TTL_S = 3600
_DEVICE_MAX_INTERVAL_S = 300


def _data_dir(request: Request) -> str:
    """The daemon's `data_dir` — not a `[link]` key, so read off settings.

    `data_dir` lives on the boot-time settings snapshot (changing it is
    restart-required anyway), while `key_file` comes from the store's
    *effective* `[link]` section — what the next restart will use (D15).
    """
    return str(request.app.state.settings.data_dir)


def _require_idempotency_key(request: Request, what: str) -> None:
    """In-route `Idempotency-Key` gate (the config-write wording, D3)."""
    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            f"A {what} requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )


def _cloud_code(response: httpx.Response, submitted_code: str) -> str:
    """Read a safe error.code, falling back to http_<status>; never read response text.

    Reject malformed tokens and case-insensitive containment of submitted_code,
    including embedded echoes. Skip containment when no secret was submitted.
    """
    try:
        payload = response.json()
        code = payload["error"]["code"]
    except (KeyError, TypeError, ValueError):
        return f"http_{response.status_code}"
    if not isinstance(code, str) or not _CLOUD_CODE_RE.fullmatch(code):
        return f"http_{response.status_code}"
    if submitted_code and submitted_code.lower() in code:
        return f"http_{response.status_code}"
    return code


def _raw_key(raw: dict, section: str, key: str) -> object:
    """One raw stored value, tolerating a non-dict section."""
    stored = raw.get(section, {})
    return stored.get(key) if isinstance(stored, dict) else None


def _refuse_pending_data_dir(store: ConfigStore, booted_data_dir: str) -> str | None:
    """409 when a staged `[nerdit].data_dir` awaits a restart; else its raw value.

    The RAW stored value, not the effective one: `effective_section` fills
    defaults, and an unset key would then read as `~/.nerdit` and
    false-refuse every daemon booted with a custom data_dir. A claim must not
    straddle the move — the key would enroll under the OLD directory while the
    required restart resolves the NEW one.
    """
    stored_data_dir = _raw_key(store.load_raw(), "nerdit", "data_dir")
    if (
        isinstance(stored_data_dir, str)
        and stored_data_dir
        and Path(stored_data_dir).expanduser() != Path(booted_data_dir).expanduser()
    ):
        raise NerditError(
            409,
            "link.restart_required",
            "A data_dir change is pending — restart the daemon before linking.",
            hint="Run 'nerdit daemon restart', then claim the code.",
        )
    return stored_data_dir if isinstance(stored_data_dir, str) else None


def _refuse_config_moved_mid_claim(
    store: ConfigStore, key_file_snapshot: object, data_dir_snapshot: str | None
) -> None:
    """409 when the key-path config moved during the awaited cloud hop.

    The config routes do not share `_LINK_MUTATION_LOCK`, so another admin
    can re-stage `[link].key_file` or `[nerdit].data_dir` while the claim
    is parked on the cloud — committing would then bind the cloud identity to
    a config whose next restart resolves a DIFFERENT key than the verifier
    just enrolled. Refusing WITHOUT committing keeps recovery real: restoring
    the snapshot paths and re-claiming replays idempotently with the same key.
    """
    raw_now = store.load_raw()
    if (
        _raw_key(raw_now, "link", "key_file") != key_file_snapshot
        or _raw_key(raw_now, "nerdit", "data_dir") != data_dir_snapshot
    ):
        raise NerditError(
            409,
            "link.config_changed",
            "The key-path configuration changed while the claim was in flight.",
            hint=(
                "Restore the previous key_file/data_dir and re-run 'nerdit link' "
                "with the same code (the claim replays), or generate a fresh code."
            ),
        )


def _relay_url_at_commit(store: ConfigStore, relay_url_to_persist: str | None) -> str:
    """Resolve the supplied or current stored relay URL immediately before commit.

    Config writers do not share the link lock and may clear the relay during awaited
    work. Refuse with 422 before writing identity, even with enable=false, so recovery
    can restore the relay and replay the grant. Omission preserves current config;
    there is no inferred default. Return the value response and audit must report.
    """
    try:
        effective = store.effective_section("link")
    except ConfigError as exc:  # pragma: no cover - "link" is a known section
        raise _as_nerdit_error(exc) from exc
    relay_url = relay_url_to_persist or effective.get("relay_url") or ""
    if not relay_url:
        raise NerditError(
            422,
            "link.relay_url_required",
            "No relay endpoint is configured for this daemon.",
            hint="Set [link].relay_url, then link again (nothing was enrolled).",
        )
    return str(relay_url)


@dataclass(frozen=True, slots=True)
class _StagedLink:
    """What the shared staging tail did, for whichever route called it."""

    #: The config-store etag the commit produced; logged truncated, never
    #: returned — a caller reads `restart_keys` to know what moved.
    etag: str
    #: The relay `[link]` holds now, re-resolved at the commit rather than
    #: before the caller's last await (`_relay_url_at_commit`). Callers
    #: report THIS, so a response and an audit row can never name an endpoint
    #: the config no longer has.
    relay_url: str
    #: The WHOLE mutation's restart keys: the `[link]` ones AND the `[mcp]`
    #: one, so automation reading this is not told half the story.
    restart_keys: list[str]
    mcp_http_enabled: bool
    mcp_skipped_reason: str | None


def _commit_link_identity(  # noqa: PLR0913 - the claim's own inline locals, unchanged
    request: Request,
    store: ConfigStore,
    *,
    node_id: str,
    slug: str,
    domain: str | None,
    relay_url_to_persist: str | None,
    enable: bool,
    key_file: Path,
    key_file_snapshot: object,
    data_dir_snapshot: str | None,
) -> _StagedLink:
    """Commit a cloud identity through the shared code, key and device-flow tail.

    Callers hold the link mutation lock. Keep stage and commit synchronous, with no
    await, and recheck pre-exchange config snapshots before writing. This preserves
    one config shape, MCP HTTP enablement, enrolled-key stamp and restart-key set.
    """
    _refuse_config_moved_mid_claim(store, key_file_snapshot, data_dir_snapshot)
    # (D5) The relay is the OTHER piece of config the grant's awaited hops leave
    # exposed, and the one the caller cannot re-check for itself without
    # re-implementing the resolution. Ordered with the key-path guard, before
    # `mark_request_side_effect` below, so a refusal here claims no durable
    # effect and replays cleanly.
    relay_url = _relay_url_at_commit(store, relay_url_to_persist)

    updates: dict[str, object] = {"node_id": node_id, "slug": slug}
    if relay_url_to_persist:
        updates["relay_url"] = relay_url_to_persist
    # Written ONLY when the cloud actually sent one. A cloud that
    # stopped sending the key (a rollback, a misconfigured origin) must not
    # null out a domain an earlier claim or `nerdit link refresh` learned —
    # forgetting it silently would disable every hosted share on the node.
    if domain is not None:
        updates["nodes_base_domain"] = domain
    # Unconditional: `--no-enable` on an enabled-but-unlinked daemon must
    # persist `false`, or the required restart starts the tunnel the
    # operator just declined. `enable=true` writing an
    # already-true value is a no-op diff.
    updates["enabled"] = enable
    # The commit below is a NON-DB durable effect (a TOML write), so the
    # idempotency claim must be pinned ahead of it (the documented
    # `mark_request_side_effect` contract, the /system/backup precedent):
    # a graceful-shutdown cancellation landing between the commit and the
    # response must replay `idempotency_interrupted` on retry, never
    # re-execute into a spurious `link.already_linked`. On the device poll
    # this is also the line that makes "only the commit branch claims a durable
    # side effect" true: a pending poll never reaches it.
    mark_request_side_effect()
    try:
        # (D12) stage + commit are CONTIGUOUS — no await between them — so
        # this read-modify-write cannot interleave with another config
        # writer on the single event loop. A ConfigError here is
        # near-unreachable: relay_url was validated by the request schema
        # before the cloud was contacted.
        staged = store.stage("link", updates)
        etag = store.commit(staged)
    except ConfigError as exc:
        raise _as_nerdit_error(exc) from exc
    # The path whose key was actually enrolled, for unlink's wipe: the
    # effective `key_file` can be re-staged through the config API at any
    # time, so at unlink it may point somewhere this identity never lived
    # (rather than the raw config value).
    request.app.state.link_enrolled_key_file = str(key_file)
    # A linked node exists for the cloud's remote MCP gateway, which
    # forwards to this daemon's `/api/mcp` — so a claim that enables the
    # link also enables the transport (a deliberate owner decision, made
    # after a fresh install was found answering 405 there otherwise).
    # Two preconditions the daemon itself enforces at boot, checked here so
    # the required restart can never be turned into a boot refusal:
    # `[daemon].auth_token` (the transport is bearer-only) and the
    # bundled `mcp` extra. Either missing → skipped, with the reason.
    mcp_http_enabled, mcp_skipped, mcp_keys = _enable_mcp_transport(store, enable=enable)
    return _StagedLink(
        etag=etag,
        relay_url=relay_url,
        restart_keys=[*staged.restart_keys, *mcp_keys],
        mcp_http_enabled=mcp_http_enabled,
        mcp_skipped_reason=mcp_skipped,
    )


@router.post(
    "/link/claim",
    response_model=LinkClaimView,
    operation_id="claim_link",
    status_code=200,
)
async def claim_link(request: Request, body: LinkClaimRequest) -> LinkClaimView:
    """Exchange exactly one console code or pre-auth key and persist the node identity.

    Admin-only with Idempotency-Key; tunnel submitters cannot change node custody.
    Both grants share locking, preflight and commit rules. Secrets travel in JSON
    bodies only and never appear in audits, logs, responses or persisted files.
    """
    require_role(request, TokenRole.admin)
    # Refused before the admin's audit params are built and before anything
    # reaches the config store: a body that names two grants, or none, is not a
    # claim this daemon can reason about. In-route rather than a model
    # validator (the `/events` `cursor`/`since_id` precedent) so the
    # answer is this daemon's structured envelope — a pydantic `value_error`
    # would echo the model's own field values back at the caller, and one of
    # those fields is a secret.
    if (body.code is None) == (body.key is None):
        raise NerditError(
            422,
            "validation_error",
            "A link claim carries exactly one of 'code' or 'key'.",
            hint="Paste a console link code, or supply a pre-auth key — not both.",
        )
    # Recorded up front so a failed claim is still audited with what it was
    # asked to do. The code is NEVER a member — that is the primary defense; the
    # `"code"` entry in the audit denylist is only depth.
    # The HOST only, never the full URL: operators embed bearer tokens in URL
    # *paths* (the webhook-URL practice the notification dispatcher already
    # refuses to audit), and this row is durable. `relay_url`
    # stays whole — its validator refuses userinfo and query strings, so it is
    # credential-free by construction, and the whole value IS the persisted
    # config the row exists to record.
    request.state.audit_params = audit_params(
        {
            "api_host": urlsplit(body.api_url).netloc,
            "relay_host": None if body.relay_url is None else urlsplit(body.relay_url).netloc,
            "enable": body.enable,
            # WHICH grant was presented, never the grant itself. The
            # row is durable and an operator reading it months later needs to
            # know whether a node arrived by console paste or by an installer's
            # pre-auth key; "code"/"key" is a two-value enum, not a secret.
            "grant": "key" if body.key is not None else "code",
        }
    )
    _require_idempotency_key(request, "link claim")

    async with _LINK_MUTATION_LOCK:
        store = _store(request)
        try:
            effective = store.effective_section("link")
        except ConfigError as exc:  # pragma: no cover - "link" is a known section
            raise _as_nerdit_error(exc) from exc

        # (D8) Refused BEFORE any cloud contact: the cloud's replay idempotency
        # is for a daemon that lost its state, not for one that still holds an
        # identity and pastes a fresh code.
        linked_as = effective.get("node_id")
        if linked_as:
            raise NerditError(
                409,
                "link.already_linked",
                f"This daemon is already linked as node {linked_as}.",
                hint="Run 'nerdit unlink' first if you mean to re-link.",
            )

        # (D5) The relay endpoint must be known before the code is spent: a
        # successful claim that cannot be started is worse than a refused one.
        # A fast-fail and nothing more, in `start_device_link`'s shape: the
        # value the response and the audit report is re-resolved at the commit
        # (`_relay_url_at_commit`), because the identity load and the
        # cloud exchange below are awaits an admin can clear `[link].relay_url`
        # inside, and a snapshot taken here would outlive its own truth.
        if not (body.relay_url or effective.get("relay_url")):
            raise NerditError(
                422,
                "link.relay_url_required",
                "No relay endpoint is configured for this daemon.",
                hint="Pass --relay-url wss://… (persisted with the claim).",
            )

        # A staged-but-not-restarted `[nerdit].data_dir` would make the claim
        # enroll a key under the OLD directory while the required restart
        # resolves the NEW one — a different key, a different verifier, and a
        # terminal relay-auth failure. Refuse until the
        # pending move is applied; a claim's entire product is for the next
        # restart, so it must not straddle one.
        stored_data_dir = _refuse_pending_data_dir(store, _data_dir(request))

        key_file = resolve_key_file(effective.get("key_file"), _data_dir(request))
        try:
            # Synchronous by design (the `load_or_create_identity` docstring),
            # so it runs off the loop.
            identity = await asyncio.to_thread(load_or_create_identity, key_file)
        except LinkIdentityError as exc:
            # The path belongs in the log (the identity.py precedent) and
            # nowhere else: the response message stays path-free (P14c M3
            # posture).
            logger.error("Node identity unavailable for link claim: %s", exc)
            raise NerditError(
                500,
                "link.identity_unavailable",
                "The node identity key could not be loaded or created.",
                hint="See the daemon log for the failing path, then fix or remove the key file.",
            ) from exc

        node_id, slug, domain = await _exchange(request, body, identity.verifier)

        # The one staging tail, shared verbatim with the device poll's
        # approved branch so the two grants cannot drift apart. Everything the
        # claim used to do inline lives in it, in the same order.
        committed = _commit_link_identity(
            request,
            store,
            node_id=node_id,
            slug=slug,
            domain=domain,
            relay_url_to_persist=body.relay_url,
            enable=bool(body.enable),
            key_file=key_file,
            key_file_snapshot=effective.get("key_file"),
            data_dir_snapshot=stored_data_dir,
        )
    logger.info("Linked as node %s (config etag %s)", node_id, committed.etag[:12])

    request.state.audit_target = node_id
    request.state.audit_params = audit_params(
        {
            "node_id": node_id,
            "slug": slug,
            # The identity.py-documented non-secret audit form of the key.
            "verifier_fingerprint": identity.fingerprint,
            # The HOST only (the api_host rule): an ACCEPTED relay URL may
            # legitimately carry a path (a relay behind a prefix), and a path
            # is where credentials get parked — the durable row does not need
            # it, the config file already holds the full value.
            "relay_host": urlsplit(committed.relay_url).netloc,
            "enabled": bool(body.enable),
            # A public DNS name, not a secret — and the row is where
            # an operator later reconstructs which suffix a node was claimed
            # under. `None` when this cloud sent none.
            "nodes_base_domain": domain,
            "mcp_http_enabled": committed.mcp_http_enabled,
            # Carried onto the success row too, not just the pre-flight
            # one it replaces: with three ways into a link, "how did this node
            # arrive" is a question the durable trail should be able to answer
            # without joining anything. The device poll's success row (which
            # audits as `link.created` through the same tail) writes
            # `"device"` in this slot, so the three are one vocabulary.
            "grant": "key" if body.key is not None else "code",
        }
    )
    return LinkClaimView(
        node_id=node_id,
        slug=slug,
        # What `[link]` holds now, for the `nodes_base_domain` reason one
        # field down: a value resolved before the exchange could name a relay
        # the config lost while the exchange was in flight.
        relay_url=committed.relay_url,
        enabled=bool(body.enable),
        verifier_fingerprint=identity.fingerprint,
        # What the daemon now HOLDS, not merely what this exchange returned: a
        # cloud that sent nothing leaves an earlier refresh's value standing,
        # and reporting `null` there would read as "the node lost it".
        nodes_base_domain=domain if domain is not None else effective.get("nodes_base_domain"),
        # Hardcoded, not read off the staged diff: every `[link]` key is
        # restart-keyed (config/store.py), so a claim ALWAYS needs a restart to
        # take effect — including the degenerate "same values" re-commit whose
        # diff would be empty. `restart_keys` still reports what changed.
        requires_restart=True,
        # The whole mutation: the [link] keys AND the [mcp] key this claim
        # staged, so automation reading restart_keys is not told half the story.
        restart_keys=committed.restart_keys,
        mcp_http_enabled=committed.mcp_http_enabled,
        mcp_skipped_reason=committed.mcp_skipped_reason,
    )


#: Skip reasons are machine-stable prefixes the CLI keys its remediation on
#: — suggesting `http_enabled=true` while the precondition is
#: missing would hand the operator the exact config `build_app` refuses to
#: boot. Human tail after the colon; never a token.
MCP_SKIP_NO_TOKEN = "no_auth_token"
MCP_SKIP_NO_EXTRA = "mcp_extra_missing"
MCP_SKIP_LINK_DISABLED = "link_not_enabled"


def _enable_mcp_transport(
    store: ConfigStore, *, enable: bool
) -> tuple[bool, str | None, list[str]]:
    """Stage `[mcp].http_enabled = true` for the post-claim restart.

    Returns `(enabled, skipped_reason, restart_keys)` — the keys so the
    claim's own `restart_keys` describes the whole mutation, not just the
    `[link]` half. Never raises past a ConfigError: the link itself is
    already committed, and a transport that could not be switched on is
    reported, not fatal. Already-true is reported as enabled with no keys.
    """
    if not enable:
        return False, f"{MCP_SKIP_LINK_DISABLED}: link not enabled", []
    try:
        if store.effective_section("mcp").get("http_enabled") is True:
            return True, None, []
        if not store.effective_section("daemon").get("auth_token"):
            return (
                False,
                f"{MCP_SKIP_NO_TOKEN}: no [daemon].auth_token — the transport is bearer-only",
                [],
            )
    except ConfigError:  # pragma: no cover - known sections
        return False, "config_unreadable: config unreadable", []
    if importlib.util.find_spec("mcp") is None:
        return False, f"{MCP_SKIP_NO_EXTRA}: the mcp extra is not installed in this build", []
    try:
        staged = store.stage("mcp", {"http_enabled": True})
        store.commit(staged)
    except ConfigError as exc:
        return False, f"config_error: could not stage [mcp].http_enabled: {exc}", []
    return True, None, list(staged.restart_keys)


def _protocol_error(hint: str) -> NerditError:
    """The one shape for "the cloud answered 200 with something else" (D11)."""
    return NerditError(
        502,
        "link.cloud_protocol",
        "The cloud's response is not what this endpoint promises.",
        hint=hint,
    )


def _validated_domain(value: object, *, submitted_code: str = "") -> str | None:
    """Validate a hosted suffix against LinkSettings, returning None when absent.

    Also reject wildcards and IP literals, which cannot form hosted app names.
    Reject reflection of submitted_code before config persistence; empty means no
    secret was sent, as on refresh.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise _protocol_error("The endpoint answered 200 without a hosted base domain.")
    if "*" in value:
        raise _protocol_error("The hosted base domain must be a concrete DNS name, not a wildcard.")
    if _is_ip_literal(value):
        raise _protocol_error("The hosted base domain must be a DNS name, not an IP address.")
    try:
        _validate_dns_name(value, field_name="nodes_base_domain")
    except ValueError as exc:
        # The VALUE is not echoed: it is attacker-influenced input on an
        # operator-supplied endpoint, and the caller already knows what it asked.
        raise _protocol_error(
            "The endpoint answered 200 with a hosted base domain that is not a DNS name."
        ) from exc
    if submitted_code and submitted_code.lower() in value.lower():
        raise _protocol_error("The endpoint answered 200 without a hosted base domain.")
    return value


async def _exchange(
    request: Request, body: LinkClaimRequest, verifier: str
) -> tuple[str, str, str | None]:
    """Exchange a grant and verifier for node_id, slug and hosted domain.

    Send keys to /api/link/preauth and codes to /api/link/claim. Both share bounded
    timeouts, refusal mapping and reflection checks. Include hostname_hint only for
    pre-auth. Errors may name the supplied netloc, never the full URL or response
    text. The caller has already enforced exactly one grant.
    """
    # The grant, whichever it is: the value every anti-reflection guard below
    # measures the cloud's answer against. Never logged, never audited, never
    # interpolated into an error message.
    submitted = body.key or body.code or ""
    if body.key is not None:
        url = body.api_url.rstrip("/") + "/api/link/preauth"
        hostname, _version, _os = _machine_facts()
        payload_out: dict[str, object] = {
            "key": body.key,
            "credential_reference": verifier,
            "hostname_hint": hostname,
        }
    else:
        url = body.api_url.rstrip("/") + "/api/link/claim"
        payload_out = {"code": body.code, "credential_reference": verifier}
    # `transport=None` is the real network; tests inject an
    # `httpx.MockTransport` through app.state so no suite ever dials out.
    transport = getattr(request.app.state, "link_claim_transport", None)
    try:
        # The scalar httpx timeout is per-operation (a read-inactivity budget),
        # so a slow-dripping endpoint could hold the exchange — and with it
        # `_LINK_MUTATION_LOCK` — indefinitely, wedging every later unlink.
        # The asyncio deadline is the WALL-CLOCK
        # bound on the whole hop.
        async with asyncio.timeout(_CLAIM_TOTAL_TIMEOUT_S):
            async with httpx.AsyncClient(
                timeout=_CLAIM_TIMEOUT_S, follow_redirects=False, transport=transport
            ) as http:
                response = await http.post(url, json=payload_out)
    except (httpx.HTTPError, TimeoutError) as exc:
        # The netloc only — never the full URL (it could carry an operator's
        # paste), never the code, never a response body.
        raise NerditError(
            502,
            "link.cloud_unreachable",
            f"The cloud endpoint {urlsplit(body.api_url).netloc} could not be reached.",
            hint="Check --api-url and this machine's outbound connectivity, then retry.",
        ) from exc

    if response.status_code >= 400:
        raise _refusal(response, submitted)

    return _parse_claim_identity(response, submitted)


def _parse_claim_identity(
    response: httpx.Response, submitted_secret: str
) -> tuple[str, str, str | None]:
    """Read one `LinkClaimResponse` and prove it is not a reflection (D11).

    Shared by all three grants because the cloud answers all three with the
    **same response class** (D-X16-O7): the claim, the pre-auth enroll and the
    approved device poll are byte-identical on the wire, so a daemon that
    parses one parses all of them, and a second copy of this function would be
    a second place for the reflection guards to rot.

    `submitted_secret` is whatever this exchange sent — a link code, a
    pre-auth key, or a device code. It is never echoed; it is only ever
    measured *against* the answer.
    """
    try:
        payload = response.json()
        node_id = payload["node_id"]
        slug = payload["node_slug"]
    except (KeyError, TypeError, ValueError) as exc:
        raise NerditError(
            502,
            "link.cloud_protocol",
            "The cloud's claim response is not a node identity.",
            hint="The endpoint answered 200 with an unexpected shape — check --api-url.",
        ) from exc
    # (D11) The keys are read RAW and validated against the FROZEN identity
    # shapes rather than merely null-checked: the cloud contract returns a UUID
    # `node_id` and a DNS-label slug (slugs become `<slug>.node.<domain>`
    # hostnames). A nonconforming `--api-url` endpoint answering 200 with the
    # submitted link code in either field would otherwise see that secret
    # persisted into config, echoed in the response, audited, and logged — the
    # success path is a reflection channel exactly like `error.code` was.
    # A link code cannot pass either shape.
    if (
        not isinstance(node_id, str)
        or not isinstance(slug, str)
        or not _is_uuid(node_id)
        or not _SLUG_RE.fullmatch(slug)
        # The schema pins codes UPPERCASE while every reflectable shape here
        # is lowercase-only, so a conforming echo of the code as-sent is
        # impossible. The LOWERCASED form is still secret-equivalent (Crockford
        # is case-insensitive, uppercasing is trivial), so it must not appear
        # in — not merely equal — the slug, and the node id must be
        # a CANONICAL lowercase UUID that equals no case-folding of the code.
        or node_id != node_id.lower()
        # Containment on BOTH fields: a hex-only code can embed inside a
        # canonical UUID's hex just as a hyphenated one can inside a slug.
        # Real issued codes contain non-hex letters, so a false
        # positive needs a pathological code AND a 1-in-10^6 collision.
        #
        # The disjointness argument above holds a fortiori for a
        # pre-auth key, whose `nk_` prefix carries an underscore that neither
        # a slug nor a canonical UUID may contain. It does NOT hold for a
        # `device_code`: 32 Crockford symbols lowercased are a perfectly legal
        # DNS label, so on the device path this containment test is not depth —
        # it is the guard, and it is why the poll passes the daemon-held code
        # in here rather than an empty string.
        or submitted_secret.lower() in node_id
        or submitted_secret.lower() in slug
    ):
        raise NerditError(
            502,
            "link.cloud_protocol",
            "The cloud's claim response is not a node identity.",
            hint="The endpoint answered 200 without a UUID node id and a slug.",
        )
    # Additive and OPTIONAL: an older cloud omits the key entirely
    # and the claim proceeds, leaving the domain to `nerdit link refresh`.
    # Present-but-malformed is a protocol error, not a silent drop — a claim
    # that persisted a bad suffix would advertise URLs nothing resolves.
    domain = _validated_domain(payload.get("nodes_base_domain"), submitted_code=submitted_secret)
    return node_id, slug, domain


def _is_uuid(value: str) -> bool:
    """Whether `value` IS the frozen contract's canonical UUID `node_id`.

    Canonical means the string equals its own parsed round-trip — lowercase,
    hyphenated, 36 chars. `uuid.UUID` alone accepts braces, URNs, bare hex
    and uppercase, which would widen the accepted space for no contract reason.
    """
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _refusal(response: httpx.Response, submitted_code: str) -> NerditError:
    """Map a cloud non-2xx onto a stable daemon code (D11).

    The cloud's machine token travels as `cloud_code` so an operator (or an
    agent) can act on the real reason; the cloud's *body* never does.
    """
    code = _cloud_code(response, submitted_code)
    status = response.status_code
    if status == 429:
        return NerditError(
            429,
            "link.claim_rate_limited",
            "The cloud is rate-limiting link claims.",
            hint="Wait a moment and run 'nerdit link' again.",
            cloud_code=code,
            cloud_status=status,
        )
    if status < 500:
        return NerditError(
            409,
            "link.claim_refused",
            "The cloud refused the link code.",
            hint=_CLAIM_HINTS.get(code, _CLAIM_HINT_FALLBACK),
            cloud_code=code,
            cloud_status=status,
        )
    return NerditError(
        502,
        "link.cloud_error",
        "The cloud could not answer the link claim.",
        hint="This is a cloud-side failure — retry shortly with the same code.",
        cloud_code=code,
        cloud_status=status,
    )


async def _wipe_node_keys(
    request: Request, effective: dict, was_linked: bool, had_live_manager: bool
) -> tuple[bool, int]:
    """Wipe the enrolled node key (+ any pending failed wipes); report the primary.

    Failed wipes are tracked in a PENDING SET on `app.state`, separate from
    the enrollment stamp: the stamp is overwritten by a subsequent claim, so a
    failed wipe of key A followed by a re-stage + re-link to key B would
    otherwise lose A's only retry marker. Every unlink drains the
    set best-effort; a never-linked no-op unlink with an empty set touches
    nothing (protecting a pre-provisioned key from an accidental delete).
    """
    enrolled = getattr(request.app.state, "link_enrolled_key_file", None)
    pending = set(getattr(request.app.state, "link_pending_key_wipes", ()))
    # `had_live_manager` joins the gate: a staged `node_id =
    # null` through the config API empties `was_linked` while the booted
    # manager still proves an enrolled identity is live — its key must go.
    linked_evidence = was_linked or had_live_manager or enrolled is not None
    if not (linked_evidence or pending):
        return False, 0
    if enrolled is not None:
        key_file = Path(enrolled)
    else:
        key_file = resolve_key_file(request.app.state.settings.link.key_file, _data_dir(request))
    targets = set(pending)
    if linked_evidence:
        targets.add(str(key_file))
        staged_path = resolve_key_file(effective.get("key_file"), _data_dir(request))
        if staged_path != key_file:
            logger.info(
                "Unlink wipes the enrolled key %s; the staged key_file %s is untouched.",
                key_file,
                staged_path,
            )

    def _wipe_all() -> tuple[bool, set[str]]:
        primary_removed = False
        still_failed: set[str] = set()
        for raw in targets:
            path = Path(raw)
            existed = path.exists()
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # A path is fine in the log (the identity.py precedent); it
                # never reaches the response or the audit row. `exc` rather
                # than `exc.strerror`: not every OSError carries a strerror,
                # and "… key node.key: None" points nowhere.
                logger.warning("Could not remove node key %s: %s", path, exc)
                still_failed.add(raw)
                continue
            if raw == str(key_file) and existed and not path.exists():
                primary_removed = True
        return primary_removed, still_failed

    key_removed, still_failed = await asyncio.to_thread(_wipe_all)
    request.app.state.link_pending_key_wipes = still_failed
    request.app.state.link_enrolled_key_file = None
    # The COUNT only, never paths: the caller reports it so "key deleted"
    # cannot conceal an older credential still stuck on disk. It is
    # in-memory like the set itself — the cross-restart persistence of pending
    # wipes is the recorded backlog residual.
    return key_removed, len(still_failed)


@router.delete(
    "/link",
    response_model=LinkUnlinkView,
    operation_id="unlink_node",
    status_code=200,
)
async def unlink_node(request: Request) -> LinkUnlinkView:
    """Drop the tunnel, clear the claim seam, and wipe the node key.

    Admin-only for the same custody reason as the claim (D2). Unlinking while
    already unlinked is an **idempotent 200 no-op** (D9) — retries and cleanup
    scripts must be safe — reported as `was_linked: false`.
    """
    require_role(request, TokenRole.admin)
    _require_idempotency_key(request, "link revocation")

    async with _LINK_MUTATION_LOCK:
        store = _store(request)
        try:
            effective = store.effective_section("link")
        except ConfigError as exc:  # pragma: no cover - "link" is a known section
            raise _as_nerdit_error(exc) from exc
        old_node_id = effective.get("node_id")
        was_linked = old_node_id is not None

        # The ONE hot action in the whole route (D7). Every `[link]` key is
        # restart-required, but a revoked node must not keep a live tunnel up
        # until the operator gets around to restarting — that is a security
        # hole, not a semantics nicety.
        manager = getattr(request.app.state, "link_manager", None)
        had_live_manager = manager is not None
        tunnel_stopped = False
        if manager is not None:
            await manager.stop()
            request.app.state.link_manager = None
            tunnel_stopped = True

        # An in-flight device request dies with the link it was going to create.
        # Unconditionally, and NOT through `_clear_pending`: that helper is
        # identity-guarded so a terminal answer settles only its own flow, which
        # is right for a poll and wrong here — unlink is not one flow ending, it
        # is the daemon's whole enrolment being torn down, so every slot goes.
        #
        # Without this the wipe below removes the key file the slot still points
        # at while every poll pre-flight keeps passing: `node_id` was just
        # cleared, the slot and its session are untouched, and the cloud happily
        # confirms the approval for the verifier minted from the DELETED key. The
        # commit would then write `node_id`/`slug`/`enabled = true` naming a
        # credential that no longer exists, the next boot would mint a fresh
        # keypair, and the tunnel could never authenticate as the enrolled node —
        # "linked but unauthenticatable" until somebody unlinks again. Cheap to
        # prevent here; invisible and baffling if it happens.
        request.app.state.link_device_pending = None

        # Non-DB durable effects follow (the config TOML write, the key wipe) —
        # pin the idempotency claim first, the documented
        # `mark_request_side_effect` contract.
        mark_request_side_effect()
        try:
            # Contiguous stage+commit (D12), after the awaited stop. The
            # null-delete is the store's documented single explicit unlink
            # path: it CLEARS node_id/slug. `relay_url`/`key_file` are
            # deliberately KEPT (D10) — operator-supplied endpoint/path config,
            # not identity, and a re-link reuses them. `enabled` goes back to
            # the ship-dark default so the daemon stops warning "enabled but
            # not linked" on every boot.
            # `nodes_base_domain` goes with the identity, not the endpoint
            # config: it is a CLAIM result (the suffix is only meaningful
            # beside a slug — the refresh rule), and a re-claim against a
            # different cloud whose claim omits the key must not inherit the
            # previous cloud's suffix and advertise hosted URLs that cannot
            # route through the new link. The claim's
            # keep-when-absent guard stays: it protects a LIVE identity from a
            # cloud rollback, not a fresh identity from a stale one.
            staged = store.stage(
                "link",
                {"node_id": None, "slug": None, "nodes_base_domain": None, "enabled": False},
            )
            store.commit(staged)
        except ConfigError as exc:  # pragma: no cover - a null-delete cannot fail
            raise _as_nerdit_error(exc) from exc

        # The second half of the D7 hot action. `app.state.settings` is the
        # BOOT snapshot and nothing re-reads it, so a read path that projects
        # the hosted identity off it (`views/hosted.py`) would keep composing
        # `https://<app>--<old-slug>.<old-domain>/` for the rest of the
        # process — advertising an address that belongs to a node the cloud has
        # just revoked. The tunnel is dropped immediately, so the in-process
        # identity it served goes with it: mirror exactly the keys the commit
        # cleared, nothing else. Not a config re-read — `effective_section`
        # is a TOML read, and the share/list hot path is held to one batched
        # query — and not a change to `addressable`, which must
        # still compose a URL for a claimed-but-not-yet-restarted node.
        live_settings = getattr(request.app.state, "settings", None)
        live_link = getattr(live_settings, "link", None) if live_settings is not None else None
        if live_link is not None:
            live_link.node_id = None
            live_link.slug = None
            live_link.nodes_base_domain = None
            live_link.enabled = False

        # AFTER the commit, best effort: a crash between the two leaves unlinked
        # config plus an orphan key, which the next claim's load_or_create simply
        # re-loads and re-enrols. The reverse order would leave a linked config with
        # no key — a daemon that cannot connect and cannot say why.
        #
        # WHICH key: the one whose verifier was actually enrolled — the path the
        # claim in this process stamped, else the path the running daemon loaded at
        # boot. The *effective* `key_file` is deliberately NOT used here: it can
        # be re-staged through the config API at any time without a restart, so it
        # may point at a pre-provisioned key this identity never lived in — wiping
        # that would destroy operator property AND leave the enrolled private key
        # on disk. A staged-but-not-yet-booted path divergence is
        # logged and left alone.
        # The wipe runs ONLY for a daemon that was actually linked: on a
        # never-linked daemon nothing was ever enrolled, so the boot-time path
        # may name a pre-provisioned operator key that an idempotent-no-op
        # unlink must not destroy (the case of `node_id` unset with a custom
        # `key_file`).
        key_removed, pending_wipes = await _wipe_node_keys(
            request, effective, was_linked, had_live_manager
        )

    if old_node_id:
        request.state.audit_target = old_node_id
    request.state.audit_params = audit_params(
        {
            "was_linked": was_linked,
            "tunnel_stopped": tunnel_stopped,
            "key_removed": key_removed,
            "pending_wipes": pending_wipes,
        }
    )
    return LinkUnlinkView(
        was_linked=was_linked,
        tunnel_stopped=tunnel_stopped,
        key_removed=key_removed,
        pending_wipes=pending_wipes,
        # The config/key wipe needs no restart to be *effective* (the manager is
        # already gone); the flag bookkeeping does, so the answer stays honest.
        requires_restart=True,
    )


async def _fetch_metadata(request: Request, body: LinkRefreshRequest) -> str:
    """GET the cloud's hosted metadata and return its `nodes_base_domain`.

    The claim hop's exact plumbing — same client, same two nested deadlines
    (the httpx scalar is a per-read inactivity budget; the `asyncio` one is
    the wall clock, and this hop also runs holding `_LINK_MUTATION_LOCK`), same
    `link_claim_transport` test seam — with one difference that shapes
    every error below: **nothing secret is sent**. There is no code to leak, so
    the refusals need only be structured, not reflection-proof.

    A `404` is an older cloud without this endpoint and gets its own code
    (`link.refresh_unsupported`): an operator whose console is simply older
    than their daemon must be told to upgrade, not handed a generic 502.
    """
    url = body.api_url.rstrip("/") + "/api/link/metadata"
    transport = getattr(request.app.state, "link_claim_transport", None)
    try:
        async with asyncio.timeout(_CLAIM_TOTAL_TIMEOUT_S):
            async with httpx.AsyncClient(
                timeout=_CLAIM_TIMEOUT_S, follow_redirects=False, transport=transport
            ) as http:
                response = await http.get(url)
    except (httpx.HTTPError, TimeoutError) as exc:
        raise NerditError(
            502,
            "link.cloud_unreachable",
            f"The cloud endpoint {urlsplit(body.api_url).netloc} could not be reached.",
            hint="Check --api-url and this machine's outbound connectivity, then retry.",
        ) from exc

    if response.status_code == 404:
        raise NerditError(
            409,
            "link.refresh_unsupported",
            "The cloud does not publish hosted metadata yet.",
            hint=(
                "Upgrade the cloud, or set [link].nodes_base_domain through "
                "PUT /config/daemon/link and restart."
            ),
        )
    if response.status_code >= 400:
        raise NerditError(
            502,
            "link.cloud_error",
            "The cloud could not answer the hosted-metadata read.",
            hint="This is a cloud-side failure — retry shortly.",
            # No submitted secret on this hop, so the cloud's machine token
            # travels as-is (the empty `submitted_code` skips the claim's
            # containment guard); the cloud's BODY still never does.
            cloud_code=_cloud_code(response, ""),
            cloud_status=response.status_code,
        )

    try:
        payload = response.json()
        raw = payload["nodes_base_domain"]
    except (KeyError, TypeError, ValueError) as exc:
        raise _protocol_error("The endpoint answered 200 without a hosted base domain.") from exc
    domain = _validated_domain(raw)
    if domain is None:
        raise _protocol_error("The endpoint answered 200 without a hosted base domain.")
    return domain


@router.post(
    "/link/refresh",
    response_model=LinkRefreshView,
    operation_id="refresh_link",
    status_code=200,
)
async def refresh_link(request: Request, body: LinkRefreshRequest) -> LinkRefreshView:
    """Re-read the cloud's hosted base domain into `[link].nodes_base_domain`.

    Admin-only for the same custody reason as the claim (D2) — it commits
    `[link]` config — and idempotent by construction: the cloud read is a
    plain GET of a static value, so running it twice writes the same key twice
    and reports `changed: false` the second time.

    Refusing on an UNLINKED node (409 `link.not_linked`) rather than writing
    the domain anyway: the suffix is only meaningful beside a slug, and a node
    that stores one without an identity would advertise nothing while looking
    configured.
    """
    require_role(request, TokenRole.admin)
    # The HOST only, never the full URL — the api_host rule the claim documents.
    request.state.audit_params = audit_params({"api_host": urlsplit(body.api_url).netloc})
    _require_idempotency_key(request, "link refresh")

    async with _LINK_MUTATION_LOCK:
        store = _store(request)
        try:
            effective = store.effective_section("link")
        except ConfigError as exc:  # pragma: no cover - "link" is a known section
            raise _as_nerdit_error(exc) from exc

        node_id = effective.get("node_id")
        if not node_id:
            raise NerditError(
                409,
                "link.not_linked",
                "This daemon is not linked to a cloud account.",
                hint="Run 'nerdit link <code>' first — the hosted domain comes with the claim.",
            )

        domain = await _fetch_metadata(request, body)
        changed = effective.get("nodes_base_domain") != domain
        # A TOML write is a non-DB durable effect, so the idempotency claim is
        # pinned ahead of it (the claim/unlink precedent): a cancellation between
        # the commit and the response must replay `idempotency_interrupted`.
        mark_request_side_effect()
        try:
            # (D12) Contiguous stage+commit — no await between them. Committing
            # even when nothing moved keeps this one code path; the store's diff
            # makes an unchanged write a no-op, so `restart_keys` is empty and
            # the caller reads `changed: false`.
            staged = store.stage("link", {"nodes_base_domain": domain})
            store.commit(staged)
        except ConfigError as exc:
            raise _as_nerdit_error(exc) from exc

    slug = effective.get("slug") or ""
    logger.info("Hosted base domain refreshed (changed=%s)", changed)

    request.state.audit_target = node_id
    request.state.audit_params = audit_params(
        {
            "api_host": urlsplit(body.api_url).netloc,
            "node_id": node_id,
            "nodes_base_domain": domain,
            "changed": changed,
        }
    )
    return LinkRefreshView(
        node_id=node_id,
        slug=slug,
        nodes_base_domain=domain,
        changed=changed,
        # Honest, not hardcoded-true like the claim's: every `[link]` key is
        # restart-keyed, but a refresh that rewrote the SAME value changed
        # nothing a restart could apply, and bouncing a healthy daemon to prove
        # that is exactly the friction D7 exists to avoid.
        requires_restart=changed,
        restart_keys=staged.restart_keys,
    )


# ---------------------------------------------------------------------------
# The device-code flow
# ---------------------------------------------------------------------------


def _machine_facts() -> tuple[str | None, str, str]:
    """Return bounded daemon-local hostname, version and OS for CLI display.

    The CLI may be remote and the cloud does not echo these facts. Omit an unreadable
    hostname rather than failing linking; truncate to cloud column widths.
    """
    try:
        hostname: str | None = socket.gethostname().strip()[:120] or None
    except OSError:  # pragma: no cover - a hostname-less host
        hostname = None
    return hostname, nerdit.__version__[:64], sys.platform[:64]


#: Told to an operator whose on-disk credential no longer matches the one the
#: cloud enrolled. Always actionable and never diagnostic: which of the several
#: ways it can happen (an unlink mid-flow, a restored backup, a swapped
#: identity) is not something the daemon can distinguish, and guessing in a
#: hint is worse than naming the one command that always works.
_CREDENTIAL_CHANGED_HINT = (
    "Run 'nerdit link --device' again to mint a request for the current credential."
)


@dataclass(slots=True)
class _DevicePending:
    """One device-link session held only in daemon memory.

    Restart abandons it; the cloud row expires after 900 seconds. A new start replaces
    the session selector so earlier terminals fail rather than poll the new grant.
    Never return, log, audit, persist or put device_code on argv.
    """

    #: The opaque selector handed to the CLI. Non-secret: it names the slot, it
    #: does not open it.
    session: str
    #: The 160-bit poll credential. NEVER leaves this process.
    device_code: str
    #: The verifier pinned at mint — re-sent on every poll because a
    #: constant-time comparison against it is the first thing the cloud does.
    verifier: str
    verifier_fingerprint: str
    key_file: Path
    #: The two config snapshots `_refuse_config_moved_mid_claim` compares
    #: at commit. Taken at the start, because the window they guard is the whole
    #: human-paced approval, not one bounded HTTP hop.
    key_file_snapshot: object
    data_dir_snapshot: str | None
    api_url: str
    relay_url: str | None
    enable: bool
    #: `time.monotonic` deadline, not a wall clock: a clock step during an
    #: attended approval must not expire a live request or resurrect a dead one.
    expires_at: float
    interval: int

    def __repr__(self) -> str:
        """Redacted by construction — the `NodeIdentity` idiom.

        A dataclass' generated `__repr__` would render `device_code` into
        any log record, traceback frame or debugger line that touched the slot.
        """
        return f"_DevicePending(session={self.session!r}, expires_at={self.expires_at!r})"


@dataclass(frozen=True, slots=True)
class _DeviceMint:
    """One cloud `201 DeviceLinkResponse`, shape-checked."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    credential_fingerprint: str
    expires_in: int
    interval: int

    def __repr__(self) -> str:
        """Redacted for the reason `_DevicePending` is, and no other.

        This one holds BOTH codes, and it is the shorter-lived of the two — so
        the generated `__repr__` was easy to leave alone and would have been
        the more embarrassing leak. No call site logs it today; that is a fact
        about today's call sites, not a property of the type, and the type is
        where the property belongs.
        """
        return f"_DeviceMint(expires_in={self.expires_in!r}, interval={self.interval!r})"


def _validated_verification_uri(value: object, *, user_code: str | None = None) -> str:
    """Validate a printable device-approval URL and its user-code fragment.

    Require HTTPS, or HTTP for loopback, without userinfo. Allow a path and fragment,
    but require the correct user_code in the fragment and reject it elsewhere to
    keep it out of HTTP/access logs and Referer headers. Missing or wrong fragments
    must fail instead of silently turning one-click approval into retyping.

    Do not pin the origin to api_url: the operator controls the complete linking
    destination. This checks endpoint conformance, not phishing; cloud approval owns
    that protection.
    """
    if not isinstance(value, str) or not value:
        raise _protocol_error("The endpoint answered without a verification address.")
    parts = urlsplit(value)
    hostname = parts.hostname
    if not hostname or parts.username or parts.password:
        raise _protocol_error("The endpoint answered with a verification address it cannot own.")
    if parts.scheme != "https" and not (parts.scheme == "http" and _is_loopback(hostname)):
        raise _protocol_error(
            "The endpoint answered with a verification address that is not https "
            "(plain http is accepted only for loopback hosts)."
        )
    if user_code is not None:
        # The "outside the fragment" half is checked against a NORMALISED copy
        # of both the code and the place it might be hiding. The console forgives
        # case and hyphenation on entry, so `?user_code=7q3mx2vf9kht` is the
        # same live credential as `?user_code=7Q3M-X2VF-9KHT` and leaks into
        # exactly the same access logs, `Referer` headers and link histories —
        # a substring test on the canonical spelling alone would wave it through.
        # The fragment half stays EXACT: what we require there is the code the
        # terminal is printing, verbatim, not a variant of it.
        canon = user_code.replace("-", "").casefold()
        if canon in parts.query.replace("-", "").casefold() or canon in (
            parts.path.replace("-", "").casefold()
        ):
            raise _protocol_error(
                "The endpoint answered with a verification address carrying the user code "
                "outside its fragment."
            )
        if user_code not in parts.fragment:
            raise _protocol_error(
                "The endpoint answered with a one-click verification address that does not "
                "carry the user code in its fragment."
            )
    return value


def _positive_int(value: object, *, ceiling: int) -> int:
    """One cadence integer from the mint, or a protocol error.

    `bool` is excluded explicitly because it is an `int` in Python and
    `True` would otherwise read as a one-second interval.
    """
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= ceiling:
        raise _protocol_error("The endpoint answered with a cadence this daemon cannot honour.")
    return value


async def _mint_device_code(
    request: Request, body: LinkDeviceStartRequest, identity_verifier: str, fingerprint: str
) -> _DeviceMint:
    """Exchange the public verifier for display and device codes.

    Use both HTTP inactivity and wall-clock deadlines under the link mutation lock.
    No secret is submitted, so refusal reflection checks receive an empty input.
    Validate returned user code and approval URL before presenting them to humans.
    """
    url = body.api_url.rstrip("/") + "/api/link/device"
    hostname, daemon_version, os_name = _machine_facts()
    transport = getattr(request.app.state, "link_claim_transport", None)
    try:
        async with asyncio.timeout(_CLAIM_TOTAL_TIMEOUT_S):
            async with httpx.AsyncClient(
                timeout=_CLAIM_TIMEOUT_S, follow_redirects=False, transport=transport
            ) as http:
                response = await http.post(
                    url,
                    json={
                        "credential_reference": identity_verifier,
                        "hostname_hint": hostname,
                        "daemon_version": daemon_version,
                        "os": os_name,
                    },
                )
    except (httpx.HTTPError, TimeoutError) as exc:
        raise NerditError(
            502,
            "link.cloud_unreachable",
            f"The cloud endpoint {urlsplit(body.api_url).netloc} could not be reached.",
            hint="Check --api-url and this machine's outbound connectivity, then retry.",
        ) from exc

    if response.status_code >= 400:
        raise _refusal(response, "")

    try:
        payload = response.json()
        device_code = payload["device_code"]
        user_code = payload["user_code"]
        cloud_fingerprint = payload["credential_fingerprint"]
    except (KeyError, TypeError, ValueError) as exc:
        raise _protocol_error(
            "The endpoint answered 200 without a device code pair — check --api-url."
        ) from exc

    if (
        not isinstance(device_code, str)
        or not 1 <= len(device_code.strip()) <= _DEVICE_CODE_MAX_LENGTH
        or not isinstance(user_code, str)
        or not _USER_CODE_RE.fullmatch(user_code)
    ):
        raise _protocol_error(
            "The endpoint answered 200 without a device code pair — check --api-url."
        )
    if not isinstance(cloud_fingerprint, str) or cloud_fingerprint != fingerprint:
        # Both sides compute `sha256` over the full `ed25519:…` verifier, so
        # a mismatch is not a disagreement about formatting — it means the peer
        # on the other end is not running the protocol this daemon vendored
        # against, and the eight characters the terminal is about to tell an
        # operator to compare against the console would be a lie. A cloud fault,
        # hence 502 rather than a refusal.
        raise NerditError(
            502,
            "link.cloud_error",
            "The cloud described a different credential than this node holds.",
            hint="The endpoint is not the Nerdit cloud this daemon speaks to — check --api-url.",
        )

    return _DeviceMint(
        # `strip` and nothing else, exactly as the cloud's own
        # `device_code_hash` strips before hashing: the value is machine-held
        # end to end and every other spelling is a mistake, not a human being
        # punished for spacing.
        device_code=device_code.strip(),
        user_code=user_code,
        # The bare URI is the "type the code yourself" address and carries no
        # code at all; only the one-click form is checked for where the code
        # sits, because only it is supposed to carry one (D-X16-O24).
        verification_uri=_validated_verification_uri(payload.get("verification_uri")),
        verification_uri_complete=_validated_verification_uri(
            payload.get("verification_uri_complete"), user_code=user_code
        ),
        credential_fingerprint=cloud_fingerprint,
        expires_in=_positive_int(payload.get("expires_in"), ceiling=_DEVICE_MAX_TTL_S),
        interval=_positive_int(payload.get("interval"), ceiling=_DEVICE_MAX_INTERVAL_S),
    )


@router.post(
    "/link/device",
    response_model=LinkDeviceStartView,
    operation_id="start_device_link",
    status_code=200,
)
async def start_device_link(request: Request, body: LinkDeviceStartRequest) -> LinkDeviceStartView:
    """Start an admin-authorized device-link flow and keep its grant in memory.

    Require Idempotency-Key for the remote durable effect. Before contacting cloud,
    check existing identity, relay availability and pending data-dir changes, then
    load identity off-loop. Entitlement is decided by cloud approval, not here.
    Return display material and a session selector only, never device_code.
    """
    require_role(request, TokenRole.admin)
    # The HOST only, never the full URL — the api_host rule the claim documents.
    # The `user_code` this route is about to mint is NOT here and never will
    # be: it is a link-code-shaped secret in transit, displayed by the CLI
    # because that is its job, and recorded nowhere.
    request.state.audit_params = audit_params(
        {
            "api_host": urlsplit(body.api_url).netloc,
            "relay_host": None if body.relay_url is None else urlsplit(body.relay_url).netloc,
            "enable": body.enable,
        }
    )
    _require_idempotency_key(request, "device link start")

    async with _LINK_MUTATION_LOCK:
        store = _store(request)
        try:
            effective = store.effective_section("link")
        except ConfigError as exc:  # pragma: no cover - "link" is a known section
            raise _as_nerdit_error(exc) from exc

        # (D8) Before any cloud contact, exactly as the claim refuses: minting a
        # device code for an already-linked daemon would put a live approval
        # screen in front of a human for a node that cannot consume it.
        linked_as = effective.get("node_id")
        if linked_as:
            raise NerditError(
                409,
                "link.already_linked",
                f"This daemon is already linked as node {linked_as}.",
                hint="Run 'nerdit unlink' first if you mean to re-link.",
            )

        # (D5) Fail fast here AND re-check at commit: the window between the two
        # is a human, and an operator can re-stage `[link].relay_url` through
        # the config API while the approval screen is open.
        if not (body.relay_url or effective.get("relay_url")):
            raise NerditError(
                422,
                "link.relay_url_required",
                "No relay endpoint is configured for this daemon.",
                hint="Pass --relay-url wss://… (persisted when the link commits).",
            )

        stored_data_dir = _refuse_pending_data_dir(store, _data_dir(request))
        key_file = resolve_key_file(effective.get("key_file"), _data_dir(request))
        try:
            # Synchronous by design (the `load_or_create_identity` docstring),
            # so it runs off the loop.
            identity = await asyncio.to_thread(load_or_create_identity, key_file)
        except LinkIdentityError as exc:
            # The path belongs in the log (the identity.py precedent) and
            # nowhere else: the response message stays path-free.
            logger.error("Node identity unavailable for device link: %s", exc)
            raise NerditError(
                500,
                "link.identity_unavailable",
                "The node identity key could not be loaded or created.",
                hint="See the daemon log for the failing path, then fix or remove the key file.",
            ) from exc

        # Pinned BEFORE the cloud hop, not after it: the mint creates a durable
        # row on the cloud, and that row is a non-DB side effect this request can
        # no longer take back. Without the mark, a cancellation landing between
        # the cloud accepting the mint and this response completing — a client
        # disconnect, or the uvicorn 30 s shutdown cancel — lets
        # `IdempotencyMiddleware` delete the in-progress claim as though
        # nothing had happened; the retry then mints a SECOND cloud row and
        # replaces the slot, instead of answering `idempotency_interrupted`
        # and letting the operator see that the first attempt got somewhere.
        # The /system/backup precedent, applied to the first hop that can
        # commit remote state.
        mark_request_side_effect()
        mint = await _mint_device_code(request, body, identity.verifier, identity.fingerprint)

        session = uuid.uuid4().hex
        # A second start REPLACES the slot and its session, so the older
        # terminal's next poll is refused `link.device_superseded` rather than
        # being handed this flow's code (D-P34-1). Assigned INSIDE the lock so
        # two concurrent starts cannot interleave mint and store.
        request.app.state.link_device_pending = _DevicePending(
            session=session,
            device_code=mint.device_code,
            verifier=identity.verifier,
            verifier_fingerprint=identity.fingerprint,
            key_file=key_file,
            key_file_snapshot=effective.get("key_file"),
            data_dir_snapshot=stored_data_dir,
            api_url=body.api_url,
            relay_url=body.relay_url,
            enable=bool(body.enable),
            expires_at=time.monotonic() + mint.expires_in,
            interval=mint.interval,
        )

    logger.info(
        "Device link requested (fingerprint %s, %s s to approve)",
        identity.fingerprint[:8],
        mint.expires_in,
    )
    request.state.audit_params = audit_params(
        {
            "api_host": urlsplit(body.api_url).netloc,
            # The identity.py-documented non-secret audit form of the key.
            "verifier_fingerprint": identity.fingerprint,
            "enable": bool(body.enable),
            "expires_in": mint.expires_in,
        }
    )
    hostname, daemon_version, os_name = _machine_facts()
    return LinkDeviceStartView(
        session=session,
        user_code=mint.user_code,
        verification_uri=mint.verification_uri,
        verification_uri_complete=mint.verification_uri_complete,
        credential_fingerprint=mint.credential_fingerprint,
        hostname=hostname,
        daemon_version=daemon_version,
        os=os_name,
        expires_in=mint.expires_in,
        interval=mint.interval,
    )


@dataclass(frozen=True, slots=True)
class _DevicePollAnswer:
    """What one cloud poll said: still waiting, or the identity to commit."""

    #: `"pending"`, `"slow_down"` or `"linked"`.
    status: str
    interval: int | None = None
    node_id: str | None = None
    slug: str | None = None
    domain: str | None = None


async def _poll_device_code(request: Request, pending: _DevicePending) -> _DevicePollAnswer:
    """Poll cloud once without the link lock or raw-body reflection.

    CLI cadence follows cloud interval/slow_down; there is no extra retry loop.
    Map refusals by HTTP status, since different statuses may share a cloud_code.
    Return that safe token as context, not as the discriminator.
    """
    url = pending.api_url.rstrip("/") + "/api/link/device/poll"
    transport = getattr(request.app.state, "link_claim_transport", None)
    try:
        async with asyncio.timeout(_CLAIM_TOTAL_TIMEOUT_S):
            async with httpx.AsyncClient(
                timeout=_CLAIM_TIMEOUT_S, follow_redirects=False, transport=transport
            ) as http:
                response = await http.post(
                    url,
                    json={
                        # Verbatim from the slot, stripped only — the cloud
                        # hashes `sha256(strip)` and forgives nothing else on
                        # this machine-held field.
                        "device_code": pending.device_code,
                        # The verifier pinned at mint: D-X16-O22's constant-time
                        # comparison against it runs before any state is
                        # revealed, cloud-side.
                        "credential_reference": pending.verifier,
                    },
                )
    except (httpx.HTTPError, TimeoutError) as exc:
        # The slot is KEPT: a transport failure says nothing about the request's
        # fate, and the CLI retries on the next interval until its own timeout.
        raise NerditError(
            502,
            "link.cloud_unreachable",
            f"The cloud endpoint {urlsplit(pending.api_url).netloc} could not be reached.",
            hint="Check this machine's outbound connectivity — polling can resume.",
        ) from exc

    if response.status_code == 202:
        # The two RFC 8628 non-terminal answers. The advertised interval is
        # passed through untouched: cadence is the cloud's to tune without a
        # daemon release, and racing a gate that fires before any database read
        # is both rude and useless.
        try:
            payload = response.json()
            status = payload["status"]
            interval = payload["interval"]
        except (KeyError, TypeError, ValueError) as exc:
            raise _protocol_error("The endpoint answered 202 without a poll status.") from exc
        if status == "slow_down":
            return _DevicePollAnswer(
                status="slow_down",
                interval=_positive_int(interval, ceiling=_DEVICE_MAX_INTERVAL_S),
            )
        if status == "authorization_pending":
            return _DevicePollAnswer(
                status="pending",
                interval=_positive_int(interval, ceiling=_DEVICE_MAX_INTERVAL_S),
            )
        raise _protocol_error("The endpoint answered 202 with an unknown poll status.")

    if response.status_code >= 400:
        raise _device_refusal(request, response, pending)

    node_id, slug, domain = _parse_claim_identity(response, pending.device_code)
    return _DevicePollAnswer(status="linked", node_id=node_id, slug=slug, domain=domain)


async def _assert_credential_unchanged(request: Request, pending: _DevicePending) -> None:
    """Refuse commit unless the enrolled on-disk credential still exists and matches.

    Check existence before load_or_create_identity to avoid minting a replacement.
    This also protects against restored backups, swapped keys and unreadable files.
    """

    if not pending.key_file.exists():
        _clear_pending(request, pending)
        raise NerditError(
            409,
            "link.node_credential_changed",
            "The node credential this request was minted for no longer exists.",
            hint=_CREDENTIAL_CHANGED_HINT,
        )
    try:
        on_disk = await asyncio.to_thread(load_or_create_identity, pending.key_file)
    except LinkIdentityError as exc:
        # The path belongs in the log and nowhere else — the claim route's
        # precedent, and the same path-free response posture.
        logger.error("Node identity unavailable for device commit: %s", exc)
        _clear_pending(request, pending)
        raise NerditError(
            500,
            "link.identity_unavailable",
            "The node identity key could not be loaded or created.",
            hint="Check the daemon data directory's permissions, then link again.",
        ) from exc
    # A plain `!=`, deliberately: the fingerprint is derived from the PUBLIC
    # key and is printed to terminals and consoles alike (D-X16-O4), so there is
    # no secret here to leak through timing, and reaching for `compare_digest`
    # would imply a property this value does not have.
    if on_disk.fingerprint != pending.verifier_fingerprint:
        _clear_pending(request, pending)
        raise NerditError(
            409,
            "link.node_credential_changed",
            "The node credential changed while this request was waiting for approval.",
            hint=_CREDENTIAL_CHANGED_HINT,
        )


def _clear_pending(request: Request, pending: _DevicePending) -> None:
    """Drop `pending`'s slot — the flow is over, one way or another.

    Identity-guarded (D-P34-1): the poll's cloud hop runs lock-free for up to
    30 s, and a concurrent `start_device_link` may have REPLACED the slot in
    that window. The replacement's own contract is "slot untouched" for the
    superseded flow — its next poll answers `link.device_superseded` — so a
    terminal answer for the OLD flow must settle only the old flow: clearing
    whatever slot is current would kill the newer terminal's live code, turn
    its next poll into a bogus `device_not_started`, and leave its approval
    to enroll an orphan cloud node — exactly the cross-talk the session
    binding exists to prevent. `is`, not `==`: the slot is the object.
    """
    if request.app.state.link_device_pending is pending:
        request.app.state.link_device_pending = None


def _forwarded_retry_after(response: httpx.Response) -> dict[str, str] | None:
    """Forward only positive Retry-After delta-seconds, reserialized locally.

    Drop dates, malformed, absent or nonpositive values so the CLI uses its fallback.
    Never copy unrelated headers describing the cloud's response.
    """
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = int(raw.strip())
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return {"Retry-After": str(seconds)}


def _device_refusal(
    request: Request, response: httpx.Response, pending: _DevicePending
) -> NerditError:
    """Map a cloud poll refusal onto a stable daemon code, and settle the slot.

    Which refusals CLEAR the slot is the whole of the table: a terminal answer
    (denied, expired, unknown, credential mismatch) means this request is dead
    and a further poll would be a lie, so the slot goes and the next poll says
    `device_not_started`. A rate limit or a cloud outage says nothing about
    the request's fate, so the slot stays and the CLI keeps polling.

    The cloud's *body* never travels (D11) — only its machine token, as
    `cloud_code`, and only after `_cloud_code` has proved it does not
    contain the device code this daemon just submitted.
    """
    code = _cloud_code(response, pending.device_code)
    status = response.status_code
    if status == 403:
        _clear_pending(request, pending)
        return NerditError(
            409,
            "link.device_denied",
            "The link request was denied in the console.",
            hint="Run 'nerdit link --device' again if that was a mistake.",
            cloud_code=code,
            cloud_status=status,
        )
    if status == 410:
        _clear_pending(request, pending)
        return NerditError(
            409,
            "link.device_expired",
            "The code expired before it was approved.",
            hint="Run 'nerdit link --device' again to mint a fresh code.",
            cloud_code=code,
            cloud_status=status,
        )
    if status == 404:
        # The cloud deliberately collapses unknown, wrong-flow and
        # attempt-cap-bricked into one answer (D-X16-O9's no-oracle rule); the
        # daemon invents no distinction it was not given.
        _clear_pending(request, pending)
        return NerditError(
            409,
            "link.device_invalid",
            "The cloud does not recognise this link request.",
            hint="Run 'nerdit link --device' again to mint a fresh code.",
            cloud_code=code,
            cloud_status=status,
        )
    if status in (400, 409):
        # `node_authentication_failed` on both, for a credential that is
        # malformed or no longer the one the request was pinned to — the node
        # key changed under a live flow. Terminal: the pin cannot be un-changed.
        _clear_pending(request, pending)
        return NerditError(
            409,
            "link.device_credential_mismatch",
            "This node's credential is not the one the link request was minted for.",
            hint="The node key changed — run 'nerdit link --device' to start a new request.",
            cloud_code=code,
            cloud_status=status,
        )
    if status == 429:
        # (D-X16-O26) The cloud computes the backoff from its own rate-limit
        # window and ships it as `Retry-After`; forwarding it is what makes
        # that knob server-tunable without a daemon release. Dropped, the CLI
        # falls back to doubling its local interval, which is safe but deaf.
        # The hint says "back off", not "wait for the interval": `interval`
        # rides on the 202 answers, and a 429 body carries none.
        return NerditError(
            429,
            "link.claim_rate_limited",
            "The cloud is rate-limiting link polls.",
            hint="Back off before polling again: honour Retry-After when it is present.",
            headers=_forwarded_retry_after(response),
            cloud_code=code,
            cloud_status=status,
        )
    if status < 500:
        return NerditError(
            409,
            "link.claim_refused",
            "The cloud refused the link request.",
            hint=_CLAIM_HINTS.get(code, _CLAIM_HINT_FALLBACK),
            cloud_code=code,
            cloud_status=status,
        )
    return NerditError(
        502,
        "link.cloud_error",
        "The cloud could not answer the link poll.",
        hint="This is a cloud-side failure — polling can resume shortly.",
        cloud_code=code,
        cloud_status=status,
    )


@router.post(
    "/link/device/poll",
    response_model=LinkDevicePollView,
    operation_id="poll_device_link",
    status_code=200,
)
async def poll_device_link(request: Request, body: LinkDevicePollRequest) -> LinkDevicePollView:
    """Poll for human approval and commit an approved device-link identity.

    Admin-only; each poll needs a fresh Idempotency-Key. Before cloud contact check,
    in order: already-linked, mismatched session, absent slot and expired deadline.
    Return the corresponding structured 409; preserve a superseded slot and clear an
    expired one. Already-linked also handles lost successful responses without
    misidentifying a different flow as this one's success.

    Poll without the lock; reacquire it only for the shared approved commit. Mark a
    durable side effect only at commit, never while pending. Cloud approval replays
    and the local already-linked response let dropped responses converge safely.
    """
    require_role(request, TokenRole.admin)
    _require_idempotency_key(request, "device link poll")

    store = _store(request)
    try:
        effective = store.effective_section("link")
    except ConfigError as exc:  # pragma: no cover - "link" is a known section
        raise _as_nerdit_error(exc) from exc

    pending: _DevicePending | None = getattr(request.app.state, "link_device_pending", None)

    # (1) The D8 refusal, and the lost-response recovery — see the docstring.
    linked_as = effective.get("node_id")
    if linked_as:
        slug = effective.get("slug")
        raise NerditError(
            409,
            "link.already_linked",
            f"This daemon is already linked as node {linked_as}"
            + (f" (slug {slug})." if slug else "."),
            hint="Run 'nerdit unlink' first if you mean to re-link.",
        )

    # (2) The session binding, BEFORE the not-started answer: a mismatch means
    # some other flow owns the slot, which is a different fact from "there is no
    # flow" and gets a different message. The slot is deliberately untouched —
    # the newer terminal still owns it.
    if pending is not None and pending.session != body.session:
        raise NerditError(
            409,
            "link.device_superseded",
            "Another link request replaced this one.",
            hint="This terminal's code is dead; the newer terminal owns the flow.",
        )

    # (3) No slot: a restarted daemon, or a poll that never had a start. The
    # session is memory-only and honestly so — nothing here pretends otherwise.
    if pending is None:
        raise NerditError(
            409,
            "link.device_not_started",
            "No device link request is pending on this daemon.",
            hint="Run 'nerdit link --device' to start a new request.",
        )

    # (4) The LOCAL deadline. The cloud has its own and would answer 410, but a
    # daemon that already knows the request is dead should not spend a poll (and
    # a rate-limit budget) proving it.
    if time.monotonic() >= pending.expires_at:
        _clear_pending(request, pending)
        raise NerditError(
            409,
            "link.device_expired",
            "The code expired before it was approved.",
            hint="Run 'nerdit link --device' again to mint a fresh code.",
        )

    # Recorded before the hop so a refusal is still audited with what it was
    # asked to do. The `session` is deliberately absent: it is grantless, but
    # it is also pure noise in a durable row.
    request.state.audit_params = audit_params(
        {
            "api_host": urlsplit(pending.api_url).netloc,
            "verifier_fingerprint": pending.verifier_fingerprint,
        }
    )

    answer = await _poll_device_code(request, pending)

    if answer.status != "linked":
        request.state.audit_params = audit_params(
            {
                "api_host": urlsplit(pending.api_url).netloc,
                "verifier_fingerprint": pending.verifier_fingerprint,
                "status": answer.status,
            }
        )
        return LinkDevicePollView(status=answer.status, interval=answer.interval)

    assert answer.node_id is not None and answer.slug is not None  # noqa: S101 - narrowing

    async with _LINK_MUTATION_LOCK:
        # The session binding closed the PRE-FLIGHT window; this closes the
        # commit one, which is the same hole one step later. `_poll_device_code`
        # runs its cloud hop lock-free for up to 30 s, and a concurrent
        # `start_device_link` may have replaced the slot inside it — so a poll
        # that comes back approved must re-establish that it is still the
        # daemon's current flow, not merely that nothing else finished linking.
        # Checking `node_id` alone is not that: it is unset in exactly this
        # race, so the superseded flow would commit, take custody of the daemon,
        # and leave the newer terminal polling a slot whose approval can never
        # land. `is`, not `==`: the slot is the object.
        if request.app.state.link_device_pending is not pending:
            raise NerditError(
                409,
                "link.device_superseded",
                "Another link request replaced this one while it was waiting.",
                hint="Run 'nerdit link --device' again to start a fresh request.",
            )
        try:
            effective = store.effective_section("link")
        except ConfigError as exc:  # pragma: no cover - "link" is a known section
            raise _as_nerdit_error(exc) from exc
        # A concurrent claim or pre-auth enroll won the lock while this poll was
        # in flight. The approval is not lost — the cloud replays 200 forever —
        # but this daemon already has an identity, and overwriting it would be a
        # silent re-link. Slot cleared: this flow is over either way.
        already = effective.get("node_id")
        if already:
            _clear_pending(request, pending)
            raise NerditError(
                409,
                "link.already_linked",
                f"This daemon is already linked as node {already}.",
                hint="Run 'nerdit unlink' first if you mean to re-link.",
            )
        # Re-resolve relay URL in the synchronous commit tail after all awaits; config
        # writers may change it during approval or credential checks. Verify the enrolled
        # key still exists and matches before committing identity. Check existence first:
        # load_or_create_identity must not silently mint a replacement.
        await _assert_credential_unchanged(request, pending)
        committed = _commit_link_identity(
            request,
            store,
            node_id=answer.node_id,
            slug=answer.slug,
            domain=answer.domain,
            relay_url_to_persist=pending.relay_url,
            enable=pending.enable,
            key_file=pending.key_file,
            key_file_snapshot=pending.key_file_snapshot,
            data_dir_snapshot=pending.data_dir_snapshot,
        )
        _clear_pending(request, pending)

    logger.info("Linked as node %s (config etag %s)", answer.node_id, committed.etag[:12])

    # The success of a device link IS a link creation, so it audits as
    # `link.created` with the claim's exact param set rather than inventing a
    # second vocabulary for the same event (D-P34-1). The override seam is the
    # one `deploy.plan` already uses; `derive_action` stays path-only, so
    # every non-approving poll still records as `link.device_poll`.
    request.state.audit_action = "link.created"
    request.state.audit_target = answer.node_id
    request.state.audit_params = audit_params(
        {
            "node_id": answer.node_id,
            "slug": answer.slug,
            "verifier_fingerprint": pending.verifier_fingerprint,
            "relay_host": urlsplit(committed.relay_url).netloc,
            "enabled": pending.enable,
            "nodes_base_domain": answer.domain,
            "mcp_http_enabled": committed.mcp_http_enabled,
            "grant": "device",
        }
    )
    return LinkDevicePollView(
        status="linked",
        node_id=answer.node_id,
        slug=answer.slug,
        relay_url=committed.relay_url,
        enabled=pending.enable,
        verifier_fingerprint=pending.verifier_fingerprint,
        # What the daemon now HOLDS, not merely what this exchange returned: a
        # cloud that sent nothing leaves an earlier refresh's value standing.
        nodes_base_domain=(
            answer.domain if answer.domain is not None else effective.get("nodes_base_domain")
        ),
        # Hardcoded, not read off the staged diff: every `[link]` key is
        # restart-keyed, so a link ALWAYS needs a restart to take effect.
        requires_restart=True,
        restart_keys=committed.restart_keys,
        mcp_http_enabled=committed.mcp_http_enabled,
        mcp_skipped_reason=committed.mcp_skipped_reason,
    )


@router.put(
    "/link/entitlement",
    response_model=EntitlementPushView,
    operation_id="push_link_entitlement",
    status_code=200,
)
async def push_link_entitlement(
    request: Request, body: EntitlementPushRequest
) -> EntitlementPushView:
    """Mirror the cloud's permission to publish a public hosted share.

    Require a synthetic link principal and x-nerdit-cloud-control: entitlement;
    operator tokens are refused. Cloud forwarding strips that header from user
    traffic. This trusts the relay; anonymous traffic remains cloud-edge controlled.

    Keep the assertion in memory, false at boot and cleared on disconnect/unlink.
    Apply a 24-hour read TTL and issued_at ordering; reject excessive future skew.
    No Idempotency-Key is required for this value-idempotent push. Older daemons may
    return either 404 or 400 idempotency_key_required; pushers must treat both as
    unsupported. Audit/event emission is change-only with the boolean alone;
    denials are always recorded.
    """
    require_cloud_principal(request, CLOUD_CONTROL_ENTITLEMENT)

    manager = getattr(request.app.state, "link_manager", None)
    if manager is None:
        # Unreachable in the real app — a tunnel principal cannot be resolved
        # without a live manager, since the manager IS what validated the
        # capability. Answering the SAME 403 rather than a 500 keeps the route
        # from becoming an oracle for daemon internals.
        raise NerditError(
            403,
            "link.cloud_principal_required",
            "This endpoint is written by the Nerdit cloud over the node link only.",
            hint="Only the cloud can confirm the linked account's access.",
        )

    # `manager.now()`, not `datetime.now(UTC)`: the seam's
    # `received_at`, its TTL and its ordering comparison all read the
    # manager's injected clock, and a skew check on a second clock would let the
    # two disagree about which stamps are in the future. Identical in
    # production — the default clock IS `datetime.now(UTC)`.
    now = manager.now()
    if body.issued_at > now + timedelta(seconds=ENTITLEMENT_MAX_FUTURE_SKEW_S):
        raise NerditError(
            422,
            "validation_error",
            "issued_at is too far in the future for this daemon's clock.",
            hint=(
                "Allowed skew is "
                f"{ENTITLEMENT_MAX_FUTURE_SKEW_S} s — check clock sync on both ends."
            ),
        )

    update = manager.set_hosted_public_entitled(body.hosted_public_entitled, body.issued_at)

    if update.changed:
        status = manager.status()
        request.state.audit_target = status.node_id
        request.state.audit_params = audit_params({"hosted_public_entitled": update.effective})
        await manager.record_entitlement_change(update.effective)
        logger.info("Hosted-public entitlement mirror changed to %s", update.effective)
    else:
        # Change-only recording: the cloud re-asserts every few minutes, and a
        # row per re-assert would bury the audit log in noise that carries no
        # information. Honoured for 2xx only, so a denial is still recorded.
        request.state.audit_skip = True

    return EntitlementPushView(
        hosted_public_entitled=update.effective,
        received_at=update.received_at,
        expires_at=update.expires_at,
        applied=update.applied,
        changed=update.changed,
    )


def _cloud_manager(request: Request, *, hint: str) -> Any:
    """The live link manager, or the same 403 the principal gate raises.

    Unreachable in the real app — a tunnel principal cannot be resolved
    without a live manager, since the manager IS what validated the
    capability. Answering the SAME 403 rather than a 500 keeps the route from
    becoming an oracle for daemon internals.
    """
    manager = getattr(request.app.state, "link_manager", None)
    if manager is None:
        raise NerditError(
            403,
            "link.cloud_principal_required",
            "This endpoint is written by the Nerdit cloud over the node link only.",
            hint=hint,
        )
    return manager


@router.put(
    "/link/github-token",
    response_model=GithubTokenPushView,
    operation_id="push_link_github_token",
    status_code=200,
)
async def push_link_github_token(
    request: Request, body: GithubTokenPushRequest
) -> GithubTokenPushView:
    """Mirror a short-lived GitHub token from the cloud control channel.

    Require a synthetic link principal and x-nerdit-cloud-control: github-token.
    Store in memory by installation ID, accepting newer issued_at only; apply expiry
    on read and clear on offline, unlink and shutdown. Resolution is by repository.
    This ordered value-set needs no Idempotency-Key.

    Never expose token values or repo names in logs, audits or responses. Change-only
    audit carries installation ID, expiry and repo count. A forged replacement can
    break cloning, which fails closed; GitHub token lifetime remains at most one hour.
    """
    require_cloud_principal(request, CLOUD_CONTROL_GITHUB_TOKEN)
    manager = _cloud_manager(
        request, hint="Operators do not set GitHub tokens; the cloud mints and pushes them."
    )

    now = manager.now()
    if body.issued_at > now + timedelta(seconds=ENTITLEMENT_MAX_FUTURE_SKEW_S):
        raise NerditError(
            422,
            "validation_error",
            "issued_at is too far in the future for this daemon's clock.",
            hint=(
                "Allowed skew is "
                f"{ENTITLEMENT_MAX_FUTURE_SKEW_S} s — check clock sync on both ends."
            ),
        )

    update = manager.set_github_token(
        installation_id=body.installation_id,
        token=body.token,
        expires_at=body.expires_at,
        issued_at=body.issued_at,
        repos=body.repos,
    )

    if update.changed:
        request.state.audit_target = manager.status().node_id
        request.state.audit_params = audit_params(
            {
                "installation_id": update.installation_id,
                "expires_at": update.expires_at.isoformat(),
                "repos_count": update.repos_count,
            }
        )
        logger.info(
            "GitHub installation token mirror updated for installation %s (%d repos)",
            update.installation_id,
            update.repos_count,
        )
    else:
        # Change-only, the entitlement precedent: the cloud re-pushes the same
        # token on a timer until it re-mints, and a row per push is noise.
        # Honoured for 2xx only, so a denial is still recorded.
        request.state.audit_skip = True

    return GithubTokenPushView(
        applied=update.applied,
        installation_id=update.installation_id,
        expires_at=update.expires_at,
        repos_count=update.repos_count,
    )


@router.post(
    "/link/git-nudge",
    response_model=GitNudgeView,
    operation_id="nudge_git",
    status_code=202,
)
async def nudge_git(request: Request, body: GitNudgeRequest) -> GitNudgeView:
    """Make matching GitWatch polls immediate after a cloud push notification.

    Require a synthetic link principal, x-nerdit-cloud-control: git-nudge and an
    Idempotency-Key (the GitHub delivery ID). The controller also deduplicates by
    service/sha. Reuse normal GitWatch guards and redeploy; polling remains fallback.

    SHA is a hint, never a checkout target: clone the recorded ref and report the
    actual resolved head. Return 202 best-effort, with empty lists when GitWatch is
    disabled. Audit canonical repo/ref/sha and matched service names.
    """
    require_cloud_principal(request, CLOUD_CONTROL_GIT_NUDGE)
    manager = _cloud_manager(
        request, hint="Operators do not nudge; GitWatch polls, and the cloud relays pushes."
    )
    _require_idempotency_key(request, "git nudge")

    request.state.audit_target = manager.status().node_id
    gitwatch = getattr(request.app.state, "gitwatch", None)
    if gitwatch is None:
        logger.info("Git nudge for %s@%s ignored: GitWatch is not running", body.repo, body.ref)
        matched: list[str] = []
        ignored: list[str] = []
        deduped: list[str] = []
    else:
        outcome = await gitwatch.nudge(body.repo, body.ref, body.sha)
        matched, ignored, deduped = outcome.matched, outcome.ignored, outcome.deduped
        logger.info(
            "Git nudge for %s@%s: matched=%s ignored=%s deduped=%s",
            body.repo,
            body.ref,
            matched,
            ignored,
            deduped,
        )
    request.state.audit_params = audit_params(
        {"repo": body.repo, "ref": body.ref, "sha": body.sha, "matched": matched}
    )
    return GitNudgeView(matched=matched, ignored=ignored, deduped=deduped)
