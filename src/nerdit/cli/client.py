"""HTTP client for communicating with nerditd."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx

from nerdit.config.defaults import DEFAULT_HOST, DEFAULT_PORT


def _decode_sse_data(payload: list[str]) -> dict | None:
    """Decode SSE data lines; return None for empty, malformed or non-object payloads."""
    if not payload:
        return None
    try:
        decoded = json.loads("\n".join(payload))
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _encode_dot_segment(segment: str) -> str:
    """Percent-encode a path segment without allowing dot-segment normalization.

    `quote(..., safe="")` leaves dots literal. Encode all-dot segments explicitly
    so httpx cannot normalize them away; other segments use ordinary quoting.
    """
    quoted = quote(segment, safe="")
    return quoted.replace(".", "%2E") if segment in (".", "..") else quoted


class NerditClient:
    """Async HTTP client wrapping the nerditd REST API."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        token: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = host
        self._base_url = f"http://{host}:{port}"
        self._headers: dict[str, str] = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._transport = transport

    def _client(self, **kwargs) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with auth headers (and optional test transport)."""
        if self._transport is not None:
            kwargs.setdefault("transport", self._transport)
        return httpx.AsyncClient(headers=self._headers, **kwargs)

    async def _request_json(self, method: str, url: str, **kwargs) -> Any:
        """Send an ordinary request with this client's auth and transport."""
        async with self._client() as client:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> dict:
        """Call `GET /health` and return the parsed JSON response."""
        return await self._request_json("GET", f"{self._base_url}/health", timeout=5.0)

    async def list_gpus(self) -> list[dict]:
        """Fetch all GPUs from the daemon."""
        return await self._request_json("GET", f"{self._base_url}/gpus", timeout=5.0)

    async def cluster_stats(self) -> dict:
        """Fetch aggregate cluster metrics from the API-only `/api/cluster/stats` route."""
        return await self._request_json("GET", f"{self._base_url}/api/cluster/stats", timeout=5.0)

    async def get_config(self, section: str | None = None) -> dict | list:
        """Read daemon config via `GET /api/config/daemon[/{section}]`.

        Returns a single `ConfigView` dict when `section` is given, or
        a list of section views otherwise. Secrets are redacted server-side.
        """
        path = "/api/config/daemon"
        if section:
            path = f"{path}/{section}"
        return await self._request_json("GET", f"{self._base_url}{path}", timeout=5.0)

    async def put_config(
        self,
        section: str,
        values: dict,
        *,
        dry_run: bool = False,
        idempotency_key: str | None = None,
        if_match: str | None = None,
    ) -> dict:
        """Write a daemon config section via `PUT /api/config/daemon/{section}`.

        Carries `Idempotency-Key` (real writes) and `If-Match` (optimistic
        concurrency) headers and the `dry_run` query flag.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if if_match:
            headers["If-Match"] = if_match
        params = {"dry_run": "true"} if dry_run else None
        return await self._request_json(
            "PUT",
            f"{self._base_url}/api/config/daemon/{section}",
            json=values,
            params=params,
            headers=headers,
            timeout=10.0,
        )

    async def apply_config(
        self,
        sections: dict[str, dict],
        *,
        dry_run: bool = False,
        idempotency_key: str | None = None,
        if_match: str | None = None,
    ) -> dict:
        """Declaratively apply a multi-section config document via
        `POST /api/config/daemon/apply`.

        Carries `Idempotency-Key` and `If-Match` headers (both required by
        the daemon on a real apply; both exempt on `dry_run`).
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if if_match:
            headers["If-Match"] = if_match
        params = {"dry_run": "true"} if dry_run else None
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/config/daemon/apply",
            json={"sections": sections},
            params=params,
            headers=headers,
            timeout=10.0,
        )

    async def get_app_config(self, name: str) -> dict:
        """Read a deployed app's config via `GET /api/config/apps/{name}`."""
        return await self._request_json(
            "GET", f"{self._base_url}/api/config/apps/{name}", timeout=5.0
        )

    async def put_app_config(
        self,
        name: str,
        section: str,
        values: dict,
        *,
        dry_run: bool = False,
        restart: bool = False,
        idempotency_key: str | None = None,
        if_match: str | None = None,
    ) -> dict:
        """Write an app config section via
        `PUT /api/config/apps/{name}/{section}`.

        Carries `Idempotency-Key` (real writes) and `If-Match` (optimistic
        concurrency) headers and the `dry_run`/`restart` query flags.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if if_match:
            headers["If-Match"] = if_match
        params: dict[str, str] = {}
        if dry_run:
            params["dry_run"] = "true"
        if restart:
            params["restart"] = "true"
        return await self._request_json(
            "PUT",
            f"{self._base_url}/api/config/apps/{name}/{section}",
            json=values,
            params=params or None,
            headers=headers,
            timeout=10.0,
        )

    async def create_token(
        self,
        name: str,
        *,
        role: str = "submitter",
        max_gpus: int | None = None,
        max_concurrent_jobs: int | None = None,
        expires_in_s: int | None = None,
        scope_services: list[str] | None = None,
    ) -> dict:
        """Create a scoped token (admin-only), returning its plaintext exactly once.

        No idempotency key is sent because replay withholds plaintext. An omitted
        `expires_in_s` uses the configured default TTL; explicit JSON null would
        instead mean no expiry.
        """
        payload: dict = {"name": name, "role": role}
        if max_gpus is not None:
            payload["max_gpus"] = max_gpus
        if max_concurrent_jobs is not None:
            payload["max_concurrent_jobs"] = max_concurrent_jobs
        if expires_in_s is not None:
            payload["expires_in_s"] = expires_in_s
        if scope_services:
            payload["scope_services"] = scope_services
        return await self._request_json(
            "POST", f"{self._base_url}/api/tokens", json=payload, timeout=10.0
        )

    async def list_tokens(self, *, include_revoked: bool = False) -> list[dict]:
        """List API tokens via `GET /api/tokens` (admin-only).

        Hashes and plaintext are never returned by the daemon.
        """
        params = {"include_revoked": "true"} if include_revoked else None
        return await self._request_json(
            "GET", f"{self._base_url}/api/tokens", params=params, timeout=5.0
        )

    async def get_self_token(self) -> dict:
        """Read the caller's token (any role).

        Legacy-global and anonymous principals receive `id: null` and
        `rotatable: false`, rather than a 404.
        """
        return await self._request_json("GET", f"{self._base_url}/api/tokens/self", timeout=5.0)

    async def rotate_self_token(
        self, *, extend: bool = False, expires_in_s: int | None = None
    ) -> dict:
        """Rotate the caller's token and return its new plaintext exactly once.

        No idempotency key is sent: replay would withhold the new plaintext after
        invalidating the old secret.
        """
        payload: dict = {"extend": extend}
        if expires_in_s is not None:
            payload["expires_in_s"] = expires_in_s
        return await self._request_json(
            "POST", f"{self._base_url}/api/tokens/self/rotate", json=payload, timeout=10.0
        )

    async def revoke_token(self, token_id: str) -> dict:
        """Revoke a token via `DELETE /api/tokens/{id}` (admin-only)."""
        return await self._request_json(
            "DELETE", f"{self._base_url}/api/tokens/{token_id}", timeout=10.0
        )

    async def get_audit(
        self,
        *,
        action: str | None = None,
        result: str | None = None,
        target: str | None = None,
        target_type: str | None = None,
        principal_id: str | None = None,
        action_prefix: str | None = None,
        since: str | None = None,
        until: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Read an audit page (admin-only; other roles receive a structured 403).

        Action, result, target, target type and principal filters match exactly.
        `action_prefix` matches a literal prefix. `since` and `until` are inclusive
        ISO-8601 UTC bounds; `cursor` pages backwards.
        """
        params: dict[str, str | int] = {"limit": limit}
        if action:
            params["action"] = action
        if result:
            params["result"] = result
        if target:
            params["target"] = target
        if target_type:
            params["target_type"] = target_type
        if principal_id:
            params["principal_id"] = principal_id
        if action_prefix:
            params["action_prefix"] = action_prefix
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        if cursor:
            params["cursor"] = cursor
        return await self._request_json(
            "GET", f"{self._base_url}/api/audit", params=params, timeout=5.0
        )

    # --- Services (P2) -------------------------------------------------------
    #
    # The ``/services`` surface is mounted under ``/api`` only (no bare-root
    # alias), so every service call hits the ``/api`` prefix unlike the legacy
    # ``/jobs`` routes. Per Decision #1 there is no ``upload_service`` in P2 —
    # services are register-only over a prebuilt image.

    async def create_service(
        self,
        *,
        name: str,
        image: str,
        port: int = 8000,
        gpus: int = 0,
        restart_policy: str = "always",
        command: str | None = None,
        script_path: str | None = None,
        env: dict[str, str] | None = None,
        vendor: str | None = None,
        health_check: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Register a service via `POST /api/services` (register-only, prebuilt image).

        When `idempotency_key` is given it is sent as the `Idempotency-Key`
        header so a retried register collapses to one service row.
        """
        payload: dict = {
            "name": name,
            "image": image,
            "port": port,
            "gpus": gpus,
            "restart_policy": restart_policy,
        }
        if command:
            payload["command"] = command
        if script_path:
            payload["script_path"] = script_path
        if env:
            payload["env"] = env
        if vendor:
            payload["vendor"] = vendor
        if health_check:
            payload["health_check"] = health_check

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        return await self._request_json(
            "POST",
            f"{self._base_url}/api/services",
            json=payload,
            headers=headers,
            timeout=10.0,
        )

    async def list_services(
        self,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict:
        """List services via `GET /api/services` → a `ServiceListPage` dict."""
        params: dict[str, str | int] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return await self._request_json(
            "GET", f"{self._base_url}/api/services", params=params, timeout=5.0
        )

    async def get_service(self, ident: str) -> dict:
        """Fetch a single service by id or name via `GET /api/services/{ident}`."""
        return await self._request_json(
            "GET", f"{self._base_url}/api/services/{ident}", timeout=5.0
        )

    async def get_service_logs(
        self,
        ident: str,
        since_id: int = 0,
        tail: int | None = None,
        *,
        grep: str | None = None,
        since: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        """Fetch logs by service ID or name, applying filters before the tail limit.

        `grep` matches a literal substring; `since` is an ISO-8601 UTC lower bound.
        `tail` counts matching lines. `source` selects build or runtime logs;
        None omits the parameter for older daemons.
        """
        entries, _ = await self.get_service_logs_page(
            ident, since_id=since_id, tail=tail, grep=grep, since=since, source=source
        )
        return entries

    async def get_service_logs_page(
        self,
        ident: str,
        *,
        since_id: int = 0,
        tail: int | None = None,
        grep: str | None = None,
        since: str | None = None,
        source: str | None = None,
    ) -> tuple[list[dict], int | None]:
        """Fetch service logs with a scan watermark for advancing filtered followers.

        Returns:
            A log page and its `X-Nerdit-Scan-Watermark`, or None when absent.
            The watermark is the pre-scan maximum log ID on filtered forward pages;
            advance `since_id` to it even when no rows matched.
        """
        params: dict[str, int | str] = {"since_id": since_id}
        if tail is not None:
            params["tail"] = tail
        if grep:
            params["grep"] = grep
        if since:
            params["since"] = since
        if source:
            # Omitted entirely when unset (never ``source=all``): an unfiltered
            # read stays byte-identical on the wire, so a pre-P34 daemon that
            # would 422 on an unknown query param is never handed one.
            params["source"] = source
        async with self._client() as client:
            resp = await client.get(
                f"{self._base_url}/api/services/{ident}/logs",
                params=params,
                timeout=5.0,
            )
            resp.raise_for_status()
            raw = resp.headers.get("X-Nerdit-Scan-Watermark")
            try:
                watermark = int(raw) if raw is not None else None
            except ValueError:
                # A cursor is an optimization, never a correctness input: an
                # unparseable one is ignored, not raised at the user.
                watermark = None
            return resp.json(), watermark

    async def wait_for_service(
        self, ident: str, *, version: int | None = None, timeout: int = 60
    ) -> dict:
        """Wait for service convergence or failure.

        The server clamps `timeout` to 1–300 seconds and returns HTTP 200 with one
        of four outcomes. The client allows `timeout + 30` seconds.
        """
        params: dict[str, int] = {"timeout": timeout}
        if version is not None:
            params["version"] = version
        return await self._request_json(
            "GET",
            f"{self._base_url}/api/services/{ident}/wait",
            params=params,
            timeout=timeout + 30,
        )

    async def diagnose_service(self, ident: str, *, log_tail: int = 50) -> dict:
        """Fetch the failure bundle via `GET /api/services/{ident}/diagnose`.

        One bounded, owner-or-admin read: forensics + fresh health probe + fresh
        binding resolution + a DB-backed log tail + a single remediation code.
        """
        return await self._request_json(
            "GET",
            f"{self._base_url}/api/services/{ident}/diagnose",
            params={"log_tail": log_tail},
            timeout=10.0,
        )

    async def get_service_stats(self, ident: str) -> dict:
        """Fetch live resource usage (any authenticated principal).

        Allow extra time for Docker sampling, cached server-side for two seconds.
        Existing services return HTTP 200 even without a live container or Docker:
        check `available`, not zero values; unavailable stats are None.
        """
        return await self._request_json(
            "GET",
            f"{self._base_url}/api/services/{ident}/stats",
            timeout=15.0,
        )

    async def get_capabilities(self) -> dict:
        """Fetch the daemon self-knowledge projection (`GET /api/capabilities`).

        Role-aware: admin callers additionally receive the `paths` block and
        `proxy.admin_addr`. Pure read, no idempotency key.
        """
        return await self._request_json("GET", f"{self._base_url}/api/capabilities", timeout=10.0)

    async def restart_daemon(
        self, *, drain_timeout_s: int | None = None, idempotency_key: str | None = None
    ) -> dict:
        """Request a daemon restart (admin-only; requires an idempotency key).

        HTTP 202 precedes the background drain. With zero drain timeout, SIGTERM
        may beat the response: a transport error can mean the restart started.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        # Body only when the caller chose a value: an omitted key leaves the
        # server default (60 s) the server's, not a client-side copy of it.
        payload = None if drain_timeout_s is None else {"drain_timeout_s": drain_timeout_s}
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/daemon/restart",
            json=payload,
            headers=headers,
            timeout=10.0,
        )

    async def get_doctor(self) -> dict:
        """Fetch the structured health checks (`GET /api/doctor`).

        Each server-side check is bounded to 2 s (whole endpoint ≤ ~5 s), so a
        10 s client timeout comfortably covers a fully-loaded run.
        """
        return await self._request_json("GET", f"{self._base_url}/api/doctor", timeout=10.0)

    async def get_system_disk(self) -> dict:
        """Fetch disk usage (any authenticated principal).

        The 30-second timeout allows bounded directory walks over a large data dir.
        """
        return await self._request_json("GET", f"{self._base_url}/api/system/disk", timeout=30.0)

    async def run_system_gc(
        self,
        *,
        include_orphan_data: bool = False,
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """Reclaim orphan images (+ opt-in data dirs) via `POST /api/system/gc`.

        Admin-only. `dry_run=True` enumerates candidates with zero writes (no
        idempotency key needed); a real run honors the `Idempotency-Key` so a
        retry collapses to one gc.
        """
        params: dict[str, str] = {}
        if dry_run:
            params["dry_run"] = "true"
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/system/gc",
            params=params,
            json={"include_orphan_data": include_orphan_data},
            headers=headers,
            timeout=60.0,
        )

    async def create_backup(self, *, idempotency_key: str | None = None) -> dict:
        """Stage a control-plane backup (admin-only, idempotent).

        The 120-second timeout allows a large `VACUUM INTO` snapshot.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/system/backup",
            headers=headers,
            timeout=120.0,
        )

    async def create_volume_backup(
        self, service: str, *, idempotency_key: str | None = None
    ) -> dict:
        """Stage a per-database volume tar via `POST /api/system/backup/volumes`.

        Admin-only. Captures `<data_dir>/services/<service>/**` into a
        `nerdit-volumes-*` tar (database data + SCRAM verifiers, never the
        secrets master key). Honors an `Idempotency-Key`; the 120 s timeout
        covers a large data tree.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/system/backup/volumes",
            json={"service": service},
            headers=headers,
            timeout=120.0,
        )

    async def claim_link(
        self,
        *,
        code: str | None = None,
        key: str | None = None,
        api_url: str,
        relay_url: str | None = None,
        enable: bool = True,
        idempotency_key: str | None = None,
    ) -> dict:
        """Claim a cloud link grant (admin-only; requires an idempotency key).

        The daemon owns the node key and cloud request. Grants are secrets: this
        client sends them once, never logs or stores them, and receives no echo.

        Args:
            code: Console-minted `NL-` grant; supply exactly one of code or key.
            key: `nk_` pre-auth grant, sent in its own field without guessing type.
            api_url: Cloud URL for this invocation; never persisted.
            relay_url: Override only when supplied; omission keeps the stored URL.
        """
        if (code is None) == (key is None):
            # Mirrors the route's own exactly-one-of refusal so a programming
            # error here fails loudly in the caller's process instead of
            # burning a network round trip to be told the same thing.
            raise ValueError("claim_link takes exactly one of code= or key=")
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        payload: dict[str, object] = {"api_url": api_url, "enable": enable}
        if code is not None:
            payload["code"] = code
        else:
            payload["key"] = key
        if relay_url is not None:
            payload["relay_url"] = relay_url
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/link/claim",
            json=payload,
            headers=headers,
            # Headroom OVER the daemon's own 30 s cloud-exchange deadline:
            # this budget also covers waiting on the claim/unlink mutation
            # lock and response settlement, so it must be comfortably
            # larger or the CLI reports failure for a claim the daemon
            # then completes.
            timeout=90.0,
        )

    async def start_device_link(
        self,
        *,
        api_url: str,
        relay_url: str | None = None,
        enable: bool = True,
        idempotency_key: str | None = None,
    ) -> dict:
        """Start browser-approved linking (admin-only; requires an idempotency key).

        The secret device code stays in daemon memory. The response contains display
        material and an opaque `session` selector that grants no authority alone.
        `relay_url` is sent only when supplied and persisted only at commit.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        payload: dict[str, object] = {"api_url": api_url, "enable": enable}
        if relay_url is not None:
            payload["relay_url"] = relay_url
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/link/device",
            json=payload,
            headers=headers,
            # The start holds the link mutation lock across one bounded
            # 30 s cloud hop, exactly as the claim does — same headroom,
            # same reason: the CLI must not report
            # failure for a start the daemon then completes.
            timeout=90.0,
        )

    async def poll_device_link(
        self,
        *,
        session: str,
        idempotency_key: str | None = None,
    ) -> dict:
        """Poll one device-link session (admin-only).

        The caller must mint a fresh idempotency key per poll and manage cadence and
        timeout. Only the opaque session selector crosses this boundary; secrets
        stay daemon-side.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/link/device/poll",
            json={"session": session},
            headers=headers,
            # The poll's cloud hop is lock-free but still runs under the
            # daemon's 30 s deadline; an approved answer then takes the
            # mutation lock to commit. Both must fit comfortably, or the
            # CLI reports a failed poll for a link the daemon then
            # finishes — the same headroom rule as the claim.
            timeout=60.0,
        )

    async def unlink_node(self, *, idempotency_key: str | None = None) -> dict:
        """Revoke the cloud link (admin-only; requires an idempotency key).

        Drops the tunnel before wiping identity; never returns the key path.
        Already-unlinked nodes return HTTP 200 with `was_linked: false`.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/link",
            headers=headers,
            # Same headroom as the claim: an unlink queued behind a claim
            # waits on the mutation lock for up to the claim's 30 s cloud
            # deadline, and a DESTRUCTIVE call must not report failure for
            # work the daemon then completes.
            timeout=90.0,
        )

    async def refresh_link(self, *, api_url: str, idempotency_key: str | None = None) -> dict:
        """Refresh hosted-domain metadata (admin-only; requires an idempotency key).

        Only the non-secret hosted base domain is fetched. `api_url` is never
        persisted. The 90-second timeout covers the cloud deadline and mutation lock.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/link/refresh",
            json={"api_url": api_url},
            headers=headers,
            timeout=90.0,
        )

    async def get_share(self, name: str) -> dict:
        """Read a service's hosted share via `GET /api/services/{name}/share`.

        `404 share.not_shared` when the service exists but is not shared — the
        caller renders that refusal rather than a made-up "private" default.
        """
        return await self._request_json(
            "GET", f"{self._base_url}/api/services/{name}/share", timeout=10.0
        )

    async def set_share(
        self,
        name: str,
        *,
        access: str = "private",
        consent: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """Share a service; requires an idempotency key.

        The daemon validates entitlement, consent, link state, service kind and
        label length. Send values unchanged so client validation cannot drift.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "PUT",
            f"{self._base_url}/api/services/{name}/share",
            json={"access": access, "consent": consent},
            headers=headers,
            timeout=15.0,
        )

    async def remove_share(self, name: str, *, idempotency_key: str | None = None) -> dict:
        """Unshare a service via `DELETE /api/services/{name}/share`.

        Idempotent by design: an unshared service answers `200` with
        `removed: false`, never a `404`, so a retried teardown converges.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/services/{name}/share",
            headers=headers,
            timeout=15.0,
        )

    async def list_domains(self, name: str) -> dict:
        """List direct domains; a service with none returns HTTP 200 and an empty list."""
        return await self._request_json(
            "GET", f"{self._base_url}/api/services/{name}/domains", timeout=10.0
        )

    async def add_domain(
        self,
        name: str,
        domain: str,
        *,
        acme: bool | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Add a direct domain; requires an idempotency key.

        Send the percent-encoded domain as typed; normalization and validation belong
        to the daemon. Omit `acme` when None to preserve the stored certificate
        setting rather than downgrade an existing public certificate.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "PUT",
            f"{self._base_url}/api/services/{name}/domains/{_encode_dot_segment(domain)}",
            json={} if acme is None else {"acme": acme},
            headers=headers,
            timeout=15.0,
        )

    async def remove_domain(
        self, name: str, domain: str, *, idempotency_key: str | None = None
    ) -> dict:
        """Remove a direct domain via `DELETE /api/services/{name}/domains/{domain}`.

        Idempotent by design (the `remove_share` rule): an unknown domain
        answers `200` with `removed: false`, never a `404`, so a retried
        teardown converges.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/services/{name}/domains/{_encode_dot_segment(domain)}",
            headers=headers,
            timeout=15.0,
        )

    async def install_license(self, blob: str, *, idempotency_key: str | None = None) -> dict:
        """Install a license on the daemon (admin-only; requires an idempotency key).

        The secret blob travels only in the JSON body and is neither logged nor
        stored locally, including when the daemon is remote.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/license",
            json={"blob": blob},
            headers=headers,
            timeout=30.0,
        )

    async def remove_license(self, *, idempotency_key: str | None = None) -> dict:
        """Remove the installed license via `DELETE /api/license`.

        Admin-only, idempotent by design (nothing installed answers `200` with
        `removed: false`), and the daemon requires an in-route
        `Idempotency-Key` like the install.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/license",
            headers=headers,
            timeout=30.0,
        )

    async def resolve_service(self, ident: str) -> dict | None:
        """Resolve a service ID or name; return None on 404 and propagate other errors."""
        async with self._client() as client:
            resp = await client.get(f"{self._base_url}/api/services/{ident}", timeout=5.0)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()

    async def stop_service(self, ident: str, *, idempotency_key: str | None = None) -> dict:
        """Stop a service via `POST /api/services/{ident}/stop` (desired_state → stopped)."""
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST", f"{self._base_url}/api/services/{ident}/stop", headers=headers, timeout=10.0
        )

    async def restart_service(self, ident: str, *, idempotency_key: str | None = None) -> dict:
        """Restart a service via `POST /api/services/{ident}/restart` (clears backoff)."""
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST", f"{self._base_url}/api/services/{ident}/restart", headers=headers, timeout=10.0
        )

    async def remove_service(
        self,
        ident: str,
        *,
        purge: str = "secrets",
        force: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """Delete a service and synchronously tear down its resources.

        `purge` is a CSV of secrets/data/images targets; `force` bypasses the model
        reference guard. Allow 120 seconds for container, directory and image removal.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        params: dict[str, str] = {"purge": purge}
        if force:
            params["force"] = "true"
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/services/{ident}",
            headers=headers,
            params=params,
            timeout=120.0,
        )

    async def run_service_command(
        self,
        ident: str,
        *,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout_s: int = 300,
        log_tail: int = 200,
        idempotency_key: str | None = None,
    ) -> dict:
        """Run a bounded one-off command in the service's current image.

        The daemon creates no workload row, kills the container at `timeout_s`, and
        returns HTTP 200 even for nonzero command exits. Environment overrides beat
        config and secrets, but not platform values (`PORT`, AI/DB bindings and
        `NERDIT_RUN_ID`). The caller supplies the idempotency key; the client timeout
        adds 30 seconds for container setup, kill and log collection.
        """
        payload: dict = {
            "command": command,
            "timeout_s": timeout_s,
            "log_tail": log_tail,
        }
        if env is not None:
            payload["env"] = env
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/services/{ident}/run",
            json=payload,
            headers=headers,
            timeout=timeout_s + 30,
        )

    # --- Deploy + Secrets (P4) ------------------------------------------------
    #
    # ``/deploy`` and ``/secrets`` are mounted under ``/api`` only. Deploy is
    # multipart (folder ZIP + form fields, like ``upload_job``); a fresh
    # ``Idempotency-Key`` is required so a retried deploy collapses to one
    # build. Secrets are write-only: no method ever returns a secret value.

    async def deploy(
        self,
        *,
        zip_bytes: bytes,
        name: str,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        env: dict[str, str | None] | None = None,
        vendor: str | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Upload an app ZIP and deploy it; an existing name triggers redeployment.

        `env` values of None preserve null-delete semantics. Unset optional fields
        use daemon defaults. Dry runs return a plan (HTTP 200), write nothing and
        remain keyless. Real deploys mint a missing idempotency key to prevent
        duplicate builds on retry.
        """
        import json
        from uuid import uuid4

        if not dry_run and not idempotency_key:
            idempotency_key = uuid4().hex

        files = {"archive": ("app.zip", zip_bytes, "application/zip")}
        data: dict[str, str] = {"name": name}
        if port is not None:
            data["port"] = str(port)
        if gpus is not None:
            data["gpus"] = str(gpus)
        if start:
            data["start"] = start
        if health:
            data["health"] = health
        if env:
            data["env"] = json.dumps(env)
        if vendor:
            data["vendor"] = vendor

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        async with self._client() as client:
            resp = await client.post(
                f"{self._base_url}/api/deploy",
                files=files,
                data=data,
                params={"dry_run": "true"} if dry_run else None,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def rollback_deploy(self, name: str, idempotency_key: str) -> dict:
        """Roll back a deploy via `POST /api/deploy/{name}/rollback`."""
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/deploy/{name}/rollback",
            headers={"Idempotency-Key": idempotency_key},
            timeout=30.0,
        )

    async def redeploy_app(
        self, name: str, *, idempotency_key: str | None = None, dry_run: bool = False
    ) -> dict:
        """Redeploy the recorded Git source and ref without a request body.

        Real redeploys mint a missing idempotency key; dry runs remain keyless.
        The timeout matches `deploy_git` to allow the server-side clone.
        """
        from uuid import uuid4

        if not dry_run and not idempotency_key:
            idempotency_key = uuid4().hex
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/deploy/{name}/redeploy",
            params={"dry_run": "true"} if dry_run else None,
            headers=headers,
            timeout=180.0,
        )

    # Git deployment uses API-only JSON routes and a longer timeout for cloning.
    # Fresh idempotency keys deduplicate retries; template reads use the embedded catalog.

    async def deploy_git(
        self,
        *,
        repo_url: str,
        name: str,
        ref: str | None = None,
        subdir: str | None = None,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        env: dict[str, str | None] | None = None,
        vendor: str | None = None,
        token_ref: str | None = None,
        idempotency_key: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Deploy a Git URL; an existing name triggers redeployment.

        Unset fields use daemon defaults. `token_ref` is a `${secrets.KEY}` reference
        resolved server-side, never a raw credential. Allow extra time for cloning.
        Dry runs return a plan (HTTP 200), write nothing and remain keyless; real
        deploys mint a missing idempotency key to prevent duplicate builds.
        """
        from uuid import uuid4

        if not dry_run and not idempotency_key:
            idempotency_key = uuid4().hex

        payload: dict = {"repo_url": repo_url, "name": name}
        if ref is not None:
            payload["ref"] = ref
        if subdir is not None:
            payload["subdir"] = subdir
        if port is not None:
            payload["port"] = port
        if gpus is not None:
            payload["gpus"] = gpus
        if start is not None:
            payload["start"] = start
        if health is not None:
            payload["health"] = health
        if env:
            payload["env"] = env
        if vendor is not None:
            payload["vendor"] = vendor
        if token_ref is not None:
            payload["token_ref"] = token_ref

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        return await self._request_json(
            "POST",
            f"{self._base_url}/api/deploy/git",
            json=payload,
            params={"dry_run": "true"} if dry_run else None,
            headers=headers,
            timeout=180.0,
        )

    async def list_app_templates(self) -> list[dict]:
        """List the app template catalog via `GET /api/app-templates`."""
        return await self._request_json("GET", f"{self._base_url}/api/app-templates", timeout=5.0)

    async def get_app_template(self, template_id: str) -> dict:
        """Fetch a single app template via `GET /api/app-templates/{template_id}`."""
        return await self._request_json(
            "GET", f"{self._base_url}/api/app-templates/{template_id}", timeout=5.0
        )

    async def deploy_template(
        self,
        template_id: str,
        *,
        name: str,
        env: dict[str, str | None] | None = None,
        secrets: dict[str, str] | None = None,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        vendor: str | None = None,
        idempotency_key: str,
    ) -> dict:
        """Deploy an app template via `POST /api/app-templates/{template_id}/deploy`.

        `secrets` are written write-only server-side before the row write (key
        names only ever come back). Unset optional fields are omitted so the
        template defaults / daemon defaults apply. The timeout matches
        `deploy_git` — the deploy clones the catalog repo server-side.
        """
        payload: dict = {"name": name}
        if env:
            payload["env"] = env
        if secrets:
            payload["secrets"] = secrets
        if port is not None:
            payload["port"] = port
        if gpus is not None:
            payload["gpus"] = gpus
        if start is not None:
            payload["start"] = start
        if health is not None:
            payload["health"] = health
        if vendor is not None:
            payload["vendor"] = vendor

        return await self._request_json(
            "POST",
            f"{self._base_url}/api/app-templates/{template_id}/deploy",
            json=payload,
            headers={"Idempotency-Key": idempotency_key},
            timeout=180.0,
        )

    # Workspace methods are library-only and mounted under /api.
    # Encode names and file-path components separately, preserving structural slashes.
    # Encode all-dot segments too: httpx would otherwise normalize them before the
    # daemon could reject an invalid path, potentially returning a different file.
    # Validation remains server-side.

    async def write_workspace_files(
        self,
        name: str,
        files: dict[str, str],
        delete: list[str] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> dict:
        """Batch write/delete workspace files via `PUT /api/workspaces/{name}/files`.

        All-or-nothing server-side: one invalid entry rejects the whole batch,
        so a retry is always of a complete batch. A key is minted when absent —
        this is a write, and a network retry must not double-apply.
        """
        from uuid import uuid4

        if not idempotency_key:
            idempotency_key = uuid4().hex
        return await self._request_json(
            "PUT",
            f"{self._base_url}/api/workspaces/{_encode_dot_segment(name)}/files",
            json={"files": files, "delete": delete or []},
            headers={"Idempotency-Key": idempotency_key},
            timeout=30.0,
        )

    async def get_workspace(self, name: str) -> dict:
        """List a workspace via `GET /api/workspaces/{name}` (owner-or-admin)."""
        return await self._request_json(
            "GET", f"{self._base_url}/api/workspaces/{_encode_dot_segment(name)}", timeout=10.0
        )

    async def read_workspace_file(self, name: str, path: str) -> str:
        """Read a workspace text file, preserving path segments on the wire.

        Encode each segment separately: slashes remain structural, `#`, `?` and `%`
        remain data, and all-dot segments cannot be normalized away by httpx.
        The daemon rejects illegal decoded paths with a structured 422.
        """
        async with self._client() as client:
            resp = await client.get(
                f"{self._base_url}/api/workspaces/{_encode_dot_segment(name)}"
                f"/files/{'/'.join(_encode_dot_segment(seg) for seg in path.split('/'))}",
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.text

    async def deploy_workspace(
        self,
        name: str,
        *,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        env: dict[str, str | None] | None = None,
        vendor: str | None = None,
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """Deploy a workspace using daemon defaults for unset fields.

        The timeout matches `deploy_git` for snapshot and build scheduling. Dry runs
        return a plan (HTTP 200), write nothing and remain keyless.
        """
        from uuid import uuid4

        if not dry_run and not idempotency_key:
            idempotency_key = uuid4().hex

        payload: dict = {}
        if port is not None:
            payload["port"] = port
        if gpus is not None:
            payload["gpus"] = gpus
        if start is not None:
            payload["start"] = start
        if health is not None:
            payload["health"] = health
        if env:
            payload["env"] = env
        if vendor is not None:
            payload["vendor"] = vendor

        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        return await self._request_json(
            "POST",
            f"{self._base_url}/api/workspaces/{_encode_dot_segment(name)}/deploy",
            json=payload,
            params={"dry_run": "true"} if dry_run else None,
            headers=headers,
            timeout=180.0,
        )

    async def set_secrets(
        self,
        service: str,
        values: dict[str, str],
        *,
        idempotency_key: str | None = None,
    ) -> dict:
        """Set/merge secrets via `POST /api/secrets/{service}` → key names only.

        When `idempotency_key` is given it is sent as the `Idempotency-Key`
        header so a retried write collapses to a single apply.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/secrets/{service}",
            json={"values": values},
            headers=headers,
            timeout=10.0,
        )

    async def list_secrets(self, service: str) -> dict:
        """List secret key names via `GET /api/secrets/{service}` (never values)."""
        return await self._request_json(
            "GET", f"{self._base_url}/api/secrets/{service}", timeout=5.0
        )

    async def delete_secret(
        self, service: str, key: str, *, idempotency_key: str | None = None
    ) -> dict:
        """Delete one secret key via `DELETE /api/secrets/{service}/{key}`."""
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/secrets/{service}/{key}",
            headers=headers,
            timeout=10.0,
        )

    async def delete_secrets(self, service: str, *, idempotency_key: str | None = None) -> dict:
        """Delete all of a service's secrets via `DELETE /api/secrets/{service}`."""
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "DELETE",
            f"{self._base_url}/api/secrets/{service}",
            headers=headers,
            timeout=10.0,
        )

    async def rotate_secrets_key(self, *, idempotency_key: str | None = None) -> dict:
        """Rotate the secrets-at-rest key via `POST /api/secrets/rotate-key`.

        Admin-only; re-encrypts every stored secret file under a fresh key and
        returns `{"services_rewritten": n}` — counts only, never key material.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/secrets/rotate-key",
            headers=headers,
            timeout=30.0,
        )

    # --- Models (P5) -----------------------------------------------------------
    #
    # ``/models`` is mounted under ``/api`` only. POST creates the ``kind=model``
    # workload; lifecycle writes (stop/restart/rm) go through the ``/services``
    # methods above (one lifecycle surface).

    async def serve_model(
        self,
        model: str,
        gpus: int = 0,
        name: str | None = None,
        backend: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Serve a model with the selected backend or the daemon's default.

        `max_model_len` and `gpu_memory_utilization` require vLLM; other backends
        return 422 `model.backend_param`. A supplied idempotency key deduplicates
        retries.
        """
        payload: dict = {"model": model, "gpus": gpus}
        if name:
            payload["name"] = name
        if backend:
            payload["backend"] = backend
        # ``is not None`` (not truthiness): 0-adjacent numerics must still ride.
        if max_model_len is not None:
            payload["max_model_len"] = max_model_len
        if gpu_memory_utilization is not None:
            payload["gpu_memory_utilization"] = gpu_memory_utilization
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/models",
            json=payload,
            headers=headers,
            timeout=10.0,
        )

    async def list_models(self, limit: int = 50, cursor: str | None = None) -> dict:
        """List served models via `GET /api/models` → a `ModelListPage` dict."""
        params: dict[str, str | int] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request_json(
            "GET", f"{self._base_url}/api/models", params=params, timeout=5.0
        )

    # Database creation is API-only; passwords are minted server-side and never returned.
    # Lifecycle operations reuse the service methods.

    async def create_database(
        self,
        backend: str | None = None,
        name: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Provision a database with the selected backend or the daemon's default.

        A supplied idempotency key deduplicates retries. Responses contain secret
        key names only, never the generated password.
        """
        payload: dict = {}
        if backend:
            payload["backend"] = backend
        if name:
            payload["name"] = name
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return await self._request_json(
            "POST",
            f"{self._base_url}/api/databases",
            json=payload,
            headers=headers,
            timeout=10.0,
        )

    async def list_databases(self, limit: int = 50, cursor: str | None = None) -> dict:
        """List managed databases via `GET /api/databases` → a `DatabaseListPage` dict."""
        params: dict[str, str | int] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request_json(
            "GET", f"{self._base_url}/api/databases", params=params, timeout=5.0
        )

    # --- Database dumps (P37, §1.6) -------------------------------------------
    #
    # Metadata only: the tar is never served over HTTP, and no method here
    # sees a database password. ``timeout_s`` defaults mirror the route's (not
    # imported: ``nerdit.cli`` does not import daemon schemas; the server cap
    # binds either way). Neither synchronous call carries a client-side READ
    # timeout (``httpx.Timeout(10.0, read=None)``): the pack/extract phases
    # are data-proportional and outside ``timeout_s``, every real bound is
    # server-owned, a dead daemon closes the socket, and a client abort would
    # only lead to a retry running a second dump. Amends D-P37-8's
    # ``timeout_s + 60`` (plan §9).

    async def create_database_dump(
        self,
        name: str,
        *,
        timeout_s: int = 900,
        idempotency_key: str | None = None,
    ) -> dict:
        """Capture a logical dump via ``POST /api/databases/{name}/dump`` (P37, D-P37-8).

        Bounded and synchronous; the response is metadata only. The httpx READ timeout
        is disabled (connect/write/pool stay 10 s): container start and the pack sit
        outside ``timeout_s`` and scale with the data, and a client abort would only
        lead to a retry running a second dump. ``idempotency_key`` is mandatory
        in-route; the command layer mints one per invocation.
        """
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        async with self._client() as client:
            resp = await client.post(
                f"{self._base_url}/api/databases/{name}/dump",
                json={"timeout_s": timeout_s},
                headers=headers,
                timeout=httpx.Timeout(10.0, read=None),
            )
            resp.raise_for_status()
            return resp.json()

    async def list_database_dumps(self, name: str) -> dict:
        """List a database's dump tars via ``GET /api/databases/{name}/dumps`` (P37).

        Names, sizes and mtimes only — the daemon never opens an archive to
        answer this, so it costs one directory scan whatever the tars weigh.
        Newest first. Same owner-or-admin + scope gate as the dump that produced
        them: enumerating a database's dumps is knowing something about its data.
        """
        async with self._client() as client:
            resp = await client.get(
                f"{self._base_url}/api/databases/{name}/dumps",
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def restore_database_dump(
        self,
        name: str,
        dump: str,
        *,
        timeout_s: int = 600,
        force: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        """Restore a dump via ``POST /api/databases/{name}/restore``. Admin-only and
        destructive. ``dump`` is a basename (a path is unrepresentable in the route's
        pattern). ``force=True`` proceeds past ``409 restore.in_use``. Read timeout
        disabled for the dump's reason.
        """
        params: dict[str, str] = {}
        if force:
            params["force"] = "true"
        headers: dict[str, str] = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        async with self._client() as client:
            resp = await client.post(
                f"{self._base_url}/api/databases/{name}/restore",
                json={"dump": dump, "timeout_s": timeout_s},
                params=params,
                headers=headers,
                timeout=httpx.Timeout(10.0, read=None),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_proxy_status(self) -> dict:
        """Fetch the embedded-proxy status via `GET /api/proxy/status`.

        Authenticated (unlike `/proxy/ca`); returns the `ProxyStatusResponse`
        dict (state, tls/ca/apex/respawn/routes/mdns projections).
        """
        return await self._request_json("GET", f"{self._base_url}/api/proxy/status", timeout=10.0)

    async def list_routes(self, *, cursor: str | None = None, limit: int | None = None) -> dict:
        """List proxy routes via `GET /api/routes` → a `RouteListPage` dict."""
        params: dict[str, str | int] = {}
        if cursor:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return await self._request_json(
            "GET", f"{self._base_url}/api/routes", params=params, timeout=10.0
        )

    async def list_events(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        types: str | None = None,
        service: str | None = None,
        since_id: int | None = None,
    ) -> dict:
        """Page the durable event feed in one direction.

        `cursor` browses backwards (descending IDs); `since_id` replays forwards
        (ascending IDs). Supplying both returns 400. The server clamps `limit`
        to 1–200.
        """
        params: dict[str, str | int] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor:
            params["cursor"] = cursor
        if types:
            params["types"] = types
        if service:
            params["service"] = service
        if since_id is not None:
            params["since_id"] = since_id
        return await self._request_json(
            "GET", f"{self._base_url}/api/events", params=params, timeout=10.0
        )

    async def stream_events(self, *, last_event_id: int | None = None) -> AsyncIterator[dict]:
        """Yield JSON-object SSE frames, skipping malformed payloads.

        `last_event_id` requests durable replay before the live tail; `feed.gap`
        arrives first when exact replay is unavailable. There is no read timeout:
        the caller owns cancellation.
        """
        headers = {"Last-Event-ID": str(last_event_id)} if last_event_id is not None else {}
        async with (
            self._client() as client,
            client.stream(
                "GET",
                f"{self._base_url}/api/events/stream",
                headers=headers,
                timeout=httpx.Timeout(10.0, read=None),
            ) as resp,
        ):
            if resp.status_code >= 400:
                await resp.aread()
                resp.raise_for_status()
            payload: list[str] = []
            async for raw in resp.aiter_lines():
                line = raw.rstrip("\r")
                if not line:
                    frame = _decode_sse_data(payload)
                    payload = []
                    if frame is not None:
                        yield frame
                    continue
                if line.startswith(":"):
                    continue
                field, _, value = line.partition(":")
                if field == "data":
                    payload.append(value[1:] if value.startswith(" ") else value)

    async def get_proxy_ca(self) -> str:
        """Fetch the public internal-CA certificate as raw PEM.

        Ignore the transport fingerprint header: an attacker can replace both.
        `nerdit trust` recomputes the fingerprint from the certificate body.
        """
        async with self._client() as client:
            resp = await client.get(f"{self._base_url}/api/proxy/ca", timeout=10.0)
            resp.raise_for_status()
            return resp.text


def get_configured_client() -> NerditClient:
    """Create a client from `~/.nerdit/config.toml`.

    Use the configured remote host and token, otherwise localhost.
    """
    from nerdit.config.settings import get_client_config

    host, port, token = get_client_config()
    return NerditClient(host=host, port=port, token=token)
