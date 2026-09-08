"""Read and update Caddy admin configuration through one shared async HTTP client."""

from __future__ import annotations

from typing import Any

import httpx

from .routing import (
    _APEX_ID,
    _ID_PREFIX,
    LiveRoute,
    _extract_auth_fingerprint,
    _extract_dial,
    _route_shape,
)


class CaddyAdmin:
    """Thin async wrapper over the Caddy admin API.

    Owns one shared `httpx.AsyncClient` (5s timeout). All routing config
    is read/written here; nothing else talks to Caddy. Tests inject an
    `httpx.MockTransport` via `transport` so the wrapper is exercised with no
    real Caddy.
    """

    def __init__(
        self,
        admin_addr: str,
        *,
        timeout: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = f"http://{admin_addr}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_idempotent(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Issue an idempotent admin request, retrying once on a dead keep-alive.

        Callers MUST only use this for admin verbs that are safe to replay
        (`DELETE` and a `PATCH`-replace) — never for the `POST` append leg
        of `upsert_route`, where a replay would create a duplicate
        `@id` twin. The shared `httpx.AsyncClient` keeps its
        connection alive across calls; on a real Caddy 2.6.2 that connection
        occasionally dies between reconcile ticks and the first request to
        reuse it raises `httpx.RemoteProtocolError` (the B21 flake,
        observed flaky on CI #518). One retry opens a fresh connection and
        replays the same idempotent verb; a second failure propagates.
        """
        try:
            return await self._client.request(method, url, **kwargs)
        except httpx.RemoteProtocolError:
            return await self._client.request(method, url, **kwargs)

    async def ping(self) -> bool:
        """Return `True` when the admin API answers (Caddy is up)."""
        try:
            resp = await self._client.get("/config/")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def has_nerdit_server(self) -> bool:
        """Return `True` when our `nerdit` HTTP server block is loaded."""
        return await self.has_server("nerdit")

    async def has_server(self, name: str) -> bool:
        """Return `True` when the HTTP server block `name` is loaded.

        Fail-**closed** on every error class (transport, non-200, unparseable
        body): the callers use this to decide whether a listener they need
        actually exists, and "we could not tell" must never read as "it is
        there". Generalised from `has_nerdit_server` for P26 WP2's
        `nerdit-acme-http` listener, which an ADOPTED Caddy from a pre-ACME
        run does not carry (review round 1).

        Thin fail-closed projection of `server_present`, which is what a
        caller that LATCHES the answer wants instead (review round 2).
        """
        return (await self.server_present(name)) is True

    async def server_present(self, name: str) -> bool | None:
        """Tri-state read of the HTTP server block `name`.

        `True` the block is loaded, `False` Caddy says it is **not**, and
        `None` we could not tell (transport error, non-200 other than 404,
        unparseable body) — the same tolerance `get_tls_config` draws, and
        for the same reason: a transient read failure must be distinguishable
        from a factual answer, otherwise a caller that caches the result caches
        a blip forever (review round 2).

        Caddy's factual "that key is absent" answer for a config sub-path is
        `200` with a JSON `null` body — measured live on v2.11.4, which is
        what makes `False` a real signal rather than an inference. A `404`
        is treated the same way for tolerance's sake; every other non-200 is
        the server failing to answer, not answering "no".
        """
        try:
            resp = await self._client.get(f"/config/apps/http/servers/{name}")
        except httpx.HTTPError:
            return None
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            return None
        try:
            return resp.json() is not None
        except ValueError:
            return None

    async def load(self, config: dict[str, Any]) -> None:
        """Replace Caddy's whole config (own process only, at bootstrap)."""
        resp = await self._client.post("/load", json=config)
        resp.raise_for_status()

    async def insert_route_first(self, obj: dict[str, Any]) -> None:
        """Insert a route object at index **0** of the `nerdit` routes array (P26 F1).

        `PUT /config/apps/http/servers/nerdit/routes/0` inserts *before* the
        element currently at that index rather than replacing it — verified live
        on Caddy v2.11.4 (spec fact F1), including against an empty array — and
        the inserted object's embedded `@id` is registered in the id map like
        any other (F2), so the very next `upsert_route` for that id takes
        the `PATCH /id` leg and replaces it **in place**, leaving the array
        position untouched (F3).

        Deliberately a bare client call rather than `_request_idempotent`:
        a replayed insert creates a second object carrying the same `@id`
        (a twin), exactly the hazard that keeps the `POST` append leg of
        `upsert_route` off the retry helper too.
        """
        resp = await self._client.put("/config/apps/http/servers/nerdit/routes/0", json=obj)
        resp.raise_for_status()

    async def upsert_route(self, obj: dict[str, Any], *, prepend: bool = False) -> None:
        """Replace a route by ID, inserting it only when absent.

        Use PATCH for replacement: POST to an ID appends a sibling and leaves a stale
        matching route. Prepend missing custom-domain routes so defaults cannot shadow
        them; append missing defaults. Patching preserves existing array position.
        """
        route_id = obj["@id"]
        resp = await self._request_idempotent("PATCH", f"/id/{route_id}", json=obj)
        if resp.status_code in (200, 201):
            return
        # Append ONLY when the id is genuinely absent — Caddy reports that as a
        # 404 or a 500 whose body says "unknown object". Any other failure
        # (e.g. a transient 5xx on an id that DOES exist) must NOT append, or we'd
        # create a second route object carrying the same `@id`.
        absent = resp.status_code == 404 or (
            resp.status_code >= 400 and "unknown object" in resp.text.lower()
        )
        if not absent:
            resp.raise_for_status()
            return
        if prepend:
            await self.insert_route_first(obj)
            return
        resp = await self._client.post("/config/apps/http/servers/nerdit/routes", json=obj)
        resp.raise_for_status()

    async def delete_route(self, route_id: str) -> bool:
        """Delete a route by `@id`; return `True` only if it existed.

        Idempotent: an absent id is reported by Caddy as **404** *or* a **500**
        whose body contains `"unknown object"` — both are treated as success
        (already gone) and return `False` so callers can skip the audit row.
        """
        resp = await self._request_idempotent("DELETE", f"/id/{route_id}")
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        if resp.status_code >= 400 and "unknown object" in resp.text.lower():
            return False
        resp.raise_for_status()
        return False

    async def live_routes(self) -> dict[str, LiveRoute] | None:
        """Read owned routes with dial, shape, auth fingerprint, and array index.

        For duplicate IDs, report the last occurrence, matching Caddy's ID map. The
        index exposes domain routes shadowed by earlier defaults.

        Returns:
            ID-to-LiveRoute mapping, empty if no routes exist; None if unreadable.
            Callers must distinguish failure from absence to avoid spurious rewrites.
        """
        try:
            resp = await self._client.get("/config/apps/http/servers/nerdit/routes")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            routes = resp.json()
        except ValueError:
            return None
        if not isinstance(routes, list):
            return None
        result: dict[str, LiveRoute] = {}
        for idx, route_obj in enumerate(routes):
            if not isinstance(route_obj, dict):
                continue
            rid = route_obj.get("@id")
            if not (isinstance(rid, str) and rid.startswith(_ID_PREFIX)):
                continue
            prev = result.get(rid)
            result[rid] = LiveRoute(
                index=idx,
                dial=_extract_dial(route_obj),
                shape=_route_shape(route_obj),
                count=prev.count + 1 if prev else 1,
                # The auth dimension is read here or nowhere: the
                # classifier compares this against the desired fingerprint, so a
                # route whose handler was stripped by hand — or by an older
                # daemon — reads as drift instead of as a converged open route.
                auth_fingerprint=_extract_auth_fingerprint(route_obj),
            )
        return result

    async def dedupe_route(self, route_id: str) -> int:
        """Delete duplicate live routes carrying `route_id`, keeping the newest.

        Stale twins cannot be addressed via `/id/` — the id map resolves to
        the newest copy only — so they are deleted by ARRAY INDEX, descending so
        the remaining indexes stay valid. The last occurrence (the id-map
        target, the one `live_routes` reported) is kept; the normal drift
        logic then converges it if needed. Returns the number removed.
        """
        resp = await self._client.get("/config/apps/http/servers/nerdit/routes")
        resp.raise_for_status()
        routes = resp.json()
        if not isinstance(routes, list):
            return 0
        indexes = [
            i
            for i, obj in enumerate(routes)
            if isinstance(obj, dict) and obj.get("@id") == route_id
        ]
        removed = 0
        for idx in sorted(indexes[:-1], reverse=True):
            resp = await self._client.delete(f"/config/apps/http/servers/nerdit/routes/{idx}")
            resp.raise_for_status()
            removed += 1
        return removed

    async def append_route(self, obj: dict[str, Any]) -> None:
        """Append a route object to the END of the `nerdit` server routes array.

        The apex reconcile uses this to land the catch-all route LAST —
        after every service route — so Caddy's top-to-bottom evaluation reaches
        the service routes first and the apex shadows nothing. Reuses the exact
        `POST …/routes` path `upsert_route` appends an absent id with.
        """
        resp = await self._client.post("/config/apps/http/servers/nerdit/routes", json=obj)
        resp.raise_for_status()

    async def apex_state(self) -> tuple[bool, bool, str | None, int] | None:
        """Read apex presence, order, dial, and duplicate count from the raw route array.

        The service-only live view excludes this ID. Caddy uses first-match order, so
        an earlier duplicate catch-all can shadow services even when the last copy
        looks correct. Callers must collapse duplicates before declaring convergence.

        Returns:
            `(present, is_last, dial, count)`, describing the last occurrence as Caddy's
            ID map does. None means unreadable, not absent; skip convergence then.
        """
        try:
            resp = await self._client.get("/config/apps/http/servers/nerdit/routes")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            routes = resp.json()
        except ValueError:
            return None
        if not isinstance(routes, list):
            return None
        present = False
        is_last = False
        dial: str | None = None
        count = 0
        for idx, route_obj in enumerate(routes):
            if isinstance(route_obj, dict) and route_obj.get("@id") == _APEX_ID:
                present = True
                is_last = idx == len(routes) - 1
                dial = _extract_dial(route_obj)
                count += 1
        return present, is_last, dial, count

    async def get_tls_config(self) -> dict[str, Any] | None:
        """Return the live `tls` app config, or `None` when it can't be read.

        Same failure tolerance as `live_routes`: `None` (NOT `{}`) on a
        transport error, a non-200, or an unparseable/non-dict body — a transient
        read failure must be distinguishable from a genuinely different config,
        otherwise the caller would re-push (and re-issue certs) on every blip.
        """
        try:
            resp = await self._client.get("/config/apps/tls")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            cfg = resp.json()
        except ValueError:
            return None
        if not isinstance(cfg, dict):
            return None
        return cfg

    async def set_tls_config(self, cfg: dict[str, Any]) -> None:
        """Replace the `tls` app subtree only — routes untouched, never `/load`."""
        resp = await self._client.post("/config/apps/tls", json=cfg)
        resp.raise_for_status()

    async def get_ca(self) -> str | None:
        """Return the internal-CA root certificate PEM, or `None`.

        `GET /pki/ca/local` answers with a JSON object whose
        `root_certificate` field is the root PEM (verified on Caddy 2.6.2;
        the GET lazily provisions the CA if it does not exist yet). Same
        failure tolerance as the other reads: any transport/parse problem is
        `None` — the caller falls back to the on-disk copy.
        """
        try:
            resp = await self._client.get("/pki/ca/local")
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        pem = data.get("root_certificate")
        if not isinstance(pem, str) or not pem.strip():
            return None
        return pem
