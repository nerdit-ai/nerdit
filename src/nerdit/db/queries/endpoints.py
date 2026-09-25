"""Durable port reservations, endpoint inventory and proxy-route writes."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime

from nerdit.config.defaults import DEFAULT_PORT
from nerdit.db.models import JobKind, JobStatus, RouteEndpoint, ServiceEndpoint

from ._base import (
    PortRangeExhausted,
    QueriesBase,
    _decode_name_cursor,
    _encode_name_cursor,
    _serialized,
)


class EndpointQueries(QueriesBase):
    """Stable host-port reservation + bounded route inventory + route writes."""

    async def list_service_endpoints(
        self, *, cursor: str | None = None, limit: int = 50
    ) -> tuple[list[RouteEndpoint], str | None]:
        """Page all endpoint rows with owning kind/status, including unrouted and terminal rows.

        Use ascending service-name cursors; clamp limit to 1–200.

        Raises:
            ValueError: The cursor is malformed.
        """
        limit = max(1, min(limit, 200))
        clauses: list[str] = []
        params: list[object] = []
        if cursor:
            clauses.append("e.service_name > ?")
            params.append(_decode_name_cursor(cursor))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT e.service_name AS service_name, j.kind AS kind, j.status AS status, "
            "e.container_port AS container_port, e.host_port AS host_port, "
            "e.active_host_port AS active_host_port, "
            "e.protocol AS protocol, e.route AS route "
            "FROM service_endpoints e JOIN jobs j ON j.service_name = e.service_name"
            f"{where} ORDER BY e.service_name ASC LIMIT ?"
        )
        params.append(limit + 1)
        cursor_obj = await self._db.conn.execute(sql, params)
        rows = list(await cursor_obj.fetchall())
        items = [
            RouteEndpoint(
                service_name=r["service_name"],
                kind=JobKind(r["kind"]),
                status=JobStatus(r["status"]),
                container_port=r["container_port"],
                host_port=r["host_port"],
                # (P24b) Raw column, NOT a COALESCE: the ONE COALESCE lives in
                # ``list_active_service_routes`` (the proxy desired-set). Every
                # other consumer derives the live port in Python off this field
                # so both numbers stay surfaceable side by side.
                active_host_port=r["active_host_port"],
                protocol=r["protocol"] or "tcp",
                route=r["route"],
            )
            for r in rows[:limit]
        ]
        next_cursor: str | None = None
        if len(rows) > limit and items:
            next_cursor = _encode_name_cursor(items[-1].service_name)
        return items, next_cursor

    async def get_reserved_service_ports(self) -> set[int]:
        """Read every stable and cutover port after pending writes settle."""
        async with self._db.write_lock:
            cursor = await self._db.conn.execute(
                "SELECT host_port, active_host_port FROM service_endpoints"
            )
            return {port for row in await cursor.fetchall() for port in row if port is not None}

    async def get_service_endpoint(self, service_name: str) -> ServiceEndpoint | None:
        """Return the durable host-port reservation for `service_name`, or `None`."""
        cursor = await self._db.conn.execute(
            "SELECT * FROM service_endpoints WHERE service_name = ?", (service_name,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_endpoint(row)

    @staticmethod
    def _row_to_endpoint(row: sqlite3.Row) -> ServiceEndpoint:
        """Convert an aiosqlite Row to a `ServiceEndpoint` model."""
        r = row
        return ServiceEndpoint(
            id=r["id"],
            service_name=r["service_name"],
            job_id=r["job_id"],
            container_port=r["container_port"],
            host_port=r["host_port"],
            protocol=r["protocol"] or "tcp",
            route=r["route"],
            active_host_port=r["active_host_port"],
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    @_serialized
    async def acquire_service_port(
        self,
        service_name: str,
        job_id: str,
        container_port: int,
        port_range: tuple[int, int],
        *,
        is_bindable: Callable[[int], bool] | None = None,
    ) -> ServiceEndpoint:
        """Reserve or reuse a stable service port across restarts and reboots.

        Repoint job/container details while preserving the reservation. If the optional
        is_bindable probe rejects a held port, replace it to avoid repeated failed
        restarts. Allocate the lowest free port in [lo, hi], excluding DEFAULT_PORT,
        reservations and unbindable ports; UNIQUE guards against duplicate reservations.

        Raises:
            PortRangeExhausted: No eligible port remains.
        """
        import sqlite3

        lo, hi = port_range
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.conn.execute(
                "SELECT * FROM service_endpoints WHERE service_name = ?", (service_name,)
            )
            existing = await cursor.fetchone()
            if existing is not None:
                held = existing["host_port"]
                cursor = await self._db.conn.execute(
                    "SELECT 1 FROM service_endpoints WHERE active_host_port = ? "
                    "AND service_name != ?",
                    (held, service_name),
                )
                used_by_cutover = await cursor.fetchone() is not None
                if not used_by_cutover and (is_bindable is None or is_bindable(held)):
                    await self._db.conn.execute(
                        "UPDATE service_endpoints SET job_id = ?, container_port = ? "
                        "WHERE service_name = ?",
                        (job_id, container_port, service_name),
                    )
                    await self._db.conn.execute("COMMIT")
                    refreshed = await self.get_service_endpoint(service_name)
                    assert refreshed is not None
                    return refreshed
                # Held port is no longer bindable: drop it and reallocate (CRIT-4).
                await self._db.conn.execute(
                    "DELETE FROM service_endpoints WHERE service_name = ?", (service_name,)
                )

            cursor = await self._db.conn.execute(
                "SELECT host_port, active_host_port FROM service_endpoints"
            )
            used = {port for row in await cursor.fetchall() for port in row if port is not None}
            chosen: int | None = None
            for port in range(lo, hi + 1):
                if port == DEFAULT_PORT or port in used:
                    continue
                if is_bindable is not None and not is_bindable(port):
                    continue
                chosen = port
                break
            if chosen is None:
                await self._db.conn.execute("ROLLBACK")
                raise PortRangeExhausted(lo, hi)

            endpoint = ServiceEndpoint(
                service_name=service_name,
                job_id=job_id,
                container_port=container_port,
                host_port=chosen,
            )
            await self._db.conn.execute(
                """INSERT INTO service_endpoints
                   (id, service_name, job_id, container_port, host_port, protocol,
                    route, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    endpoint.id,
                    endpoint.service_name,
                    endpoint.job_id,
                    endpoint.container_port,
                    endpoint.host_port,
                    endpoint.protocol,
                    endpoint.route,
                    endpoint.created_at.isoformat(),
                ),
            )
            await self._db.conn.execute("COMMIT")
            return endpoint
        except sqlite3.IntegrityError:
            # host_port UNIQUE backstop — a logic race slipped a duplicate past
            # the in-transaction scan; roll back and let the caller retry next tick.
            try:
                await self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        except Exception:
            try:
                await self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    @_serialized
    async def release_service_endpoint(self, service_name: str) -> None:
        """Delete a service's host-port reservation (only on service *delete*).

        Never called on a restart — the endpoint outlives container churn so the
        `host_port` (and future route) stay stable.
        """
        await self._db.conn.execute(
            "DELETE FROM service_endpoints WHERE service_name = ?", (service_name,)
        )
        await self._db.conn.commit()

    @_serialized
    async def set_endpoint_route(self, service_name: str, route: str | None) -> None:
        """Set a service endpoint's proxy `route`.

        Contract: `NULL` = unrouted (proxy off, or a `kind=model` row —
        models are never routed), `"/name"` = path route, `""` (empty
        string) = subdomain route. Callers must test `route is not None`,
        never truthiness — an empty string is a valid, routed state.
        """
        await self._db.conn.execute(
            "UPDATE service_endpoints SET route = ? WHERE service_name = ?",
            (route, service_name),
        )
        await self._db.conn.commit()

    @_serialized
    async def set_endpoint_active_port(self, service_name: str, port: int | None) -> None:
        """Set the transient cutover dial, or None to fall back to the reserved port.

        This nullable pointer is neither unique nor a second allocation from the
        service-port range. Repeated calls and missing endpoint rows are harmless.
        """
        await self._db.conn.execute(
            "UPDATE service_endpoints SET active_host_port = ? WHERE service_name = ?",
            (port, service_name),
        )
        await self._db.conn.commit()
