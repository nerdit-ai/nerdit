"""Manage Caddy spawn, adoption, readiness, respawn backoff, and teardown.

`ProxyManager` initializes all mixin state and supplies bootstrap/TLS methods.
Calls stay on the composed instance so lifecycle hooks remain overridable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tls import ACME_HTTP_SERVER

if TYPE_CHECKING:
    from nerdit.config.settings import ProxySettings

    from .admin import CaddyAdmin

logger = logging.getLogger(__name__)


class CaddySupervisorMixin:
    """Owns the Caddy subprocess: spawn, kill, ready-poll, respawn backoff."""

    # -- state provided by ProxyManager.__init__ (composed, never set here) --
    _settings: ProxySettings
    _admin: CaddyAdmin
    _binary: str | None
    _proc: subprocess.Popen[bytes] | None
    _data_dir: Path
    _storage_root: Path
    _bootstrap_file: Path
    _pid_file: Path
    _log_file: Path
    _available: bool
    _acme_listener_live: bool | None
    _tls_synced: bool
    _tls_hash: str | None
    _live_domain_ids: frozenset[str]
    _spawn_attempts: int
    _last_spawn_at: float
    _foreign_warned: bool
    _conflict: bool

    # -- methods provided by CaddyTlsMixin (route/TLS territory, cross-mixin) --
    # Declared as TYPE_CHECKING-only method stubs (never real methods at
    # runtime — a real body here would sit ahead of CaddyTlsMixin's actual
    # implementation in ProxyManager's MRO and silently shadow it) so mypy can
    # typecheck this file in isolation with signatures that stay
    # Liskov-compatible with the real definitions once both mixins compose.
    if TYPE_CHECKING:

        def _bootstrap_config(self) -> dict[str, Any]: ...

        async def _converge_tls(
            self,
            domains: Sequence[str] | None = None,
            acme_domains: Sequence[str] | None = None,
        ) -> bool: ...

    # -- process lifecycle ----------------------------------------------------

    def _pid_alive(self) -> bool:
        """Return `True` when the recorded Caddy PID is a live process."""
        if not self._pid_file.exists():
            return False
        try:
            pid = int(self._pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False

    def _spawn(self) -> None:
        """Launch `caddy run --config <bootstrap.json>` detached.

        The bootstrap config (native JSON — no adapter) is written to disk and
        passed at launch so Caddy binds *our* configured `admin_addr` (not the
        hard-coded default :2019, which a system Caddy may already own) and comes
        up with the `nerdit` server + internal-CA TLS already loaded. Every
        subsequent route change still goes through the admin API, never a reload.
        """
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._bootstrap_file.write_text(json.dumps(self._bootstrap_config()))
        log_fh = open(self._log_file, "a")
        try:
            self._proc = subprocess.Popen(
                [
                    self._binary or self._settings.caddy_binary,
                    "run",
                    "--config",
                    str(self._bootstrap_file),
                ],
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
            )
        finally:
            log_fh.close()
        self._pid_file.write_text(str(self._proc.pid))
        logger.info("Caddy started (PID %d, log %s)", self._proc.pid, self._log_file)

    async def _kill(self) -> None:
        """SIGTERM the recorded Caddy PID, escalating to SIGKILL.

        Async so the SIGTERM grace wait yields to the event loop (`asyncio.sleep`)
        instead of blocking the single shared loop with `time.sleep` — a hung
        Caddy must never freeze the daemon's HTTP/SSE/health handling.
        """
        if not self._pid_file.exists():
            return
        try:
            pid = int(self._pid_file.read_text().strip())
        except ValueError:
            self._pid_file.unlink(missing_ok=True)
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            if sig is signal.SIGTERM:
                # Give it a moment to exit cleanly before escalating.
                for _ in range(20):
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        break
                    await asyncio.sleep(0.1)
                else:
                    continue
                break
        self._pid_file.unlink(missing_ok=True)
        self._proc = None

    async def _wait_ready(self, timeout: float = 10.0, interval: float = 0.3) -> bool:
        """Poll the admin API until Caddy answers or `timeout` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self._admin.ping():
                return True
            await asyncio.sleep(interval)
        return False

    def _backoff_seconds(self) -> float:
        """Exponential respawn backoff capped at 60s (self-heal, not hot-loop)."""
        return min(60.0, 2.0**self._spawn_attempts)

    def _invalidate_convergence(self) -> None:
        """Forget everything we believe about the CURRENT Caddy process's state.

        Three latches, one lifetime: the TLS one-shot (`_tls_synced`), the
        digest it was set for (`_tls_hash`) and the set of custom-domain route
        ids last observed live (`_live_domain_ids`). All three describe *a
        particular Caddy process*, so every transition that ends one — a
        foreign process taking the admin port, a backoff window with nothing
        answering, a failed respawn, and (Codex round 1, P1 #3831777092) a
        SUCCESSFUL respawn — must clear them together.

        The successful-respawn case is the one that bit: the fresh process loads
        `_bootstrap_config`, whose `tls` app is `_tls_desired_app(())`
        — deliberately no custom domains, because the DB is not readable at
        spawn time. Leaving the previous process's converged digest in place
        would make the next `_converge_tls` short-circuit with zero admin
        I/O and open the D-P26-4 route gate against a Caddy carrying only the
        bootstrap policies, i.e. exactly the "Host route ahead of its
        internal-issuer policy" state that gate exists to prevent.
        """
        self._tls_synced = False
        self._tls_hash = None
        self._live_domain_ids = frozenset()
        # The ACME listener belongs to a specific Caddy PROCESS —
        # it is part of the bootstrap that process was spawned with. Whatever we
        # knew about it is void the moment we stop believing in that process.
        self._acme_listener_live = None

    async def _sync_acme_listener(self) -> None:
        """Latch whether the live Caddy actually carries `nerdit-acme-http`.

        Called at every point where the proxy becomes available. One
        **conclusive** admin read per availability episode (the latch), never
        per reconcile tick; an inconclusive read latches nothing and is retried
        on the next healthy tick (review round 2).

        This exists because adoption is content-checked on the `nerdit` server
        ALONE: a Caddy that outlived an ungraceful daemon death is adopted on
        that evidence, and if ACME was enabled after it was spawned it carries no
        `:80` listener at all. Issuance may still limp along on certmagic's own
        temporary solver, but the redirect and the terminal 404 are simply not
        there — and reporting `listening: true` for that state sent the
        operator looking anywhere but at the actual cause (review round 1).

        The tri-state matters because the latch is only reset when an
        availability episode ENDS: a single admin blip (a reload in flight, the
        5 s client timeout) against a Caddy that then stays healthy forever used
        to pin `listening: false` and a doctor row telling the operator to
        restart a daemon whose :80 listener was serving all along. The
        fresh-spawn call site is the sharpest case — the bootstrap we just
        loaded is KNOWN to carry the server, since Caddy refuses to start when a
        configured listener cannot bind.
        """
        if not self._settings.acme.enabled:
            self._acme_listener_live = None
            return
        if self._acme_listener_live is not None:
            return
        live = await self._admin.server_present(ACME_HTTP_SERVER)
        if live is None:
            # Not an answer — leave the latch unset so the guard above re-probes
            # next tick. Status/doctor read an unset latch as False meanwhile
            # (fail-closed), which is right for a transient window and is what
            # the retry then corrects.
            logger.debug(
                "[proxy] ACME listener probe for '%s' was inconclusive; retrying next tick",
                ACME_HTTP_SERVER,
            )
            return
        self._acme_listener_live = live
        if not live:
            logger.warning(
                "[proxy] ACME is enabled but the running Caddy carries no '%s' server "
                "— it was started before [proxy.acme] was turned on. HTTP-01 "
                "validation on :%d is not served by this process; restart the daemon "
                "(or stop the stray Caddy) to bind it.",
                ACME_HTTP_SERVER,
                self._settings.acme.http_port,
            )

    async def _ensure_alive(self) -> bool:
        """Make sure *our* Caddy is up; respawn with backoff if it died.

        Ownership is verified by *content* (`has_nerdit_server`), not just
        reachability or the pid file: the admin port may be answered by a
        *foreign* Caddy (e.g. a system `caddy.service` on the default :2019), in
        which case we must NOT respawn a doomed process or push routes into it.
        Adoption is content-first — mirroring `nerdit.core.proxy.ProxyManager.start` —
        so a live nerdit Caddy with a stale/missing pid file (e.g. an
        out-of-band restart) is adopted rather than misclassified as dead and
        doomed-respawned.
        """
        pinged = await self._admin.ping()
        if pinged and await self._admin.has_nerdit_server():
            # A nerdit Caddy is answering — it is ours by content. Adopt it
            # whether or not we currently track its pid (matches start()).
            self._available = True
            self._spawn_attempts = 0
            self._foreign_warned = False
            self._conflict = False
            await self._sync_acme_listener()
            # Flag-guarded: this branch runs on EVERY healthy reconcile tick
            # (~5s); the guard keeps the TLS sync a one-shot, not a per-tick
            # admin read/write.
            if not self._tls_synced:
                await self._converge_tls()
            return True
        # Admin port answers (but not our nerdit server, from the check above) and
        # we hold no live process of our own → a foreign Caddy owns the port.
        # Refuse: don't spawn a doomed process, don't push routes into it.
        if pinged and not self._pid_alive():
            if not self._foreign_warned:
                logger.error(
                    "[proxy] admin API %s is owned by a foreign process (not nerdit); "
                    "proxy disabled. Set [proxy].admin_addr to a free port or stop the "
                    "other Caddy.",
                    self._settings.admin_addr,
                )
                self._foreign_warned = True
            self._available = False
            self._invalidate_convergence()
            self._conflict = True
            return False
        # Down or hung → backoff-gated respawn (non-blocking on the shared loop).
        # Nothing answers the admin ping, so the port is demonstrably not
        # foreign-owned: clear any stale conflict flag BEFORE the backoff
        # early-return so the backoff/starting state becomes visible (a foreign
        # process that has since exited must not keep masking the real spawn
        # failure). While a foreign process is still alive we take the branch
        # above and keep the flag set — the one-tick staleness there is correct.
        self._conflict = False
        now = time.monotonic()
        if now - self._last_spawn_at < self._backoff_seconds():
            self._available = False
            self._invalidate_convergence()
            return False
        self._last_spawn_at = now
        self._spawn_attempts += 1
        logger.warning(
            "[proxy] Caddy not responding; respawning (attempt %d)", self._spawn_attempts
        )
        try:
            if self._pid_alive():
                await self._kill()
            # BEFORE the spawn, not after a successful one: the process we are
            # about to replace is already gone, so anything we believed about
            # its TLS subtree or its live domain routes is void whichever way
            # the spawn goes (see `_invalidate_convergence`).
            self._invalidate_convergence()
            self._spawn()
            # Only declare available once OUR process answers with the nerdit
            # server loaded — a foreign Caddy answering the ping (while our spawned
            # process died failing to bind the taken port) must not pass.
            if (
                await self._wait_ready(timeout=5.0)
                and self._pid_alive()
                and await self._admin.has_nerdit_server()
            ):
                # Config is loaded at launch (--config); routes start empty and
                # the rest of this reconcile tick re-adds the live set.
                self._available = True
                self._spawn_attempts = 0
                self._foreign_warned = False
                self._conflict = False
                return True
        except Exception:
            logger.exception("[proxy] respawn failed")
        self._available = False
        self._invalidate_convergence()
        return False

    async def stop(self) -> None:
        """Tear down the Caddy process (best-effort) and close the admin client.

        Reaps our process whenever a pid is tracked — even when disabled — so a
        leftover nerdit-owned Caddy never outlives the daemon (the pid file is
        written only by `_spawn`, so a live pid there is ours).
        """
        if self._pid_alive():
            await self._kill()
        try:
            await self._admin.aclose()
        except Exception:
            logger.debug("[proxy] admin client close failed", exc_info=True)
