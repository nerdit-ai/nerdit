"""Service-lifecycle MCP tools (P2) + diagnose (P13c) + run (P20) + stats (P24).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call, _clamp
from nerdit.mcp.tools._shared import (
    DEFAULT_DIAGNOSE_LOG_TAIL,
    DEFAULT_LOG_TAIL,
    DEFAULT_RUN_LOG_TAIL,
    DEFAULT_RUN_TIMEOUT_S,
    DEFAULT_SERVICE_LIMIT,
    MAX_DIAGNOSE_LOG_TAIL,
    MAX_LOG_TAIL,
    MAX_RUN_LOG_TAIL,
    MAX_SERVICE_LIMIT,
)
from nerdit.mcp.transport import _request_client


async def _list_services_impl(
    client: NerditClient,
    *,
    status: str | None = None,
    cursor: str | None = None,
    limit: int = DEFAULT_SERVICE_LIMIT,
) -> Any:
    """List services (optionally by status), bounded to ``limit`` per page.

    The bound is enforced server-side (``?limit=``) so the daemon never returns
    an unbounded page; the cursor is passed straight through for paging.
    """
    bound = _clamp(limit, MAX_SERVICE_LIMIT)
    return await _call(client.list_services(status=status, cursor=cursor, limit=bound))


async def _get_service_impl(client: NerditClient, ident: str) -> Any:
    """Return a single service resolved by id or name."""
    return await _call(client.get_service(ident))


async def _wait_for_service_impl(
    client: NerditClient,
    ident: str,
    *,
    version: int | None = None,
    timeout: int = 60,
) -> Any:
    """Block until a service converges/fails, or the timeout elapses.

    A read (no idempotency key). The server long-polls and **always** returns a
    four-way ``outcome`` (``converged``/``failed``/``timeout``/``superseded``);
    the daemon clamps ``timeout`` to [1, 300]. Pass ``version`` **explicitly**
    from a deploy's 201 ``last_deploy.version`` so a concurrent redeploy is
    reported as ``superseded`` rather than silently converged on.
    """
    return await _call(client.wait_for_service(ident, version=version, timeout=timeout))


async def _service_logs_impl(
    client: NerditClient,
    ident: str,
    *,
    since_id: int = 0,
    tail: int = DEFAULT_LOG_TAIL,
    grep: str | None = None,
    since: str | None = None,
    source: str = "all",
) -> Any:
    """Return the last ``tail`` log entries for a service (by id or name).

    The bound is enforced server-side (``?tail=``) so the daemon never
    materializes the whole log set. ``grep``/``since``/``source`` filter inside
    that same bounded query, so ``tail`` counts *matching* lines.

    ``source`` stays a plain ``str`` and the default is passed as ``None`` on
    the wire: the daemon owns the enum (it 422s anything outside it), so the
    tool schema can never drift from the route's, and an unfiltered read sends
    no ``source`` parameter at all.
    """
    bound = _clamp(tail, MAX_LOG_TAIL)
    result = await _call(
        client.get_service_logs(
            ident,
            since_id=since_id,
            tail=bound,
            grep=grep,
            since=since,
            source=None if source == "all" else source,
        )
    )
    if isinstance(result, list):
        return result[-bound:]
    return result


async def _serve_impl(
    client: NerditClient,
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
    health_check: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Register a service from a prebuilt image, auto-minting an idempotency key.

    The agent rarely supplies its own key, so we mint a UUID to make every
    register safe to retry: a replayed call collapses to the same service rather
    than failing on the unique ``service_name``.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.create_service(
            name=name,
            image=image,
            port=port,
            gpus=gpus,
            restart_policy=restart_policy,
            command=command,
            script_path=script_path,
            env=env,
            vendor=vendor,
            health_check=health_check,
            idempotency_key=idempotency_key,
        )
    )


async def _stop_service_impl(
    client: NerditClient,
    ident: str,
    *,
    idempotency_key: str | None = None,
) -> Any:
    """Stop a service (desired_state → stopped), auto-minting an idempotency key."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.stop_service(ident, idempotency_key=idempotency_key))


async def _restart_service_impl(
    client: NerditClient,
    ident: str,
    *,
    idempotency_key: str | None = None,
) -> Any:
    """Restart a service (clears backoff), auto-minting an idempotency key."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.restart_service(ident, idempotency_key=idempotency_key))


async def _remove_service_impl(
    client: NerditClient,
    ident: str,
    *,
    purge: str = "secrets",
    force: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Delete a service (teardown + remove row), auto-minting an idempotency key."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.remove_service(ident, purge=purge, force=force, idempotency_key=idempotency_key)
    )


async def _diagnose_service_impl(
    client: NerditClient, ident: str, *, log_tail: int = DEFAULT_DIAGNOSE_LOG_TAIL
) -> Any:
    """Fetch a service's structured failure diagnosis, bounding the log tail.

    ``log_tail`` is clamped to [1, 200] here and server-side (the daemon does
    the same), so an agent can never pull an unbounded tail through one call.
    """
    bound = _clamp(log_tail, MAX_DIAGNOSE_LOG_TAIL)
    return await _call(client.diagnose_service(ident, log_tail=bound))


async def _service_stats_impl(client: NerditClient, ident: str) -> Any:
    """Return one live resource sample for a service.

    A read (no idempotency key), and unbounded by construction: the payload is
    a fixed set of numeric counters, so there is nothing to clamp. The daemon
    caches the sample for 2 s, which is what makes a polling agent cheap.
    """
    return await _call(client.get_service_stats(ident))


async def _run_command_impl(
    client: NerditClient,
    ident: str,
    *,
    command: list[str],
    env: dict[str, str] | None = None,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    log_tail: int = DEFAULT_RUN_LOG_TAIL,
    idempotency_key: str | None = None,
) -> Any:
    """Run a one-off command against a service (P20), auto-minting a key.

    The two bounds are treated asymmetrically on purpose. ``log_tail`` is
    clamped here because its server bound is the compile-time 200 the route
    itself clamps to — mirroring it costs nothing and can never disagree.
    ``timeout_s`` is only *floored* at 1 and otherwise passed straight through:
    its ceiling is config-driven (``[services].run_timeout_max_s``), so a
    client-side clamp would silently truncate the caller's request whenever an
    operator raises the cap, and the authoritative 422 ``run.timeout_too_large``
    could then never fire for an MCP caller. MCP stays a thin projection of the
    API — the daemon owns the policy.

    What makes a dropped connection on this long blocking call retry-safe is a
    *caller-supplied* key REUSED on the retry: the replay returns the recorded
    envelope instead of executing the command a second time (D-P20-6). The
    mint is only the fallback when the agent supplies none — it keeps a
    single call idempotent against the middleware's claim, but it protects
    nothing across calls, since a fresh UUID is a fresh execution.

    **That protection covers a run that COMPLETED (2xx) — and only that.** The
    P1 idempotency rule is "never pin a failure" (``daemon/idempotency.py``),
    so any non-2xx deletes the key and a same-key retry executes for real. Two
    consequences an agent must plan around, neither of which this layer can
    fix:

    * ``503 run.interrupted`` is a failure by that rule, yet it is exactly the
      case whose own hint says the command may have PARTIALLY applied. Retrying
      it with the same key re-runs it. Treat that code as "unknown outcome,
      needs a human or an idempotent command", never as "safe to retry".
    * If the daemon dies mid-run the key stays claimed ``in_progress`` until
      its 24 h TTL, so the retry gets ``409 idempotency_in_progress`` for a
      request that can never complete. Mint a fresh key once you have
      established the run is not still going.

    Making ``run.interrupted`` pin (treating container-started as committed) is
    a deliberate carve-out from the P1 rule and is P22-scoped work.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.run_service_command(
            ident,
            command=command,
            env=env,
            timeout_s=max(1, int(timeout_s)),
            log_tail=_clamp(log_tail, MAX_RUN_LOG_TAIL),
            idempotency_key=idempotency_key,
        )
    )


# fmt: off
async def list_services(
        status: str | None = None,
        cursor: str | None = None,
        limit: int = DEFAULT_SERVICE_LIMIT,
    ) -> Any:
        """List services, optionally filtered by status, bounded to ``limit`` per page."""
        return await _list_services_impl(
            _request_client(), status=status, cursor=cursor, limit=limit
        )

async def get_service(name: str) -> Any:
        """Get one service. ``name`` accepts the service name or the job id."""
        return await _get_service_impl(_request_client(), name)

async def wait_for_service(name: str, version: int | None = None, timeout: int = 60) -> Any:
        """Use when: a deploy is in flight and you need its real outcome.

        Blocks until the service converges (healthy) or fails, then returns
        ``{outcome: converged|failed|timeout|superseded, ...}`` — always, never
        raising on a timeout. ``name`` accepts the service name or job id.
        After a ``deploy``/``deploy_git``/``deploy_app``/``deploy_template``
        call, pass the 201's ``last_deploy.version`` as ``version`` so a
        concurrent redeploy resolves ``superseded`` instead of being mistaken
        for your deploy. ``timeout`` is clamped server-side to [1, 300] seconds.

        On ``outcome: failed`` the response also carries ``diagnosis`` — the
        FULL ``diagnose_service`` bundle (``remediation.code`` + ``detail``,
        crash forensics, health probe, binding readiness and a bounded log
        tail), folded in so a failure needs no second call. Branch on
        ``diagnosis.remediation.code``. It is ``null`` on every non-failed
        outcome, and ``null`` on a failure your token is not owner-or-admin for
        (the bundle carries app log lines) — in that case call
        ``diagnose_service`` yourself with a token that owns the app.
        """
        return await _wait_for_service_impl(
            _request_client(), name, version=version, timeout=timeout
        )

async def service_logs(
        name: str,
        since_id: int = 0,
        tail: int = DEFAULT_LOG_TAIL,
        grep: str | None = None,
        since: str | None = None,
        source: str = "all",
    ) -> Any:
        """Use when: you need the raw output a service or its build printed.

        ``name`` accepts the service name or the job id. Polling-only: pass
        the last entry's id as ``since_id`` to fetch only newer lines.

        ``source`` picks which output you get. ``'runtime'`` is the container's
        own stdout/stderr plus the crash tail; ``'build'`` is the image build's
        output; ``'all'`` (the default) is both plus the daemon's own lifecycle
        lines. **Pass ``source='runtime'`` when you are debugging a running
        app** — the build and the app used to share one stream, so a bare
        ``grep`` over a deployed app still matches BuildKit layer chatter
        (``404B``, ``CACHED``, ``sha256:…``) from the build that produced it.
        Lines written by an older daemon are stored as stdout and therefore read
        as ``'runtime'`` whichever program printed them.

        ``grep`` keeps only lines CONTAINING that text. It is a plain
        **literal substring**, never a regex — ``grep='100%'`` finds a literal
        ``100%``, ``.*`` finds a literal ``.*``, and there is no anchoring, no
        character class and no escaping to do. ``since`` keeps only lines at or
        after an ISO-8601 UTC timestamp (``2026-08-07T10:00:00Z``).

        All three filter server-side inside the ``tail`` bound, so ``tail``
        counts matching lines: ``grep='Traceback', source='runtime', tail=20``
        gives the last 20 app tracebacks out of a million-line log in one
        bounded query. Prefer that over pulling a large tail and searching it
        yourself.
        """
        return await _service_logs_impl(
            _request_client(),
            name,
            since_id=since_id,
            tail=tail,
            grep=grep,
            since=since,
            source=source,
        )

async def serve(
        name: str,
        image: str,
        port: int = 8000,
        gpus: int = 0,
        restart_policy: str = "always",
        command: str | None = None,
        script_path: str | None = None,
        env: dict[str, str] | None = None,
        vendor: str | None = None,
        health_check: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Register a service from a prebuilt image. Idempotency key auto-generated if omitted."""
        return await _serve_impl(
            _request_client(),
            name=name,
            image=image,
            port=port,
            gpus=gpus,
            restart_policy=restart_policy,
            command=command,
            script_path=script_path,
            env=env,
            vendor=vendor,
            health_check=health_check,
            idempotency_key=idempotency_key,
        )

async def stop_service(name: str, idempotency_key: str | None = None) -> Any:
        """Stop a service (desired_state → stopped).

        ``name`` accepts the service name or the job id. An idempotency key is
        auto-generated if omitted.
        """
        return await _stop_service_impl(_request_client(), name, idempotency_key=idempotency_key)

async def restart_service(name: str, idempotency_key: str | None = None) -> Any:
        """Restart a service (clears backoff).

        ``name`` accepts the service name or the job id. An idempotency key is
        auto-generated if omitted.
        """
        return await _restart_service_impl(_request_client(), name, idempotency_key=idempotency_key)

async def remove_service(
        name: str,
        purge: str = "secrets",
        force: bool = False,
        idempotency_key: str | None = None,
    ) -> Any:
        """Delete a service (teardown + remove).

        ``name`` accepts the service name or the job id. ``purge`` is a CSV of
        durable-state targets to destroy alongside the row
        (``secrets``/``data``/``images``; default ``secrets``). ``force`` bypasses
        the model reference guard (deleting a model still bound by other apps).
        An idempotency key is auto-generated if omitted.
        """
        return await _remove_service_impl(
            _request_client(), name, purge=purge, force=force, idempotency_key=idempotency_key
        )

async def diagnose_service(name: str, log_tail: int = DEFAULT_DIAGNOSE_LOG_TAIL) -> Any:
        """Use when: a service is failing and you need the reason, not the logs.

        Bundles the deploy phase, crash forensics (``last_exit_code``,
        ``oom_killed``), a fresh health probe, AI-binding readiness, a bounded
        log tail and a single ``remediation_code`` to branch on — call it after
        ``wait_for_service`` returns ``failed`` instead of scraping logs.
        ``name`` accepts the service name or the job id. Owner-or-admin read:
        the bundle carries env/secret key *names* and app log lines, never
        secret values. ``log_tail`` is clamped to [1, 200].
        """
        return await _diagnose_service_impl(_request_client(), name, log_tail=log_tail)

async def service_stats(name: str) -> Any:
        """Get live CPU, memory, network and PID usage for a running service.

        ``name`` accepts the service name or the job id. Returns
        ``{available, stats: {cpu_pct, mem_used_bytes, mem_limit_bytes,
        mem_pct, net_rx_bytes, net_tx_bytes, pids}, gpus, cached}``.

        **Branch on ``available``, never on zeroes.** ``available: false`` with
        ``stats: null`` is the normal answer when the service has no running
        container or the daemon cannot reach Docker; every individual counter
        is likewise ``null`` when unknown, so a missing metric never
        masquerades as an idle one. ``cpu_pct`` follows the docker convention —
        ``100.0`` is one fully-used core, so a multi-core host can exceed 100.

        For *why* a service is unhealthy use ``diagnose_service``; this tool
        answers *how much* a healthy one is consuming. The daemon caches the
        sample for 2 seconds, so polling it in a loop is cheap.
        """
        return await _service_stats_impl(_request_client(), name)

async def run_command(
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
        log_tail: int = DEFAULT_RUN_LOG_TAIL,
        idempotency_key: str | None = None,
    ) -> Any:
        """Run a one-off command in a deployed service's image (migrations, seeds).

        The command runs in a throwaway container built from the service's
        current image with the same env, secrets, AI/DB bindings and named
        volumes as the live container — the service itself keeps serving.
        ``name`` accepts the service name or the job id.

        ``command`` is argv, run unshelled (no globbing, no pipes — wrap in
        ``["sh", "-c", "..."]`` if you need them). It REPLACES the image's
        ``CMD`` and is APPENDED to its ``ENTRYPOINT`` when the image declares
        one, so on an image whose entrypoint is e.g. ``python -m gunicorn``
        your argv arrives as arguments to that, not as a fresh command line.

        **Bounded-synchronous**: this call BLOCKS until the command exits or
        ``timeout_s`` seconds elapse (default 300; the daemon rejects anything
        above ``[services].run_timeout_max_s`` with ``run.timeout_too_large``),
        then returns ``{exit_code, timed_out, oom_killed, duration_s, run_id,
        log_tail, ...}``. Branch on ``exit_code``/``timed_out`` — a non-zero
        exit is a normal 200 result, not an error. One run at a time per
        service (409 ``service.run_in_progress``). ``log_tail`` is clamped to
        [1, 200].

        **Never put credentials in ``command`` argv**: argv is visible via
        ``docker inspect`` on the host and is stored verbatim in the service's
        ``last_run``. Pass them in ``env`` instead — those values are masked in
        the audit trail. ``env`` overrides beat the app's config env and its
        secrets, but never platform-computed keys (resolved AI/DB binding vars,
        ``PORT``, ``NERDIT_RUN_ID``), which always win.

        Because the call blocks for minutes, **supply your own
        ``idempotency_key`` and REUSE it verbatim if you retry** — that is the
        only thing standing between a client-side read timeout and a migration
        running twice. Omitting it auto-generates one, which makes the single
        call safe but leaves every retry a fresh execution. A retried key
        REPLAYS an ``idempotent_replay`` envelope, NOT the command's output —
        the run's stdout is deliberately never cached at rest. Re-run under a
        fresh key if you need the output again, or read it from
        ``diagnose_service``'s ``last_run``.

        **Key reuse protects a COMPLETED run only.** A failed call un-pins the
        key, so a same-key retry really re-executes. In particular ``503
        run.interrupted`` means the container started and was lost — the
        command may have partially applied — and retrying it, with any key,
        runs it again: stop and inspect instead. And if the daemon died
        mid-run, the key stays claimed for 24 h and the retry returns ``409
        idempotency_in_progress``; mint a fresh key once you have confirmed
        the run is over.
        """
        return await _run_command_impl(
            _request_client(), name, command=command, env=env,
            timeout_s=timeout_s, log_tail=log_tail, idempotency_key=idempotency_key,
        )
# fmt: on


TOOLS = (
    list_services,
    get_service,
    wait_for_service,
    service_logs,
    serve,
    stop_service,
    restart_service,
    remove_service,
    diagnose_service,
    service_stats,
    run_command,
)
