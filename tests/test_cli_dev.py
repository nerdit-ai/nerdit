"""Test watch-and-redeploy with a mocked client and scripted filesystem changes.

Patch command-local sleep/time while keeping the event loop's real clock. Each
poll applies one mutation; script exhaustion raises KeyboardInterrupt.

Coalesce save bursts, ignore excluded trees and exit zero with a Ctrl-C summary.
Pin git-source redeployment versus ZIP upload, wait-version handoff, remediation
codes on failure and the startup warning when cutover is disabled.
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.app import app
from nerdit.cli.commands import dev as dev_mod

runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Keep Rich from wrapping the sentences the assertions look for."""
    monkeypatch.setenv("COLUMNS", "300")


# --- harness ------------------------------------------------------------------


class _Clock:
    """A stand-in for the ``time`` module exposing only ``monotonic``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t


def _install_loop(monkeypatch, script: list) -> _Clock:
    """Drive the poll loop off a script: one entry per poll, then Ctrl-C.

    Each fake ``sleep`` advances the fake clock by the requested delay and runs
    the next scripted step; an exhausted script raises ``KeyboardInterrupt``,
    standing in for the user's Ctrl-C.
    """
    clock = _Clock()
    steps = iter(script)

    async def _sleep(delay: float) -> None:
        clock.t += delay
        try:
            step = next(steps)
        except StopIteration:
            raise KeyboardInterrupt from None
        if step is not None:
            step()

    monkeypatch.setattr(dev_mod, "time", clock)
    monkeypatch.setattr(dev_mod, "asyncio", types.SimpleNamespace(run=asyncio.run, sleep=_sleep))
    return clock


def _write(directory, name: str, content: str) -> None:
    """Write a file (creating parents) — the scripted 'a developer saved'."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _app(directory) -> None:
    """Seed a watched folder whose ``[deploy].name`` is the stable ``app``.

    Without a nerdit.toml the service name defaults to the folder name, which
    pytest randomizes — the name-bearing assertions pin the config path instead.
    """
    _write(directory, "nerdit.toml", '[deploy]\nname = "app"\nport = 8000\n')
    _write(directory, "app.py", "v0")


def _service(**over) -> dict:
    body = {
        "id": "svc123456789",
        "name": "app",
        "kind": "service",
        "status": "building",
        "gpu_count": 0,
        "build_version": 4,
        "last_deploy": {"version": 4, "action": "redeploy", "phase": "queued"},
        "endpoint": {"host_port": 30001, "url": "http://127.0.0.1:30001", "public_url": None},
        "source": {"type": "zip"},
    }
    body.update(over)
    return body


def _client(**methods) -> AsyncMock:
    fake = AsyncMock()
    # Startup reads the service through `resolve_service` (404 → None, every
    # other failure propagates) — see tests/test_cli_services.py for its grammar.
    fake.resolve_service = AsyncMock(return_value=_service())
    fake.get_app_config = AsyncMock(return_value={"deploy": {"cutover": None}})
    fake.deploy = AsyncMock(return_value=_service())
    fake.redeploy_app = AsyncMock(return_value=_service(source={"type": "git"}))
    fake.wait_for_service = AsyncMock(
        return_value={
            "outcome": "converged",
            "service_name": "app",
            "public_url": "https://host.local/app",
        }
    )
    for name, value in methods.items():
        setattr(fake, name, AsyncMock(**value))
    return fake


def _invoke(tmp_path, fake, args: list[str] | None = None):
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        return runner.invoke(app, ["dev", str(tmp_path), *(args or [])])


# --- the three plan-pinned behaviours -----------------------------------------


def test_debounce_collapses_a_burst_into_one_deploy(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client()
    _install_loop(
        monkeypatch,
        [
            # poll 1 + 2: a burst spanning two polls — re-arms, never deploys.
            lambda: _write(tmp_path, "a.py", "1"),
            lambda: (_write(tmp_path, "b.py", "1"), _write(tmp_path, "c.py", "1")),
            None,  # poll 3: quiet past the debounce → exactly one deploy
            None,  # poll 4: still quiet, nothing pending → no second deploy
        ],
    )

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    assert fake.deploy.await_count == 1
    assert fake.redeploy_app.await_count == 0


def test_excluded_paths_do_not_trigger_a_deploy(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client()
    _install_loop(
        monkeypatch,
        [
            lambda: _write(tmp_path, "node_modules/dep/index.js", "1"),
            lambda: _write(tmp_path, "__pycache__/app.cpython-311.pyc", "1"),
            lambda: _write(tmp_path, ".git/objects/ab/cdef", "1"),
            lambda: _write(tmp_path, "data/blob.bin", "1"),
            None,
            None,
        ],
    )

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    fake.deploy.assert_not_awaited()


def test_ctrl_c_exits_cleanly(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client()
    _install_loop(monkeypatch, [])  # the very first poll is the Ctrl-C

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    assert "Stopped watching after 0 deploy(s)." in result.output
    fake.deploy.assert_not_awaited()


def test_a_save_made_during_a_deploy_is_not_swallowed(tmp_path, monkeypatch):
    """No post-deploy re-baseline: an edit landing mid-flight deploys next tick."""
    _app(tmp_path)
    fake = _client()

    async def _deploy(**kwargs):
        _write(tmp_path, "app.py", "saved-while-the-build-was-running")
        return _service()

    fake.deploy = AsyncMock(side_effect=_deploy)
    _install_loop(
        monkeypatch,
        [
            lambda: _write(tmp_path, "app.py", "v1"),
            None,  # deploy #1 — writes app.py while in flight
            None,  # that write is seen, not swallowed → re-arm
            None,  # quiet → deploy #2
            None,  # quiet, nothing pending
        ],
    )

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    assert fake.deploy.await_count == 2


# --- source-aware dispatch ----------------------------------------------------


def test_git_sourced_service_redeploys_from_source(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client(resolve_service={"return_value": _service(source={"type": "git"})})
    _install_loop(monkeypatch, [lambda: _write(tmp_path, "app.py", "v1"), None])

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    fake.redeploy_app.assert_awaited_once_with("app")
    fake.deploy.assert_not_awaited()
    # The push-first caveat is stated, not hidden.
    assert "push first" in result.output


def test_unknown_service_falls_back_to_a_zip_deploy(tmp_path, monkeypatch):
    """A folder that has never been deployed: the first change creates it.

    ``resolve_service`` answers ``None`` on the 404 — the ONLY error shape the
    watch session treats as "not deployed yet".
    """
    _write(tmp_path, "app.py", "v0")
    fake = _client(resolve_service={"return_value": None})
    _install_loop(monkeypatch, [lambda: _write(tmp_path, "app.py", "v1"), None])

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    assert fake.deploy.await_count == 1
    assert fake.deploy.await_args.kwargs["name"] == tmp_path.name


def test_an_unreachable_daemon_at_startup_aborts(tmp_path, monkeypatch):
    """Not "not deployed yet": a transport failure must not be reclassified.

    Answering ``None`` here would start uploading local ZIPs over a service
    whose recorded source is a git repo, so the session refuses to start.
    """
    _app(tmp_path)
    fake = _client(
        resolve_service={"side_effect": httpx.ConnectError("connection refused")},
    )
    _install_loop(monkeypatch, [lambda: _write(tmp_path, "app.py", "v1"), None])

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 1
    fake.deploy.assert_not_awaited()
    fake.redeploy_app.assert_not_awaited()


# --- the /wait handoff --------------------------------------------------------


def test_wait_receives_the_version_from_the_201_body(tmp_path, monkeypatch):
    """The P13 §1.2 handoff lock: the version is explicit, never defaulted."""
    _app(tmp_path)
    fake = _client(deploy={"return_value": _service(last_deploy={"version": 9}, build_version=4)})
    _install_loop(monkeypatch, [lambda: _write(tmp_path, "app.py", "v1"), None])

    result = _invoke(tmp_path, fake, ["--timeout", "45"])

    assert result.exit_code == 0
    fake.wait_for_service.assert_awaited_once_with("app", version=9, timeout=45)
    assert "https://host.local/app" in result.output


def test_failed_generation_prints_the_remediation_code(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client(
        wait_for_service={
            "return_value": {
                "outcome": "failed",
                "service_name": "app",
                "reason": "build_failed",
            }
        },
        diagnose_service={
            "return_value": {
                "remediation": {"code": "build.failed", "detail": "Check the build logs."}
            }
        },
    )
    _install_loop(monkeypatch, [lambda: _write(tmp_path, "app.py", "v1"), None])

    result = _invoke(tmp_path, fake)

    # A failed generation is information for the next save, not an exit.
    assert result.exit_code == 0
    assert "build.failed" in result.output
    assert "Stopped watching after 1 deploy(s)." in result.output


def test_deploy_error_keeps_the_watch_alive(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client(deploy={"side_effect": RuntimeError("boom")})
    _install_loop(
        monkeypatch,
        [
            lambda: _write(tmp_path, "app.py", "v1"),
            None,  # deploy #1 raises
            lambda: _write(tmp_path, "app.py", "v2"),
            None,  # deploy #2 still attempted — the loop survived
        ],
    )

    result = _invoke(tmp_path, fake)

    assert result.exit_code == 0
    assert fake.deploy.await_count == 2
    fake.wait_for_service.assert_not_awaited()


# --- startup warning ----------------------------------------------------------


def test_gpu_service_warns_it_is_not_cutover_armed(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client(resolve_service={"return_value": _service(gpu_count=1)})
    _install_loop(monkeypatch, [])

    result = _invoke(tmp_path, fake)

    assert "not cutover-armed" in result.output
    assert "downtime window" in result.output


def test_running_service_without_a_route_warns(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client(resolve_service={"return_value": _service(status="running")})
    _install_loop(monkeypatch, [])

    result = _invoke(tmp_path, fake)

    assert "not cutover-armed" in result.output


def test_api_set_cutover_opt_out_warns_even_when_the_toml_is_silent(tmp_path, monkeypatch):
    """`nerdit config app set … cutover=false` carries forward server-side only.

    The local nerdit.toml says nothing, the service is healthy and routed —
    every client-visible eligibility fact passes — yet redeploys have a
    downtime window, so the ONE extra startup GET is what makes it honest.
    """
    _app(tmp_path)
    fake = _client(
        resolve_service={
            "return_value": _service(
                status="running",
                endpoint={
                    "host_port": 30001,
                    "url": "http://127.0.0.1:30001",
                    "public_url": "https://host.local/app",
                },
            )
        },
        get_app_config={"return_value": {"deploy": {"cutover": False}}},
    )
    _install_loop(monkeypatch, [])

    result = _invoke(tmp_path, fake)

    assert "not cutover-armed" in result.output
    assert "cutover = false" in result.output
    fake.get_app_config.assert_awaited_once_with("app")


def test_local_toml_opt_out_short_circuits_the_app_config_fetch(tmp_path, monkeypatch):
    _write(tmp_path, "nerdit.toml", '[deploy]\nname = "app"\nport = 8000\ncutover = false\n')
    _write(tmp_path, "app.py", "v0")
    fake = _client()
    _install_loop(monkeypatch, [])

    result = _invoke(tmp_path, fake)

    assert "not cutover-armed" in result.output
    fake.get_app_config.assert_not_awaited()


def test_still_building_service_is_not_warned_about_its_route(tmp_path, monkeypatch):
    """A generation that has not registered its route yet is not a verdict."""
    _app(tmp_path)
    fake = _client(resolve_service={"return_value": _service(status="building")})
    _install_loop(monkeypatch, [])

    result = _invoke(tmp_path, fake)

    assert "not cutover-armed" not in result.output


def test_routed_cpu_service_is_not_warned(tmp_path, monkeypatch):
    _app(tmp_path)
    fake = _client(
        resolve_service={
            "return_value": _service(
                status="running",
                endpoint={
                    "host_port": 30001,
                    "url": "http://127.0.0.1:30001",
                    "public_url": "https://host.local/app",
                },
            )
        }
    )
    _install_loop(monkeypatch, [])

    result = _invoke(tmp_path, fake)

    assert "not cutover-armed" not in result.output


def test_not_a_directory_exits_1(tmp_path, monkeypatch):
    fake = _client()
    _install_loop(monkeypatch, [])
    missing = tmp_path / "nope"

    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["dev", str(missing)])

    assert result.exit_code == 1
    assert "Not a directory" in result.output


# --- the snapshot itself ------------------------------------------------------


def test_snapshot_prunes_excluded_directories(tmp_path):
    _write(tmp_path, "app.py", "v0")
    _write(tmp_path, "node_modules/dep/index.js", "1")
    _write(tmp_path, ".venv/lib/thing.py", "1")
    _write(tmp_path, "src/main.py", "1")

    snap = dev_mod._snapshot(tmp_path)

    assert set(snap) == {"app.py", "src/main.py"}
