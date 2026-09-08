"""Tests for the DockerRuntime image-build primitive (P4 / S1).

Since P29 leg E ``build_image`` shells out to the ``docker`` CLI with BuildKit
instead of driving docker-py's classic ``api.build`` (inert on Docker >= 29:
zero stream lines, never terminates). These tests therefore monkeypatch
``asyncio.create_subprocess_exec`` with a scripted fake process; the real-Docker
gate lives in ``tests/test_deploy_build_smoke.py``.
"""

import asyncio
import os
import shlex
import sys
import time

import pytest

from nerdit.core.runtime import docker as docker_mod
from nerdit.core.runtime.protocol import BuildError, BuildPlatformError


class _FakeStdout:
    """Byte-line reader over a scripted script of output lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        await asyncio.sleep(0)  # interleave with the consumer, like a real pipe
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeProc:
    def __init__(self, lines: list[bytes], rc: int = 0, *, hang: bool = False) -> None:
        self.stdout = _FakeStdout(lines)
        self._rc = rc
        self._hang = hang
        self.returncode: int | None = None
        self.killed = False

    async def wait(self) -> int:
        if self._hang and not self.killed:
            await asyncio.Event().wait()  # never returns until killed
        self.returncode = self._rc
        return self._rc

    def kill(self) -> None:
        self.killed = True
        self._rc = -9


def _patch_exec(monkeypatch, proc, captured: dict | None = None):
    """Monkeypatch ``asyncio.create_subprocess_exec`` to hand back *proc*."""

    async def _fake_exec(*argv, **kwargs):
        if captured is not None:
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return proc


@pytest.fixture
def docker_on_path(monkeypatch):
    """Pretend the docker CLI resolves, wherever the test host actually has it."""
    monkeypatch.setattr(
        "nerdit.core.runtime.docker.shutil.which", lambda name: "/usr/local/bin/docker"
    )


def _patch_probe(monkeypatch, verdict: str, counter: list | None = None):
    """Pin the (BUG-1) buildx probe to *verdict*, counting calls into *counter*."""

    async def _fake_probe(_docker_bin: str) -> str:
        if counter is not None:
            counter.append(_docker_bin)
        return verdict

    monkeypatch.setattr(docker_mod, "_probe_buildx", _fake_probe)


@pytest.fixture
def buildx_present(monkeypatch):
    """(BUG-1) Pin the failure-path buildx probe to 'present'.

    Applied to every rc!=0 case that is NOT about the probe: without it the
    shared ``create_subprocess_exec`` fake would hand the probe the build's own
    scripted process and a non-zero rc would read as "buildx missing".
    """
    _patch_probe(monkeypatch, "present")


# --------------------------------------------------------------------------- #
# (Codex 3804646864) Real-subprocess process-tree harness                      #
# --------------------------------------------------------------------------- #

# The scripted ``_FakeProc`` above cannot express the defect: the leak is that a
# spawned CLI's OWN child (``docker``'s ``docker-buildx`` plugin, BuildKit's
# helpers) outlives a head-only ``proc.kill()``. So these two tests spawn a real
# ``sh`` stand-in that reproduces exactly that topology — a long-lived
# background grandchild plus a hanging head — and assert the grandchild dies.
# No Docker is involved and nothing but ``sh`` and ``sleep`` is assumed.
_TREE_STANDIN = """#!/bin/sh
sleep 30 &
echo $! > "$NERDIT_TEST_PIDFILE"
echo "#1 building"
sleep 30
"""


def _tree_standin(tmp_path, monkeypatch):
    """Install the stand-in as the resolved ``docker`` CLI; return its pidfile."""
    script = tmp_path / "docker-standin.sh"
    script.write_text(_TREE_STANDIN)
    script.chmod(0o755)
    pidfile = tmp_path / "grandchild.pid"
    monkeypatch.setenv("NERDIT_TEST_PIDFILE", str(pidfile))
    monkeypatch.setattr("nerdit.core.runtime.docker.shutil.which", lambda name: str(script))
    return str(script), pidfile


def _await_grandchild(pidfile) -> int:
    """Block until the stand-in has published its background child's pid."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pidfile.exists():
            text = pidfile.read_text().strip()
            if text:
                return int(text)
        time.sleep(0.02)
    raise AssertionError("the stand-in never wrote its grandchild pid")


def _assert_reaped(pid: int) -> None:
    """Poll until *pid* is gone; fail if it survives the kill."""
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return  # ProcessLookupError (gone) or EPERM (recycled by another uid)
        time.sleep(0.02)
    os.kill(pid, 9)  # never leak out of the test suite itself
    raise AssertionError(f"grandchild {pid} survived — only the process head was killed")


@pytest.mark.asyncio
async def test_build_image_streams_lines(mock_docker, monkeypatch, docker_on_path):
    """build_image yields the CLI's plain-progress lines, blanks dropped."""
    from nerdit.core.runtime.docker import DockerRuntime

    captured: dict = {}
    _patch_exec(
        monkeypatch,
        _FakeProc(
            [
                b"#1 [internal] load build definition from Dockerfile\n",
                b"\n",  # blank lines are never yielded
                b"#5 [2/3] COPY . .\n",
                b"#8 exporting to image DONE\n",
            ]
        ),
        captured,
    )

    runtime = DockerRuntime(client=mock_docker)
    lines = [line async for line in runtime.build_image("/ctx", "nerdit-app/x:1")]

    assert lines == [
        "#1 [internal] load build definition from Dockerfile",
        "#5 [2/3] COPY . .",
        "#8 exporting to image DONE",
    ]

    argv = captured["argv"]
    assert argv[0] == "/usr/local/bin/docker"
    assert argv[1] == "build"
    assert "--progress=plain" in argv
    assert argv[argv.index("-t") + 1] == "nerdit-app/x:1"
    # -f is joined onto the context: the CLI resolves a relative -f against CWD.
    assert argv[argv.index("-f") + 1] == "/ctx/Dockerfile"
    # The context is the last positional.
    assert argv[-1] == "/ctx"
    # Never shell=True, and BuildKit forced on over the inherited environment.
    assert captured["kwargs"]["env"]["DOCKER_BUILDKIT"] == "1"
    assert "shell" not in captured["kwargs"]
    assert captured["kwargs"]["stderr"] is asyncio.subprocess.STDOUT


@pytest.mark.asyncio
async def test_build_image_stamps_ownership_labels(mock_docker, monkeypatch, docker_on_path):
    """Every built image carries ``managed-by`` + ``nerdit-instance`` (Track 0.4).

    Passed as ``--label`` flags, not as a Dockerfile ``LABEL``, so a passthrough
    (user-supplied) Dockerfile is labelled too — the image GC uses the label to
    tell this daemon's images from a co-located sibling's.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    captured: dict = {}
    _patch_exec(monkeypatch, _FakeProc([b"ok\n"]), captured)
    runtime = DockerRuntime(client=mock_docker, instance_id="beta")
    _ = [line async for line in runtime.build_image("/ctx", "nerdit-app/x:1")]

    argv = captured["argv"]
    labels = [argv[i + 1] for i, a in enumerate(argv) if a == "--label"]
    assert labels == ["managed-by=nerdit", "nerdit-instance=beta"]


@pytest.mark.asyncio
async def test_build_image_labels_default_instance(mock_docker, monkeypatch, docker_on_path):
    """A daemon that never configured an instance id builds under ``default``."""
    from nerdit.core.runtime.docker import DockerRuntime

    captured: dict = {}
    _patch_exec(monkeypatch, _FakeProc([b"ok\n"]), captured)
    runtime = DockerRuntime(client=mock_docker)
    _ = [line async for line in runtime.build_image("/ctx", "nerdit-app/x:1")]

    assert "nerdit-instance=default" in captured["argv"]


@pytest.mark.asyncio
async def test_build_image_custom_dockerfile(mock_docker, monkeypatch, docker_on_path):
    from nerdit.core.runtime.docker import DockerRuntime

    captured: dict = {}
    _patch_exec(monkeypatch, _FakeProc([b"ok\n"]), captured)
    runtime = DockerRuntime(client=mock_docker)
    _ = [line async for line in runtime.build_image("/ctx", "t:1", dockerfile="Dockerfile.gen")]

    argv = captured["argv"]
    assert argv[argv.index("-f") + 1] == "/ctx/Dockerfile.gen"


@pytest.mark.asyncio
async def test_build_image_raises_on_nonzero_exit(
    mock_docker, monkeypatch, docker_on_path, buildx_present
):
    """A non-zero exit becomes a BuildError carrying the code + the output tail."""
    from nerdit.core.runtime.docker import DockerRuntime

    _patch_exec(
        monkeypatch,
        _FakeProc(
            [
                b"#5 [2/3] RUN npm ci\n",
                b"ERROR: failed to solve: process /bin/sh -c npm ci did not complete\n",
                # BuildKit's real last line is a deep link, not the diagnosis.
                b"View build details: docker-desktop://dashboard/build/x/y/z\n",
            ],
            rc=1,
        ),
    )

    runtime = DockerRuntime(client=mock_docker)
    seen: list[str] = []
    with pytest.raises(BuildError) as exc:
        async for line in runtime.build_image("/ctx", "t:1"):
            seen.append(line)

    assert "exit 1" in str(exc.value)
    # The ERROR line wins over the trailing deep link — that message is what an
    # agent reads back beside ``last_deploy.reason = "build_failed"``.
    assert "npm ci did not complete" in str(exc.value)
    assert "docker-desktop://" not in str(exc.value)
    # Everything the build printed still streamed into the caller's log capture.
    assert "#5 [2/3] RUN npm ci" in seen


# ---------------------------------------------------------------------------
# (BUG-1) buildx availability: the probe, the asymmetric cache, and the
# failure-path classification it drives.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_failure_classifies_platform_when_buildx_is_missing(
    mock_docker, monkeypatch, docker_on_path
):
    """A failed build on a daemon WITHOUT buildx is a HOST fault, not the app's.

    Falsify-both-ways leg A: same non-zero build, probe says ``missing`` ⇒
    ``BuildPlatformError`` (which ``app_build`` maps to ``PLATFORM_ERROR``).
    """
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.core.runtime.protocol import BuildPlatformError

    _patch_exec(monkeypatch, _FakeProc([b"ERROR: failed to solve\n"], rc=1))
    _patch_probe(monkeypatch, "missing")

    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(BuildPlatformError) as exc:
        async for _ in runtime.build_image("/ctx", "t:1"):
            pass

    message = str(exc.value)
    # It names the missing plugin and the two supported escape hatches.
    assert "buildx" in message
    assert "DOCKER_CONFIG" in message
    # The build output still rides along so the operator sees what broke.
    assert "failed to solve" in message
    # Doctor/agent discipline: no host path leaks out of the runtime.
    assert "/.docker" not in message
    assert "/usr/libexec" not in message


@pytest.mark.asyncio
async def test_build_failure_stays_user_error_when_buildx_is_present(
    mock_docker, monkeypatch, docker_on_path
):
    """Leg B: with buildx present the message is byte-identical to today's."""
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.core.runtime.protocol import BuildPlatformError

    _patch_exec(monkeypatch, _FakeProc([b"ERROR: failed to solve\n"], rc=1))
    _patch_probe(monkeypatch, "present")

    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(BuildError) as exc:
        async for _ in runtime.build_image("/ctx", "t:1"):
            pass

    assert not isinstance(exc.value, BuildPlatformError)
    assert str(exc.value) == "docker build failed (exit 1): ERROR: failed to solve"


@pytest.mark.asyncio
async def test_build_failure_stays_user_error_when_the_buildx_probe_is_unknown(
    mock_docker, monkeypatch, docker_on_path
):
    """Fail-open pin: an inconclusive probe must never accuse the host.

    Reading ``unknown`` as "missing" would reclassify a genuine app build error
    as a platform fault and send the user to fix a machine that is fine.
    """
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.core.runtime.protocol import BuildPlatformError

    _patch_exec(monkeypatch, _FakeProc([b"ERROR: failed to solve\n"], rc=1))
    _patch_probe(monkeypatch, "unknown")

    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(BuildError) as exc:
        async for _ in runtime.build_image("/ctx", "t:1"):
            pass
    assert not isinstance(exc.value, BuildPlatformError)


@pytest.mark.asyncio
async def test_buildx_probe_argv_is_docker_buildx_version(monkeypatch):
    """The probe is ``docker buildx version`` — the CLI's own plugin resolution.

    A stderr regex would be version- and locale-dependent; the probe is the
    authority the classification rests on, so its argv is pinned.
    """
    captured: dict = {}
    _patch_exec(monkeypatch, _FakeProc([], rc=0), captured)

    verdict = await docker_mod._probe_buildx("/usr/local/bin/docker")

    assert verdict == "present"
    assert captured["argv"] == ["/usr/local/bin/docker", "buildx", "version"]
    # The probe must not pollute the build log: both streams go to /dev/null.
    assert captured["kwargs"]["stdout"] is asyncio.subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_buildx_probe_reports_missing_on_a_nonzero_exit(monkeypatch):
    _patch_exec(monkeypatch, _FakeProc([], rc=125))
    assert await docker_mod._probe_buildx("/usr/local/bin/docker") == "missing"


@pytest.mark.asyncio
async def test_buildx_probe_is_unknown_and_reaps_the_child_on_timeout(monkeypatch):
    """A hung probe fails open AND never leaks the child process."""
    proc = _FakeProc([], rc=0, hang=True)
    _patch_exec(monkeypatch, proc)
    monkeypatch.setattr(docker_mod, "_BUILDX_PROBE_TIMEOUT_S", 0.01)

    assert await docker_mod._probe_buildx("/usr/local/bin/docker") == "unknown"
    assert proc.killed is True


@pytest.mark.asyncio
async def test_buildx_probe_is_unknown_when_the_spawn_fails(monkeypatch):
    async def _boom(*argv, **kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
    assert await docker_mod._probe_buildx("/usr/local/bin/docker") == "unknown"


@pytest.mark.asyncio
async def test_buildx_probe_starts_its_own_session(monkeypatch):
    """(Codex 3804646864) The probe child must lead its own session.

    ``start_new_session=True`` is what makes the child's pid equal its pgid, so
    one ``killpg`` reaps the CLI *and* its plugin child. Pinned separately from
    the behavioural test below because it is the precondition that fix rests on.
    """
    captured: dict = {}
    _patch_exec(monkeypatch, _FakeProc([], rc=0), captured)

    assert await docker_mod._probe_buildx("/usr/local/bin/docker") == "present"
    assert captured["kwargs"]["start_new_session"] is True


@pytest.mark.asyncio
async def test_buildx_probe_kills_the_whole_plugin_process_tree_on_timeout(tmp_path, monkeypatch):
    """(Codex 3804646864) A hung probe reaps the CLI's own child too.

    ``docker buildx version`` runs the separate ``docker-buildx`` executable as
    a child, so killing only the ``docker`` head reparents the plugin to init.
    A timed-out verdict is never cached, so every later /doctor call leaked one
    more. Real subprocesses: the defect is a process-topology defect.
    """
    _, pidfile = _tree_standin(tmp_path, monkeypatch)
    monkeypatch.setattr(docker_mod, "_BUILDX_PROBE_TIMEOUT_S", 0.5)

    verdict = await docker_mod._probe_buildx(str(tmp_path / "docker-standin.sh"))

    assert verdict == "unknown"  # still fail-open
    _assert_reaped(_await_grandchild(pidfile))


@pytest.mark.asyncio
async def test_buildx_available_memoizes_present_but_reprobes_missing(
    mock_docker, monkeypatch, docker_on_path
):
    """Asymmetric cache: 'present' sticks, 'missing' always re-probes.

    A plugin does not vanish, so caching the positive is safe; caching the
    negative would force a daemon restart after the operator installs buildx.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    verdicts = ["missing", "missing", "present"]
    calls: list[str] = []

    async def _fake_probe(docker_bin: str) -> str:
        calls.append(docker_bin)
        return verdicts.pop(0) if verdicts else "present"

    monkeypatch.setattr(docker_mod, "_probe_buildx", _fake_probe)

    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.buildx_available() == "missing"
    assert await runtime.buildx_available() == "missing"  # re-probed, not cached
    assert len(calls) == 2
    assert await runtime.buildx_available() == "present"
    assert len(calls) == 3
    assert await runtime.buildx_available() == "present"
    assert len(calls) == 3  # the positive is memoized


@pytest.mark.asyncio
async def test_buildx_available_reports_no_cli_without_the_docker_cli(mock_docker, monkeypatch):
    """(Codex 3804646811) No docker CLI is a DEFINITE host fault, not ``unknown``.

    ``unknown`` is the fail-open value reserved for a probe that could not
    conclude; "the daemon's PATH has no docker CLI" concluded perfectly well,
    and it means every image build on this node will fail. Reporting it as
    ``unknown`` left /doctor ``ok`` and every build classified USER_ERROR.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    monkeypatch.setattr("nerdit.core.runtime.docker.shutil.which", lambda name: None)
    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.buildx_available() == "no_cli"


@pytest.mark.asyncio
async def test_buildx_positive_cache_expires_after_the_ttl(mock_docker, monkeypatch):
    """(Codex 3804646823) The positive memo is bounded, not permanent.

    A plugin CAN vanish (uninstall, a broken package upgrade, a lost DOCKER_CONFIG),
    and a permanent memo made the cache asymmetric in the wrong direction: /doctor
    kept claiming a builder that was gone while failed builds fell back to
    BuildError/USER_ERROR until the daemon restarted. Both legs run against the
    real clock — no clock patching, only the TTL constant moves.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    monkeypatch.setattr(
        "nerdit.core.runtime.docker.shutil.which", lambda name: "/usr/local/bin/docker"
    )

    def _scripted():
        verdicts = ["present", "missing"]
        calls: list[str] = []

        async def _fake_probe(docker_bin: str) -> str:
            calls.append(docker_bin)
            return verdicts.pop(0) if verdicts else "missing"

        return _fake_probe, calls

    # Leg A — inside the window the memo still answers, at zero subprocess cost.
    probe, calls = _scripted()
    monkeypatch.setattr(docker_mod, "_probe_buildx", probe)
    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.buildx_available() == "present"
    assert await runtime.buildx_available() == "present"
    assert len(calls) == 1

    # Leg B — past the window the verdict is re-derived, so a builder that went
    # away is seen without a daemon restart.
    probe, calls = _scripted()
    monkeypatch.setattr(docker_mod, "_probe_buildx", probe)
    monkeypatch.setattr(docker_mod, "_BUILDX_CACHE_TTL_S", 0.0)
    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.buildx_available() == "present"
    assert await runtime.buildx_available() == "missing"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_build_image_does_not_probe_buildx_on_a_successful_build(
    mock_docker, monkeypatch, docker_on_path
):
    """Happy-path cost pin: the probe only runs when the build already failed."""
    from nerdit.core.runtime.docker import DockerRuntime

    _patch_exec(monkeypatch, _FakeProc([b"#8 exporting to image DONE\n"], rc=0))
    calls: list[str] = []
    _patch_probe(monkeypatch, "present", calls)

    runtime = DockerRuntime(client=mock_docker)
    async for _ in runtime.build_image("/ctx", "t:1"):
        pass

    assert calls == []


def test_build_failure_tail_picks_the_informative_line():
    """The BuildError message must never be BuildKit's trailing deep link."""
    from nerdit.core.runtime.docker import _build_failure_tail

    link = "View build details: docker-desktop://dashboard/build/x/y/z"
    # ERROR wins, innermost (last) one first.
    assert _build_failure_tail(["ERROR: first", "#4 ok", "ERROR: real cause", link]) == (
        "ERROR: real cause"
    )
    # No ERROR marker: the last non-chatter line.
    assert _build_failure_tail(["#4 ok", "something odd", link]) == "something odd"
    # Chatter only: better the link than nothing.
    assert _build_failure_tail([link]) == link
    # No output at all.
    assert _build_failure_tail([]) == "no output"


@pytest.mark.asyncio
async def test_build_image_without_the_docker_cli_is_a_platform_error(mock_docker, monkeypatch):
    """(Codex 3804646811) Leg A: a missing docker CLI is a HOST fault.

    It is still a ``BuildError`` subclass (every ``except ContainerRuntimeError``
    settle path is unchanged), but the *type* now carries the distinction, so
    the generation settles PLATFORM_ERROR instead of telling the caller to fix
    an application whose source is fine. The message stays path-free: it is
    persisted into ``config['last_deploy']`` and re-served over /diagnose to
    every role, where the daemon's PATH is not the caller's business.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    monkeypatch.setattr("nerdit.core.runtime.docker.shutil.which", lambda name: None)
    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(BuildPlatformError) as exc:
        async for _ in runtime.build_image("/ctx", "t:1"):
            pass
    msg = str(exc.value)
    assert "docker CLI not found" in msg
    assert "/usr/" not in msg
    assert "/.docker" not in msg


@pytest.mark.asyncio
async def test_build_image_kills_the_build_process_tree_when_abandoned(
    mock_docker, tmp_path, monkeypatch
):
    """(Codex 3804646864) The abandoned-build promise covers the WHOLE tree.

    ``build_image``'s ``finally`` already promised to "never leave a detached
    ``docker build`` running against the daemon", but a head-only kill leaves
    BuildKit's own children burning CPU and disk. Same real-subprocess topology
    as the probe test, driven through the generator's drain path.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    captured: dict = {}
    real_exec = asyncio.create_subprocess_exec

    async def _spy_exec(*argv, **kwargs):
        captured["kwargs"] = kwargs
        return await real_exec(*argv, **kwargs)

    _, pidfile = _tree_standin(tmp_path, monkeypatch)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spy_exec)

    runtime = DockerRuntime(client=mock_docker)
    agen = runtime.build_image("/ctx", "t:1")
    assert await agen.__anext__() == "#1 building"
    grandchild = _await_grandchild(pidfile)
    await agen.aclose()

    assert captured["kwargs"]["start_new_session"] is True
    _assert_reaped(grandchild)


@pytest.mark.asyncio
async def test_build_image_wraps_spawn_failure(mock_docker, monkeypatch, docker_on_path):
    """An OSError spawning the CLI is delivered as a BuildError."""
    from nerdit.core.runtime.docker import DockerRuntime

    async def _boom(*argv, **kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)

    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(BuildError) as exc:
        async for _ in runtime.build_image("/ctx", "t:1"):
            pass
    assert "fork failed" in str(exc.value)


@pytest.mark.asyncio
async def test_build_image_kills_child_when_abandoned(mock_docker, monkeypatch, docker_on_path):
    """Closing the generator early (the P20 drain) kills the build subprocess.

    The classic path had no wall-clock bound and this one adds none: consumer
    cancellation IS the bound, so it must actually reap the child rather than
    leave a detached ``docker build`` running against the daemon.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    proc = _FakeProc([b"#1 line\n", b"#2 line\n", b"#3 line\n"], hang=True)
    _patch_exec(monkeypatch, proc)

    runtime = DockerRuntime(client=mock_docker)
    agen = runtime.build_image("/ctx", "t:1")
    first = await agen.__anext__()
    assert first == "#1 line"
    await agen.aclose()

    assert proc.killed is True


@pytest.mark.asyncio
async def test_build_image_thread_handoff_many_lines(mock_docker, monkeypatch, docker_on_path):
    """A large stream drains fully without hanging or dropping lines."""
    from nerdit.core.runtime.docker import DockerRuntime

    n = 500
    _patch_exec(
        monkeypatch,
        _FakeProc([f"Step {i}\n".encode() for i in range(n)] + [b"Successfully built abc\n"]),
    )

    runtime = DockerRuntime(client=mock_docker)
    lines: list[str] = []
    async for line in runtime.build_image("/ctx", "t:1"):
        lines.append(line)
        await asyncio.sleep(0)

    assert len(lines) == n + 1
    assert lines[0] == "Step 0"
    assert lines[-1] == "Successfully built abc"


@pytest.mark.asyncio
async def test_remove_image_ignores_missing(mock_docker):
    import docker  # mocked module
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.images.remove.side_effect = docker.errors.ImageNotFound("gone")
    runtime = DockerRuntime(client=mock_docker)
    # Must not raise
    await runtime.remove_image("nerdit-app/x:1", force=True)
    mock_docker.images.remove.assert_called_once_with("nerdit-app/x:1", force=True)


@pytest.mark.asyncio
async def test_remove_image_swallows_in_use(mock_docker):
    """APIError (e.g. 'image is in use') is logged and swallowed — prune is best-effort."""
    import docker  # mocked module
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.images.remove.side_effect = docker.errors.APIError("image is referenced")
    runtime = DockerRuntime(client=mock_docker)
    await runtime.remove_image("nerdit-app/x:1")  # must not raise


# --- long build lines (P29 review round-2, Codex 3803596893) -----------------
#
# These two need a REAL subprocess: the hazard lives in asyncio's
# ``StreamReader`` limit, which the scripted ``_FakeStdout`` above cannot model.
# So the fake binary is a real executable on disk and
# ``create_subprocess_exec`` is left alone.


@pytest.fixture
def fake_docker_binary(tmp_path, monkeypatch):
    """Install a real executable that stands in for the ``docker`` CLI."""

    def _install(script_body: str) -> None:
        script = tmp_path / "docker"
        script.write_text(
            "#!/bin/sh\nexec " + sys.executable + " -c " + shlex.quote(script_body) + "\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        monkeypatch.setattr("nerdit.core.runtime.docker.shutil.which", lambda name: str(script))

    return _install


@pytest.mark.asyncio
async def test_build_image_survives_a_line_over_64_kib(mock_docker, fake_docker_binary):
    """A 200 KiB BuildKit line must stream, not raise a bare ``ValueError``.

    asyncio's default subprocess stream limit is 64 KiB and ``readline()``
    raises ``ValueError`` past it — which is NOT a ``ContainerRuntimeError``, so
    ``AppBuildManager.build`` (which catches only that) never reached
    ``_settle_failed_generation`` and the row wedged in ``building`` forever.
    ``--progress=plain`` legitimately emits such lines (echoed long ``RUN``s,
    inline-cache base64, big ``COPY`` file lists).
    """
    from nerdit.core.runtime.docker import DockerRuntime

    fake_docker_binary(
        "import sys; sys.stdout.write('L' * 204800 + '\\n'); sys.stdout.write('ok\\n')"
    )
    runtime = DockerRuntime(client=mock_docker)
    lines = [line async for line in runtime.build_image("/ctx", "nerdit-app/x:1")]

    assert lines[0] == "L" * 204800
    assert lines[-1] == "ok"


@pytest.mark.asyncio
async def test_build_image_marks_a_line_over_the_raised_limit(mock_docker, fake_docker_binary):
    """Past 1 MiB the line is replaced by one marker and the stream continues.

    CPython's ``StreamReader`` clears its buffer before raising, so the loop
    resumes safely; the marker keeps the truncation visible in the log tail
    instead of silently losing output — and the ``ValueError`` never escapes.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    fake_docker_binary(
        "import sys; sys.stdout.write('M' * 2_200_000 + '\\n'); sys.stdout.write('ok\\n')"
    )
    runtime = DockerRuntime(client=mock_docker)
    lines = [line async for line in runtime.build_image("/ctx", "nerdit-app/x:1")]

    assert any(line == docker_mod._BUILD_LINE_TRUNCATED for line in lines)
    assert lines[-1] == "ok"
    assert not any(len(line) > docker_mod._BUILD_STREAM_LIMIT for line in lines)
