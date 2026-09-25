"""Remove an installation without relying on a running daemon or network calls.

Load corrupt config tolerantly. Stop the unit, verify the daemon is gone, then
hold the exclusive data-dir lock before removing anything, including the unit.
Record stop failures; never orphan a supervised daemon by deleting its unit.

Capture Caddy's PID and print certificate-derived trust-removal commands before
deleting their files. Reaping is PID-file-only: an adopted Caddy with no PID
file may survive until reboot. Escape dynamic Rich text.
"""

from __future__ import annotations

import os
import shutil
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer
from rich.markup import escape

from nerdit.cli.commands.trust import _untrust_commands
from nerdit.cli.display import _plain, console, fmt_bytes
from nerdit.config.settings import NerditSettings, load_settings
from nerdit.utils.certs import parse_single_certificate
from nerdit.utils.disk import du_bytes, resolve_archive_dir
from nerdit.utils.install_layout import (
    InstallLayout,
    ServiceUnit,
    detect_install_layout,
    detect_service_unit,
    path_present,
)

#: Container label selectors — the exact pair every managed container carries.
_MANAGED_LABEL = "managed-by=nerdit"

#: How long a daemon gets to drain after SIGTERM before SIGKILL. Deliberately
#: generous: the daemon's own graceful shutdown bounds in-flight runs/builds at
#: ~30 s (``/daemon/restart`` drain + uvicorn ``timeout_graceful_shutdown``), so
#: anything under that would routinely escalate to SIGKILL on a healthy box.
_DAEMON_DRAIN_S = 60.0

#: Filename of the whole-life shared flock the daemon holds on its data dir
#: (D6/H1). Taking it ``LOCK_EX`` is the authoritative "no daemon owns this data
#: dir" answer, and the single gate every removal below sits behind.
_DATA_DIR_LOCK = ".restore.lock"


class _DockerImages(Protocol):
    """Structural shape of `docker.DockerClient.images` (no upstream stubs)."""

    def list(self) -> list[Any]: ...
    def remove(self, image: str) -> None: ...


class _DockerContainers(Protocol):
    """Structural shape of `docker.DockerClient.containers` (no upstream stubs)."""

    def list(self, all: bool = ..., filters: dict[str, Any] | None = ...) -> list[Any]: ...


class _DockerClient(Protocol):
    """Structural shape of a docker-py client, narrowed to what this module uses."""

    images: _DockerImages
    containers: _DockerContainers


def _default_docker_client():  # noqa: ANN202 — a lazily-imported docker client
    """Construct docker-py lazily so importing this module does not require Docker."""
    import docker

    return docker.from_env()


# --------------------------------------------------------------------------- #
# manifest (pure, unit-testable)
# --------------------------------------------------------------------------- #


@dataclass
class _ExtraPath:
    """A path outside (or logically separate from) `data_dir` to remove.

    `kind` drives the `--keep-data` deletion policy: `"key_file"` and
    `"config"` are kept whenever data is kept (the data is useless without the
    key, and the config lets the daemon restart), while `pid` / `boot_log` /
    `upload_dir` are kept only when they fall *inside* the preserved
    `data_dir`.
    """

    path: Path
    label: str
    kind: str  # "pid" | "boot_log" | "upload_dir" | "key_file" | "config"


@dataclass
class UninstallManifest:
    """Read-only inventory of uninstall targets, safe to build for dry runs."""

    data_dir: Path
    data_dir_bytes: int
    extra_paths: list[_ExtraPath] = field(default_factory=list)
    archive_dir: Path | None = None
    daemon_pid: int | None = None
    caddy_pid: int | None = None
    ca_certs: list[Path] = field(default_factory=list)
    containers: list[tuple[str, str, str]] = field(default_factory=list)
    images: list[tuple[str, int]] = field(default_factory=list)
    docker_error: str | None = None
    purge_images: bool = False
    instance_id: str = "default"
    base_image_tags: list[str] = field(default_factory=list)
    owned_repos: set[str] = field(default_factory=set)
    db_error: str | None = None
    # (P30 D-P30-8/9) The installer-made layout, when there is one. A unit-
    # managed daemon that is only pid-killed is respawned within seconds, and
    # code left under /opt/nerdit (or ~/.nerdit/versions) would keep a `nerdit`
    # on PATH after its data is gone — both are worse than no uninstall.
    service_unit: ServiceUnit | None = None
    layout: InstallLayout | None = None


def _is_under(child: Path, parent: Path) -> bool:
    """True iff *child* is *parent* or nested under it (symlink-resolved)."""
    try:
        c = child.resolve()
        p = parent.resolve()
    except OSError:
        c, p = child.absolute(), parent.absolute()
    return c == p or p in c.parents


def _read_pid(path: Path) -> int | None:
    """Read a pid file's integer content; `None` on missing/garbage.

    Deliberately does NOT go through `DaemonLifecycle.is_running` — that
    call unlinks a stale pid file as a side effect, which would race the
    ordered deletion below.
    """
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    # A corrupt/tampered "0" or negative would make os.kill signal the whole
    # process group (0) or every permitted process (-1) — never a valid target.
    return pid if pid > 0 else None


def _base_image_tags(settings: NerditSettings) -> list[str]:
    """The configured (not hard-coded) base-image tags removed under `--purge-images`.

    Ollama/vLLM/Postgres/Redis images are operator-configurable, so read the
    live settings rather than the defaults (recon fact #7).
    """
    return [
        settings.containers.default_image,
        settings.models.ollama_image,
        settings.models.vllm_image,
        settings.databases.postgres_image,
        settings.databases.redis_image,
    ]


def _owned_app_repos(db_path: Path) -> tuple[set[str], str | None]:
    """Read this instance's app image repositories before deleting its database.

    Attribute image ownership from the database's persisted image_repo,
    image and previous_image references, never service names shared by model/DB
    rows. Missing or unreadable databases return an empty set and reason; callers
    skip app-image removal rather than guess ownership.
    """
    import json
    import sqlite3

    if not db_path.exists():
        return set(), "no database file"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT config FROM jobs WHERE kind = 'service' AND config IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return set(), f"unreadable database ({exc})"

    repos: set[str] = set()
    for (cfg_text,) in rows:
        try:
            cfg = json.loads(cfg_text)
        except (TypeError, ValueError):
            continue
        if not isinstance(cfg, dict):
            continue
        repo = cfg.get("image_repo")
        if isinstance(repo, str):
            repos.add(repo)
        for key in ("image", "previous_image"):
            tag = cfg.get(key)
            if isinstance(tag, str) and tag:
                repos.add(tag.split(":", 1)[0])
    return {r for r in repos if r.startswith("nerdit-app/")}, None


def _is_target_image(tag: str, extra_tags: list[str], owned_repos: set[str]) -> bool:
    """Whether an image *tag* is one THIS instance owns: a build of one of its
    own `nerdit-app/<name>` repos, plus the configured base set when
    `extra_tags` is populated (`--purge-images` — base images are shared
    across instances by nature, hence the explicit opt-in).
    """
    return tag.split(":", 1)[0] in owned_repos or tag in extra_tags


def _container_filters(instance_id: str) -> dict[str, list[str]]:
    """The exact label filter pinning containers to THIS install (recon fact #7)."""
    return {"label": [_MANAGED_LABEL, f"nerdit-instance={instance_id}"]}


def _describe_docker_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _c_id(c: object) -> str:
    return (getattr(c, "id", "") or "")[:12]


def _c_name(c: object) -> str:
    return getattr(c, "name", "") or ""


def _c_status(c: object) -> str:
    return getattr(c, "status", "") or ""


def _list_target_image_tags(
    client: _DockerClient, extra_tags: list[str], owned_repos: set[str]
) -> list[tuple[str, int]]:
    """Every owned `(tag, size)` currently present on the docker host."""
    out: list[tuple[str, int]] = []
    for img in client.images.list():
        for tag in getattr(img, "tags", None) or []:
            if _is_target_image(tag, extra_tags, owned_repos):
                size = (getattr(img, "attrs", None) or {}).get("Size", 0) or 0
                out.append((tag, size))
    return out


def _probe_docker(
    factory: Callable[[], _DockerClient],
    instance_id: str,
    *,
    extra_tags: list[str],
    owned_repos: set[str],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, int]], str | None]:
    """Read-only docker inventory for the manifest.

    Returns `(containers, images, error)`; a non-`None` error means docker is
    unreachable (client construction OR the first call failed) — the manifest
    still builds and teardown degrades to printed manual commands (§2.7).
    """
    try:
        client = factory()
        # all=True is the whole point: stopped/exited managed containers are
        # invisible to the default all=False listing (recon fact #2), and a
        # stopped app container is exactly what we must reap.
        containers = client.containers.list(all=True, filters=_container_filters(instance_id))
        cinfo = [(_c_id(c), _c_name(c), _c_status(c)) for c in containers]
        images = _list_target_image_tags(client, extra_tags, owned_repos)
    except Exception as exc:  # noqa: BLE001 — any docker/transport failure degrades
        return [], [], _describe_docker_error(exc)
    return cinfo, images, None


def build_manifest(
    settings: NerditSettings,
    *,
    purge_images: bool,
    docker_client_factory: Callable[[], _DockerClient],
) -> UninstallManifest:
    """Compute the read-only `UninstallManifest` (no side effects)."""
    data_dir = Path(settings.data_dir).expanduser()
    instance_id = settings.daemon.instance_id
    base_tags = _base_image_tags(settings)
    extra_tags = base_tags if purge_images else []

    # Paths that are NOT structurally under data_dir (recon fact #3) and so need
    # explicit handling — read here so a later rmtree(data_dir) cannot hide them.
    # File-valued artifacts (pid/logs/key/config) require is_file(): a config
    # typo pointing one at an existing DIRECTORY must be refused as invalid,
    # never recursively removed (Codex P2).
    extra_paths: list[_ExtraPath] = []

    def _add_file(path: Path, label: str, kind: str) -> None:
        if not path.exists():
            return
        if not path.is_file():
            console.print(
                f"[yellow]Ignoring {escape(label)} ({_plain(path)}): not a regular file.[/yellow]"
            )
            return
        extra_paths.append(_ExtraPath(path, label, kind))

    pid_file = Path(settings.daemon.pid_file).expanduser()
    _add_file(pid_file, "daemon pid file", "pid")
    for name in ("nerditd.boot.log", "nerditd.boot.log.1"):
        _add_file(pid_file.parent / name, "daemon boot log", "boot_log")
    upload_dir = Path(settings.daemon.upload_dir).expanduser()
    if upload_dir.is_dir() and not _is_under(upload_dir, data_dir):
        extra_paths.append(_ExtraPath(upload_dir, "upload dir", "upload_dir"))
    key_override = settings.security.secrets_key_file
    if key_override:
        key_path = Path(key_override).expanduser()
        if not _is_under(key_path, data_dir):
            _add_file(key_path, "secrets key (override)", "key_file")

    # The daemon config lives at a FIXED ~/.nerdit/config.toml, independent of
    # data_dir (settings.py). In the default install it sits inside data_dir and
    # dies with the rmtree; with a custom data_dir it is separate and would
    # otherwise orphan ~/.nerdit (blocking the empty-dir rmdir below), so
    # enumerate it explicitly (deleted normally, kept under --keep-data).
    config_toml = Path.home() / ".nerdit" / "config.toml"
    if not _is_under(config_toml, data_dir):
        _add_file(config_toml, "daemon config.toml", "config")

    # Saved CA copies live under the literal home (~/.nerdit), which trust.py
    # hardcodes regardless of a data_dir override (recon fact #3/#5).
    ca_dir = Path.home() / ".nerdit"
    ca_certs = sorted(ca_dir.glob("nerdit-root-*.crt"))

    # Audit archive: reported + LEFT IN PLACE only when it resolves OUTSIDE
    # data_dir (a within-data_dir archive dies with the rmtree).
    archive_dir = None
    resolved_archive = resolve_archive_dir(settings.retention, data_dir)
    if (
        resolved_archive is not None
        and resolved_archive.exists()
        and not _is_under(resolved_archive, data_dir)
    ):
        archive_dir = resolved_archive

    owned_repos, db_error = _owned_app_repos(data_dir / "nerdit.db")
    containers, images, docker_error = _probe_docker(
        docker_client_factory, instance_id, extra_tags=extra_tags, owned_repos=owned_repos
    )

    # Never walk an unsafe tree: data_dir = "/" or $HOME would make even a
    # --dry-run traverse it for sizing before _validate_targets refuses.
    data_dir_bytes = 0 if _unsafe_target("data dir", data_dir) else du_bytes(data_dir)

    return UninstallManifest(
        data_dir=data_dir,
        data_dir_bytes=data_dir_bytes,
        extra_paths=extra_paths,
        archive_dir=archive_dir,
        daemon_pid=_read_pid(pid_file),
        caddy_pid=_read_pid(data_dir / "caddy.pid"),
        ca_certs=ca_certs,
        containers=containers,
        images=images,
        docker_error=docker_error,
        purge_images=purge_images,
        instance_id=instance_id,
        base_image_tags=base_tags,
        owned_repos=owned_repos,
        db_error=db_error,
        service_unit=detect_service_unit(),
        layout=detect_install_layout(),
    )


# --------------------------------------------------------------------------- #
# process teardown
# --------------------------------------------------------------------------- #


def _process_command(pid: int) -> str | None:
    """Best-effort command line of *pid* via `ps` (POSIX).

    `None` when identity cannot be checked (no `ps`, timeout); `""` when
    the process does not exist.
    """
    import subprocess

    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def _terminate(
    pid: int, label: str, *, grace_s: float = 5.0, expect_cmd: str | None = None
) -> bool:
    """SIGTERM → poll → SIGKILL a process, tolerantly.

    `True` when the process is gone (already dead, or reaped here); `False`
    only for a `PermissionError` — the pid belongs to another user, so
    the caller must NOT touch its pid file. `ProcessLookupError` anywhere
    means the process is already gone (`True`).

    `expect_cmd` guards against pid reuse: a hard-crashed daemon leaves its
    pid file behind and the OS may hand that pid to an unrelated process — an
    identity mismatch skips the kill entirely (stale file, gone-equivalent).
    Unverifiable identity (no `ps`) falls through to signalling, preserving
    the reap on minimal systems.
    """
    if expect_cmd is not None:
        cmd = _process_command(pid)
        if cmd:  # a live process that is NOT ours — never signal it
            if expect_cmd not in cmd:
                console.print(
                    f"[yellow]pid {pid} is not {escape(label)} (found: "
                    f"{escape(cmd[:60])}) — stale pid file, not signalled.[/yellow]"
                )
                return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        console.print(f"[yellow]No permission to stop {escape(label)} (pid {pid}).[/yellow]")
        return False

    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)

    # Still alive after the grace window — escalate.
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    final = time.monotonic() + 1.0
    while time.monotonic() < final:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)
    console.print(f"[yellow]{escape(label)} (pid {pid}) did not exit.[/yellow]")
    return False


# --------------------------------------------------------------------------- #
# liveness (shared with ``nerdit exit`` — the two must never disagree)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DaemonLiveness:
    """Daemon liveness evidence shared by exit and uninstall."""

    alive: bool
    pid: int | None
    #: ``"pidfile"`` (a live process whose identity checks out), ``"data_dir_lock"``
    #: (something still holds the flock) or ``"none"``.
    evidence: str

    def describe(self) -> str:
        if not self.alive:
            return "no daemon is running on this data dir"
        if self.evidence == "pidfile":
            return f"a nerditd is running (pid {self.pid})"
        return "a daemon still holds the data dir lock (pid unknown)"


def _pid_alive(pid: int) -> bool:
    """Whether *pid* exists. A `PermissionError` means it exists as someone else."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_data_dir_lock(data_dir: Path, *, attempts: int = 10) -> tuple[bool, int | None]:
    """Take the exclusive data-dir flock. Returns `(free, fd)`.

    `free` is False **only** when something still holds it — i.e. a daemon is
    alive on this data dir. `fd` is `None` when there is nothing to lock
    (non-POSIX, or no data dir yet), which counts as free; the caller must close
    a non-`None` fd when the deletion phase is over.
    """
    if os.name != "posix" or not data_dir.exists():
        return True, None
    import fcntl

    fd = os.open(str(data_dir / _DATA_DIR_LOCK), os.O_CREAT | os.O_RDWR, 0o600)
    for _ in range(attempts):
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True, fd
        except BlockingIOError:
            time.sleep(0.2)
    os.close(fd)
    return False, None


def _data_dir_is_free(data_dir: Path) -> bool:
    """Read-only probe of the same flock: acquire, release immediately.

    Never creates the lock file — a data dir that has never hosted a daemon has
    none, and `nerdit exit` must not leave one behind just by asking.
    """
    if os.name != "posix":
        return True
    lock_path = data_dir / _DATA_DIR_LOCK
    if not lock_path.exists():
        return True
    import fcntl

    try:
        fd = os.open(str(lock_path), os.O_RDWR)
    except OSError:
        return True  # unreadable: no evidence of a daemon, and none to gain
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def probe_daemon(*, data_dir: Path, pid_file: Path, expect_cmd: str = "nerdit") -> DaemonLiveness:
    """Check daemon liveness through PID identity, then the data-dir lock.

    The identity check rejects reused PIDs; the lock detects a live daemon even
    when its PID file was deleted.
    """
    pid = _read_pid(pid_file)
    if pid is not None and _pid_alive(pid):
        cmd = _process_command(pid)
        # Empty/None = unverifiable (no ``ps``, or a race): assume it is ours,
        # matching ``_terminate``'s fall-through. A non-empty NON-matching
        # command line is a reused pid, so the pid file proves nothing.
        if not cmd or expect_cmd in cmd:
            return DaemonLiveness(alive=True, pid=pid, evidence="pidfile")
    if not _data_dir_is_free(data_dir):
        return DaemonLiveness(alive=True, pid=None, evidence="data_dir_lock")
    return DaemonLiveness(alive=False, pid=None, evidence="none")


# --------------------------------------------------------------------------- #
# docker teardown
# --------------------------------------------------------------------------- #


def _teardown_docker(
    factory: Callable[[], _DockerClient],
    instance_id: str,
    *,
    extra_tags: list[str],
    owned_repos: set[str],
) -> tuple[int, int, list[str]]:
    """Remove managed containers then owned images. Returns `(removed_containers,
    removed_images, leftover_image_tags)`; per-item failures are reported, never
    raised.
    """
    client = factory()
    removed_c = 0
    for c in client.containers.list(all=True, filters=_container_filters(instance_id)):
        try:
            c.remove(force=True)  # force covers a still-running container
            removed_c += 1
        except Exception as exc:  # noqa: BLE001 — best-effort per container
            console.print(
                f"[yellow]Could not remove container {_plain(_c_name(c))}: {_plain(exc)}[/yellow]"
            )

    targets = [tag for tag, _size in _list_target_image_tags(client, extra_tags, owned_repos)]
    removed_i = 0
    for tag in targets:
        try:
            client.images.remove(tag)
            removed_i += 1
        except Exception as exc:  # noqa: BLE001 — APIError/in-use etc.
            console.print(f"[yellow]left: {_plain(tag)} ({_plain(exc)})[/yellow]")
    # Re-list: images.remove swallows some failures, so trust the host, not the
    # return count (the remove_image-swallows-errors lesson).
    leftover = [tag for tag, _size in _list_target_image_tags(client, extra_tags, owned_repos)]
    return removed_c, removed_i, leftover


def _print_docker_manual(manifest: UninstallManifest) -> None:
    """The exact manual teardown commands, with the real instance id inlined (§2.7)."""
    console.print(
        "[yellow]Docker is unreachable — remove the containers and images "
        "manually once it is back:[/yellow]"
    )
    iid = escape(manifest.instance_id)
    console.print(
        f"[dim]  docker ps -aq --filter label={_MANAGED_LABEL} "
        f"--filter label=nerdit-instance={iid} | xargs -r docker rm -f[/dim]"
    )
    console.print(
        "[dim]  docker images --format '{{.Repository}}:{{.Tag}}' "
        "| grep '^nerdit-app/' | xargs -r docker rmi[/dim]"
    )
    console.print(
        "[dim]  (the image command removes EVERY instance's nerdit-app images "
        "— skip it if another Nerdit instance shares this docker host)[/dim]"
    )


# --------------------------------------------------------------------------- #
# service-unit teardown (P30 D-P30-8)
# --------------------------------------------------------------------------- #


def _run_unit_command(argv: list[str], *, note: str = "") -> tuple[bool, str | None]:
    """Run a service-manager command, returning success and failure detail.

    Never raise. Callers must handle failed stops; disable/bootout failures are
    harmless only after the data dir is proven free.
    """
    import subprocess

    label = " ".join(argv)
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv from install_layout, no shell
            argv, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        console.print(f"[yellow]Could not run '{escape(label)}': {_plain(exc)}[/yellow]")
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        detail = next(
            (line.strip() for line in (proc.stderr or "").splitlines() if line.strip()),
            f"exit {proc.returncode}",
        )
        console.print(f"[yellow]'{escape(label)}' failed ({_plain(detail)}){escape(note)}[/yellow]")
        return False, detail
    return True, None


def _stop_unit(unit: ServiceUnit, *, data_dir: Path, pid_file: Path) -> tuple[bool, str]:
    """Stop through the service manager and wait for the daemon to release its state.

    Do not disable or delete the unit. Signalling alone may trigger respawn, and
    a failed stop must be reported before any removal.
    """
    label = " ".join(unit.stop_argv)
    ok, detail = _run_unit_command(unit.stop_argv)
    if ok:
        # launchctl bootout can return before the daemon finishes draining.
        deadline = time.monotonic() + _DAEMON_DRAIN_S
        while probe_daemon(data_dir=data_dir, pid_file=pid_file).alive:
            if time.monotonic() >= deadline:
                detail = f"daemon did not stop within {_DAEMON_DRAIN_S:g}s"
                console.print(f"[yellow]{detail} after '{escape(label)}'.[/yellow]")
                return False, f"'{label}' succeeded, but {detail}"
            time.sleep(0.2)
        console.print(f"[dim]Service unit stopped ('{escape(label)}').[/dim]")
        return True, f"'{label}' succeeded"
    return False, f"'{label}' failed ({detail})"


def _remove_unit(unit: ServiceUnit) -> None:
    """Disable and remove the unit only after the data-dir lock proves the daemon is gone."""
    if unit.disable_argv != unit.stop_argv:  # launchd's bootout is both
        _run_unit_command(unit.disable_argv, note=" — continuing")
    if path_present(unit.unit_path):
        _delete_path(unit.unit_path)
        console.print(f"[dim]Removed the service unit ({_plain(unit.unit_path)}).[/dim]")
    if unit.kind == "systemd-system":
        _run_unit_command(["systemctl", "daemon-reload"], note=" — continuing")
    elif unit.kind == "systemd-user":
        _run_unit_command(["systemctl", "--user", "daemon-reload"], note=" — continuing")


# --------------------------------------------------------------------------- #
# render + CA hint
# --------------------------------------------------------------------------- #


def _render_manifest(manifest: UninstallManifest, *, keep_data: bool) -> None:
    console.print("[bold]Nerdit uninstall plan[/bold]")
    console.print(
        f"  data dir:   {_plain(manifest.data_dir)} "
        f"({fmt_bytes(manifest.data_dir_bytes)})"
        + ("  [dim](kept — --keep-data)[/dim]" if keep_data else "")
    )
    if manifest.daemon_pid is not None:
        console.print(f"  daemon pid: {manifest.daemon_pid}")
    if manifest.caddy_pid is not None:
        console.print(f"  caddy pid:  {manifest.caddy_pid}")

    if manifest.service_unit is not None:
        console.print(
            f"  service unit ({escape(manifest.service_unit.kind)}): "
            f"{_plain(manifest.service_unit.unit_path)}"
        )
    if manifest.layout is not None:
        layout = manifest.layout
        console.print(
            f"  install ({escape(layout.mode)}, version "
            f"{_plain(layout.current_version or 'unknown')}):"
        )
        console.print(f"    versions dir: {_plain(layout.versions_dir)}")
        console.print(f"    current link: {_plain(layout.current)}")
        console.print(f"    shim:         {_plain(layout.shim)}")

    for ep in manifest.extra_paths:
        console.print(f"  {escape(ep.label)}: {_plain(ep.path)}")
    for cert in manifest.ca_certs:
        console.print(f"  saved CA cert: {_plain(cert)}")
    if manifest.archive_dir is not None:
        console.print(f"  audit archive (left in place): {_plain(manifest.archive_dir)}")

    if manifest.docker_error is not None:
        console.print(f"  docker: [yellow]unreachable ({_plain(manifest.docker_error)})[/yellow]")
    else:
        console.print(f"  containers: {len(manifest.containers)}")
        for _id, name, status in manifest.containers:
            console.print(f"    - {_plain(name)} [dim]({_plain(status)})[/dim]")
        console.print(f"  images: {len(manifest.images)}")
        for tag, size in manifest.images:
            console.print(f"    - {_plain(tag)} [dim]({fmt_bytes(size)})[/dim]")
        if manifest.db_error is not None:
            # Without a readable DB there is no way to tell this instance's
            # nerdit-app/* images from another instance's on a shared host, so
            # none are removed (the manual command is printed instead).
            console.print(
                f"[yellow]  app images unknown ({_plain(manifest.db_error)}) — "
                "none will be removed; a manual cleanup command is printed "
                "instead.[/yellow]"
            )
        if not manifest.purge_images:
            console.print("[dim]  (base images kept — pass --purge-images to remove them)[/dim]")

    console.print("")
    if keep_data:
        console.print(
            "[bold yellow]This removes Nerdit's containers, images, the daemon "
            "pid/boot logs and any trust artifacts. --keep-data preserves the "
            "data dir and the secrets master key.[/bold yellow]"
        )
    else:
        console.print(
            "[bold red]This permanently deletes the Nerdit database, all service "
            "data, all secrets AND the secrets master key. Backups under "
            f"{escape(str(manifest.data_dir / 'backups'))} (which contain the "
            "master key) are deleted too. This cannot be undone.[/bold red]"
        )


def _print_untrust_hint(manifest: UninstallManifest) -> None:
    """Print certificate-derived removal commands before deleting the saved certificate."""
    if not manifest.ca_certs:
        console.print("[dim]No locally saved CA certificates found — nothing to untrust.[/dim]")
        return
    console.print(
        "[bold]This machine still trusts the Nerdit internal CA.[/bold] "
        "Remove it from the OS trust store with these commands (not run for you):"
    )
    for cert in manifest.ca_certs:
        try:
            parsed = parse_single_certificate(cert.read_text())
        except (OSError, ValueError):
            console.print(f"[yellow]  (could not parse {_plain(cert)})[/yellow]")
            continue
        commands = _untrust_commands(parsed)
        if commands is None:
            console.print(f"[dim]  remove {_plain(cert)} from your trust store manually[/dim]")
            continue
        for cmd in commands:
            console.print(f"[dim]  {' '.join(escape(part) for part in cmd)}[/dim]")
    console.print("[dim]  or run 'nerdit untrust' before removing the package.[/dim]")


# --------------------------------------------------------------------------- #
# deletion
# --------------------------------------------------------------------------- #


def _unsafe_target(label: str, path: Path) -> str | None:
    """First reason *path* is too dangerous to `rmtree`, else `None`.

    `[nerdit].data_dir` is an unvalidated `str` (settings.py) mapped verbatim
    from `config.toml` — an empty value expands to `Path('.')` and a relative
    one is CWD-relative, so an unguarded `rmtree` would wipe the working
    directory. Fail closed on a non-absolute target, a root, `$HOME`, the CWD,
    or any ancestor of them.
    """
    if not path.is_absolute():
        return f"{label} is not an absolute path ({path!r})"
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    if resolved == resolved.parent:  # a filesystem root ('/')
        return f"{label} is a filesystem root ({resolved})"
    # The live $HOME, PLUS both recorded homes. The live one is the original
    # guard and must stay; the recorded ones exist because adoption rewrites
    # $HOME mid-run, so afterwards `Path.home()` is the ADOPTED home and this
    # check would silently stop protecting the invoking user's.
    homes: list[Path] = []
    try:
        homes.append(Path.home())
    except (OSError, RuntimeError):
        pass
    homes.extend(h for h in (_INVOKING_HOME, _ADOPTED_HOME) if h is not None)
    for home in homes:
        try:
            if home.resolve().is_relative_to(resolved):
                return f"{label} is a home directory or contains one ({resolved})"
        except OSError:
            continue
    if Path.cwd().resolve().is_relative_to(resolved):
        return f"{label} is or contains the current directory ({resolved})"

    # Containment. Once a system unit's home has been adopted, every
    # config-derived target comes from a file an unprivileged principal can
    # write while this command runs as ROOT — so a denylist is the wrong shape:
    # `/etc`, `/usr`, `/var` and `/boot` all pass one. Require the target to sit
    # inside the tree the adoption exists to reach. Anything outside it is, by
    # construction, not this install's state.
    if _ADOPTED_HOME is not None:
        try:
            adopted = _ADOPTED_HOME.resolve()
        except OSError:
            adopted = _ADOPTED_HOME
        if not (resolved == adopted or adopted in resolved.parents):
            return (
                f"{label} ({resolved}) is outside the service user's home "
                f"({adopted}); refusing to remove it"
            )
    return None


def _validate_targets(manifest: UninstallManifest) -> str | None:
    """First unsafe deletion target among the config-derived paths.

    `data_dir` and every config-derived extra path get the same fail-closed
    guard: an operator typo pointing e.g. `upload_dir` or
    `[security].secrets_key_file` at `$HOME` must refuse, not rmtree
    (security-review symmetry note).
    """
    err = _unsafe_target("data dir", manifest.data_dir)
    if err is not None:
        return err
    for ep in manifest.extra_paths:
        err = _unsafe_target(ep.label, ep.path)
        if err is not None:
            return err
    return None


def _delete_path(path: Path) -> None:
    """Remove a file or directory best-effort (never raises)."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        console.print(f"[yellow]Could not remove {_plain(path)}: {_plain(exc)}[/yellow]")


def _delete_everything(
    manifest: UninstallManifest, *, keep_data: bool, daemon_stopped: bool
) -> int:
    """The ordered deletion phase. Returns the number of top-level items removed.

    Every item is best-effort: a failure is reported and the rest continue.
    """
    data_dir = manifest.data_dir
    deleted = 0

    if not keep_data and data_dir.exists():
        try:
            shutil.rmtree(data_dir)
            deleted += 1
        except OSError as exc:
            console.print(f"[yellow]Could not remove {_plain(data_dir)}: {_plain(exc)}[/yellow]")

    for ep in manifest.extra_paths:
        if keep_data:
            if ep.kind in ("key_file", "config"):
                continue  # data is useless without the key; keep config to restart
            if _is_under(ep.path, data_dir):
                continue  # inside the preserved data dir
        # A foreign daemon's pid file must not be deleted (we could not stop it).
        if ep.kind == "pid" and not daemon_stopped:
            continue
        if ep.path.exists():
            _delete_path(ep.path)
            deleted += 1

    if not keep_data:
        for cert in manifest.ca_certs:
            if cert.exists():
                _delete_path(cert)
                deleted += 1

    # (P30 D-P30-8) The installed CODE, last: an installer-made layout is never
    # "data", so it goes even under --keep-data — otherwise a `nerdit` stays on
    # PATH pointing at a half-removed install. Deleting the files of the running
    # frozen binary is safe on POSIX (the image is already mapped), but code
    # still goes after the data teardown so nothing above needs it.
    if manifest.layout is not None:
        for path in (manifest.layout.shim, manifest.layout.current, manifest.layout.versions_dir):
            if path_present(path):
                _delete_path(path)
                deleted += 1
        # A now-empty ~/.nerdit/bin would keep ~/.nerdit non-empty and block the
        # rmdir below; never recursive (an operator may keep their own scripts
        # there) and USER MODE ONLY — a system install's shim dir is
        # /usr/local/bin, which is not ours to remove under any circumstance.
        bin_dir = manifest.layout.shim.parent
        if manifest.layout.mode == "user" and bin_dir != manifest.layout.root and bin_dir.is_dir():
            try:
                bin_dir.rmdir()
            except OSError:
                pass

    # The literal ~/.nerdit may survive a custom-data_dir uninstall once its CA
    # copies and config.toml are gone. rmdir ONLY when empty — never rmtree it
    # blindly (it may hold unrelated files, or be the kept data_dir itself).
    home_nerdit = Path.home() / ".nerdit"
    if home_nerdit.is_dir():
        try:
            home_nerdit.rmdir()
        except OSError:
            pass

    if manifest.archive_dir is not None:
        console.print(f"[dim]Left in place (audit archive): {_plain(manifest.archive_dir)}[/dim]")

    return deleted


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #


#: Where a P30 system install records the identity of the daemon it manages.
_SYSTEM_UNIT_PATH = Path("/etc/systemd/system/nerdit.service")

#: The home adopted from the system unit, and the home of the user who actually
#: invoked the command. Both are needed by `_unsafe_target`: after
#: adoption ``Path.home()`` is the ADOPTED one, so a guard written against it
#: silently stops protecting root's home and starts protecting the unit user's.
_ADOPTED_HOME: Path | None = None
_INVOKING_HOME: Path | None = None


def _adopt_unit_home() -> str | None:
    """Use the system unit's recorded home when different from the invoking home.

    This finds an unprivileged daemon's data and credentials during root uninstall
    without widening deletion scope. Return None when no change is needed.
    """
    global _ADOPTED_HOME, _INVOKING_HOME
    # Reset first: these are module state, so a second call in the same process
    # must not inherit the first's adoption. (A leak across calls is invisible
    # in the one-shot CLI but makes the guard below depend on call order.)
    _ADOPTED_HOME = None
    _INVOKING_HOME = None

    if os.name != "posix" or sys.platform == "darwin":
        return None
    try:
        text = _SYSTEM_UNIT_PATH.read_text()
    except OSError:
        return None

    recorded = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Environment=HOME="):
            recorded = stripped.split("=", 2)[2].strip().strip('"')
    if not recorded.startswith("/"):
        return None
    try:
        if Path(recorded) == Path.home():
            return None
    except (OSError, RuntimeError):
        return None

    try:
        _INVOKING_HOME = Path.home().resolve()
    except (OSError, RuntimeError):
        _INVOKING_HOME = None
    _ADOPTED_HOME = Path(recorded)
    os.environ["HOME"] = recorded
    return recorded


def _adopted_config_is_trustworthy(adopted_home: str) -> bool:
    """Allow adopted config to supply deletion targets only when owned by root or the unit user.

    An unrelated owner could redirect privileged deletion through a substituted file.
    """
    config = Path(adopted_home) / ".nerdit" / "config.toml"
    try:
        cfg_uid = config.stat().st_uid
        home_uid = Path(adopted_home).stat().st_uid
    except OSError:
        # No config (or unreadable): nothing to distrust — defaults are used.
        return True
    return cfg_uid in (0, home_uid)


def uninstall(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the typed confirmation prompt."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed and exit without touching anything."
    ),
    purge_images: bool = typer.Option(
        False,
        "--purge-images",
        help="Also remove the base images (Ollama/vLLM/Postgres/Redis/runtime).",
    ),
    keep_data: bool = typer.Option(
        False,
        "--keep-data",
        help="Preserve the data dir and secrets key; still remove containers/images/logs.",
    ),
) -> None:
    """Remove this installation, its containers, images, data and credentials.

    Stop the daemon and acquire its data-dir lock before deleting anything,
    including the service unit. Abort if it remains alive. Other removal steps
    are best-effort and work with a dead daemon or corrupt config.
    """
    # (P30) BEFORE settings are resolved: a system unit records the home whose
    # `.nerdit` actually holds the state. `sudo` resets $HOME to /root, so
    # without this the documented `sudo nerdit uninstall` reports success while
    # the secrets key, every .enc envelope, the internal CA and node.key stay
    # on disk under the unit user's home.
    adopted_home = _adopt_unit_home()
    if adopted_home is not None:
        console.print(
            f"[yellow]This system install runs as the user whose home is "
            f"{_plain(adopted_home)}; resolving its data directory there rather "
            "than under the invoking $HOME.[/yellow]"
        )
        if not _adopted_config_is_trustworthy(adopted_home):
            console.print(
                "[red]Refusing to uninstall — the config.toml under "
                f"{_plain(adopted_home)}/.nerdit is owned by neither root nor "
                "that user, so a third party controls which paths this command "
                "would remove as root. Investigate before retrying.[/red]"
            )
            raise typer.Exit(1)

    # Tolerant settings load — a mangled config.toml is exactly a state this
    # command must survive (mirrors cli/checks.py).
    try:
        settings = load_settings()
    except Exception as exc:  # noqa: BLE001 — any parse/validation error
        console.print(
            f"[yellow]Could not read config.toml ({type(exc).__name__}); "
            "assuming defaults.[/yellow]"
        )
        settings = NerditSettings()

    manifest = build_manifest(
        settings, purge_images=purge_images, docker_client_factory=_default_docker_client
    )

    # Fail closed on a dangerous data_dir BEFORE rendering or deleting anything:
    # an empty/relative/root/$HOME/CWD value would otherwise rmtree the wrong
    # tree (guards --dry-run too — a "." target must never be presented as safe).
    guard = _validate_targets(manifest)
    if guard is not None:
        console.print(
            f"[red]Refusing to uninstall — {escape(guard)}. "
            "Fix the data_dir setting in config.toml and retry.[/red]"
        )
        raise typer.Exit(1)

    # Refuse non-root system uninstall before preview or confirmation, even in dry run.
    # A partial teardown could delete data while leaving a unit respawning the daemon.
    if manifest.layout is not None and manifest.layout.mode == "system" and os.geteuid() != 0:
        console.print(
            f"[red]This is a system install ({_plain(manifest.layout.root)}) and you "
            "are not root — nothing was removed. Re-run with sudo: "
            "sudo nerdit uninstall[/red]"
        )
        raise typer.Exit(1)

    _render_manifest(manifest, keep_data=keep_data)

    if dry_run:
        console.print("\n[dim]Dry run — nothing was removed.[/dim]")
        return

    # An adopted config that relocates the data dir out of its default position
    # is the shape a tampered file takes, so it never rides --yes: the operator
    # must see the path and type the word. Containment in `_unsafe_target` has
    # already refused anything outside the adopted home; this covers the rest.
    relocated = adopted_home is not None and manifest.data_dir != (Path(adopted_home) / ".nerdit")
    if relocated and yes:
        console.print(
            f"[yellow]The service user's config relocates the data directory to "
            f"{_plain(str(manifest.data_dir))}, so --yes does not apply here.[/yellow]"
        )

    if not yes or relocated:
        answer = typer.prompt("Type 'uninstall' to confirm")
        if answer.strip() != "uninstall":
            console.print("[red]Aborted — nothing was removed.[/red]")
            raise typer.Exit(1)

    # Stop the unit first to prevent respawn. Record failures and retain the unit
    # until the data-dir lock proves the daemon is gone.
    if manifest.service_unit is not None:
        _stopped_by_unit, service_result = _stop_unit(
            manifest.service_unit,
            data_dir=manifest.data_dir,
            pid_file=Path(settings.daemon.pid_file).expanduser(),
        )
    else:
        service_result = "no service unit manages this install"

    # 1. Stop the daemon process itself, whether or not step 0 succeeded (recon
    #    fact #1: a bare SIGTERM does not wait — we do our own
    #    TERM → drain → KILL escalation, bounded by ``_DAEMON_DRAIN_S``).
    #    ``_terminate`` is safe on a dead or reused pid: it verifies the command
    #    line first and treats ProcessLookupError as "already gone".
    daemon_stopped = True
    if manifest.daemon_pid is not None:
        console.print(f"[dim]Signalling the daemon (pid {manifest.daemon_pid})…[/dim]")
        daemon_stopped = _terminate(
            manifest.daemon_pid, "nerditd", grace_s=_DAEMON_DRAIN_S, expect_cmd="nerdit"
        )
        signal_result = f"pid {manifest.daemon_pid} signalled (SIGTERM, then SIGKILL); " + (
            "it is gone" if daemon_stopped else "it did NOT exit"
        )
    else:
        signal_result = "no daemon pid file to signal"
    if daemon_stopped:
        Path(settings.daemon.pid_file).expanduser().unlink(missing_ok=True)

    # Hold the exclusive data-dir lock before removing any file or resource.
    # On failure, abort with the install, unit, Caddy, containers and data intact.
    free, lock_fd = _acquire_data_dir_lock(manifest.data_dir)
    if not free:
        console.print(
            "[red]Refusing to uninstall — a daemon is still running on this data "
            "dir, so nothing was removed.[/red]"
        )
        console.print(f"[yellow]  service manager: {_plain(service_result)}[/yellow]")
        console.print(f"[yellow]  daemon process:  {_plain(signal_result)}[/yellow]")
        console.print(
            f"[yellow]  the data dir ({_plain(manifest.data_dir)}) is still held.[/yellow]"
        )
        console.print(
            "[dim]The service unit, the install tree, the containers and the data "
            "are untouched. Stop the daemon (`nerdit exit`) and retry.[/dim]"
        )
        raise typer.Exit(1)

    # ======================= REMOVE ========================================= #
    # 3. The data dir is free. Disable the unit and delete its file — only now
    #    can that not orphan a live daemon.
    if manifest.service_unit is not None:
        _remove_unit(manifest.service_unit)

    # 4. Reap Caddy (survives a killed daemon; pid read pre-deletion).
    if manifest.caddy_pid is not None:
        # Identity needle follows [proxy].caddy_binary so a custom binary name
        # still verifies; falls back to "caddy".
        caddy_needle = Path(settings.proxy.caddy_binary).name or "caddy"
        _terminate(manifest.caddy_pid, "caddy", expect_cmd=caddy_needle)
    (manifest.data_dir / "caddy.pid").unlink(missing_ok=True)

    # 5. Docker teardown, or degrade to printed manual commands. Sweeps exactly
    #    this instance's ``managed-by=nerdit`` containers (``_container_filters``
    #    pins both labels). ``nerdit exit`` has no sweep to reuse — it stops the
    #    daemon and deliberately leaves the containers running — so the label
    #    pair, shared with the daemon's own sweeps, is the reused seam.
    extra_tags = manifest.base_image_tags if purge_images else []
    if manifest.docker_error is not None:
        _print_docker_manual(manifest)
    else:
        try:
            removed_c, removed_i, leftover = _teardown_docker(
                _default_docker_client,
                manifest.instance_id,
                extra_tags=extra_tags,
                owned_repos=manifest.owned_repos,
            )
            console.print(f"[dim]Removed {removed_c} container(s), {removed_i} image(s).[/dim]")
            if manifest.db_error is not None:
                console.print(
                    "[yellow]App images were NOT removed (this instance's DB was "
                    "unreadable, so its images cannot be told apart from another "
                    "instance's). Clean up manually if none share this host:[/yellow]"
                )
                console.print(
                    "[dim]  docker images --format '{{.Repository}}:{{.Tag}}' "
                    "| grep '^nerdit-app/' | xargs -r docker rmi[/dim]"
                )
            if leftover:
                console.print(
                    f"[yellow]{len(leftover)} image(s) could not be removed "
                    f"(still in use?): {_plain(', '.join(leftover))}[/yellow]"
                )
        except Exception as exc:  # noqa: BLE001 — docker died between probe + teardown
            console.print(f"[yellow]Docker teardown failed: {_plain(exc)}[/yellow]")
            _print_docker_manual(manifest)

    # 6. Print the trust-store removal commands BEFORE deleting the .crt files
    #    (the fingerprint is unrecoverable afterwards).
    if not keep_data:
        _print_untrust_hint(manifest)

    # 7. The ordered deletion, under the flock taken in step 2.
    try:
        deleted = _delete_everything(manifest, keep_data=keep_data, daemon_stopped=daemon_stopped)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)

    # 8. Closing.
    console.print("\n[green]Nerdit uninstalled.[/green]")
    console.print(f"[dim]Removed {deleted} filesystem item(s).[/dim]")
    if manifest.layout is None:
        # Only a pip/uv install still has a package left to remove; an
        # installer-made layout WAS the binaries, and they are gone.
        console.print(
            "[dim]Remove the package itself with: pip uninstall nerdit  "
            "(or: uv tool uninstall nerdit)[/dim]"
        )
