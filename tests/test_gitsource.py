"""gitsource tests (P11.5 / Part A) — the clone guards, subprocess fully mocked.

``asyncio.create_subprocess_exec`` is replaced by a fake that records argv/env
and (for success paths) materializes the checkout on disk. No real ``git`` runs,
so the suite is deterministic offline. Covers U1-U8 plus the ``[git]`` settings.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from nerdit.config.settings import GitSettings
from nerdit.core import gitsource
from nerdit.core.gitsource import GitSourceError, clone_source, ls_remote_head

_SHA = "a" * 40


class FakeProc:
    def __init__(self, *, returncode=0, stdout=b"", stderr=b"", hang=False):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        self.killed = False

    async def communicate(self):
        if self._hang:
            await asyncio.sleep(3600)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


def _install(monkeypatch, handler):
    """Patch git_available True + a recording fake create_subprocess_exec."""
    monkeypatch.setattr(gitsource, "git_available", lambda: True)
    calls: list[dict] = []

    async def fake_exec(*args, env=None, stdout=None, stderr=None):
        proc = handler(args, env)
        calls.append({"args": list(args), "env": env, "proc": proc})
        return proc

    monkeypatch.setattr(gitsource.asyncio, "create_subprocess_exec", fake_exec)
    return calls


def _success_handler(*, branch=b"main\n"):
    """A handler that materializes a clone and answers rev-parse."""

    def handler(args, env):
        if "clone" in args:
            dest = Path(args[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "app.py").write_text("print('hi')\n")
            git_dir = dest / ".git"
            git_dir.mkdir()
            (git_dir / "config").write_text("x")
            return FakeProc(returncode=0)
        if "rev-parse" in args and "--abbrev-ref" in args:
            return FakeProc(returncode=0, stdout=branch)
        if "rev-parse" in args:
            return FakeProc(returncode=0, stdout=_SHA.encode() + b"\n")
        return FakeProc(returncode=0)

    return handler


# --- U1: https-only rejection matrix -----------------------------------------


@pytest.mark.parametrize(
    "bad_url",
    [
        "ssh://git@github.com/o/r.git",
        "git@github.com:o/r.git",
        "git://github.com/o/r.git",
        "file:///etc/passwd",
        "/local/path/repo",
        "http://github.com/o/r.git",
    ],
)
async def test_u1_non_https_rejected_no_subprocess(monkeypatch, tmp_path, bad_url):
    calls = _install(monkeypatch, _success_handler())
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            bad_url,
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_url_invalid"
    assert calls == []  # never spawned a subprocess


# --- U2: host allowlist ------------------------------------------------------


async def test_u2_host_not_in_allowlist_forbidden(monkeypatch, tmp_path):
    calls = _install(monkeypatch, _success_handler())
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://gitlab.com/o/r.git",
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_host_forbidden"
    assert calls == []


async def test_u2_custom_allowed_host_admitted(monkeypatch, tmp_path):
    _install(monkeypatch, _success_handler())
    info = await clone_source(
        "https://gitlab.com/o/r.git",
        dest_dir=tmp_path / "d",
        timeout_s=10,
        max_bytes=10**9,
        allowed_hosts=["gitlab.com"],
    )
    assert info.commit_sha == _SHA


# --- U3: timeout -------------------------------------------------------------


async def test_u3_timeout_kills_and_cleans(monkeypatch, tmp_path):
    def handler(args, env):
        return FakeProc(hang=True)

    calls = _install(monkeypatch, handler)
    dest = tmp_path / "d"
    dest.mkdir()  # simulate a partial clone tree already on disk
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            dest_dir=dest,
            timeout_s=0.05,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_timeout"
    assert calls[0]["proc"].killed is True
    assert not dest.exists()  # partial tree rmtree'd


# --- U4: size cap ------------------------------------------------------------


async def test_u4_too_large_rejected_and_cleaned(monkeypatch, tmp_path):
    def handler(args, env):
        if "clone" in args:
            dest = Path(args[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "big.bin").write_bytes(b"x" * 4096)
            (dest / ".git").mkdir()
            return FakeProc(returncode=0)
        return FakeProc(returncode=0, stdout=_SHA.encode() + b"\n")

    _install(monkeypatch, handler)
    dest = tmp_path / "d"
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            dest_dir=dest,
            timeout_s=10,
            max_bytes=1024,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_too_large"
    assert not dest.exists()


# --- U5: subdir handling -----------------------------------------------------


@pytest.mark.parametrize("bad_subdir", ["../x", "/abs", "a/../../b"])
async def test_u5_subdir_traversal_rejected_no_subprocess(monkeypatch, tmp_path, bad_subdir):
    calls = _install(monkeypatch, _success_handler())
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            subdir=bad_subdir,
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_subdir_invalid"
    assert calls == []  # syntax guard runs before any clone


async def test_u5_missing_subdir_dir_rejected(monkeypatch, tmp_path):
    _install(monkeypatch, _success_handler())  # clone makes no 'svc' dir
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            subdir="svc",
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_subdir_invalid"


async def test_u5_valid_subdir_returns_nested_context(monkeypatch, tmp_path):
    def handler(args, env):
        if "clone" in args:
            dest = Path(args[-1])
            (dest / "svc").mkdir(parents=True, exist_ok=True)
            (dest / "svc" / "main.py").write_text("x")
            (dest / ".git").mkdir()
            return FakeProc(returncode=0)
        return FakeProc(returncode=0, stdout=_SHA.encode() + b"\n")

    _install(monkeypatch, handler)
    dest = tmp_path / "d"
    info = await clone_source(
        "https://github.com/o/r.git",
        subdir="svc",
        dest_dir=dest,
        timeout_s=10,
        max_bytes=10**9,
        allowed_hosts=["github.com"],
    )
    assert info.context_dir == (dest / "svc").resolve()


# --- U6: success path --------------------------------------------------------


async def test_u6_success_strips_git_and_pins_commit(monkeypatch, tmp_path):
    _install(monkeypatch, _success_handler())
    dest = tmp_path / "d"
    info = await clone_source(
        "https://github.com/o/r.git",
        dest_dir=dest,
        timeout_s=10,
        max_bytes=10**9,
        allowed_hosts=["github.com"],
    )
    assert info.commit_sha == _SHA
    assert info.resolved_ref == "main"  # abbrev-ref fallback when ref is None
    assert info.context_dir == dest
    assert not (dest / ".git").exists()  # history stripped
    assert (dest / "app.py").exists()


async def test_u6_explicit_ref_is_honored(monkeypatch, tmp_path):
    _install(monkeypatch, _success_handler())
    info = await clone_source(
        "https://github.com/o/r.git",
        ref="v1.0.0",
        dest_dir=tmp_path / "d",
        timeout_s=10,
        max_bytes=10**9,
        allowed_hosts=["github.com"],
    )
    assert info.resolved_ref == "v1.0.0"


# --- U9: global git config isolation (F2) ------------------------------------


async def test_u9_global_git_config_isolated(monkeypatch, tmp_path):
    # A planted real HOME must never leak: a url.<base>.insteadOf there could
    # rewrite the validated https URL to file://... before the clone runs.
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = _install(monkeypatch, _success_handler())
    await clone_source(
        "https://github.com/o/r.git",
        dest_dir=tmp_path / "d",
        timeout_s=10,
        max_bytes=10**9,
        allowed_hosts=["github.com"],
    )
    assert calls  # at least one git invocation recorded
    for call in calls:
        env = call["env"]
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["HOME"] == "/dev/null"  # the real HOME never leaks
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert "XDG_CONFIG_HOME" not in env


# --- U7: credential hygiene --------------------------------------------------


async def test_u7a_stderr_scrubbed_in_error(monkeypatch, tmp_path):
    token = "tok_secret_ABC123"
    leaky = f"fatal: could not read from https://x-access-token:{token}@github.com/o/r.git"

    def handler(args, env):
        if "clone" in args:
            return FakeProc(returncode=128, stderr=leaky.encode())
        return FakeProc(returncode=0)

    _install(monkeypatch, handler)
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            token=token,
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    msg = exc.value.message
    assert exc.value.code == "deploy.git_clone_failed"
    assert token not in msg
    assert "x-access-token" not in msg
    assert "***" in msg


async def test_clone_auth_challenge_gives_not_found_hint(monkeypatch, tmp_path):
    """A missing/private repo over https surfaces git's 'could not read Username'
    (prompts disabled); the hint must point at not-found/private, not a raw ref
    error. This is the store-template-not-published case the user hit."""
    stderr = b"fatal: could not read Username for 'https://github.com': terminal prompts disabled"

    def handler(args, env):
        if "clone" in args:
            return FakeProc(returncode=128, stderr=stderr)
        return FakeProc(returncode=0)

    _install(monkeypatch, handler)
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/example-org/does-not-exist",
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_clone_failed"
    assert "not found or private" in (exc.value.hint or "")
    assert "token_ref" in (exc.value.hint or "")


def test_clone_failed_hint_generic_vs_auth():
    """The hint helper distinguishes an auth challenge from an ordinary bad ref."""
    auth = gitsource._clone_failed_hint("fatal: Authentication failed", had_token=False)
    assert "not found or private" in auth
    token = gitsource._clone_failed_hint("fatal: Authentication failed", had_token=True)
    assert "token_ref" in token and "expired" in token
    generic = gitsource._clone_failed_hint("fatal: Remote branch nope not found", had_token=False)
    assert "branch or tag" in generic


async def test_u7b_token_never_in_argv_only_in_child_env(monkeypatch, tmp_path):
    token = "tok_secret_DEF456"
    captured: dict = {}

    def handler(args, env):
        if "clone" in args and env and "GIT_ASKPASS" in env:
            helper = Path(env["GIT_ASKPASS"])
            captured["helper"] = helper
            captured["helper_mode"] = helper.stat().st_mode & 0o777
            captured["helper_content"] = helper.read_text()
            captured["ntoken"] = env.get("NERDIT_GIT_TOKEN")
        return _success_handler()(args, env)

    calls = _install(monkeypatch, handler)
    await clone_source(
        "https://github.com/o/r.git",
        token=token,
        dest_dir=tmp_path / "d",
        timeout_s=10,
        max_bytes=10**9,
        allowed_hosts=["github.com"],
    )
    # token nowhere in any argv
    for call in calls:
        assert all(token not in str(arg) for arg in call["args"])
    # token only reaches the child through the env
    assert captured["ntoken"] == token
    assert captured["helper_mode"] == 0o700
    assert token not in captured["helper_content"]
    # the helper is unlinked once the clone finishes (no leftover in upload_dir)
    assert not captured["helper"].exists()


async def test_u7c_userinfo_url_rejected_before_any_work(monkeypatch, tmp_path):
    calls = _install(monkeypatch, _success_handler())
    dest = tmp_path / "d"
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://user:tok@github.com/o/r.git",
            dest_dir=dest,
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_url_credentials"
    assert calls == []
    assert not dest.exists()


# --- U8: argv injection ------------------------------------------------------


@pytest.mark.parametrize("bad_ref", ["--upload-pack=/tmp/x", "-b", "a/../b"])
async def test_u8_bad_ref_rejected(monkeypatch, tmp_path, bad_ref):
    calls = _install(monkeypatch, _success_handler())
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            ref=bad_ref,
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_ref_invalid"
    assert calls == []


async def test_u8_repo_url_leading_dash_rejected(monkeypatch, tmp_path):
    calls = _install(monkeypatch, _success_handler())
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "-oProxyCommand=evil",
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.code == "deploy.git_url_invalid"
    assert calls == []


# --- git_available -----------------------------------------------------------


async def test_git_unavailable_raises_503(monkeypatch, tmp_path):
    monkeypatch.setattr(gitsource.shutil, "which", lambda _name: None)
    with pytest.raises(GitSourceError) as exc:
        await clone_source(
            "https://github.com/o/r.git",
            dest_dir=tmp_path / "d",
            timeout_s=10,
            max_bytes=10**9,
            allowed_hosts=["github.com"],
        )
    assert exc.value.status_code == 503
    assert exc.value.code == "deploy.git_unavailable"


def test_git_available_reflects_which(monkeypatch):
    monkeypatch.setattr(gitsource.shutil, "which", lambda _name: "/usr/bin/git")
    assert gitsource.git_available() is True
    monkeypatch.setattr(gitsource.shutil, "which", lambda _name: None)
    assert gitsource.git_available() is False


# --- GitSettings -------------------------------------------------------------


def test_git_settings_defaults():
    s = GitSettings()
    assert s.enabled is True
    assert s.allowed_hosts == ["github.com"]
    assert s.clone_timeout_s == 120
    assert s.max_clone_bytes == 500 * 1024 * 1024


def test_git_settings_allowed_hosts_normalized():
    s = GitSettings(allowed_hosts=["  GitHub.com ", "Gitlab.COM"])
    assert s.allowed_hosts == ["github.com", "gitlab.com"]


def test_git_settings_rejects_empty_allowed_host():
    with pytest.raises(ValidationError):
        GitSettings(allowed_hosts=["github.com", "  "])


def test_git_settings_github_clone_base_url(monkeypatch):
    monkeypatch.delenv("NERDIT_DEV_GITHUB_CLONE_BASE", raising=False)
    s = GitSettings()
    assert s.github_clone_base_url == "https://github.com"
    assert s.github_host == "github.com"
    # Host compare is case-insensitive.
    assert GitSettings(github_clone_base_url="https://GitHub.COM").github_host == "github.com"
    # A non-github host LOADS with grammar-only validation — no env guard at
    # load (security review F2 / D9: the host gate moved to token resolution).
    s = GitSettings(github_clone_base_url="https://git.localhost/")
    assert s.github_clone_base_url == "https://git.localhost"
    assert s.github_host == "git.localhost"
    for bad in (
        "http://git.localhost",  # https only — clone URLs are https-only too
        "https://",
        "https://user:tok@github.com",
        "https://github.com/path",
        "https://github.com?x=1",
    ):
        # Grammar failures are all that remains at load — the host is not pinned.
        with pytest.raises(ValidationError):
            GitSettings(github_clone_base_url=bad)


def test_git_settings_github_clone_base_url_host_is_dev_gated(monkeypatch):
    """Security review F2 / D9: the github.com-only trust decision moved from
    settings LOAD to token RESOLUTION — settings load in every process (the CLI
    included), so a load-time refusal would hard-fail `nerdit link`/`doctor` on
    a fixture config they never act on. (a) A non-github host loads (grammar
    only). (b) `resolve_github_installation_token` refuses the cloud token for a
    non-github host without the env guard, and offers it with the guard set."""
    from types import SimpleNamespace

    from nerdit.daemon.deploy_pipeline import resolve_github_installation_token

    monkeypatch.delenv("NERDIT_DEV_GITHUB_CLONE_BASE", raising=False)

    # (a) LOAD succeeds for a fixture host with no flag — grammar only.
    s = GitSettings(github_clone_base_url="https://git.localhost")
    assert s.github_host == "git.localhost"
    GitSettings(github_clone_base_url="https://evil.example.com")  # also loads
    # The offerability predicate is the moved gate.
    assert gitsource.installation_token_allowed_for_host("github.com") is True
    assert gitsource.installation_token_allowed_for_host("git.localhost") is False

    repo = "https://git.localhost/owner/repo"
    app = SimpleNamespace(
        state=SimpleNamespace(
            link_manager=SimpleNamespace(github_token_for_repo=lambda slug: "ghs_tok"),
            settings=SimpleNamespace(git=s),
        )
    )
    # (b) resolve refuses without the flag ...
    assert resolve_github_installation_token(app, repo) is None
    # ... the empty string is falsy ...
    monkeypatch.setenv("NERDIT_DEV_GITHUB_CLONE_BASE", "")
    assert resolve_github_installation_token(app, repo) is None
    assert gitsource.installation_token_allowed_for_host("git.localhost") is False
    # ... and resolves once any non-empty value opens the hatch.
    monkeypatch.setenv("NERDIT_DEV_GITHUB_CLONE_BASE", "1")
    assert gitsource.installation_token_allowed_for_host("git.localhost") is True
    assert resolve_github_installation_token(app, repo) == "ghs_tok"

    # github.com is always offerable, no flag needed.
    monkeypatch.delenv("NERDIT_DEV_GITHUB_CLONE_BASE", raising=False)
    app_gh = SimpleNamespace(
        state=SimpleNamespace(
            link_manager=SimpleNamespace(github_token_for_repo=lambda slug: "ghs_tok"),
            settings=SimpleNamespace(git=GitSettings()),
        )
    )
    assert resolve_github_installation_token(app_gh, "https://github.com/owner/repo") == "ghs_tok"


def test_git_settings_bounds():
    with pytest.raises(ValidationError):
        GitSettings(clone_timeout_s=0)
    with pytest.raises(ValidationError):
        GitSettings(max_clone_bytes=0)


def test_git_settings_watch_interval():
    """[git].watch_interval_s (P24c WP10): default 60, floor 15."""
    assert GitSettings().watch_interval_s == 60
    assert GitSettings(watch_interval_s=15).watch_interval_s == 15
    with pytest.raises(ValidationError):
        GitSettings(watch_interval_s=14)


# --- ls_remote_head (P24c WP10) ----------------------------------------------
#
# The poller is an unattended, repeating egress, so its guard set is not
# "similar to" the clone's — it is the same one. The parity matrix below runs
# every guard against BOTH operations from one parametrize, so a guard that is
# ever added to one and not the other fails here.

_LS_STDOUT = f"{_SHA}\trefs/heads/main\n".encode()


def _parity_handler(args, env):
    """A handler that answers an ls-remote and materializes a clone."""
    if "ls-remote" in args:
        return FakeProc(returncode=0, stdout=_LS_STDOUT)
    return _success_handler()(args, env)


def _op(op, repo_url, *, tmp_path, ref=None, token=None, timeout_s=10):
    """Invoke ``clone_source``/``ls_remote_head`` with equivalent arguments."""
    hosts = ["github.com"]
    if op == "clone":
        return clone_source(
            repo_url,
            ref=ref,
            dest_dir=tmp_path / "d",
            token=token,
            timeout_s=timeout_s,
            max_bytes=10**9,
            allowed_hosts=hosts,
        )
    return ls_remote_head(repo_url, ref=ref, token=token, timeout_s=timeout_s, allowed_hosts=hosts)


_OPS = ["clone", "ls_remote"]


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize(
    ("bad_url", "code"),
    [
        ("http://github.com/o/r.git", "deploy.git_url_invalid"),
        ("ssh://git@github.com/o/r.git", "deploy.git_url_invalid"),
        ("-oProxyCommand=evil", "deploy.git_url_invalid"),
        ("https://gitlab.com/o/r.git", "deploy.git_host_forbidden"),
        ("https://user:tok@github.com/o/r.git", "deploy.git_url_credentials"),
    ],
)
async def test_parity_url_guards_reject_before_any_subprocess(
    monkeypatch, tmp_path, op, bad_url, code
):
    calls = _install(monkeypatch, _parity_handler)
    with pytest.raises(GitSourceError) as exc:
        await _op(op, bad_url, tmp_path=tmp_path)
    assert exc.value.code == code
    assert calls == []


@pytest.mark.parametrize("op", _OPS)
@pytest.mark.parametrize("bad_ref", ["--upload-pack=/tmp/x", "-b", "a/../b"])
async def test_parity_bad_ref_rejected_before_any_subprocess(monkeypatch, tmp_path, op, bad_ref):
    calls = _install(monkeypatch, _parity_handler)
    with pytest.raises(GitSourceError) as exc:
        await _op(op, "https://github.com/o/r.git", tmp_path=tmp_path, ref=bad_ref)
    assert exc.value.code == "deploy.git_ref_invalid"
    assert calls == []


@pytest.mark.parametrize("op", _OPS)
async def test_parity_git_unavailable_is_503(monkeypatch, tmp_path, op):
    monkeypatch.setattr(gitsource.shutil, "which", lambda _name: None)
    with pytest.raises(GitSourceError) as exc:
        await _op(op, "https://github.com/o/r.git", tmp_path=tmp_path)
    assert (exc.value.status_code, exc.value.code) == (503, "deploy.git_unavailable")


@pytest.mark.parametrize("op", _OPS)
async def test_parity_timeout_kills_the_child(monkeypatch, tmp_path, op):
    calls = _install(monkeypatch, lambda args, env: FakeProc(hang=True))
    with pytest.raises(GitSourceError) as exc:
        await _op(op, "https://github.com/o/r.git", tmp_path=tmp_path, timeout_s=0.05)
    assert (exc.value.status_code, exc.value.code) == (504, "deploy.git_timeout")
    assert calls[0]["proc"].killed is True


@pytest.mark.parametrize("op", _OPS)
async def test_parity_token_never_in_argv(monkeypatch, tmp_path, op):
    token = "tok_secret_GHI789"  # noqa: S105 — a fake, not a credential
    calls = _install(monkeypatch, _parity_handler)
    await _op(op, "https://github.com/o/r.git", tmp_path=tmp_path, token=token)
    assert calls
    for call in calls:
        assert all(token not in str(arg) for arg in call["args"])
        assert call["env"]["NERDIT_GIT_TOKEN"] == token
        assert call["env"]["GIT_ASKPASS"]


@pytest.mark.parametrize("op", _OPS)
async def test_parity_env_isolation(monkeypatch, tmp_path, op):
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = _install(monkeypatch, _parity_handler)
    await _op(op, "https://github.com/o/r.git", tmp_path=tmp_path)
    assert calls
    for call in calls:
        env = call["env"]
        assert env["HOME"] == "/dev/null"
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize("op", _OPS)
async def test_parity_stderr_is_scrubbed(monkeypatch, tmp_path, op):
    token = "tok_secret_JKL012"  # noqa: S105 — a fake, not a credential
    leaky = f"fatal: could not read from https://x-access-token:{token}@github.com/o/r.git"
    _install(monkeypatch, lambda args, env: FakeProc(returncode=128, stderr=leaky.encode()))
    with pytest.raises(GitSourceError) as exc:
        await _op(op, "https://github.com/o/r.git", tmp_path=tmp_path, token=token)
    assert exc.value.code == "deploy.git_clone_failed"
    assert token not in exc.value.message
    assert "x-access-token" not in exc.value.message
    assert "***" in exc.value.message


# --- ls_remote_head only ------------------------------------------------------


async def test_ls_remote_argv_is_exact(monkeypatch, tmp_path):
    calls = _install(monkeypatch, _parity_handler)
    sha = await ls_remote_head(
        "https://github.com/o/r.git",
        ref="v1.0.0",
        token=None,
        timeout_s=10,
        allowed_hosts=["github.com"],
    )
    assert sha == _SHA
    assert calls[0]["args"] == [
        "git",
        "-c",
        "credential.helper=",
        "ls-remote",
        "--exit-code",
        "--",
        "https://github.com/o/r.git",
        "v1.0.0",
        "v1.0.0^{}",
    ]


async def test_ls_remote_none_ref_asks_for_head(monkeypatch, tmp_path):
    calls = _install(monkeypatch, _parity_handler)
    await ls_remote_head(
        "https://github.com/o/r.git",
        ref=None,
        token=None,
        timeout_s=10,
        allowed_hosts=["github.com"],
    )
    assert calls[0]["args"][-2:] == ["HEAD", "HEAD^{}"]


@pytest.mark.parametrize(
    ("ref", "stdout", "expected"),
    [
        # Annotated tag: the exact-pattern line carries the TAG OBJECT id, the
        # ``^{}`` line the commit — and the commit is what config['source']
        # records, so it is what must come back.
        ("v1.0.1", f"{'c' * 40}\trefs/tags/v1.0.1\n{_SHA}\trefs/tags/v1.0.1^{{}}\n", _SHA),
        # A branch and an annotated tag sharing a name (real `git ls-remote`
        # output, verified against git 2.50). ``clone --branch dup`` checks out
        # the BRANCH, so preferring the peeled tag here would hand the poller a
        # sha the clone never produces — permanent phantom drift, a redeploy
        # every tick.
        (
            "dup",
            f"{_SHA}\trefs/heads/dup\n{'c' * 40}\trefs/tags/dup\n{'d' * 40}\trefs/tags/dup^{{}}\n",
            _SHA,
        ),
        # Lightweight tag / branch: one line, unchanged behaviour.
        ("v1.0.1", f"{_SHA}\trefs/tags/v1.0.1\n", _SHA),
        ("main", f"{_SHA}\trefs/heads/main\n", _SHA),
        # ``ref=None`` ⇒ the default branch, listed under the literal ``HEAD``.
        (None, f"{_SHA}\tHEAD\n", _SHA),
        # A fully-qualified recorded ref still resolves, peeled first.
        (
            "refs/tags/v1.0.1",
            f"{'c' * 40}\trefs/tags/v1.0.1\n{_SHA}\trefs/tags/v1.0.1^{{}}\n",
            _SHA,
        ),
        # Last resort: an unanticipated refname shape still has to clear the
        # 40-hex gate, but is not refused outright.
        ("main", f"{_SHA}\trefs/remotes/origin/main\n", _SHA),
    ],
)
async def test_ls_remote_selects_the_ref_clone_would_check_out(
    monkeypatch, tmp_path, ref, stdout, expected
):
    """Selection mirrors ``clone --branch``: branch over same-named tag, and an
    annotated tag resolved to its peeled commit rather than the tag object."""
    _install(monkeypatch, lambda args, env: FakeProc(returncode=0, stdout=stdout.encode()))
    sha = await ls_remote_head(
        "https://github.com/o/r.git",
        ref=ref,
        token=None,
        timeout_s=10,
        allowed_hosts=["github.com"],
    )
    assert sha == expected


@pytest.mark.parametrize(
    "stdout",
    [
        b"",
        b"\n",
        b"not-a-sha\trefs/heads/main\n",
        ("a" * 39 + "\trefs/heads/main\n").encode(),
        ("A" * 40 + "\trefs/heads/main\n").encode(),
        b"ref: refs/heads/main\tHEAD\n",
    ],
)
async def test_ls_remote_untrusted_output_is_refused(monkeypatch, tmp_path, stdout):
    """Remote-controlled output never becomes a commit_sha on trust alone."""
    _install(monkeypatch, lambda args, env: FakeProc(returncode=0, stdout=stdout))
    with pytest.raises(GitSourceError) as exc:
        await ls_remote_head(
            "https://github.com/o/r.git",
            ref="main",
            token=None,
            timeout_s=10,
            allowed_hosts=["github.com"],
        )
    assert (exc.value.status_code, exc.value.code) == (400, "deploy.git_clone_failed")


@pytest.mark.parametrize("failing", [False, True])
async def test_ls_remote_askpass_tempdir_is_removed(monkeypatch, tmp_path, failing):
    """The named deviation from clone_source: the helper lives in a private
    TemporaryDirectory, removed on the way out of BOTH exits."""

    def handler(args, env):
        if failing:
            return FakeProc(returncode=128, stderr=b"boom")
        return FakeProc(returncode=0, stdout=_LS_STDOUT)

    calls = _install(monkeypatch, handler)
    coro = ls_remote_head(
        "https://github.com/o/r.git",
        ref="main",
        token="tok_secret_MNO345",  # noqa: S106 — a fake, not a credential
        timeout_s=10,
        allowed_hosts=["github.com"],
    )
    if failing:
        with pytest.raises(GitSourceError):
            await coro
    else:
        await coro
    helper = Path(calls[0]["env"]["GIT_ASKPASS"])
    assert not helper.exists()
    assert not helper.parent.exists()


# --- P33 WP-R: ${github.installation} resolution key (D-GH-3) ------------------


def test_github_installation_ref_is_the_exact_literal():
    assert gitsource.GITHUB_INSTALLATION_REF == "${github.installation}"


@pytest.mark.parametrize(
    ("repo_url", "slug"),
    [
        ("https://github.com/Acme/App.git", "acme/app"),
        ("https://GitHub.com/acme/app/", "acme/app"),
        ("https://github.com/acme/app", "acme/app"),
        # Not GitHub: an installation token is never offered to another host.
        ("https://gitlab.com/acme/app", None),
        ("https://ghe.example.com/acme/app", None),
        # Not owner/name shaped.
        ("https://github.com/acme", None),
        ("https://github.com/acme/app/extra", None),
        ("https://github.com/", None),
        # canonical_repo's own guards narrow to None rather than raise.
        ("https://user:tok@github.com/acme/app", None),
        ("not a url", None),
    ],
)
def test_github_repo_slug(repo_url, slug):
    assert gitsource.github_repo_slug(repo_url) == slug


def test_github_repo_slug_configurable_host():
    """D-GH-3 amended: the pin's host follows [git].github_clone_base_url.

    Host-exclusivity is unchanged — with the fixture host configured, the
    real github.com no longer resolves, and vice versa.
    """
    fixture = "https://git.localhost/nerdit-dev/private-demo.git"
    assert gitsource.github_repo_slug(fixture, "git.localhost") == "nerdit-dev/private-demo"
    assert gitsource.github_repo_slug("https://github.com/acme/app", "git.localhost") is None
    assert gitsource.github_repo_slug("https://git.localhost/acme/app", "github.com") is None
    # Case-insensitive on the configured host, like the default.
    slug = gitsource.github_repo_slug("https://Git.LocalHost/Acme/App", "GIT.LOCALHOST")
    assert slug == "acme/app"
