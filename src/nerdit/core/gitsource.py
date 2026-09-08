"""Acquire deploy contexts from Git and probe remote refs safely.

Both clone and `ls_remote_head` enforce HTTPS, a deny-by-default host allowlist,
no URL userinfo, and guarded ref/subdir syntax. They ignore global/system Git
config so URL rewrite rules cannot bypass validation. Credentials use
`GIT_ASKPASS`, never argv or persisted source URLs.

Clones have a wall-clock timeout and post-clone size cap. After pinning the
commit, remove `.git` so image builds cannot include history or old secrets.
Delete partial trees on every failure. Pure validators let routes apply these
same guards before writing audit params.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

# git ref grammar: start alphanumeric, then the safe subset of git's own ref
# rules. Deliberately stricter than `git check-ref-format` — it rejects any
# leading `-` (argv-injection: `--upload-pack=…`) and any `..` component.
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
# A full object name as git prints it in an ls-remote listing. Remote output is
# untrusted text: nothing is carried forward until it matches this exactly.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# Scrub an embedded `scheme://userinfo@` from any git-echoed text.
_URL_CRED_RE = re.compile(r"(https?://)[^/@\s]+@")
# Keep bubbled stderr short; it is defense-in-depth, not the primary barrier.
_STDERR_TAIL = 500

# Static GIT_ASKPASS helper: git invokes it with the prompt as $1. The token is
# read from the child environment (NERDIT_GIT_TOKEN) — never written to disk, so
# the script content is constant and carries no secret.
_ASKPASS_SCRIPT = """#!/bin/sh
case "$1" in
  Username*) printf '%s\\n' "x-access-token" ;;
  *) printf '%s\\n' "$NERDIT_GIT_TOKEN" ;;
esac
"""


@dataclass
class GitSourceInfo:
    """The result of a successful clone."""

    commit_sha: str  # full 40-hex from `git rev-parse HEAD`
    resolved_ref: str  # requested ref, else the default branch (abbrev-ref HEAD)
    context_dir: Path  # dest_dir, or dest_dir/<subdir> when a subdir was given


class GitSourceError(Exception):
    """A clone failure mapped 1:1 to a structured error envelope by the route."""

    def __init__(self, status_code: int, code: str, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.hint = hint


def git_available() -> bool:
    """True when a `git` binary is resolvable on `PATH`."""
    return shutil.which("git") is not None


def _scrub(text: str, token: str | None) -> str:
    """Redact any credential a git subprocess may echo before it reaches a client.

    Replaces the resolved token literal and any `scheme://userinfo@` prefix
    with `***`. Applied to every stderr fragment that can reach an exception
    message (defense-in-depth — the token is never in argv in the first place).
    """
    if token:
        text = text.replace(token, "***")
    return _URL_CRED_RE.sub(r"\1***@", text)


# git prints these when a clone hits an authentication challenge with prompts
# disabled — which is what a MISSING or PRIVATE repo looks like over https (the
# host issues a 401 rather than leaking existence). The raw "could not read
# Username" is baffling for a public-repo-not-found, so we detect it and hand
# back a targeted hint.
_AUTH_SIGNATURES = (
    "could not read username",
    "could not read password",
    "authentication failed",
    "terminal prompts disabled",
    "repository not found",
    "invalid username or password",
)


def _clone_failed_hint(stderr: str, *, had_token: bool) -> str:
    """Pick a hint for a failed clone: auth-challenge → not-found/private guidance."""
    if any(sig in stderr.lower() for sig in _AUTH_SIGNATURES):
        if had_token:
            return (
                "authentication failed — check the token_ref value has repo read "
                "access and has not expired."
            )
        return (
            "repository not found or private — verify the URL is correct and the "
            "repo is public, or pass token_ref (a ${secrets.shared.KEY} reference) "
            "for a private repository."
        )
    return "check the URL/ref; ref must be a branch or tag."


#: The one `token_ref` literal that resolves to a cloud-pushed GitHub App
#: installation token instead of a stored secret (P33 D-GH-3). Matched exactly:
#: there is no `${github.<anything else>}` grammar, and `${secrets.*}` refs
#: are untouched so a daemon with no cloud at all behaves byte-identically.
GITHUB_INSTALLATION_REF = "${github.installation}"
#: The only host an installation token is ever presented to (the default).
#: The mirror is keyed by `owner/name` (GitHub's own repo identity, no
#: host), so a clone URL on any other allow-listed host never resolves
#: through it — a GitHub credential must not be offered to a non-GitHub
#: remote. The pin's HOST follows `[git].github_clone_base_url` (default
#: `https://github.com`) so the cloud dev stack's git fixture can stand in
#: for GitHub in `make e2e-real`; the invariant is host-exclusivity, not
#: the literal name.
_GITHUB_HOST = "github.com"

#: The env guard that opens the installation-token gate to a non-github host
#: (security review F2). Any non-empty value opens it — the `make e2e-real`
#: gate daemon sets it to `1` so `git.localhost` can play GitHub.
_DEV_GITHUB_CLONE_BASE_ENV = "NERDIT_DEV_GITHUB_CLONE_BASE"


def installation_token_allowed_for_host(github_host: str) -> bool:
    """Whether the cloud-minted GitHub installation token may be offered to
    `github_host` (P33 D-GH-3; security review F2 / D9).

    The token is presented to `github.com` only. A non-github host is the
    dev/e2e fixture case (`make e2e-real`'s `git.localhost`) and is allowed
    ONLY when the `NERDIT_DEV_GITHUB_CLONE_BASE` env guard is set non-empty.
    This is the SAME trust decision the settings validator used to make at load
    time; it moved here — the security boundary where the credential is actually
    handed to a host — so a CLI process that loads a fixture config but never
    resolves a token no longer hard-fails at load (D9). Fails closed: a
    non-github host without the guard offers nothing.
    """
    if github_host.strip().lower() == _GITHUB_HOST:
        return True
    return bool(os.environ.get(_DEV_GITHUB_CLONE_BASE_ENV))


def github_repo_slug(repo_url: str, github_host: str = _GITHUB_HOST) -> str | None:
    """`owner/name` (lower-case) of a clone URL on `github_host`, else `None`.

    The key `LinkManager.github_token_for_repo` resolves by (D-GH-3):
    the host-less tail of `canonical_repo`, accepted only when the host
    is the configured GitHub host and the path is exactly two segments. A URL
    that fails `canonical_repo`'s own guards (no host, embedded
    userinfo) is `None` too — the caller has already run
    `validate_repo_url`, so this is a narrowing, never a second error
    surface.
    """
    try:
        canonical = canonical_repo(repo_url)
    except ValueError:
        return None
    host, _, path = canonical.partition("/")
    if host != github_host.strip().lower():
        return None
    parts = path.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return path


def canonical_repo(repo_url: str) -> str:
    """Canonical `host/owner/repo` identity of a clone URL (P33 D-GH-6).

    Lower-cased, a trailing `.git` stripped, no trailing slash, and no port,
    query, or fragment — so `https://GitHub.com/Acme/App.git/` and
    `https://github.com/acme/app` name the same repository. Derived once at
    deploy time and recorded beside `ref` and the resolved sha; a cloud
    nudge matches on it. Pure, and stricter than a plain parse: a URL with no
    host or with embedded userinfo raises `ValueError` (callers run
    `validate_repo_url` first, which maps the same faults to the
    structured 422s — this guard only keeps a credential out of an identity
    string if a caller ever forgets).
    """
    parts = urlsplit(repo_url)
    if not parts.hostname:
        raise ValueError("repo_url has no host")
    if parts.username or parts.password:
        raise ValueError("repo_url must not embed credentials")
    path = parts.path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return f"{parts.hostname}/{path}".lower().rstrip("/")


def git_source_meta(
    info: GitSourceInfo,
    repo_url: str,
    *,
    subdir: str | None = None,
    token_ref: str | None = None,
    template_id: str | None = None,
) -> dict:
    """Build the `config['source']` provenance blob for a git deploy.

    One writer for the three git ingresses (`POST /deploy/git`, the
    GitWatch/nudge redeploy, and app templates) so the recorded shape cannot
    drift: `type`/`repo_url`/`repo` (D-GH-6 canonical identity)/`ref`/
    `commit_sha` always; `subdir`, `token_ref` (the `${…}` reference
    NAME, never a token — D-P24-14) and `template_id` only when given.
    """
    meta: dict = {
        "type": "git",
        "repo_url": repo_url,
        "repo": canonical_repo(repo_url),
        "ref": info.resolved_ref,
        "commit_sha": info.commit_sha,
    }
    if subdir:
        meta["subdir"] = subdir
    if token_ref is not None:
        meta["token_ref"] = token_ref
    if template_id is not None:
        meta["template_id"] = template_id
    return meta


def validate_repo_url(repo_url: str, allowed_hosts: list[str]) -> None:
    """Enforce the URL guards (https-only, no userinfo, host allowlist).

    Raises `GitSourceError` with the matching code; a caller can invoke
    this before writing any audit/provenance so a rejected URL leaves no trace.
    """
    if repo_url.startswith("-"):
        raise GitSourceError(
            422,
            "deploy.git_url_invalid",
            "repo_url must not start with '-'.",
            hint="Use an https://<host>/<owner>/<repo>.git URL.",
        )
    parts = urlsplit(repo_url)
    if parts.scheme != "https" or not parts.hostname:
        raise GitSourceError(
            422,
            "deploy.git_url_invalid",
            "repo_url must be an https:// URL with a host.",
            hint="Only https:// clone URLs are accepted (no ssh/git/file/local paths).",
        )
    if parts.username or parts.password:
        raise GitSourceError(
            422,
            "deploy.git_url_credentials",
            "repo_url must not embed credentials.",
            hint="Pass a private-repo token via token_ref (a secret reference), not in the URL.",
        )
    host = parts.hostname.lower()
    if host not in {h.lower() for h in allowed_hosts}:
        raise GitSourceError(
            422,
            "deploy.git_host_forbidden",
            f"Host '{host}' is not in the allowed clone hosts.",
            hint="Add it to [git].allowed_hosts to permit clones from this forge.",
        )


def validate_ref(ref: str | None) -> None:
    """Enforce the ref grammar (no argv injection, no `..` component)."""
    if ref is None:
        return
    if not _REF_RE.match(ref) or ".." in ref.split("/"):
        raise GitSourceError(
            422,
            "deploy.git_ref_invalid",
            f"ref '{ref}' is not a valid branch/tag name.",
            hint="A ref must be a branch or tag name with no leading '-' or '..' component.",
        )


def validate_subdir(subdir: str | None) -> None:
    """Enforce the subdir grammar (relative, no leading `-`, no `..`)."""
    if subdir is None:
        return
    pp = PurePosixPath(subdir)
    if subdir.startswith("-") or pp.is_absolute() or ".." in pp.parts:
        raise GitSourceError(
            422,
            "deploy.git_subdir_invalid",
            f"subdir '{subdir}' must be a relative path with no '..' components.",
            hint="Point at a directory inside the repo, e.g. 'services/api'.",
        )


def _forward_ca_bundle(child_env: dict[str, str]) -> None:
    """Honour a daemon-level `SSL_CERT_FILE` override for git's TLS too.

    The daemon's own HTTPS clients (the link claim, the tunnel) already trust
    a custom CA bundle through `SSL_CERT_FILE`; the git child env is an
    allowlist, so without this line a clone would verify against a DIFFERENT
    trust store than the process that authorized it. libcurl does not read
    `SSL_CERT_FILE`, so the value is forwarded as `GIT_SSL_CAINFO`. Only
    ever a bundle the operator set on the daemon process — nothing request-
    controlled reaches here.
    """
    bundle = os.environ.get("SSL_CERT_FILE")
    if bundle:
        child_env["GIT_SSL_CAINFO"] = bundle


def _write_askpass_helper(parent: Path) -> Path:
    """Write the static GIT_ASKPASS helper (0o700) beside the clone dir."""
    fd, path = tempfile.mkstemp(prefix=".nerdit-askpass-", dir=str(parent))
    with os.fdopen(fd, "w") as handle:
        handle.write(_ASKPASS_SCRIPT)
    os.chmod(path, 0o700)
    return Path(path)


async def _run_git(
    args: list[str], *, env: dict[str, str], timeout_s: float, token: str | None
) -> tuple[int, bytes, bytes]:
    """Run a git subprocess with a hard timeout; kill + raise on expiry.

    Never a shell (argv list only) and never `os.environ` wholesale (an
    allowlisted `env`). Returns `(returncode, stdout, stderr)`.
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        proc.kill()
        await proc.wait()
        raise GitSourceError(
            504,
            "deploy.git_timeout",
            f"git operation timed out after {timeout_s:g}s.",
            hint="Increase [git].clone_timeout_s or check the repository size/connectivity.",
        ) from exc
    assert proc.returncode is not None  # communicate() has completed
    return proc.returncode, stdout, stderr


def _tree_size(root: Path) -> int:
    """Sum the byte size of the checkout (no symlink following)."""
    total = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                total += _tree_size(Path(entry.path))
            else:
                total += entry.stat(follow_symlinks=False).st_size
    return total


async def clone_source(
    repo_url: str,
    *,
    ref: str | None = None,
    subdir: str | None = None,
    dest_dir: Path,
    token: str | None = None,
    timeout_s: float,
    max_bytes: int,
    allowed_hosts: list[str],
) -> GitSourceInfo:
    """Shallow-clone `repo_url` into `dest_dir` and return a pinned source.

    Order: git-available → URL/ref/subdir guards (no subprocess yet) →
    `git clone --depth 1` → `rev-parse` pin → strip `.git` → size cap →
    subdir containment. Any failure after `dest_dir` may exist `rmtree`s it
    (never leak a tree). The token, when present, reaches the child only via
    `GIT_ASKPASS` + `NERDIT_GIT_TOKEN` — never argv, never disk, never logs.
    """
    if not git_available():
        raise GitSourceError(
            503,
            "deploy.git_unavailable",
            "The git binary is not installed on the daemon host.",
            hint="Install git, or set [git].enabled = false to disable deploy-from-git.",
        )
    validate_repo_url(repo_url, allowed_hosts)
    validate_ref(ref)
    validate_subdir(subdir)

    dest_dir = Path(dest_dir)
    child_env = {
        "PATH": os.environ.get("PATH", ""),
        # Isolate HOME and null the global config: a url.<base>.insteadOf in
        # the daemon user's ~/.gitconfig (or XDG git config) would rewrite the
        # validated https URL to file://... or another host BEFORE the clone,
        # bypassing the https-only + allowed_hosts egress guards. /dev/null is
        # never a directory, so this holds on git versions without
        # GIT_CONFIG_GLOBAL support too.
        "HOME": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    _forward_ca_bundle(child_env)
    askpass_path: Path | None = None
    try:
        if token:
            askpass_path = _write_askpass_helper(dest_dir.parent)
            child_env["GIT_ASKPASS"] = str(askpass_path)
            child_env["NERDIT_GIT_TOKEN"] = token

        clone_cmd = [
            "git",
            "-c",
            "credential.helper=",
            "clone",
            "--depth",
            "1",
            "--single-branch",
            *(["--branch", ref] if ref else []),
            "--",
            repo_url,
            str(dest_dir),
        ]
        try:
            rc, _out, err = await _run_git(
                clone_cmd, env=child_env, timeout_s=timeout_s, token=token
            )
            if rc != 0:
                tail = _scrub(err.decode("utf-8", "replace"), token)[-_STDERR_TAIL:]
                raise GitSourceError(
                    400,
                    "deploy.git_clone_failed",
                    f"git clone failed: {tail}".rstrip(),
                    hint=_clone_failed_hint(tail, had_token=token is not None),
                )

            rc, out, err = await _run_git(
                ["git", "-C", str(dest_dir), "rev-parse", "HEAD"],
                env=child_env,
                timeout_s=timeout_s,
                token=token,
            )
            if rc != 0:
                tail = _scrub(err.decode("utf-8", "replace"), token)[-_STDERR_TAIL:]
                raise GitSourceError(
                    400,
                    "deploy.git_clone_failed",
                    f"git rev-parse failed: {tail}".rstrip(),
                    hint="check the URL/ref; ref must be a branch or tag.",
                )
            commit_sha = out.decode("utf-8", "replace").strip()

            if ref is None:
                rc, out, err = await _run_git(
                    ["git", "-C", str(dest_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                    env=child_env,
                    timeout_s=timeout_s,
                    token=token,
                )
                if rc != 0:
                    tail = _scrub(err.decode("utf-8", "replace"), token)[-_STDERR_TAIL:]
                    raise GitSourceError(
                        400,
                        "deploy.git_clone_failed",
                        f"git rev-parse failed: {tail}".rstrip(),
                        hint="check the URL/ref; ref must be a branch or tag.",
                    )
                resolved_ref = out.decode("utf-8", "replace").strip()
            else:
                resolved_ref = ref

            # Strip .git — parity with ZIP_EXCLUDE_PATTERNS: history (and any
            # committed secrets) must not be baked into the image.
            await asyncio.to_thread(shutil.rmtree, dest_dir / ".git", ignore_errors=True)

            size = await asyncio.to_thread(_tree_size, dest_dir)
            if size > max_bytes:
                raise GitSourceError(
                    413,
                    "deploy.git_too_large",
                    f"Cloned tree is {size} bytes; limit is {max_bytes} bytes.",
                    hint="Deploy a smaller repository or raise [git].max_clone_bytes.",
                )

            if subdir:
                resolved = (dest_dir / subdir).resolve()
                if not resolved.is_relative_to(dest_dir.resolve()) or not resolved.is_dir():
                    raise GitSourceError(
                        422,
                        "deploy.git_subdir_invalid",
                        f"subdir '{subdir}' is not an existing directory inside the repo.",
                        hint="Point at a directory that exists in the repository.",
                    )
                context_dir = resolved
            else:
                context_dir = dest_dir

            return GitSourceInfo(
                commit_sha=commit_sha,
                resolved_ref=resolved_ref,
                context_dir=context_dir,
            )
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, dest_dir, ignore_errors=True)
            raise
    finally:
        if askpass_path is not None:
            try:
                askpass_path.unlink()
            except OSError:
                pass


async def ls_remote_head(
    repo_url: str,
    *,
    ref: str | None,
    token: str | None,
    timeout_s: float,
    allowed_hosts: list[str],
) -> str:
    """Resolve a ref's peeled commit SHA with one bounded, read-only Git probe.

    Check Git availability, URL, and ref before any subprocess, matching clone
    security guards. Request the peeled tag pattern too: comparing a tag-object ID
    with the deployed commit would trigger endless redeploys.

    Create the mode-0700 askpass helper in a private temporary directory and remove
    it on every exit. Pass tokens through child env, never argv.
    """
    if not git_available():
        raise GitSourceError(
            503,
            "deploy.git_unavailable",
            "The git binary is not installed on the daemon host.",
            hint="Install git, or set [git].enabled = false to disable deploy-from-git.",
        )
    validate_repo_url(repo_url, allowed_hosts)
    validate_ref(ref)

    child_env = {
        "PATH": os.environ.get("PATH", ""),
        # Same isolation as the clone: an insteadOf rule in the daemon user's
        # git config must not rewrite the validated URL past the guards.
        "HOME": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    _forward_ca_bundle(child_env)
    with tempfile.TemporaryDirectory(prefix="nerdit-lsremote-") as scratch:
        if token:
            askpass_path = _write_askpass_helper(Path(scratch))
            child_env["GIT_ASKPASS"] = str(askpass_path)
            child_env["NERDIT_GIT_TOKEN"] = token
        # Both patterns: the ref itself, and its peeled form. `--exit-code`
        # still succeeds as long as ONE of them matches (a branch, a lightweight
        # tag and `HEAD` simply have no `^{}` line). The suffix is appended
        # after `validate_ref`, so the ref grammar guard is unaffected.
        pattern = ref or "HEAD"
        rc, out, err = await _run_git(
            [
                "git",
                "-c",
                "credential.helper=",
                "ls-remote",
                "--exit-code",
                "--",
                repo_url,
                pattern,
                f"{pattern}^{{}}",
            ],
            env=child_env,
            timeout_s=timeout_s,
            token=token,
        )

    if rc != 0:
        tail = _scrub(err.decode("utf-8", "replace"), token)[-_STDERR_TAIL:]
        raise GitSourceError(
            400,
            "deploy.git_clone_failed",
            f"git ls-remote failed: {tail}".rstrip(),
            hint=_clone_failed_hint(tail, had_token=token is not None),
        )

    # Remote-controlled output, selected by REFNAME so the answer is the object
    # `git clone --depth 1 --single-branch --branch <ref>` would check out.
    # Precedence mirrors clone's for a named ref (verified against git 2.50):
    # a branch WINS over a same-named tag — preferring any `^{}` line would
    # hand back the tag's commit while the clone checks out the branch, so the
    # poller would see permanent phantom drift and redeploy on every tick — and
    # an annotated tag resolves to its peeled commit, never the tag object.
    # Either way the object name is accepted only when it is a full lowercase
    # sha. Anything else (an empty listing, a symref line, a truncated id) is an
    # error, never a string that goes on to be compared against a persisted
    # commit_sha.
    lines = [
        (fields[1], fields[0])
        for fields in (line.split() for line in out.decode("utf-8", "replace").splitlines())
        if len(fields) >= 2
    ]
    # First line wins per refname: a duplicate is a malformed listing, not a
    # later override.
    by_ref: dict[str, str] = {}
    for refname, obj in lines:
        by_ref.setdefault(refname, obj)
    if ref is None:
        # `HEAD` has no peeled form and is never a `refs/…` path.
        preferred: tuple[str, ...] = ("HEAD",)
    else:
        preferred = (
            f"refs/heads/{ref}",
            f"refs/tags/{ref}^{{}}",
            f"refs/tags/{ref}",
            # A fully-qualified ref (`refs/tags/v1`) matches itself, peeled first.
            f"{ref}^{{}}",
            ref,
        )
    sha = next(
        (by_ref[candidate] for candidate in preferred if candidate in by_ref),
        # Last resort — an unanticipated refname shape still has to clear the
        # sha gate below.
        lines[0][1] if lines else "",
    )
    if not _SHA_RE.match(sha):
        raise GitSourceError(
            400,
            "deploy.git_clone_failed",
            "git ls-remote returned an unexpected ref listing.",
            hint="check the URL/ref; ref must be a branch or tag.",
        )
    return sha
