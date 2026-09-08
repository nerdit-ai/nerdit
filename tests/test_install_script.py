"""Tests for `packaging/install.sh` (P30 WP-3).

Three layers:

1. **Static** — the script is valid POSIX sh (`sh -n`) and still carries the
   shared P30 constants verbatim (artifact grammar, env knobs, the signing
   pubkey placeholder block).
2. **Fail-closed** — an un-keyed installer refuses *before* touching the
   network or the filesystem. Pinned with a stub PATH whose `curl` writes a
   sentinel: the sentinel must stay empty and HOME must stay untouched.
3. **Rehearsal** — the whole install/update/verify flow driven against a
   locally staged, really-P-256-signed "release" with every external command
   stubbed *except `openssl`*, which is the real system one: the whole point
   of D-P30-11 rev 1.2 is that stock LibreSSL verifies these signatures, so
   stubbing it would test the wrong thing. Hermetic otherwise: no network, no
   service manager, no Docker.
"""

from __future__ import annotations

import functools
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "packaging" / "install.sh"
PLACEHOLDER = "__NERDIT_RELEASE_PUBKEY_PLACEHOLDER__"
#: The shipped installer carries the REAL release key (D-P30-11 rev 1.2), so
#: the rehearsal swaps that body for a per-test one and the fail-closed test
#: puts the placeholder back to simulate an un-keyed installer.
_PEM_HEAD = 'NERDIT_RELEASE_PUBKEY_PEM="-----BEGIN PUBLIC KEY-----'
_PEM_TAIL = '-----END PUBLIC KEY-----"'


def _with_pubkey_body(text: str, body: str) -> str:
    """Return `text` with the embedded public-key body replaced by `body`."""
    start = text.index(_PEM_HEAD) + len(_PEM_HEAD)
    end = text.index(_PEM_TAIL, start)
    return text[:start] + "\n" + body + "\n" + text[end:]


_OS = "macos" if sys.platform == "darwin" else "linux"
_ARCH = "arm64" if platform.machine() in ("arm64", "aarch64") else "x86_64"

# The rehearsal drives the *user* install layout, which exists on both
# supported platforms (macOS always; Linux via NERDIT_INSTALL_MODE=user).
_EXEC_SUPPORTED = (
    sys.platform == "darwin"
    or (sys.platform.startswith("linux") and Path("/run/systemd/system").is_dir())
) and os.geteuid() != 0
_exec_only = pytest.mark.skipif(
    not _EXEC_SUPPORTED,
    reason="install.sh execution rehearsal needs a non-root macOS or systemd Linux host",
)


def _script_text() -> str:
    return SCRIPT.read_text()


# ---------------------------------------------------------------------------
# 1. Static
# ---------------------------------------------------------------------------


class TestStatic:
    def test_shebang_and_strict_mode(self):
        text = _script_text()
        assert text.startswith("#!/bin/sh\n")
        assert "\nset -eu\n" in text

    def test_posix_sh_syntax(self):
        proc = subprocess.run(["sh", "-n", str(SCRIPT)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    def test_no_bashisms(self):
        text = _script_text()
        # `[[:space:]]` etc. are POSIX character classes inside a regex, not the
        # bash `[[` test operator — which is never followed by ':'. Without this
        # the check fires on legitimate portable sed.
        scanned = text.replace("[[:", "")
        for bashism in ("[[", "function ", "local ", "$'", "declare "):
            assert bashism not in scanned, f"bashism in install.sh: {bashism!r}"

    def test_carries_a_real_release_signing_pubkey(self):
        """The shipped installer must be keyed — a placeholder fails closed."""
        text = _script_text()
        assert _PEM_HEAD in text
        assert PLACEHOLDER not in text, "install.sh is not release-keyed"
        body = text[text.index(_PEM_HEAD) + len(_PEM_HEAD) : text.index(_PEM_TAIL)]
        key = serialization.load_pem_public_key(
            f"-----BEGIN PUBLIC KEY-----{body}-----END PUBLIC KEY-----\n".encode()
        )
        # P-256 and not Ed25519: stock macOS LibreSSL cannot parse an Ed25519
        # key, so an Ed25519 release signature is unverifiable on platform #2.
        assert isinstance(key, ec.EllipticCurvePublicKey)
        assert isinstance(key.curve, ec.SECP256R1)

    def test_env_knobs(self):
        text = _script_text()
        assert 'REPO="${NERDIT_RELEASES_REPO:-nerdit-ai/nerdit}"' in text
        assert 'REQ_VERSION="${NERDIT_VERSION:-}"' in text
        assert "${NERDIT_INSTALL_MODE:-}" in text

    def test_artifact_and_url_grammar(self):
        text = _script_text()
        assert 'ASSET="nerdit-$VERSION-$OS-$ARCH.tar.gz"' in text
        assert 'BASE_URL="https://github.com/$REPO/releases/download/v$VERSION"' in text
        assert "https://api.github.com/repos/$REPO/releases/latest" in text
        assert "SHA256SUMS.sig" in text

    def test_verifies_signature_then_checksum_before_extracting(self):
        text = _script_text()
        verify = text.index("dgst -sha256 -verify")
        checksum = text.index('EXPECTED="$(awk')
        extract = text.index('tar -xzf "$TMP/$ASSET"')
        assert verify < checksum < extract

    def test_every_refusal_precedes_the_first_mutation(self):
        """No mkdir/ln/mv/rm before the last preflight refusal."""
        text = _script_text()
        # (P34 D2) The link-input preflight is the new last refusal.
        last_preflight = text.index("the pre-auth key file is not readable")
        head = text[:last_preflight]
        for mutator in ('mkdir -p "', 'ln -sfn "', 'mv "', 'rm -rf "'):
            assert mutator not in head, f"{mutator!r} runs before the preflights finish"

    def test_supported_platform_matrix(self):
        text = _script_text()
        assert "nerdit supports Linux and macOS only." in text
        assert "nerdit ships for Apple Silicon only; Intel Macs are not supported." in text
        assert "nerdit v1 requires systemd to manage the daemon (see docs)." in text

    def test_service_manager_commands_match_the_shared_constants(self):
        text = _script_text()
        for cmd in (
            "systemctl stop nerdit.service",
            "systemctl --user stop nerdit.service",
            # `enable` + `restart`, never `enable --now` (a no-op on a unit
            # systemd still counts as active — the state that left a deleted
            # 0.5.2 tree running the new `current`, E2E 2026-08-23).
            "systemctl enable nerdit.service",
            "systemctl restart nerdit.service",
            "systemctl --user enable nerdit.service",
            "systemctl --user restart nerdit.service",
            "systemctl is-active --quiet nerdit.service",
            'launchctl bootout "gui/$EUID_NOW/ai.nerdit.daemon"',
            'launchctl bootstrap "gui/$EUID_NOW" "$PLIST_DST"',
            'launchctl kickstart -k "gui/$EUID_NOW/ai.nerdit.daemon"',
        ):
            assert cmd in text, f"missing canonical service command: {cmd}"
        assert "systemctl enable --now" not in text

    def test_unit_templates_only_use_placeholders_the_installer_renders(self):
        """Cross-package pin: install.sh sed-renders exactly these three."""
        units = sorted((REPO_ROOT / "packaging" / "units").glob("*"))
        if not units:  # units belong to WP-4; nothing to check before they land
            pytest.skip("packaging/units is empty")
        rendered = {"__NERDITD__", "__USER__", "__HOME__"}
        for unit in units:
            found = set(re.findall(r"__[A-Z_]+__", unit.read_text()))
            assert found <= rendered, (
                f"{unit.name} uses unrendered placeholders: {found - rendered}"
            )

    def test_not_linked_banner_replaced_the_final_next_step_line(self):
        """(P34 D2) The install no longer ENDS on a second manual command.

        ``Next step:  nerdit link <code>`` narrated D-ENT-1's two-step
        onboarding: install, then leave for a browser on another surface, copy
        an ``NL-`` code, come back and type it. D2 makes the installer drive
        the link itself, so the closing lines report the OUTCOME instead of
        prescribing homework — and when the node really is unlinked they say so
        loudly (D-X16-O16: loud, but still exit 0).
        """
        text = _script_text()
        assert "Next step:  nerdit link <code>" not in text
        assert 'say "nerdit $VERSION is installed and this node is linked."' in text
        assert 'say "NOT LINKED"' in text
        assert "nerdit link --device" in text
        assert "nerdit link --key-stdin" in text
        # D-ENT-2 rendered as copy: the banner must never suggest that the
        # absence of a link degrades anything on this machine.
        assert "Deploys, models and databases keep working." in text

    def test_argument_loop_rejects_an_unknown_flag_before_any_download(self):
        """(P34 D2) The loop is the first thing that can refuse, and it refuses
        through ``die`` — which is why it sits AFTER the helper definitions and
        not, as first sketched, between ``set -eu`` and the
        constants block, where ``die`` does not exist yet and every refusal it
        is supposed to make would abort with ``die: not found``.
        """
        text = _script_text()
        loop = text.index("while [ $# -gt 0 ]; do")
        assert loop < text.index("# 1. Platform detection")
        assert text.index("die() {") < loop, "the argument loop calls die before it is defined"
        assert 'die "unknown option: $1 (see docs/guide/install.md)"' in text
        # A stray positional is most likely a mistyped pre-auth key, so the
        # refusal must not echo it — D-X16-O11 covers error text too.
        assert 'die "this installer takes options only' in text

    def test_key_flags_fall_back_to_the_documented_env_vars_and_flags_win(self):
        """``curl | sh`` gives a script no argv at all, so the env vars are the
        channel that works everywhere and the flags are the download-then-run
        refinement layered on top of them."""
        text = _script_text()
        assert 'LINK_KEY="${NERDIT_AUTH_KEY:-}"' in text
        assert 'LINK_KEY_FILE="${NERDIT_AUTH_KEY_FILE:-}"' in text
        assert 'SKIP_LINK="${NERDIT_SKIP_LINK:-0}"' in text
        assert "LINK_TIMEOUT=600" in text
        loop = text[text.index("while [ $# -gt 0 ]; do") : text.index("# 1. Platform detection")]
        # Each key flag clears the other source, so a flag beats an inherited
        # env twin whichever of the two the operator happened to set.
        assert 'LINK_KEY="$2"\n\t\tLINK_KEY_FILE=""' in loop
        assert 'LINK_KEY_FILE="$2"\n\t\tLINK_KEY=""' in loop
        # Two competing FLAGS are a mistake, not a precedence question.
        assert 'die "--key and --key-file are mutually exclusive; pass exactly one."' in text

    def test_the_documented_piped_synopsis_puts_the_assignment_on_the_sh_side(self):
        """A prefix assignment scopes to the ONE command it prefixes.

        The header used to synopsize the unattended piped install as
        ``NERDIT_AUTH_KEY_FILE=… curl … | sh``, which puts the variable in
        *curl's* environment; the ``sh`` on the other side of the pipe is a
        different process and never sees it. An operator following that line
        got neither the key path nor the attended browser path: a silent exit 0
        with the node unlinked. Same trap the section 4 note already calls out
        for ``sudo NERDIT_VERSION=…``, and the correction is the same shape, so
        this pins the ONE placement rather than two spellings of it.
        """
        text = _script_text()
        assert "#     curl -fsSL ... | NERDIT_AUTH_KEY_FILE=/run/... sh" in text
        # The broken shape, in any variable: an assignment, then curl, then a
        # pipe into sh. Matched loosely on purpose, because a future example that
        # reintroduces the trap with a different variable is the same bug.
        broken = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"\b[A-Z_]+=\S*\s+curl\b.*\|\s*(sudo\s+)?sh\b", line)
        ]
        assert not broken, f"prefix assignment on curl's side of the pipe: {broken}"

    def test_the_env_key_is_unset_the_moment_it_has_been_copied(self):
        """Defence in depth, not a ruling being repaired.

        D-X16-O11 governs argv, and an inherited environment is the weaker
        exposure (``/proc/<pid>/environ`` is owner-only where
        ``/proc/<pid>/cmdline`` is world-readable), which is exactly why the
        env vars are the documented channel. But once the value lives in
        ``$LINK_KEY`` there is no reason for it to keep riding the environment
        of every child of a multi-minute install, so the script drops it
        immediately and the section 12b ``printf`` pipe becomes the only live
        channel below that line.
        """
        text = _script_text()
        assert 'LINK_KEY="${NERDIT_AUTH_KEY:-}"' in text
        copy = text.index('LINK_KEY="${NERDIT_AUTH_KEY:-}"')
        drop = text.index("\nunset NERDIT_AUTH_KEY\n")
        assert copy < drop, "the key is unset before it is copied"
        # Nothing may EXPAND it afterwards: the copy is the script's only read,
        # which is what makes the unset safe (`nerdit link` takes the key on
        # stdin, and `nerdit update` re-execs a fresh `sh` from the CLI's own
        # os.environ, so the update path is untouched). Matched on the
        # expansion, not the bare name: `NERDIT_AUTH_KEY_FILE` is a different
        # variable that legitimately survives, and prose mentions both.
        tail = text[drop + len("\nunset NERDIT_AUTH_KEY\n") :]
        assert not re.findall(r"\$\{?NERDIT_AUTH_KEY(?![A-Z_])", tail)

    def test_link_step_sits_inside_the_same_is_update_guard_as_the_token_mint(self):
        """``nerdit update`` re-executes this script on every fielded machine; a
        link step running there would prompt, block on a browser, or hang a
        non-interactive updater on a node linked months ago. Same literal guard
        as the 10b token mint, so a reader looking for "fresh-install only"
        finds one spelling rather than two."""
        text = _script_text()
        mint = text.index('if [ "$IS_UPDATE" = 0 ]')
        link = text.index("# 12b. Drive the link")
        guard = text.index('if [ "$IS_UPDATE" = 0 ] && [ "$SKIP_LINK" = 0 ]', link)
        assert mint < link < guard

    def test_link_step_runs_after_the_health_probe_and_before_the_closing_doctor(self):
        text = _script_text()
        probe = text.index("HEALTHY=0")
        link = text.index("# 12b. Drive the link")
        doctor = text.index("# The closing doctor MUST run as the unit user.")
        assert probe < link < doctor

    def test_the_key_variable_is_never_expanded_outside_its_assignments_and_the_printf_pipe(
        self,
    ):
        """D-X16-O11: no hop the installer controls puts the key on an argv.

        ``printf`` is a builtin in every shell this script targets, so the pipe
        forms no exec argv at all, and ``cat`` receives the key FILE's path and
        never its contents. This pins the whole custody claim as a static
        property of the script rather than as a claim in a comment.
        """
        text = _script_text()
        allowed_prefixes = (
            'if [ -z "$LINK_KEY" ]',
            'if [ -n "$LINK_KEY" ]',
            "if printf '%s\\n' \"$LINK_KEY\" |",
            "elif printf '%s\\n' \"$LINK_KEY\" |",
        )
        offenders = [
            line.strip()
            for line in text.splitlines()
            if re.search(r'"\$LINK_KEY"', line) and not line.strip().startswith(allowed_prefixes)
        ]
        assert not offenders, f"$LINK_KEY expanded outside its allowed forms: {offenders}"
        # Cleared the moment the pipe has consumed it (D-X16-O11's clearing
        # clause) — the installer keeps no copy for the rest of the run.
        assert '\t\tLINK_KEY=""\n' in text
        # The file is read BY PATH; its contents never become a command word.
        assert 'LINK_KEY=$(cat "$LINK_KEY_FILE")' in text

    def test_link_cli_invocations_are_exit_status_guarded_so_set_e_cannot_abort_the_install(
        self,
    ):
        """OD-P34-1 / D-X16-O16: a healthy daemon exits 0 even when the link
        failed. Under ``set -eu`` a bare failing command aborts the script with
        the CLI's own exit code, so every link call sits in an ``if``."""
        text = _script_text()
        block = text[text.index("# 12b. Drive the link") : text.index("RUN_DOCTOR=1")]
        calls = [ln.strip() for ln in block.splitlines() if '"$SHIM" link' in ln]
        assert calls, "no link invocation found in the link step"
        for call in calls:
            assert call.startswith(("if ", "elif ", '"$SHIM" link')) or "sudo" in call, call
        # The key path is a pipeline whose head is the guard.
        assert 'if printf \'%s\\n\' "$LINK_KEY" | run_as_unit "$SHIM" link --key-stdin' in text
        assert "the link step did not complete; the node is installed but not linked" in text

    def test_link_step_uses_the_same_sudo_unit_user_shape_as_the_mint_and_doctor(self):
        """D-P30-12 mints the auth token into the UNIT user's config.toml, so a
        link run as the invoking root would authenticate with nothing and 401
        against the daemon this script just installed.

        The shape now lives in exactly one place, ``run_as_unit()``. Pinning the
        literal ``sudo -n -u ...`` at each call site is what let the four steps
        drift apart in the first place — only the mint refused to fall back to
        running as root, so the two link paths and the doctor silently did the
        wrong thing when sudo was absent. The contract is therefore restated as
        the stronger one: ONE escalation in the file, inside ``run_as_unit``,
        and every step that needs the unit user's ``$HOME`` routed through it.
        """
        text = _script_text()
        escalation = 'sudo -n -u "$UNIT_USER" env HOME="$UNIT_HOME" "$@"'
        assert text.count(escalation) == 1, "the sudo escalation is not in exactly one place"
        start = text.index("run_as_unit() {")
        body = text[start : text.index("\n}", start)]
        assert escalation in body, "the one escalation is not inside run_as_unit()"
        for step in (
            'run_as_unit "$SHIM" init --auth-token-only',
            'run_as_unit "$SHIM" link --key-stdin',
            'run_as_unit "$SHIM" link --device',
            'run_as_unit "$SHIM" doctor',
        ):
            assert step in text, f"step does not run as the unit user: {step}"

    def test_device_branch_gates_on_stdout_tty_not_stdin(self):
        """Under ``curl ... | sh`` stdin IS the script and is never a tty — yet
        that piped install is exactly the attended onboarding D2 exists to fix.
        A tty on STDOUT is the honest "a human is watching this" signal."""
        text = _script_text()
        assert 'elif [ -t 1 ] && [ "$UNATTENDED" = 0 ]; then' in text

    def test_the_attended_branch_also_stands_down_for_automation(self):
        """A tty on stdout is honest but not sufficient.

        ``ansible`` with ``pty: yes``, ``docker run -t`` and several CI runners
        all produce one, and there the browser nobody is watching would hold the
        install for the full ``--link-timeout`` before falling through to the
        banner. Requiring a tty on STDIN instead is the obvious guard and the
        wrong one: it is false for ``curl | sh``, the documented primary form,
        so it would disable attended onboarding for most humans to fix a case
        about machines. These markers are narrow, conventional, and opt-out-able.
        """
        text = _script_text()
        start = text.index("UNATTENDED=0")
        # Only the detection block itself — a wider slice sweeps in unrelated
        # traps and turns the TERM assertion below into noise.
        block = text[start : text.index("\nfi", start)]
        assert '[ -n "${CI:-}" ]' in block
        assert '[ -n "${NERDIT_NONINTERACTIVE:-}" ]' in block
        assert '[ "${DEBIAN_FRONTEND:-}" = noninteractive ]' in block
        # TERM=dumb is deliberately NOT a marker: editors and some login shells
        # set it on perfectly attended sessions.
        assert "TERM" not in block
        assert "[ -t 0 ]" not in text

    def test_device_branch_shields_the_install_from_the_keyboard_sigint(self):
        """ "Ctrl-C skips" must be MADE true: the keyboard delivers SIGINT to the
        whole foreground process group, so without a shield the script's own
        ``exit 130`` INT trap would abort the install the moment the CLI prints
        its stop line — no NOT LINKED banner, rc 130, a wrapper records a failed
        install of a healthy daemon (the D-X16-O16 outcome). The shield must be
        a COMMAND trap, never ``trap '' INT``: an ignored disposition is
        inherited by children (and Python then installs no KeyboardInterrupt
        handler at all), while a caught one resets to default in children — so
        the CLI answers the ^C itself and the shell carries on to the banner.
        And it must be restored, so a Ctrl-C anywhere else still exits 130."""
        text = _script_text()
        attended = text[text.index('elif [ -t 1 ] && [ "$UNATTENDED" = 0 ]; then') :]
        shield = attended.index("trap ':' INT")
        restore = attended.index("trap 'exit 130' INT")
        # Shield before the CLI runs, restore after — both inside the branch.
        # Anchored on the invocation, not the banner copy that names the verb.
        assert shield < attended.index('"$SHIM" link --device') < restore
        # No line may EXECUTE an ignore-trap (a comment may explain why not).
        assert not any(
            line.strip().startswith("trap '' INT")
            for line in text.splitlines()
            if not line.strip().startswith("#")
        )

    def test_system_mode_link_step_is_followed_by_a_root_side_unit_restart(self):
        """On a systemd SYSTEM unit the CLI's own ``_restart_for_tunnel``
        escalates through ``sudo -n systemctl restart``, which the de-escalated
        unit user the link step runs as cannot execute — so on exactly the
        headless fleet path the pre-auth key exists for, the install would end
        linked with the tunnel down. The installer is still root here: it
        performs that restart itself, through ``start_unit``, the ONE place the
        service-manager verbs live."""
        text = _script_text()
        tail = text[text.index('if [ "$LINKED_NOW" -eq 1 ]; then') :]
        assert 'if [ "$MODE" = system ]; then\n\t\tstart_unit ||' in tail
        # No second dialect of "restart the daemon" (the section 5 rule).
        assert "systemctl restart nerdit.service" not in tail

    def test_link_step_reprobes_health_before_the_closing_doctor(self):
        """``_restart_for_tunnel`` returns WITHOUT waiting for health, so the
        closing doctor would otherwise run against a daemon mid-drain (uvicorn's
        graceful shutdown is 30 s) and print a red ``daemon | fail |
        unreachable`` row on a perfectly good install. Re-probe first; on
        timeout warn once and SKIP the doctor rather than show a false red."""
        text = _script_text()
        assert "wait_healthy() {" in text
        tail = text[text.index('if [ "$LINKED_NOW" -eq 1 ]; then') :]
        assert "if wait_healthy; then" in tail
        assert "RUN_DOCTOR=0" in tail
        assert 'if [ "$RUN_DOCTOR" -eq 1 ]; then' in text

    def test_extraction_and_staging_precede_the_service_stop(self):
        """Nothing that can still fail may run with the daemon already down.

        The only step between "stop" and "flip" is a same-filesystem rename.
        """
        text = _script_text()
        extract = text.index('tar -xzf "$TMP/$ASSET"')
        stage = text.index('mv "$SRC_DIR" "$STAGING"')
        stop = text.index('say "stopping the running nerdit service"')
        swap = text.index('mv "$STAGING" "$VDIR"')
        assert extract < stage < stop < swap

    def test_both_filesystems_are_size_checked(self):
        """$ROOT and the mktemp staging dir are frequently different mounts."""
        text = _script_text()
        assert 'check_free_space "$ROOT"' in text
        assert 'check_free_space "$TMP"' in text

    def test_failed_swap_rolls_back_and_restarts(self):
        text = _script_text()
        assert "trap cleanup EXIT" in text
        assert 'mv "$OLD_VDIR" "$VDIR"' in text
        assert "restored the previous version at $VDIR" in text
        assert "restarting the nerdit service that was stopped for this install" in text

    def test_an_existing_system_unit_keeps_its_recorded_user_and_home(self):
        """A re-render from $SUDO_USER would move a live node onto an empty
        data dir; the recorded identity wins over this invocation's."""
        text = _script_text()
        assert "RECORDED_USER=\"$(sed -n 's/^ *User *= *" in text
        assert "RECORDED_HOME=\"$(sed -n 's/^ *Environment=HOME=" in text
        assert 'UNIT_USER="$RECORDED_USER"' in text
        assert 'UNIT_HOME="$RECORDED_HOME"' in text

    def test_system_unit_grants_the_privileged_port_capability(self):
        """Live-run defect (P30 fresh-VM run, 2026-08-19): the system unit runs
        as an unprivileged user while ``[proxy].https_port`` defaults to 443, so
        without an ambient capability the bundled Caddy dies with EACCES and
        respawn-loops — every deployed app left without a public_url. Measured
        fix: ``AmbientCapabilities`` is inherited across the fork+exec into the
        Caddy child. ``setcap`` is NOT an equivalent: D-P30-9 puts each release's
        Caddy at a new versioned path, so an update silently drops it.
        """
        unit = (REPO_ROOT / "packaging" / "units" / "nerdit.service").read_text()
        directives = [ln for ln in unit.splitlines() if not ln.lstrip().startswith("#")]
        assert "AmbientCapabilities=CAP_NET_BIND_SERVICE" in directives
        # A narrowed bounding set would strip CAP_FOWNER/CAP_DAC_OVERRIDE/
        # CAP_CHOWN from the User=root rendering, which the daemon does use.
        assert not any(ln.startswith("CapabilityBoundingSet=") for ln in directives)

    def test_user_unit_must_not_ask_for_ambient_capabilities(self):
        """An unprivileged user manager cannot raise one: the directive does not
        degrade, it fails the unit outright (measured: ``Failed to apply ambient
        capabilities (before UID change): Operation not permitted`` →
        ``status=218/CAPABILITIES``). Copying the system unit's line here would
        break every ``--user`` install."""
        unit = (REPO_ROOT / "packaging" / "units" / "nerdit-user.service").read_text()
        directives = [ln for ln in unit.splitlines() if not ln.lstrip().startswith("#")]
        assert not any(ln.startswith("AmbientCapabilities") for ln in directives)

    def test_closing_doctor_runs_as_the_unit_user(self):
        """D-P30-12 mints the auth token into the UNIT user's home, so a doctor
        run as the invoking root gets 401 and ends a good install on a red
        table (observed on the fresh-VM run). Same escalation shape as the
        10b mint — now literally the same code path, ``run_as_unit()``, whose
        single escalation is pinned by
        ``test_link_step_uses_the_same_sudo_unit_user_shape_as_the_mint_and_doctor``."""
        text = _script_text()
        assert 'run_as_unit "$SHIM" doctor' in text, (
            "the closing doctor does not run as the unit user"
        )

    def test_user_mode_warns_that_it_cannot_bind_80_or_443(self):
        """(P26 WP2) The warning covers both privileged ports now.

        A --user unit cannot bind ``[proxy].https_port = 443`` and — since
        ACME HTTP-01 shipped — cannot bind ``[proxy.acme].http_port = 80``
        either. One ``setcap`` on the caddy binary covers both, so the note
        must name both ports or the operator fixes half the problem.

        (review round 2) And the *other* route has to be routed correctly per
        port. ``config set`` only produces scalars, so ``proxy acme.http_port``
        is a 422 — the ACME port goes through ``config apply`` (build log §13,
        review-round-1 finding 2, and ``docs/guide/configuration.md``). The note
        used to present the single ``config set proxy https_port=8443`` command
        as the alternative "for both ports", which leaves a node with ACME on
        still trying to bind :80 — the respawn loop the note exists to prevent.
        """
        text = _script_text()
        assert "a per-user install cannot bind :80 or :443" in text
        assert "cap_net_bind_service" in text
        assert "[proxy.acme]" in text
        # The ACME port is routed through the surface that can actually set it.
        assert "config apply" in text
        assert "http_port" in text
        # And the scalar example is no longer sold as covering both ports.
        assert "set unprivileged ports ($SHIM config set proxy https_port=8443)" not in text

    def test_install_roots_get_an_explicit_mode(self):
        """sudo unions the caller's umask — a hardened 027 would make the tree
        untraversable for the non-root user the system unit runs as."""
        text = _script_text()
        assert "umask 022" in text
        assert 'chmod 0755 "$VERSIONS_DIR"' in text


# ---------------------------------------------------------------------------
# Rig — stub PATH, staged signed release
# ---------------------------------------------------------------------------


def _write_exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _pub_body(key: ec.EllipticCurvePrivateKey) -> str:
    pem = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    return "".join(line for line in pem.splitlines() if "-----" not in line)


#: A sanitised copy of the system bin dirs: every real tool EXCEPT `docker`.
#:
#: The rig used to fall back to `/usr/bin:/bin` on PATH, which made it hermetic
#: on macOS and not on Linux — docker lives in /usr/local/bin on one and
#: /usr/bin on the other, so `test_missing_docker_refuses` removed the stub and
#: then found the REAL docker on CI and installed successfully. A hand-curated
#: allowlist replaced that and promptly missed `gzip`, which `tar -xzf` execs on
#: Linux but not macOS. Enumerating the system dirs is the only version that
#: does not depend on guessing the transitive tool set correctly.
_SYSTEM_BIN_DIRS = ("/usr/bin", "/bin", "/usr/sbin", "/sbin", "/usr/local/bin")
_EXCLUDED_TOOLS = frozenset({"docker"})


@functools.lru_cache(maxsize=1)
def _system_tool_farm() -> Path:
    """Build the farm once per session and share it across rigs."""
    farm = Path(tempfile.mkdtemp(prefix="nerdit-toolfarm-"))
    for d in _SYSTEM_BIN_DIRS:
        src_dir = Path(d)
        if not src_dir.is_dir():
            continue
        for entry in src_dir.iterdir():
            if entry.name in _EXCLUDED_TOOLS or (farm / entry.name).exists():
                continue
            try:
                (farm / entry.name).symlink_to(entry)
            except OSError:
                continue
    # Fail loudly here rather than as a confusing `tar: Cannot exec` later.
    for essential in ("tar", "gzip", "sed", "awk", "openssl", "mktemp"):
        assert (farm / essential).exists(), f"tool farm is missing {essential}"
    assert not (farm / "docker").exists(), "the farm must not expose docker"
    return farm


def _base_env(rig) -> dict[str, str]:
    env = {
        # Stubs first, then the docker-free farm — see _system_tool_farm.
        "PATH": f"{rig.bin}:{_system_tool_farm()}",
        "HOME": str(rig.home),
        "SENTINEL": str(rig.log),
    }
    if sys.platform.startswith("linux"):
        env["NERDIT_INSTALL_MODE"] = "user"
    return env


def _rig(tmp_path: Path, *, keyed: bool = True, curl_fails: bool = False) -> SimpleNamespace:
    home = tmp_path / "home"
    home.mkdir()
    binn = tmp_path / "bin"
    binn.mkdir()
    rel = tmp_path / "rel"
    rel.mkdir()
    log = tmp_path / "calls.log"

    key = ec.generate_private_key(ec.SECP256R1())
    text = _script_text()
    text = _with_pubkey_body(text, _pub_body(key) if keyed else PLACEHOLDER)
    script = tmp_path / "install.sh"
    script.write_text(text)

    if curl_fails:
        _write_exe(
            binn / "curl",
            '#!/bin/sh\necho "curl $*" >> "$SENTINEL"\nexit 1\n',
        )
    else:
        _write_exe(
            binn / "curl",
            "#!/bin/sh\n"
            'OUT=""\n'
            'URL=""\n'
            "while [ $# -gt 0 ]; do\n"
            '  case "$1" in\n'
            '    -o) OUT="$2"; shift ;;\n'
            "    -*) ;;\n"
            '    *) URL="$1" ;;\n'
            "  esac\n"
            "  shift\n"
            "done\n"
            'echo "curl $URL" >> "$SENTINEL"\n'
            'case "$URL" in\n'
            "  */health) exit 0 ;;\n"
            "  *api.github.com*)\n"
            # Quoted: the rehearsal drives paths containing spaces (P34 D2).
            f'    printf \'{{"tag_name":"v%s"}}\' "$(cat "{rel}/LATEST")"\n'
            "    exit 0 ;;\n"
            "esac\n"
            'BN=$(basename "$URL")\n'
            f'[ -f "{rel}/$BN" ] || exit 22\n'
            f'cp "{rel}/$BN" "$OUT"\n',
        )

    _write_exe(binn / "docker", '#!/bin/sh\necho "docker $*" >> "$SENTINEL"\nexit 0\n')
    for name in ("systemctl", "launchctl"):
        _write_exe(binn / name, f'#!/bin/sh\necho "{name} $*" >> "$SENTINEL"\nexit 0\n')

    # NO openssl stub: the real system `openssl` (LibreSSL on macOS, OpenSSL
    # on Linux) must verify these P-256 signatures — that is the property
    # D-P30-11 rev 1.2 exists to guarantee, so the test exercises it for real.

    return SimpleNamespace(home=home, bin=binn, rel=rel, log=log, key=key, script=script)


def _publish(rig, version: str, *, tamper: str | None = None) -> None:
    """Stage a signed release of `version` in the rig's fake releases repo."""
    stage = rig.rel / f"stage-{version}"
    top = stage / f"nerdit-{version}"
    (top / "units").mkdir(parents=True)
    # (P34 D2) The stub records what the installer asked of the CLI. Three
    # separate sinks on purpose: argv goes to the shared call log, a
    # `--key-stdin` invocation's STDIN goes to a file of its own — which is
    # what lets a test prove the key crossed on the pipe and never on an argv,
    # and the child's inherited environment goes to a third, as variable NAMES
    # only. Names, because the whole point of that sink is to prove a secret is
    # absent from the child env, and a rig that wrote the value out to disk to
    # check it would be a poor place to look for one; a name is enough, since
    # `NERDIT_AUTH_KEY` is either inherited or it is not.
    _write_exe(
        top / "nerdit",
        "#!/bin/sh\n"
        'echo "nerdit $*" >> "$SENTINEL"\n'
        "env | sed -n 's/^\\([A-Za-z_][A-Za-z0-9_]*\\)=.*/\\1/p' >> \"$SENTINEL.env\"\n"
        'case " $* " in\n'
        '  *" --key-stdin "*) cat >> "$SENTINEL.stdin" ;;\n'
        "esac\n"
        'echo "nerdit-stub $*"\n'
        'exit "${NERDIT_STUB_EXIT:-0}"\n',
    )
    _write_exe(top / "nerditd", "#!/bin/sh\nexit 0\n")
    _write_exe(top / "caddy", "#!/bin/sh\nexit 0\n")
    (top / "VERSION").write_text(f"{version}\n")
    shutil.copy(SCRIPT, top / "install.sh")
    (top / "units" / "nerdit.service").write_text("ExecStart=__NERDITD__\nUser=__USER__\n")
    (top / "units" / "nerdit-user.service").write_text(
        "ExecStart=__NERDITD__\nEnvironment=HOME=__HOME__\n"
    )
    (top / "units" / "ai.nerdit.daemon.plist").write_text(
        "<string>__NERDITD__</string><string>__HOME__</string>\n"
    )

    asset = f"nerdit-{version}-{_OS}-{_ARCH}.tar.gz"
    tar_path = rig.rel / asset
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(top, arcname=f"nerdit-{version}")
    shutil.rmtree(stage)

    # SHA256SUMS always describes the pristine tarball; a "tarball" tamper
    # corrupts the asset afterwards, exactly like a swapped release asset.
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    if tamper == "tarball":
        with tar_path.open("ab") as fh:
            fh.write(b"corrupted")

    sums = rig.rel / "SHA256SUMS"
    sums.write_text(f"{digest}  {asset}\n")
    (rig.rel / "SHA256SUMS.sig").write_bytes(
        rig.key.sign(sums.read_bytes(), ec.ECDSA(hashes.SHA256()))
    )
    if tamper == "sums-after-signing":
        with sums.open("ab") as fh:
            fh.write(b"tampered\n")
    (rig.rel / "LATEST").write_text(version)


def _run(rig, *args, **extra_env):
    """Run the installer. Positional `args` become the script's own argv —
    which only the download-then-run form has (P34 D2); `curl | sh` has none,
    which is why every flag has an environment twin."""
    env = _base_env(rig)
    env.update(extra_env)
    return subprocess.run(
        ["sh", str(rig.script), *args],
        env=env,
        capture_output=True,
        text=True,
    )


def _link_calls(rig) -> list[str]:
    return [ln for ln in _calls(rig).splitlines() if ln.startswith("nerdit link")]


def _piped_key(rig) -> str:
    stdin_log = Path(f"{rig.log}.stdin")
    return stdin_log.read_text() if stdin_log.exists() else ""


def _child_env_names(rig) -> set[str]:
    """The environment variable NAMES every stubbed CLI invocation inherited."""
    env_log = Path(f"{rig.log}.env")
    return set(env_log.read_text().split()) if env_log.exists() else set()


def _calls(rig) -> str:
    return rig.log.read_text() if rig.log.exists() else ""


# ---------------------------------------------------------------------------
# 2. Fail-closed on the un-keyed installer
# ---------------------------------------------------------------------------


@_exec_only
def test_unkeyed_installer_refuses_before_any_network_or_write(tmp_path):
    rig = _rig(tmp_path, keyed=False, curl_fails=True)
    proc = _run(rig)

    assert proc.returncode == 1
    assert "not yet release-keyed" in proc.stderr
    # The unkeyed hint points at packaging/README.md: RELEASING.md does not
    # ship in the public tree, so the runtime message must not name it.
    assert "packaging/README.md" in proc.stderr
    # never reached the network, never probed Docker, never wrote a byte
    assert _calls(rig) == ""
    assert list(rig.home.iterdir()) == []


# ---------------------------------------------------------------------------
# 3. Rehearsal — fresh install, update, and both verification failures
# ---------------------------------------------------------------------------


def _unit_dst(home: Path) -> Path:
    if sys.platform == "darwin":
        return home / "Library" / "LaunchAgents" / "ai.nerdit.daemon.plist"
    return home / ".config" / "systemd" / "user" / "nerdit.service"


@_exec_only
class TestRehearsal:
    def test_fresh_install(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig)
        assert proc.returncode == 0, proc.stderr

        assert "release signature verified." in proc.stdout
        assert "checksum verified." in proc.stdout
        # (P34 D2) The stub CLI never writes a config.toml, so this rehearsal
        # install ends genuinely unlinked — and says so, loudly, at exit 0.
        lines = [line for line in proc.stdout.strip().splitlines() if line]
        assert "nerdit 0.5.0 is installed." in lines
        assert "NOT LINKED" in lines
        assert lines[-1] == (
            "    nerdit link --key-stdin    # pipe a pre-auth key from https://app.nerdit.ai"
        )

        root = rig.home / ".nerdit"
        assert (root / "versions" / "0.5.0" / "nerdit").is_file()
        assert (root / "current").is_symlink()
        assert (root / "current").resolve() == (root / "versions" / "0.5.0").resolve()
        shim = root / "bin" / "nerdit"
        assert shim.is_symlink() and os.readlink(shim) == "../current/nerdit"
        assert shim.resolve().is_file()

        unit = _unit_dst(rig.home)
        rendered = unit.read_text()
        assert (
            "__NERDITD__" not in rendered
            and "__HOME__" not in rendered
            and "__USER__" not in rendered
        )
        assert str(root / "current" / "nerditd") in rendered

        calls = _calls(rig)
        assert "docker info" in calls
        if sys.platform == "darwin":
            assert "launchctl bootstrap gui/" in calls
            assert "launchctl kickstart -k gui/" in calls
        else:
            assert "systemctl --user enable nerdit.service" in calls
            assert "systemctl --user restart nerdit.service" in calls

    def test_pinned_version_skips_the_latest_lookup(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, NERDIT_VERSION="v0.5.0")
        assert proc.returncode == 0, proc.stderr
        assert "api.github.com" not in _calls(rig)
        assert (rig.home / ".nerdit" / "versions" / "0.5.0").is_dir()

    def test_update_stops_the_service_and_flips_current(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        assert _run(rig).returncode == 0
        rig.log.unlink()

        _publish(rig, "0.5.1")
        proc = _run(rig)
        assert proc.returncode == 0, proc.stderr

        root = rig.home / ".nerdit"
        assert (root / "current").resolve() == (root / "versions" / "0.5.1").resolve()
        assert (root / "versions" / "0.5.0").is_dir()  # previous version kept for rollback

        calls = _calls(rig)
        if sys.platform == "darwin":
            assert "launchctl bootout gui/" in calls
        else:
            assert "systemctl --user stop nerdit.service" in calls

    def test_reinstalling_the_same_version_is_idempotent(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        assert _run(rig).returncode == 0
        proc = _run(rig)
        assert proc.returncode == 0, proc.stderr
        root = rig.home / ".nerdit"
        assert (root / "current").resolve() == (root / "versions" / "0.5.0").resolve()

    def test_unpinned_downgrade_is_refused(self, tmp_path):
        """A releases-repo compromise cannot forge a signature, but it can
        re-point `latest` at an older, validly signed release."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.1")
        assert _run(rig).returncode == 0

        _publish(rig, "0.5.0")  # `latest` now points backwards
        proc = _run(rig)
        assert proc.returncode == 1
        assert "refusing to downgrade silently" in proc.stderr

        root = rig.home / ".nerdit"
        assert (root / "current").resolve() == (root / "versions" / "0.5.1").resolve()

    def test_an_explicit_pin_still_downgrades(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.1")
        assert _run(rig).returncode == 0
        _publish(rig, "0.5.0")

        proc = _run(rig, NERDIT_VERSION="0.5.0")
        assert proc.returncode == 0, proc.stderr
        root = rig.home / ".nerdit"
        assert (root / "current").resolve() == (root / "versions" / "0.5.0").resolve()

    def test_superseded_versions_are_pruned_to_current_plus_previous(self, tmp_path):
        rig = _rig(tmp_path)
        for version in ("0.5.0", "0.5.1", "0.5.2"):
            _publish(rig, version)
            assert _run(rig).returncode == 0

        versions = rig.home / ".nerdit" / "versions"
        assert sorted(p.name for p in versions.iterdir()) == ["0.5.1", "0.5.2"]

    def test_tampered_manifest_fails_the_signature_and_installs_nothing(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0", tamper="sums-after-signing")
        proc = _run(rig)

        assert proc.returncode == 1
        assert "signature verification FAILED" in proc.stderr
        assert not (rig.home / ".nerdit" / "current").exists()
        assert not _unit_dst(rig.home).exists()

    def test_tampered_tarball_fails_the_checksum_and_installs_nothing(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0", tamper="tarball")
        proc = _run(rig)

        assert proc.returncode == 1
        assert "checksum mismatch" in proc.stderr
        assert not (rig.home / ".nerdit" / "current").exists()
        assert not _unit_dst(rig.home).exists()

    def test_missing_docker_refuses_with_a_pointer(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        (rig.bin / "docker").unlink()  # PATH is stub:/usr/bin:/bin — no docker there
        proc = _run(rig)

        assert proc.returncode == 1
        assert "Docker is required and was not found" in proc.stderr
        assert "docs.docker.com" in proc.stderr
        assert not (rig.home / ".nerdit").exists()

    def test_unreachable_docker_socket_refuses(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        _write_exe(rig.bin / "docker", '#!/bin/sh\necho "docker $*" >> "$SENTINEL"\nexit 1\n')
        proc = _run(rig)

        assert proc.returncode == 1
        assert "socket is not reachable" in proc.stderr
        assert not (rig.home / ".nerdit").exists()

    # -----------------------------------------------------------------------
    # P34 D2 — the installer drives the link
    # -----------------------------------------------------------------------

    def test_a_key_file_install_links_with_the_key_on_stdin_and_never_on_argv(self, tmp_path):
        """The custody claim of D-X16-O11, executed rather than asserted.

        The stub CLI records its argv in one sink and, for `--key-stdin`, its
        stdin in another: the key must appear in the second and nowhere else —
        not in the recorded argv of any process the installer spawned, not on
        stdout, not on stderr.
        """
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        keyfile = tmp_path / "preauth.key"
        keyfile.write_text("nk_0123456789ABCDEFGHJKMNPQRSTVWXYZ\n")

        proc = _run(rig, "--key-file", str(keyfile), "--link-timeout", "42")
        assert proc.returncode == 0, proc.stderr

        assert _link_calls(rig) == ["nerdit link --key-stdin --timeout 42"]
        assert _piped_key(rig).strip() == "nk_0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        secret = "nk_0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        assert secret not in _calls(rig)
        assert secret not in proc.stdout
        assert secret not in proc.stderr

    def test_the_env_var_key_links_the_same_way_as_the_flag(self, tmp_path):
        """`curl | sh` has no argv, so NERDIT_AUTH_KEY is the channel that
        works on the documented primary install form."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, NERDIT_AUTH_KEY="nk_ENVENVENVENVENVENVENVENVENVENVEN")
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == ["nerdit link --key-stdin --timeout 600"]
        assert _piped_key(rig).strip() == "nk_ENVENVENVENVENVENVENVENVENVENVEN"

    def test_the_env_var_key_does_not_ride_along_into_the_children(self, tmp_path):
        """The installer copies the key into `$LINK_KEY` and drops the variable
        there and then, so the stdin pipe is the only live channel below that
        line and no child of a multi-minute install carries the credential in
        its environment.

        Defence in depth rather than a D-X16-O11 repair: that ruling is about
        argv, and `/proc/<pid>/environ` is owner-only where
        `/proc/<pid>/cmdline` is world-readable, the very distinction that
        makes the env var the documented channel. The narrowing is still worth
        having, and the CLI is the child best placed to prove it, since it is
        the one that legitimately receives the key (on stdin).
        """
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, NERDIT_AUTH_KEY="nk_INHERITINHERITINHERITINHERITINH")
        assert proc.returncode == 0, proc.stderr
        # The link still happens, and still on the pipe: the unset narrows the
        # exposure without removing the channel.
        assert _link_calls(rig) == ["nerdit link --key-stdin --timeout 600"]
        assert _piped_key(rig).strip() == "nk_INHERITINHERITINHERITINHERITINH"
        names = _child_env_names(rig)
        assert names, "the CLI stub recorded no environment at all"
        assert "NERDIT_AUTH_KEY" not in names

    def test_a_key_flag_overrides_the_key_file_env_var(self, tmp_path):
        """Flags beat their inherited env twin: each key flag clears the other
        source, so precedence never depends on which of the two was set."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        stale = tmp_path / "stale.key"
        stale.write_text("nk_STALESTALESTALESTALESTALESTALES\n")
        proc = _run(
            rig,
            "--key",
            "nk_FLAGFLAGFLAGFLAGFLAGFLAGFLAGFLAGF",
            NERDIT_AUTH_KEY_FILE=str(stale),
        )
        assert proc.returncode == 0, proc.stderr
        assert _piped_key(rig).strip() == "nk_FLAGFLAGFLAGFLAGFLAGFLAGFLAGFLAGF"

    def test_no_link_skips_the_link_step_entirely(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        keyfile = tmp_path / "preauth.key"
        keyfile.write_text("nk_0123456789ABCDEFGHJKMNPQRSTVWXYZ\n")

        proc = _run(rig, "--no-link", "--key-file", str(keyfile))
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []
        assert _piped_key(rig) == ""
        assert "NOT LINKED" in proc.stdout

    def test_an_update_never_runs_the_link_step(self, tmp_path):
        """`nerdit update` re-executes this script on every fielded machine. It
        passes no argv, but an exported NERDIT_AUTH_KEY in the operator's shell
        survives its `dict(os.environ)` copy — and must still never re-link."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        assert _run(rig).returncode == 0
        rig.log.unlink()

        _publish(rig, "0.5.1")
        proc = _run(rig, NERDIT_AUTH_KEY="nk_SHOULDNEVERBEUSEDONANUPDATEXXXXX")
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []
        assert _piped_key(rig) == ""

    def test_an_already_linked_node_gets_the_linked_closing_line(self, tmp_path):
        """The banner reads link state back from the unit user's config.toml
        rather than trusting this run's own outcome, so a reinstall of a node
        linked long ago closes honestly."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        cfg = rig.home / ".nerdit"
        cfg.mkdir()
        (cfg / "config.toml").write_text(
            '[daemon]\nhost = "127.0.0.1"\n\n[link]\nenabled = true\nnode_id = "a1b2c3d4e5f6"\n'
        )
        proc = _run(rig)
        assert proc.returncode == 0, proc.stderr
        assert "nerdit 0.5.0 is installed and this node is linked." in proc.stdout
        assert "NOT LINKED" not in proc.stdout

    def test_a_failing_link_still_exits_zero_with_the_not_linked_banner(self, tmp_path):
        """OD-P34-1 / D-X16-O16: the install fails only when the daemon never
        became healthy. A refused, expired or unreachable link is reported by
        the banner, not by the exit code — a non-zero exit here would break
        `nerdit update` and every automation wrapper."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, NERDIT_AUTH_KEY="nk_REFUSEDREFUSEDREFUSEDREFUSEDREF", NERDIT_STUB_EXIT="1")

        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == ["nerdit link --key-stdin --timeout 600"]
        assert "the link step did not complete" in proc.stderr
        assert "NOT LINKED" in proc.stdout

    def test_an_unknown_flag_refuses_before_any_download_or_write(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--linkk")
        assert proc.returncode == 1
        assert "unknown option: --linkk" in proc.stderr
        assert _calls(rig) == ""
        assert list(rig.home.iterdir()) == []

    def test_a_positional_argument_is_refused_without_echoing_it(self, tmp_path):
        """The likeliest positional a human types here is a pre-auth key meant
        for --key; a secret must not reach error text on its way to being
        rejected (D-X16-O11)."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "nk_MISTYPEDMISTYPEDMISTYPEDMISTYPED")
        assert proc.returncode == 1
        assert "takes options only" in proc.stderr
        assert "nk_MISTYPED" not in proc.stderr
        assert _calls(rig) == ""

    def test_key_and_key_file_together_are_refused_before_any_download(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        keyfile = tmp_path / "preauth.key"
        keyfile.write_text("nk_0123456789ABCDEFGHJKMNPQRSTVWXYZ\n")
        proc = _run(rig, "--key", "nk_BOTHBOTHBOTHBOTHBOTHBOTHBOTHBOTH", "--key-file", str(keyfile))

        assert proc.returncode == 1
        assert "mutually exclusive" in proc.stderr
        assert "nk_BOTH" not in proc.stderr
        # Refused with the other argument errors: nothing probed, downloaded
        # or written — not even the Docker socket check has run.
        assert _calls(rig) == ""
        assert list(rig.home.iterdir()) == []

    def test_an_unreadable_key_file_is_refused_before_any_download(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--key-file", str(tmp_path / "nope.key"))
        assert proc.returncode == 1
        assert "pre-auth key file is not readable" in proc.stderr
        # A filesystem preflight, so it sits with the others in section 3 —
        # after the (read-only) Docker probe, and still before the first
        # download and the first write.
        assert "api.github.com" not in _calls(rig)
        assert "releases/download" not in _calls(rig)
        assert list(rig.home.iterdir()) == []

    def test_a_non_numeric_link_timeout_is_refused(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--link-timeout", "ten minutes")
        assert proc.returncode == 1
        assert "whole number of seconds" in proc.stderr
        assert _calls(rig) == ""

    def test_a_flag_missing_its_value_is_refused(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--key-file")
        assert proc.returncode == 1
        assert "--key-file needs a value" in proc.stderr
        assert _calls(rig) == ""

    def test_the_version_flag_pins_the_release_like_the_env_var(self, tmp_path):
        """`--version` assigns REQ_VERSION directly — the single consumer of
        NERDIT_VERSION — so the pin behaves identically on both channels,
        leading `v` and all."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--version", "v0.5.0")
        assert proc.returncode == 0, proc.stderr
        assert "api.github.com" not in _calls(rig)
        assert (rig.home / ".nerdit" / "versions" / "0.5.0").is_dir()

    def test_an_install_into_a_home_directory_containing_a_space(self, tmp_path):
        """Every new path expansion — the key file, the config read behind the
        banner, the re-probe — has to survive the space that breaks unquoted
        shell. Cheap to check, expensive to discover in the field."""
        base = tmp_path / "a space"
        base.mkdir()
        rig = _rig(base)
        _publish(rig, "0.5.0")
        keyfile = base / "pre auth.key"
        keyfile.write_text("nk_SPACESPACESPACESPACESPACESPACES1\n")

        proc = _run(rig, "--key-file", str(keyfile))
        assert proc.returncode == 0, proc.stderr
        assert " " in str(rig.home)
        assert _link_calls(rig) == ["nerdit link --key-stdin --timeout 600"]
        assert _piped_key(rig).strip() == "nk_SPACESPACESPACESPACESPACESPACES1"
        assert (rig.home / ".nerdit" / "current").is_symlink()
        assert "NOT LINKED" in proc.stdout


class TestUnattendedAttendedBranch:
    """The attended browser wait must not fire where nobody can approve."""

    def test_ci_skips_the_attended_wait_and_still_exits_zero(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, CI="true")
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []
        assert "NOT LINKED" in proc.stdout

    def test_noninteractive_provisioning_skips_it_too(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, DEBIAN_FRONTEND="noninteractive")
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []

    def test_the_explicit_escape_hatch_skips_it_too(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, NERDIT_NONINTERACTIVE="1")
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []

    def test_an_unattended_marker_never_blocks_the_key_path(self, tmp_path):
        """Only the browser branch stands down. An unattended install WITH a key
        is the whole point of the key, so CI must still link."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        key = tmp_path / "key"
        key.write_text("nk_" + "A" * 32)
        key.chmod(0o600)
        proc = _run(rig, "--key-file", str(key), CI="true")
        assert proc.returncode == 0, proc.stderr
        assert any("--key-stdin" in call for call in _link_calls(rig)), _link_calls(rig)


class TestKeyFilePreflightGating:
    """The key-file preflight must not refuse installs that never read the file."""

    def test_an_unreadable_key_file_still_refuses_a_real_link_attempt(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--key-file", str(tmp_path / "gone"))
        assert proc.returncode == 1
        assert "not readable" in proc.stderr
        # Preflight (f) runs after the environment probes, so `docker info` has
        # already happened — what must NOT have happened is any download or any
        # mutation of the install tree.
        assert "api.github.com" not in _calls(rig)
        assert not (rig.home / ".nerdit" / "versions").exists()

    def test_a_stale_key_file_never_blocks_an_update(self, tmp_path):
        """`nerdit update` re-executes this script with NERDIT_SKIP_LINK=1 but
        inherits the caller's whole environment. One stale
        NERDIT_AUTH_KEY_FILE pointing at a path since cleaned up would
        otherwise abort every future update on that machine before it
        downloaded anything — over a variable the update told us to ignore."""
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(
            rig,
            NERDIT_SKIP_LINK="1",
            NERDIT_AUTH_KEY_FILE=str(tmp_path / "deleted-by-a-previous-run"),
        )
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []

    def test_a_stale_key_file_never_blocks_an_install_only_run(self, tmp_path):
        rig = _rig(tmp_path)
        _publish(rig, "0.5.0")
        proc = _run(rig, "--no-link", NERDIT_AUTH_KEY_FILE=str(tmp_path / "gone"))
        assert proc.returncode == 0, proc.stderr
        assert _link_calls(rig) == []
