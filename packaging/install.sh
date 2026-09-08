#!/bin/sh
# nerdit installer — https://get.nerdit.ai
#
# Installs the nerdit node (CLI + daemon + bundled Caddy) from a signed
# release tarball and registers a service-manager unit (systemd on Linux,
# launchd on macOS).  Usage:
#
#     curl -fsSL https://get.nerdit.ai | sh          # Linux (as root) / macOS
#     curl -fsSL https://get.nerdit.ai | sudo sh     # Linux
#     NERDIT_INSTALL_MODE=user sh install.sh         # Linux, per-user install
#     NERDIT_VERSION=0.5.0 sh install.sh             # pin a version
#
# Onboarding is ONE step (D-ENT-1 as reworded by P34): on a FRESH install this
# script also links the node to the console -- attended, by approving it in a
# browser, or unattended from a pre-auth key.  Flags only exist on the
# download-then-run form, because the documented `curl | sh` pipe gives a
# script no argv at all; the environment variables are the channel that works
# everywhere.  See docs/guide/install.md and section 0b below.
#
# On the piped form the assignment goes on the `sh` side of the pipe (and
# AFTER any `sudo`, per the section 4 note): a prefix assignment scopes to the
# ONE command it prefixes, so writing it before `curl` puts it in curl's
# environment, and the `sh` on the other side of the pipe -- a different
# process entirely -- never sees it.  That install then takes neither the key
# path nor the attended browser path and exits 0 with the node unlinked.
#
#     sh install.sh --key-file /run/secrets/nerdit-key   # unattended
#     curl -fsSL ... | NERDIT_AUTH_KEY_FILE=/run/... sh  # the same, piped
#     sh install.sh --no-link                            # install only
#
# This script is the SOURCE OF TRUTH, versioned in the private development
# repository at packaging/install.sh and deployed to get.nerdit.ai.  It is
# NEVER hand-edited on the live host:
# edit it here, run the deploy one-liner.  The same file is shipped inside
# every release tarball and is what `nerdit update` re-executes.
#
# It is also its own updater: re-running it over an existing install stops the
# service, swaps the version directory, flips `current`, and restarts.  The
# data directory (~/.nerdit) and config.toml are never touched.

set -eu

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REPO="${NERDIT_RELEASES_REPO:-nerdit-ai/nerdit}"
REQ_VERSION="${NERDIT_VERSION:-}"
REQ_VERSION="${REQ_VERSION#v}"

DAEMON_PORT=9321
CONSOLE_URL="https://app.nerdit.ai"
MIN_FREE_KB=1048576 # 1 GiB

# --- release signing public key (ECDSA P-256 — D-P30-11 rev 1.2). P-256 and
# --- not Ed25519 because stock macOS ships LibreSSL, which cannot parse an
# --- Ed25519 key at all; `dgst -sha256 -verify` works on every LibreSSL and
# --- OpenSSL build we support, so macOS onboarding stays two commands.
NERDIT_RELEASE_PUBKEY_PEM="-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAErD99ocEK39T60IgnOpitYeNoHD+V
qwskFP+Z5xTkEFhI+zNwUBr06bBcVTzBH71xc472T0ZHRrr5aK6evs6oSg==
-----END PUBLIC KEY-----"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

say() {
	echo "$*"
}

warn() {
	echo "warning: $*" >&2
}

die() {
	echo "error: $*" >&2
	exit 1
}

# Free space on the filesystem holding $1 (walking up to the first existing
# directory), against MIN_FREE_KB. Both the install root AND the staging temp
# dir go through this: they are frequently different filesystems (/opt vs a
# tmpfs /tmp), and a full /tmp used to abort the install *after* the service
# had already been stopped.
check_free_space() {
	_ck_target="$1"
	while [ ! -d "$_ck_target" ]; do
		_ck_target="$(dirname "$_ck_target")"
	done
	_ck_free="$(df -Pk "$_ck_target" 2>/dev/null | awk 'NR>1 {print $4; exit}')"
	case "${_ck_free:-}" in
	'' | *[!0-9]*)
		warn "could not determine free space on $_ck_target; continuing."
		;;
	*)
		if [ "$_ck_free" -lt "$MIN_FREE_KB" ]; then
			die "not enough free space on $_ck_target: $((_ck_free / 1024)) MiB available, $((MIN_FREE_KB / 1024)) MiB required."
		fi
		;;
	esac
}

# True when version $1 sorts strictly before version $2. Dot-separated numeric
# comparison in awk (`sort -V` is not POSIX and BSD sort lacks it); a
# non-numeric component compares as 0, which is enough for the one thing this
# gates: refusing a silent downgrade.
version_lt() {
	awk -v a="$1" -v b="$2" '
	BEGIN {
		na = split(a, x, ".")
		nb = split(b, y, ".")
		n = (na > nb) ? na : nb
		for (i = 1; i <= n; i++) {
			av = (i <= na) ? x[i] + 0 : 0
			bv = (i <= nb) ? y[i] + 0 : 0
			if (av < bv) { exit 0 }
			if (av > bv) { exit 1 }
		}
		exit 1
	}'
}

# --------------------------------------------------------------------------
# 0b. Arguments
#
# The documented primary form is `curl -fsSL https://get.nerdit.ai | sh`, where
# no script argv exists at all -- so every flag below has an environment twin
# and the ENV VARS, not the flags, are the channel that works everywhere. When
# both are given the flag wins: it is the more specific statement of intent and
# it belongs to this invocation, while the env var may be inherited.
#
# Placement note: this loop cannot sit between `set -eu` and the constants
# block (which would let `--version` precede the `NERDIT_VERSION` read),
# because it calls `die`, which is defined further down in the helpers
# section, so running it that early aborts with
# "die: not found" (exit 127) on every refusal it is supposed to make. It
# therefore sits immediately after the helpers, and `--version` assigns
# REQ_VERSION -- the ONLY consumer of NERDIT_VERSION -- directly, with the same
# leading-`v` strip. Every consumer of what this loop sets runs later: the
# key preflight in section 3, REQ_VERSION in section 4, the link step in 12b.
#
# Key custody: `--key` puts the secret on THIS invocation's argv, a hop the
# OPERATOR controls and an accepted, documented cost (D-X16-O12) -- on Linux it
# then sits in this shell's world-readable /proc/<pid>/cmdline for the whole
# multi-minute install. D-X16-O11 governs every hop BELOW this line, and none
# of them ever puts the key on an argv again. Prefer --key-file or
# NERDIT_AUTH_KEY outside ephemeral CI runners.
# --------------------------------------------------------------------------

LINK_KEY="${NERDIT_AUTH_KEY:-}"
LINK_KEY_FILE="${NERDIT_AUTH_KEY_FILE:-}"
# Defence in depth, NOT a ruling being repaired: D-X16-O11 governs argv, and
# an inherited environment is a strictly weaker exposure than one
# (/proc/<pid>/environ is owner-only where /proc/<pid>/cmdline is
# world-readable), which is exactly why the env vars are the documented
# channel in the first place.  But once the value is copied into $LINK_KEY
# there is no reason left for it to keep riding the environment of every
# child this script spawns -- curl, tar, openssl, the service manager, the
# CLI, the closing doctor -- for the whole multi-minute install.  Narrow the
# blast radius: below this line the section 12b printf pipe is the only live
# channel the key travels on.
#
# Safe because this is the script's ONLY read of $NERDIT_AUTH_KEY, and no
# child needs it: `nerdit link` takes the key on stdin (--key-stdin), never
# from the environment.  The update path is untouched too -- `nerdit update`
# re-executes a COPY of this script as a fresh `sh` built from the CLI's own
# os.environ, so an operator's exported key still arrives (and is still
# ignored there, behind NERDIT_SKIP_LINK=1 and the IS_UPDATE guard).
unset NERDIT_AUTH_KEY
SKIP_LINK="${NERDIT_SKIP_LINK:-0}"
LINK_TIMEOUT=600
# Is anything watching? Only the ATTENDED browser branch consults this — an
# unattended install with a key still links, and skipping is still `--no-link`.
# `CI` is the near-universal convention (GitHub Actions, GitLab, CircleCI,
# Jenkins and Buildkite all set it); `DEBIAN_FRONTEND=noninteractive` is the
# same statement from the provisioning side, which is what Ansible and cloud
# images set; `NERDIT_NONINTERACTIVE` is the explicit escape for everyone else.
# Deliberately not `TERM=dumb`: editors and some login shells set it on
# perfectly attended sessions.
UNATTENDED=0
if [ -n "${CI:-}" ] || [ -n "${NERDIT_NONINTERACTIVE:-}" ] ||
	[ "${DEBIAN_FRONTEND:-}" = noninteractive ]; then
	UNATTENDED=1
fi
# Flag provenance, tracked apart from the values so the check below this loop
# can refuse two contradictory FLAGS without also refusing the documented "env
# var set, flag overrides it" case.
KEY_FLAG=0
KEY_FILE_FLAG=0

while [ $# -gt 0 ]; do
	case "$1" in
	--key)
		[ $# -ge 2 ] || die "--key needs a value"
		# The two sources are exclusive by construction, so a flag always beats
		# an inherited env twin whichever of the two the operator set. Two
		# competing FLAGS are refused below the loop, never resolved here.
		LINK_KEY="$2"
		LINK_KEY_FILE=""
		KEY_FLAG=1
		shift 2
		;;
	--key-file)
		[ $# -ge 2 ] || die "--key-file needs a value"
		LINK_KEY_FILE="$2"
		LINK_KEY=""
		KEY_FILE_FLAG=1
		shift 2
		;;
	--no-link)
		SKIP_LINK=1
		shift
		;;
	--link-timeout)
		[ $# -ge 2 ] || die "--link-timeout needs a value"
		case "$2" in
		'' | *[!0-9]*) die "--link-timeout takes a whole number of seconds" ;;
		esac
		LINK_TIMEOUT="$2"
		shift 2
		;;
	--version)
		[ $# -ge 2 ] || die "--version needs a value"
		REQ_VERSION="${2#v}"
		shift 2
		;;
	-*)
		die "unknown option: $1 (see docs/guide/install.md)"
		;;
	*)
		# Deliberately does NOT echo the argument. The likeliest positional a
		# human types here is a pre-auth key they meant to pass to --key, and a
		# secret must never reach error text (D-X16-O11) -- not even on the way
		# to being rejected.
		die "this installer takes options only, not positional arguments (see docs/guide/install.md)"
		;;
	esac
done

# Two competing key FLAGS are a contradiction in this invocation's own argv,
# not a precedence question, so it is refused here beside the other argument
# errors rather than at the filesystem preflight — and refused before anything
# at all has been probed, downloaded or written. (An env var supplying one
# source and a flag the other is NOT a contradiction: that is the documented
# override, and the loop already cleared the source the flag replaced.)
if [ "$KEY_FLAG" = 1 ] && [ "$KEY_FILE_FLAG" = 1 ]; then
	die "--key and --key-file are mutually exclusive; pass exactly one."
fi

# --------------------------------------------------------------------------
# 1. Platform detection
# --------------------------------------------------------------------------

case "$(uname -s)" in
Linux) OS=linux ;;
Darwin) OS=macos ;;
*) die "nerdit supports Linux and macOS only." ;;
esac

case "$(uname -m)" in
x86_64 | amd64) ARCH=x86_64 ;;
aarch64 | arm64) ARCH=arm64 ;;
*) die "nerdit does not ship a build for this CPU architecture ($(uname -m))." ;;
esac

if [ "$OS" = macos ] && [ "$ARCH" = x86_64 ]; then
	die "nerdit ships for Apple Silicon only; Intel Macs are not supported."
fi

if [ "$OS" = linux ] && [ ! -d /run/systemd/system ]; then
	die "nerdit v1 requires systemd to manage the daemon (see docs)."
fi

# --------------------------------------------------------------------------
# 2. Mode selection and layout
# --------------------------------------------------------------------------

EUID_NOW="$(id -u)"

if [ "$OS" = macos ]; then
	[ "$EUID_NOW" != 0 ] ||
		die "do not install nerdit on macOS with sudo; the LaunchAgent is per-user, so rerun this as your normal user."
	MODE=user
elif [ "$EUID_NOW" = 0 ]; then
	if [ "${NERDIT_INSTALL_MODE:-}" = user ]; then
		die "NERDIT_INSTALL_MODE=user is a non-root install; rerun it without sudo."
	fi
	MODE=system
elif [ "${NERDIT_INSTALL_MODE:-}" = user ]; then
	MODE=user
else
	die "installing nerdit on Linux needs root: rerun 'curl -fsSL https://get.nerdit.ai | sudo sh', or set NERDIT_INSTALL_MODE=user for a per-user install under \$HOME/.nerdit."
fi

PLIST_DST=""
if [ "$MODE" = system ]; then
	ROOT=/opt/nerdit
	VERSIONS_DIR=/opt/nerdit
	SHIM=/usr/local/bin/nerdit
	SHIM_TARGET=/opt/nerdit/current/nerdit
	UNIT_SRC=units/nerdit.service
	UNIT_DST=/etc/systemd/system/nerdit.service
	UNIT_USER="${SUDO_USER:-root}"
else
	[ -n "${HOME:-}" ] || die "HOME is not set; a per-user install needs it."
	ROOT="$HOME/.nerdit"
	VERSIONS_DIR="$HOME/.nerdit/versions"
	SHIM="$HOME/.nerdit/bin/nerdit"
	SHIM_TARGET="../current/nerdit"
	if [ "$OS" = macos ]; then
		UNIT_SRC=units/ai.nerdit.daemon.plist
		UNIT_DST="$HOME/Library/LaunchAgents/ai.nerdit.daemon.plist"
		PLIST_DST="$UNIT_DST"
	else
		UNIT_SRC=units/nerdit-user.service
		UNIT_DST="$HOME/.config/systemd/user/nerdit.service"
	fi
	UNIT_USER="$(id -un)"
fi

case "$UNIT_USER" in
'' | *[!A-Za-z0-9._-]*)
	die "refusing to build a service unit for the unusual user name '$UNIT_USER'."
	;;
esac

# An install that already exists OWNS its service identity, and this script
# must not re-derive it. The daemon's data dir is $HOME of the unit user, so
# rewriting User=/Environment=HOME= from whoever happens to run *this*
# invocation (SUDO_USER is set under `sudo nerdit update`, unset in a root
# shell) would silently move a live node onto an empty data dir — services,
# secrets, the internal CA and the node identity all still on disk, none of
# them visible to the daemon. Read what the unit records and keep it.
RECORDED_HOME=""
if [ "$MODE" = system ] && [ -f "$UNIT_DST" ]; then
	RECORDED_USER="$(sed -n 's/^ *User *= *\([^ ]*\).*/\1/p' "$UNIT_DST" | head -n 1)"
	RECORDED_HOME="$(sed -n 's/^ *Environment=HOME=\([^ "]*\).*/\1/p' "$UNIT_DST" | head -n 1)"
	case "${RECORDED_USER:-}" in
	'' | *[!A-Za-z0-9._-]*) RECORDED_USER="" ;;
	esac
	if [ -n "${RECORDED_USER:-}" ] && [ "$RECORDED_USER" != "$UNIT_USER" ]; then
		say "keeping the service user recorded in $UNIT_DST: '$RECORDED_USER' (this run would have used '$UNIT_USER'; the recorded one owns the data dir)"
		UNIT_USER="$RECORDED_USER"
	fi
	case "$RECORDED_HOME" in
	/*) ;;
	*) RECORDED_HOME="" ;;
	esac
fi

if [ "$MODE" = system ]; then
	UNIT_HOME="$RECORDED_HOME"
	if [ -z "$UNIT_HOME" ] && command -v getent >/dev/null 2>&1; then
		UNIT_HOME="$(getent passwd "$UNIT_USER" | cut -d: -f6)"
	fi
	if [ -z "${UNIT_HOME:-}" ]; then
		UNIT_HOME="$(eval "echo ~$UNIT_USER")"
	fi
	case "$UNIT_HOME" in
	/*) ;;
	*) UNIT_HOME=/root ;;
	esac
else
	UNIT_HOME="$HOME"
fi

CURRENT_LINK="$ROOT/current"
NERDITD_PATH="$ROOT/current/nerditd"

# --------------------------------------------------------------------------
# 3. Preflights — every refusal happens here, before the first mutation.
#    A failed preflight leaves `current` and the running service untouched.
# --------------------------------------------------------------------------

# (a) required tools
for _tool in curl tar; do
	command -v "$_tool" >/dev/null 2>&1 ||
		die "$_tool is required and was not found; install it and rerun."
done

# Any openssl will do. The release signature is ECDSA P-256 verified through
# `dgst -sha256 -verify`, which stock macOS LibreSSL supports (D-P30-11
# rev 1.2) — so there is no Homebrew path to hunt for and no prerequisite.
OPENSSL=""
command -v openssl >/dev/null 2>&1 && OPENSSL=openssl
[ -n "$OPENSSL" ] ||
	die "openssl is required to verify the release signature; install it and rerun."

if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
	sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
	die "sha256sum or shasum is required to verify the download; install one and rerun."
fi

# (b) the installer must be release-keyed
case "$NERDIT_RELEASE_PUBKEY_PEM" in
*PLACEHOLDER*)
	die "this installer is not yet release-keyed — see packaging/README.md."
	;;
esac

# (c) Docker
command -v docker >/dev/null 2>&1 ||
	die "Docker is required and was not found — install it first: https://docs.docker.com/engine/install/"

DOCKER_PROBE_USER="$(id -un)"
DOCKER_OK=0
if [ "$MODE" = system ] && [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
	DOCKER_PROBE_USER="$SUDO_USER"
	if sudo -n -u "$SUDO_USER" docker info >/dev/null 2>&1; then
		DOCKER_OK=1
	fi
fi
if [ "$DOCKER_OK" -eq 0 ]; then
	if docker info >/dev/null 2>&1; then
		DOCKER_OK=1
	fi
fi
[ "$DOCKER_OK" -eq 1 ] ||
	die "Docker is installed but its socket is not reachable by $DOCKER_PROBE_USER — is the daemon running, and is the user in the docker group?"

# (d) disk space on the install filesystem (the staging temp dir is checked
#     the same way once mktemp has picked it — they are often different
#     filesystems, and only one of them was ever measured before)
check_free_space "$ROOT"

# (e) on update, the invoking user must own the existing install
IS_UPDATE=0
INSTALLED_VERSION=""
if [ -e "$CURRENT_LINK" ] || [ -L "$CURRENT_LINK" ]; then
	IS_UPDATE=1
	if [ -f "$CURRENT_LINK/VERSION" ]; then
		INSTALLED_VERSION="$(tr -d '\r\n' <"$CURRENT_LINK/VERSION" 2>/dev/null || true)"
	fi
	[ -L "$CURRENT_LINK" ] ||
		die "$CURRENT_LINK exists but is not a symlink; move it aside and rerun."
	# shellcheck disable=SC2012 # fixed path; `stat` uid flags differ GNU vs BSD
	OWNER_UID="$(ls -ldn "$CURRENT_LINK" | awk 'NR==1 {print $3; exit}')"
	case "${OWNER_UID:-}" in
	'' | *[!0-9]*) warn "could not determine the owner of $CURRENT_LINK; continuing." ;;
	*)
		if [ "$OWNER_UID" != "$EUID_NOW" ]; then
			die "the existing install at $ROOT is owned by uid $OWNER_UID; run the update as that user."
		fi
		;;
	esac
fi

# (f) link inputs (P34 D2) — the last refusal, and still well before the first
#     download and the first mutation. Only the key FILE's path is examined
#     here; the key itself is not read until section 12b, and no message on
#     this branch ever names a secret value (D-X16-O11).
#
#     Gated on the link step actually being reachable, and gated on exactly the
#     same three conditions section 12b uses, so the two can never disagree
#     about whether the file matters. Without the gate this refuses installs
#     that were never going to read the file: `nerdit update` re-executes this
#     script with NERDIT_SKIP_LINK=1 but inherits the caller's whole
#     environment, so one stale NERDIT_AUTH_KEY_FILE pointing at a path that
#     has since been cleaned up would abort every future update on that machine
#     before it downloaded anything — a permanent, confusing failure caused by
#     a variable the update explicitly told us to ignore.
if [ "$IS_UPDATE" = 0 ] && [ "$SKIP_LINK" = 0 ] &&
	[ -n "$LINK_KEY_FILE" ] && [ ! -r "$LINK_KEY_FILE" ]; then
	die "the pre-auth key file is not readable: $LINK_KEY_FILE"
fi

# --------------------------------------------------------------------------
# 4. Resolve the version
# --------------------------------------------------------------------------

if [ -n "$REQ_VERSION" ]; then
	VERSION="$REQ_VERSION"
	say "version pinned by NERDIT_VERSION: $VERSION"
else
	VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
		sed -n 's/.*"tag_name"[^"]*"v\{0,1\}\([^"]*\)".*/\1/p' | head -n 1)"
	say "resolved latest release from $REPO: ${VERSION:-<none>}"
	if [ -n "${SUDO_COMMAND:-}" ] && [ "$IS_UPDATE" = 0 ]; then
		# The documented pin is `| sudo NERDIT_VERSION=x sh`; writing the
		# assignment BEFORE sudo puts it in sudo's own environment, where
		# env_reset drops it — a silently ignored pin.
		say "(if you meant to pin a version under sudo, the assignment goes AFTER sudo: curl … | sudo NERDIT_VERSION=x.y.z sh)"
	fi
fi
[ -n "${VERSION:-}" ] ||
	die "could not resolve the latest nerdit release from $REPO; set NERDIT_VERSION=x.y.z to pin one."

# A releases-repo compromise cannot forge a signature, but it CAN re-point
# `latest` at an older, still-validly-signed release and roll every unpinned
# install back onto known-vulnerable code. Going backwards therefore requires
# an explicit pin (`nerdit update --version x.y.z` supplies one).
if [ -z "$REQ_VERSION" ] && [ -n "$INSTALLED_VERSION" ] &&
	version_lt "$VERSION" "$INSTALLED_VERSION"; then
	die "$REPO offers $VERSION but $INSTALLED_VERSION is installed — refusing to downgrade silently. Pin it explicitly if you mean it: nerdit update --version $VERSION"
fi

ASSET="nerdit-$VERSION-$OS-$ARCH.tar.gz"
BASE_URL="https://github.com/$REPO/releases/download/v$VERSION"

say "installing nerdit $VERSION ($OS/$ARCH, $MODE install) into $ROOT"

# --------------------------------------------------------------------------
# 5. Download
# --------------------------------------------------------------------------

# Service-manager verbs live in exactly one place each: the installer, the
# rollback handler below and `nerdit link` must never drift into two dialects
# of "restart the daemon" (they are mirrored in utils/install_layout.py).
stop_unit() {
	if [ "$OS" = macos ]; then
		launchctl bootout "gui/$EUID_NOW/ai.nerdit.daemon" >/dev/null 2>&1 || true
	elif [ "$MODE" = system ]; then
		systemctl stop nerdit.service >/dev/null 2>&1 || true
	else
		systemctl --user stop nerdit.service >/dev/null 2>&1 || true
	fi
}

unit_active() {
	if [ "$OS" = macos ]; then
		launchctl print "gui/$EUID_NOW/ai.nerdit.daemon" >/dev/null 2>&1
	elif [ "$MODE" = system ]; then
		systemctl is-active --quiet nerdit.service 2>/dev/null
	else
		systemctl --user is-active --quiet nerdit.service 2>/dev/null
	fi
}

start_unit() {
	if [ "$OS" = macos ]; then
		launchctl bootstrap "gui/$EUID_NOW" "$PLIST_DST" >/dev/null 2>&1 || true
		launchctl kickstart -k "gui/$EUID_NOW/ai.nerdit.daemon" >/dev/null 2>&1 || true
	elif [ "$MODE" = system ]; then
		# `enable` + `restart`, not `enable --now`: --now is a no-op on a unit
		# systemd still counts as active, which is exactly the state that
		# leaves the previous version's process running the new `current`.
		systemctl daemon-reload
		systemctl enable nerdit.service >/dev/null
		systemctl restart nerdit.service
	else
		systemctl --user daemon-reload
		systemctl --user enable nerdit.service >/dev/null
		systemctl --user restart nerdit.service
	fi
}

# Run "$@" as the UNIT user — the one decision, in one place.
#
# Four steps need it (the 10b token mint, both link paths, the closing doctor)
# and each used to spell the rule out again, with two different answers to the
# same awkward case. D-P30-12 mints the auth token into the unit user's
# ~/.nerdit/config.toml, and the CLI reads token, config and data dir out of
# $HOME — so a step run as the invoking root writes into the wrong home, or
# authenticates with nothing, or both.
#
# When escalation is REQUIRED (system mode, a different unit user) and sudo is
# not there, this warns and returns non-zero rather than running the command as
# root anyway: the mint's semantics, now shared. Running it as root is not a
# degraded version of the right thing, it is a different thing — a token in the
# wrong home, a link the daemon never sees. Callers treat the non-zero the way
# the mint always has: skip that step and print the remedy.
#
# $RUN_AS_UNIT_ESCALATED records which arm was taken (1 = escalation was
# required), so a caller can word its remedy with or without `sudo -u` without
# re-deciding the mode question here. It is set BEFORE the command runs, and is
# unreliable in a pipeline (the right-hand side of a `|` runs in a subshell);
# only the mint, which is not piped, reads it.
RUN_AS_UNIT_ESCALATED=0
run_as_unit() {
	if [ "$MODE" = system ] && [ "$UNIT_USER" != "$(id -un)" ]; then
		RUN_AS_UNIT_ESCALATED=1
		if command -v sudo >/dev/null 2>&1; then
			sudo -n -u "$UNIT_USER" env HOME="$UNIT_HOME" "$@"
			return $?
		fi
		warn "cannot run this step as '$UNIT_USER': sudo was not found. Run it by hand: sudo -u $UNIT_USER $*"
		return 1
	fi
	RUN_AS_UNIT_ESCALATED=0
	"$@"
}

# Rollback state. Declared here — after the last preflight refusal, so the
# "nothing is mutated before a refusal" contract still holds.
TMP=""
STAGING=""
VDIR=""
OLD_VDIR=""
SERVICE_STOPPED=0
SWAP_DONE=0

cleanup() {
	_rc=$?
	# The dangerous window is: previous tree moved aside, replacement not yet
	# in place. Failing there used to leave a dangling `current`, a broken
	# shim, a stopped service and the only good tree under a name nobody has
	# seen. Put it back, say so, and bring the service up again.
	if [ "$_rc" -ne 0 ] && [ "$SWAP_DONE" -eq 0 ] && [ -n "$OLD_VDIR" ]; then
		if [ -d "$OLD_VDIR" ] && [ ! -e "$VDIR" ] && mv "$OLD_VDIR" "$VDIR" 2>/dev/null; then
			OLD_VDIR=""
			echo "error: install failed — restored the previous version at $VDIR" >&2
		else
			echo "error: install failed and $VDIR could not be restored; the previous tree is at $OLD_VDIR — move it back by hand." >&2
		fi
	fi
	# Deliberately NOT gated on SWAP_DONE, unlike the rollback above. The tree
	# must not be moved back after a swap, but the service must come up either
	# way: before the swap the restored old version should run, after it the
	# new one should. Several steps still fail AFTER SWAP_DONE=1 (a read-only
	# /usr or /etc, a full disk, a tarball missing units/) and gating the
	# restart on it left the daemon stopped with nothing said.
	if [ "$_rc" -ne 0 ] && [ "$SERVICE_STOPPED" -eq 1 ]; then
		echo "error: restarting the nerdit service that was stopped for this install" >&2
		start_unit || true
	fi
	if [ -n "$STAGING" ]; then
		rm -rf "$STAGING"
	fi
	if [ -n "$TMP" ]; then
		rm -rf "$TMP"
	fi
	exit "$_rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TMP="$(mktemp -d)"

# The download and the extracted tree live HERE, not on the install
# filesystem — a tmpfs /tmp on a small VM is the common case, and running out
# of room mid-extract used to abort with the service already stopped.
check_free_space "$TMP"

for _f in "$ASSET" SHA256SUMS SHA256SUMS.sig; do
	curl -fsSL -o "$TMP/$_f" "$BASE_URL/$_f" ||
		die "download failed: $BASE_URL/$_f"
done

# --------------------------------------------------------------------------
# 6. Verify — signature first, then checksum. Nothing is extracted before
#    both pass.
# --------------------------------------------------------------------------

printf '%s\n' "$NERDIT_RELEASE_PUBKEY_PEM" >"$TMP/pub.pem"

"$OPENSSL" dgst -sha256 -verify "$TMP/pub.pem" \
	-signature "$TMP/SHA256SUMS.sig" "$TMP/SHA256SUMS" >/dev/null 2>&1 ||
	die "release signature verification FAILED — refusing to install."
say "release signature verified."

EXPECTED="$(awk -v f="$ASSET" '$2 == f || $2 == "*" f {print $1; exit}' "$TMP/SHA256SUMS")"
[ -n "${EXPECTED:-}" ] ||
	die "SHA256SUMS does not list $ASSET — refusing to install."
ACTUAL="$(sha256 "$TMP/$ASSET")"
[ "$EXPECTED" = "$ACTUAL" ] ||
	die "checksum mismatch for $ASSET — refusing to install."
say "checksum verified."

# --------------------------------------------------------------------------
# 7. Extract and stage — BEFORE the service is stopped, and onto the
#    destination filesystem. Everything that can still fail (a full /tmp, a
#    truncated tarball, a full /opt) happens with the daemon still running.
# --------------------------------------------------------------------------

tar -xzf "$TMP/$ASSET" -C "$TMP"
SRC_DIR="$TMP/nerdit-$VERSION"
[ -d "$SRC_DIR" ] ||
	die "the release tarball does not contain the expected nerdit-$VERSION directory."

VDIR="$VERSIONS_DIR/$VERSION"

# Explicit modes, never the inherited umask: sudo unions the invoking umask,
# so a hardened 027/077 login would create /opt/nerdit unreadable by the
# non-root user the system unit runs as — the daemon would fail with 203/EXEC
# and the shim would be permission-denied for everyone.
umask 022
if [ "$MODE" = system ]; then
	mkdir -p "$VERSIONS_DIR"
	chmod 0755 "$VERSIONS_DIR"
else
	if [ ! -d "$ROOT" ]; then
		# $ROOT is also the data dir on a user install; 0700 is what the
		# daemon's own data_dir_perms check wants.
		mkdir -p "$ROOT"
		chmod 0700 "$ROOT"
	fi
	mkdir -p "$VERSIONS_DIR"
fi

# Stage on the destination filesystem so the swap below is a rename, not a
# cross-filesystem copy that can die halfway with `current` already dangling.
STAGING="$VERSIONS_DIR/.staging.$$"
rm -rf "$STAGING"
mv "$SRC_DIR" "$STAGING"
if [ "$MODE" = system ]; then
	# tar run as root preserves the build machine's uid/gid; a system install
	# must be root-owned, not owned by whatever uid built the tarball.
	chown -R 0:0 "$STAGING"
fi

# --------------------------------------------------------------------------
# 8. Stop the running service (update path only)
# --------------------------------------------------------------------------

# Gated on the service being ACTIVE, not on the unit file existing: a
# half-done uninstall can remove the file while systemd keeps running the
# process, and an upgrade that skipped the stop left a 0.5.2 daemon executing
# a deleted tree (E2E 2026-08-23 — its CA bundle was gone, every HTTPS call
# 500'd). The file test stays so a stopped-but-installed unit is restarted too.
if [ -f "$UNIT_DST" ] || unit_active; then
	say "stopping the running nerdit service"
	stop_unit
	SERVICE_STOPPED=1
fi

# --------------------------------------------------------------------------
# 9. Flip — two same-filesystem renames, then the symlink
# --------------------------------------------------------------------------

if [ -e "$VDIR" ]; then
	# Reinstall of the same version: move the old tree aside rather than
	# deleting it in place, so a `nerdit update` running from inside it keeps
	# resolving its own files until the new tree is in position.
	OLD_VDIR="$VDIR.old.$$"
	rm -rf "$OLD_VDIR"
	mv "$VDIR" "$OLD_VDIR"
fi
mv "$STAGING" "$VDIR"
STAGING=""
SWAP_DONE=1

ln -sfn "$VDIR" "$CURRENT_LINK"

if [ "$MODE" = system ]; then
	mkdir -p /usr/local/bin
	ln -sfn "$SHIM_TARGET" "$SHIM"
else
	mkdir -p "$ROOT/bin"
	ln -sfn "$SHIM_TARGET" "$SHIM"
	case ":$PATH:" in
	*":$ROOT/bin:"*) ;;
	*)
		say "add nerdit to your PATH:  export PATH=\"\$HOME/.nerdit/bin:\$PATH\""
		;;
	esac
fi

# --------------------------------------------------------------------------
# 10. Install / refresh the service unit
# --------------------------------------------------------------------------

UNIT_TEMPLATE="$VDIR/$UNIT_SRC"
[ -f "$UNIT_TEMPLATE" ] ||
	die "the release tarball is missing $UNIT_SRC — refusing to leave the daemon unmanaged."

mkdir -p "$(dirname "$UNIT_DST")"
sed -e "s|__NERDITD__|$NERDITD_PATH|g" \
	-e "s|__USER__|$UNIT_USER|g" \
	-e "s|__HOME__|$UNIT_HOME|g" \
	"$UNIT_TEMPLATE" >"$UNIT_DST"
chmod 644 "$UNIT_DST"

# --------------------------------------------------------------------------
# 10b. Mint the daemon auth token — BEFORE the unit is first started.
#      A tokenless daemon attaches the LOCAL *admin* principal to every
#      loopback request, so on a multi-user box every user on it is admin.
#      Idempotent (`--auth-token-only` never overwrites an existing
#      config.toml) and fresh-install only: minting one under an existing node
#      would break whatever already talks to it unauthenticated.
#
#      It MUST run as the unit user. The daemon's data dir is $HOME of that
#      user, and `nerdit link` later reads the same config.toml as the same
#      user — a token written into root's home would be invisible to both.
# --------------------------------------------------------------------------

if [ "$IS_UPDATE" = 0 ]; then
	if ! run_as_unit "$SHIM" init --auth-token-only; then
		if [ "$RUN_AS_UNIT_ESCALATED" = 1 ]; then
			warn "could not mint the daemon auth token as '$UNIT_USER'; the daemon will treat every user on this box as admin. Fix with: sudo -u $UNIT_USER $SHIM init --auth-token-only && <restart the service>"
		else
			warn "could not mint the daemon auth token; the daemon will treat every user on this box as admin. Fix with: $SHIM init --auth-token-only && <restart the service>"
		fi
	fi
fi

start_unit
if [ "$MODE" != system ] && [ "$OS" != macos ]; then
	say "run 'loginctl enable-linger $UNIT_USER' so the daemon keeps running after you log out."
	# A --user unit cannot carry AmbientCapabilities (an unprivileged user
	# manager cannot raise one; the directive fails the unit with
	# 218/CAPABILITIES), so unlike the system install this mode cannot bind the
	# default [proxy].https_port = 443 -- nor the default
	# [proxy.acme].http_port = 80, the HTTP-01 listener public certificates
	# need. One setcap on the caddy binary covers both. Say so once, here,
	# rather than letting the operator discover it as a Caddy respawn loop.
	#
	# setcap is named FIRST because it is the only one-command answer. The
	# move-the-ports route needs two different commands (review round 2): the
	# scalar `config set` reaches [proxy].https_port but cannot address a nested
	# table, so `config set proxy acme.enabled=...` / `acme.http_port=...` is a
	# 422 and the ACME port has to go through `config apply` (or config.toml).
	# Presenting the one scalar command as covering "both ports" left a node
	# with ACME on still trying to bind :80, i.e. exactly the respawn loop this
	# note exists to prevent.
	say "note: a per-user install cannot bind :80 or :443. If you enable the URL layer, run 'sudo setcap cap_net_bind_service=+ep $CURRENT_LINK/caddy' after every update -- one setcap covers both ports."
	say "      to move the ports above 1023 instead: '$SHIM config set proxy https_port=8443' for HTTPS, and for the ACME HTTP-01 listener a [proxy.acme] block with http_port applied via '$SHIM config apply <file.toml>' (config set is scalar-only and cannot address that sub-table). Public certificates still need :80 reachable from the internet, so a moved http_port needs something forwarding :80 to it."
fi

# --------------------------------------------------------------------------
# 11. Prune superseded version directories
#     Keep the one just installed and the one it replaced (that is what
#     `nerdit update --version <prev>` rolls back to). A frozen bundle plus a
#     static Caddy is ~150-200 MiB per release; unbounded accumulation
#     eventually fails the disk preflight with an unrelated-looking message.
# --------------------------------------------------------------------------

if [ -n "$OLD_VDIR" ]; then
	rm -rf "$OLD_VDIR"
	OLD_VDIR=""
fi

# The newest version dir that is neither the one just installed nor the one it
# replaced — the rollback target to preserve across a same-version reinstall.
KEEP_ROLLBACK=""
for _entry in "$VERSIONS_DIR"/*; do
	[ -d "$_entry" ] && [ ! -L "$_entry" ] && [ -f "$_entry/VERSION" ] || continue
	_n="$(basename "$_entry")"
	[ "$_n" = "$VERSION" ] && continue
	[ -n "$INSTALLED_VERSION" ] && [ "$_n" = "$INSTALLED_VERSION" ] && continue
	KEEP_ROLLBACK="$_n"
done

for _entry in "$VERSIONS_DIR"/*; do
	[ -d "$_entry" ] || continue
	if [ -L "$_entry" ]; then
		continue
	fi
	# A version dir is one the installer made: it carries the release stamp.
	[ -f "$_entry/VERSION" ] || continue
	_name="$(basename "$_entry")"
	if [ "$_name" = "$VERSION" ]; then
		continue
	fi
	if [ -n "$INSTALLED_VERSION" ] && [ "$_name" = "$INSTALLED_VERSION" ]; then
		continue
	fi
	# On a same-version reinstall $VERSION and $INSTALLED_VERSION are the same
	# directory, so the two keep-rules above collapse into one and every other
	# version — including the rollback target this loop's header promises to
	# keep — would be deleted. Keep the newest other one.
	if [ "$VERSION" = "$INSTALLED_VERSION" ] && [ "$_name" = "$KEEP_ROLLBACK" ]; then
		continue
	fi
	say "removing superseded version $_name"
	rm -rf "$_entry"
done

# --------------------------------------------------------------------------
# 12. Post-install
# --------------------------------------------------------------------------

# A pre-existing config.toml may set a non-default [daemon].port, which
# `init --auth-token-only` deliberately preserves. Probing the compiled default
# would then time out against a perfectly healthy daemon — and could falsely
# pass if something unrelated answers on 9321. Read the effective port.
PROBE_PORT="$DAEMON_PORT"
if [ -f "$UNIT_HOME/.nerdit/config.toml" ]; then
	_cfg_port="$(sed -n 's/^[[:space:]]*port[[:space:]]*=[[:space:]]*\([0-9]\{1,5\}\).*/\1/p' \
		"$UNIT_HOME/.nerdit/config.toml" | head -n 1)"
	case "$_cfg_port" in
	'' | *[!0-9]*) ;;
	*) [ "$_cfg_port" -ge 1 ] && [ "$_cfg_port" -le 65535 ] && PROBE_PORT="$_cfg_port" ;;
	esac
fi

# One bounded probe, defined once and run twice: here, and again after a
# successful link step, which restarts the daemon out from under this script
# (12b). Two hand-rolled copies of the same loop would be two budgets to keep
# in step — the same one-place-each rule the service-manager verbs follow.
wait_healthy() {
	_wh_i=0
	while [ "$_wh_i" -lt 30 ]; do
		if curl -fs "http://127.0.0.1:$PROBE_PORT/health" >/dev/null 2>&1; then
			return 0
		fi
		_wh_i=$((_wh_i + 1))
		sleep 1
	done
	return 1
}

HEALTHY=0
if wait_healthy; then
	HEALTHY=1
fi
if [ "$HEALTHY" -eq 0 ]; then
	if [ "$IS_UPDATE" = 0 ]; then
		# A node that never answered /health is not installed, whatever the
		# filesystem looks like. Say so with a non-zero exit rather than
		# ending on "Next step: nerdit link".
		die "the daemon did not answer http://127.0.0.1:$PROBE_PORT/health within 30s. The files are in place at $ROOT; check the service logs (systemctl status nerdit.service / journalctl -u nerdit.service, or ~/.nerdit/nerditd-launchd.log on macOS) and rerun once the cause is fixed."
	fi
	warn "daemon did not report healthy within 30s; check the service logs"
fi

# --------------------------------------------------------------------------
# 12b. Drive the link (P34 D2) — FRESH INSTALLS ONLY.
#
#      `nerdit update` re-executes this very script on every fielded machine
#      (cli/commands/update.py runs it as `["sh", <copy>]`, no argv), so a link
#      step that ran there would prompt, block on a browser, or hang a
#      non-interactive updater on a node that has been linked for months. The
#      guard is the same `$IS_UPDATE` literal the token mint uses at 10b, on
#      purpose: a reader scanning for "what only happens on a fresh install"
#      finds one spelling, not two.
#
#      The whole block is best-effort. D-X16-O16 / OD-P34-1: an install whose
#      daemon is healthy exits 0 even when the link failed, timed out, or was
#      refused — the NOT LINKED banner at the end is what communicates it, and
#      a non-zero exit here would break `nerdit update` and every automation
#      wrapper to repeat what the banner already says. Hence every CLI call
#      below sits in an `if` condition: under `set -eu` a bare failing command
#      aborts the script with the CLI's own exit code.
# --------------------------------------------------------------------------

LINKED_NOW=0
# Set when a named key file turns out to be unusable here. "skipping the link"
# then has to MEAN it: the warning used to be followed by a fall-through into
# the attended browser wait below, which contradicts the line just printed and,
# on the headless fleet node the key path exists for, holds the install for the
# full --link-timeout waiting for a human who is not there. An operator who
# named a key asked for the unattended path; a key that cannot be used ends the
# link step, and the NOT LINKED banner says so.
LINK_SKIP=0
if [ "$IS_UPDATE" = 0 ] && [ "$SKIP_LINK" = 0 ] && [ "$HEALTHY" -eq 1 ]; then
	if [ -z "$LINK_KEY" ] && [ -n "$LINK_KEY_FILE" ]; then
		# Re-checked rather than assumed: the preflight proved it readable
		# several minutes and one download ago, and a key delivered as a
		# reaped tmpfile is a real deployment shape. A late `die` here would
		# fail an install whose daemon is up and serving.
		if [ -r "$LINK_KEY_FILE" ]; then
			# `cat` receives the PATH; the key never becomes an argv word.
			LINK_KEY=$(cat "$LINK_KEY_FILE")
			# Readable but empty is the same outcome by a different route (a
			# secret not yet materialised, a truncated tmpfile) and left
			# $LINK_KEY empty, which is exactly the value the attended branch
			# below keys off.
			if [ -z "$LINK_KEY" ]; then
				warn "the pre-auth key file is empty: $LINK_KEY_FILE; skipping the link"
				LINK_SKIP=1
			fi
		else
			warn "cannot read $LINK_KEY_FILE (it was readable at preflight); skipping the link"
			LINK_SKIP=1
		fi
	fi
	if [ "$LINK_SKIP" = 0 ]; then
		if [ -n "$LINK_KEY" ]; then
			say "Linking this node with the pre-auth key..."
			# `printf` is a BUILTIN in every shell this script targets (dash,
			# bash-as-sh, BusyBox ash, macOS /bin/sh), so no exec argv is ever
			# formed for the key: it crosses on the pipe fd and nowhere else
			# (D-X16-O11). `run_as_unit` owns the de-escalation, for the reason
			# stated where it is defined — D-P30-12 mints the auth token into
			# the UNIT user's config.toml, so a link run as the invoking root
			# would authenticate with nothing.
			if printf '%s\n' "$LINK_KEY" | run_as_unit "$SHIM" link --key-stdin --timeout "$LINK_TIMEOUT"; then
				LINKED_NOW=1
			else
				warn "the link step did not complete; the node is installed but not linked"
			fi
			LINK_KEY=""
		elif [ -t 1 ] && [ "$UNATTENDED" = 0 ]; then
			# STDOUT, not stdin: under `curl ... | sh` stdin is the script itself
			# and is never a tty, yet that piped install is exactly the attended
			# onboarding this step exists to fix. A tty on stdout is the honest
			# "a human is watching this" signal.
			#
			# It is honest but not sufficient, which is why `$UNATTENDED` joins it.
			# A tty on stdout is also what `ansible` with `pty: yes`, `docker run
			# -t` and several CI runners produce, and there the browser nobody is
			# watching would hold the install for the full `--link-timeout` (600 s
			# by default) before falling through to the banner. Requiring a tty on
			# STDIN as well would be the obvious guard and is the wrong one: it is
			# false for `curl | sh`, the documented primary form, so it would
			# disable attended onboarding for most humans to fix a case about
			# machines. The automation markers below are narrow, conventional, and
			# say what they mean; anything they miss still has `--no-link`,
			# `NERDIT_SKIP_LINK`, and a key for the unattended path proper.
			say "Approve this node in your browser to finish setup (Ctrl-C skips; link later with: nerdit link --device)."
			# "Ctrl-C skips" has to be MADE true: the keyboard delivers SIGINT to
			# the whole foreground process group, so without a guard the script's
			# own `exit 130` INT trap would abort the install the moment the CLI
			# prints its stop line — no banner, rc 130, a wrapper records a failed
			# install of a perfectly healthy daemon (the D-X16-O16 outcome the
			# exit-0 ruling exists to prevent). A COMMAND trap, deliberately not
			# `trap '' INT`: an ignored signal is inherited by children as SIG_IGN
			# (and Python then installs no KeyboardInterrupt handler at all, making
			# the CLI un-interruptible), while a caught one is reset to default in
			# children — so the CLI answers the ^C itself (dim stop line, exit 1),
			# the shell runs `:` and carries on to the NOT LINKED banner and exit 0.
			# Restored immediately after; a Ctrl-C anywhere else still exits 130.
			trap ':' INT
			if run_as_unit "$SHIM" link --device --timeout "$LINK_TIMEOUT"; then
				LINKED_NOW=1
			fi
			trap 'exit 130' INT
		fi
	fi
fi

# What a successful link owes the closing doctor, in two parts.
#
# (1) The system-mode restart. `nerdit link` finishes with _restart_for_tunnel,
#     which on a systemd SYSTEM unit escalates through `sudo -n systemctl
#     restart` — something the de-escalated $UNIT_USER the step above ran as
#     cannot do non-interactively. So on exactly the headless fleet path the
#     pre-auth key exists for, the CLI prints its "restart it by hand" hint and
#     returns, leaving the node linked with the tunnel down. This script is
#     still root, so it performs that restart itself, through `start_unit` —
#     the one place the service-manager verbs live (see the note above it). The
#     CLI's yellow hint on that path is cosmetic, not a failure.
# (2) The health re-probe. Where the CLI's own restart DOES succeed (user-mode
#     systemd, launchd) it fires and returns WITHOUT waiting for health, so the
#     daemon is mid-drain (uvicorn's 30 s graceful shutdown) at the exact
#     moment the closing doctor would run — printing a red
#     "daemon | fail | unreachable" table on a perfectly good install. Re-probe
#     first; if it times out, say so once and SKIP the doctor rather than end a
#     good install on a false red.
RUN_DOCTOR=1
if [ "$LINKED_NOW" -eq 1 ]; then
	if [ "$MODE" = system ]; then
		start_unit || warn "could not restart the nerdit service after linking; restart it by hand to bring the tunnel up"
	fi
	if wait_healthy; then
		:
	else
		warn "the daemon is restarting after the link and did not answer /health within 30s; skipping the closing doctor. Check it with: $SHIM doctor"
		RUN_DOCTOR=0
	fi
fi

# The closing doctor MUST run as the unit user. D-P30-12 mints the auth token
# into that user's ~/.nerdit/config.toml, and the CLI reads the token from the
# config of whoever runs it — so a doctor run as the invoking user (root, under
# `sudo sh install.sh`) authenticates with nothing, gets 401 from the daemon it
# just installed, and ends a perfectly good install on a red
# "daemon | fail | unreachable: 401 Unauthorized" table. Through `run_as_unit`,
# the same one place the mint in 10b and both link paths go through — including
# its refusal to fall back to root when it cannot become that user: a doctor run
# as root is the red table this note exists to prevent, so skipping it is the
# better answer.
if [ "$RUN_DOCTOR" -eq 1 ]; then
	run_as_unit "$SHIM" doctor || true
fi

# Link state is read back from the unit user's config.toml rather than trusted
# from $LINKED_NOW, so the banner is right in the cases the variable cannot
# know about: an already-linked node being reinstalled, an update, and a link
# the CLI committed on a path this script did not drive. Same hand-parse
# technique as the effective-port read above; plain `say` lines only, no ANSI
# and no box drawing, matching the one banner precedent in this file.
NODE_LINKED=0
if [ -f "$UNIT_HOME/.nerdit/config.toml" ] &&
	sed -n '/^\[link\]/,/^\[/p' "$UNIT_HOME/.nerdit/config.toml" |
	grep -q '^node_id[[:space:]]*=[[:space:]]*"..*"'; then
	NODE_LINKED=1
fi

if [ "$NODE_LINKED" = 1 ]; then
	say "nerdit $VERSION is installed and this node is linked."
else
	say "nerdit $VERSION is installed."
	say ""
	say "NOT LINKED"
	say "The daemon is serving locally, but this node is not linked to $CONSOLE_URL:"
	say "no remote access, no hosted URLs. Deploys, models and databases keep working."
	say "Link it any time:"
	say "    nerdit link --device       # approve in your browser"
	say "    nerdit link --key-stdin    # pipe a pre-auth key from $CONSOLE_URL"
fi
