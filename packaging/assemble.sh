#!/bin/sh
# assemble.sh — build one Nerdit release tarball from a PyInstaller onedir dist.
#
# P30 WP-1. THIS SCRIPT IS THE SINGLE OWNER OF THE TARBALL INNER LAYOUT.
# CI (.github/workflows/release.yml) and any local build call it; nobody
# re-implements the layout. Change the layout here and only here.
#
# Usage:
#   sh packaging/assemble.sh \
#       --version 0.5.0 \
#       --os linux|macos \
#       --arch x86_64|arm64 \
#       --dist  path/to/dist/nerdit \
#       --caddy path/to/caddy \
#       --out   path/to/output-dir
#
# Produces <out>/nerdit-<version>-<os>-<arch>.tar.gz around a single top-level
# directory nerdit-<version>/ :
#
#   nerdit                        frozen CLI executable
#   nerditd                       frozen daemon executable
#   _internal/                    shared PyInstaller payload
#   caddy                         pinned static Caddy binary (0755)
#   units/nerdit.service          systemd system unit template
#   units/nerdit-user.service     systemd --user unit template
#   units/ai.nerdit.daemon.plist  launchd agent template
#   install.sh                    byte-identical copy of packaging/install.sh
#   VERSION                       "<version>\n"
#
# install.sh is copied in so an installed node carries its own installer:
# `nerdit update` re-execs <ROOT>/current/install.sh.
#
# The last line printed on stdout is the absolute-or-given path of the
# artifact, so a caller can do:  ART=$(sh packaging/assemble.sh … | tail -1)
#
# The unit templates and install.sh belong to other packages; this script only
# READS them and fails naming the file when one is missing.

set -eu

# `CDPATH= cd` is a one-shot assignment prefixing the command, not a stray
# empty assignment — it stops a user's CDPATH from making `cd` print/jump.
# shellcheck disable=SC1007
SELF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

die() {
    printf 'assemble.sh: %s\n' "$1" >&2
    exit 1
}

usage() {
    printf 'usage: sh %s --version <v> --os <linux|macos> --arch <x86_64|arm64> --dist <dir> --caddy <file> --out <dir>\n' "$0" >&2
    exit 2
}

VERSION=
OS=
ARCH=
DIST=
CADDY=
OUT=

while [ $# -gt 0 ]; do
    case "$1" in
        --version) [ $# -ge 2 ] || usage; VERSION=$2; shift 2 ;;
        --os)      [ $# -ge 2 ] || usage; OS=$2;      shift 2 ;;
        --arch)    [ $# -ge 2 ] || usage; ARCH=$2;    shift 2 ;;
        --dist)    [ $# -ge 2 ] || usage; DIST=$2;    shift 2 ;;
        --caddy)   [ $# -ge 2 ] || usage; CADDY=$2;   shift 2 ;;
        --out)     [ $# -ge 2 ] || usage; OUT=$2;     shift 2 ;;
        -h|--help) usage ;;
        *) die "unknown argument '$1' (try --help)" ;;
    esac
done

[ -n "$VERSION" ] || die "--version is required"
[ -n "$OS" ]      || die "--os is required"
[ -n "$ARCH" ]    || die "--arch is required"
[ -n "$DIST" ]    || die "--dist is required"
[ -n "$CADDY" ]   || die "--caddy is required"
[ -n "$OUT" ]     || die "--out is required"

# The version rides the artifact name and the VERSION file; keep it a bare
# release number (no leading v — the git TAG is v-prefixed, the artifact is not).
case "$VERSION" in
    v*) die "--version must be bare (0.5.0), not tag-shaped ('$VERSION')" ;;
    *[!0-9.a-zA-Z-]*) die "--version '$VERSION' contains characters that do not belong in a file name" ;;
esac

case "$OS" in
    linux|macos) ;;
    *) die "--os must be 'linux' or 'macos' (got '$OS')" ;;
esac

case "$ARCH" in
    x86_64|arm64) ;;
    *) die "--arch must be 'x86_64' or 'arm64' (got '$ARCH')" ;;
esac

# macOS x86_64 is not a published target (D-P30-2: Linux x86_64 first, macOS
# arm64 second, Linux arm64 third; Intel Macs are out of scope for v1).
if [ "$OS" = macos ] && [ "$ARCH" = x86_64 ]; then
    die "macOS x86_64 is not a supported target in v1"
fi

[ -d "$DIST" ] || die "--dist '$DIST' is not a directory (expected a PyInstaller onedir, e.g. dist/nerdit)"
[ -x "$DIST/nerdit" ]  || die "missing executable '$DIST/nerdit' — build packaging/nerdit.spec first"
[ -x "$DIST/nerditd" ] || die "missing executable '$DIST/nerditd' — build packaging/nerdit.spec first"
[ -d "$DIST/_internal" ] || die "missing '$DIST/_internal' — the onedir payload"

[ -f "$CADDY" ] || die "--caddy '$CADDY' is not a file (see packaging/caddy.pin for the pinned release)"

UNITS_SRC="$SELF_DIR/units"
for unit in nerdit.service nerdit-user.service ai.nerdit.daemon.plist; do
    [ -f "$UNITS_SRC/$unit" ] || die "missing service template '$UNITS_SRC/$unit'"
done

INSTALL_SRC="$SELF_DIR/install.sh"
[ -f "$INSTALL_SRC" ] || die "missing installer '$INSTALL_SRC'"

STAGE_NAME="nerdit-$VERSION"
ARTIFACT="nerdit-$VERSION-$OS-$ARCH.tar.gz"

mkdir -p "$OUT"
STAGE="$OUT/$STAGE_NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# --- frozen bundle -------------------------------------------------------
cp -p "$DIST/nerdit"  "$STAGE/nerdit"
cp -p "$DIST/nerditd" "$STAGE/nerditd"
chmod 0755 "$STAGE/nerdit" "$STAGE/nerditd"
# `cp -R <dir>/. <dest>` copies the CONTENTS, portably on both GNU and BSD cp.
mkdir -p "$STAGE/_internal"
cp -R "$DIST/_internal/." "$STAGE/_internal/"

# --- pinned Caddy (D-P30-5: bundled, never fetched at install time) -------
cp -p "$CADDY" "$STAGE/caddy"
chmod 0755 "$STAGE/caddy"

# --- service-manager templates (D-P30-8) ---------------------------------
mkdir -p "$STAGE/units"
for unit in nerdit.service nerdit-user.service ai.nerdit.daemon.plist; do
    cp -p "$UNITS_SRC/$unit" "$STAGE/units/$unit"
    chmod 0644 "$STAGE/units/$unit"
done

# --- installer (byte-identical; `nerdit update` re-execs this copy) -------
cp -p "$INSTALL_SRC" "$STAGE/install.sh"
chmod 0755 "$STAGE/install.sh"

# --- version stamp -------------------------------------------------------
printf '%s\n' "$VERSION" > "$STAGE/VERSION"
chmod 0644 "$STAGE/VERSION"

# --- tar -----------------------------------------------------------------
rm -f "$OUT/$ARTIFACT"
tar -czf "$OUT/$ARTIFACT" -C "$OUT" "$STAGE_NAME"
rm -rf "$STAGE"

printf '%s\n' "$OUT/$ARTIFACT"
