#!/usr/bin/env bash
# publish_public.sh — assemble the public seed tree from an explicit ALLOWLIST.
#
# WP-PUB / D-OS-4: the public repo is a release-synced mirror, and the leak
# hazard is answered structurally — an internal file that nobody listed here
# does not ship. Never an exclude-list over the whole tree: a new private file
# must be private by default, which only holds if the default is "not copied".
#
# Everything is read out of git at --ref (never the dirty working tree), so a
# stray local file cannot be published either.
#
# Usage:
#   scripts/publish_public.sh --out <dir> [--ref <git ref>]
#
# --ref defaults to HEAD, so a branch is verified on its committed head: run
# this after committing, or the seed is the tree before the work, and its gate
# passing says nothing about the work.

set -euo pipefail

OUT=""
REF="HEAD"

while [ $# -gt 0 ]; do
    case "$1" in
        --out)
            OUT="${2:-}"
            shift 2
            ;;
        --ref)
            REF="${2:-}"
            shift 2
            ;;
        -h | --help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "publish_public.sh: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$OUT" ]; then
    echo "publish_public.sh: --out <dir> is required" >&2
    exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! git rev-parse --verify --quiet "${REF}^{commit}" >/dev/null; then
    echo "publish_public.sh: not a commit-ish: ${REF}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# The allowlist.
# ---------------------------------------------------------------------------

# Copied verbatim out of the private tree. Directories are taken whole, so a
# new file under one of them ships without a further edit here — which is why
# no directory holding internal material (tasks/, .baton/) may ever be added.
# packaging/public/ is pruned back out below: those files are the OVERLAY, and
# shipping them a second time under packaging/ would leak the source layout.
# User guides and API documentation are maintained separately for the website.
# Keep tests beside the public source; neither docs/ nor mkdocs.yml ships.
ALLOW_TREES=(
    "src"
    "tests"
    "docker"
    "examples"
    "packaging"
)

# CHANGELOG.md is NOT here: the private changelog narrates unreleased phases and
# cites internal plans, so the public tree gets the packaging/public/ variant
# below instead — a fresh public history starting at the first public release.
ALLOW_FILES=(
    "pyproject.toml"
    "nerdit.toml.example"
    # Generic ruff/pytest/mypy hooks; the shipped Makefile's `setup` target
    # installs them, so leaving it out would ship a broken command.
    ".pre-commit-config.yaml"
)

# scripts/ is curated file-by-file, not taken whole: it mixes publish-safe
# helpers with cross-repo tooling that points at the private cloud repo.
# When in doubt a script stays out and is named in the manifest's tail.
ALLOW_SCRIPTS=(
    "scripts/check-deps.sh"
    "scripts/import_cycle_scan.py" # tests/test_import_cycle_scan.py drives it
    "scripts/license_tool.py"      # tests/test_license_tool.py drives it
    "scripts/publish_public.sh"    # this script — the allowlist is public too
)

# Public variants, kept under packaging/public/ in the private repo and laid
# down at the seed root ON TOP of the copied tree. The private README.md,
# CONTRIBUTING.md, CLAUDE.md and .github/ are NOT in the allowlist above, so
# these are additions, not replacements — nothing private is ever overwritten
# late enough to have already been staged.
OVERLAY_ROOT="packaging/public"
OVERLAY_FILES=(
    "Makefile" # Contributor commands only; no website build targets.
    "README.md"
    "CONTRIBUTING.md"
    "CLAUDE.md"
    "LICENSE"
    "NOTICE"
    "SECURITY.md"
    "CODE_OF_CONDUCT.md"
    # The public changelog: a fresh history whose first entry is the first
    # public release. The private CHANGELOG.md narrates unreleased phases and
    # cites internal plans, so it is not in ALLOW_FILES.
    "CHANGELOG.md"
    # The private .gitignore names internal tooling (.baton/, .wrangler/,
    # local.md), so the public one is a curated variant rather than a copy.
    # Required, not optional: a seed without one invites a first contributor to
    # commit __pycache__/ and node_modules/.
    ".gitignore"
)
# Required overlay directories (public CI + issue templates), taken whole.
OVERLAY_TREES=(
    ".github"
)

# ---------------------------------------------------------------------------
# Gate: every required public variant must exist at REF, before anything is
# written. A missing LICENSE must abort the publish, not ship a seed without
# one.
# ---------------------------------------------------------------------------

overlay_listing="$(git ls-tree -r --name-only "$REF" -- "$OVERLAY_ROOT" || true)"

missing=()
for f in "${OVERLAY_FILES[@]}"; do
    printf '%s\n' "$overlay_listing" | grep -qxF "${OVERLAY_ROOT}/${f}" ||
        missing+=("${OVERLAY_ROOT}/${f}")
done
for d in "${OVERLAY_TREES[@]}"; do
    printf '%s\n' "$overlay_listing" | grep -qE "^${OVERLAY_ROOT}/${d}/" ||
        missing+=("${OVERLAY_ROOT}/${d}/ (no files)")
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "publish_public.sh: missing required public variant(s) at ref ${REF}:" >&2
    for m in "${missing[@]}"; do
        echo "  - ${m}" >&2
    done
    echo "publish_public.sh: nothing assembled." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Assemble into a staging dir, then move into place: a half-written --out is
# never left behind for a later step to publish.
# ---------------------------------------------------------------------------

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "publish_public.sh: --out ${OUT} exists and is not empty" >&2
    exit 2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/seed" "$STAGE/overlay"

# git archive fails loudly on a pathspec that matches nothing at REF — that is
# the intended behaviour: an allowlist entry that has been renamed away must
# stop the publish, not silently drop a directory from the seed.
git archive --format=tar "$REF" -- \
    "${ALLOW_TREES[@]}" "${ALLOW_FILES[@]}" "${ALLOW_SCRIPTS[@]}" |
    tar -x -C "$STAGE/seed"

rm -rf "${STAGE:?}/seed/${OVERLAY_ROOT}"

# The dashboard bundle is built in CI (§3 Q2), never seeded. Pruned rather than
# assumed untracked so the seed is identical whether or not the private repo
# still carries a committed copy at REF.
rm -rf "${STAGE:?}/seed/src/nerdit/daemon/web/dist"

# Sub-trees pruned back out of the whole-tree entries above. Each one is
# internal material that a whole-directory allowlist would otherwise ship, so
# the prune list is part of the allowlist, not a cleanup: adding a file under
# any of these paths keeps it private without a further edit here.
#
#   packaging/hosting/   — production VPS topology for get.nerdit.ai (host
#                          addresses, ssh/scp deploy steps, what else that box
#                          fronts). No secret values, but an attack map.
#   packaging/RELEASING.md — the private repo's CI secret inventory and
#                          self-hosted-runner runbook.
#   packaging/OSS-GOLIVE.md — the open-sourcing go-live runbook (names the
#                          PUBLISH_SOURCE switch, token handling, key
#                          revocation timing).
SEED_PRUNE=(
    "packaging/hosting"
    "packaging/RELEASING.md"
    "packaging/OSS-GOLIVE.md"
    # The two frozen vendored-meta files under tests/data/node_link_v1/ name the
    # private repository in prose (the X1 freeze resolution text and the pin
    # record). They are frozen — byte-compared against the cloud repo and
    # re-hashed against UPSTREAM_PIN by scripts/check_node_link_drift.py — so
    # they cannot be reworded; they are pruned from the seed instead. The
    # fixture JSONs still ship: they are what defines the wire contract.
    "tests/data/node_link_v1/README.md"
    "tests/data/node_link_v1/UPSTREAM_PIN"
    # …and the conformance test whose subject is that freeze machinery (it
    # asserts README/UPSTREAM_PIN exist and reads the pin): the drift script
    # is not in ALLOW_SCRIPTS either, so the machinery has no public consumer.
    # The protocol tests that exercise the fixture JSONs themselves all ship.
    "tests/test_node_link_conformance.py"
)
for p in "${SEED_PRUNE[@]}"; do
    rm -rf "${STAGE:?}/seed/${p}"
done

git archive --format=tar "$REF" -- "$OVERLAY_ROOT" | tar -x -C "$STAGE/overlay"

for f in "${OVERLAY_FILES[@]}"; do
    cp -p "$STAGE/overlay/$OVERLAY_ROOT/$f" "$STAGE/seed/$f"
done
for d in "${OVERLAY_TREES[@]}"; do
    rm -rf "${STAGE:?}/seed/${d}"
    cp -R "$STAGE/overlay/$OVERLAY_ROOT/$d" "$STAGE/seed/$d"
done

# ---------------------------------------------------------------------------
# Forbidden-reference gate — the last guard, run over the WHOLE assembled seed
# (copied tree + overlay), so a new file that names internal material aborts the
# publish instead of shipping. This mechanizes the manual grep checklist in
# packaging/OSS-GOLIVE.md; that checklist is now a double-check, not the gate.
#
#   the private repo's directory name — zero matches, no exception. Spelled as
#                     two concatenated halves below so this script, which ships
#                     inside the seed, is not itself a match and needs no
#                     exemption: the check stays absolute.
#   tasks/          — the private planning layer, which never ships. Tolerated
#                     only in this script, which ships and names its own prune
#                     paths: an allowlist stating what it excludes is not a
#                     dangling reference. Matched by path, so a *different* file
#                     mentioning tasks/ still aborts.
#   an absolute home path — a maintainer's own machine path left in a docstring
#                     or a comment is a privacy leak and a reference no reader
#                     can follow. Zero matches, no exception.
#   the working-agreement doc — it is not in the allowlist, so any mention of it
#                     in the seed points a public reader at a file that does not
#                     ship. Matched on word boundaries so a longer identifier
#                     that merely contains the name does not abort.
#
# The last two are spelled as two concatenated halves below for the same reason
# as the repo name: this script ships inside the seed and must not be its own
# match, so the checks stay absolute rather than needing an exemption.
# ---------------------------------------------------------------------------

PRIVATE_REPO_NAME="nerdit""_private"
ABSOLUTE_HOME_PATH="/Users""/"
WORKING_AGREEMENT_DOC="AGENTS"".md"

forbidden_hits=""
if hits="$(grep -rnF "$PRIVATE_REPO_NAME" "$STAGE/seed" 2>/dev/null)"; then
    forbidden_hits="${forbidden_hits}${hits}"$'\n'
fi
if hits="$(grep -rnF --exclude=publish_public.sh 'tasks/' "$STAGE/seed" 2>/dev/null)"; then
    forbidden_hits="${forbidden_hits}${hits}"$'\n'
fi
if hits="$(grep -rnF "$ABSOLUTE_HOME_PATH" "$STAGE/seed" 2>/dev/null)"; then
    forbidden_hits="${forbidden_hits}${hits}"$'\n'
fi
if hits="$(grep -rnwF "$WORKING_AGREEMENT_DOC" "$STAGE/seed" 2>/dev/null)"; then
    forbidden_hits="${forbidden_hits}${hits}"$'\n'
fi

if [ -n "${forbidden_hits//[$'\n']/}" ]; then
    echo "publish_public.sh: forbidden private reference(s) in the assembled seed:" >&2
    printf '%s' "$forbidden_hits" | sed "s|^${STAGE}/seed/|  |" >&2
    echo "publish_public.sh: nothing published." >&2
    exit 1
fi

mkdir -p "$OUT"
# Dotfiles included: .github/ is an overlay tree.
(cd "$STAGE/seed" && tar -cf - .) | (cd "$OUT" && tar -xf -)

# ---------------------------------------------------------------------------
# Manifest — every file placed, sorted, plus what was deliberately left out.
# ---------------------------------------------------------------------------

echo "# seed manifest (ref=${REF}, out=${OUT})"
(cd "$OUT" && find . -type f | sed 's|^\./||' | LC_ALL=C sort)

file_count="$(cd "$OUT" && find . -type f | wc -l | tr -d ' ')"
echo "# ${file_count} file(s)"

echo "# excluded top-level paths tracked at ${REF} (not in the allowlist):"
git ls-tree --name-only "$REF" | LC_ALL=C sort | while read -r top; do
    case " ${ALLOW_TREES[*]} ${ALLOW_FILES[*]} scripts " in
        *" ${top} "*) continue ;;
    esac
    echo "#   - ${top}"
done

echo "# pruned back out of the allowlisted trees (internal material):"
for p in "${SEED_PRUNE[@]}"; do
    echo "#   - ${p}"
done
echo "#   - ${OVERLAY_ROOT} (laid down at the seed root instead)"
echo "#   - src/nerdit/daemon/web/dist (built in CI)"

echo "# excluded scripts/ entries (not publish-safe or not needed publicly):"
git ls-tree -r --name-only "$REF" -- scripts | LC_ALL=C sort | while read -r s; do
    case " ${ALLOW_SCRIPTS[*]} " in
        *" ${s} "*) continue ;;
    esac
    echo "#   - ${s}"
done
