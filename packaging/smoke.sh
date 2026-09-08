#!/bin/sh
# smoke.sh — the WP-1 acceptance bar for a frozen Nerdit bundle (P30, plan §3).
#
# `nerdit --version` proves nothing about a PyInstaller build: the risks are
# dynamic imports (uvicorn protocol/loop lookups, pydantic plugins), data files
# that live only inside the PYZ archive, and the bundled Caddy. So this script
# drives a REAL loop against a REAL Docker on a machine with NO Python:
#
#   1  extract + version stamp        5  serve the app through the bundled Caddy
#   2  boot the frozen daemon         6  survive POST /daemon/restart
#   3  nerdit doctor (docker ok)      7  clean up after itself
#   4  deploy the node-starter template end to end
#
# Usage:  sh packaging/smoke.sh <path-to-nerdit-…tar.gz>
#
# The argument must be an ASSEMBLED tarball (packaging/assemble.sh) — the
# canonical layout, so the bundled `caddy` and `VERSION` are present. Run it on
# a clean Linux x86_64 VM and a clean macOS arm64 machine, both with Docker
# running and network access to github.com (the template store clones).
#
# Prerequisites on the host: docker, curl, tar, git (the template store shells
# out to the host's git — the bundle does not carry one).
#
# Everything lands in mktemp dirs under a scratch HOME and is torn down by leg
# 7 (and by the EXIT trap on failure). The daemon is killed by the PID captured
# at spawn — never `pkill` by name, which would hit a real daemon on the box.
# `nerdit daemon restart` is never used either: it prompts.

set -eu

# --- knobs (defaults keep clear of a real daemon on 9321/443) ---------------
PORT=${NERDIT_SMOKE_PORT:-9333}
HTTPS_PORT=${NERDIT_SMOKE_HTTPS_PORT:-9443}
APP=${NERDIT_SMOKE_APP:-smoke-app}
TEMPLATE=node-starter
INSTANCE=nerdit-smoke
HOSTNAME_OVERRIDE=localhost
BOOT_TIMEOUT=${NERDIT_SMOKE_BOOT_TIMEOUT:-60}
DEPLOY_BUDGET=${NERDIT_SMOKE_DEPLOY_BUDGET:-900}
API="http://127.0.0.1:$PORT"

BUNDLE=
WORK=
SMOKE_HOME=
SMOKE_TOKEN=
DAEMON_PID=
CLEANED=0

# The daemon runs under a scratch HOME, and the `docker` CLI resolves its
# BuildKit plugin from $DOCKER_CONFIG/cli-plugins (default $HOME/.docker) — so
# a scratch HOME hides a per-user buildx and EVERY build fails with
# BuildPlatformError. Point DOCKER_CONFIG at the invoking user's real docker
# config, which is exactly the accommodation the daemon's own doctor hint asks
# a service unit to make (D-P30-8).
REAL_HOME=$HOME
SMOKE_DOCKER_CONFIG=${DOCKER_CONFIG:-$REAL_HOME/.docker}

# --- helpers ---------------------------------------------------------------
pass() { printf 'SMOKE %s PASS\n' "$1"; }

fail() {
    printf 'SMOKE %s FAIL: %s\n' "$1" "$2" >&2
    exit 1
}

note() { printf '  · %s\n' "$1"; }

# GET on the daemon API. Never `-f`: callers want the body even on 4xx.
api() { curl -sS -m 30 -H "Authorization: Bearer $SMOKE_TOKEN" "$API$1"; }

# First value of a flat JSON string field ("key":"value").
json_str() { sed -n 's/.*"'"$1"'":"\([^"]*\)".*/\1/p' | head -1; }

# First value of a flat JSON number field ("key":123 or 123.4).
json_num() { sed -n 's/.*"'"$1"'":\([0-9][0-9.]*\).*/\1/p' | head -1; }

wait_for_health() {
    _budget=$1
    _i=0
    while [ "$_i" -lt "$_budget" ]; do
        if curl -fsk -m 3 "$API/health" >/dev/null 2>&1; then
            return 0
        fi
        _i=$((_i + 1))
        sleep 1
    done
    return 1
}

cleanup() {
    [ "$CLEANED" -eq 1 ] && return 0
    CLEANED=1
    if [ -n "$BUNDLE" ] && [ -n "$SMOKE_HOME" ] && curl -fsk -m 3 "$API/health" >/dev/null 2>&1; then
        HOME="$SMOKE_HOME" "$BUNDLE/nerdit" services rm "$APP" \
            --purge secrets,data,images --force --yes >/dev/null 2>&1 || true
    fi
    if [ -n "$DAEMON_PID" ]; then
        kill "$DAEMON_PID" >/dev/null 2>&1 || true
        _i=0
        while [ "$_i" -lt 20 ] && kill -0 "$DAEMON_PID" >/dev/null 2>&1; do
            _i=$((_i + 1))
            sleep 1
        done
        kill -9 "$DAEMON_PID" >/dev/null 2>&1 || true
    fi
    # Any image the deploy built, plus containers this instance owns. Scoped by
    # the smoke instance label so a real daemon on the same Docker host is
    # untouched (post-P15 instance-scoped ownership).
    if command -v docker >/dev/null 2>&1; then
        _cids=$(docker ps -aq --filter "label=nerdit-instance=$INSTANCE" 2>/dev/null || true)
        # shellcheck disable=SC2086  # deliberate word splitting: a list of ids
        [ -n "$_cids" ] && docker rm -f $_cids >/dev/null 2>&1 || true
        _imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
            | grep "^nerdit-app/$APP:" || true)
        # shellcheck disable=SC2086  # deliberate word splitting: a list of tags
        [ -n "$_imgs" ] && docker rmi -f $_imgs >/dev/null 2>&1 || true
    fi
    [ -n "$SMOKE_HOME" ] && rm -rf "$SMOKE_HOME"
    [ -n "$WORK" ] && rm -rf "$WORK"
    return 0
}

trap cleanup EXIT INT TERM

# --- argument --------------------------------------------------------------
[ $# -eq 1 ] || { printf 'usage: sh %s <path-to-tarball>\n' "$0" >&2; exit 2; }
TARBALL=$1
[ -f "$TARBALL" ] || { printf 'smoke.sh: no such tarball: %s\n' "$TARBALL" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { printf 'smoke.sh: docker is required\n' >&2; exit 2; }
docker info >/dev/null 2>&1 || { printf 'smoke.sh: the docker daemon is unreachable\n' >&2; exit 2; }

printf 'smoke: %s\n' "$TARBALL"

# ===========================================================================
# 1 — extract; the CLI reports the version the tarball claims
# ===========================================================================
WORK=$(mktemp -d "${TMPDIR:-/tmp}/nerdit-smoke.XXXXXX") || fail 1 "mktemp failed"
tar -xzf "$TARBALL" -C "$WORK" || fail 1 "tar extraction failed"

# Exactly one top-level directory, named nerdit-<version>.
_tops=$(ls "$WORK")
_count=$(printf '%s\n' "$_tops" | wc -l | tr -d ' ')
[ "$_count" -eq 1 ] || fail 1 "expected one top-level dir in the tarball, got: $_tops"
BUNDLE="$WORK/$_tops"

for _f in nerdit nerditd VERSION caddy install.sh units/nerdit.service \
          units/nerdit-user.service units/ai.nerdit.daemon.plist; do
    [ -e "$BUNDLE/$_f" ] || fail 1 "tarball is missing $_f"
done
[ -d "$BUNDLE/_internal" ] || fail 1 "tarball is missing _internal/"
[ -x "$BUNDLE/caddy" ] || fail 1 "bundled caddy is not executable"

VERSION=$(cat "$BUNDLE/VERSION") || fail 1 "unreadable VERSION file"
[ -n "$VERSION" ] || fail 1 "empty VERSION file"

CLI_VERSION=$("$BUNDLE/nerdit" --version 2>&1) || fail 1 "nerdit --version exited non-zero: $CLI_VERSION"
case "$CLI_VERSION" in
    *"$VERSION"*) ;;
    *) fail 1 "nerdit --version said '$CLI_VERSION', VERSION file says '$VERSION'" ;;
esac
note "version $VERSION"
pass 1

# ===========================================================================
# 2 — the frozen daemon boots and answers /health
# ===========================================================================
SMOKE_HOME=$(mktemp -d "${TMPDIR:-/tmp}/nerdit-smoke-home.XXXXXX") || fail 2 "mktemp failed"
mkdir -p "$SMOKE_HOME/.nerdit"

# The proxy is opt-in in the product and stays that way on a real install; the
# smoke turns it on explicitly because legs 5/6 serve through it. `caddy_binary`
# points at the BUNDLED binary (D-P30-5) so this leg proves the bundle, not
# whatever caddy happens to sit on PATH. `instance_id` scopes container
# ownership so the smoke's sweeps never touch another daemon's containers.
# `[mcp].http_enabled` mounts the streamable-HTTP transport the cloud's remote
# connector forwards to; it refuses to boot without an auth token (it would be
# an unauthenticated admin surface) and without the bundled mcp extra — both
# are exactly what leg 3b proves. The token is minted here, for this run only.
SMOKE_TOKEN=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
cat > "$SMOKE_HOME/.nerdit/config.toml" <<EOF
[daemon]
port = $PORT
instance_id = "$INSTANCE"
auth_token = "$SMOKE_TOKEN"

[mcp]
http_enabled = true

[proxy]
enabled = true
https_port = $HTTPS_PORT
hostname_override = "$HOSTNAME_OVERRIDE"
caddy_binary = "$BUNDLE/caddy"
EOF

if curl -fsk -m 3 "$API/health" >/dev/null 2>&1; then
    fail 2 "something already listens on $API — pick another NERDIT_SMOKE_PORT"
fi

if [ -d "$SMOKE_DOCKER_CONFIG" ]; then
    note "DOCKER_CONFIG=$SMOKE_DOCKER_CONFIG (buildx plugin resolution)"
else
    SMOKE_DOCKER_CONFIG=
fi

HOME="$SMOKE_HOME" DOCKER_CONFIG="$SMOKE_DOCKER_CONFIG" \
    NERDIT_BOOT_LOG=1 \
    nohup "$BUNDLE/nerditd" > "$SMOKE_HOME/nerditd.out" 2>&1 &
DAEMON_PID=$!

wait_for_health "$BOOT_TIMEOUT" || {
    tail -30 "$SMOKE_HOME/nerditd.out" >&2 || true
    fail 2 "daemon did not answer $API/health within ${BOOT_TIMEOUT}s"
}
HEALTH=$(api /health)
case "$HEALTH" in
    *"\"version\":\"$VERSION\""*) ;;
    *) fail 2 "/health reported an unexpected version: $HEALTH" ;;
esac
note "daemon pid $DAEMON_PID, $HEALTH"
pass 2

# ===========================================================================
# 3 — nerdit doctor runs; Docker is reachable from the frozen daemon
# ===========================================================================
# Exit code is NOT asserted: doctor exits 1 whenever the worst check is `fail`,
# and this smoke host may legitimately fail unrelated checks. (A brand-new data
# dir no longer reports `secrets_key: fail` — the key is created on the first
# secret write, so with no ciphertexts at rest that row is `skipped`.) What must
# hold is that the frozen CLI renders the table and that the frozen daemon
# really reached Docker.
DOCTOR_OUT=$(HOME="$SMOKE_HOME" "$BUNDLE/nerdit" doctor 2>&1) || true
printf '%s\n' "$DOCTOR_OUT"
case "$DOCTOR_OUT" in
    *config_restart_pending*) ;;
    *) fail 3 "nerdit doctor did not print the checks table" ;;
esac
# 3b — the bundled MCP transport answers an `initialize` on /api/mcp. A
# bundle built without the mcp extra never gets here (the daemon refuses to
# boot with http_enabled and no extra); a regression in the mount shows as a
# non-200 or a reply without the protocol handshake.
MCP_BODY='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}'
MCP_OUT=$(curl -sS -m 30 -X POST "$API/api/mcp" \
    -H "Authorization: Bearer $SMOKE_TOKEN" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    -d "$MCP_BODY" -w '\n%{http_code}') || fail 3 "POST /api/mcp did not answer"
MCP_CODE=${MCP_OUT##*$'\n'}
[ "$MCP_CODE" = "200" ] || fail 3 "POST /api/mcp initialize answered $MCP_CODE (mcp extra missing from the bundle?): ${MCP_OUT%$'\n'*}"
case "$MCP_OUT" in
    *'"protocolVersion"'*) ;;
    *) fail 3 "POST /api/mcp initialize returned no protocolVersion: ${MCP_OUT%$'\n'*}" ;;
esac
echo "  mcp: /api/mcp initialize -> 200"

DOCTOR_JSON=$(api /api/doctor)
# `ok`, not merely "not fail": a `warn` here is the buildx-plugin verdict, and
# a daemon that cannot reach BuildKit fails leg 4 anyway — better to say so now.
case "$DOCTOR_JSON" in
    *'"name":"docker","status":"ok"'*) ;;
    *) fail 3 "the docker check is not ok: $DOCTOR_JSON" ;;
esac
case "$DOCTOR_JSON" in
    *'"name":"proxy","status":"ok"'*) ;;
    *) fail 3 "the bundled caddy did not come up: $DOCTOR_JSON" ;;
esac
# `$SMOKE_HOME` is mktemp'd for this run and leg 4 is the first thing that could
# store a secret, so this data dir has no key file and no ciphertext: the row
# must be exactly `skipped`. Asserted positively, not merely "not fail", so it
# catches BOTH regressions — `fail` (what a customer saw as "doctor is red on a
# brand-new install") and `ok` (the key minted as a side effect of the check or
# of boot, which the never-create-while-diagnosing invariant forbids).
case "$DOCTOR_JSON" in
    *'"name":"secrets_key","status":"skipped"'*) ;;
    *) fail 3 "a fresh install must report secrets_key: skipped: $DOCTOR_JSON" ;;
esac
pass 3

# ===========================================================================
# 4 — deploy the node-starter template end to end
# ===========================================================================
command -v git >/dev/null 2>&1 || fail 4 "git is required on the host (the template store clones server-side)"

HOME="$SMOKE_HOME" "$BUNDLE/nerdit" store deploy "$TEMPLATE" --name "$APP" \
    || fail 4 "store deploy $TEMPLATE failed"

# `services wait` is server-clamped to 300 s; a cold `npm install` can exceed
# that, so poll until converged or the budget runs out. Exit 0 converged,
# 1 failed/superseded, 3 timeout.
ELAPSED=0
CONVERGED=0
while [ "$ELAPSED" -lt "$DEPLOY_BUDGET" ]; do
    set +e
    HOME="$SMOKE_HOME" "$BUNDLE/nerdit" services wait "$APP" --timeout 120
    WAIT_CODE=$?
    set -e
    ELAPSED=$((ELAPSED + 120))
    case "$WAIT_CODE" in
        0) CONVERGED=1; break ;;
        3) ;;  # still building — keep waiting
        *) fail 4 "services wait exited $WAIT_CODE (deploy failed or was superseded)" ;;
    esac
done
[ "$CONVERGED" -eq 1 ] || fail 4 "deploy did not converge within ${DEPLOY_BUDGET}s"

SVC=$(api "/api/services/$APP")
STATUS=$(printf '%s' "$SVC" | json_str status)
[ "$STATUS" = running ] || fail 4 "service '$APP' is '$STATUS', expected 'running'"
note "$APP is running"
pass 4

# ===========================================================================
# 5 — the app answers through the bundled Caddy, over the internal-CA TLS
# ===========================================================================
PUBLIC_URL=$(printf '%s' "$SVC" | json_str public_url)
[ -n "$PUBLIC_URL" ] || fail 5 "no public_url on the service — the proxy did not route it"

CA_PEM="$SMOKE_HOME/nerdit-ca.pem"
curl -fsS -m 15 "$API/api/proxy/ca" -o "$CA_PEM" || fail 5 "could not fetch the internal CA from /proxy/ca"
grep -q 'BEGIN CERTIFICATE' "$CA_PEM" || fail 5 "/proxy/ca did not return a PEM certificate"

# Caddy provisions the leaf on first use; give it a few seconds rather than
# racing it. --cacert (never -k): trusting the CA is the point of the leg.
CODE=
_i=0
while [ "$_i" -lt 30 ]; do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 15 --cacert "$CA_PEM" "$PUBLIC_URL" || printf '000')
    [ "$CODE" = 200 ] && break
    _i=$((_i + 1))
    sleep 2
done
[ "$CODE" = 200 ] || fail 5 "$PUBLIC_URL returned HTTP $CODE through the bundled Caddy"
note "$PUBLIC_URL -> 200"
pass 5

# ===========================================================================
# 6 — the daemon survives its own restart, and the app keeps serving
# ===========================================================================
UPTIME_BEFORE=$(api /api/capabilities | json_num uptime_s)
[ -n "$UPTIME_BEFORE" ] || fail 6 "could not read uptime_s from /capabilities"

RESTART_CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST \
    -H "Authorization: Bearer $SMOKE_TOKEN" \
    -H "Idempotency-Key: smoke-restart-$$" "$API/api/daemon/restart")
[ "$RESTART_CODE" = 202 ] || fail 6 "POST /daemon/restart returned HTTP $RESTART_CODE, expected 202"

# The daemon drains, then re-execs itself in place (same PID). Observing the
# down window is best-effort — the drain can be quicker than the poll — so the
# real assertion is the uptime reset below.
_i=0
while [ "$_i" -lt 30 ]; do
    curl -fsk -m 2 "$API/health" >/dev/null 2>&1 || break
    _i=$((_i + 1))
    sleep 1
done
wait_for_health 120 || fail 6 "the daemon did not come back within 120s of the restart"

UPTIME_AFTER=$(api /api/capabilities | json_num uptime_s)
[ -n "$UPTIME_AFTER" ] || fail 6 "could not read uptime_s after the restart"
# Integer compare on the whole-second part; a re-exec resets the clock.
case "$(printf '%s' "$UPTIME_BEFORE" | cut -d. -f1)" in ''|*[!0-9]*) fail 6 "bad uptime_s '$UPTIME_BEFORE'" ;; esac
case "$(printf '%s' "$UPTIME_AFTER" | cut -d. -f1)" in ''|*[!0-9]*) fail 6 "bad uptime_s '$UPTIME_AFTER'" ;; esac
[ "$(printf '%s' "$UPTIME_AFTER" | cut -d. -f1)" -lt "$(printf '%s' "$UPTIME_BEFORE" | cut -d. -f1)" ] \
    || fail 6 "uptime_s did not reset ($UPTIME_BEFORE -> $UPTIME_AFTER): the daemon never restarted"

STATUS=$(api "/api/services/$APP" | json_str status)
[ "$STATUS" = running ] || fail 6 "after the restart '$APP' is '$STATUS', expected 'running'"

CODE=
_i=0
while [ "$_i" -lt 30 ]; do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 15 --cacert "$CA_PEM" "$PUBLIC_URL" || printf '000')
    [ "$CODE" = 200 ] && break
    _i=$((_i + 1))
    sleep 2
done
[ "$CODE" = 200 ] || fail 6 "after the restart $PUBLIC_URL returned HTTP $CODE"
note "uptime_s $UPTIME_BEFORE -> $UPTIME_AFTER; $APP still serving"
pass 6

# ===========================================================================
# 7 — leave the machine as we found it
# ===========================================================================
HOME="$SMOKE_HOME" "$BUNDLE/nerdit" services rm "$APP" \
    --purge secrets,data,images --force --yes >/dev/null 2>&1 \
    || fail 7 "could not remove service '$APP'"
CLEANED=0
cleanup || fail 7 "cleanup failed"
pass 7

printf 'SMOKE OK — %s\n' "$TARBALL"
