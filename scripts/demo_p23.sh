#!/usr/bin/env bash
# demo_p23.sh — P23 CLI + dashboard surface-parity runbook (NOT run in CI).
#
# The scripted form of the live agentic CLI run that every daemon/CLI change
# ends with, for this phase's surface. It boots its OWN sandboxed daemon
# (scratch HOME, port 9333, real Docker, real Caddy) and drives every surface
# P23 added, printing PASS/FAIL/SKIP per leg:
#
#   nerdit capabilities [--json]        # WP1 — daemon self-knowledge, flat verb
#   nerdit proxy status                 # WP1 — proxy sub-typer
#   nerdit routes                       # WP1 — DB-authoritative route inventory
#   nerdit services wait NAME           # WP2 — standalone converge wait (0/1/3)
#   nerdit daemon restart --yes --wait  # WP3 — client + CLI + audit + 409 guard
#   restart_daemon over stdio MCP       # WP3 — the 36th tool, round-tripped
#   readonly token → 403 envelope       # §7 negative contracts
#   instance-scoped image GC            # Track 0.4 (R6b (a)) — WP5 prerequisite
#
# READ BEFORE RUNNING:
# * It is SELF-CONTAINED: it creates a scratch HOME, writes its own
#   ~/.nerdit/config.toml (port 9333, a freshly minted throwaway auth token,
#   [proxy].enabled with the admin API off :2019 so a system Caddy is not
#   disturbed) and starts `nerditd` under nohup. It NEVER touches the
#   operator's real ~/.nerdit, and it refuses to start when something is
#   already listening on the sandbox port.
# * Containers and images are stamped `nerdit-instance=demop23` ([daemon].
#   instance_id), so every sweep/list/kill path here is scoped to this run and
#   a co-located real daemon's workloads are invisible to it.
# * Docker is MANDATORY (real buildpack builds). A GPU is OPTIONAL: with one,
#   leg 2b serves a real model so `nerdit routes` shows the unrouted tri-state
#   for real; without one it deploys a second app instead and says so.
# * Legs 7 and 8 RESTART the daemon three times on purpose. Leg 10 DELETES the
#   demo services and garbage-collects their images.
# * SPA legs (Tokens page, Disk & GC card, Routes card) are not scriptable
#   here — the script ends with an explicit MANUAL checklist for them.
# * There is no automated CI counterpart; the mocked test suites are not a
#   substitute for this — this script is the live proof.
#
# Env knobs: PORT, MODEL_REF, APP_WAIT_S, KEEP_SANDBOX=1 (skip cleanup).

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOSTIP="127.0.0.1"
PORT="${PORT:-9333}"
API="http://${HOSTIP}:${PORT}"
INSTANCE_ID="demop23"
# Outside [services].service_port_range (9400-9499) so a service port can never
# collide with the embedded Caddy's TLS listener.
PROXY_HTTPS_PORT="${PROXY_HTTPS_PORT:-9343}"
# Off Caddy's default :2019 so a system Caddy on this host is left alone.
PROXY_ADMIN_ADDR="${PROXY_ADMIN_ADDR:-127.0.0.1:2119}"

APP_NAME="demo-p23-app"
APP2_NAME="demo-p23-app-b"
BROKEN_NAME="demo-p23-broken"
MODEL_NAME="demo-p23-model"
MODEL_REF="${MODEL_REF:-llama3.2:1b}"
FAKE_IMAGE="nerdit-app/zzz-fake:1"

APP_WAIT_S="${APP_WAIT_S:-300}"
MODEL_WAIT_S="${MODEL_WAIT_S:-600}"
BOOT_WAIT_S="${BOOT_WAIT_S:-90}"
KEEP_SANDBOX="${KEEP_SANDBOX:-0}"

# Wide terminal so Rich never wraps a table cell we later grep for.
export COLUMNS=200

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '   proves: %s\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m %s\n' "$*"; PASS_COUNT=$((PASS_COUNT + 1)); }
bad()  { printf '   \033[31mFAIL\033[0m %s\n' "$*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
skip() { printf '   \033[33mSKIP\033[0m %s\n' "$*"; SKIP_COUNT=$((SKIP_COUNT + 1)); }

# ---------------------------------------------------------------------------
# HTTP + JSON helpers (python3 everywhere — jq is not a house dependency, and
# the existing runbooks all parse with python3; see demo_p11.sh/demo_p14c.sh)
# ---------------------------------------------------------------------------
api_get() {  # api_get <path> — authenticated GET, raw JSON on stdout
  curl -sf -H "Authorization: Bearer ${TOKEN}" "${API}$1"
}

api_code() {  # api_code <path> [curl args...] — HTTP status only
  local path="$1"; shift
  curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${TOKEN}" "$@" "${API}${path}"
}

api_body() {  # api_body <path> [curl args...] — body at any status
  local path="$1"; shift
  curl -s -H "Authorization: Bearer ${TOKEN}" "$@" "${API}${path}"
}

as_code() {  # as_code <token> <path> [curl args...] — status for another principal
  local tok="$1" path="$2"; shift 2
  curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${tok}" "$@" "${API}${path}"
}

as_body() {  # as_body <token> <path> [curl args...] — body for another principal
  local tok="$1" path="$2"; shift 2
  curl -s -H "Authorization: Bearer ${tok}" "$@" "${API}${path}"
}

json_field() {  # json_field <json> <python-expr over parsed `d`>
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2" \
    2>/dev/null || echo ""
}

# Idempotency keys are minted per call at the caller layer, exactly as the CLI
# does (uuid4().hex, the `nerdit gc` precedent). `date +%N` is a GNU-ism, so the
# uniqueness here comes from the label + seconds + $RANDOM, which is portable.
idem() { printf 'demo-p23-%s-%s-%s' "$1" "$(date +%s)" "${RANDOM}"; }

wait_health() {  # wait_health [seconds] — block until /health answers
  local budget="${1:-30}" deadline
  deadline=$(( $(date +%s) + budget ))
  while [ "$(date +%s)" -le "$deadline" ]; do
    curl -sf "${API}/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

uptime_s() {  # uptime_s — the daemon's current uptime, "" when unreachable
  json_field "$(api_get /api/capabilities 2>/dev/null || echo '{}')" "d.get('uptime_s', '')"
}

# ---------------------------------------------------------------------------
# Cleanup (trap-based, idempotent, scoped to this run's instance_id)
# ---------------------------------------------------------------------------
CLEANED=0
DAEMON_PID=""
SCRATCH_HOME=""
RO_TOKEN_ID=""

cleanup() {
  [ "$CLEANED" = "1" ] && return 0
  CLEANED=1
  say "Cleanup"

  if [ "$KEEP_SANDBOX" = "1" ]; then
    echo "   KEEP_SANDBOX=1 — leaving the daemon (pid ${DAEMON_PID:-?}) and ${SCRATCH_HOME} in place."
    echo "   Stop it with: kill ${DAEMON_PID:-<pid>}   (or: pgrep -f bin/nerditd)"
    return 0
  fi

  # Services + scratch token first — they need a live daemon.
  if curl -sf "${API}/health" >/dev/null 2>&1; then
    for name in "$APP_NAME" "$APP2_NAME" "$BROKEN_NAME" "$MODEL_NAME"; do
      nerdit services rm "$name" --purge secrets,data,images --force --yes >/dev/null 2>&1 \
        && echo "   removed service ${name}"
    done
    if [ -n "$RO_TOKEN_ID" ]; then
      if curl -sf -X DELETE -H "Authorization: Bearer ${TOKEN:-}" \
           "${API}/api/tokens/${RO_TOKEN_ID}" >/dev/null 2>&1; then
        echo "   revoked scratch readonly token ${RO_TOKEN_ID}"
      else
        echo "   WARN: could not revoke scratch token ${RO_TOKEN_ID} (the whole DB is discarded below)"
      fi
    fi
  fi

  # Stop the daemon. The recorded PID is authoritative and survives a restart
  # (POST /daemon/restart re-execs in place, so the PID does not change); the
  # pidfile is deliberately never consulted by a live run. `pgrep -f
  # bin/nerditd` is NOT used to kill: on a host that also runs the operator's
  # real daemon it would match that one too.
  if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
    kill "$DAEMON_PID" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "$DAEMON_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$DAEMON_PID" 2>/dev/null
    echo "   sandbox daemon (pid ${DAEMON_PID}) stopped"
  fi
  if curl -sf "${API}/health" >/dev/null 2>&1; then
    echo "   WARN: something still answers on ${API}. Inspect: pgrep -f bin/nerditd"
  fi

  # Images + containers, filtered on THIS run's instance label so a co-located
  # daemon's `nerdit-app/*` images are untouchable from here.
  if command -v docker >/dev/null 2>&1; then
    local cids imgs
    cids="$(docker ps -aq --filter "label=nerdit-instance=${INSTANCE_ID}" 2>/dev/null)"
    [ -n "$cids" ] && docker rm -f $cids >/dev/null 2>&1 && echo "   removed this run's containers"
    imgs="$(docker images --filter "label=nerdit-instance=${INSTANCE_ID}" \
      --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep '^nerdit-app/' || true)"
    if [ -n "$imgs" ]; then
      # shellcheck disable=SC2086
      docker rmi -f $imgs >/dev/null 2>&1 && echo "   removed this run's nerdit-app/* images"
    fi
    docker rmi "$FAKE_IMAGE" >/dev/null 2>&1 && echo "   untagged ${FAKE_IMAGE}"
  fi

  if [ -n "$SCRATCH_HOME" ] && [ -d "$SCRATCH_HOME" ]; then
    case "$SCRATCH_HOME" in
      /tmp/*|/private/tmp/*|/var/folders/*)
        rm -rf "$SCRATCH_HOME" && echo "   removed scratch HOME ${SCRATCH_HOME}" ;;
      *)
        echo "   WARN: refusing to rm -rf a non-temp scratch HOME (${SCRATCH_HOME}) — delete it yourself" ;;
    esac
  fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 0. Preflight (uncounted — a failure here is an operator problem, not a result)
# ---------------------------------------------------------------------------
say "0. Preflight"
command -v python3 >/dev/null || { echo "python3 required."; exit 1; }
command -v curl    >/dev/null || { echo "curl required."; exit 1; }
command -v docker  >/dev/null || { echo "docker required (real buildpack builds)."; exit 1; }
NERDIT_BIN="$(command -v nerdit)"  || { echo "nerdit CLI not on PATH (pip install -e .)."; exit 1; }
NERDITD_BIN="$(command -v nerditd)" || { echo "nerditd not on PATH (pip install -e .)."; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not reachable."; exit 1; }
if curl -sf "${API}/health" >/dev/null 2>&1; then
  echo "REFUSING: something already answers on ${API}. This runbook owns its own daemon —"
  echo "stop that process (or set PORT=<free port>) and re-run."
  exit 1
fi
echo "nerdit:  ${NERDIT_BIN}"
echo "nerditd: ${NERDITD_BIN}"

# ---------------------------------------------------------------------------
# 1. Sandbox boot
# ---------------------------------------------------------------------------
say "1. Sandbox boot — scratch HOME, port ${PORT}, proxy on"
SCRATCH_HOME="$(mktemp -d "${TMPDIR:-/tmp}/nerdit-p23.XXXXXX")"
mkdir -p "${SCRATCH_HOME}/.nerdit"
# A throwaway credential minted per run, living only inside the scratch HOME
# that cleanup deletes: a demo mints its own scratch credential, never reuses a
# real one, and never commits one.
AUTH_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(24))')"
cat > "${SCRATCH_HOME}/.nerdit/config.toml" <<EOF
# Generated by scripts/demo_p23.sh — scratch sandbox, discarded on exit.
[daemon]
host = "${HOSTIP}"
port = ${PORT}
auth_token = "${AUTH_TOKEN}"
instance_id = "${INSTANCE_ID}"

[proxy]
enabled = true
admin_addr = "${PROXY_ADMIN_ADDR}"
https_port = ${PROXY_HTTPS_PORT}
EOF
chmod 600 "${SCRATCH_HOME}/.nerdit/config.toml"

export HOME="$SCRATCH_HOME"
DATA_DIR="${SCRATCH_HOME}/.nerdit"
DAEMON_LOG="${SCRATCH_HOME}/nerditd-nohup.log"
WORKDIR="${SCRATCH_HOME}/work"
mkdir -p "$WORKDIR"

nohup "$NERDITD_BIN" >"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
if wait_health "$BOOT_WAIT_S"; then
  ok "sandbox daemon up at ${API} (pid ${DAEMON_PID}, HOME=${SCRATCH_HOME})"
else
  bad "daemon did not answer /health within ${BOOT_WAIT_S}s — log tail:"
  tail -30 "$DAEMON_LOG" | sed 's/^/     /'
  say "Summary"
  echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
  echo "0/1 PASS"
  echo "P23 surface runbook: FAIL (daemon never booted)"
  exit 1
fi
TOKEN="$(nerdit token)"
# Every CLI verb below runs WITHOUT `nerdit connect`, against a daemon on a
# NON-default port. A successful client round-trip here is the WP0 fix in
# anger: before it, the local fallback dialled the constant 9321 and every one
# of the five new read verbs would have targeted the wrong daemon.
nerdit services list >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ] && [ "$TOKEN" = "$AUTH_TOKEN" ]; then
  ok "the CLI reached :${PORT} with no 'nerdit connect' (WP0: the local default follows [daemon].port)"
else
  bad "CLI round-trip failed (exit ${rc}) or the token did not match — later legs would be meaningless"
fi
note "a disposable control plane, isolated by HOME and by instance_id=${INSTANCE_ID}"

# ---------------------------------------------------------------------------
# 2. Workloads: one app, plus a model (GPU) or a second app (no GPU)
# ---------------------------------------------------------------------------
say "2. Deploy a self-contained app"
mk_app() {  # mk_app <dir> <name> [start-override]
  local dir="$1" name="$2" start="${3:-}"
  mkdir -p "$dir"
  cat > "${dir}/package.json" <<EOF
{ "name": "${name}", "version": "1.0.0", "private": true,
  "scripts": { "start": "node server.js" } }
EOF
  cat > "${dir}/server.js" <<EOF
const http = require("http");
const port = process.env.PORT || 3000;
http.createServer((req, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("hello from ${name}\\n");
}).listen(port, () => console.log("${name} listening on " + port));
EOF
  {
    echo "[deploy]"
    echo "name = \"${name}\""
    echo "port = 3000"
    [ -n "$start" ] && echo "start = \"${start}\""
  } > "${dir}/nerdit.toml"
}

mk_app "${WORKDIR}/app" "$APP_NAME"
nerdit deploy "${WORKDIR}/app" --wait --timeout "$APP_WAIT_S"
rc=$?
if [ "$rc" -eq 0 ]; then
  ok "${APP_NAME} deployed + converged (exit 0)"
else
  bad "nerdit deploy --wait exited ${rc} for ${APP_NAME} — later legs will be thin"
fi

say "2b. A second routable-vs-unrouted row"
GPU_COUNT="$(json_field "$(api_get /api/gpus 2>/dev/null || echo '[]')" \
  "sum(1 for g in d if g.get('schedulable'))")"
GPU_COUNT="${GPU_COUNT:-0}"
MODEL_SERVED=0
if [ "$GPU_COUNT" -ge 1 ]; then
  echo "   ${GPU_COUNT} schedulable GPU(s) — serving ${MODEL_REF} (a kind=model row is the unrouted case)"
  nerdit serve "$MODEL_REF" --gpus 1 --name "$MODEL_NAME" >/dev/null 2>&1
  nerdit services wait "$MODEL_NAME" --timeout 300 >/dev/null 2>&1
  # The weights pull can outlive one clamped wait; poll the model list too.
  deadline=$(( $(date +%s) + MODEL_WAIT_S ))
  while [ "$(date +%s)" -le "$deadline" ]; do
    state="$(json_field "$(api_get /api/models 2>/dev/null || echo '{"items":[]}')" \
      "next(((i['status'], bool(i.get('model_pulled'))) for i in d['items'] if i['name'] == '${MODEL_NAME}'), ('absent', False))")"
    case "$state" in
      "('running', True)") MODEL_SERVED=1; break ;;
      "('failed',"*) break ;;
    esac
    sleep 5
  done
  if [ "$MODEL_SERVED" = "1" ]; then
    ok "model ${MODEL_NAME} running — the routes table gets a real unrouted (route=null) row"
  else
    skip "model ${MODEL_NAME} did not reach running+pulled within ${MODEL_WAIT_S}s (state ${state:-?}) — routes tri-state is thinner"
  fi
else
  mk_app "${WORKDIR}/app-b" "$APP2_NAME"
  nerdit deploy "${WORKDIR}/app-b" --wait --timeout "$APP_WAIT_S"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    ok "no GPU — deployed ${APP2_NAME} instead, so the routes table still has >= 2 rows"
  else
    bad "second app deploy exited ${rc}"
  fi
  echo "   NOTE (limitation): with no GPU there is no kind=model row, so the"
  echo "   route=null 'unrouted' leg of the tri-state is NOT exercised here."
  echo "   Re-run this runbook on the GPU box to close it (plan §6 gate 6)."
fi
note "at least two endpoint rows exist, so the P23 read verbs have something to render"

# ---------------------------------------------------------------------------
# 3. nerdit capabilities (WP1)
# ---------------------------------------------------------------------------
say "3. nerdit capabilities"
caps_out="$(nerdit capabilities 2>&1)"; rc=$?
printf '%s\n' "$caps_out" | sed 's/^/     /'
if [ "$rc" -eq 0 ] && printf '%s' "$caps_out" | grep -q "buildpacks"; then
  ok "nerdit capabilities rendered the facts/proxy/capability tables (exit 0)"
else
  bad "nerdit capabilities exited ${rc} / did not render a capability table"
fi

caps_json="$(nerdit capabilities --json 2>/dev/null)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$caps_json" | python3 -m json.tool >/dev/null 2>&1; then
  role="$(json_field "$caps_json" "(d.get('caller') or {}).get('role')")"
  ok "nerdit capabilities --json parses (caller.role=${role:-?}, machine escape hatch intact)"
else
  bad "nerdit capabilities --json did not emit parseable JSON (exit ${rc})"
fi
note "the read an agent used to need MCP for is now a first-class CLI verb"

# ---------------------------------------------------------------------------
# 4. nerdit proxy status (WP1)
# ---------------------------------------------------------------------------
say "4. nerdit proxy status"
proxy_out="$(nerdit proxy status 2>&1)"; rc=$?
printf '%s\n' "$proxy_out" | sed 's/^/     /'
if [ "$rc" -ne 0 ]; then
  bad "nerdit proxy status exited ${rc}"
elif printf '%s' "$proxy_out" | grep -qi "state"; then
  ok "nerdit proxy status rendered the proxy state ([proxy].enabled = true in this sandbox)"
elif printf '%s' "$proxy_out" | grep -qi "Proxy off"; then
  bad "the proxy reported disabled although the sandbox config enables it"
else
  bad "nerdit proxy status printed neither a state row nor the disabled sentence"
fi
PROXY_AVAILABLE="$(json_field "$(api_get /api/proxy/status 2>/dev/null || echo '{}')" \
  "str(d.get('available'))")"
[ "$PROXY_AVAILABLE" = "True" ] || \
  echo "   NOTE: proxy available=${PROXY_AVAILABLE:-?} (no caddy binary, or a bind failure) —"
[ "$PROXY_AVAILABLE" = "True" ] || \
  echo "   routes stay DB-authoritative but 'live' will read as unknown/missing."
note "the operator's first question when a URL misbehaves, answerable without curl"

# ---------------------------------------------------------------------------
# 5. nerdit routes (WP1)
# ---------------------------------------------------------------------------
say "5. nerdit routes"
routes_out="$(nerdit routes 2>&1)"; rc=$?
printf '%s\n' "$routes_out" | sed 's/^/     /'
if [ "$rc" -eq 0 ] && printf '%s' "$routes_out" | grep -q "$APP_NAME"; then
  ok "nerdit routes listed ${APP_NAME} (exit 0)"
else
  bad "nerdit routes exited ${rc} / did not list ${APP_NAME}"
fi
if printf '%s' "$routes_out" | grep -qi "live table"; then
  ok "the page-level live_table tri-state footer is printed (so a '-' cell is explained)"
else
  bad "no 'live table:' footer — the live tri-state would be unexplained"
fi
if [ "$MODEL_SERVED" = "1" ]; then
  if printf '%s' "$routes_out" | grep -q "$MODEL_NAME" \
     && printf '%s' "$routes_out" | grep -qi "unrouted"; then
    ok "the model row renders as 'unrouted' (route=null — Invariant #2, models are never routed)"
  else
    bad "the model row is missing or not rendered as unrouted"
  fi
else
  skip "no served model — the route=null 'unrouted' rendering is not exercised (see leg 2b)"
fi
note "one DB-authoritative inventory, models included, whether or not Caddy is up"

# ---------------------------------------------------------------------------
# 6. nerdit services wait (WP2)
# ---------------------------------------------------------------------------
say "6. nerdit services wait — the converge signal, detached from the deploy"
nerdit services wait "$APP_NAME" --timeout 60 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
  ok "wait on the converged ${APP_NAME} exited 0"
else
  bad "wait on a healthy service exited ${rc} (expected 0)"
fi

echo "   deploying a deliberately broken image (${BROKEN_NAME}: a start command that cannot run)…"
mk_app "${WORKDIR}/broken" "$BROKEN_NAME" "node does-not-exist.js"
nerdit deploy "${WORKDIR}/broken" --wait --timeout "$APP_WAIT_S" >/dev/null 2>&1
nerdit services wait "$BROKEN_NAME" --timeout 60 >/dev/null 2>&1
rc=$?
if [ "$rc" -eq 1 ]; then
  ok "wait on the failed ${BROKEN_NAME} exited 1 (0 converged · 1 failed|superseded · 3 timeout)"
elif [ "$rc" -eq 3 ]; then
  bad "wait on the broken app exited 3 (timeout) — it never settled terminal within 60s"
else
  bad "wait on the broken app exited ${rc} (expected 1)"
fi
note "a script can block on any service, not only the one it just deployed"

# ---------------------------------------------------------------------------
# 7. nerdit daemon restart (WP3) + the 409 drain guard
# ---------------------------------------------------------------------------
say "7. nerdit daemon restart --yes --wait"
UPTIME_BEFORE="$(uptime_s)"
echo "   uptime before: ${UPTIME_BEFORE:-?}s"
nerdit daemon restart --yes --wait --wait-timeout 90
rc=$?
if [ "$rc" -eq 0 ]; then
  ok "nerdit daemon restart --yes --wait exited 0 (a freshly booted daemon answered)"
else
  bad "nerdit daemon restart --yes --wait exited ${rc}"
fi
wait_health 60 || true
UPTIME_AFTER="$(uptime_s)"
if [ -n "$UPTIME_BEFORE" ] && [ -n "$UPTIME_AFTER" ] \
   && [ "$UPTIME_AFTER" -lt "$UPTIME_BEFORE" ] 2>/dev/null; then
  ok "uptime reset (${UPTIME_BEFORE}s → ${UPTIME_AFTER}s) — the process really re-execed"
else
  bad "uptime did not reset (${UPTIME_BEFORE:-?}s → ${UPTIME_AFTER:-?}s)"
fi
if kill -0 "$DAEMON_PID" 2>/dev/null; then
  ok "same pid ${DAEMON_PID} after the restart (execv in place, the pidfile stays valid)"
else
  bad "pid ${DAEMON_PID} is gone — the daemon did not re-exec in place"
fi
audit_hit="$(json_field "$(api_get '/api/audit?action=daemon.restart&limit=5' 2>/dev/null || echo '{"items":[]}')" \
  "any(e.get('action') == 'daemon.restart' for e in d.get('items', []))")"
if [ "$audit_hit" = "True" ]; then
  ok "audit carries a daemon.restart row (Invariant #3: the write is on the record)"
else
  bad "no daemon.restart entry in GET /api/audit"
fi

say "7b. A second restart DURING the drain → 409 daemon.restart_in_progress"
# The drain returns immediately when nothing is in flight, so the 409 window is
# sub-millisecond by default. Widen it honestly: a one-off `sleep` run counts
# in busy_runs(), so the drain parks on it until the deadline kill (P20).
RACE_OK=0
run_body='{"command":["sleep","20"],"timeout_s":25}'
curl -s -o /dev/null -X POST -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" -H "Idempotency-Key: $(idem run)" \
  -d "$run_body" "${API}/api/services/${APP_NAME}/run" &
RUN_CURL_PID=$!
sleep 4
code1="$(api_code /api/daemon/restart -X POST -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(idem r1)" -d '{"drain_timeout_s": 8}')"
race_body="$(api_body /api/daemon/restart -X POST -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(idem r2)" -d '{"drain_timeout_s": 8}')"
race_code="$(json_field "$race_body" "d.get('code')")"
if [ "$code1" = "202" ] && [ "$race_code" = "daemon.restart_in_progress" ]; then
  RACE_OK=1
  ok "first POST 202, second answered the structured 409 daemon.restart_in_progress"
elif [ "$code1" != "202" ]; then
  skip "the first restart POST returned HTTP ${code1} (not 202) — the race was not set up; body: $(printf '%s' "$race_body" | head -c 160)"
else
  skip "the drain window closed before the second POST (code=${race_code:-none}) — best-effort race, see plan §2.3 R1/R2"
fi
wait "$RUN_CURL_PID" 2>/dev/null
if wait_health 90; then
  [ "$RACE_OK" = "1" ] && ok "daemon came back after the drained restart" \
    || echo "   daemon back up."
else
  bad "the daemon did not come back after leg 7b — remaining legs will fail"
fi
note "restart is admin-gated, idempotency-mandatory, audited, and 409-guarded"

# ---------------------------------------------------------------------------
# 8. MCP stdio round-trip of restart_daemon (WP3, D-P23-1 (b))
# ---------------------------------------------------------------------------
say "8. restart_daemon over stdio MCP (the 36th tool)"
if ! python3 -c 'import mcp' >/dev/null 2>&1; then
  skip "the 'mcp' extra is not importable — run MANUALLY: pip install 'nerdit[mcp]' then re-run this leg"
else
  # Leg 7b just restarted the daemon, so its truncated uptime_s may still be
  # 0 — and "uptime_after < uptime_before" can never hold against a 0 baseline
  # (the same truncation trap the CLI's boot-time predicate exists for). Wait
  # until the reset is observable before restarting again.
  for _ in $(seq 1 20); do
    u="$(uptime_s)"
    [ -n "$u" ] && [ "$u" -ge 3 ] 2>/dev/null && break
    sleep 1
  done
  UPTIME_BEFORE="$(uptime_s)"
  NERDIT_BIN="$NERDIT_BIN" python3 - <<'PYEOF'
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(
    command=os.environ["NERDIT_BIN"],
    args=["mcp"],
    # The scratch HOME is what points the inner REST hop at :9333.
    env=dict(os.environ),
)


async def main() -> int:
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            print(f"  tools/list returned {len(names)} tools", flush=True)
            if "restart_daemon" not in names:
                print("  restart_daemon is NOT registered", flush=True)
                return 3
            try:
                result = await session.call_tool("restart_daemon", {"drain_timeout_s": 5})
            except Exception as exc:  # the daemon may die before the reply lands
                print(f"  call dropped ({type(exc).__name__}) — probably restarting", flush=True)
                return 4
            if getattr(result, "isError", False):
                text = "\n".join(
                    getattr(b, "text", "") or "" for b in (getattr(result, "content", []) or [])
                )
                print(f"  restart_daemon returned an error: {text[:200]}", flush=True)
                return 5
            print("  restart_daemon accepted (202 body returned through the tool)", flush=True)
            return 0


sys.exit(asyncio.run(main()))
PYEOF
  rc=$?
  case "$rc" in
    0) mcp_note="clean 202 through the tool" ;;
    4) mcp_note="response dropped mid-restart (the documented R1 shape — confirm by uptime)" ;;
    *) mcp_note="" ;;
  esac
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 4 ]; then
    # /health can race the re-exec (the OLD process answers, then the socket
    # dies while the new one boots), and every later leg needs a settled
    # daemon. So the accept signal is the CLI's own: poll until an ANSWERING
    # process reports an uptime strictly below the pre-restart baseline —
    # empty reads (mid-boot dead socket) just keep polling.
    UPTIME_AFTER=""
    for _ in $(seq 1 90); do
      u="$(uptime_s)"
      if [ -n "$u" ] && [ "$u" -lt "$UPTIME_BEFORE" ] 2>/dev/null; then
        UPTIME_AFTER="$u"
        break
      fi
      sleep 1
    done
    if [ -n "$UPTIME_AFTER" ]; then
      ok "stdio MCP restart_daemon round-tripped: ${mcp_note}; uptime ${UPTIME_BEFORE}s → ${UPTIME_AFTER}s"
    else
      bad "no freshly booted daemon answered within 90s of the MCP restart (baseline ${UPTIME_BEFORE:-?}s)"
    fi
  else
    bad "the stdio MCP session failed (driver exit ${rc}) — see the output above"
    wait_health 90 || true
  fi
fi
note "over stdio the MCP process outlives the daemon it restarts (R1 applies to HTTP only)"

# ---------------------------------------------------------------------------
# 9. Negative contracts: the live run and the credential gate
# ---------------------------------------------------------------------------
say "9. Negative contracts — a readonly token, and where the plaintext may live"
TOKEN="$(nerdit token)"
ro_json="$(api_body /api/tokens -X POST -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(idem ro)" \
  -d '{"name": "demo-p23-readonly", "role": "readonly"}')"
RO_TOKEN="$(json_field "$ro_json" "d.get('token', '')")"
RO_TOKEN_ID="$(json_field "$ro_json" "d.get('id', '')")"
if [ -z "$RO_TOKEN" ]; then
  bad "could not mint a readonly token: $(printf '%s' "$ro_json" | head -c 200)"
else
  ok "minted a scratch readonly token ${RO_TOKEN_ID} (plaintext returned exactly once)"

  ro_disk_code="$(as_code "$RO_TOKEN" /api/system/disk)"
  if [ "$ro_disk_code" = "200" ]; then
    ok "readonly GET /api/system/disk → 200 (the disk report is doctor-posture: any authenticated)"
  else
    bad "readonly GET /api/system/disk → HTTP ${ro_disk_code} (expected 200)"
  fi

  ro_gc_body="$(as_body "$RO_TOKEN" /api/system/gc -X POST -H "Content-Type: application/json" \
    -H "Idempotency-Key: $(idem rogc)" -d '{}')"
  ro_gc_code_http="$(as_code "$RO_TOKEN" /api/system/gc -X POST -H "Content-Type: application/json" \
    -H "Idempotency-Key: $(idem rogc2)" -d '{}')"
  ro_gc_code="$(json_field "$ro_gc_body" "d.get('code')")"
  if [ "$ro_gc_code_http" = "403" ] && [ "$ro_gc_code" = "forbidden" ]; then
    ok "readonly POST /api/system/gc → 403 with the structured envelope code=forbidden"
  else
    bad "readonly GC gave HTTP ${ro_gc_code_http} / code=${ro_gc_code:-none} (expected 403 / forbidden)"
  fi

  # The create response is the ONLY place the plaintext may appear. The daemon
  # log is the cheap, load-bearing place to prove no secret value reaches it.
  leaked=0
  for f in "$DAEMON_LOG" "${DATA_DIR}/nerditd.log"; do
    [ -f "$f" ] || continue
    if grep -qF -- "$RO_TOKEN" "$f"; then
      leaked=1
      echo "     LEAK in ${f}"
    fi
  done
  # And it must not come back out of the audit trail either.
  audit_dump="$(api_get '/api/audit?action=token.create&limit=20' 2>/dev/null || echo '{}')"
  if printf '%s' "$audit_dump" | grep -qF -- "$RO_TOKEN"; then
    leaked=1
    echo "     LEAK in the audit trail"
  fi
  if [ "$leaked" = "0" ]; then
    ok "the minted plaintext appears in the create response only — absent from the daemon log and audit"
  else
    bad "the minted token plaintext leaked (see above) — this is a merge blocker"
  fi
fi
note "role redaction and once-only plaintext hold across the new surfaces"

# ---------------------------------------------------------------------------
# 10. Instance-scoped image GC (Track 0.4, the WP5 prerequisite — R6b (a))
# ---------------------------------------------------------------------------
say "10. Image GC is instance-scoped (a co-located daemon's images are safe)"
APP_IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
  | grep "^nerdit-app/${APP_NAME}:" | head -1)"
if [ -z "$APP_IMAGE" ]; then
  skip "no nerdit-app/${APP_NAME} image found — the deploy in leg 2 did not build one"
else
  label="$(docker inspect --format '{{index .Config.Labels "nerdit-instance"}}' "$APP_IMAGE" 2>/dev/null)"
  if [ "$label" = "$INSTANCE_ID" ]; then
    ok "${APP_IMAGE} carries nerdit-instance=${INSTANCE_ID} (stamped at build, passthrough Dockerfiles included)"
  else
    bad "${APP_IMAGE} has nerdit-instance='${label:-<none>}' (expected ${INSTANCE_ID})"
  fi

  # Orphan it: drop the row, keep the image.
  nerdit services rm "$APP_NAME" --purge secrets --yes >/dev/null 2>&1
  sleep 2
  gc_plan="$(nerdit gc --dry-run 2>&1)"; rc=$?
  printf '%s\n' "$gc_plan" | sed 's/^/     /'
  if [ "$rc" -eq 0 ] && printf '%s' "$gc_plan" | grep -q "nerdit-app/${APP_NAME}"; then
    ok "nerdit gc --dry-run lists the now-orphan OWN-instance repo nerdit-app/${APP_NAME}"
  else
    bad "nerdit gc --dry-run (exit ${rc}) did not list nerdit-app/${APP_NAME} as reclaimable"
  fi

  # A same-namespace image this daemon did not build must be invisible to GC.
  # `node:20-slim` is present (the Node buildpack base) and carries no
  # nerdit-instance label, so tagging it produces an honest foreign/unlabelled
  # repo without pulling anything.
  if docker tag node:20-slim "$FAKE_IMAGE" >/dev/null 2>&1; then
    fake_label="$(docker inspect --format '{{index .Config.Labels "nerdit-instance"}}' "$FAKE_IMAGE" 2>/dev/null)"
    if [ -n "$fake_label" ]; then
      skip "the tagged base image unexpectedly carries nerdit-instance=${fake_label} — cannot build an unlabelled fake"
    else
      gc_plan2="$(nerdit gc --dry-run 2>&1)"
      if printf '%s' "$gc_plan2" | grep -q "zzz-fake"; then
        bad "nerdit gc --dry-run listed the UNLABELLED ${FAKE_IMAGE} — the ownership filter is not holding"
      else
        ok "the unlabelled ${FAKE_IMAGE} is NOT listed — 'no evidence of ownership' never means 'mine'"
      fi
    fi
    docker rmi "$FAKE_IMAGE" >/dev/null 2>&1
  else
    skip "could not tag ${FAKE_IMAGE} (node:20-slim absent?) — the foreign-image leg is unexercised"
  fi
fi
note "the PR #81 multi-daemon regression class stays closed with the GC card one click away"

# ---------------------------------------------------------------------------
# Summary + the manual SPA checklist
# ---------------------------------------------------------------------------
say "Summary"
TOTAL=$(( PASS_COUNT + FAIL_COUNT + SKIP_COUNT ))
echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
echo "${PASS_COUNT}/${TOTAL} PASS"

cat <<'MANUAL'

=== MANUAL — SPA legs (not scriptable here; plan §6 gate 6, dashboard bullet) ===
Open the dashboard (https://<host>/ with [proxy].dashboard_apex, or
http://127.0.0.1:<port>/ directly) against a daemon carrying this state, and
check each by hand:

  [ ] Tokens page (/tokens, system nav group + reachable from the Cmd-K palette)
      - the table lists tokens; the "include revoked" toggle works
      - create → the plaintext modal appears EXACTLY once, CopyButton copies it
      - close the modal → the plaintext is gone from the page and is NOT
        recoverable (DevTools: no localStorage/sessionStorage entry, not in the
        URL, and the create mutation is absent from the react-query
        MutationCache — D-P23-5 needs BOTH gcTime: 0 and reset())
      - revoke goes through the ConfirmDialog, then the list refreshes
      - with a non-admin token: the friendly empty state, not an error loop
  [ ] Settings → Disk & GC card
      - the disk report is populated; a null docker aggregate reads
        "docker unavailable", never zeros
      - GC button shows the DRY-RUN plan first; no non-dry-run POST /system/gc
        fires until the ConfirmDialog is confirmed (watch the Network tab)
      - "include orphan data" carries the IRREVERSIBLE wording verbatim
  [ ] Settings → Routes card
      - a routed service, and (GPU box) a model row rendered "unrouted"
      - the live_table caption explains any "-" cell
      - ProxyStatusCard no longer shows a bare "Routes" count row (D-P23-7)
MANUAL

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo ""
  echo "P23 surface runbook: FAIL"
  exit 1
fi
echo ""
echo "P23 surface runbook: PASS"
