#!/usr/bin/env bash
# demo_p6.sh — P6 surface runbook (NOT run in CI).
#
# Reproducible, copy-paste runnable from the repo root against a live daemon.
# It proves the P6 surface end-to-end and prints PASS/FAIL per step:
#
#   nerdit deploy <app>                     # folder → build → run → URL
#   GET /api/services                       # rollback_available/build_version
#   GET /api/cluster/stats                  # services_up > 0
#   GET /api/audit                          # the mutations are on the record
#   GET /api/services/{name}/logs?tail=N    # bounded reads (agent-grade)
#   GET /api/audit with a readonly token    # admin-only (403)
#
# NOTES (read before running):
# * The deployed app is a self-contained, dependency-free Node HTTP server
#   generated into a temp dir — no example folder or GPU required. Docker and
#   the Node buildpack path are exercised for real.
# * The readonly-403 step needs auth enabled ([security] / a global token). In
#   v0.1 no-auth local mode every request is the LOCAL admin principal, so the
#   step is SKIPPED (not failed) with a note.
# * The automated CI counterpart is tests/test_p6_surface.py.

set -euo pipefail

HOST="${NERDIT_HOST:-127.0.0.1}"
PORT="${NERDIT_PORT:-9321}"
API="http://${HOST}:${PORT}"
APP_NAME="demo-p6-app"
APP_WAIT_S="${APP_WAIT_S:-300}"
KEEP_APP="${KEEP_APP:-0}"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '   proves: %s\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m %s\n' "$*"; PASS_COUNT=$((PASS_COUNT + 1)); }
bad()  { printf '   \033[31mFAIL\033[0m %s\n' "$*"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
skip() { printf '   \033[33mSKIP\033[0m %s\n' "$*"; SKIP_COUNT=$((SKIP_COUNT + 1)); }

api_get() {  # api_get <path> — authenticated GET, raw JSON on stdout
  curl -sf -H "Authorization: Bearer ${TOKEN}" "${API}$1"
}

api_code() {  # api_code <path> [extra curl args...] — HTTP status only
  local path="$1"; shift
  curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${TOKEN}" "$@" "${API}${path}"
}

json_field() {  # json_field <json> <python-expr over parsed `d`>
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2"
}

# --- 0. preflight ------------------------------------------------------------
say "0. Preflight"
command -v nerdit >/dev/null || { echo "nerdit CLI not on PATH (pip install -e .)."; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for JSON parsing."; exit 1; }
curl -sf "${API}/health" >/dev/null || { echo "Daemon not reachable at ${API} — start nerditd."; exit 1; }
TOKEN="$(nerdit token)"
# Auth mode probe: an unauthenticated protected read is 401 when auth is on.
UNAUTH_CODE="$(curl -s -o /dev/null -w '%{http_code}' "${API}/api/cluster/stats")"
if [ "$UNAUTH_CODE" = "401" ]; then AUTH_ENABLED=1; else AUTH_ENABLED=0; fi
echo "Daemon up at ${API} (auth enabled: ${AUTH_ENABLED})."
note "the control plane is reachable and authenticated"

# --- 1. generate + deploy a self-contained app --------------------------------
say "1. nerdit deploy (self-contained Node app)"
APP_DIR="$(mktemp -d)"
trap 'rm -rf "$APP_DIR"' EXIT
cat > "${APP_DIR}/package.json" <<'EOF'
{
  "name": "demo-p6-app",
  "version": "1.0.0",
  "private": true,
  "scripts": { "start": "node server.js" }
}
EOF
cat > "${APP_DIR}/server.js" <<'EOF'
const http = require("http");
const port = process.env.PORT || 3000;
http
  .createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("hello from demo-p6-app\n");
  })
  .listen(port, () => console.log(`demo-p6-app listening on ${port}`));
EOF
cat > "${APP_DIR}/nerdit.toml" <<EOF
[deploy]
name = "${APP_NAME}"
port = 3000
EOF

T0=$(date +%s)
if nerdit deploy "$APP_DIR"; then
  ok "deploy accepted (folder → ZIP → server-side Node buildpack build)"
else
  bad "nerdit deploy failed"
  exit 1
fi
note "one command: build + run + restart policy (+ URL when [proxy].enabled)"

say "2. Wait for the app: status=running"
deadline=$(( $(date +%s) + APP_WAIT_S ))
while :; do
  svc_json="$(api_get "/api/services/${APP_NAME}")"
  status="$(json_field "$svc_json" "d['status']")"
  echo "  app status: ${status}"
  [ "$status" = "running" ] && break
  [ "$status" = "failed" ] && { bad "app failed — check: nerdit services logs ${APP_NAME}"; exit 1; }
  [ "$(date +%s)" -lt "$deadline" ] || { bad "timed out waiting for the app"; exit 1; }
  sleep 3
done
T1=$(date +%s)
ok "app running in $(( T1 - T0 ))s"

# --- 3. GET /services surfaces the app + the P6 response fields ---------------
say "3. GET /api/services — listed with rollback_available/build_version"
list_json="$(api_get "/api/services?limit=200")"
found="$(json_field "$list_json" "any(i['name'] == '${APP_NAME}' for i in d['items'])")"
if [ "$found" = "True" ]; then
  ok "the app appears in the paginated services list"
else
  bad "the app is missing from GET /api/services"
fi
# A re-run redeploys onto the existing name (version bumps, rollback becomes
# available), so assert the D3 fields are present and typed, and print values.
fields="$(json_field "$svc_json" "(d.get('rollback_available'), d.get('build_version'))")"
fields_ok="$(json_field "$svc_json" \
  "isinstance(d.get('rollback_available'), bool) and isinstance(d.get('build_version'), int)")"
if [ "$fields_ok" = "True" ]; then
  ok "ServiceResponse carries (rollback_available, build_version)=${fields} (D3 fields)"
else
  bad "rollback_available/build_version missing or mistyped: ${fields}"
fi
note "agents/dashboard read rollback availability from the schema, never the config blob"

# --- 4. services_up in /cluster/stats ------------------------------------------
say "4. GET /api/cluster/stats — services_up > 0"
services_up="$(json_field "$(api_get /api/cluster/stats)" "d['services_up']")"
if [ "$services_up" -ge 1 ]; then
  ok "services_up=${services_up}"
else
  bad "services_up=${services_up} (expected >= 1 with the app running)"
fi
note "the Overview leads with services, not GPUs — the PaaS framing is live data"

# --- 5. the deploy mutation is on the audit record ------------------------------
say "5. GET /api/audit — the deploy is audited"
audit_json="$(api_get "/api/audit?action=deploy.create&limit=50")"
audited="$(json_field "$audit_json" "any(e['result'] == 'ok' for e in d['items'])")"
if [ "$audited" = "True" ]; then
  ok "audit has a deploy.create entry with result=ok (principal + action recorded)"
else
  bad "no ok deploy.create entry in GET /api/audit"
fi
note "every write is on the append-only record (Invariant #3)"

# --- 6. bounded service logs (the MCP service_logs projection) ------------------
say "6. GET /api/services/${APP_NAME}/logs?tail=5 — bounded reads"
logs_json="$(api_get "/api/services/${APP_NAME}/logs?tail=5")"
log_count="$(json_field "$logs_json" "len(d)")"
if [ "$log_count" -le 5 ]; then
  ok "tail=5 returned ${log_count} entries (<= 5)"
else
  bad "tail=5 returned ${log_count} entries"
fi
over_code="$(api_code "/api/services/${APP_NAME}/logs?tail=999999")"
if [ "$over_code" = "422" ]; then
  ok "absurd tail rejected with 422 (server-side cap)"
else
  bad "tail=999999 returned HTTP ${over_code} (expected 422)"
fi
note "agent reads are bounded by construction — MCP service_logs wraps this route"

# --- 7. audit is admin-only ------------------------------------------------------
say "7. GET /api/audit with a readonly token — 403"
if [ "$AUTH_ENABLED" = "1" ]; then
  ro_json="$(curl -sf -X POST -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"name": "demo-p6-readonly", "role": "readonly"}' "${API}/api/tokens")"
  RO_TOKEN="$(json_field "$ro_json" "d['token']")"
  RO_ID="$(json_field "$ro_json" "d['id']")"
  ro_code="$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${RO_TOKEN}" "${API}/api/audit")"
  if [ "$ro_code" = "403" ]; then
    ok "readonly token got 403 on the audit log"
  else
    bad "readonly token got HTTP ${ro_code} on /api/audit (expected 403)"
  fi
  curl -sf -X DELETE -H "Authorization: Bearer ${TOKEN}" \
    "${API}/api/tokens/${RO_ID}" >/dev/null || true
else
  skip "no-auth local mode: every principal is admin, 403 not observable here"
fi
note "the audit trail is privileged; MCP get_audit documents the admin-403"

# --- 8. cleanup -------------------------------------------------------------------
say "8. Cleanup"
if [ "$KEEP_APP" = "1" ]; then
  skip "KEEP_APP=1 — leaving ${APP_NAME} running"
else
  if nerdit services rm "$APP_NAME"; then
    ok "service removed (container stopped, endpoint + route released)"
  else
    bad "nerdit services rm ${APP_NAME} failed"
  fi
fi

# --- summary -----------------------------------------------------------------------
say "Summary"
echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "P6 surface runbook: FAIL"
  exit 1
fi
echo "P6 surface runbook: PASS"
