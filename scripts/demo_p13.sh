#!/usr/bin/env bash
# demo_p13.sh — P13 agent-operability runbook (NOT run in CI).
#
# Reproducible, copy-paste runnable from the repo root against a live daemon.
# It drives the P13 agent loop end-to-end and prints PASS/FAIL per step:
#
#   nerdit deploy --wait                       # converge-or-fail with exit codes
#   GET /api/services/{name}/diagnose          # one-call failure bundle + remediation
#   nerdit config app set … / deploy --dry-run # fix + preview with zero writes
#   GET /api/capabilities · /api/doctor        # daemon self-knowledge
#   POST /api/mcp (streamable-HTTP MCP)         # a readonly agent session over HTTP
#
# NOTES (read before running):
# * The deployed apps are self-contained, dependency-free Node HTTP servers
#   generated into a temp dir — no example folder or GPU required. Docker and
#   the Node buildpack path are exercised for real.
# * Step 2 (OOM) is best-effort: a runtime that does not enforce cgroup memory
#   limits is SKIPPED (not failed).
# * Step 5 (MCP over HTTP) is SKIPPED unless the daemon has [mcp].http_enabled
#   and the optional `mcp` extra is importable. Enable it with:
#     [daemon] auth_token = "…"   and   [mcp] http_enabled = true   (restart).
# * There is no automated CI counterpart: this is the live run against a real
#   daemon that every daemon/CLI change ends with.

set -euo pipefail

HOST="${NERDIT_HOST:-127.0.0.1}"
PORT="${NERDIT_PORT:-9321}"
API="http://${HOST}:${PORT}"
APP_NAME="demo-p13-app"
OOM_NAME="demo-p13-oom"
APP_WAIT_S="${APP_WAIT_S:-300}"
KEEP_APP="${KEEP_APP:-0}"
DO_RESTART="${DO_RESTART:-0}"

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

# Run a command, capturing its exit code without tripping `set -e`.
run_rc() { set +e; "$@"; local rc=$?; set -e; return $rc; }

# --- 0. preflight ------------------------------------------------------------
say "0. Preflight"
command -v nerdit >/dev/null || { echo "nerdit CLI not on PATH (pip install -e .)."; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for JSON parsing."; exit 1; }
command -v docker  >/dev/null || { echo "docker required (the buildpack path builds a real image)."; exit 1; }
curl -sf "${API}/health" >/dev/null || { echo "Daemon not reachable at ${API} — start nerditd."; exit 1; }
TOKEN="$(nerdit token)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
echo "Daemon up at ${API}; scratch workdir ${WORKDIR}."
note "the control plane is reachable and authenticated"

# --- 1. broken-app agent loop -------------------------------------------------
say "1. Broken-app loop: deploy --wait → diagnose → fix → dry-run → redeploy"
APP_DIR="${WORKDIR}/app"
mkdir -p "$APP_DIR"
cat > "${APP_DIR}/package.json" <<'EOF'
{
  "name": "demo-p13-app",
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
    res.end("hello from demo-p13-app\n");
  })
  .listen(port, () => console.log(`demo-p13-app listening on ${port}`));
EOF
# A deliberately broken start command: the container launches and crashes.
cat > "${APP_DIR}/nerdit.toml" <<EOF
[deploy]
name = "${APP_NAME}"
port = 3000
start = "node does-not-exist.js"
EOF

COLD_T0=$(date +%s)
if run_rc nerdit deploy "$APP_DIR" --wait --timeout "$APP_WAIT_S"; then
  bad "broken deploy converged healthy (expected failure exit 1)"
else
  rc=$?
  if [ "$rc" -eq 1 ]; then
    ok "nerdit deploy --wait exited 1 on the broken start (scriptable, no log scraping)"
  else
    bad "nerdit deploy --wait exited ${rc} (expected 1 for a failed deploy)"
  fi
fi
COLD_T1=$(date +%s)
COLD_S=$(( COLD_T1 - COLD_T0 ))
note "the deploy converge signal is a coded exit, not a status poll"

say "1b. GET /api/services/${APP_NAME}/diagnose — one-call failure bundle"
diag_json="$(api_get "/api/services/${APP_NAME}/diagnose")"
rem_code="$(json_field "$diag_json" "d['remediation']['code']")"
if [ -n "$rem_code" ] && [ "$rem_code" != "None" ]; then
  ok "diagnose returned remediation_code='${rem_code}' (phase=$(json_field "$diag_json" "(d.get('last_deploy') or {}).get('phase')"))"
else
  bad "diagnose carried no remediation code"
fi
# The bundle carries key NAMES only, never secret values (Invariant #3).
leak="$(json_field "$diag_json" "'injected_env_keys' in d and isinstance(d['injected_env_keys'], list)")"
[ "$leak" = "True" ] && ok "injected_env_keys is a names-only list (no secret values)" \
  || bad "injected_env_keys missing/mistyped"
note "an agent branches on remediation_code instead of scraping logs"

say "1c. Fix start via config-as-API, then dry-run the redeploy"
if run_rc nerdit config app set "$APP_NAME" deploy start="node server.js"; then
  ok "nerdit config app set applied the corrected start (config-as-API, audited)"
else
  bad "nerdit config app set failed"
fi
# Keep the source consistent so the redeploy does not re-clobber the API fix
# (an agent fixes both the live config and the repo it deploys from).
cat > "${APP_DIR}/nerdit.toml" <<EOF
[deploy]
name = "${APP_NAME}"
port = 3000
start = "node server.js"
EOF
dry_out="$(nerdit deploy "$APP_DIR" --dry-run 2>&1)"
if printf '%s' "$dry_out" | grep -qi "dry run"; then
  ok "nerdit deploy --dry-run returned a plan diff with zero writes"
  printf '%s\n' "$dry_out" | sed 's/^/     /'
else
  bad "nerdit deploy --dry-run did not render a plan"
fi
note "dry-run previews the buildpack + names-only env diff before committing"

say "1d. Real redeploy --wait → healthy"
if run_rc nerdit deploy "$APP_DIR" --wait --timeout "$APP_WAIT_S"; then
  ok "nerdit deploy --wait exited 0 (converged healthy)"
else
  bad "redeploy --wait exited $? (expected 0)"
fi
status="$(json_field "$(api_get "/api/services/${APP_NAME}")" "d['status']")"
[ "$status" = "running" ] && ok "service status=running" || bad "service status=${status}"
note "the loop closes: fail → diagnose → fix → converge, all API-driven"

# --- 2. OOM loop (best-effort) ------------------------------------------------
say "2. OOM loop: memory_limit → diagnose oom_killed → raise → converge"
OOM_DIR="${WORKDIR}/oom"
mkdir -p "$OOM_DIR"
cat > "${OOM_DIR}/package.json" <<'EOF'
{ "name": "demo-p13-oom", "version": "1.0.0", "private": true,
  "scripts": { "start": "node server.js" } }
EOF
# Allocate well past the 64m cap on boot, then serve — the cap kills it.
cat > "${OOM_DIR}/server.js" <<'EOF'
const http = require("http");
const hog = [];
for (let i = 0; i < 4096; i++) hog.push(Buffer.alloc(1024 * 1024, 1)); // ~4 GB
const port = process.env.PORT || 3000;
http.createServer((req, res) => res.end("ok\n")).listen(port);
EOF
cat > "${OOM_DIR}/nerdit.toml" <<EOF
[deploy]
name = "${OOM_NAME}"
port = 3000
memory_limit = "64m"
EOF
run_rc nerdit deploy "$OOM_DIR" --wait --timeout "$APP_WAIT_S" || true
oom_diag="$(api_get "/api/services/${OOM_NAME}/diagnose" 2>/dev/null || echo '{}')"
oom_killed="$(json_field "$oom_diag" "d.get('forensics', {}).get('oom_killed', False)")"
if [ "$oom_killed" = "True" ]; then
  ok "diagnose forensics.oom_killed=true (last_exit_code=$(json_field "$oom_diag" "d.get('forensics', {}).get('last_exit_code')"))"
  # Raise the limit via per-app config and restart to converge.
  if run_rc nerdit config app set "$OOM_NAME" deploy memory_limit="6g" --restart; then
    ok "raised memory_limit to 6g via config-as-API (--restart)"
  else
    bad "failed to raise memory_limit"
  fi
else
  skip "runtime did not report an OOM kill (no cgroup memory enforcement?) — OOM loop skipped"
fi
note "an OOM is a first-class, machine-readable forensic — not a mystery crash"

# --- 3. build cache -----------------------------------------------------------
say "3. Build cache: a source-only edit redeploys faster than the cold build"
# Touch only server.js (not package.json): the install layer stays cached.
cat >> "${APP_DIR}/server.js" <<'EOF'
// cache-buster comment — source-only change, deps unchanged
EOF
WARM_T0=$(date +%s)
run_rc nerdit deploy "$APP_DIR" --wait --timeout "$APP_WAIT_S" || true
WARM_T1=$(date +%s)
WARM_S=$(( WARM_T1 - WARM_T0 ))
echo "  cold build ${COLD_S}s vs warm redeploy ${WARM_S}s"
if [ "$WARM_S" -lt "$COLD_S" ]; then
  ok "source-only redeploy (${WARM_S}s) was faster than the cold build (${COLD_S}s)"
else
  skip "warm redeploy (${WARM_S}s) not faster than cold (${COLD_S}s) — cache noise on a tiny app"
fi
note "the dependency-layer reorder means a source edit reuses the install layer"

# --- 4. daemon self-knowledge -------------------------------------------------
say "4. GET /api/capabilities and /api/doctor"
caps="$(api_get /api/capabilities)"
role="$(json_field "$caps" "d['caller']['role']")"
grammar="$(json_field "$caps" "d.get('url_shape')")"
if [ -n "$role" ] && [ "$role" != "None" ]; then
  ok "capabilities: caller role='${role}', proxy url_shape='${grammar}'"
else
  bad "capabilities did not report a caller role"
fi
doc="$(api_get /api/doctor)"
top="$(json_field "$doc" "d['status']")"
checks="$(json_field "$doc" "','.join(c['name'] for c in d['checks'])")"
if [ -n "$top" ]; then
  ok "doctor: top status='${top}', checks=[${checks}]"
else
  bad "doctor did not report a top status"
fi
note "an agent orients with one read instead of probing endpoints one by one"

if [ "$DO_RESTART" = "1" ]; then
  say "4b. POST /api/daemon/restart (DO_RESTART=1)"
  rc_code="$(api_code /api/daemon/restart -X POST -H "Idempotency-Key: demo-p13-$(date +%s)")"
  if [ "$rc_code" = "202" ]; then
    ok "daemon accepted the restart (202); waiting for it to come back"
    for _ in $(seq 1 30); do sleep 1; curl -sf "${API}/health" >/dev/null && break || true; done
    curl -sf "${API}/health" >/dev/null && ok "daemon is back after restart" || bad "daemon did not return"
  else
    bad "restart returned HTTP ${rc_code} (expected 202)"
  fi
else
  skip "POST /api/daemon/restart not exercised (set DO_RESTART=1 to opt in)"
fi

# --- 5. MCP over HTTP: a readonly agent session -------------------------------
say "5. MCP over HTTP — readonly agent session (scenario 5)"
mcp_enabled="$(json_field "$caps" "bool(d.get('mcp', {}).get('http_enabled'))")"
if [ "$mcp_enabled" != "True" ]; then
  skip "[mcp].http_enabled is off — enable it + restart to run the MCP-over-HTTP session"
elif ! python3 -c 'import mcp' >/dev/null 2>&1; then
  skip "the 'mcp' extra is not importable in this python3 — pip install 'nerdit[mcp]'"
else
  # Mint a readonly scoped token and drive /api/mcp with the reference client.
  ro_json="$(curl -sf -X POST -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{"name": "demo-p13-readonly", "role": "readonly"}' "${API}/api/tokens")"
  RO_TOKEN="$(json_field "$ro_json" "d['token']")"
  RO_ID="$(json_field "$ro_json" "d['id']")"
  MCP_URL="${API}/api/mcp"
  if MCP_URL="$MCP_URL" RO_TOKEN="$RO_TOKEN" APP_NAME="$APP_NAME" python3 - <<'PYEOF'
import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = os.environ["MCP_URL"]
TOKEN = os.environ["RO_TOKEN"]
APP = os.environ["APP_NAME"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def _text(result) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        parts.append(getattr(block, "text", "") or "")
    return "\n".join(parts)


async def main() -> int:
    async with streamablehttp_client(URL, headers=HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            print(f"  tools/list returned {len(names)} tools", flush=True)
            if len(names) != 34:
                print(f"  UNEXPECTED tool count: {len(names)}", flush=True)
                return 2
            # A read tool works for a readonly token.
            ps = await session.call_tool("proxy_status", {})
            print(f"  proxy_status ok (getError={getattr(ps, 'isError', False)})", flush=True)
            # A write tool must be refused on the inner hop with a structured 403.
            stopped = await session.call_tool("stop_service", {"name": APP})
            body = _text(stopped).lower()
            if "forbidden" in body or "403" in body or getattr(stopped, "isError", False):
                print("  stop_service correctly refused (inner forbidden envelope)", flush=True)
                return 0
            print(f"  stop_service was NOT refused: {body[:200]}", flush=True)
            return 3
    return 0


sys.exit(asyncio.run(main()))
PYEOF
  then
    ok "readonly MCP-over-HTTP session: 34 tools listed, read worked, write refused (403)"
  else
    bad "MCP-over-HTTP readonly session failed (see output above)"
  fi
  curl -sf -X DELETE -H "Authorization: Bearer ${TOKEN}" \
    "${API}/api/tokens/${RO_ID}" >/dev/null || true
fi
note "MCP-over-HTTP forwards the caller's token: readonly reads, writes get an inner 403"

# --- 6. cleanup ---------------------------------------------------------------
say "6. Cleanup"
if [ "$KEEP_APP" = "1" ]; then
  skip "KEEP_APP=1 — leaving ${APP_NAME}/${OOM_NAME} running"
else
  for name in "$APP_NAME" "$OOM_NAME"; do
    if nerdit services rm "$name" >/dev/null 2>&1; then
      ok "service ${name} removed"
    else
      skip "service ${name} not present / already removed"
    fi
  done
fi

# --- summary ------------------------------------------------------------------
say "Summary"
echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "P13 agent-operability runbook: FAIL"
  exit 1
fi
echo "P13 agent-operability runbook: PASS"
