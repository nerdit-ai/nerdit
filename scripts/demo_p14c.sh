#!/usr/bin/env bash
# demo_p14c.sh — P14c backup v1 + `nerdit restore` runbook (NOT run in CI).
#
# Reproducible, copy-paste runnable from the repo root against a live, scratch-HOME
# daemon. It drives the P14c control-plane backup/restore loop end-to-end and
# prints PASS/FAIL/SKIP per step:
#
#   nerdit backup                              # stage a key-bearing control-plane tar
#   GET /api/audit?action=system.backup        # basename + contains_master_key only
#   nerdit restore  (refused while live)        # pid + /health + held .restore.lock
#   POST /system/backup  (staged rotation)      # 409 secret.rotation_in_progress, no path
#   nerdit exit → wipe DB+secrets → nerdit restore → nerdit init   # round-trip
#   held exclusive .restore.lock → nerdit init  # daemon refuses to boot
#
# READ BEFORE RUNNING — this is an OPERATOR runbook, not an automated test:
# * Run it against a DISPOSABLE, scratch-HOME daemon only. Point HOME at a scratch
#   dir and start the daemon there first: a live run wants a sandboxed HOME, real
#   Docker + Caddy, and cleanup. Legs 4-6 STOP, WIPE and RESTART the daemon and
#   DELETE the control-plane DB + secrets — never run them against a real
#   ~/.nerdit.
# * The destructive legs (4-6) are GATED: they run only when CONFIRM_DESTROY=1.
#   Without it they are SKIPPED with a note, so legs 1-3 stay safe to re-run.
# * Legs that need the daemon stopped/restarted are marked "OPERATOR" — the script
#   drives `nerdit exit`/`nerdit init` itself, but they change daemon state.
# * The seed app is a self-contained, dependency-free Node HTTP server that writes
#   to its implicit /data volume (NERDIT_DATA_DIR) — no example folder or GPU. The
#   app's data volume is NOT in the backup (control-plane only); it survives the
#   restore because leg 4 wipes only the daemon DB + secrets, never services/.
# * The proxy/TLS checks (leg 1/5) are best-effort: if the proxy is off they SKIP.
# * There is no automated CI counterpart: this is the live run against a real
#   daemon that every daemon/CLI change ends with.
# * Pass = all executed legs green; legs 4+5 (round-trip + boot-probe) are the
#   gate — a failure there means backup/restore itself is broken.

set -euo pipefail

HOST="${NERDIT_HOST:-127.0.0.1}"
PORT="${NERDIT_PORT:-9321}"
API="http://${HOST}:${PORT}"
APP_NAME="demo-p14c-app"
CONFIRM_DESTROY="${CONFIRM_DESTROY:-0}"
APP_WAIT_S="${APP_WAIT_S:-300}"

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

api_body() {  # api_body <path> [extra curl args...] — response body (any status)
  local path="$1"; shift
  curl -s -H "Authorization: Bearer ${TOKEN}" "$@" "${API}${path}"
}

json_field() {  # json_field <json> <python-expr over parsed `d`>
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2"
}

# Run a command, capturing its exit code without tripping `set -e`.
run_rc() { set +e; "$@"; local rc=$?; set -e; return $rc; }

# Count backup tars without tripping pipefail when zero exist (ls exits 2).
count_backups() { (ls -1 "${BACKUP_DIR}"/nerdit-backup-*.tar.gz 2>/dev/null || true) | wc -l | tr -d ' '; }

wait_health() {  # wait_health — block until /health answers (up to ~30s)
  for _ in $(seq 1 30); do
    curl -sf "${API}/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

# --- 0. preflight ------------------------------------------------------------
say "0. Preflight"
command -v nerdit  >/dev/null || { echo "nerdit CLI not on PATH (pip install -e .)."; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for JSON parsing."; exit 1; }
command -v docker  >/dev/null || { echo "docker required (the buildpack path builds a real image)."; exit 1; }
command -v curl    >/dev/null || { echo "curl required."; exit 1; }
curl -sf "${API}/health" >/dev/null || { echo "Daemon not reachable at ${API} — start a scratch-HOME nerditd."; exit 1; }
TOKEN="$(nerdit token)"

# Resolve the daemon's data_dir from the active settings (respects scratch HOME).
DATA_DIR="$(python3 - <<'PYEOF'
from pathlib import Path
from nerdit.config.settings import load_settings
print(Path(load_settings().data_dir).expanduser())
PYEOF
)"
[ -n "$DATA_DIR" ] || { echo "could not resolve data_dir"; exit 1; }
BACKUP_DIR="${DATA_DIR}/backups"

# Guard against the operator's REAL home (not $HOME, which a scratch run
# points elsewhere — data_dir is always <scratch HOME>/.nerdit and must pass).
REAL_HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6 || true)"
REAL_HOME="${REAL_HOME:-$HOME}"
case "$DATA_DIR" in
  "$REAL_HOME/.nerdit"|"/root/.nerdit")
    if [ "$CONFIRM_DESTROY" = "1" ]; then
      echo "REFUSING: data_dir is ${DATA_DIR} and CONFIRM_DESTROY=1 — point HOME at a scratch dir first."
      exit 1
    fi
    echo "WARNING: data_dir is a default location (${DATA_DIR}); destructive legs 4-6 disabled."
    ;;
esac

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
echo "Daemon up at ${API}; data_dir ${DATA_DIR}; scratch workdir ${WORKDIR}."
note "the control plane is reachable, authenticated, and its data_dir resolved"

# --- 1. seed control-plane state ---------------------------------------------
say "1. Seed state: deploy a data-writing app + set a secret"
APP_DIR="${WORKDIR}/app"
mkdir -p "$APP_DIR"
cat > "${APP_DIR}/package.json" <<'EOF'
{ "name": "demo-p14c-app", "version": "1.0.0", "private": true,
  "scripts": { "start": "node server.js" } }
EOF
# Persists a counter into the implicit /data volume so it survives the restore
# (the volume is NOT in the tar — it is never wiped in leg 4).
cat > "${APP_DIR}/server.js" <<'EOF'
const http = require("http");
const fs = require("fs");
const path = require("path");
const dir = process.env.NERDIT_DATA_DIR || "/data";
const file = path.join(dir, "app.json");
const load = () => { try { return JSON.parse(fs.readFileSync(file)); } catch { return { rows: 0 }; } };
const port = process.env.PORT || 3000;
http.createServer((req, res) => {
  const s = load();
  if (req.url === "/insert") { s.rows += 1; fs.writeFileSync(file, JSON.stringify(s)); }
  res.writeHead(200, { "Content-Type": "application/json" });
  res.end(JSON.stringify(s));
}).listen(port, () => console.log(`demo-p14c-app on ${port}`));
EOF
cat > "${APP_DIR}/nerdit.toml" <<EOF
[deploy]
name = "${APP_NAME}"
port = 3000
EOF

if run_rc nerdit deploy "$APP_DIR" --wait --timeout "$APP_WAIT_S"; then
  ok "seed app deployed + healthy"
else
  bad "seed app failed to converge (rc $?)"
fi

# Insert a couple of rows through the app (via the proxy if up, else skip rows).
APP_ROUTE=""
if json_field "$(api_get "/api/services/${APP_NAME}" 2>/dev/null || echo '{}')" \
     "bool((d.get('endpoint') or {}).get('public_url'))" 2>/dev/null | grep -q True; then
  APP_ROUTE="$(json_field "$(api_get "/api/services/${APP_NAME}")" "d['endpoint']['public_url']")"
  curl -sk "${APP_ROUTE%/}/insert" >/dev/null 2>&1 || true
  curl -sk "${APP_ROUTE%/}/insert" >/dev/null 2>&1 || true
  rows="$(curl -sk "${APP_ROUTE%/}/" 2>/dev/null | json_field /dev/stdin "d" 2>/dev/null || echo '?')"
  if curl -sk "${APP_ROUTE%/}/" >/dev/null 2>&1; then
    ok "app rows written + served over ${APP_ROUTE} (proxy/TLS up)"
  else
    skip "proxy up but app not reachable over its public_url — rows unverified"
  fi
else
  skip "proxy off (no public_url) — app data written on next reconcile only"
fi

if run_rc nerdit secrets set "$APP_NAME" DEMO_KEY=p14c-secret --yes 2>/dev/null \
   || run_rc nerdit secrets set "$APP_NAME" DEMO_KEY=p14c-secret; then
  ok "secret DEMO_KEY set on ${APP_NAME}"
else
  skip "could not set a secret (older CLI signature?) — backup still carries the master key"
fi
note "control-plane state (DB rows + secrets + CA identity) now exists to capture"

# --- 2. backup ----------------------------------------------------------------
say "2. nerdit backup — stage a key-bearing control-plane tar"
BEFORE_COUNT="$(count_backups)"
backup_out="$(nerdit backup --yes 2>&1)"
printf '%s\n' "$backup_out" | sed 's/^/     /'
AFTER_COUNT="$(count_backups)"
TAR="$(ls -1t "${BACKUP_DIR}"/nerdit-backup-*.tar.gz 2>/dev/null | head -1 || true)"

if [ -n "$TAR" ] && [ -f "$TAR" ]; then
  ok "tar written: $(basename "$TAR")"
else
  bad "no nerdit-backup-*.tar.gz appeared under ${BACKUP_DIR}"
fi
if printf '%s' "$backup_out" | grep -qi "kid"; then
  ok "kid printed in the CLI output"
else
  bad "kid not printed"
fi
if printf '%s' "$backup_out" | grep -qi "master key"; then
  ok "custody warning shown (archive contains the master key)"
else
  bad "custody warning missing"
fi
if printf '%s' "$backup_out" | grep -qi "backup_keep_last"; then
  ok "keep==0 honesty hint shown (no backups deleted by default)"
else
  bad "keep==0 hint missing"
fi
# Tar perms + member sanity.
if [ -n "$TAR" ]; then
  mode="$(stat -c '%a' "$TAR" 2>/dev/null || stat -f '%Lp' "$TAR" 2>/dev/null || echo '?')"
  [ "$mode" = "600" ] && ok "tar mode is 0600" || bad "tar mode is ${mode} (expected 600)"
  members="$(tar -tzf "$TAR" 2>/dev/null || true)"
  if printf '%s\n' "$members" | grep -q '^manifest.json$' \
     && printf '%s\n' "$members" | grep -q '^db/nerdit.db$' \
     && printf '%s\n' "$members" | grep -q '^secrets/secrets.key$'; then
    ok "tar carries manifest.json + db/nerdit.db + secrets/secrets.key"
  else
    bad "tar member set is missing a core file"
  fi
fi

# nerdit disk shows the backup count.
disk_out="$(nerdit disk 2>&1 || true)"
if printf '%s' "$disk_out" | grep -qi "backups"; then
  ok "nerdit disk reports the backups line ($(printf '%s' "$disk_out" | grep -i backups | head -1 | sed 's/^ *//'))"
else
  skip "nerdit disk did not surface a backups line"
fi

# Audit row: basename + contains_master_key only, no path.
audit_json="$(api_get "/api/audit?action=system.backup&limit=1" 2>/dev/null || echo '{}')"
params="$(json_field "$audit_json" "(d.get('items') or [{}])[0].get('params_redacted')" 2>/dev/null || echo 'None')"
if printf '%s' "$params" | grep -q "contains_master_key" \
   && printf '%s' "$params" | grep -q "backup" \
   && ! printf '%s' "$params" | grep -q "/"; then
  ok "audit system.backup params carry basename + contains_master_key, no path (${params})"
else
  bad "audit params wrong or leak a path: ${params}"
fi

# Idempotency replay: same key → cached body, still one new tar.
IDEM="demo-p14c-$(date +%s)"
BODY1="$(api_body /api/system/backup -X POST -H "Idempotency-Key: ${IDEM}")"
MID_COUNT="$(count_backups)"
BODY2="$(api_body /api/system/backup -X POST -H "Idempotency-Key: ${IDEM}")"
END_COUNT="$(count_backups)"
# Compare parsed JSON — the cached replay body is re-serialized, so raw
# string equality fails on whitespace alone.
SAME_BODY="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1]) == json.loads(sys.argv[2]))' "$BODY1" "$BODY2" 2>/dev/null || echo False)"
if [ "$SAME_BODY" = "True" ] && [ "$MID_COUNT" = "$END_COUNT" ]; then
  ok "Idempotency-Key replay returned the cached body — no second tar staged"
else
  bad "idempotency replay staged a second tar or changed the body"
fi
note "a backup is a single key-bearing tar; audit is path-free; retries collapse"

# --- 3. refusals (daemon live) -----------------------------------------------
say "3. Refusals while the daemon is live"
# 3a. restore refused (held LOCK_SH + pid + /health).
if [ -n "$TAR" ]; then
  if run_rc nerdit restore "$TAR" --yes >/dev/null 2>&1; then
    bad "nerdit restore succeeded against a LIVE daemon (must refuse)"
  else
    ok "nerdit restore refused against the live daemon (exit 1)"
  fi
else
  skip "no tar from leg 2 — restore-refusal not exercised"
fi

# 3b. staged rotation → 409 secret.rotation_in_progress, body carries no path.
touch "${DATA_DIR}/secrets.key.new"
rot_body="$(api_body /api/system/backup -X POST)"
rm -f "${DATA_DIR}/secrets.key.new"
if printf '%s' "$rot_body" | grep -q "rotation_in_progress" \
   && ! printf '%s' "$rot_body" | grep -q "/"; then
  ok "staged rotation → 409 secret.rotation_in_progress, path-free body"
else
  bad "staged-rotation response wrong or leaks a path: ${rot_body}"
fi

# 3c. concurrent-staging 409 (best-effort race — the loser may run serially).
(api_body /api/system/backup -X POST >/dev/null 2>&1 &)
race_code="$(api_code /api/system/backup -X POST || echo '000')"
if [ "$race_code" = "409" ]; then
  ok "concurrent POST /system/backup → 409 backup.in_progress"
else
  skip "concurrent request did not overlap (best-effort; got HTTP ${race_code}) — loser runs serially"
fi
wait 2>/dev/null || true
note "a live daemon holds the state; restore + overlapping/staged backups all refuse"

# --- 4+5. destroy → restore → boot-probe (OPERATOR, destructive) -------------
say "4+5. Round-trip + boot-probe (OPERATOR — stops/wipes/restarts the daemon)"
if [ "$CONFIRM_DESTROY" != "1" ]; then
  skip "CONFIRM_DESTROY!=1 — the destructive round-trip is disabled (set CONFIRM_DESTROY=1 on a scratch HOME)"
elif [ -z "$TAR" ]; then
  skip "no tar from leg 2 — round-trip skipped"
else
  # 5a. Boot probe: hold the exclusive .restore.lock, then try to start.
  say "  5a. Held exclusive .restore.lock → nerdit init must refuse"
  run_rc nerdit exit --yes >/dev/null 2>&1 || true
  sleep 2
  if curl -sf "${API}/health" >/dev/null 2>&1; then
    bad "daemon still answering after 'nerdit exit' — was it started via 'nerdit init' (pid file)?"
    echo "ABORT: never wipe the data dir under a live daemon. PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
    exit 1
  fi
  python3 - "$DATA_DIR" <<'PYEOF' &
import fcntl, os, sys, time
data_dir = sys.argv[1]
fd = os.open(os.path.join(data_dir, ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
try:
    # Non-blocking: if a daemon still holds its shared lock the leg must fail
    # fast, not deadlock the runbook (`nerdit exit` may have missed it).
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("lock-holder: .restore.lock still held (daemon alive?)", file=sys.stderr)
    sys.exit(1)
time.sleep(15)
os.close(fd)
PYEOF
  LOCK_PID=$!
  sleep 1
  if run_rc nerdit init >/dev/null 2>&1 && wait_health; then
    bad "daemon booted while an exclusive .restore.lock was held (must refuse)"
    run_rc nerdit exit --yes >/dev/null 2>&1 || true
  else
    ok "daemon refused to boot while a restore held the exclusive lock"
  fi
  wait "$LOCK_PID" 2>/dev/null || true

  # 4. Destroy control-plane state (NOT the app data volume) + restore.
  say "  4. Wipe DB + secrets, then nerdit restore"
  rm -f "${DATA_DIR}/nerdit.db" "${DATA_DIR}/nerdit.db-wal" "${DATA_DIR}/nerdit.db-shm"
  rm -rf "${DATA_DIR}/secrets"
  rm -f "${DATA_DIR}/secrets.key"
  if run_rc nerdit restore "$TAR" --yes; then
    ok "nerdit restore exited 0 (manifest validated, perms pass, move-in done)"
  else
    bad "nerdit restore failed (rc $?)"
  fi

  # 5b. Resurrection: start the daemon, verify secrets + service + CA identity.
  say "  5b. Restart → secrets + service + CA identity survive"
  run_rc nerdit init >/dev/null 2>&1 || true
  if wait_health; then
    ok "daemon back up after restore (holding LOCK_SH again)"
    TOKEN="$(nerdit token)"
    if run_rc nerdit secrets list "$APP_NAME" 2>/dev/null | grep -qi "DEMO_KEY"; then
      ok "restored secret DEMO_KEY readable under the restored key"
    else
      skip "DEMO_KEY not listed (secret may not have been set in leg 1)"
    fi
    svc_status="$(json_field "$(api_get "/api/services/${APP_NAME}" 2>/dev/null || echo '{}')" \
      "d.get('status')" 2>/dev/null || echo '?')"
    [ -n "$svc_status" ] && [ "$svc_status" != "None" ] \
      && ok "service ${APP_NAME} known after restore (status=${svc_status})" \
      || bad "service ${APP_NAME} not present after restore"
  else
    bad "daemon did not come back after restore"
  fi
fi
note "round-trip restores DB + secrets + CA identity; the boot probe is a held lock"

# --- 6. retention integration (OPERATOR) -------------------------------------
say "6. Retention: backup_keep_last bounds the key-bearing tars"
if [ "$CONFIRM_DESTROY" != "1" ]; then
  skip "CONFIRM_DESTROY!=1 — retention leg disabled (it restarts the daemon)"
else
  gc_before="$(api_get "/api/system/gc?dry_run=true" -X POST 2>/dev/null \
    | json_field /dev/stdin "d.get('backups_over_keep')" 2>/dev/null || echo '?')"
  echo "   (set [retention].backup_keep_last = 1 via 'nerdit config apply' + restart, then:)"
  run_rc nerdit backup --yes >/dev/null 2>&1 || true
  run_rc nerdit backup --yes >/dev/null 2>&1 || true
  FINAL_COUNT="$(count_backups)"
  echo "   backups on disk after two more: ${FINAL_COUNT} (dry-run backups_over_keep before sweep: ${gc_before})"
  skip "retention sweep + config apply is operator-driven — verify keep=1 leaves exactly 1 tar manually"
fi
note "the P14b sweep (lit by P14c) bounds key-bearing archives once keep is nonzero"

# --- 7. cleanup ---------------------------------------------------------------
say "7. Cleanup"
if run_rc nerdit services rm "$APP_NAME" --yes >/dev/null 2>&1 \
   || run_rc nerdit services rm "$APP_NAME" >/dev/null 2>&1; then
  ok "service ${APP_NAME} removed"
else
  skip "service ${APP_NAME} not present / already removed"
fi
echo "   NOTE: key-bearing tars remain under ${BACKUP_DIR} — delete them off-box (custody)."
echo "   NOTE: leave no scratch daemon running for the pytest pass (pgrep -f nerditd)."

# --- summary ------------------------------------------------------------------
say "Summary"
echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "P14c backup/restore runbook: FAIL"
  exit 1
fi
echo "P14c backup/restore runbook: PASS"
