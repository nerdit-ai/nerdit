#!/usr/bin/env bash
# demo_p5.sh — P5 north-star manual runbook (NOT run in CI).
#
# Reproducible, copy-paste runnable from the repo root, against a live daemon
# on real hardware. It proves the whole AI wedge end-to-end:
#
#   nerdit serve llama3.1:8b --gpus 1     # local OpenAI-compatible model
#   nerdit deploy examples/ai-app          # both [ai.*] bindings wired via env
#   curl <app>/ + /chat (+ ?binding=cheap) # app answers via BOTH providers
#   daemon restart → re-curl               # reboot survival
#
# NOTES (read before running):
# * Decision #6: the FIRST-EVER model weights download (~5 GB for an 8B model)
#   is EXCLUDED from the <5-min demo clock — the persistent
#   <data_dir>/models/ollama volume makes every later run fast. This script
#   reports the model-ready time separately from the deploy-to-live time.
# * Hardened configs: with [containers].read_only_rootfs = true, verify the
#   ollama container tolerates a read-only rootfs (the weights volume is
#   writable, but the runtime may need scratch space) — flagged during review.
#   If the model container crash-loops, check `nerdit logs` for EROFS errors.
# * The automated CI counterpart is tests/test_p5_northstar.py.

set -euo pipefail

HOST="${NERDIT_HOST:-127.0.0.1}"
PORT="${NERDIT_PORT:-9321}"
API="http://${HOST}:${PORT}"
MODEL_REF="llama3.1:8b"
APP_DIR="examples/ai-app"
APP_NAME="ai-app"
MODEL_WAIT_S="${MODEL_WAIT_S:-1800}"   # generous: covers a first-ever pull
APP_WAIT_S="${APP_WAIT_S:-300}"

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
note() { printf '   proves: %s\n' "$*"; }

api_get() {  # api_get <path> — authenticated GET, raw JSON on stdout
  curl -sf -H "Authorization: Bearer ${TOKEN}" "${API}$1"
}

json_field() {  # json_field <json> <python-expr over parsed `d`>
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2]))' "$1" "$2"
}

# --- 0. preflight ------------------------------------------------------------
say "0. Preflight"
[ -d "$APP_DIR" ] || { echo "Run this from the repo root (missing $APP_DIR)."; exit 1; }
command -v nerdit >/dev/null || { echo "nerdit CLI not on PATH (pip install -e .)."; exit 1; }
curl -sf "${API}/health" >/dev/null || { echo "Daemon not reachable at ${API} — start nerditd."; exit 1; }
TOKEN="$(nerdit token)"
echo "Daemon up at ${API}."
note "the control plane is reachable and authenticated"

# --- 1. serve the local model ------------------------------------------------
say "1. nerdit serve ${MODEL_REF} --gpus 1"
T0=$(date +%s)
# A re-run against an already-served model 409s (name taken) — that's fine,
# the wait loop below still gates on running+pulled.
nerdit serve "$MODEL_REF" --gpus 1 || echo "(already served — continuing)"
note "a model is just a kind=model workload: POST /api/models, audited + idempotent"

say "2. Wait for the model: status=running AND model_pulled=true"
echo "(first-ever weights download can take many minutes — excluded from the demo clock)"
deadline=$(( $(date +%s) + MODEL_WAIT_S ))
while :; do
  models_json="$(api_get /api/models)"
  state="$(json_field "$models_json" \
    "next(((i['status'], i['model_pulled']) for i in d['items'] if i['model'] == '${MODEL_REF}'), ('absent', False))")"
  echo "  model state: ${state}"
  case "$state" in
    "('running', True)") break ;;
    "('failed',"*) echo "Model failed — check: nerdit logs"; exit 1 ;;
  esac
  [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for the model."; exit 1; }
  sleep 5
done
T_MODEL=$(date +%s)
echo "Model ready in $(( T_MODEL - T0 ))s (includes any first-ever pull)."
note "off-tick image pull + ensure_model persisted model_pulled; endpoint is OpenAI-compatible (/v1)"

# --- 3. deploy the app --------------------------------------------------------
# NOTE (ordering): secrets attach to an EXISTING service row — `nerdit secrets
# set` 404s on a name with no service yet (anti-squatting, by design). So we
# deploy FIRST, then set the secret; the app harmlessly parks in "Waiting on AI
# binding" until the [ai.cheap] secret lands, then launches on the next tick.
# The OpenAI key is read BEFORE the deploy clock so human typing time never
# counts against the <5-min deploy-to-live measurement.
say "3. Read the external-API key (before the clock; set after deploy)"
read -r -s -p "Enter the OpenAI API key for the [ai.cheap] binding: " OPENAI_KEY
echo
note "secrets are write-only (names echoed, values never readable back)"

say "4. nerdit deploy ${APP_DIR}  (+ secrets set once the service row exists)"
T1=$(date +%s)
nerdit deploy "$APP_DIR"
note "folder → ZIP → server-side Node buildpack build → [ai.*] parsed + gated server-side"
# The service row now exists — set the secret so [ai.cheap] resolves.
nerdit secrets set "$APP_NAME" "OPENAI_KEY=${OPENAI_KEY}"
unset OPENAI_KEY
note "the app waits on the binding until this lands, then launches (all-or-nothing)"

say "5. Wait for the app: status=running"
deadline=$(( $(date +%s) + APP_WAIT_S ))
while :; do
  svc_json="$(api_get "/api/services/${APP_NAME}")"
  status="$(json_field "$svc_json" "d['status']")"
  echo "  app status: ${status}"
  [ "$status" = "running" ] && break
  [ "$status" = "failed" ] && { echo "App failed — check: nerdit services logs ${APP_NAME}"; exit 1; }
  [ "$(date +%s)" -lt "$deadline" ] || { echo "Timed out waiting for the app."; exit 1; }
  sleep 3
done
# Public URL when [proxy].enabled, else the loopback endpoint.
APP_URL="$(json_field "$svc_json" \
  "(d.get('endpoint') or {}).get('public_url') or 'http://127.0.0.1:%s' % (d.get('endpoint') or {}).get('host_port')")"
T2=$(date +%s)
echo "App live at ${APP_URL} — deploy-to-live: $(( T2 - T1 ))s (the <5-min clock)."
note "build + run + URL + restart policy, in one command"

# --- 6. exercise BOTH bindings -------------------------------------------------
# -k: the embedded Caddy uses an internal CA for TLS.
say "6a. GET / — binding env echo (key NAMES only, never values)"
curl -sk "${APP_URL}/" && echo
note "OPENAI_BASE_URL/OPENAI_MODEL + NERDIT_AI_* injected; app code is provider-agnostic"

say "6b. GET /chat — [ai.default] via the LOCAL Ollama model"
curl -sk "${APP_URL}/chat" && echo
note "the app talks OpenAI to a local GPU model it never configured"

say "6c. GET /chat?binding=cheap — [ai.cheap] via the external API"
curl -sk "${APP_URL}/chat?binding=cheap" && echo
note "same app, same contract, remote provider — swapped by config, not code"

# --- 7. reboot survival ---------------------------------------------------------
say "7. Reboot survival"
echo "Restart the daemon now (e.g. 'nerdit exit && nerditd' or systemctl restart nerditd)."
read -r -p "Press Enter once the daemon is back up... "
until curl -sf "${API}/health" >/dev/null; do sleep 2; done
deadline=$(( $(date +%s) + APP_WAIT_S ))
while :; do
  status="$(json_field "$(api_get "/api/services/${APP_NAME}")" "d['status']")"
  [ "$status" = "running" ] && break
  [ "$(date +%s)" -lt "$deadline" ] || { echo "App did not come back after reboot."; exit 1; }
  sleep 3
done
say "7b. Re-curl after reboot"
curl -sk "${APP_URL}/" >/dev/null && echo "GET / OK"
curl -sk "${APP_URL}/chat" && echo
note "model + app re-adopted from the DB; bindings re-resolved identically; weights NOT re-pulled"

# --- 8. timings ------------------------------------------------------------------
say "8. Timings"
echo "model serve → ready : $(( T_MODEL - T0 ))s (first-ever pull excluded from the demo clock)"
echo "deploy → live       : $(( T2 - T1 ))s (pass/fail: < 300s)"
echo "Record these in the project's manual runbook notes."
