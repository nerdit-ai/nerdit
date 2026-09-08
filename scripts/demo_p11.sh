#!/usr/bin/env bash
# demo_p11.sh — vLLM model-plane runbook (P11 + P21, NOT run in CI).
#
# Reproducible, copy-paste runnable from the repo root against a live daemon on
# real hardware. It closes the real-hardware runbook P11 deferred, and proves
# the P21 model-plane operability work end-to-end, printing PASS/FAIL per step:
#
#   nerdit serve <hf-repo> --backend vllm --gpus 1   # DEFAULT config, no tuning
#   GET /api/models → running + model_pulled          # D3 auto-bounds made it fit
#   POST <endpoint>/chat/completions                  # a real completion (Invariant #1)
#   nerdit serve … --max-model-len 2048               # D4 per-serve typed override
#   GET /api/services/{name}/diagnose                 # D1+D2 GPU_OOM is self-explanatory
#
# NOTES (read before running):
# * Run it against a DISPOSABLE, scratch-HOME daemon only: a live run drives a
#   sandboxed HOME, real Docker and a real GPU. Point HOME at a scratch dir and
#   start the daemon there first:
#       export HOME=/tmp/nerdit-p11 && mkdir -p "$HOME" && nerditd
#   The script deletes every model row it creates, but it drives a real GPU and
#   pulls multi-GB weights into <data_dir>/models.
# * A GPU is MANDATORY: vLLM is GPU-only (`model.gpu_required` on gpus=0). With
#   no schedulable GPU every leg is SKIPPED (not failed) and the script exits 0.
# * The point of leg 1 is that the daemon config is UNTUNED. On a card under
#   16 GB the daemon injects `--max-model-len 4096` +
#   `--gpu-memory-utilization 0.85` at launch (P21 D3), which is what makes the
#   validated 2026-07-12 RTX 4060 recipe unnecessary as a manual step. If you
#   already have `[models].vllm_extra_args` set, leg 1b is SKIPPED (your flags
#   suppress the injection, by design).
# * The FIRST-EVER weights download is excluded from any timing claim — the
#   persistent <data_dir>/models volume makes every later run fast. WAIT_S
#   defaults to 900s to cover it.
# * Leg 4 (forced GPU OOM) is opt-in with CRASH_LEG=1: it deliberately
#   crash-loops a container to exhaustion (~4 attempts × ~60s).
# * The automated CI counterparts are tests/test_model_backend.py,
#   tests/test_services_reconcile.py and tests/test_diagnose.py.

set -euo pipefail

HOST="${NERDIT_HOST:-127.0.0.1}"
PORT="${NERDIT_PORT:-9321}"
API="http://${HOST}:${PORT}"
MODEL_REF="${MODEL_REF:-Qwen/Qwen2.5-0.5B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen-vllm-demo}"
OOM_NAME="${OOM_NAME:-qwen-vllm-oom}"
WAIT_S="${WAIT_S:-900}"          # generous: covers a first-ever HF weights download
CRASH_LEG="${CRASH_LEG:-0}"      # opt-in: leg 4 forces a real GPU OOM
KEEP_MODEL="${KEEP_MODEL:-0}"

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

# Poll GET /api/models until <name> is (running, pulled) — echoes the outcome:
# "ready", "failed" or "timeout".
wait_model() {  # wait_model <name> <deadline-seconds>
  local name="$1" budget="$2" deadline state
  deadline=$(( $(date +%s) + budget ))
  while :; do
    state="$(json_field "$(api_get /api/models)" \
      "next(((i['status'], i['model_pulled']) for i in d['items'] if i['name'] == '${name}'), ('absent', False))")"
    printf '   model state: %s\n' "$state" >&2
    case "$state" in
      "('running', True)") echo "ready"; return 0 ;;
      "('failed',"*)       echo "failed"; return 0 ;;
    esac
    [ "$(date +%s)" -lt "$deadline" ] || { echo "timeout"; return 0; }
    sleep 5
  done
}

# The launch argv of the running vLLM container serving <name>, as a JSON array
# (empty when docker is unavailable or no container matches).
vllm_cmd() {  # vllm_cmd <model-ref>
  local ref="$1" cid cmd ids
  command -v docker >/dev/null || { echo ""; return 0; }
  ids="$(docker ps -q --filter label=managed-by=nerdit 2>/dev/null || true)"
  for cid in $ids; do
    cmd="$(docker inspect --format '{{json .Config.Cmd}}' "$cid" 2>/dev/null || echo '')"
    case "$cmd" in *"\"${ref}\""*) echo "$cmd"; return 0 ;; esac
  done
  echo ""
}

# Count occurrences of a flag token in a JSON argv array.
cmd_flag_count() {  # cmd_flag_count <json-argv> <flag>
  python3 -c 'import json,sys; a=json.loads(sys.argv[1] or "[]"); print(sum(1 for t in a if t == sys.argv[2] or t.startswith(sys.argv[2] + "=")))' \
    "$1" "$2"
}

# Value that follows a flag in a JSON argv array ("" when absent).
cmd_flag_value() {  # cmd_flag_value <json-argv> <flag>
  python3 - "$1" "$2" <<'PYEOF'
import json, sys
argv = json.loads(sys.argv[1] or "[]")
flag = sys.argv[2]
for i, tok in enumerate(argv):
    if tok == flag and i + 1 < len(argv):
        print(argv[i + 1]); break
    if tok.startswith(flag + "="):
        print(tok.split("=", 1)[1]); break
else:
    print("")
PYEOF
}

cleanup_model() {  # cleanup_model <name> — best-effort delete, never fails the run
  nerdit services rm "$1" >/dev/null 2>&1 || true
}

# --- 0. preflight -------------------------------------------------------------
say "0. Preflight"
command -v nerdit  >/dev/null || { echo "nerdit CLI not on PATH (pip install -e .)."; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for JSON parsing."; exit 1; }
curl -sf "${API}/health" >/dev/null || {
  echo "Daemon not reachable at ${API} — start a scratch-HOME nerditd."; exit 1; }
TOKEN="$(nerdit token)"
echo "Daemon up at ${API}."

# Credential gate, demos and runbooks mint scratch credentials: the raw API legs
# run on a minted scratch token, revoked in cleanup, never the daemon's
# long-lived admin token.
mint_json="$(curl -sf -X POST -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' \
  -H "Idempotency-Key: demo-p11-mint-$(date +%s)" \
  -d '{"name": "demo-p11-scratch", "role": "admin"}' "${API}/api/tokens" || echo '{}')"
SCRATCH_TOKEN="$(json_field "$mint_json" "d.get('token') or ''")"
SCRATCH_TOKEN_ID="$(json_field "$mint_json" "d.get('id') or ''")"
if [ -n "$SCRATCH_TOKEN" ]; then
  TOKEN="$SCRATCH_TOKEN"
  echo "minted scratch admin token ${SCRATCH_TOKEN_ID} for the raw API legs (revoked in cleanup)"
else
  echo "WARN: could not mint a scratch token — raw API legs fall back to the configured token"
fi

gpus_json="$(api_get /api/gpus)"
GPU_COUNT="$(json_field "$gpus_json" "sum(1 for g in d if g.get('schedulable'))")"
if [ "$GPU_COUNT" -lt 1 ]; then
  skip "no schedulable GPU — vLLM is GPU-only (model.gpu_required); nothing to run here"
  say "Summary"
  echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
  echo "P11/P21 vLLM runbook: SKIPPED (no GPU)"
  exit 0
fi
MIN_VRAM_MB="$(json_field "$gpus_json" \
  "min((g['memory_mb'] for g in d if g.get('schedulable') and g.get('memory_mb')), default=0)")"
echo "${GPU_COUNT} schedulable GPU(s); smallest has ${MIN_VRAM_MB} MB VRAM."
note "the control plane is reachable, authenticated, and has a GPU to allocate"

# The daemon-global escape hatch suppresses the P21 auto-injection per flag —
# read it so leg 1b can SKIP honestly instead of failing on a tuned daemon.
EXTRA_ARGS="$(json_field "$(api_get /api/config/daemon/models 2>/dev/null || echo '{}')" \
  "json.dumps((d.get('values') or d).get('vllm_extra_args', []))" 2>/dev/null || echo '[]')"
echo "[models].vllm_extra_args = ${EXTRA_ARGS}"

# Start from a clean slate so a re-run measures a real serve, not an adoption.
cleanup_model "$MODEL_NAME"
cleanup_model "$OOM_NAME"

# --- 1. serve on an UNTUNED daemon ---------------------------------------------
say "1. nerdit serve ${MODEL_REF} --backend vllm --gpus 1 (no manual tuning)"
T0=$(date +%s)
# A re-run against an already-served name 409s — harmless, the wait gates below.
run_rc nerdit serve "$MODEL_REF" --backend vllm --gpus 1 --name "$MODEL_NAME" \
  || echo "(serve returned non-zero — already served? continuing to the wait loop)"
note "a model is a kind=model workload: POST /api/models, audited + idempotent"

say "1a. Wait for status=running AND model_pulled=true"
echo "(a first-ever HF weights download can take many minutes — excluded from any timing claim)"
outcome="$(wait_model "$MODEL_NAME" "$WAIT_S")"
T1=$(date +%s)
case "$outcome" in
  ready)
    ok "model ready in $(( T1 - T0 ))s on a default config (P21 D3: it fits without hand-tuning)" ;;
  failed)
    bad "model row went failed — inspect: nerdit diagnose ${MODEL_NAME}"
    api_get "/api/services/${MODEL_NAME}/diagnose" | sed 's/^/     /' || true ;;
  *)
    bad "timed out after ${WAIT_S}s waiting for ${MODEL_NAME}" ;;
esac
note "the P11 trigger — 'crash-loops to failed on an 8 GB card' — is fixed by default"

say "1b. The launch argv carries the VRAM-aware bounds (P21 D3)"
CMD_JSON="$(vllm_cmd "$MODEL_REF")"
if ! command -v docker >/dev/null; then
  skip "docker CLI absent — cannot read the container launch argv"
elif [ -z "$CMD_JSON" ]; then
  skip "no running vLLM container matched ${MODEL_REF} (model not up?)"
elif [ "$EXTRA_ARGS" != "[]" ]; then
  skip "[models].vllm_extra_args is set (${EXTRA_ARGS}) — it suppresses the matching injection, by design"
elif [ "$MIN_VRAM_MB" -le 0 ]; then
  skip "the daemon reports no VRAM for the allocated GPU — injection keys off KNOWN VRAM only"
elif [ "$MIN_VRAM_MB" -ge 16384 ]; then
  len_n="$(cmd_flag_count "$CMD_JSON" --max-model-len)"
  if [ "$len_n" = "0" ]; then
    ok "GPU has ${MIN_VRAM_MB} MB (>= 16 GB): nothing injected, vLLM defaults kept"
  else
    bad "GPU has ${MIN_VRAM_MB} MB but --max-model-len was injected ${len_n}x (threshold is < 16 GB)"
  fi
else
  len_v="$(cmd_flag_value "$CMD_JSON" --max-model-len)"
  util_v="$(cmd_flag_value "$CMD_JSON" --gpu-memory-utilization)"
  if [ "$len_v" = "4096" ] && [ "$util_v" = "0.85" ]; then
    ok "under 16 GB: injected --max-model-len 4096 --gpu-memory-utilization 0.85"
  else
    bad "expected the injected bounds (4096 / 0.85), argv had (${len_v:-none} / ${util_v:-none})"
  fi
fi
note "conservative bounds are injected only under 16 GB, only when the flag is absent"

# --- 2. a real chat completion --------------------------------------------------
say "2. POST <endpoint>/chat/completions — the OpenAI contract, served locally"
ENDPOINT="$(json_field "$(api_get /api/models)" \
  "next((i.get('endpoint') or '' for i in d['items'] if i['name'] == '${MODEL_NAME}'), '')")"
if [ -z "$ENDPOINT" ]; then
  bad "no endpoint published for ${MODEL_NAME}"
else
  echo "  endpoint: ${ENDPOINT}"
  chat_json="$(curl -sf -X POST -H 'Content-Type: application/json' \
    -d "{\"model\": \"${MODEL_REF}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hi in five words.\"}], \"max_tokens\": 32}" \
    "${ENDPOINT}/chat/completions" || true)"
  [ -n "$chat_json" ] || chat_json='{}'
  reply="$(json_field "$chat_json" \
    "((d.get('choices') or [{}])[0].get('message') or {}).get('content') or ''" 2>/dev/null || true)"
  if [ -n "$reply" ]; then
    ok "vLLM answered: $(printf '%s' "$reply" | tr '\n' ' ' | head -c 120)"
  else
    bad "no completion returned (raw: $(printf '%s' "$chat_json" | head -c 200))"
  fi
fi
note "Invariant #1: the app-facing contract is OpenAI, whatever backend is behind it"

# --- 3. per-serve typed override (P21 D4) ---------------------------------------
say "3. Re-serve with --max-model-len 2048 (per-serve override, no daemon restart)"
cleanup_model "$MODEL_NAME"
sleep 3
if run_rc nerdit serve "$MODEL_REF" --backend vllm --gpus 1 --name "$MODEL_NAME" \
     --max-model-len 2048; then
  ok "serve accepted the typed per-serve bound (no raw argv passthrough — D4)"
else
  bad "nerdit serve --max-model-len 2048 was rejected"
fi
outcome="$(wait_model "$MODEL_NAME" "$WAIT_S")"
[ "$outcome" = "ready" ] && ok "re-serve reached running + model_pulled" \
  || bad "re-serve outcome: ${outcome}"

CMD_JSON="$(vllm_cmd "$MODEL_REF")"
if ! command -v docker >/dev/null; then
  skip "docker CLI absent — cannot assert the launch argv"
elif [ -z "$CMD_JSON" ]; then
  skip "no running vLLM container matched ${MODEL_REF}"
else
  n="$(cmd_flag_count "$CMD_JSON" --max-model-len)"
  v="$(cmd_flag_value "$CMD_JSON" --max-model-len)"
  if [ "$n" = "1" ] && [ "$v" = "2048" ]; then
    ok "--max-model-len 2048 appears EXACTLY ONCE (override beats the injected default)"
  else
    bad "--max-model-len appeared ${n}x with value '${v}' (expected exactly 1x 2048)"
  fi
fi
# The override must be vLLM-only: the same field on an ollama serve is a 422.
bp_code="$(api_code /api/models -X POST -H 'Content-Type: application/json' \
  -H "Idempotency-Key: demo-p11-bp-$(date +%s)" \
  -d '{"model": "llama3.2:1b", "backend": "ollama", "gpus": 0, "max_model_len": 2048}')"
if [ "$bp_code" = "422" ]; then
  ok "max_model_len on an ollama serve refused with 422 (model.backend_param)"
else
  bad "ollama + max_model_len returned HTTP ${bp_code} (expected 422)"
fi
note "one model is tunable without a daemon restart, and only via typed, bounded fields"

# --- 4. forced GPU OOM → self-explanatory failure (P21 D1 + D2) ------------------
say "4. Forced GPU OOM → error.class=GPU_OOM + remediation.code=model.gpu_oom"
if [ "$CRASH_LEG" != "1" ]; then
  skip "CRASH_LEG!=1 — the destructive crash-loop leg is disabled (set CRASH_LEG=1 to opt in)"
else
  # A second engine claiming 0.95 of a card the first model already occupies
  # cannot fit: the allocator raises, the row crash-loops to failed.
  run_rc nerdit serve "$MODEL_REF" --backend vllm --gpus 1 --name "$OOM_NAME" \
    --gpu-memory-utilization 0.95 || true
  echo "  waiting for the restart budget to be exhausted (~4 attempts)..."
  outcome="$(wait_model "$OOM_NAME" "$WAIT_S")"
  if [ "$outcome" != "failed" ]; then
    skip "the OOM row settled '${outcome}' instead of failed — the card had room; crash leg inconclusive"
  else
    # log_tail=200 (the max): the root-cause line sits ~50 lines above vLLM's
    # closing APIServer traceback, outside the default 50-line window.
    diag="$(api_get "/api/services/${OOM_NAME}/diagnose?log_tail=200" 2>/dev/null || echo '{}')"
    ecls="$(json_field "$diag" "(d.get('error') or {}).get('class')")"
    rcode="$(json_field "$diag" "(d.get('remediation') or {}).get('code')")"
    gflag="$(json_field "$diag" "(d.get('forensics') or {}).get('gpu_oom', False)")"
    tail_hit="$(json_field "$diag" \
      "any(l.get('stream') == 'crash' and ('out of memory' in str(l.get('line', '')).lower() or 'gpu memory utilization' in str(l.get('line', '')).lower()) for l in (d.get('logs') or []))")"
    [ "$ecls" = "GPU_OOM" ] && ok "error.class=GPU_OOM (distinct from the cgroup OOM class)" \
      || bad "error.class=${ecls} (expected GPU_OOM)"
    [ "$gflag" = "True" ] && ok "forensics.gpu_oom=true" || bad "forensics.gpu_oom=${gflag}"
    [ "$rcode" = "model.gpu_oom" ] && ok "remediation.code=model.gpu_oom (names the GPU knobs, not memory_limit)" \
      || bad "remediation.code=${rcode} (expected model.gpu_oom)"
    [ "$tail_hit" = "True" ] && ok "the crash tail survived container removal (D1 capture)" \
      || bad "diagnose logs carried no CUDA OOM line — the capture did not land"
    # Negative contract (plan §2 leg 3): captured tails never reach audit params.
    audit_json="$(api_get "/api/audit?action=model.serve&limit=50" 2>/dev/null || echo '{"items": []}')"
    leaked="$(json_field "$audit_json" \
      "any('out of memory' in str(e.get('params_redacted') or '').lower() for e in d['items'])")"
    [ "$leaked" = "False" ] && ok "no captured log content in the audit params" \
      || bad "audit params contain crash-tail content"
  fi
fi
note "the 2026-07-12 dead end (empty log_tail, USER_ERROR, generic message) is gone"

# --- 5. cleanup -------------------------------------------------------------------
say "5. Cleanup"
if [ "$KEEP_MODEL" = "1" ]; then
  skip "KEEP_MODEL=1 — leaving ${MODEL_NAME}/${OOM_NAME} in place"
else
  for name in "$MODEL_NAME" "$OOM_NAME"; do
    if nerdit services rm "$name" >/dev/null 2>&1; then
      ok "model row ${name} removed (container stopped, endpoint released)"
    else
      skip "model row ${name} not present / already removed"
    fi
  done
  echo "  NOTE: the downloaded weights stay under <data_dir>/models (reused by later runs)."
fi
if [ -n "${SCRATCH_TOKEN_ID:-}" ]; then
  if curl -sf -X DELETE -H "Authorization: Bearer $(nerdit token)" \
       "${API}/api/tokens/${SCRATCH_TOKEN_ID}" >/dev/null 2>&1; then
    ok "scratch token ${SCRATCH_TOKEN_ID} revoked"
  else
    bad "could not revoke scratch token ${SCRATCH_TOKEN_ID} — revoke it manually"
  fi
fi

# --- summary ------------------------------------------------------------------------
say "Summary"
echo "PASS=${PASS_COUNT} FAIL=${FAIL_COUNT} SKIP=${SKIP_COUNT}"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "P11/P21 vLLM runbook: FAIL"
  exit 1
fi
echo "P11/P21 vLLM runbook: PASS"
