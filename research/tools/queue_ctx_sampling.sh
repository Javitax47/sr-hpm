#!/usr/bin/env bash
# Unattended test queue (30 oct. 2026): sampling / probabilistic verification, long context with the KV cache in RAM,
# KV-cache quantization quality. Runs detached from the Claude session; every result is appended to
# ~/srhpm_runs/summary.txt and the queue ends by writing ~/srhpm_runs/DONE.
#   phase A  decode, short prompts: greedy vs temperature 1, draft sampling greedy vs probabilistic, chat-like
#   phase B  decode, short prompts, with large context windows (32k f16 on GPU, 128k / 262k q8 in RAM)
#   phase E  KLD of q8_0 vs f16 KV cache
#   phase C  prompt processing + decode at 8k / 32k / 64k / 120k tokens of context (128k window, q8 in RAM)
#   phase D  one ~240k-token prompt (262k window, q8 in RAM)
set -u
PUB=/c/Users/Javier/Desktop/SR-HPM/publish
LAB=/c/Users/Javier/Desktop/SR-HPM/f3
OUT=$HOME/srhpm_runs
PY=$HOME/.venvs/f0/Scripts/python.exe
MODEL=$HOME/f0_gguf/q_speed1.gguf
DRAFT=$HOME/f0_gguf/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
TEXT=$HOME/f0_corpus_full.txt
PORT=8090
export LIMIT_GB=48
mkdir -p "$OUT"
rm -f "$OUT/DONE"
S=$OUT/summary.txt
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$S"; }

stop_server() {
  taskkill //F //IM llama-server.exe >/dev/null 2>&1
  for _ in $(seq 30); do tasklist | grep -qi llama-server || break; sleep 2; done
}

# serve <label> <launcher args...>: start llama-server through the launcher (commit guard on), wait until healthy
serve() {
  local label=$1; shift
  stop_server
  ( cd "$PUB" && bash "$LAB/guard.sh" "$PY" -m srhpm.launch "$MODEL" --draft "$DRAFT" --port $PORT --run "$@" \
      > "$OUT/server_$label.log" 2>&1 ) &
  for _ in $(seq 600); do curl -s 127.0.0.1:$PORT/health | grep -q ok && return 0; sleep 2; done
  log "[$label] server did not become healthy"; return 1
}

server_notes() {  # VRAM / large pages / acceptance / pass times of the run, for the summary
  local f=$OUT/server_$1.log
  grep -a "srv_dgpu: VRAM free" "$f" | head -1 | sed "s/^/   [$1] /" >> "$S"
  grep -a "large pages: .*MiB" "$f" | tr '\n' ';' | sed "s/^/   [$1] /" >> "$S"; echo >> "$S"
  grep -a "draft acceptance" "$f" | tail -1 | sed "s/.*draft acceptance/   [$1] last acceptance/" >> "$S"
  grep -a "out of memory\|GGML_ASSERT\|failed to allocate" "$f" | head -2 | sed "s/^/   [$1] ERROR /" >> "$S"
}

# dec <label> <dec_bench args> -- <launcher args>
dec() {
  local label=$1; shift
  local bargs=() ; while [ $# -gt 0 ] && [ "$1" != "--" ]; do bargs+=("$1"); shift; done; shift
  if serve "$label" "$@"; then
    ( cd "$PUB" && "$PY" -m bench.dec_bench --port $PORT --n 10 --max-tokens 200 --label "$label" "${bargs[@]}" 2>&1 ) \
      | grep -a "\[dec\]\|Error" | tee -a "$S"
    server_notes "$label"
  fi
  stop_server
}

# ctxrun <label> <lengths> <launcher args>
ctxrun() {
  local label=$1 lengths=$2; shift 2
  if serve "$label" "$@"; then
    ( cd "$PUB" && "$PY" -m bench.ctx_bench --port $PORT --text "$TEXT" --lengths "$lengths" --max-tokens 200 \
        --out "$OUT/ctx_$label.json" 2>&1 ) | grep -a "\[ctx\]\|Error" | sed "s/^/[$label] /" | tee -a "$S"
    server_notes "$label"
  fi
  stop_server
}

log "==== queue start"
# keep the machine awake while the queue runs (a runtime request, like a media player's; no setting is changed)
powershell -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$LAB/keepawake.ps1")" -Until "$(cygpath -w "$OUT/DONE")" &

log "==== phase A: sampling"
dec A_t0_greedy             --                       # reference: the launcher's default, greedy requests
dec A_t1_greedy  --temp 1.0 --                       # temperature 1, top-k 20 / top-p 0.95 (the web UI's values)
dec A_t1_prob7   --temp 1.0 -- --draft-sampling probabilistic
dec A_t1_prob3   --temp 1.0 -- --draft-sampling probabilistic --draft-max 3
dec A_t0_prob7              -- --draft-sampling probabilistic
dec A_t1_think   --temp 1.0 --think --                # chat-like: thinking on, temperature 1
dec A_t1_think_prob --temp 1.0 --think -- --draft-sampling probabilistic
dec A_t0_greedy_b           --                       # repeat of the reference (drift check)

log "==== phase B: large context windows, short prompts"
dec B_ctx32k_f16   -- --ctx 32768
dec B_ctx128k_q8ram -- --ctx 131072 --kv-type q8_0 --kv-ram
dec B_ctx262k_q8ram -- --ctx 262144 --kv-type q8_0 --kv-ram
dec B_ctx8k_q8ram  -- --kv-type q8_0 --kv-ram          # cost of the KV in RAM alone, at the default window
dec B_ctx8k_ref    --                                 # reference again

log "==== phase E: KV cache quantization quality (KLD vs Q8_0 weights + f16 KV)"
stop_server
for kv in f16 q8_0; do
  "$HOME/src/llama.cpp/build/bin/llama-perplexity.exe" -m "$MODEL" -f "$TEXT" -c 512 --chunks 16 -fa on -ngl 21 \
      -ctk $kv -ctv $kv --kl-divergence-base "$HOME/kld_base_q8.bin" --kl-divergence > "$OUT/kld_kv_$kv.log" 2>&1
  log "[E_kv_$kv] $(grep -a 'Mean    KLD' "$OUT/kld_kv_$kv.log")  $(grep -a 'Same top p' "$OUT/kld_kv_$kv.log")"
done

log "==== phase C: long contexts (128k window, q8 KV in RAM)"
ctxrun C_128k 8000,32000,64000,110000 --ctx 131072 --kv-type q8_0 --kv-ram

log "==== phase D: one ~240k-token prompt (262k window, q8 KV in RAM)"
ctxrun D_262k 220000 --ctx 262144 --kv-type q8_0 --kv-ram

log "==== queue done"
touch "$OUT/DONE"
