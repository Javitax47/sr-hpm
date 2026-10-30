#!/usr/bin/env bash
# Decode benchmark of the launcher's configuration: starts llama-server through srhpm.launch, runs the 10-prompt
# decode set (bench/dec_bench.py) and stops it. Extra launcher arguments after the label; environment variables set
# before the call reach the server (e.g. GGML_LARGE_PAGES=0 bash bench/bench.sh nolp).
#
#   MODEL=model.gguf DRAFT=drafter.gguf bash bench/bench.sh <label> [launcher args...]
#
# Compare configurations in pairs within one session (A B A B): laptops drift by +-10 % with temperature.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
: "${MODEL:?set MODEL=<model.gguf>}"
LABEL="${1:-run}"; shift || true
PY="${PYTHON:-python}"
PORT="${PORT:-8090}"
DRAFT_ARG=()
[ -n "${DRAFT:-}" ] && DRAFT_ARG=(--draft "$DRAFT")
LOG="${LOGDIR:-$REPO/bench}/bench_$LABEL.log"
"$PY" -m srhpm.launch "$MODEL" "${DRAFT_ARG[@]}" --port "$PORT" --run "$@" > "$LOG" 2>&1 &
PID=$!
for _ in $(seq 300); do curl -s "127.0.0.1:$PORT/health" | grep -q ok && break; sleep 2; done
"$PY" -m bench.dec_bench --port "$PORT" --n "${N:-10}" --max-tokens "${MT:-200}" --temp "${TEMP:-0}" --label "$LABEL"
kill $PID 2>/dev/null
pkill -f "llama-server.*--port $PORT" 2>/dev/null || taskkill //F //IM llama-server.exe >/dev/null 2>&1
wait $PID 2>/dev/null
