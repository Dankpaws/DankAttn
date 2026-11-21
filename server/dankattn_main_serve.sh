#!/bin/sh
# Production profile: this tree's engine behind the OpenAI front end on :8340.
#   server/dankattn_main_serve.sh start|stop|status|smoke
# Defaults: pool mode, 1 conversation slot of 200000 tokens, 6 lanes of 32768 for subagents/fanouts, K/V caches int8 (OPUS_KV8),
# slot checkpoints: 4 newest + one per 16384 tokens.
# Any OPUS_* value in the environment overrides its default. Log: <tree>/server.log
T=$(cd "$(dirname "$0")/.." && pwd)
PACK=${DANKATTN_PACK:-${OPUS_PACK:-$T/pack}}
PORT=${DANKATTN_PORT:-8340}
export OPUS_CAP=${OPUS_CAP:-200000}
export OPUS_SLOTS=${OPUS_SLOTS:-1}
export OPUS_SLOT_CAP=${OPUS_SLOT_CAP:-98304}
export OPUS_LANES=${OPUS_LANES:-6}
export OPUS_LANE_CAP=${OPUS_LANE_CAP:-32768}
export OPUS_POOL=${OPUS_POOL:-1}
export OPUS_KV8=${OPUS_KV8:-1}
# Tolerant acceptance (2026-11-15): a draft within this many logits of the target's top token is
# accepted. Greedy paths only (slot and lanes); a stop token is never tolerated. 10 prompts x 200
# tokens: tok/round 3.44 -> 3.65, wall 126.9 -> 133.0 tok/s, teacher-forced NLL 0.8614 -> 0.8613.
export OPUS_TOL=${OPUS_TOL:-0.5}
export OPUS_CKPT_KEEP=${OPUS_CKPT_KEEP:-4}
export OPUS_CKPT_FAR=${OPUS_CKPT_FAR:-16384}
PY=${PYTHON:-python3}
LOG=$T/server.log
case "$1" in
  start)
    if curl -sf -m 2 localhost:$PORT/v1/models >/dev/null; then echo "already up"; exit 0; fi
    [ -x "$T/build/dankattn" ] || { echo "no engine at $T/build/dankattn (run build.sh)"; exit 1; }
    [ -e "$T/src" ] || ln -s . "$T/src"  # the front end runs the engine with cwd <tree>/src
    # the subshell's own output goes to the log (exec), so an ssh caller is not held open by the server
    (cd "$T" && exec nohup "$PY" server/dankattn_server.py \
      --pack "$PACK" --engine "$T/build/dankattn" --port $PORT --cap "$OPUS_CAP") >> "$LOG" 2>&1 < /dev/null &
    # /v1/models answers 200 only once the engine reports ready (503 while it loads)
    for i in $(seq 1 600); do curl -sf -m 2 localhost:$PORT/v1/models >/dev/null && { echo "DANKATTN_MAIN_READY ${i}s"; exit 0; }; sleep 1; done
    echo "start failed; see $LOG"; exit 1 ;;
  status)
    # the front end restarts a dead engine; /health says whether the engine itself is serving
    h=$(curl -s -m 2 localhost:$PORT/health) || { echo DOWN; exit 1; }
    echo "$h" | grep -q '"state":"up"' && echo "UP :$PORT $h" || { echo "NOT SERVING :$PORT $h"; exit 1; } ;;
  *) exec sh "$T/server/dankattn_serve.sh" "$@" ;;
esac
