#!/bin/sh
# hy4/serve_hy4.sh start|stop|status|smoke -- DankAttn (Hy4) behind hy4_server.py on :8341.
# Env: HY4_PACK (default <tree>/pack-v4-h128-w6), HY4_PORT (8341), HY4_CAP (524288), HY4_MP (1024),
# HY4_SIDE_LANES (3, extra concurrent lanes), HY4_SIDE_CAP (32768, each side lane), HY4_MIN_P (0.02),
# HY4_DRAFT_CUT (0.2), PYTHON (default python3). Engine defaults: HY4_LOOKUP (1), HY4_DBLOCK (8),
# HY4_PCACHE (<tree>/pcache, 0 = off), HY4_PCACHE_GB (40), HY4_PCACHE_FREE_GB (15).
T=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "${HY4_PACK:-}" ]; then PACK="$HY4_PACK"
elif [ -d "$T/models/pack-dfly2b6" ]; then PACK="$T/models/pack-dfly2b6"
elif [ -d "$T/models/pack-v4-h128-w6" ]; then PACK="$T/models/pack-v4-h128-w6"
else PACK="$T/pack-v4-h128-w6"; fi
export HY4_SIDE_LANES=${HY4_SIDE_LANES:-3}
case "$PACK" in
  *dfly2b6*)
    export HY4_SIDE_CAP=${HY4_SIDE_CAP:-16384}
    export HY4_CAP=${HY4_CAP:-393216} ;;
  *)
    export HY4_SIDE_CAP=${HY4_SIDE_CAP:-32768}
    export HY4_CAP=${HY4_CAP:-524288} ;;
esac
PORT=${HY4_PORT:-8341}
export HY4_MP=${HY4_MP:-1024}
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$T/.venv/bin/python" ]; then PY="$T/.venv/bin/python"
else PY=python3; fi
LOG=$T/hy4-server.log
case "$1" in
  start)
    if curl -sf -m 2 localhost:$PORT/health >/dev/null; then echo "already up"; exit 0; fi
    [ -x "$T/build/dankattn-hy4" ] || { echo "no engine at $T/build/dankattn-hy4 (run hy4/build_hy4.sh)"; exit 1; }
    (cd "$T" && exec nohup "$PY" server/hy4_server.py \
      --pack "$PACK" --engine "$T/build/dankattn-hy4" --port $PORT --cap "$HY4_CAP") >> "$LOG" 2>&1 < /dev/null &
    for i in $(seq 1 600); do curl -sf -m 2 localhost:$PORT/health >/dev/null && { echo "HY4_READY ${i}s"; exit 0; }; sleep 1; done
    echo "start failed; see $LOG"; exit 1 ;;
  stop)
    p=$(lsof -ti tcp:$PORT -sTCP:LISTEN)
    [ -n "$p" ] && kill $p && echo "stopped $p" || echo "not running"
    for i in $(seq 1 30); do pgrep -f "build/dankattn-hy4 .* serve" >/dev/null || exit 0; sleep 1; done ;;
  status)
    h=$(curl -s -m 2 localhost:$PORT/health) || { echo DOWN; exit 1; }
    echo "$h" | grep -qE '"state":"(up|sleeping)"' && echo "UP :$PORT $h" || { echo "NOT SERVING :$PORT $h"; exit 1; } ;;
  smoke)
    if [ -n "${DANKATTN_API_KEY:-}" ]; then
      set -- -H "Authorization: Bearer $DANKATTN_API_KEY"
    else
      set --
    fi
    curl -s -m 600 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' "$@" \
      -d '{"model":"hy4-flash","messages":[{"role":"user","content":"What is 19*17? Answer with the number only."}],"max_tokens":400,"temperature":0}' \
      | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"].get("content"), d.get("usage",{}).get("dankattn"))' ;;
  *) echo "usage: $0 start|stop|status|smoke"; exit 2 ;;
esac
