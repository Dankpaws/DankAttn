#!/bin/sh
# DankAttn ERNIE-5.5-Flash-Preview + OpenAI front end on loopback :8343.
#   ernie55_serve.sh start|stop|status|smoke
T=$(cd "$(dirname "$0")/.." && pwd)
if [ -n "${E5_PACK:-}" ]; then
  PACK="$E5_PACK"
elif [ -f "$T/models/pack-ernie55-w2-g32-w8attn/manifest.json" ]; then
  PACK="$T/models/pack-ernie55-w2-g32-w8attn"
else
  PACK="$HOME/pack-ernie55-w2-g32-w8attn"
fi
PORT=${E5_PORT:-8343}
CAP=${E5_CAP:-262144}
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$T/.venv/bin/python" ]; then
  PY="$T/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
LOG=$T/ernie55-server.log
pids() { pgrep -f "ernie55_server.py --pack" ; pgrep -f "build/dankattn-ernie55 .* serve"; }
case "$1" in
  start)
    if curl -sf -m 2 localhost:$PORT/health >/dev/null; then echo "already up"; exit 0; fi
    cd "$T" && nohup "$PY" server/ernie55_server.py \
      --pack "$PACK" --port $PORT --cap $CAP >> "$LOG" 2>&1 < /dev/null &
    for i in $(seq 1 120); do curl -sf -m 2 localhost:$PORT/health >/dev/null && { echo "ERNIE55_READY ${i}s"; exit 0; }; sleep 1; done
    echo "start failed; see $LOG"; exit 1 ;;
  stop)
    pkill -f "ernie55_server.py --pack"; sleep 2; pkill -f "build/dankattn-ernie55 .* serve"
    for i in $(seq 1 20); do [ -z "$(pids)" ] && { echo stopped; exit 0; }; sleep 1; done
    echo "still running: $(pids)"; exit 1 ;;
  status)
    curl -sf -m 2 localhost:$PORT/health >/dev/null && echo "UP :$PORT pids $(pids | tr '\n' ' ')" || { echo DOWN; exit 1; } ;;
  smoke)
    if [ -n "${DANKATTN_API_KEY:-}" ]; then
      set -- -H "Authorization: Bearer $DANKATTN_API_KEY"
    else
      set --
    fi
    curl -s -m 120 localhost:$PORT/v1/chat/completions -H 'content-type: application/json' "$@" \
      -d '{"messages":[{"role":"user","content":"What is 19*17? Reply with the number only."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      | "$PY" -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"], d["usage"]["dankattn"])' ;;
  *) echo "usage: $0 start|stop|status|smoke"; exit 64 ;;
esac
