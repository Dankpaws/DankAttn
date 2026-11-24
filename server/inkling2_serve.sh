#!/bin/sh
# DankAttn Inkling-2-Mini + OpenAI front end on loopback (default :8342).
#   inkling2_serve.sh start|stop|status|smoke
# Env: INK_PACK (default ~/pack-inkling2-mini-w8), INK_PORT, INK_CAP (per lane, default 32768), INK_LANES (4),
#      INK_RING_SLACK (extra local-ring rows for rewinding past a reply, default 4096), PYTHON.
T=$(cd "$(dirname "$0")/.." && pwd)
PACK=${INK_PACK:-$HOME/pack-inkling2-mini-w8}
PORT=${INK_PORT:-8342}
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$T/.venv/bin/python" ]; then PY="$T/.venv/bin/python"
else PY=python3; fi
LOG=$T/inkling2-server.log
export INK_CAP=${INK_CAP:-32768} INK_LANES=${INK_LANES:-4} INK_RING_SLACK=${INK_RING_SLACK:-4096}
pids() { pgrep -f "inkling2_server.py --pack" ; pgrep -f "build/dankattn-inkling2 .* serve"; }
case "$1" in
  start)
    if curl -s -m 2 localhost:$PORT/v1/models >/dev/null; then echo "already up"; exit 0; fi
    cd "$T" && nohup "$PY" server/inkling2_server.py --pack "$PACK" --port $PORT >> "$LOG" 2>&1 < /dev/null &
    for i in $(seq 1 120); do curl -s -m 2 localhost:$PORT/v1/models >/dev/null && { echo "INKLING2_READY ${i}s"; exit 0; }; sleep 1; done
    echo "start failed; see $LOG"; exit 1 ;;
  stop)
    pkill -f "inkling2_server.py --pack"; sleep 2; pkill -f "build/dankattn-inkling2 .* serve"
    for i in $(seq 1 20); do [ -z "$(pids)" ] && { echo stopped; exit 0; }; sleep 1; done
    echo "still running: $(pids)"; exit 1 ;;
  status)
    curl -s -m 2 localhost:$PORT/v1/models >/dev/null && echo "UP :$PORT pids $(pids | tr '\n' ' ')" || echo DOWN ;;
  smoke)
    curl -s -m 300 localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"What is 19*17? Reply with the number only."}],"max_tokens":2000,"reasoning_effort":"low"}' \
      | "$PY" -c 'import sys,json;d=json.load(sys.stdin);m=d["choices"][0]["message"];print(repr(m.get("content")), "| reasoning", len(m.get("reasoning_content") or ""), "chars |", d["usage"]["dankattn"])' ;;
  *) echo "usage: $0 start|stop|status|smoke"; exit 64 ;;
esac
