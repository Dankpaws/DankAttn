#!/bin/sh
# DankAttn (Flash-Next) + OpenAI front end on loopback :8340.
#   dankattn_serve.sh start|stop|status|smoke
T=$(cd "$(dirname "$0")/.." && pwd)
PACK=${DANKATTN_PACK:-${OPUS_PACK:-$T/pack}}
PORT=${DANKATTN_PORT:-${OPUS_PORT:-8340}}
CAP=${DANKATTN_CAP:-${OPUS_CAP:-131072}}
PY=${PYTHON:-python3}
LOG=$T/server.log
pids() { pgrep -f "dankattn_server.py --pack" ; pgrep -f "build/dankattn .* serve"; }
case "$1" in
  start)
    if curl -s -m 2 localhost:$PORT/v1/models >/dev/null; then echo "already up"; exit 0; fi
    cd "$T" && nohup "$PY" server/dankattn_server.py \
      --pack "$PACK" --port $PORT --cap $CAP >> "$LOG" 2>&1 < /dev/null &
    for i in $(seq 1 120); do curl -s -m 2 localhost:$PORT/v1/models >/dev/null && { echo "DANKATTN_READY ${i}s"; exit 0; }; sleep 1; done
    echo "start failed; see $LOG"; exit 1 ;;
  stop)
    pkill -f "dankattn_server.py --pack"; sleep 2; pkill -f "build/dankattn .* serve"
    for i in $(seq 1 20); do [ -z "$(pids)" ] && { echo stopped; exit 0; }; sleep 1; done
    echo "still running: $(pids)"; exit 1 ;;
  status)
    curl -s -m 2 localhost:$PORT/v1/models >/dev/null && echo "UP :$PORT pids $(pids | tr '\n' ' ')" || echo DOWN ;;
  smoke)
    curl -s -m 120 localhost:$PORT/v1/chat/completions -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"What is 19*17? Reply with the number only."}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      | "$PY" -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"], d["usage"]["dankattn"])' ;;
  *) echo "usage: $0 start|stop|status|smoke"; exit 64 ;;
esac
