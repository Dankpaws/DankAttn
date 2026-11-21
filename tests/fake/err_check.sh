#!/bin/sh
# err_check.sh <server.py> [pack_dir]: run the front end on :8399 against the fake engine and exercise every error path
S=$1
PACK=${2:-${OPUS_PACK:-$HOME/opus/pack}}
PY=${PYTHON:-python3}
mkdir -p /tmp/hcb/fake/bin /tmp/hcb/fake/src
cp "$(dirname "$0")/fake_engine.py" /tmp/hcb/fake/bin/opus && chmod +x /tmp/hcb/fake/bin/opus
"$PY" "$S" --pack "$PACK" \
  --engine /tmp/hcb/fake/bin/opus --port 8399 --cap 4096 > /tmp/hcb/fake/server.log 2>&1 < /dev/null &
fe=$!
for i in $(seq 1 60); do curl -sf -m 1 localhost:8399/v1/models >/dev/null && break; sleep 0.5; done
big=$(python3 -c 'import json;print(json.dumps({"messages":[{"role":"user","content":" ".join("w%d"%i for i in range(3000))}],"max_tokens":20}))')
req() {  # req <label> <json>
  printf '%-34s ' "$1"
  curl -s -m 20 -o /tmp/hcb/fake/body -w '%{http_code} ' localhost:8399/v1/chat/completions -H 'content-type: application/json' -d "$2"
  tr '\n' ' ' < /tmp/hcb/fake/body | cut -c1-200; echo
}
M='"messages":[{"role":"user","content":"hi"}]'
req "plain ok"                       "{$M,\"max_tokens\":20}"
req "stream ok"                      "{$M,\"max_tokens\":20,\"stream\":true}"
req "oversize (non-stream)"          "$big"
req "oversize (stream)"              "$(echo "$big" | sed 's/}$/,"stream":true}/')"
req "group refused (stream)"         "{$M,\"max_tokens\":20,\"stream\":true,\"start_group\":{\"id\":\"g\",\"count\":2,\"index\":0}}"
req "group refused (non-stream)"     "{$M,\"max_tokens\":20,\"start_group\":{\"id\":\"g2\",\"count\":2,\"index\":0}}"
req "mid-stream failure (stream)"    "{$M,\"max_tokens\":7,\"stream\":true}"
req "mid-stream failure (non-stream)" "{$M,\"max_tokens\":7}"
req "capacity cut -> finish length"  "{$M,\"max_tokens\":3}"
req "bad max_tokens -> 400"          "{$M,\"max_tokens\":-5}"
req "bad top_p -> 400"               "{$M,\"top_p\":1.5}"
req "negative seed ok"               "{$M,\"max_tokens\":20,\"seed\":-1}"
req "no messages -> 400"             "{\"max_tokens\":20}"
req "engine crash -> 503"            "{$M,\"max_tokens\":9}"
printf '%-34s ' "health right after";  curl -s localhost:8399/health; echo
for i in $(seq 1 20); do curl -sf -m 1 localhost:8399/v1/models >/dev/null && break; sleep 0.5; done
printf '%-34s ' "health after restart"; curl -s localhost:8399/health; echo
req "plain ok after restart"         "{$M,\"max_tokens\":20}"
printf '%-34s ' "load after"; curl -s localhost:8399/v1/load | grep -o '"running":[0-9]*'
kill $fe; wait $fe 2>/dev/null
grep -iE "Traceback|Error" /tmp/hcb/fake/server.log | head -5
