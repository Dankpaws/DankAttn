#!/bin/sh
# One-command regression gate for DankAttn Hy4 on the Mac Studio.
# Stops :8341, runs vtest + ntest + ctest + jtest + bench 12k + spec, and restarts :8341 + smoke.
set -e
T=$(cd "$(dirname "$0")/.." && pwd)
BIN=${1:-$T/build/dankattn-hy4}
PACK=${2:-$T/pack-g64-gptq-v3}
DEV=${HY4_DEV:-$HOME/hy4-dev}

sh "$T/hy4/serve_hy4.sh" stop
sleep 2
echo "=== [1/7] vtest ==="
"$BIN" "$PACK" vtest "$DEV/ids3k.bin"
echo "=== [2/7] ntest 4 ==="
"$BIN" "$PACK" ntest "$DEV/ids3k.bin" 4
echo "=== [3/7] ctest ==="
"$BIN" "$PACK" ctest "$DEV/ids3k.bin"
echo "=== [4/7] jtest 4 chat ==="
"$BIN" "$PACK" jtest "$DEV/chat_code_no_think.bin" 200 "$DEV/chat_doc_no_think.bin" "$DEV/chat_code_high.bin" "$DEV/chat_doc_high.bin"
echo "=== [5/7] bench 12k (1,5,8,16) ==="
"$BIN" "$PACK" bench "$DEV/ids12k.bin" 1,5,8,16
echo "=== [6/7] spec (ids12k, chat_code, edit_spec) ==="
"$BIN" "$PACK" spec "$DEV/ids12k.bin" 400 2>&1 | grep "spec:"
"$BIN" "$PACK" spec "$DEV/chat_code_no_think.bin" 400 2>&1 | grep "spec:"
"$BIN" "$PACK" spec "$DEV/edit_spec.bin" 400 2>&1 | grep "spec:"
echo "=== [7/7] restart :8341 + smoke ==="
sh "$T/hy4/serve_hy4.sh" start
sh "$T/hy4/serve_hy4.sh" smoke
