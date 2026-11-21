#!/bin/sh
# accrun.sh [extra env assignments...] : spec on three workloads with acceptance statistics
cd "$(dirname "$0")/.." || exit 2
one() {  # one <ids file> <tokens> <label>
  env OPUS_CLI_CAP=32768 OPUS_WARM=0 OPUS_CONFLOG=1 OPUS_TRACE=1 $EXTRA ./build/opus ${OPUS_PACK:-$HOME/opus/pack} spec "$1" "$2" > /tmp/acc_$3.out 2> /tmp/acc_$3.err
  echo "=== $3 ($2 tokens) $EXTRA"
  grep -E "^spec ==|^wall" /tmp/acc_$3.out | cut -c1-120
  [ -n "$QUIET" ] || python3 tools/accstat.py /tmp/acc_$3.err
}
EXTRA="$*"
one $HOME/opus/chat_code.bin 500 code
one $HOME/opus/chat_fact.bin 500 prose
one tests/data/ids8k.bin 300 doc8k
