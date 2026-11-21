#!/bin/sh
# Standing exactness gate on the real pack (about 2 minutes, no mini packs).
#   tests/gate.sh            run every check; exit 0 only if all pass
#   tests/gate.sh --bless    record the current logit hash as the golden one (after an intended numeric change)
# Extra environment (for example OPUS_HCFUSE=1) is passed through to the engine, so a flag can be gated as is.
# What each check covers:
#   vtest   verify/commit vs one token at a time, 4- and 16-row blocks (short context)
#   score   logits of 8,192 tokens in 16-row steps, hashed: block selection runs from 2k on; compared to golden
#   ptest   2,048-row prefill vs 16-row steps at 8,000 tokens: logits and continuation
#   spec    speculative vs plain decode at 8k context, greedy and sampled
#   ctest   resume from a checkpoint vs a fresh run, without and with speculation
cd "$(dirname "$0")/.." || exit 2
B=${OPUS_BIN:-./build/opus}
P=${OPUS_PACK:-$HOME/opus/pack}
D=tests/data
G=${OPUS_GATE_GOLDEN:-tests/golden_ids8k.sha256}  # kv8: tests/golden_ids8k_kv8.sha256
export OPUS_WARM=0
fail=0
check() {  # check <name> <pattern that must appear> <command...>
  name=$1; pat=$2; shift 2
  all=$("$@" 2>&1)
  out=$(echo "$all" | grep -E "^(vtest|ptest|ctest|spec ==)" | tail -1)
  if echo "$out" | grep -q "$pat" && ! echo "$out" | grep -Eq "DIFFER|MISMATCH"; then echo "ok    $name: $out"
  else echo "FAIL  $name: ${out:-$(echo "$all" | tail -1 | cut -c1-160)}"; fail=1; fi
}
OPUS_CLI_CAP=8192 check "vtest 4 rows" "64/64 bit-identical" $B $P vtest $D/ids300.bin 64 4
OPUS_CLI_CAP=8192 check "vtest 16 rows" "64/64 bit-identical" $B $P vtest $D/ids300.bin 64 16
h=$(OPUS_CLI_CAP=32768 $B $P score $D/ids8k.bin /dev/stdout 2>/tmp/opus_gate_score.err | shasum -a 256 | cut -d' ' -f1)
grep -q "^scored 8192 tokens" /tmp/opus_gate_score.err || { echo "FAIL  score run: $(tail -1 /tmp/opus_gate_score.err | cut -c1-160)"; fail=1; }
if [ "$1" = "--bless" ]; then echo "$h" > $G; echo "blessed score hash $h"; fi
if [ -f $G ] && [ "$h" = "$(cat $G)" ]; then echo "ok    score hash: $h"; else echo "FAIL  score hash: $h (golden $(cat $G 2>/dev/null || echo none))"; fail=1; fi
OPUS_CLI_CAP=32768 check "ptest 8000" "logits bit-identical, continuation identical" $B $P ptest $D/ids8k.bin 8000 8
OPUS_CLI_CAP=32768 check "spec greedy" "(identical)" $B $P spec $D/ids8k.bin 120
OPUS_CLI_CAP=32768 check "spec sampled" "(identical)" $B $P spec $D/ids8k.bin 120 1.0
OPUS_CLI_CAP=32768 check "ctest plain" "continuation identical" $B $P ctest $D/ids8k.bin 0
OPUS_CLI_CAP=32768 check "ctest spec" "continuation identical" $B $P ctest $D/ids8k.bin 1
[ $fail = 0 ] && echo "GATE PASSED" || echo "GATE FAILED"
exit $fail
