#!/bin/sh
# Exactness suite on the mini packs: verify/commit, prefill chunking, speculation, checkpoints.
# Run on the Mac from ~/opus/src after build.sh. Any line without identical/bit-identical fails.
B=../build/opus
for p in ~/opus/mini4f-pack ~/opus/mini4mf-pack; do
  echo "== $p"
  $B $p vtest ~/opus/ids300.bin 64 4 2>&1 | tail -1
  $B $p vtest ~/opus/ids300.bin 64 16 2>&1 | tail -1
  $B $p ptest ~/opus/ids2200.bin 2200 8 2>&1 | tail -1
  $B $p spec ~/opus/ids300.bin 64 2>&1 | grep '^spec =='
  $B $p spec ~/opus/ids300.bin 64 1.0 2>&1 | grep '^spec =='
  $B $p ctest ~/opus/ids8k.bin 1 2>&1 | tail -1
done
