#!/bin/sh
# Build the DankAttn (Flash-Next) engine on the Mac (CLT clang is enough; kernels compile at runtime).
# Usage: sh build.sh [output_path] [extra clang flags...]
set -e
cd "$(dirname "$0")"
out="${1:-build/dankattn}"
[ $# -gt 0 ] && shift
mkdir -p "$(dirname "$out")"
clang++ -std=c++20 -fobjc-arc -O2 -Wno-deprecated-declarations "$@" \
  engine/model.mm engine/ngram.mm engine/main.mm \
  -framework Metal -framework Foundation -o "$out.new"
mv "$out.new" "$out"
echo "built $out"
