#!/bin/sh
# Build the DankAttn Inkling-2-Mini Metal 4 engine on Apple Silicon.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
out="${1:-build/dankattn-inkling2}"
clang++ -std=c++20 -fobjc-arc -O2 -Wno-deprecated-declarations \
  engine/inkling2/inkling2.mm engine/inkling2/main.mm \
  -framework Metal -framework Foundation -o "$out.new"
mv "$out.new" "$out"
echo "built $out"
