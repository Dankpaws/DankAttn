#!/bin/sh
# Build the DankAttn ERNIE-5.5 Metal engine on the Mac Studio.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
out="${1:-build/dankattn-ernie55}"
clang++ -std=c++20 -fobjc-arc -O2 -Wno-deprecated-declarations \
  engine/ernie55/ernie55.mm engine/ernie55/main.mm \
  -framework Metal -framework Foundation -o "$out.new"
mv "$out.new" "$out"
echo "built $out"
