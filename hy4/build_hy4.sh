#!/bin/sh
# Build the DankAttn Hy4 Metal engine on the Mac Studio.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
out="${1:-build/dankattn-hy4}"
clang++ -std=c++20 -fobjc-arc -O2 -Wno-deprecated-declarations \
  engine/hy4/hy4.mm engine/hy4/main.mm \
  -framework Metal -framework Foundation -o "$out.new"
mv "$out.new" "$out"
echo "built $out"
