#!/bin/sh
# Unified build script for DankAttn Metal 4 engines on Apple Silicon.
# Usage:
#   sh build.sh [all|hy4|inkling2|ernie55|flash|bench] [output_path] [extra clang flags...]
#   sh build.sh build/dankattn [extra clang flags...]   (legacy Flash-Next output path)
set -e
cd "$(dirname "$0")"

CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++20 -fobjc-arc -O2 -Wno-deprecated-declarations"
FRAMEWORKS="-framework Metal -framework Foundation"

build_one() {
  out="$1"
  shift
  mkdir -p "$(dirname "$out")"
  $CXX $CXXFLAGS "$@" -o "$out.new"
  mv "$out.new" "$out"
  echo "built $out"
}

target="${1:-all}"
[ $# -gt 0 ] && shift

# Legacy compatibility: `sh build.sh build/dankattn [flags...]`
case "$target" in
  */*)
    build_one "$target" "$@" engine/model.mm engine/ngram.mm engine/main.mm $FRAMEWORKS
    exit 0
    ;;
esac

out=""
if [ $# -gt 0 ]; then
  case "$1" in
    -*) ;;
    *) out="$1"; shift ;;
  esac
fi

case "$target" in
  all)
    build_one "build/dankattn-hy4"      "$@" engine/hy4/hy4.mm engine/hy4/main.mm $FRAMEWORKS
    build_one "build/dankattn-inkling2" "$@" engine/inkling2/inkling2.mm engine/inkling2/main.mm $FRAMEWORKS -framework IOKit
    build_one "build/dankattn-ernie55"  "$@" engine/ernie55/ernie55.mm engine/ernie55/main.mm $FRAMEWORKS
    build_one "build/dankattn"          "$@" engine/model.mm engine/ngram.mm engine/main.mm $FRAMEWORKS
    ;;
  hy4)
    build_one "${out:-build/dankattn-hy4}" "$@" engine/hy4/hy4.mm engine/hy4/main.mm $FRAMEWORKS
    ;;
  inkling2)
    build_one "${out:-build/dankattn-inkling2}" "$@" engine/inkling2/inkling2.mm engine/inkling2/main.mm $FRAMEWORKS -framework IOKit
    ;;
  ernie55)
    build_one "${out:-build/dankattn-ernie55}" "$@" engine/ernie55/ernie55.mm engine/ernie55/main.mm $FRAMEWORKS
    ;;
  flash|opus)
    build_one "${out:-build/dankattn}" "$@" engine/model.mm engine/ngram.mm engine/main.mm $FRAMEWORKS
    ;;
  bench)
    build_one "build/quant_test"    "$@" bench/quant_test.mm $FRAMEWORKS
    build_one "build/kv_radix_test" "$@" bench/kv_radix_test.mm $FRAMEWORKS
    build_one "build/mtl4_floor"    "$@" bench/mtl4_floor.mm $FRAMEWORKS
    build_one "build/sample_test"   "$@" bench/sample_test.mm $FRAMEWORKS
    build_one "build/moe_bw"        "$@" bench/moe_bw.mm $FRAMEWORKS
    ;;
  *)
    echo "usage: sh build.sh [all|hy4|inkling2|ernie55|flash|bench] [out] [extra clang flags...]" >&2
    exit 2
    ;;
esac
