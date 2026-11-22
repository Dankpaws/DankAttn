#!/bin/sh
# Pull the latest DankAttn changes and rebuild any installed engine binaries.
# Usage: sh update.sh [inkling2|hy4|all]
set -e
cd "$(dirname "$0")"

before="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
git pull --ff-only
after="$(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD 2>/dev/null || echo unknown)"

target="${1:-auto}"
built=0

if [ "$target" = "inkling2" ] || [ "$target" = "all" ] || { [ "$target" = "auto" ] && [ -x build/dankattn-inkling2 ]; }; then
  sh inkling2/build_inkling2.sh
  built=1
fi
if [ "$target" = "hy4" ] || [ "$target" = "all" ] || { [ "$target" = "auto" ] && [ -x build/dankattn-hy4 ]; }; then
  sh hy4/build_hy4.sh
  built=1
fi
if [ "$target" = "auto" ] && [ -x build/dankattn ]; then
  sh build.sh
  built=1
fi

if [ "$built" -eq 0 ]; then
  echo "No existing binaries found in build/. Build one with:"
  echo "  sh inkling2/build_inkling2.sh   # Inkling-2-Mini"
  echo "  sh hy4/build_hy4.sh             # Hy4-flash"
fi

echo "DankAttn updated: $before -> $after"
echo "If a server is running, restart it to load the new binary (existing model packs do not need to be re-downloaded unless noted in the release)."
