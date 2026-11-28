#!/bin/sh
# Build the DankAttn Inkling-2-Mini Metal 4 engine on Apple Silicon.
set -e
exec sh "$(dirname "$0")/../build.sh" inkling2 "$@"
