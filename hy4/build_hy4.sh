#!/bin/sh
# Build the DankAttn Hy4 Metal engine on Apple Silicon.
set -e
exec sh "$(dirname "$0")/../build.sh" hy4 "$@"
