#!/bin/sh
# Build the DankAttn ERNIE-5.5 Metal engine on Apple Silicon.
set -e
exec sh "$(dirname "$0")/../build.sh" ernie55 "$@"
