#!/bin/sh
# Push the source tree to a remote Mac host and optionally run a command there.
# Usage: DANKATTN_HOST=user@host [DANKATTN_DIR=dankattn] [DANKATTN_SSH_OPTS="..."] sh sync.sh [cmd...]
set -e
: "${DANKATTN_HOST:?set DANKATTN_HOST=user@host}"
DIR="${DANKATTN_DIR:-dankattn}"
S="ssh ${DANKATTN_SSH_OPTS:-} -o ConnectTimeout=10"
rsync -a -e "$S" --exclude .git --exclude ref --exclude 'build/' --exclude '*.bin' --exclude '__pycache__/' \
  "$(dirname "$0")/" "$DANKATTN_HOST:$DIR/"
[ $# -gt 0 ] && $S "$DANKATTN_HOST" "cd ~/$DIR && $*"
