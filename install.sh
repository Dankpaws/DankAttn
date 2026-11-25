#!/bin/sh
# DankAttn one-liner installer for Apple Silicon macOS.
#
# Remote install:
#   curl -fsSL https://raw.githubusercontent.com/Dankpaws/DankAttn/main/install.sh | sh
#
# Local install (inside an existing clone):
#   sh install.sh
#
# Env overrides:
#   DANKATTN_HOME   Install directory (default: current dir if inside a clone, else ~/.dankattn)
#   DANKATTN_BIN    Symlink directory for the `dankattn` CLI (default: ~/.local/bin)
#   DANKATTN_REPO   Git repo URL (default: https://github.com/Dankpaws/DankAttn.git)
#   DANKATTN_BRANCH Git branch (default: main)
set -eu

REPO="${DANKATTN_REPO:-https://github.com/Dankpaws/DankAttn.git}"
BRANCH="${DANKATTN_BRANCH:-main}"
BIN_DIR="${DANKATTN_BIN:-$HOME/.local/bin}"
SKIP_BUILD=0

for arg in "$@"; do
  case "$arg" in
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help)
      echo "Usage: sh install.sh [--skip-build]"
      echo "Env: DANKATTN_HOME, DANKATTN_BIN, DANKATTN_REPO, DANKATTN_BRANCH"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 64
      ;;
  esac
done

say() { printf '\033[1;34m[dankattn]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dankattn]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[dankattn]\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$SKIP_BUILD" -eq 0 ]; then
  if [ "$OS" != "Darwin" ] || [ "$ARCH" != "arm64" ]; then
    die "DankAttn requires Apple Silicon macOS (detected: $OS/$ARCH). Pass --skip-build for non-Mac development."
  fi
fi

# Detect hardware info on macOS
CHIP=""
RAM_GB=0
if [ "$OS" = "Darwin" ]; then
  CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
  MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
  RAM_GB="$((MEM_BYTES / 1073741824))"
  say "Detected: $CHIP (${RAM_GB} GB unified memory, macOS $(sw_vers -productVersion 2>/dev/null || echo unknown))"
fi

# Check prerequisites
command -v git >/dev/null 2>&1 || die "git is required. Install Xcode Command Line Tools with: xcode-select --install"

if [ "$SKIP_BUILD" -eq 0 ]; then
  if ! command -v clang++ >/dev/null 2>&1 || ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools (clang++) are required to compile the Metal 4 engine (~2 seconds)."
    warn "Run: xcode-select --install"
    warn "Then re-run this installer."
    exit 1
  fi
fi

# Find Python 3.11+
find_python() {
  for cand in "${PYTHON:-}" python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    [ -n "$cand" ] || continue
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        command -v "$cand"
        return 0
      fi
    fi
  done
  return 1
}

PY="$(find_python || true)"
if [ -z "$PY" ]; then
  die "Python 3.11+ is required for the OpenAI front end. Install via Homebrew (brew install python@3.12) or python.org, then re-run."
fi
say "Using Python: $PY ($("$PY" --version 2>&1))"

# Determine install directory: use current checkout if run inside DankAttn
if [ -n "${DANKATTN_HOME:-}" ]; then
  HOME_DIR="$DANKATTN_HOME"
elif [ -f "./engine/inkling2/inkling2.mm" ] && [ -f "./bin/dankattn" ]; then
  HOME_DIR="$(pwd)"
else
  HOME_DIR="$HOME/.dankattn"
fi

if [ -f "$HOME_DIR/engine/inkling2/inkling2.mm" ]; then
  say "Using existing DankAttn directory: $HOME_DIR"
  if [ -d "$HOME_DIR/.git" ] && [ "$HOME_DIR" = "$HOME/.dankattn" ]; then
    say "Updating repository..."
    git -C "$HOME_DIR" pull --ff-only || warn "git pull skipped (local changes or detached HEAD)"
  fi
else
  if [ -e "$HOME_DIR" ] && [ ! -d "$HOME_DIR/.git" ]; then
    die "$HOME_DIR already exists and is not a DankAttn git checkout. Set DANKATTN_HOME to another path."
  fi
  say "Cloning DankAttn into $HOME_DIR..."
  git clone --branch "$BRANCH" --single-branch "$REPO" "$HOME_DIR"
fi

# Set up Python virtual environment
VENV="$HOME_DIR/.venv"
if command -v uv >/dev/null 2>&1; then
  say "Setting up Python environment with uv..."
  [ -d "$VENV" ] || uv venv --python "$PY" "$VENV"
  uv pip install --python "$VENV/bin/python" -r "$HOME_DIR/requirements.txt"
else
  say "Setting up Python virtual environment..."
  [ -d "$VENV" ] || "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$HOME_DIR/requirements.txt"
fi

# Build Metal 4 engine binaries
if [ "$SKIP_BUILD" -eq 0 ]; then
  say "Compiling Inkling-2-Mini engine..."
  sh "$HOME_DIR/inkling2/build_inkling2.sh"
  say "Compiling Hy4-flash engine..."
  sh "$HOME_DIR/hy4/build_hy4.sh"
  say "Compiling ERNIE-5.5-Flash engine..."
  sh "$HOME_DIR/ernie55/build_ernie55.sh"
else
  say "Skipping engine compilation (--skip-build)"
fi

# Install `dankattn` CLI symlink
chmod +x "$HOME_DIR/bin/dankattn"
mkdir -p "$BIN_DIR"
ln -sf "$HOME_DIR/bin/dankattn" "$BIN_DIR/dankattn"
say "Installed CLI: $BIN_DIR/dankattn -> $HOME_DIR/bin/dankattn"

# Check PATH
case ":${PATH:-}:" in
  *":$BIN_DIR:"*) PATH_OK=1 ;;
  *) PATH_OK=0 ;;
esac

echo ""
say "DankAttn is installed!"
if [ "$PATH_OK" -eq 0 ]; then
  warn "$BIN_DIR is not in your PATH yet. Add it with:"
  echo "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
  echo ""
fi

echo "Next steps:"
echo "  1. Check your hardware & auto-sized profile:"
echo "       dankattn doctor"
echo "  2. Download Inkling-2-Mini (~19 GB, runs on 32 GB+ Apple Silicon):"
echo "       dankattn pull inkling2"
echo "  3. Start the OpenAI-compatible server (foreground, Ctrl-C to stop):"
echo "       dankattn run inkling2"
echo "     Or run in the background:"
echo "       dankattn serve inkling2 start"
echo "       dankattn serve inkling2 smoke"
if [ "$RAM_GB" -ge 96 ]; then
  echo ""
  echo "Your Mac has ${RAM_GB} GB unified memory — it can also run Hy4-flash (220B MoE):"
  echo "       dankattn pull hy4 && dankattn run hy4"
fi
