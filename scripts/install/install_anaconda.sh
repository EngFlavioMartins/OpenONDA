#!/usr/bin/env bash
# Compatibility entry point. Prefer install_conda.sh in new documentation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/install_conda.sh" "$@"
