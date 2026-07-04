#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
rm -rf solution figures __pycache__ assets/__pycache__
echo "[allclean] cylinderIBM cleaned."
