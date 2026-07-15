#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
rm -rf solution/ figures/
find assets -name '*.msh' -delete 2>/dev/null || true
find assets -name '*.vtk' -delete 2>/dev/null || true
rm -f stepProfile.msh
