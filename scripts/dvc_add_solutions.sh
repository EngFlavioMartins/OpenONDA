#!/bin/bash
# dvc_add_solutions.sh — track OpenONDA simulation RESULTS with DVC.
#
# Policy (see docs/data_management.md):
#   * Files needed to RUN a case live in git   (scripts, system/, constant/
#     {transportProperties,turbulenceProperties}, constant/polyMesh.orig/,
#     0.orig/, assets/, figures/ PNG·PDF).
#   * Files PRODUCED by a run live in DVC       (solution/, referenceFlow/, and
#     OpenFOAM reconstructed time directories such as 0.08, 0.16, …, 15).
#   * Regenerable scratch is ignored entirely   (processor*/, constant/polyMesh/,
#     the runtime 0/ copy, *.foam, log.*, VTK/, postProcessing/).
#
# This script walks tutorials/ and `dvc add`s every result directory that is not
# yet tracked, then stages the resulting .dvc / .gitignore files for git.  It
# does NOT commit and does NOT `dvc push` — do those yourself once you have
# reviewed `git status` (see the hints printed at the end).
#
# Usage:
#   scripts/dvc_add_solutions.sh                 # track everything under tutorials/
#   scripts/dvc_add_solutions.sh tutorials/VPM   # restrict to a sub-tree
#   DVC=dvc scripts/dvc_add_solutions.sh         # override the dvc invocation

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# Resolve repo root (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROOT="${1:-tutorials}"

# DVC invocation: prefer an explicit $DVC, else the OpenONDA env, else PATH.
if [ -n "${DVC:-}" ]; then
    :
elif conda run -n OpenONDA dvc --version >/dev/null 2>&1; then
    DVC="conda run -n OpenONDA dvc"
elif command -v dvc >/dev/null 2>&1; then
    DVC="dvc"
else
    echo -e "${RED}ERROR: dvc not found (install it or set \$DVC).${NC}"; exit 1
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo -e "${RED}ERROR: not in a git repository.${NC}"; exit 1
fi
if [ ! -d ".dvc" ]; then
    echo -e "${RED}ERROR: DVC not initialised (run 'dvc init').${NC}"; exit 1
fi

echo "======================================="
echo " DVC result tracker  (root: $ROOT)"
echo "======================================="

added=0 skipped=0

# Track one directory if it exists and is not already tracked.
add_to_dvc() {
    local dir="$1"
    [ -d "$dir" ] || return 0
    if [ -f "${dir}.dvc" ]; then
        echo -e "  ${YELLOW}skip${NC}  $dir (already tracked)"
        skipped=$((skipped + 1))
        return 0
    fi
    echo -e "  ${GREEN}add ${NC}  $dir"
    $DVC add "$dir" >/dev/null
    git add "${dir}.dvc" "$(dirname "$dir")/.gitignore" >/dev/null 2>&1 || true
    added=$((added + 1))
}

# 1) VPM / coupler diagnostics and backups, plus reference data.
while IFS= read -r d; do add_to_dvc "$d"; done \
    < <(find "$ROOT" -type d \( -name solution -o -name referenceFlow \) 2>/dev/null | sort)

# 2) OpenFOAM reconstructed time directories: numeric names > 0 at the case root.
#    Prunes processor*/, postProcessing/ and VTK/ (their nested numeric subdirs
#    are scratch, not top-level time results).  Excludes 0 and 0.orig.  Matches
#    integer (15, 100) and decimal (0.08, 1.36) time names.
while IFS= read -r d; do add_to_dvc "$d"; done \
    < <(find "$ROOT" \
            \( -name 'processor*' -o -name postProcessing -o -name VTK \) -prune -o \
            -type d -regextype posix-extended \
            -regex '.*/[0-9]+(\.[0-9]+)?' \
            ! -regex '.*/0' \
            -print 2>/dev/null | sort)

echo "---------------------------------------"
echo -e " added: ${GREEN}${added}${NC}   skipped: ${YELLOW}${skipped}${NC}"
echo "---------------------------------------"

if [ "$added" -gt 0 ]; then
    cat <<EOF

Next steps (review first, then run):
  git status                       # confirm only *.dvc / .gitignore are staged
  ${DVC} push                      # upload results to the Nextcloud remote
  git commit -m "Track new simulation results with DVC"
  git push
EOF
else
    echo "Nothing new to track."
fi
