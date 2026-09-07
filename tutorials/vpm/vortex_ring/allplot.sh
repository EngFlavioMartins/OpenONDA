#!/usr/bin/env bash
# Rebuild the vortex-ring figures from the samples that are available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.vortex_ring"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/figures"

STRICT=0
case "${1:-}" in
    "") ;;
    --strict) STRICT=1 ;;
    *) printf 'Usage: %s [--strict]\n' "$0" >&2; exit 2 ;;
esac

if (( STRICT )); then
    "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" --pre-plot
else
    "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" --available --pre-plot
fi
for figure_format in png pdf; do
    "${PYTHON_BIN}" -m "${MODULE}.assets.plot_vortex_ring_motion" --format "${figure_format}"
    "${PYTHON_BIN}" -m "${MODULE}.assets.plot_vortex_ring_energy" --format "${figure_format}"
    "${PYTHON_BIN}" -m "${MODULE}.assets.plot_vortex_ring_circulation" --format "${figure_format}"
    "${PYTHON_BIN}" -m "${MODULE}.assets.plot_vortex_ring_stability" --format "${figure_format}"
done
"${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" --manifest
if (( STRICT )); then
    "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess"
else
    "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" --available
fi
