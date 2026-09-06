#!/usr/bin/env bash
# Rebuild the Lamb--Oseen diagnostics and figures from existing samples.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.lamb_oseen_vortex"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/figures"

"${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" --extract-fields

"${PYTHON_BIN}" -m "${MODULE}.assets.plot_vortex_comparison" --format both
"${PYTHON_BIN}" -m "${MODULE}.assets.plot_dipole_comparison" --format both
"${PYTHON_BIN}" -m "${MODULE}.assets.plot_merging_comparison" --format both
"${PYTHON_BIN}" -m "${MODULE}.assets.plot_vortex_surface_fields" --format both
"${PYTHON_BIN}" -m "${MODULE}.assets.plot_lamboseen_energy" --format both

"${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" --manifest
"${PYTHON_BIN}" -m "${MODULE}.assets.postprocess"
