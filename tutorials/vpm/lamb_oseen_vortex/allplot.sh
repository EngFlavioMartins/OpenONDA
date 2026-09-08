#!/usr/bin/env bash
# Rebuild the Lamb--Oseen diagnostics and figures from existing samples.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${SCRIPT_DIR}/.cache/numba}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}" "${SCRIPT_DIR}/figures"

"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.postprocess --extract-fields

"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_vortex_comparison --format both
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_dipole_comparison --format both
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_merging_comparison --format both
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_vortex_surface_fields --format both
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_lamboseen_energy --format both
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_merging_snapshots --format both

"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.postprocess --manifest
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.postprocess
