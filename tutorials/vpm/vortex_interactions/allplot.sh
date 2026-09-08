#!/usr/bin/env bash
# Plot every available vortex-interaction stabilization result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}" "${SCRIPT_DIR}/figures"

STRICT=0
case "${1:-}" in
    "") ;;
    --strict) STRICT=1 ;;
    *) printf 'Usage: %s [--strict]\n' "$0" >&2; exit 2 ;;
esac

shopt -s nullglob
SAMPLE_FILES=("${SCRIPT_DIR}"/samples/*/flow_integrals.csv)
shopt -u nullglob
# Plane figures also cover study_results, independently of legacy CSV outputs.
"${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_core_sections
shopt -s nullglob
les_runs=()
for index in "${SCRIPT_DIR}"/study_results/les_*leapfrog*/samples/diagnostics/core_section.pvd; do
    folder="${index%/samples/diagnostics/core_section.pvd}"
    les_runs+=("${folder##*/}")
done
shopt -u nullglob
if (( ${#les_runs[@]} )); then
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" assess_lbm_agreement "${les_runs[@]}"
fi
if (( ${#SAMPLE_FILES[@]} == 0 )); then
    if (( STRICT )); then
        "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" validate_stabilization_suite --strict
    else
        "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" validate_stabilization_suite
    fi
    exit $?
fi

for figure_format in png pdf; do
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_circulation --format "${figure_format}"
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_conservation --format "${figure_format}"
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_energy --format "${figure_format}"
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_energy_budget --format "${figure_format}"
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_resolution --format "${figure_format}"
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_stability --format "${figure_format}"
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" plot_rings_trajectory --format "${figure_format}"
done

if (( STRICT )); then
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" validate_stabilization_suite --strict
else
    "${PYTHON_BIN}" -m openonda.tutorial_runner "${SCRIPT_DIR}" validate_stabilization_suite
fi
