#!/usr/bin/env bash
# Run the Lamb--Oseen vortex, dipole, and merging-vortex comparisons.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.lamb_oseen_vortex"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"

CACHE_PARENT="${TI_OFFLINE_CACHE_FILE_PATH:-${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}/taichi}"
mkdir -p "${CACHE_PARENT}"
RUN_CACHE_DIR="$(mktemp -d "${CACHE_PARENT%/}/lamb-oseen.XXXXXX")"
export TI_OFFLINE_CACHE_FILE_PATH="${RUN_CACHE_DIR}"
CURRENT_PHASE="setup"
finish() {
    local status=$?
    if (( status != 0 )); then
        printf '\n[campaign] FAILED | %s | exit %s\n' "${CURRENT_PHASE}" "${status}" >&2
    fi
    rm -rf -- "${RUN_CACHE_DIR}"
    trap - EXIT
    exit "${status}"
}
trap finish EXIT

run_phase() {
    CURRENT_PHASE="$1"
    shift
    local started=${SECONDS}
    printf '\n[campaign] START | %s\n' "${CURRENT_PHASE}"
    "$@"
    printf '[campaign] DONE  | %s | %ss\n' "${CURRENT_PHASE}" "$((SECONDS - started))"
}

printf '[campaign] Lamb–Oseen | vortex → dipole → merging | 4 methods per case\n'
run_phase "Clean previous outputs" "${SCRIPT_DIR}/allclean.sh"

run_physics_case() {
    local physics="$1"
    run_phase "${physics} / CS" "${PYTHON_BIN}" -u -m "${MODULE}.setup" "${physics}" CS

    run_phase "${physics} / RWM / 10 realizations" \
        "${PYTHON_BIN}" -u -m "${MODULE}.assets.rwm_ensemble" "${physics}" \
        --number-of-realizations 10
    run_phase "${physics} / aggregate RWM" "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" \
        --aggregate-rwm-case "${physics}" --expected-rwm-members 10

    run_phase "${physics} / DVH" "${PYTHON_BIN}" -u -m "${MODULE}.setup" "${physics}" DVH

    run_phase "${physics} / GBD" "${PYTHON_BIN}" -u -m "${MODULE}.setup" "${physics}" GBD

    run_phase "${physics} / extract diagnostics" "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" \
        --extract-fields --case "${physics}"
    run_phase "${physics} / validate" "${PYTHON_BIN}" -m "${MODULE}.assets.postprocess" \
        --pre-plot --validate-case "${physics}"
}

run_physics_case vortex
run_physics_case dipole
run_physics_case merging

run_phase "Figures and final validation" "${SCRIPT_DIR}/allplot.sh"
printf '\n[campaign] COMPLETED | all cases and final validation passed\n'
