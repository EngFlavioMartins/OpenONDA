#!/usr/bin/env bash
# Compare the instability-onset time of three DNS schemes and transposed LES.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.vortex_ring"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"

CACHE_PARENT="${TI_OFFLINE_CACHE_FILE_PATH:-${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}/taichi}"
mkdir -p "${CACHE_PARENT}"
RUN_CACHE_DIR="$(mktemp -d "${CACHE_PARENT%/}/vortex-ring.XXXXXX")"
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

CLEAN=0
STEPS=""
while (( $# > 0 )); do
    case "$1" in
        --resume) shift ;;
        --clean) CLEAN=1; shift ;;
        --steps)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
                printf 'Usage: %s [--resume|--clean] [--steps N]\n' "$0" >&2
                exit 2
            }
            STEPS="$2"
            shift 2
            ;;
        *)
            printf 'Usage: %s [--resume|--clean] [--steps N]\n' "$0" >&2
            exit 2
            ;;
    esac
done

printf '[campaign] Vortex ring | DNS stretching schemes and transposed LES\n'
if (( CLEAN )); then
    run_phase "Clean previous outputs" "${SCRIPT_DIR}/allclean.sh"
fi

run_variant() {
    local variant="$1"
    local label="$2"
    local arguments=(--variant "${variant}" --resume)
    if [[ -n "${STEPS}" ]]; then
        arguments+=(--steps "${STEPS}")
    fi
    run_phase "${label}" "${PYTHON_BIN}" -u -m "${MODULE}.setup" "${arguments[@]}"
}

run_variant dns_direct "DNS Direct"
run_variant dns_transposed "DNS Transposed"
run_variant dns_mixed "DNS Mixed"
run_variant les_transposed "LES Transposed"
run_phase "Figures and final validation" "${SCRIPT_DIR}/allplot.sh" --strict
printf '\n[campaign] COMPLETE | vortex-ring instability comparison and plots passed\n'
