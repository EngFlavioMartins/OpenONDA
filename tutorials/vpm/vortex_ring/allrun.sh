#!/usr/bin/env bash
# Run the complete Widnall vortex-ring comparison on Metal.
set -euo pipefail

cd "$(dirname "$0")"

python_bin="${OPENONDA_PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi
python_path=$(command -v "$python_bin")
export PATH="$(dirname "$python_path"):$PATH"

# Pinned production resolution; only C_s is an intentional calibration knob.
export OPENONDA_COMPUTE_DEVICE=METAL
export OPENONDA_RING_PARTICLE_SPACING=0.035
export OPENONDA_RING_MAX_N_PARTICLES=100000
export OPENONDA_RING_ENABLE_STABILIZATION=0
les_cs="${OPENONDA_RING_SMAGORINSKY_COEFFICIENT:-0.20}"

printf '%s\n' \
    "===== VORTEX-RING WIDNALL CAMPAIGN =====" \
    "backend=METAL" \
    "distribution=toroidal" \
    "particle_spacing=0.035" \
    "widnall_modes=1..24" \
    "widnall_amplitude=0.05" \
    "les_smagorinsky_coefficient=${les_cs}" \
    "stabilization=disabled" \
    "steps=3000"

"$python_bin" -c \
    'import taichi as ti; ti.init(arch=ti.metal, default_fp=ti.f32, offline_cache=False); ti.sync(); ti.reset(); print("[OK] Metal preflight passed.")'

./allclean.sh
mkdir -p solution

run_error=0
run_case() {
    local variant="$1"
    local coefficient="0.0"
    if [[ "$variant" == les_transposed ]]; then
        coefficient="$les_cs"
    fi
    echo "===== vortex ring: ${variant} ====="
    if "$python_bin" -u ring_setup.py \
        --variant "$variant" \
        --particle-distribution toroidal \
        --widnall-amplitude 0.05 \
        --widnall-modes 24 \
        --n-steps 3000 \
        --smagorinsky-coefficient "$coefficient" \
        --compute-device METAL \
        --velocity-method TREECODE \
        --treecode-theta 0.30 \
        2>&1 | tee "solution/${variant}.log"; then
        return
    fi
    echo "[WARN] ${variant} exited with an execution error; continuing the comparison." >&2
    run_error=1
}

run_case dns_direct
run_case dns_transposed
run_case dns_mixed
run_case les_transposed

echo "===== FIGURES ====="
plot_error=0
./allplot.sh png || plot_error=1
./allplot.sh pdf || plot_error=1

echo "===== STRICT DIAGNOSTIC REPORT ====="
if ! "$python_bin" assets/validate_results.py; then
    echo "[INFO] Strict completion failed for one or more variants; failure ordering remains available in the manifests and plots."
fi

if [[ "$run_error" -ne 0 || "$plot_error" -ne 0 ]]; then
    exit 1
fi
echo "[OK] Campaign execution and plotting completed."
