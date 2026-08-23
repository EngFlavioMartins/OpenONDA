#!/usr/bin/env bash
# Make every coupled cube-flow comparison figure from sampled results.
#
# Usage:
#   ./allplot.sh                         active run, PNG (default)
#   ./allplot.sh pdf                     active run, PDF
#   ./allplot.sh png run_backups/<run>   archived run
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

format="${1:-png}"
case "$format" in
    png|pdf) ;;
    *) echo "Usage: $0 [png|pdf] [run-directory]" >&2; exit 2 ;;
esac

results_root="${2:-$PWD}"
if [ ! -d "$results_root" ]; then
    echo "Run directory does not exist: $results_root" >&2
    exit 2
fi
results_root="$(cd "$results_root" && pwd)"
target="$results_root/figures"
staging="$results_root/.figures.tmp.$$"

rm -rf "$staging"
mkdir -p "$staging"
trap 'rm -rf "$staging"' EXIT

export OPENONDA_SAMPLES_DIR="$results_root/samples"
export OPENONDA_SOLUTION_DIR="$results_root/solution"
export OPENONDA_REFERENCE_SAMPLES_DIR="$PWD/reference_flow/samples"
export OPENONDA_FIGURES_DIR="$staging"

plot() {
    python "$@" --format "$format"
}

python assets/validate_plot_inputs.py
plot assets/plot_velocity_profiles.py
plot assets/plot_velocity_fields.py
plot assets/plot_coupling_diagnostics.py
# plot assets/plot_wake_errors.py

rm -rf "$target"
mv "$staging" "$target"
trap - EXIT
echo "Figures are complete in $target"
