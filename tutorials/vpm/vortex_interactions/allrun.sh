#!/usr/bin/env bash
# Reproducible classical-VPM comparisons; one GPU calculation at a time.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
MODULE="tutorials.vpm.vortex_interactions"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"
CAMPAIGN=physics
SCENARIO=""
STEPS=""
SPACING=""
DT=""
SUPPORT=disturbed
CLEAN=0
PLOT=1
usage() {
    cat <<'HELP'
Usage: allrun.sh [options]
  --campaign physics|recommended|strategies|screen|legacy  default: physics
  --scenario leapfrog|collision|both   physics: leapfrog; other campaigns: both
  --steps N                           physics: 400; other campaigns: 1200
  --spacing H                         physics: .04; other campaigns: .035
  --dt DT                             physics default: .015; others use setup.py
  --support circular|disturbed         default: disturbed
  --quick                             set 400 steps and spacing .04
  --resume                            reuse completed compatible runs (default)
  --clean                             remove legacy outputs; study results are archived on replacement
  --no-plot                           skip summary generation

Physics: corrected Gaussian, no seed or LES; compare core spreading with
fixed-core GBD using M4' and experimental six-point Lagrange remapping.
These are validation controls, not certified LBM matches. All use sigma=h
and the same initial particles representing the prescribed Gaussian core.
Tree theta=.5/order=3 is a qualified cost setting; strict default remains
available through study.py. Inspect particle caps and field peaks before
interpreting late-time behaviour. No rVPM is used.
Recommended: the earlier baseline control; added stabilization has not yet qualified
as a physically reliable extension through collision breakdown.
Strategies: baseline, filament splitting, Gaussian core remeshing, and weak
moment-corrected P-relaxation plus remeshing. Screen adds residual viscosity,
strong normalized realignment, constrained divergence relaxation, weak
realignment alone, the combined split/remesh method and solenoidal remeshing.
Legacy reproduces the six original setup.py cases and their original support.
A completed run is a numerical outcome, not certification of late-time physics.
HELP
}
while (( $# > 0 )); do
    case "$1" in
        --campaign) CAMPAIGN="${2:?missing campaign}"; shift 2 ;;
        --scenario) SCENARIO="${2:?missing scenario}"; shift 2 ;;
        --steps) STEPS="${2:?missing steps}"; shift 2 ;;
        --spacing) SPACING="${2:?missing spacing}"; shift 2 ;;
        --dt) DT="${2:?missing dt}"; shift 2 ;;
        --support) SUPPORT="${2:?missing support}"; shift 2 ;;
        --quick) STEPS=400; SPACING=.04; shift ;;
        --resume) shift ;;
        --clean) CLEAN=1; shift ;;
        --no-plot) PLOT=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ "$CAMPAIGN" == physics ]]; then
    SCENARIO="${SCENARIO:-leapfrog}"; STEPS="${STEPS:-400}"; SPACING="${SPACING:-.04}"
else
    SCENARIO="${SCENARIO:-both}"; STEPS="${STEPS:-1200}"; SPACING="${SPACING:-.035}"
fi
[[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 0 ]] || { usage >&2; exit 2; }
[[ "$CAMPAIGN" =~ ^(physics|recommended|strategies|screen|legacy)$ ]] || { usage >&2; exit 2; }
[[ "$SCENARIO" =~ ^(leapfrog|collision|both)$ ]] || { usage >&2; exit 2; }
[[ "$SUPPORT" =~ ^(circular|disturbed)$ ]] || { usage >&2; exit 2; }
CACHE_PARENT="${TI_OFFLINE_CACHE_FILE_PATH:-${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}/taichi}"
mkdir -p "$CACHE_PARENT"
RUN_CACHE_DIR="$(mktemp -d "${CACHE_PARENT%/}/vortex-interactions.XXXXXX")"
export TI_OFFLINE_CACHE_FILE_PATH="$RUN_CACHE_DIR"
trap 'rm -rf -- "$RUN_CACHE_DIR"' EXIT
if (( CLEAN )); then "${SCRIPT_DIR}/allclean.sh"; fi

run() {
    local label="$1"; shift
    printf '\n[campaign] START | %s\n' "$label"
    if "$PYTHON_BIN" -u -m "$@"; then
        printf '[campaign] DONE  | %s | inspect recorded termination status\n' "$label"
    else
        local status=$?
        printf '[campaign] STOP  | %s | exit %s; continuing\n' "$label" "$status" >&2
    fi
}
if [[ "$CAMPAIGN" == physics ]]; then
    scenarios=("$SCENARIO")
    if [[ "$SCENARIO" == both ]]; then scenarios=(leapfrog collision); fi
    runs=()
    reuse=()
    if (( ! CLEAN )); then reuse=(--resume); fi
    for scenario in "${scenarios[@]}"; do
        for variant in cs gbd_m4 gbd_lagrange6; do
            tag="physics_${scenario}_${variant}_h${SPACING}"
            runs+=("$tag")
            tuning=(--diffusion CS --core-ratio 1)
            if [[ "$variant" != cs ]]; then
                tuning=(--diffusion GBD --core-ratio 1 --diffusion-tail .0001)
            fi
            if [[ "$variant" == gbd_lagrange6 ]]; then tuning+=(--gbd-remeshing LAGRANGE6); fi
            run "$scenario / $variant" "${MODULE}.study" --scenario "$scenario" --method baseline \
                --steps "$STEPS" --dt "${DT:-.015}" --spacing "$SPACING" \
                --support circular --amplitude 0 --smagorinsky 0 --initial-tail .0001 \
                --tree-theta .5 --tree-order 3 --capacity 500000 --snapshot-interval 20 \
                "${tuning[@]}" --tag "$tag" "${reuse[@]}"
        done
    done
    if (( PLOT )); then
        "$PYTHON_BIN" -m "${MODULE}.assets.track_vorticity_cores" "${runs[@]}"
        "$PYTHON_BIN" -m "${MODULE}.assets.compare_physics" "${runs[@]}"
    fi
    exit 0
fi
if [[ "$CAMPAIGN" == legacy ]]; then
    names=(baseline stretching_viscosity pedrizzetti splitting divergence_relaxation remeshing)
    labels=(Baseline 'Stretching viscosity' 'Pedrizzetti relaxation' 'Filament refinement' 'Divergence relaxation' 'Conservative regularization')
    for index in "${!names[@]}"; do
        run "${labels[$index]}" "${MODULE}.setup" "${names[$index]}" --steps "$STEPS" --resume
    done
    if (( PLOT )); then "${SCRIPT_DIR}/allplot.sh" --strict; fi
    exit 0
fi
methods=(baseline)
if [[ "$CAMPAIGN" != recommended ]]; then methods+=(splitting remeshing p_remesh); fi
if [[ "$CAMPAIGN" == screen ]]; then
    methods+=(stretching_viscosity pedrizzetti divergence_relaxation p_moments p_split_remesh solenoidal_remeshing)
fi
scenarios=("$SCENARIO")
if [[ "$SCENARIO" == both ]]; then scenarios=(leapfrog collision); fi
runs=()
reuse=()
if (( ! CLEAN )); then reuse=(--resume); fi
for scenario in "${scenarios[@]}"; do
    for method in "${methods[@]}"; do
        tag="${CAMPAIGN}_${scenario}_${method}_h${SPACING}_${SUPPORT}"
        runs+=("$tag")
        tuning=(--tail-budget .01)
        if [[ "$method" == solenoidal_remeshing ]]; then
            # Experimental cost-controlled profile; no full collision validation.
            tuning=(--tail-budget .02 --remesh-spacing .05 --remesh-start 50 --remesh-interval 10 --divergence-trigger .08)
        fi
        if [[ -n "$DT" ]]; then tuning+=(--dt "$DT"); fi
        run "$scenario / $method" "${MODULE}.study" --scenario "$scenario" --method "$method" \
            --steps "$STEPS" --spacing "$SPACING" --support "$SUPPORT" \
            --frequency .384684814725 "${tuning[@]}" --tag "$tag" "${reuse[@]}"
    done
done
if (( PLOT )); then
    "$PYTHON_BIN" -m "${MODULE}.assets.analyze_study" --runs "${runs[@]}" --output "${SCRIPT_DIR}/figures/study/${CAMPAIGN}"
fi
