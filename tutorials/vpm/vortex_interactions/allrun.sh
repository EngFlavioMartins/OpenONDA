#!/usr/bin/env bash
# Reproducible classical-VPM comparisons; one GPU calculation at a time.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${OPENONDA_PYTHON:-python}"
CAMPAIGN=budget
VISCOUS=""
SCENARIO=""
STEPS=""
SPACING=""
DT=""
INTEGRATOR=""
CORE_RATIO=""
SUPPORT=disturbed
CLEAN=0
PLOT=1
usage() {
    cat <<'HELP'
Usage: allrun.sh [options]
  --campaign budget|qualification|baseline|stabilized|recommended|strategies|screen|legacy
                                      default: budget (bounded LES study)
  --viscous cs|gbd_m4|gbd_lagrange6     required for baseline/stabilized
  --scenario leapfrog|collision|both   qualification/baseline/stabilized: leapfrog; other campaigns: both
  --steps N                           qualification: 40; baseline/stabilized: 1600; other campaigns: 1200
  --spacing H                         new campaigns: .03; other campaigns: .035
  --dt DT                             new campaigns: .00375; others use setup.py
  --integrator RK4|SSPRK3              default: SSPRK3; required for LES campaigns
  --core-ratio S                      new campaigns sigma/h: 4/3; other campaigns: 2
  --support circular|disturbed         default: disturbed
  --quick                             set 400 steps and spacing .04
  --resume                            reuse completed compatible runs (default)
  --clean                             remove legacy outputs; study results are archived on replacement
  --no-plot                           skip summary generation

Budget: fixed-core GBD/Lagrange6, LES/RK3/transposed, h=.04, sigma=.04,
dt=.0075, no imposed disturbance. Run a baseline and weak moment-preserving
realignment comparison to t=6, each capped at 50 minutes; then a half-dt
baseline check capped at 30 minutes. This is an exploratory, budgeted study,
not a certified viscous winner or converged LBM match. Output/initialization
adds overhead beyond the caps. Do not launch this alongside another campaign.
Qualification: LES (Smagorinsky Cs=.20), SSPRK3, transposed stretching,
corrected Gaussian and no imposed seed, matching the unperturbed LBM target.
Compare CS, GBD/M4' and GBD/Lagrange6 at dt and dt/2 at the same physical time.
This is a short temporal/viscous qualification, not a full LBM validation.
Baseline: run the explicitly selected viscous candidate without stabilization.
Stabilized: first run that identical baseline, then matched stabilization cases.
Use the observed baseline failure to decide which operations warrant study.
All new campaigns use VPM flow/ring samplers, SurfaceSampler fields and normal
self-diagnostics. Postprocessing only reads recorded CSV/VTS/PVD outputs.
No particle snapshot reconstruction or auxiliary continuation is required.
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
        --viscous) VISCOUS="${2:?missing viscous variant}"; shift 2 ;;
        --campaign) CAMPAIGN="${2:?missing campaign}"; shift 2 ;;
        --scenario) SCENARIO="${2:?missing scenario}"; shift 2 ;;
        --steps) STEPS="${2:?missing steps}"; shift 2 ;;
        --spacing) SPACING="${2:?missing spacing}"; shift 2 ;;
        --dt) DT="${2:?missing dt}"; shift 2 ;;
        --integrator) INTEGRATOR="${2:?missing integrator}"; shift 2 ;;
        --core-ratio) CORE_RATIO="${2:?missing core ratio}"; shift 2 ;;
        --support) SUPPORT="${2:?missing support}"; shift 2 ;;
        --quick) STEPS=400; SPACING=.04; shift ;;
        --resume) shift ;;
        --clean) CLEAN=1; shift ;;
        --no-plot) PLOT=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ "$CAMPAIGN" == budget ]]; then
    SCENARIO="${SCENARIO:-leapfrog}"; STEPS="${STEPS:-800}"; SPACING="${SPACING:-.04}"
    DT="${DT:-.0075}"; INTEGRATOR="${INTEGRATOR:-SSPRK3}"; CORE_RATIO="${CORE_RATIO:-1}"
elif [[ "$CAMPAIGN" =~ ^(qualification|baseline|stabilized)$ ]]; then
    default_steps=1600
    if [[ "$CAMPAIGN" == qualification ]]; then default_steps=40; fi
    SCENARIO="${SCENARIO:-leapfrog}"; STEPS="${STEPS:-$default_steps}"; SPACING="${SPACING:-.03}"
    INTEGRATOR="${INTEGRATOR:-SSPRK3}"; CORE_RATIO="${CORE_RATIO:-1.3333333333333333}"
else
    SCENARIO="${SCENARIO:-both}"; STEPS="${STEPS:-1200}"; SPACING="${SPACING:-.035}"
    INTEGRATOR="${INTEGRATOR:-SSPRK3}"; CORE_RATIO="${CORE_RATIO:-2}"
fi
[[ "$STEPS" =~ ^[0-9]+$ && "$STEPS" -gt 0 ]] || { usage >&2; exit 2; }
[[ "$CAMPAIGN" =~ ^(budget|qualification|baseline|stabilized|recommended|strategies|screen|legacy)$ ]] || { usage >&2; exit 2; }
[[ "$SCENARIO" =~ ^(leapfrog|collision|both)$ ]] || { usage >&2; exit 2; }
[[ "$SUPPORT" =~ ^(circular|disturbed)$ ]] || { usage >&2; exit 2; }
[[ "$INTEGRATOR" =~ ^(RK4|SSPRK3)$ ]] || { usage >&2; exit 2; }
CACHE_PARENT="${TI_OFFLINE_CACHE_FILE_PATH:-${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}/taichi}"
mkdir -p "$CACHE_PARENT"
RUN_CACHE_DIR="$(mktemp -d "${CACHE_PARENT%/}/vortex-interactions.XXXXXX")"
export TI_OFFLINE_CACHE_FILE_PATH="$RUN_CACHE_DIR"
trap 'rm -rf -- "$RUN_CACHE_DIR"' EXIT
if (( CLEAN )); then "${SCRIPT_DIR}/allclean.sh"; fi

run() {
    local label="$1"; shift
    printf '\n[campaign] START | %s\n' "$label"
    if "$PYTHON_BIN" -u -m openonda.tutorial_runner "${SCRIPT_DIR}" "$@"; then
        printf '[campaign] DONE  | %s | inspect recorded termination status\n' "$label"
    else
        local status=$?
        printf '[campaign] STOP  | %s | exit %s; continuing\n' "$label" "$status" >&2
    fi
}
if [[ "$CAMPAIGN" == budget ]]; then
    [[ "$INTEGRATOR" == SSPRK3 && "$SCENARIO" == leapfrog ]] || { usage >&2; exit 2; }
    [[ -z "$VISCOUS" || "$VISCOUS" == gbd_lagrange6 ]] || { usage >&2; exit 2; }
    runs=(); reuse=()
    if (( ! CLEAN )); then reuse=(--resume); fi
    for variant in baseline p_moments halfdt; do
        method="$variant"; run_dt="$DT"; run_steps="$STEPS"; wall_minutes=50
        if [[ "$variant" == halfdt ]]; then
            method=baseline; wall_minutes=30; run_steps=$((STEPS * 2))
            run_dt="$(awk -v dt="$DT" 'BEGIN {printf "%.12g", dt/2}')"
        fi
        tag="les_budget_leapfrog_gbd_l6_${variant}_dt${run_dt}_h${SPACING}"
        runs+=("$tag")
        "$PYTHON_BIN" -u -m openonda.tutorial_runner "${SCRIPT_DIR}" study --scenario leapfrog --method "$method" \
            --steps "$run_steps" --dt "$run_dt" --spacing "$SPACING" \
            --wall-minutes "$wall_minutes" --integrator SSPRK3 --stretching TRANSPOSED \
            --core-ratio "$CORE_RATIO" --support circular --amplitude 0 --smagorinsky .20 \
            --initial-tail .0001 --tree-theta .5 --tree-order 3 --capacity 1000000 \
            --field-interval .15 --diffusion GBD --gbd-remeshing LAGRANGE6 --diffusion-tail .0001 \
            --frequency .384684814725 --tag "$tag" "${reuse[@]}"
    done
    if (( PLOT )); then
        "$PYTHON_BIN" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_core_sections --runs "${runs[@]}"
        "$PYTHON_BIN" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.assess_lbm_agreement "${runs[@]}"
    fi
    exit 0
fi
if [[ "$CAMPAIGN" =~ ^(qualification|baseline|stabilized)$ ]]; then
    [[ "$INTEGRATOR" == SSPRK3 ]] || { printf 'LES study requires SSPRK3.\n' >&2; exit 2; }
    variants=(cs gbd_m4 gbd_lagrange6)
    if [[ -n "$VISCOUS" ]]; then
        [[ "$VISCOUS" =~ ^(cs|gbd_m4|gbd_lagrange6)$ ]] || { usage >&2; exit 2; }
        variants=("$VISCOUS")
    elif [[ "$CAMPAIGN" != qualification ]]; then
        printf 'Select --viscous from qualification evidence before baseline/stabilized runs.\n' >&2
        exit 2
    fi
    scenarios=("$SCENARIO")
    if [[ "$SCENARIO" == both ]]; then scenarios=(leapfrog collision); fi
    runs=(); leapfrog_runs=(); reuse=()
    if (( ! CLEAN )); then reuse=(--resume); fi
    for scenario in "${scenarios[@]}"; do
        for variant in "${variants[@]}"; do
            tuning=(--diffusion CS)
            if [[ "$variant" != cs ]]; then tuning=(--diffusion GBD --diffusion-tail .0001); fi
            if [[ "$variant" == gbd_lagrange6 ]]; then tuning+=(--gbd-remeshing LAGRANGE6); fi
            methods=(baseline)
            if [[ "$CAMPAIGN" == stabilized ]]; then
                methods+=(splitting)
                if [[ "$variant" == cs ]]; then methods+=(remeshing p_remesh)
                else methods+=(p_moments); fi
            fi
            refinements=(1)
            if [[ "$CAMPAIGN" == qualification ]]; then refinements+=(2); fi
            for method in "${methods[@]}"; do
                for refinement in "${refinements[@]}"; do
                    run_dt="${DT:-.00375}"
                    if [[ "$refinement" == 2 ]]; then run_dt="$(awk -v dt="$run_dt" 'BEGIN {printf "%.12g", dt/2}')"; fi
                    run_steps=$((STEPS * refinement))
                    tag="les_${CAMPAIGN}_${scenario}_${variant}_${method}_dt${run_dt}_h${SPACING}"
                    runs+=("$tag")
                    if [[ "$scenario" == leapfrog ]]; then leapfrog_runs+=("$tag"); fi
                    # Fail on an execution error; a recorded health stop remains valid evidence.
                    "$PYTHON_BIN" -u -m openonda.tutorial_runner "${SCRIPT_DIR}" study --scenario "$scenario" --method "$method" \
                        --steps "$run_steps" --dt "$run_dt" --spacing "$SPACING" \
                        --integrator SSPRK3 --stretching TRANSPOSED --core-ratio "$CORE_RATIO" \
                        --support circular --amplitude 0 --smagorinsky .20 --initial-tail .0001 \
                        --tree-theta .5 --tree-order 3 --capacity 1000000 --field-interval .15 \
                        --frequency .384684814725 "${tuning[@]}" --tag "$tag" "${reuse[@]}"
                done
            done
        done
    done
    if (( PLOT )); then
        "$PYTHON_BIN" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.plot_core_sections --runs "${runs[@]}"
        if (( ${#leapfrog_runs[@]} )); then
            "$PYTHON_BIN" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.assess_lbm_agreement "${leapfrog_runs[@]}"
        fi
    fi
    exit 0
fi
if [[ "$CAMPAIGN" == legacy ]]; then
    names=(baseline stretching_viscosity pedrizzetti splitting divergence_relaxation remeshing)
    labels=(Baseline 'Stretching viscosity' 'Pedrizzetti relaxation' 'Filament refinement' 'Divergence relaxation' 'Conservative regularization')
    for index in "${!names[@]}"; do
        run "${labels[$index]}" setup "${names[$index]}" --steps "$STEPS" --resume
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
        run "$scenario / $method" study --scenario "$scenario" --method "$method" \
            --steps "$STEPS" --spacing "$SPACING" --support "$SUPPORT" \
            --integrator "$INTEGRATOR" --core-ratio "$CORE_RATIO" \
            --frequency .384684814725 "${tuning[@]}" --tag "$tag" "${reuse[@]}"
    done
done
if (( PLOT )); then
    "$PYTHON_BIN" -m openonda.tutorial_runner "${SCRIPT_DIR}" assets.analyze_study --runs "${runs[@]}" --output "${SCRIPT_DIR}/figures/study/${CAMPAIGN}"
fi
