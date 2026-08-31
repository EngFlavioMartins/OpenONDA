#!/usr/bin/env bash
# Run the compact three-level spatial-convergence study for the cube reference.
#
# The current h64 archive is reused. Only h16 and h32 are simulated, one at a
# time, and only force/line samples are retained under samples/<grid>/.
#
# Usage:
#   ./grid_converge_study.sh
#   ./grid_converge_study.sh --dry-run
#   ./grid_converge_study.sh --analyse-only
set -euo pipefail

cd "$(dirname "$0")/.."
case_dir=$PWD
samples_root="$case_dir/samples"
runner="$case_dir/assets/run_trial.py"
analyser="$case_dir/assets/analyse_grid_study.py"
python_bin=${OPENONDA_PYTHON:-python}

cores=${OPENONDA_GRID_STUDY_CORES:-1}
end_time=${OPENONDA_GRID_STUDY_END_TIME:-20.0}
background_h=${OPENONDA_GRID_STUDY_BACKGROUND_SIZE:-0.5}
force_interval=${OPENONDA_GRID_STUDY_FORCE_INTERVAL:-0.05}
line_interval=${OPENONDA_GRID_STUDY_LINE_INTERVAL:-0.25}
window_start=${OPENONDA_GRID_STUDY_WINDOW_START:-15.0}
drag_tolerance=${OPENONDA_GRID_STUDY_DRAG_TOLERANCE:-0.01}
field_tolerance=${OPENONDA_GRID_STUDY_FIELD_TOLERANCE:-0.02}
minimum_available_gib=${OPENONDA_GRID_STUDY_MIN_AVAILABLE_GIB:-4.0}
maximum_load=${OPENONDA_GRID_STUDY_MAX_LOAD:-2.5}
maximum_start_temperature=${OPENONDA_GRID_STUDY_START_TEMP_C:-65.0}
maximum_run_temperature=${OPENONDA_GRID_STUDY_MAX_TEMP_C:-80.0}
resume_run_temperature=${OPENONDA_GRID_STUDY_RESUME_TEMP_C:-65.0}
thermal_check_steps=${OPENONDA_GRID_STUDY_THERMAL_CHECK_STEPS:-1}
minimum_step_wall_time=${OPENONDA_GRID_STUDY_MIN_STEP_SECONDS:-1.0}
resource_poll_seconds=${OPENONDA_GRID_STUDY_RESOURCE_POLL_SECONDS:-30}
required_stable_checks=${OPENONDA_GRID_STUDY_STABLE_CHECKS:-3}
maximum_wait_seconds=${OPENONDA_GRID_STUDY_MAX_WAIT_SECONDS:-0}
# Positive niceness lowers the scheduler priority.  CPUQuota is enforced by a
# transient user scope, preventing a single serial solver from turbo-heating
# the laptop between the temperature checks in run_trial.py.
nice_level=${OPENONDA_GRID_STUDY_NICE:-10}
cpu_quota_percent=${OPENONDA_GRID_STUDY_CPU_QUOTA_PERCENT:-20}

mode=run
if [ "${1:-}" = "--dry-run" ]; then
    mode=dry-run
elif [ "${1:-}" = "--analyse-only" ]; then
    mode=analyse-only
elif [ "${1:-}" = "--help" ]; then
    sed -n '1,13p' "$0"
    exit 0
elif [ "$#" -ne 0 ]; then
    echo "Unknown argument: $1" >&2
    exit 2
fi

for command_path in "$runner" "$analyser"; do
    if [ ! -f "$command_path" ]; then
        echo "Required helper is missing: $command_path" >&2
        exit 1
    fi
done
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python executable not found: $python_bin" >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to prevent overlapping grid studies" >&2
    exit 1
fi
if ! command -v systemd-run >/dev/null 2>&1; then
    echo "systemd-run is required to enforce the grid-study CPU quota" >&2
    exit 1
fi
if ! awk -v quota="$cpu_quota_percent" 'BEGIN {exit !(quota > 0.0 && quota <= 100.0)}'; then
    echo "OPENONDA_GRID_STUDY_CPU_QUOTA_PERCENT must be in (0, 100]" >&2
    exit 1
fi

mkdir -p "$samples_root"
lock_file="${TMPDIR:-/tmp}/openonda-cube-grid-study-${UID}.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
    echo "Another cube grid-convergence study is already active." >&2
    exit 1
fi

available_memory_gib() {
    awk '/^MemAvailable:/ {printf "%.3f", $2 / 1024 / 1024}' /proc/meminfo
}

package_temperature_c() {
    local preferred=""
    local fallback=""
    local type_path zone_type temperature
    for type_path in /sys/class/thermal/thermal_zone*/type; do
        [ -f "$type_path" ] || continue
        zone_type=$(<"$type_path")
        [ -f "${type_path%/type}/temp" ] || continue
        temperature=$(awk '{printf "%.3f", $1 / 1000}' "${type_path%/type}/temp")
        if [ "$zone_type" = "TCPU" ]; then
            if [ -z "$preferred" ] || awk -v a="$temperature" -v b="$preferred" 'BEGIN {exit !(a > b)}'; then
                preferred=$temperature
            fi
        elif [ "$zone_type" = "x86_pkg_temp" ]; then
            if [ -z "$fallback" ] || awk -v a="$temperature" -v b="$fallback" 'BEGIN {exit !(a > b)}'; then
                fallback=$temperature
            fi
        fi
    done
    if [ -n "$preferred" ]; then
        echo "$preferred"
    else
        echo "$fallback"
    fi
}

less_than() {
    awk -v first="$1" -v second="$2" 'BEGIN {exit !(first < second)}'
}

greater_than() {
    awk -v first="$1" -v second="$2" 'BEGIN {exit !(first > second)}'
}

wait_for_resources() {
    local grid=$1
    local waited=0
    local stable=0
    local memory load temperature reason
    while true; do
        memory=$(available_memory_gib)
        read -r load _ </proc/loadavg
        temperature=$(package_temperature_c)
        reason=""
        if less_than "$memory" "$minimum_available_gib"; then
            reason="available RAM ${memory} GiB < ${minimum_available_gib} GiB"
        elif greater_than "$load" "$maximum_load"; then
            reason="one-minute load ${load} > ${maximum_load}"
        elif [ -n "$temperature" ] && greater_than "$temperature" "$maximum_start_temperature"; then
            reason="CPU package ${temperature} C > ${maximum_start_temperature} C"
        fi
        if [ -z "$reason" ]; then
            stable=$((stable + 1))
            if [ "$stable" -ge "$required_stable_checks" ]; then
                echo "Resource gate passed for $grid: RAM=${memory} GiB load=${load} temp=${temperature:-unavailable} C"
                return
            fi
            echo "Resource gate stable for $grid ($stable/$required_stable_checks): RAM=${memory} GiB load=${load} temp=${temperature:-unavailable} C"
        else
            stable=0
            echo "Waiting to start $grid: $reason"
        fi
        if [ "$maximum_wait_seconds" -gt 0 ] && [ "$waited" -ge "$maximum_wait_seconds" ]; then
            echo "Resource gate timed out for $grid: $reason" >&2
            exit 1
        fi
        sleep "$resource_poll_seconds"
        waited=$((waited + resource_poll_seconds))
    done
}

grid_is_complete() {
    local destination=$1
    [ -s "$destination/forces_history.csv" ] \
        && [ -s "$destination/centreline.csv" ] \
        && [ -s "$destination/offaxis_y075.csv" ] \
        && [ -s "$destination/grid_metadata.json" ]
}

show_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_grid() {
    local grid=$1
    local surface_h=$2
    local destination="$samples_root/$grid"
    local work_directory
    if grid_is_complete "$destination"; then
        echo "Skipping completed grid $grid in $destination"
        return
    fi
    if [ -e "$destination" ]; then
        echo "Refusing to overwrite incomplete grid directory: $destination" >&2
        exit 1
    fi
    local -a command=(
        systemd-run --user --scope --quiet -p "CPUQuota=${cpu_quota_percent}%"
        nice -n "$nice_level" "$python_bin" -u "$runner"
        --grid-name "$grid"
        --surface-cell-size "$surface_h"
        --background-cell-size "$background_h"
        --end-time "$end_time"
        --cores "$cores"
        --force-interval "$force_interval"
        --line-interval "$line_interval"
        --max-cpu-temperature "$maximum_run_temperature"
        --resume-cpu-temperature "$resume_run_temperature"
        --thermal-check-steps "$thermal_check_steps"
        --minimum-step-wall-time "$minimum_step_wall_time"
    )
    if [ "$mode" = "dry-run" ]; then
        command+=(--output-directory "$case_dir/.grid-study-${grid}.XXXXXX")
        echo "Would run $grid:"
        show_command "${command[@]}"
        return
    fi
    wait_for_resources "$grid"
    work_directory=$(mktemp -d "$case_dir/.grid-study-${grid}.XXXXXX")
    command+=(--output-directory "$work_directory")
    echo
    echo "===== GRID $grid (surface h/D=$surface_h) ====="
    echo "Scratch directory: $work_directory"
    if ! "${command[@]}" 2>&1 | tee "$work_directory/grid_run.log"; then
        echo "Grid $grid failed; scratch data were retained in $work_directory" >&2
        exit 1
    fi
    if ! grid_is_complete "$work_directory/samples"; then
        echo "Grid $grid ended without a complete compact sample set: $work_directory" >&2
        exit 1
    fi
    cp "$work_directory/grid_run.log" "$work_directory/samples/grid_run.log"
    mv "$work_directory/samples" "$destination"
    rm -rf -- "$work_directory"
    echo "Retained compact grid data in $destination"
}

echo "Cube reference grid-convergence study"
echo "  Re=1000, dt=0.01, end=$end_time, ranks=$cores, nice=$nice_level, cpu quota=${cpu_quota_percent}%"
echo "  start gates: RAM>=${minimum_available_gib} GiB, load<=${maximum_load}, temp<=${maximum_start_temperature} C"
echo "  in-run thermal guard: pause>${maximum_run_temperature} C, resume<=${resume_run_temperature} C"

if [ "$mode" != "analyse-only" ]; then
    run_grid h16 0.0625
    run_grid h32 0.03125
fi

if [ "$mode" = "dry-run" ]; then
    echo "Would compact the existing samples into $samples_root/h64 and analyse all three grids."
    exit 0
fi

"$python_bin" "$analyser" prepare-fine \
    --source "$samples_root" \
    --destination "$samples_root/h64" \
    --line-interval "$line_interval"

"$python_bin" "$analyser" analyse \
    --samples-root "$samples_root" \
    --output "$samples_root/grid_convergence_report.json" \
    --window-start "$window_start" \
    --window-end "$end_time" \
    --drag-tolerance "$drag_tolerance" \
    --field-tolerance "$field_tolerance"

echo
echo "===== GRID STUDY COMPLETE ====="
echo "Report: $samples_root/grid_convergence_report.md"
if [ -f "$samples_root/grid_recommendation.env" ]; then
    echo "Load-converged settings: $samples_root/grid_recommendation.env"
    echo "To use them for a production reference run:"
    echo "  set -a; . samples/grid_recommendation.env; set +a; ./allrun.sh"
else
    echo "No mesh recommendation was emitted; inspect the report before extending the study."
fi
