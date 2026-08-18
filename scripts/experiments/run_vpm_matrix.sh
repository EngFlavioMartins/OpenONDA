#!/usr/bin/env bash
#
# Run the VPM/coupling isolation matrix.
#
# Each variant is an independent coupled run, so the order below is by
# information value, not dependency: phase B (one component at a time) first,
# because that is what identifies the culprit; phase C (resolution floor) and
# phase D (everything on) afterwards to put the answer in context.
#
# State lives in tutorials/coupled_FVM_VPM/cubeFlow/matrix/.  Completed variants
# are skipped, so an interrupted sweep resumes where it stopped.
#
#   ./scripts/experiments/run_vpm_matrix.sh                 # phases A+B
#   ./scripts/experiments/run_vpm_matrix.sh --phase all
#   ./scripts/experiments/run_vpm_matrix.sh --status
#   ./scripts/experiments/run_vpm_matrix.sh --only B6_prune
#   ./scripts/experiments/run_vpm_matrix.sh --redo A0_bare
#
# Unattended:  nohup ./scripts/experiments/run_vpm_matrix.sh > matrix.log 2>&1 &
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUBE="$REPO/tutorials/coupled_FVM_VPM/cubeFlow"
OUT="$CUBE/matrix"
FLAG="$OUT/CALL_CLAUDE"
TIMELINE="$OUT/TIMELINE.log"
T_END="${OPENONDA_MATRIX_T_END:-2.4}"
CORES="${OPENONDA_MATRIX_CORES:-4}"
# When a rank raises, MPI leaves the survivors spinning in a collective and
# the run never exits.  The sweep therefore enforces its own deadline instead
# of trusting the process to terminate.
TIMEOUT="${OPENONDA_MATRIX_TIMEOUT:-5400}"

# A0 must run first: every phase-B delta is measured against it.
PHASE_AB=(A0_bare B5_f64cpu B1_treecode B2_gbd B6_prune B7_cap B8_thinbuf B8b_thickbuf
          B9_nopanel B10_resync B3_stretch B4_les)
PHASE_C=(C1_h020 C2_h010 C3_h006 C4_h003)
PHASE_D=(D1_full_h015 D2_production)

mkdir -p "$OUT"

bar() { printf '%s\n' "================================================================"; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '[%s] %s\n' "$(stamp)" "$*" >>"$TIMELINE"; }
banner() {
    local sym="$1"; shift
    echo; bar; printf '  %s  %s\n' "$sym" "$1"; shift
    for line in "$@"; do printf '     %s\n' "$line"; done
    bar; echo
}

PHASE="ab"; ONLY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)
            printf 'matrix state in %s\n\n' "$OUT"
            for v in "${PHASE_AB[@]}" "${PHASE_C[@]}" "${PHASE_D[@]}"; do
                if   [[ -f "$OUT/$v.done"   ]]; then printf '  %-16s done\n'    "$v"
                elif [[ -f "$OUT/$v.failed" ]]; then printf '  %-16s FAILED\n'  "$v"
                else                                 printf '  %-16s pending\n' "$v"; fi
            done
            echo; [[ -f "$TIMELINE" ]] && cat "$TIMELINE"
            [[ -f "$FLAG" ]] && { echo; bar; cat "$FLAG"; bar; }
            exit 0 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --only)  ONLY="$2"; shift 2 ;;
        --redo)  rm -f "$OUT/$2.done" "$OUT/$2.failed"; shift 2 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done

case "$PHASE" in
    # A fast pass over every variant: catches configuration failures that only
    # surface at solver start-up (device/precision, backend support) for minutes
    # instead of hours.  Run this before committing to the full sweep.
    smoke) QUEUE=("${PHASE_AB[@]}" "${PHASE_C[@]}" "${PHASE_D[@]}"); T_END=0.15 ;;
    ab)  QUEUE=("${PHASE_AB[@]}") ;;
    c)   QUEUE=("${PHASE_C[@]}") ;;
    d)   QUEUE=("${PHASE_D[@]}") ;;
    all) QUEUE=("${PHASE_AB[@]}" "${PHASE_C[@]}" "${PHASE_D[@]}") ;;
    *)   echo "unknown phase: $PHASE (smoke|ab|c|d|all)"; exit 2 ;;
esac
[[ -n "$ONLY" ]] && QUEUE=("$ONLY")

banner "🔬" "VPM ISOLATION MATRIX" \
    "variants: ${#QUEUE[@]}   t_end=$T_END   cores=$CORES" \
    "out:      $OUT" \
    "" \
    "Metric is amplitude retention of the interior centreline profile." \
    "Report:   python scripts/experiments/cube_vpm_matrix_report.py"

cd "$REPO"
failed=0
for v in "${QUEUE[@]}"; do
    if [[ -f "$OUT/$v.done" ]]; then
        echo "-- skipping $v (done; --redo $v to force)"
        continue
    fi
    rm -f "$OUT/$v.failed"
    banner "▶" "$v" "started $(stamp)" "log: matrix/$v.log"
    note "start  $v"
    t0=$(date +%s)
    python "$REPO/scripts/experiments/cube_vpm_matrix.py" \
        --variant "$v" --t-end "$T_END" --cores "$CORES" >"$OUT/$v.log" 2>&1 &
    run_pid=$!
    ( sleep "$TIMEOUT"
      if kill -0 "$run_pid" 2>/dev/null; then
          echo "[watchdog] exceeded ${TIMEOUT}s, killing" >>"$OUT/$v.log"
          pkill -f "cube_vpm_matrix.py --variant $v" 2>/dev/null
          kill -9 "$run_pid" 2>/dev/null
      fi ) &
    watchdog=$!
    wait "$run_pid"; rc=$?
    kill "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
    mins=$(( ($(date +%s) - t0) / 60 ))

    # A blow-up does not reliably give a non-zero exit, so check the samples.
    if [[ $rc -eq 0 ]]; then
        python3 - "$OUT/$v/samples/centerline.csv" <<'PY'
import csv, math, sys, pathlib
p = pathlib.Path(sys.argv[1])
if not p.exists():
    sys.exit("no centreline samples written")
rows = list(csv.DictReader(open(p)))
if not rows:
    sys.exit("centreline file is empty")
if not all(math.isfinite(float(r["Ux"])) for r in rows):
    sys.exit("centreline contains non-finite velocity (blow-up)")
print(f"  {len(rows)} centreline samples, to t={float(rows[-1]['flow_time']):.2f}")
PY
        rc=$?
    fi

    if [[ $rc -ne 0 ]]; then
        : >"$OUT/$v.failed"; failed=$((failed + 1))
        note "FAILED $v (rc=$rc, ${mins}m)"
        banner "❌" "$v FAILED after ${mins} min" "continuing with the rest of the sweep"
    else
        : >"$OUT/$v.done"
        note "done   $v (${mins}m)"
        banner "✅" "$v complete in ${mins} min"
    fi
done

banner "🏁" "MATRIX SWEEP COMPLETE" "$(stamp)" "${failed} failed of ${#QUEUE[@]}" \
    "" "    say:  \"matrix done\""
{ echo "Matrix sweep finished at $(stamp): ${failed} failed of ${#QUEUE[@]}."
  echo; echo 'Say to Claude:  "matrix done"'
} >"$FLAG"
python "$REPO/scripts/experiments/cube_vpm_matrix_report.py" || true
