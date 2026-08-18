#!/usr/bin/env bash
#
# Run the cube coupling study end to end, in order, unattended.
#
# The stages are sequential because each one's input is the previous one's
# output: the oracle replays the reference's boundary trace, and the hybrid is
# only interpretable against the oracle on the same mesh.
#
# State lives in tutorials/coupled_FVM_VPM/cubeFlow/sequence/.  Completed stages
# are skipped on re-invocation, so an interrupted run resumes where it stopped.
#
#   ./scripts/experiments/run_cube_sequence.sh              # run everything
#   ./scripts/experiments/run_cube_sequence.sh --status     # what happened so far
#   ./scripts/experiments/run_cube_sequence.sh --from oracle
#   ./scripts/experiments/run_cube_sequence.sh --only smoke
#   ./scripts/experiments/run_cube_sequence.sh --redo reference
#
# Unattended:  nohup ./scripts/experiments/run_cube_sequence.sh > sequence.log 2>&1 &
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUBE="$REPO/tutorials/coupled_FVM_VPM/cubeFlow"
SEQ="$CUBE/sequence"
FLAG="$SEQ/CALL_CLAUDE"
TIMELINE="$SEQ/TIMELINE.log"

STAGES=(reference oracle smoke hybrid)

# Stages after which there is something for Claude to analyse.  The others are
# either preparation or a pass/fail gate that only needs attention on failure.
is_analysis_stage() { [[ "$1" == "oracle" || "$1" == "hybrid" ]]; }

mkdir -p "$SEQ"

# ---------------------------------------------------------------- presentation
bar() { printf '%s\n' "================================================================"; }
say() { printf '%s\n' "$*"; }
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
note() { printf '[%s] %s\n' "$(stamp)" "$*" >>"$TIMELINE"; }

banner() {  # banner <symbol> <line1> [line2...]
    local sym="$1"; shift
    echo; bar; printf '  %s  %s\n' "$sym" "$1"; shift
    for line in "$@"; do printf '     %s\n' "$line"; done
    bar; echo
}

call_claude() {  # call_claude <stage> <what to say>
    local stage="$1" phrase="$2"
    { echo "Stage '$stage' finished at $(stamp) and is ready to analyse."
      echo
      echo "Say to Claude:  $phrase"
    } >"$FLAG"
    banner "⏸" "ANALYSIS CHECKPOINT — call Claude back now" \
        "" "    say:  $phrase" "" \
        "Later stages do not depend on this analysis; the run continues."
}

# ------------------------------------------------------------------ arguments
FROM=""; ONLY=""; REDO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)
            say "sequence state in $SEQ"; echo
            for s in "${STAGES[@]}"; do
                if   [[ -f "$SEQ/$s.done"    ]]; then printf '  %-10s done\n'    "$s"
                elif [[ -f "$SEQ/$s.failed"  ]]; then printf '  %-10s FAILED\n'  "$s"
                elif [[ -f "$SEQ/$s.running" ]]; then printf '  %-10s running\n' "$s"
                else                                  printf '  %-10s pending\n' "$s"; fi
            done
            echo; [[ -f "$TIMELINE" ]] && { say "timeline:"; cat "$TIMELINE"; }
            [[ -f "$FLAG" ]] && { echo; bar; cat "$FLAG"; bar; }
            exit 0 ;;
        --from) FROM="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --redo) REDO="$2"; rm -f "$SEQ/$2.done" "$SEQ/$2.failed"; shift 2 ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) say "unknown argument: $1"; exit 2 ;;
    esac
done

should_run() {  # should_run <stage>
    local s="$1"
    [[ -n "$ONLY" && "$s" != "$ONLY" ]] && return 1
    if [[ -n "$FROM" ]]; then
        local seen=0
        for t in "${STAGES[@]}"; do
            [[ "$t" == "$FROM" ]] && seen=1
            [[ "$t" == "$s" ]] && { [[ $seen -eq 1 ]] && return 0 || return 1; }
        done
    fi
    [[ -f "$SEQ/$s.done" ]] && { say "-- skipping '$s' (already done; --redo $s to force)"; return 1; }
    return 0
}

run_stage() {  # run_stage <stage> <description> <command...>
    local stage="$1" desc="$2"; shift 2
    should_run "$stage" || return 0
    rm -f "$SEQ/$stage.failed" "$FLAG"
    : >"$SEQ/$stage.running"
    banner "▶" "STAGE ${stage} — ${desc}" "started $(stamp)" "log: sequence/${stage}.log"
    note "start  $stage"
    local t0 rc; t0=$(date +%s)
    ( "$@" ) >"$SEQ/$stage.log" 2>&1
    rc=$?
    local mins=$(( ($(date +%s) - t0) / 60 ))
    rm -f "$SEQ/$stage.running"
    if [[ $rc -ne 0 ]]; then
        : >"$SEQ/$stage.failed"
        note "FAILED $stage (rc=$rc, ${mins}m)"
        { echo "Stage '$stage' FAILED at $(stamp) after ${mins} min (exit $rc)."
          echo "Tail of sequence/${stage}.log:"; echo
          tail -30 "$SEQ/$stage.log"
        } >"$FLAG"
        banner "❌" "STAGE ${stage} FAILED after ${mins} min (exit ${rc})" \
            "" "    say:  \"$stage failed\"" "" \
            "Sequence stopped. Nothing after this stage can be interpreted."
        exit "$rc"
    fi
    : >"$SEQ/$stage.done"
    note "done   $stage (${mins}m)"
    banner "✅" "STAGE ${stage} complete in ${mins} min"
    if is_analysis_stage "$stage"; then
        call_claude "$stage" "\"$stage done\""
    fi
}

# --------------------------------------------------------------------- stages

# A blow-up does not always give a non-zero exit, so check the physics too.
guard_reference() {
    cd "$CUBE/referenceFlow" || return 1
    # allclean.sh drops solution/ and constant/ but keeps samples/.  The stale
    # traces carry the inflated geometry at a different cadence and share the
    # step-numbered filenames the new run writes, so clear them explicitly.
    rm -rf samples
    ./allrun.sh || return 1
    python3 - <<'PY' || return 1
import csv, math, sys, pathlib
p = pathlib.Path("samples/forces_history.csv")
rows = list(csv.DictReader(open(p)))
if not rows:
    sys.exit("reference produced no force history")
cd = [float(r["Cd"]) for r in rows]
t = [float(r["time"]) for r in rows]
if not all(math.isfinite(c) for c in cd):
    sys.exit("reference Cd contains non-finite values (blow-up)")
tail = [c for tt, c in zip(t, cd) if tt >= 15.0]
if not tail:
    sys.exit(f"reference stopped early at t={t[-1]:.2f}")
mean = sum(tail) / len(tail)
print(f"reference settled Cd(t>=15) = {mean:.4f} over {len(tail)} samples, t_end={t[-1]:.2f}")
if not 0.7 < mean < 1.5:
    sys.exit(f"reference settled Cd {mean:.4f} is implausible")
PY
}

guard_hybrid() {
    cd "$CUBE" || return 1
    ./allrun.sh || return 1
    python3 - <<'PY' || return 1
import csv, math, sys, pathlib
p = pathlib.Path("samples/forces_history.csv")
if not p.exists():
    sys.exit("hybrid produced no force history")
rows = list(csv.DictReader(open(p)))
cd = [float(r["Cd"]) for r in rows]
if not cd or not all(math.isfinite(c) for c in cd):
    sys.exit("hybrid Cd contains non-finite values (blow-up)")
print(f"hybrid reached t={float(rows[-1]['time']):.2f}, {len(rows)} samples, last Cd={cd[-1]:.4f}")
PY
}

smoke_hybrid() {
    cd "$CUBE" || return 1
    OPENONDA_SMOKE=1 ./allrun.sh
}

cd "$REPO"

# The reference alone writes ~3.6 GB of per-step face traces plus its volumes.
free_gb=$(df -g "$REPO" | awk 'NR==2 {print $4}')
if [[ -n "${free_gb:-}" && "$free_gb" -lt 20 ]]; then
    banner "⚠" "Only ${free_gb} GB free — the sequence wants ~20 GB" \
        "" "Stage 1 frees ~6 GB when it clears the old reference output," \
        "but per-step face sampling then writes ~3.6 GB back."
fi

banner "🧊" "CUBE COUPLING SEQUENCE" \
    "repo:  $REPO" \
    "state: $SEQ" \
    "" \
    "Mesh fix applied: both cases now resolve an exact unit cube at h=0.015625." \
    "The pre-fix reference data is superseded and will be replaced."

# Preserve the small pre-fix force histories.  The face traces are not worth
# keeping: they carry the inflated geometry and are regenerated by stage 1.
if [[ ! -d "$SEQ/archive_pre_fix" ]]; then
    mkdir -p "$SEQ/archive_pre_fix"
    for f in "$CUBE/referenceFlow/samples/forces_history.csv" \
             "$CUBE/referenceFlow/samples/centerline.csv" \
             "$CUBE/referenceFlow/samples/offaxis_y075.csv" \
             "$CUBE/oracleFlow/samples/forces_history.csv" \
             "$CUBE/oracleFlow_k18/samples/forces_history.csv"; do
        [[ -f "$f" ]] || continue
        tag="$(basename "$(dirname "$(dirname "$f")")")_$(basename "$f")"
        cp "$f" "$SEQ/archive_pre_fix/$tag"
    done
    note "archived pre-fix force histories"
    say "-- archived pre-fix force histories to sequence/archive_pre_fix/"
fi

run_stage reference "full-mesh ground truth, ~4-5 h" guard_reference
run_stage oracle    "exact-trace replay on the matched mesh, ~30 min" \
    python "$REPO/scripts/experiments/cube_oracle_bc.py" --t-end 3.0 --cores 4
run_stage smoke     "hybrid configuration check, minutes" smoke_hybrid
run_stage hybrid    "full coupled run" guard_hybrid

banner "🏁" "SEQUENCE COMPLETE" "$(stamp)" "" "    say:  \"sequence done\""
{ echo "All stages complete at $(stamp)."; echo; echo 'Say to Claude:  "sequence done"'; } >"$FLAG"
