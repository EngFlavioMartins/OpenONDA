#!/usr/bin/env bash
# Continue the qualified cylinder study without touching the completed preflight.
set -euo pipefail

case_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$case_dir"

python -c '
import json
from pathlib import Path

quality = json.loads(Path("solution/mesh_quality_study.json").read_text())
if quality.get("status") != "passed":
    raise SystemExit("mesh-quality study has not passed")

diagnostics = Path("solution/very_coarse/diagnostics.jsonl")
last = json.loads(diagnostics.read_text().splitlines()[-1])
if abs(float(last["time"]) - 60.0) > 1.0e-10:
    raise SystemExit("very_coarse is not complete at t=60")
'

run_case() {
    local name="$1"
    local dx="$2"
    if [[ -e "solution/$name" || -e "samples/$name" ]]; then
        if [[ -d "solution/$name" && -d "samples/$name" ]] && python -c '
import json
from pathlib import Path
import sys

diagnostics = Path("solution") / sys.argv[1] / "diagnostics.jsonl"
last = json.loads(diagnostics.read_text().splitlines()[-1])
raise SystemExit(0 if abs(float(last["time"]) - 60.0) <= 1.0e-10 else 1)
' "$name"; then
            echo "Preserving completed $name output"
            return 0
        fi
        echo "Refusing to overwrite incomplete existing $name output" >&2
        return 1
    fi
    python -u setup.py --dx "$dx" --case-name "$name"
    python -c '
import json
from pathlib import Path
import sys

name = sys.argv[1]
diagnostics = Path("solution") / name / "diagnostics.jsonl"
last = json.loads(diagnostics.read_text().splitlines()[-1])
if abs(float(last["time"]) - 60.0) > 1.0e-10:
    raise SystemExit(f"{name} ended at t={last['"'"'time'"'"']}, expected t=60")
' "$name"
}

run_case coarse 0.041666666666666664
run_case medium 0.027777777777777776
run_case fine 0.018518518518518517

python assets/postprocess.py
python assets/plot_grid_study.py
