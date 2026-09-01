#!/usr/bin/env python3
"""Measure end-to-end accepted VPM step cost at production particle counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import setup

from assets.run_full_checkpoint import CONFIGS, build_solver, load_checkpoint, upload, write_csv

COUNTS = (4_000, 14_000, 35_000, 70_200)
EXACT_70K_LIMIT_SECONDS = 120.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", choices=CONFIGS, required=True)
    args = parser.parse_args()
    setup.mkdirs()
    configuration = args.configuration
    solver = build_solver("rotor", configuration, max(COUNTS) + 32)
    rows: list[dict] = []
    prediction = None
    try:
        for n in COUNTS:
            if (
                configuration == "exact_pair_rk3_isolated"
                and n == 70_200
                and (prediction is None or prediction > EXACT_70K_LIMIT_SECONDS)
            ):
                rows.append(
                    {
                        "particles": n,
                        "configuration": configuration,
                        "status": "skipped_by_measured_feasibility_gate",
                        "predicted_step_wall_s": prediction,
                        "gate_s": EXACT_70K_LIMIT_SECONDS,
                    }
                )
                write_csv(setup.RESULTS / f"scale_timing_{configuration}.csv", rows)
                continue
            data = load_checkpoint("rotor", n)
            if len(solver) == 0:
                upload(solver, data)
            else:
                upload(solver, data, replace=True)
            # Compile and populate all production paths, then restore the state so
            # the timed step always begins at the checkpoint.
            solver.advance(defer_output=True)
            solver.synchronize()
            upload(solver, data, replace=True)
            start = time.perf_counter()
            solver.advance(defer_output=True)
            solver.synchronize()
            elapsed = time.perf_counter() - start
            row = {
                "particles": n,
                "configuration": configuration,
                "status": "measured",
                "complete_step_wall_s": elapsed,
                "precision": "f32",
                "device_request": "VULKAN",
                "source_checkpoint": "rotor_000520_uniform_index_subset",
                "includes": "velocity,all_RK_stages,stretching,viscosity_or_inviscid,LES_if_production,stabilization_if_production",
                "forcing": "none"
                if configuration != "production_numerics_unforced"
                else "freestream_only_no_vlm",
            }
            rows.append(row)
            if configuration == "exact_pair_rk3_isolated" and n == 35_000:
                prediction = elapsed * (70_200 / 35_000) ** 2
                row["predicted_70200_step_wall_s_quadratic"] = prediction
            write_csv(setup.RESULTS / f"scale_timing_{configuration}.csv", rows)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        solver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
