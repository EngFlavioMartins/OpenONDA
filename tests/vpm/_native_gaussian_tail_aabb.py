"""Read-only native prepared-source/AABB tail cost and conservatism census."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.vpm._gaussian_tail_aabb_certificate import prepare_tail_source, query_tail_bound
from tests.vpm._native_gaussian_tail_bounds import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-target-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__)
    helpers = [source.with_name(name) for name in ("_gaussian_tail_aabb_certificate.py",
                 "_gaussian_tail_arithmetic.py", "_gaussian_tail_coefficient_enclosure.py",
                 "_native_gaussian_tail_bounds.py")]
    paths = [source, *helpers, args.all_target_report, args.checkpoint]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    original = json.loads(args.all_target_report.read_text())
    if hashes[str(args.checkpoint.resolve())] != original["input_sha256"][str(args.checkpoint.resolve())]:
        raise ValueError("native checkpoint identity mismatch")
    with h5py.File(args.checkpoint, "r") as saved:
        x = saved["particles/position"][:].astype(np.float64)
        g = saved["particles/vortex_strength"][:].astype(np.float64)
        sigma = saved["particles/core_radius"][:].astype(np.float64)
        configuration = json.loads(saved["solver"].attrs["numerical_configuration"])
    config = configuration["induction"]
    if len(x) != original["particles"] or config != original["induction_configuration"]:
        raise ValueError("native input mismatch")
    begin = perf_counter()
    prepared = prepare_tail_source(x, g, sigma, z_min=config["z_min"], z_max=config["z_max"])
    preparation_seconds = perf_counter()-begin
    begin = perf_counter()
    lower, upper = x.min(axis=0), x.max(axis=0)
    target_bounds_seconds = perf_counter()-begin
    previous = {r["K"]: r for r in original["records"]}
    records = []
    for k in (32, 64, 128):
        begin = perf_counter()
        bound = query_tail_bound(prepared, lower, upper, shells=k)
        seconds = perf_counter()-begin
        individual = previous[k]["whole"]
        if (bound.velocity_upper < individual["velocity"]["maximum"]
                or bound.gradient_upper < individual["gradient"]["maximum"]):
            raise AssertionError("continuous-box bound did not dominate native per-target bounds")
        records.append({"K": k, "bound": asdict(bound), "query_seconds": seconds,
                        "whole_query_box_passes_tail_tolerance": bool(
                            bound.velocity_upper <= config["tail_tolerance"]*config["velocity_scale"]
                            and bound.gradient_upper <= config["tail_tolerance"]*config["gradient_scale"]),
                        "previous_all_target_velocity": individual["velocity"]["maximum"],
                        "previous_all_target_gradient": individual["gradient"]["maximum"]})
    if any(digest(path) != hashes[str(path.resolve())] for path in paths):
        raise RuntimeError("qualification source or evidence changed")
    result = {"status": "complete", "particles": len(x), "input_sha256": hashes,
              "induction_configuration": config, "records": records,
              "preparation_seconds": preparation_seconds, "target_bounds_seconds": target_bounds_seconds,
              "source_snapshot_sha256": prepared.source_sha256,
              "scope": "Immutable source-moment snapshot plus O(1) continuous query-AABB tail bounds. No finite field computed or changed.",
              "ownership": "Snapshot represents copied source values at preparation; a future runtime must enforce an explicit immutable source epoch before reusing it.",
              "arithmetic": "IEEE754 round-to-nearest basic operations with gradual underflow; all relevant operations outward-enclosed.",
              "excluded": "Finite-image interpolation/FFT/GPU accuracy; this is not a whole-operator certificate.",
              "production_admissible": False, "original_gate_changed": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"preparation_seconds": preparation_seconds, "target_bounds_seconds": target_bounds_seconds,
                      "records": [{"K": row["K"], "u": row["bound"]["velocity_upper"],
                                   "J": row["bound"]["gradient_upper"], "seconds": row["query_seconds"],
                                   "pass": row["whole_query_box_passes_tail_tolerance"]} for row in records]}, indent=2))


if __name__ == "__main__":
    main()
