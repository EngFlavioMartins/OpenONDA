"""NEW read-only all-target enclosed infinite-image completion remainder."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.vpm._gaussian_tail_arithmetic import add, leading_tail, point
from tests.vpm._gaussian_tail_remainder_interval import tail_remainder
from tests.vpm._native_gaussian_tail_bounds import digest, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moment-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__)
    helpers = [source.with_name(name) for name in ("_gaussian_tail_arithmetic.py",
                 "_gaussian_tail_coefficient_enclosure.py", "_gaussian_tail_remainder_interval.py",
                 "_native_gaussian_tail_bounds.py")]
    paths = [source, *helpers, args.moment_report, args.checkpoint]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    original = json.loads(args.moment_report.read_text())
    if hashes[str(args.checkpoint.resolve())] != original["input_sha256"][str(args.checkpoint.resolve())]:
        raise ValueError("native checkpoint identity mismatch")
    start = perf_counter()
    with h5py.File(args.checkpoint, "r") as saved:
        x = saved["particles/position"][:].astype(np.float64)
        g = saved["particles/vortex_strength"][:].astype(np.float64)
        sigma = saved["particles/core_radius"][:].astype(np.float64)
        configuration = json.loads(saved["solver"].attrs["numerical_configuration"])
    config = configuration["induction"]
    if (len(x) != original["particles"] or config != original["induction_configuration"]
            or configuration["particle_kernel"] != "GAUSSIAN"):
        raise ValueError("native input mismatch")
    records = []
    for k in (32, 64, 128):
        leading = leading_tail(x, g, x, z_min=config["z_min"], z_max=config["z_max"], shells=k)
        remainder = tail_remainder(x, g, sigma, x, z_min=config["z_min"], z_max=config["z_max"], shells=k)
        total_u = add(point(leading.velocity_error_bound), point(remainder.velocity)).upper
        total_j = add(point(leading.gradient_error_bound), point(remainder.gradient)).upper
        records.append({"K": k,
                        "velocity_completed_tail_error": summary(total_u, config["tail_tolerance"]*config["velocity_scale"], x),
                        "gradient_completed_tail_error": summary(total_j, config["tail_tolerance"]*config["gradient_scale"], x),
                        "leading_arithmetic_velocity_max": float(leading.velocity_error_bound.max()),
                        "leading_arithmetic_gradient_max": float(leading.gradient_error_bound.max()),
                        "singular_remainder_velocity_max": float(remainder.singular_velocity.max()),
                        "singular_remainder_gradient_max": float(remainder.singular_gradient.max()),
                        "rational_gaussian_velocity_max": float(remainder.gaussian_velocity.max()),
                        "rational_gaussian_gradient_max": float(remainder.gaussian_gradient.max()),
                        "remainder_diagnostics": remainder.diagnostics})
    if any(digest(path) != hashes[str(path.resolve())] for path in paths):
        raise RuntimeError("evidence changed during qualification")
    result = {"status": "complete", "particles": len(x), "input_sha256": hashes,
              "induction_configuration": config, "records": records, "seconds": perf_counter()-start,
              "scope": "Enclosed mathematical infinite-image tail error after adding stored affine leading completion: singular Taylor remainder plus Gaussian defect plus all leading coefficient/moment/field arithmetic. Exact supplied binary64 inputs; IEEE754 basic round-to-nearest operations with gradual underflow.",
              "excluded": ["finite-image approximation, interpolation and FFT", "finite-image GPU arithmetic", "adding completion to stored finite fields", "physical primary/self operator error"],
              "production_admissible": False, "original_two_block_gate_changed": False,
              "remaining_validation": "Independent review, empirical full finite-field accuracy/repeatability and matched diagnostic continuation. This analytic certificate does not certify finite mesh error."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{"K": record["K"], "velocity": record["velocity_completed_tail_error"]["maximum"],
                       "gradient": record["gradient_completed_tail_error"]["maximum"],
                       "all_velocity_pass": record["velocity_completed_tail_error"]["all_targets_bounded_at_requested_tolerance"],
                       "all_gradient_pass": record["gradient_completed_tail_error"]["all_targets_bounded_at_requested_tolerance"]}
                      for record in records], indent=2))


if __name__ == "__main__":
    main()
