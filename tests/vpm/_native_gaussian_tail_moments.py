"""UNWIRED O(N) native omitted-tail and analytic-completion qualification."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.vpm._gaussian_image_tail_moments import moment_tail_bound
from tests.vpm._native_gaussian_tail_bounds import digest, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    proof = Path(__file__).with_name("_gaussian_image_tail_moments.py")
    helper = Path(__file__).with_name("_native_gaussian_tail_bounds.py")
    inputs = [Path(__file__), proof, helper, args.checkpoint, args.reference_report]
    hashes = {str(path.resolve()): digest(path) for path in inputs}
    reference = json.loads(args.reference_report.read_text())
    if digest(args.checkpoint) != reference["identity"]["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from native field evidence")
    start = perf_counter()
    with h5py.File(args.checkpoint, "r") as saved:
        config = json.loads(saved["solver"].attrs["numerical_configuration"])
        if config != reference["identity"]["configuration"]:
            raise ValueError("checkpoint numerical configuration differs from field evidence")
        x = saved["particles/position"][:].astype(np.float64)
        gamma = saved["particles/vortex_strength"][:].astype(np.float64)
        sigma = saved["particles/core_radius"][:].astype(np.float64)
        time = float(saved["solver"].attrs["time"])
    if len(x) != reference["identity"]["particles"] or time != reference["identity"]["time"]:
        raise ValueError("native point count/time mismatch")
    induction = config["induction"]
    if config["particle_kernel"] != "GAUSSIAN" or induction["method"] != "SLIP_SLAB":
        raise ValueError("Gaussian slab configuration required")
    uscale, jscale = induction["velocity_scale"], induction["gradient_scale"]
    tolerance = induction["tail_tolerance"]
    records = []
    for k in (16, 32, 64, 128, 256):
        bound = moment_tail_bound(x, gamma, sigma, x, z_min=induction["z_min"],
                                  z_max=induction["z_max"], shells=k)
        remainder_u = bound.singular_velocity_remainder+bound.gaussian_velocity_defect
        remainder_j = bound.singular_gradient_remainder+bound.gaussian_gradient_defect
        records.append({
            "K": k, "explicit_images_excluding_primary": 4*k+1,
            "omitted_whole_tail": {
                "velocity_absolute": summary(bound.velocity_bound, tolerance*uscale, x),
                "gradient_frobenius_absolute": summary(bound.gradient_bound, tolerance*jscale, x),
                "maximum_scaled_bound": float(np.maximum(bound.velocity_bound/uscale,
                                                           bound.gradient_bound/jscale).max()),
            },
            "residual_after_analytic_leading_tail_completion": {
                "velocity_absolute": summary(remainder_u, tolerance*uscale, x),
                "gradient_frobenius_absolute": summary(remainder_j, tolerance*jscale, x),
                "maximum_scaled_bound": float(np.maximum(remainder_u/uscale, remainder_j/jscale).max()),
            },
            "leading_velocity_max_norm": float(np.linalg.norm(bound.leading_velocity, axis=1).max()),
            "leading_gradient_max_frobenius": float(np.linalg.norm(bound.leading_gradient, axis=(1, 2)).max()),
            "gaussian_velocity_defect_max": float(bound.gaussian_velocity_defect.max()),
            "gaussian_gradient_defect_max": float(bound.gaussian_gradient_defect.max()),
            "moment_diagnostics": bound.diagnostics,
        })
    if any(digest(path) != hashes[str(path.resolve())] for path in inputs):
        raise RuntimeError("qualification source or preserved evidence changed")
    result = {
        "status": "complete", "particles": len(x), "time": time, "input_sha256": hashes,
        "induction_configuration": induction, "records": records,
        "seconds": perf_counter()-start,
        "targets": "every physical native particle; source moments and target AABB extents require O(N), no pair matrix",
        "scope": "Real-arithmetic Gaussian infinite-image truncation bounds after finite +/-K shells, with and without derivative-consistent analytic leading paired-tail completion. Does not include existing finite-field, FFT, interpolation or f32/f64 evaluation error.",
        "arithmetic": "Padded ordinary f64, not a directed interval certificate; Gaussian defect is explicit with a positive1e-300 upper floor per family.",
        "production_admissible": False, "unchanged_config": True,
        "original_two_consecutive_block_gate_replaced": False,
        "required_before_adoption": "A new explicit stopping contract, total-error budget including finite-field error and arithmetic, independent validation and matched continuation. Passing this analytic truncation inequality does not pass the current two-consecutive-block gate.",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: row[key] for key in ("K", "leading_velocity_max_norm", "leading_gradient_max_frobenius")}
                     | {"whole_tail": row["omitted_whole_tail"]["maximum_scaled_bound"],
                        "completed_tail_residual": row["residual_after_analytic_leading_tail_completion"]["maximum_scaled_bound"]}
                     for row in records], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
