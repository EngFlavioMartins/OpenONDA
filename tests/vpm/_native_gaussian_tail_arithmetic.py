"""Read-only native leading-tail rounding and rational-defect qualification."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.vpm._gaussian_tail_arithmetic import (
    add,
    gaussian_defect_upper,
    leading_tail,
    mul,
    point,
    sub,
)
from tests.vpm._native_gaussian_tail_bounds import digest


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
                                                  "_gaussian_tail_coefficient_enclosure.py")]
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
    if len(x) != original["particles"] or config != original["induction_configuration"]:
        raise ValueError("native input mismatch")
    period = mul(point(2.), sub(point(config["z_max"]), point(config["z_min"])))
    span = sub(point(x.max(axis=0)), point(x.min(axis=0)))
    # L1 distance dominates Euclidean distance. These deliberately coarse
    # global extents suffice for the tiny Gaussian defect, not Taylor bounds.
    xy = add(point(float(span.upper[0])), point(float(span.upper[1])))
    physical_extent = add(xy, point(float(span.upper[2])))
    reflected_extent = add(xy, mul(point(2.), sub(point(float(x[:, 2].max())), point(config["z_min"]))))
    rows = []
    for k in (32, 64, 128):
        result = leading_tail(x, g, x, z_min=config["z_min"], z_max=config["z_max"], shells=k)
        defects = []
        for extent in (physical_extent, reflected_extent):
            gap = sub(mul(period, point(float(k))), point(float(extent.upper)))
            defects.append(gaussian_defect_upper(
                gap_lower=float(gap.lower), sigma_upper=float(sigma.max()),
                period_lower=float(period.lower),
                absolute_strength_upper=result.moments["source_l1_strength_upper"]))
        rows.append({"K": k,
                     "velocity_leading_rounding_and_coefficient_bound_max": float(result.velocity_error_bound.max()),
                     "gradient_leading_rounding_and_coefficient_bound_max": float(result.gradient_error_bound.max()),
                     "moment_intervals": {name: [float(result.moments[name].lower), float(result.moments[name].upper)]
                                          for name in ("net_z", "moment_x", "moment_y", "coefficient")},
                     "h3": asdict(result.moments["h3"]),
                     "source_l1_strength_upper": result.moments["source_l1_strength_upper"],
                     "gaussian_family_bounds": defects})
    if any(digest(path) != hashes[str(path.resolve())] for path in paths):
        raise RuntimeError("evidence changed during qualification")
    output = {"status": "complete", "particles": len(x), "input_sha256": hashes, "records": rows,
              "seconds": perf_counter()-start,
              "scope": "The leading infinite-tail field and its stored affine evaluation are enclosed, including signed moment aggregation, coefficient uncertainty, pi and period. Gaussian defects use outward rational bounds, no libm transcendental accuracy assumption.",
              "assumptions": "IEEE754 binary64 round-to-nearest basic operations and gradual underflow on exact supplied floating inputs; no concurrent mutation.",
              "not_certified_here": ["singular Taylor remainder evaluation", "finite image mesh approximation", "FFT and GPU arithmetic", "complete induction or runtime stop"],
              "production_admissible": False, "existing_gate_changed": False}
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
