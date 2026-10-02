"""Read-only native whole-tail and optional completion certificate census."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.vpm._gaussian_tail_certificate import normalized_bound, tail_certificate
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
    helpers = [source.with_name(name) for name in ("_gaussian_tail_arithmetic.py", "_gaussian_tail_certificate.py",
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
        certificate = tail_certificate(x, g, sigma, x, z_min=config["z_min"], z_max=config["z_max"], shells=k)
        record = {"K": k, "finite_images_excluding_primary": 4*k+1, "diagnostics": certificate.diagnostics}
        for kind in ("whole", "completed"):
            u, j = getattr(certificate, f"{kind}_velocity"), getattr(certificate, f"{kind}_gradient")
            normalized = normalized_bound(u, j, velocity_scale=config["velocity_scale"], gradient_scale=config["gradient_scale"])
            record[kind] = {
                "velocity": summary(u, config["tail_tolerance"]*config["velocity_scale"], x),
                "gradient": summary(j, config["tail_tolerance"]*config["gradient_scale"], x),
                "normalized_maximum": float(normalized.max()),
                "all_targets_pass_analytic_tail_tolerance": bool(np.all(normalized <= config["tail_tolerance"]))}
        records.append(record)
    if any(digest(path) != hashes[str(path.resolve())] for path in paths):
        raise RuntimeError("evidence changed during qualification")
    result = {"status": "complete", "particles": len(x), "input_sha256": hashes,
              "induction_configuration": config, "records": records, "seconds": perf_counter()-start,
              "scope": "Analytic mathematical infinite omitted-image tail enclosures, evaluated with outward basic arithmetic on exact supplied binary64 inputs. Whole-tail policy adds no completion and retains exactly the existing finite image set.",
              "arithmetic_assumptions": "IEEE754 basic operations round-to-nearest with gradual underflow; no concurrent input mutation.",
              "separate_accuracy_obligation": "Finite-image mesh/interpolation/FFT/GPU approximation is not certified by the tail bound. It must be honestly qualified independently; this report does not imply a universal finite-field certificate.",
              "production_admissible": False, "original_two_block_gate_changed": False,
              "no_production_fields_modified": True}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{"K": row["K"], "whole": row["whole"]["normalized_maximum"],
                       "completed": row["completed"]["normalized_maximum"],
                       "all_whole_pass": row["whole"]["all_targets_pass_analytic_tail_tolerance"]}
                      for row in records], indent=2))


if __name__ == "__main__":
    main()
