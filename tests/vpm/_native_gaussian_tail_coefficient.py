"""Read-only coefficient-error attribution for the preserved native tail report."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from tests.vpm._gaussian_tail_coefficient_enclosure import coefficient_enclosure
from tests.vpm._native_gaussian_tail_bounds import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moment-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    helper = Path(__file__).with_name("_gaussian_tail_coefficient_enclosure.py")
    paths = [Path(__file__), helper, args.moment_report, args.checkpoint]
    hashes = {str(path.resolve()): digest(path) for path in paths}
    original = json.loads(args.moment_report.read_text())
    if hashes[str(args.checkpoint.resolve())] != original["input_sha256"][str(args.checkpoint.resolve())]:
        raise ValueError("native checkpoint identity mismatch")
    start = perf_counter()
    with h5py.File(args.checkpoint, "r") as saved:
        x = saved["particles/position"][:].astype(np.float64)
    if len(x) != original["particles"]:
        raise ValueError("native particle count mismatch")
    rows = []
    for record in original["records"]:
        diag = record["moment_diagnostics"]
        origin = np.asarray(diag["moment_origin"])
        net = np.asarray(diag["two_family_net_strength"])
        moment = np.asarray(diag["two_family_cross_first_moment"])
        period = diag["period"]
        u_factor = (np.cross(net, (x-origin)*[1., 1., -2.])-moment)/(2*np.pi*period**3)
        j_factor = np.array([[0., -net[2], net[1]], [net[2], 0., -net[0]],
                             [-net[1], net[0], 0.]])*np.array([1., 1., -2.])[None, :]/(2*np.pi*period**3)
        enclosure = coefficient_enclosure(record["K"])
        # These norm/field-scale calculations intentionally retain the prior
        # ordinary-f64 status; only the scalar coefficient is enclosed here.
        u_charge = enclosure.radius*float(np.linalg.norm(u_factor, axis=1).max())
        j_charge = enclosure.radius*float(np.linalg.norm(j_factor))
        previous = record["residual_after_analytic_leading_tail_completion"]
        rows.append({"K": record["K"], "coefficient": asdict(enclosure),
                     "velocity_coefficient_only_charge_max": u_charge,
                     "gradient_coefficient_only_charge_max": j_charge,
                     "previous_velocity_remainder_max": previous["velocity_absolute"]["maximum"],
                     "previous_gradient_remainder_max": previous["gradient_frobenius_absolute"]["maximum"],
                     "remainder_plus_coefficient_velocity": previous["velocity_absolute"]["maximum"]+u_charge,
                     "remainder_plus_coefficient_gradient": previous["gradient_frobenius_absolute"]["maximum"]+j_charge})
    if any(digest(path) != hashes[str(path.resolve())] for path in paths):
        raise RuntimeError("input evidence changed during coefficient attribution")
    result = {"status": "complete", "input_sha256": hashes, "records": rows,
              "seconds": perf_counter()-start, "particles": len(x),
              "coefficient_scope": "Positive prefix terms/divisions/sums and integral endpoints are outward-rounded binary64 enclosures; the midpoint radius includes midpoint rounding. No scipy zeta is used.",
              "field_scope": "Charges use the prior ordinary-f64 moment/prefactor values. They quantify the coefficient uncertainty but do not upgrade those moments, Gaussian defect or finite-image arithmetic to interval-certified fields.",
              "production_admissible": False, "original_gate_changed": False}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: row[key] for key in ("K", "velocity_coefficient_only_charge_max",
                      "gradient_coefficient_only_charge_max", "remainder_plus_coefficient_velocity")}
                     for row in rows], indent=2))


if __name__ == "__main__":
    main()
