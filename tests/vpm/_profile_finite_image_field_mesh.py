"""Separate bounded CPU screen of analytic-field finite-image convolution."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._finite_image_field_mesh_reference import finite_image_field_mesh
from tests.vpm._finite_image_mesh_reference import direct_finite_images, observed_error
from tests.vpm._profile_finite_image_mesh import cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(__file__).with_name(name) for name in (
        "_profile_finite_image_field_mesh.py", "_finite_image_field_mesh_reference.py",
        "_finite_image_mesh_reference.py", "_profile_finite_image_mesh.py",
        "_gaussian_broadening_reference.py", "_slip_periodic_gaussian_oracle.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    records = []
    cases_to_run = list(cases())
    x = np.array([[.013, -.017, .081], [.036, .027, .137]])
    cases_to_run.append(("noncommensurate_reflected_coincidence", x,
                         np.array([[.2, .7, -.4], [0., 0., 0.]]), np.full(2, .04), x[:1], [(.162, True)]))
    for name, x, gamma, sigma, query, images in cases_to_run:
        u, j, conditioning = direct_finite_images(x, gamma, sigma, query, images)
        for tau in (.08, .10, .12):
            for order in (8, 10):
                started = time.perf_counter()
                result = finite_image_field_mesh(x, gamma, sigma, query, images, tau=tau, spacing=.04, order=order)
                velocity = observed_error(result.velocity, u, conditioning[:, 0])
                gradient = observed_error(result.gradient, j, conditioning[:, 1])
                velocity_ok = (velocity["relative_l2"] < 1e-4 if velocity["relative_l2"] is not None
                               else velocity["absolute_l2"] < 1e-5)
                record = {"case": name, "spacing": .04, "tau": tau, "tau_over_h": tau/.04, "order": order,
                          "velocity": velocity, "gradient": gradient, "runtime_admissible": False,
                          "passed_small_cloud_norm_screen": bool(velocity_ok and gradient["relative_l2"] < 1e-4),
                          "trace_max_abs": float(np.max(np.abs(np.trace(result.gradient, axis1=1, axis2=2)))),
                          "cpu_seconds": time.perf_counter()-started, "diagnostics": result.diagnostics}
                records.append(record)
                print(json.dumps({key: record[key] for key in ("case", "tau", "order", "passed_small_cloud_norm_screen")} |
                                 {"u_error": velocity["relative_l2"], "u_absolute": velocity["absolute_l2"],
                                  "J_error": gradient["relative_l2"]}), flush=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise RuntimeError("qualification source changed during run")
    report = {"status": "complete", "runtime_admissible": False, "source_hashes": before,
              "records": records, "qualification_limits": [
                  "Analytic J is not identically the derivative of interpolated u.",
                  "No mesh, FFT, interpolation or roundoff error certificate.",
                  "All local corrections evaluated; CPU time is not production/GPU performance.",
                  "Small-cloud norm screen does not establish old-operator pointwise noninferiority."]}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
