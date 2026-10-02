"""Bounded CPU-only finite-block mesh parameter screen; no production imports."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._finite_image_mesh_reference import (
    direct_finite_images,
    finite_image_mesh,
    observed_error,
)


def cases():
    position = np.array([[-0.082, 0.012, 0.023], [-0.042, 0.012, 0.023],
                         [-0.002, 0.012, 0.023], [0.038, 0.012, 0.023], [0.078, 0.012, 0.023]])
    strength = np.tile([0.3, -0.2, 0.7], (5, 1))
    targets = np.array([[0.011, 0.019, 0.018], [-0.073, 0.058, 0.061], [0.112, -0.036, 0.04]])
    images = [(0.0, True), *[(shift, odd) for shift in (-3.84, -1.92, 1.92, 3.84) for odd in (False, True)]]
    yield "near_plane_nonzero_gamma_z", position, strength, np.full(5, 0.04), targets, images
    yield "near_plane_fourth_moment_cancellation", position, strength*np.array([1, -4, 6, -4, 1])[:, None], np.full(5, 0.04), targets, images
    single = np.array([[0.013, -0.017, 0.0]])
    yield "reflected_coincident", single, np.array([[0.2, 0.7, -0.4]]), np.array([0.04]), single, [(0.0, True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(__file__), Path(__file__).with_name("_finite_image_mesh_reference.py"),
             Path(__file__).with_name("_gaussian_broadening_reference.py"),
             Path(__file__).with_name("_slip_periodic_gaussian_oracle.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    records = []
    for name, x, gamma, sigma, query, images in cases():
        u, j, conditioning = direct_finite_images(x, gamma, sigma, query, images)
        for tau in (0.08, 0.10, 0.12):
            for order in (8, 10):
                started = time.perf_counter()
                result = finite_image_mesh(x, gamma, sigma, query, images, tau=tau, spacing=0.04, order=order)
                velocity = observed_error(result.velocity, u, conditioning[:, 0])
                gradient = observed_error(result.gradient, j, conditioning[:, 1])
                # A preregistered small-cloud screen, NOT runtime admission.
                velocity_ok = (velocity["relative_l2"] < 1e-4 if velocity["relative_l2"] is not None
                               else velocity["absolute_l2"] < 1e-5)
                gradient_ok = gradient["relative_l2"] < 1e-4
                record = {"case": name, "spacing": 0.04, "tau": tau, "tau_over_h": tau/0.04,
                          "order": order, "velocity": velocity, "gradient": gradient,
                          "passed_small_cloud_norm_screen": bool(velocity_ok and gradient_ok),
                          "runtime_admissible": False, "cpu_seconds": time.perf_counter()-started,
                          "diagnostics": result.diagnostics}
                records.append(record)
                print(json.dumps({key: record[key] for key in ("case", "tau", "order", "passed_small_cloud_norm_screen")} |
                                 {"u_error": velocity["relative_l2"], "u_absolute": velocity["absolute_l2"],
                                  "J_error": gradient["relative_l2"]}), flush=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise RuntimeError("qualification sources changed during execution")
    report = {"status": "complete", "runtime_admissible": False, "production_imports": False,
              "source_hashes": before, "records": records,
              "qualification_limits": ["No mesh, FFT, interpolation or roundoff error certificate.",
                                       "All local correction pairs evaluated; no scalable correction-neighbor implementation.",
                                       "Small-cloud norm screen is not pointwise noninferiority to existing production.",
                                       "CPU timing is not a production or GPU performance estimate."]}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
