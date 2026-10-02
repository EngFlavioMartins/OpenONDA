"""Preregistered auxiliary refinement/phase screen; no production imports."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tests.vpm._finite_image_mesh_reference import direct_finite_images, observed_error
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh, slab_world_images


def cases():
    x = np.array([[-.082, .012, .023], [-.042, .012, .023], [-.002, .012, .023],
                  [.038, .012, .023], [.078, .012, .023]])
    q = np.array([[.011, .019, .018], [-.073, .058, .061], [.112, -.036, .04]])
    images = [(0, True), *[(k, odd) for k in (-2, -1, 1, 2) for odd in (False, True)]]
    for name, weights in (("nonneutral", np.ones(5)), ("cancelled", np.array([1, -4, 6, -4, 1]))):
        yield name, x, weights[:, None]*np.array([[.3, -.2, .7]]), np.full(5, .04), q, images, 0., .96, False
    for upper in (False, True):
        zmin, zmax = .081, .274
        points = np.array([[.013, -.017, zmax if upper else zmin], [.036, .027, .152]])
        yield f"{'upper' if upper else 'lower'}_coincidence", points, np.array([[.2, .7, -.4], [0., 0., 0.]]), np.full(2, .04), points[:1], [(int(upper), True)], zmin, zmax, False
    x = np.array([[-.047, .018, .073], [.056, -.036, .041], [.011, .029, .096]])
    gamma = np.array([[.3, -.2, .5], [-.7, .4, .2], [.4, -.2, .3]])
    query = np.array([[.012, -.013, .068], [-.021, .027, .059]])
    yield "mixed_core_derivative", x, gamma, np.array([.04, .07, .09]), query, [(0, True), (1, False)], 0., .96, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = [Path(__file__).with_name(name) for name in (
        "_profile_finite_slab_field_mesh.py", "_finite_slab_field_mesh_reference.py",
        "_finite_image_field_mesh_reference.py", "_finite_image_mesh_reference.py",
        "_gaussian_broadening_reference.py", "_slip_periodic_gaussian_oracle.py")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    records = []
    for h in (.04, .035, .03):
        for phase in ((0., 0.), (.17, .29), (.43, .61), (.79, .11)):
            for name, x, gamma, sigma, query, images, zmin, zmax, derivative_check in cases():
                translation = h*np.array([*phase, 0.])
                xx, qq = x+translation, query+translation
                world, _ = slab_world_images(images, zmin, zmax, 1)
                u, j, conditioning = direct_finite_images(xx, gamma, sigma, qq, world)
                options = {"zmin": zmin, "zmax": zmax, "tau": .12, "spacing": h, "order": 10}
                started = time.perf_counter()
                result = finite_slab_field_mesh(xx, gamma, sigma, qq, images, **options)
                velocity = observed_error(result.velocity, u, conditioning[:, 0])
                gradient = observed_error(result.gradient, j, conditioning[:, 1])
                passed = (velocity["relative_l2"] < 1e-4 if velocity["relative_l2"] is not None
                          else velocity["absolute_l2"] < 1e-5) and gradient["relative_l2"] < 1e-4
                derivative = None
                if derivative_check:
                    fd = np.empty_like(j)
                    epsilon = 1e-6
                    for axis in range(3):
                        step = np.zeros(3)
                        step[axis] = epsilon
                        plus = finite_slab_field_mesh(xx, gamma, sigma, qq+step, images, **options)
                        minus = finite_slab_field_mesh(xx, gamma, sigma, qq-step, images, **options)
                        for key in ("grid_origin_lattice", "grid_shape", "fft_shape"):
                            if plus.diagnostics[key] != minus.diagnostics[key] or plus.diagnostics[key] != result.diagnostics[key]:
                                raise RuntimeError("finite-difference test changed auxiliary grid")
                        fd[:, :, axis] = (plus.velocity-minus.velocity)/(2*epsilon)
                    derivative = {"relative_to_true_J": float(np.linalg.norm(fd-j)/np.linalg.norm(j)),
                                  "relative_discrepancy_to_gathered_J": float(np.linalg.norm(fd-result.gradient)/np.linalg.norm(j)),
                                  "epsilon": epsilon}
                    passed = passed and all(derivative[key] < 1e-4 for key in ("relative_to_true_J", "relative_discrepancy_to_gathered_J"))
                record = {"case": name, "requested_spacing": h, "xy_phase_cells": phase,
                          "velocity": velocity, "gradient": gradient, "derivative": derivative,
                          "passed_small_cloud_screen": bool(passed), "runtime_admissible": False,
                          "cpu_seconds": time.perf_counter()-started, "diagnostics": result.diagnostics}
                records.append(record)
                print(json.dumps({"case": name, "h": h, "phase": phase, "passed": bool(passed),
                                  "u": velocity["relative_l2"], "u_abs": velocity["absolute_l2"],
                                  "J": gradient["relative_l2"], "derivative": derivative}), flush=True)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise RuntimeError("qualification source changed during run")
    report = {"status": "complete", "runtime_admissible": False, "source_hashes": before,
              "records": records, "qualification_limits": [
                  "No mesh/FFT/roundoff or pointwise old-operator noninferiority certificate.",
                  "Preregistered1e-4 u/J/derivative guards and1e-5 absolute zero-u cap unchanged.",
                  "Physical particle mesh and positions unchanged; only auxiliary FFT grid refined.",
                  "All local correction pairs evaluated; no production/GPU cost claim."]}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
