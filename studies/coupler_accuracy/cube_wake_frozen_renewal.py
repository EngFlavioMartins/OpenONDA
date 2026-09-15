"""Hold the saved 3D FVM donor field fixed and repeat only particle renewal.

This is a component experiment, with no time advancement, boundary iteration,
particle advection, stretching, diffusion, or panel solve. The exact saved cell
velocities and boundary trace reconstruct the native Gauss donor gradient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cube_wake_drift_audit import frame, ordered_fields
from cube_wake_operator_audit import evaluate, field_metrics, reflection_asymmetry
from cube_wake_particle_probe import load_case, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


def read_archive(path, decode):
    with np.load(path) as data:
        return decode({key: data[key].copy() for key in data.files})


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.coupler.stable_renewal import (
        renew_stable_overlap,
        vortex_strength_from_velocity_trace,
    )
    from source.coupler.vorticity_transfer import VorticityTransfer, _smoothstep
    from source.solvers.fvm.fields.gradients import compute_gauss_gradient
    from source.solvers.fvm.fields.mixed_velocity_boundary import (
        update_normal_velocity_tangential_gradient_boundary,
    )
    from source.solvers.fvm.io.backup import decode_state
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

    mesh = load_native_mesh(args.solution / "mesh.npz")
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    n, nf, ni = (mesh[k] for k in ("n_cells", "n_faces", "n_interior_faces"))
    values = np.full((n + nf - ni, 3), np.nan)
    seen = np.zeros(n, dtype=bool)
    files = list((args.solution / "backups/fvm_000800").glob("rank*.npz"))
    for file in files:
        state = read_archive(file, decode_state)
        ids = state["global_cell_id"]
        velocity = state["velocity"][: len(ids)]
        if np.any(seen[ids]):
            np.testing.assert_allclose(values[ids][seen[ids]], velocity[seen[ids]], atol=1e-14)
        values[ids] = velocity
        seen[ids] = True
    assert np.all(seen)
    boundary_path = args.solution / "backups/vpm_boundary_condition_000800.npz"
    boundary = read_archive(boundary_path, decode_state)
    for patch in mesh["boundary"]:
        faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
        if patch["name"] == "cube":
            patch["velocity_type"] = "fixedValue"
            values[n + faces - ni] = 0
        elif patch["name"] == "numericalBoundary":
            patch["velocity_type"] = "normalValueTangentialGradient"
            patch["normal_velocity_field"] = boundary["normal_velocity"]
            patch["tangential_gradient_field"] = boundary["tangential_gradient"]
            assert len(faces) == len(boundary["velocity"])
            update_normal_velocity_tangential_gradient_boundary(values, patch, mesh, geometry)
        else:
            raise ValueError(f"Unrecognized physical boundary {patch['name']}")
    geometry["_operator_backend"] = "numba"
    gradient = compute_gauss_gradient(values, mesh, geometry)[:n]
    native = ordered_fields(frame(args.solution, "coupled_replacement_flow", 8), n)
    omega = np.column_stack(
        (
            gradient[:, 1, 2] - gradient[:, 2, 1],
            gradient[:, 2, 0] - gradient[:, 0, 2],
            gradient[:, 0, 1] - gradient[:, 1, 0],
        )
    )
    wall = next(p for p in mesh["boundary"] if p["name"] == "cube")
    faces = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
    normals = geometry["face_area_vector"][faces] / geometry["face_area"][faces, None]
    holder = SimpleNamespace(
        setup=case.COUPLER_SETUP,
        kinematic_viscosity=0.001,
        fvm_box=np.array([-1.5, 1.5] * 3),
        vpm_core_radius_ratio=1.1,
        vpm_particle_spacing=0.06,
        vpm_time_step_size=0.01,
        vpm_solver=SimpleNamespace(viscous_scheme="GBD"),
    )
    fvm = SimpleNamespace(
        setup=SimpleNamespace(boundaries=[SimpleNamespace(name="cube", mesh_type="wall")]),
        get_cell_centre_coordinates=lambda: geometry["cell_centre"],
        get_cell_volume=lambda: geometry["cell_volume"],
        get_boundary_face_centre_coordinates=lambda _: geometry["face_centre"][faces],
        get_boundary_face_normal=lambda _: normals,
    )
    transfer = VorticityTransfer(holder)
    transfer.setup(fvm)
    lattice = transfer._stable_renewal_lattice

    def velocity_at(points):
        return (
            transfer._velocity_trace.sample(points, values[:n], gradient)
            * _smoothstep(transfer._signed_solid_distance(points), 0.0, 0.06)[:, None]
        )

    target = vortex_strength_from_velocity_trace(lattice.positions, 0.06, velocity_at)
    with np.load(args.operator_audit / "fields_t8.npz") as data:
        points = data["points"].copy()
    masks = {
        "authority_ramp": points[:, 0] < 1.25,
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    report = {
        "description": __doc__,
        "source_tree": str(args.source_tree.resolve()),
        "velocity_reconstruction_vs_vtk_rms": rms(values[:n] - native["velocity"]),
        "curl_reconstruction_vs_vtk_rms": rms(omega - native["vorticity"]),
        "donor_and_boundary_sha256": {
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [*files, boundary_path]
        },
        "iterations": [],
    }
    for variant, prune in (("native", transfer.transfer_prune_threshold_abs), ("no_pruning", 0.0)):
        with np.load(args.operator_audit / "particles_accepted_t8.npz") as data:
            state = {k: data[k].copy() for k in data.files}
        for iteration in range(args.iterations + 1):
            if iteration in {0, 1, 2, 5, 10, args.iterations}:
                fields = evaluate(points, state)
                row = {
                    "variant": variant,
                    "iteration": iteration,
                    "particles": len(state["position"]),
                    "metrics": field_metrics(fields, masks),
                }
                report["iterations"].append(row)
                np.savez_compressed(
                    args.output / f"fields_{variant}_{iteration}.npz", points=points, **fields
                )
                print(json.dumps(row), flush=True)
                (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            if iteration == args.iterations:
                break
            result = renew_stable_overlap(
                state["position"],
                state["vortex_strength"],
                lattice,
                fvm_vortex_strength_at_node=lambda _: target,
                particle_fluid_weight=lambda p: _smoothstep(
                    transfer._signed_solid_distance(p), -0.06, 0.0
                ),
                particle_in_solid=lambda p: transfer._points_in_solid(p, include_boundary=False),
                prune_threshold=prune,
                core_radius_ratio=1.1,
                amplification_cap=transfer.transfer_amplification_cap,
                boundary_prune_multiplier=transfer.transfer_boundary_prune_multiplier,
                compute_diagnostics=False,
            )
            state = {
                "position": result.position,
                "vortex_strength": result.vortex_strength,
                "core_radius": result.core_radius,
            }
        np.savez_compressed(args.output / f"particles_{variant}_final.npz", **state)
    summarize_saved_fields(args.output)


def summarize_saved_fields(directory):
    """Reproduce the frozen-transfer reflection measurement from saved fields."""
    rows = []
    for variant in ("native", "no_pruning"):
        for iteration in (0, 1, 20):
            path = directory / f"fields_{variant}_{iteration}.npz"
            if not path.exists():
                continue
            with np.load(path) as data:
                points = data["points"]
                mask = (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62)
                asymmetry = reflection_asymmetry(points, data["velocity"])
                rows.append(
                    {
                        "variant": variant,
                        "iteration": iteration,
                        "seam_reflection_asymmetry_rms": rms(asymmetry[mask]),
                    }
                )
    (directory / "asymmetry.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
