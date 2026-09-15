"""Isolate lattice phase using frozen, fully 3D fine-reference velocity.

Only the lattice anchor changes. The production velocity-trace curl, Gaussian
representation correction, body mask, pruning and moment recovery are used.
No elapsed time or particle advection is claimed by this component experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from cube_wake_drift_audit import frame, ordered_fields
from cube_wake_particle_probe import direct_gaussian, rms
from numba import set_num_threads
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


def smooth(value, low, high):
    x = np.clip((value - low) / (high - low), 0, 1)
    return x * x * (3 - 2 * x)


def distance(points):
    q = np.abs(points) - 0.5
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)


def reference_gradient(velocity, mesh, geometry):
    from source.solvers.fvm.fields.gradients import compute_gauss_gradient

    n, ni, nf = (mesh[key] for key in ("n_cells", "n_interior_faces", "n_faces"))
    values = np.empty((n + nf - ni, 3))
    values[:n] = velocity
    for patch in mesh["boundary"]:
        faces = np.arange(patch["start_face"], patch["start_face"] + patch["n_faces"])
        owners = mesh["owners"][faces]
        name = patch["name"]
        boundary = velocity[owners].copy()
        if name == "cube":
            boundary[:] = 0
            patch["velocity_type"] = "fixedValue"
        elif name == "inlet":
            boundary[:] = [1, 0, 0]
            patch["velocity_type"] = "fixedValue"
        elif name == "outlet":
            patch["velocity_type"] = "zeroGradient"
        else:
            normal = geometry["face_area_vector"][faces] / geometry["face_area"][faces, None]
            boundary -= np.einsum("ij,ij->i", boundary, normal)[:, None] * normal
            patch["velocity_type"] = "slip"
        values[n + faces - ni] = boundary
    geometry["_operator_backend"] = "numba"
    return compute_gauss_gradient(values, mesh, geometry)[:n]


def production_geometry(mesh, geometry):
    """Fingerprint the production geometry setup using the native cube wall."""
    from source.coupler.config.types import CouplerSetup
    from source.coupler.vorticity_transfer import VorticityTransfer

    wall = next(p for p in mesh["boundary"] if p["name"] == "cube")
    faces = np.arange(wall["start_face"], wall["start_face"] + wall["n_faces"])
    normals = geometry["face_area_vector"][faces] / geometry["face_area"][faces, None]
    config = CouplerSetup(
        transfer_method="buffered_m4_renewal",
        transfer_region_bounds=(-1.25, 1.25) * 3,
        eta_blend_width=0.36,
    )
    coupler = SimpleNamespace(
        setup=config,
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
        get_boundary_face_centre_coordinates=lambda _name: geometry["face_centre"][faces],
        get_boundary_face_normal=lambda _name: normals,
    )
    transfer = VorticityTransfer(coupler)
    transfer.setup(fvm)
    lattice = transfer._stable_renewal_lattice
    return {
        "body_bounds": transfer._body_bounds.tolist(),
        "anchor": transfer._lattice_anchor.tolist(),
        "shape": list(lattice.shape),
        "buffer_length": lattice.buffer_length,
        "array_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(getattr(lattice, name)).tobytes()).hexdigest()
            for name in (
                "positions",
                "mesh_weight",
                "fluid_weight",
                "fvm_authority",
                "solid_interior",
            )
        },
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_tree.resolve()))
    from source.coupler.interpolation import FVMVelocityInterpolator
    from source.coupler.stable_renewal import (
        build_stable_renewal_lattice,
        renew_stable_overlap,
        vortex_strength_from_velocity_trace,
    )
    from source.solvers.fvm.io.mesh_storage import load_native_mesh
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry

    refdir = args.baseline / "solution/reference_fine"
    mesh = load_native_mesh(refdir / "mesh.npz")
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    resolved_geometry = production_geometry(mesh, geometry)
    (args.output / "production_geometry.json").write_text(
        json.dumps(resolved_geometry, indent=2) + "\n"
    )
    if args.geometry_only:
        print(json.dumps(resolved_geometry), flush=True)
        return
    centres = geometry["cell_centre"]
    trace = FVMVelocityInterpolator(centres, cKDTree(centres), neighbour_count=4)
    h, box = 0.06, np.array([-1.25, 1.25] * 3)

    def fluid_weight(p):
        return smooth(distance(p), -h, 0)

    def solid(p):
        return np.all(np.abs(p) < 0.5, axis=1)

    def mesh_weight(p):
        return np.all(np.abs(p) <= 1.5, axis=1).astype(float)

    base = np.stack(
        np.meshgrid(
            np.linspace(-1.3, 1.5, 15), [0.0, 0.3, 0.7, 1.1], [0.0, 0.3, 0.7], indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    base = base[~np.all(np.abs(base) < 0.55, axis=1)]
    reflected = base * [1, -1, -1]
    points = np.vstack((base, reflected))
    anchors = {
        "minimum_corner": [-0.53, -0.53, -0.53],
        "mirrored_transverse": [-0.53, 0.53, 0.53],
        "centred_transverse": [-0.53, -0.03, -0.03],
        "centred_all": [-0.03, -0.03, -0.03],
    }
    if args.body_mask_ablation:
        # Diagnostic only: retain every control cell containing fluid, even
        # when its centre lies inside the cube. These are not admissible
        # advecting particles; the released phase rule avoids that placement.
        anchors["corner_keep_cut_cells"] = anchors["minimum_corner"]

    def fully_solid_cell(p):
        return np.all(np.abs(p) + h / 2 <= 0.5, axis=1)

    def contains_fluid(p):
        return (~fully_solid_cell(p)).astype(float)

    report = {"description": __doc__, "spacing": h, "anchors": anchors, "times": []}
    for time in args.times:
        fields = ordered_fields(frame(refdir, "fine", time), len(centres))
        velocity = fields["velocity"]
        gradient = reference_gradient(velocity, mesh, geometry)
        curl = np.column_stack(
            (
                gradient[:, 1, 2] - gradient[:, 2, 1],
                gradient[:, 2, 0] - gradient[:, 0, 2],
                gradient[:, 0, 1] - gradient[:, 1, 0],
            )
        )

        def velocity_at(p, velocity=velocity, gradient=gradient):
            return trace.sample(p, velocity, gradient) * smooth(distance(p), 0, h)[:, None]

        donor = trace.sample(points, velocity, gradient)
        row = {
            "time": time,
            "native_vorticity_reconstruction_rms": rms(curl - fields["vorticity"]),
            "donor_rotation_defect_rms": rms(donor[: len(base)] - donor[len(base) :] * [1, -1, -1]),
            "phases": {},
        }
        for name, anchor in anchors.items():
            cell_ablation = name == "corner_keep_cut_cells"
            fluid_at = contains_fluid if cell_ablation else fluid_weight
            solid_at = fully_solid_cell if cell_ablation else solid
            lattice = build_stable_renewal_lattice(
                box,
                h,
                buffer_length=0.135,
                authority_ramp_width=0.36,
                lattice_anchor=np.array(anchor),
                mesh_weight_at_node=mesh_weight,
                fluid_weight_at_node=fluid_at,
                interior_at_node=solid_at,
            )
            target = vortex_strength_from_velocity_trace(lattice.positions, h, velocity_at)
            cell_lower = np.maximum(lattice.positions - 0.5 * h, -0.5)
            cell_upper = np.minimum(lattice.positions + 0.5 * h, 0.5)
            fluid_volume = h**3 - np.prod(np.maximum(cell_upper - cell_lower, 0), axis=1)
            affected = lattice.solid_interior & (fluid_volume > 1e-12 * h**3)
            geometry_check = {
                "solid_centre_cells_containing_fluid": int(affected.sum()),
                "fluid_volume_in_solid_centre_cells": float(fluid_volume[affected].sum()),
                "raw_target_strength_l1_in_solid_centre_cells": float(
                    np.linalg.norm(target[affected], axis=1).sum()
                ),
                "raw_target_strength_net_in_solid_centre_cells": target[affected]
                .sum(axis=0)
                .tolist(),
            }
            position, strength = np.empty((0, 3)), np.empty((0, 3))
            records = []
            for iteration in range(1, args.renewals + 1):
                result = renew_stable_overlap(
                    position,
                    strength,
                    lattice,
                    fvm_vortex_strength_at_node=lambda _p, target=target: target,
                    particle_fluid_weight=fluid_at,
                    particle_in_solid=solid_at,
                    prune_threshold=0.05 * h**3,
                    core_radius_ratio=1.1,
                    amplification_cap=1.8,
                    boundary_prune_multiplier=10,
                    freestream_speed=1,
                    time_step_size=0.01,
                    compute_diagnostics=True,
                )
                position, strength = result.position, result.vortex_strength
                if iteration in {1, args.renewals}:
                    induced, _ = direct_gaussian(points, position, strength, result.core_radius)
                    a, b = induced[: len(base)], induced[len(base) :] * [1, -1, -1]
                    record = {
                        "renewal": iteration,
                        "geometry": geometry_check,
                        "particles": len(position),
                        "rotation_defect_rms": rms(a - b),
                        "rotation_defect_max": float(np.linalg.norm(a - b, axis=1).max()),
                        "strength_net": strength.sum(axis=0).tolist(),
                        "strength_l1": float(np.linalg.norm(strength, axis=1).sum()),
                        "representation_residual_after_prune": result.representation_residual_after_prune,
                    }
                    records.append(record)
                    np.savez_compressed(
                        args.output / f"{name}_t{time:g}_r{iteration}.npz",
                        points=points,
                        velocity=induced,
                        position=position,
                        vortex_strength=strength,
                        core_radius=result.core_radius,
                    )
                    print(json.dumps({"time": time, "phase": name, **record}), flush=True)
            row["phases"][name] = records
        report["times"].append(row)
        (args.output / "phase.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[1, 6])
    parser.add_argument("--renewals", type=int, default=10)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--body-mask-ablation", action="store_true")
    args = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
