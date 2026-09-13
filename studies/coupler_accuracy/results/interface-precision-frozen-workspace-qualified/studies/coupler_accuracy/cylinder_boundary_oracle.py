#!/usr/bin/env python3
"""Compare a small cylinder FVM domain with a full FVM supplying its boundary.

Both domains inherit identical native cells from one midspan layer of a saved
reference mesh. Both start from the same checkpoint cell fields and rebuild
their flux/history through the FVM initial-state API. This is a new short
initial-value experiment, not a restart or reproduction of the archived run.
There is no VPM, transfer, or macro-step interpolation. Cut-face normal velocity
comes from the full solve's native conservative flux; tangential data comes
from its reconstructed velocity and gradient.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from source.coupler.boundary import apply_fvm_boundary, tangential_normal_velocity_gradient
from source.coupler.config.types import CouplerSetup
from source.coupler.interpolation import FVMVelocityInterpolator
from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    RunSchedule,
    TimeConfig,
    TransportConfig,
    create_fvm_solver,
)
from source.solvers.fvm.fields.gradients import compute_lsq_gradient
from source.solvers.fvm.io.backup import decode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import extract_cell_subset_mesh
from source.solvers.fvm.sampling.forces import ForceSampler

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow/reference_flow"


def restrict_mesh(mesh, selected_ids):
    """Preserve physical patches and track original face IDs/orientations."""
    selected_ids = np.unique(selected_ids)
    selected = np.zeros(mesh["n_cells"], dtype=bool)
    selected[selected_ids] = True
    owners, neighbours = mesh["owners"], mesh["neighbours"]
    n_internal = mesh["n_interior_faces"]
    left, right = selected[owners[:n_internal]], selected[neighbours]
    retained = np.flatnonzero(left & right)
    exposed = np.flatnonzero(left ^ right)
    boundary = n_internal + np.flatnonzero(selected[owners[n_internal:]])
    original_faces = np.concatenate((retained, exposed, boundary))
    orientation = np.ones(len(original_faces))
    orientation[len(retained) : len(retained) + len(exposed)] = np.where(left[exposed], 1, -1)
    subset = extract_cell_subset_mesh(mesh, selected_ids)

    names = np.full(mesh["n_faces"], "numericalBoundary", dtype=object)
    for patch in mesh["boundary"]:
        start = patch["start_face"]
        names[start : start + patch["n_faces"]] = patch["name"]
    # A newly exposed span plane remains a slip span plane. Lateral cuts
    # become the artificial coupling boundary.
    points = mesh["vertex_position"]
    for face in exposed:
        polygon = points[mesh["faces"][face]]
        if np.ptp(polygon[:, 2]) < 1e-12:
            area = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
            sign = 1 if left[face] else -1
            names[face] = "zmax" if area[2] * sign > 0 else "zmin"
    boundary_names = names[original_faces[len(retained) :]]
    order = list(range(len(retained)))
    patches = []
    for name in sorted(set(boundary_names)):
        rows = len(retained) + np.flatnonzero(boundary_names == name)
        patches.append(
            {
                "name": name,
                "type": "wall" if name == "cylinder" else "patch",
                "start_face": len(order),
                "n_faces": len(rows),
            }
        )
        order.extend(rows)
    subset["faces"] = [subset["faces"][row] for row in order]
    subset["owners"] = np.asarray(subset["owners"])[order]
    subset["boundary"] = patches
    return subset, original_faces[order], orientation[order]


def checkpoint_fields(solution, output):
    manifest_path = solution / "backup/manifest.json"
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    count = manifest["n_global_cells"]
    velocity, pressure = np.empty((count, 3)), np.empty(count)
    seen = np.zeros(count, dtype=bool)
    source = []
    checkpoint_time = None
    for name in manifest["files"]:
        if Path(name).name != name:
            raise ValueError("Invalid checkpoint member path")
        path = manifest_path.parent / name
        with np.load(path, allow_pickle=False) as archive:
            state = decode_state({key: archive[key] for key in archive.files})
        ids = state["global_cell_id"]
        local_u = state["velocity"][: len(ids)]
        local_p = state["kinematic_pressure"][: len(ids)]
        if np.any(seen[ids]):
            repeated = seen[ids]
            np.testing.assert_allclose(
                velocity[ids[repeated]], local_u[repeated], atol=1e-13, rtol=0
            )
            np.testing.assert_allclose(
                pressure[ids[repeated]], local_p[repeated], atol=1e-13, rtol=0
            )
        velocity[ids], pressure[ids], seen[ids] = local_u, local_p, True
        if checkpoint_time is not None and checkpoint_time != float(state["time"]):
            raise ValueError("Checkpoint ranks have different times")
        checkpoint_time = float(state["time"])
        source.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not seen.all():
        raise ValueError("Checkpoint cells are incomplete")
    (output / "source-manifest.json").write_text(manifest_text)
    return velocity, pressure, checkpoint_time, source


def setup_for(mesh, name, dt, steps):
    boundaries = []
    for patch in mesh["boundary"]:
        patch_name = patch["name"]
        if patch_name == "cylinder":
            boundaries.append(BoundaryConfig.wall(patch_name))
        elif patch_name in {"zmin", "zmax", "ymin", "ymax"}:
            boundaries.append(BoundaryConfig.slip(patch_name))
        elif patch_name == "inlet":
            boundaries.append(BoundaryConfig.inlet(patch_name, [1, 0, 0]))
        elif patch_name == "outlet":
            boundaries.append(BoundaryConfig.outlet(patch_name, kinematic_pressure=0))
        elif patch_name == "numericalBoundary":
            boundaries.append(
                BoundaryConfig(
                    name=patch_name,
                    velocity_type="fixedValue",
                    velocity_value=[1, 0, 0],
                    pressure_type="fixedFluxPressure",
                )
            )
        else:
            raise ValueError(f"Unexpected boundary {patch_name}")
    return FVMSetup(
        case_name=name,
        time=TimeConfig(
            time_step_size=dt,
            end_time=dt * steps,
            output_schedule=RunSchedule(every_n_steps=100000),
        ),
        schemes=DiscretizationConfig(
            convection_scheme="limitedLinear", gradient_scheme="lsq", time_scheme="euler_implicit"
        ),
        linear=LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-12,
            momentum_tolerance=1e-12,
            pressure_relative_tolerance=0,
            momentum_relative_tolerance=0,
        ),
        pimple=PimpleControl(
            n_outer_correctors=3, n_correctors=2, velocity_relaxation=1, pressure_relaxation=1
        ),
        transport=TransportConfig(kinematic_viscosity=1 / 150),
        boundaries=boundaries,
        initial_velocity=[1, 0, 0],
    )


def run(reference, output, dt, steps):
    output.mkdir(parents=True, exist_ok=True)
    solution = CASE / "solution" / reference
    original = load_native_mesh(solution / "mesh.npz")
    geometry = compute_mesh_geometry(original, gradient_scheme="lsq", compute_lsq=False)
    centre = geometry["cell_centre"][: original["n_cells"]]
    z_layer = centre[np.argmin(np.abs(centre[:, 2])), 2]
    slab_ids = np.flatnonzero(np.isclose(centre[:, 2], z_layer, rtol=0, atol=1e-10))
    full_mesh, _, _ = restrict_mesh(original, slab_ids)
    full_centre = centre[slab_ids]
    bounds = np.array([-1.5, 2.0, -1.5, 1.5])
    small_ids = np.flatnonzero(
        np.all((full_centre[:, :2] >= bounds[::2]) & (full_centre[:, :2] <= bounds[1::2]), axis=1)
    )
    small_mesh, face_map, face_sign = restrict_mesh(full_mesh, small_ids)
    velocity, pressure, checkpoint_time, source = checkpoint_fields(solution, output)
    source.append(
        {
            "path": str((solution / "mesh.npz").relative_to(ROOT)),
            "sha256": hashlib.sha256((solution / "mesh.npz").read_bytes()).hexdigest(),
        }
    )
    velocity, pressure = velocity[slab_ids], pressure[slab_ids]
    span = float(np.ptp(full_mesh["vertex_position"][:, 2]))
    force = ForceSampler(patch_names=["cylinder"], reference_area=span)
    records = []
    with ExitStack() as stack:
        full = stack.enter_context(
            create_fvm_solver(
                setup_for(full_mesh, "full", dt, steps), case_dir=output / "full", mesh=full_mesh
            )
        )
        full.set_initial_state(velocity, pressure)
        np.testing.assert_allclose(
            full.get_cell_centre_coordinates(), full_centre, atol=1e-13, rtol=0
        )
        trace = FVMVelocityInterpolator(full_centre, cKDTree(full_centre))
        solvers = {}
        for mode in (
            "dirichlet",
            "vorticity_mixed",
            "pressure_gradient",
            "mixed_discrete_trace",
            "mixed_pressure_gradient",
        ):
            small = stack.enter_context(
                create_fvm_solver(
                    setup_for(small_mesh, mode, dt, steps),
                    case_dir=output / mode,
                    mesh=copy.deepcopy(small_mesh),
                )
            )
            for previous in solvers.values():
                if {id(patch) for patch in small.boundaries} & {
                    id(patch) for patch in previous.boundaries
                }:
                    raise RuntimeError("Oracle solvers must not share mutable boundary state")
            solvers[mode] = small
            np.testing.assert_allclose(
                small.get_cell_centre_coordinates(), full_centre[small_ids], atol=1e-13, rtol=0
            )
            np.testing.assert_allclose(
                small.get_cell_volume(), full.get_cell_volume()[small_ids], atol=1e-14, rtol=1e-13
            )
        patch = next(p for p in small_mesh["boundary"] if p["name"] == "numericalBoundary")
        rows = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
        face = small.get_boundary_face_centre_coordinates("numericalBoundary")
        normal = small.get_boundary_face_normal("numericalBoundary")
        area = small.geo_data["face_area"][rows]
        owner = small_ids[small_mesh["owners"][rows]]
        owner_to_face = face - full_centre[owner]
        normal_distance = np.einsum("ij,ij->i", owner_to_face, normal)
        skew = owner_to_face - normal_distance[:, None] * normal
        np.testing.assert_allclose(
            small.geo_data["face_area_vector"][rows],
            full.geo_data["face_area_vector"][face_map[rows]] * face_sign[rows, None],
            atol=1e-13,
            rtol=0,
        )

        def boundary_data():
            full_gradient = full.get_velocity_gradient_field()
            vector = trace.sample(face, full.get_velocity_field(), full_gradient)
            # FVM gradients are [derivative, velocity], opposite the VPM Jacobian.
            gradient = np.stack(
                [trace.sample_cell_field(face, full_gradient[:, i, :]) for i in range(3)], axis=1
            ).transpose(0, 2, 1)
            normal_velocity = full.volumetric_face_flux[face_map[rows]] * face_sign[rows] / area
            vector += (normal_velocity - np.einsum("ij,ij->i", vector, normal))[:, None] * normal
            tangent = tangential_normal_velocity_gradient(gradient, normal)
            pressure_gradient = compute_lsq_gradient(
                full.kinematic_pressure, full.mesh_data, full.geo_data
            )[: len(full_centre), :, 0]
            return (
                vector,
                normal_velocity,
                tangent,
                trace.sample_cell_field(face, pressure_gradient),
            )

        def discrete_tangent(vector):
            # Diagnostic oracle: use a compatible discrete owner-to-face
            # difference, not a pointwise physical normal derivative.
            gradient = (vector - full.get_velocity_field()[owner]) / normal_distance[:, None]
            return gradient - np.einsum("ij,ij->i", gradient, normal)[:, None] * normal

        vector, normal_velocity, tangent, pressure_gradient = boundary_data()
        for mode, small in solvers.items():
            if mode not in {"vorticity_mixed", "mixed_discrete_trace", "mixed_pressure_gradient"}:
                small.set_dirichlet_velocity_boundary_condition_vec(vector, "numericalBoundary")
            else:
                small.set_normal_velocity_tangential_gradient_boundary_condition(
                    normal_velocity,
                    discrete_tangent(vector) if mode == "mixed_discrete_trace" else tangent,
                    "numericalBoundary",
                )
            if mode in {"pressure_gradient", "mixed_pressure_gradient"}:
                small.set_neumann_pressure_boundary_condition(
                    pressure_gradient, "numericalBoundary"
                )
            small.set_initial_state(velocity[small_ids], pressure[small_ids])

        for step in range(steps + 1):
            if step:
                full.advance()
                vector, normal_velocity, tangent, pressure_gradient = boundary_data()
                for mode, small in solvers.items():
                    boundary_mode = {
                        "mixed_discrete_trace": "vorticity_mixed",
                        "mixed_pressure_gradient": "vorticity_mixed_pressure_gradient",
                    }.get(mode, mode)
                    apply_fvm_boundary(
                        SimpleNamespace(
                            fvm_solver=small,
                            setup=CouplerSetup(boundary_condition_mode=boundary_mode),
                        ),
                        "numericalBoundary",
                        vector,
                        normal_velocity=normal_velocity,
                        tangential_gradient=discrete_tangent(vector)
                        if mode == "mixed_discrete_trace"
                        else tangent,
                        pressure_gradient=pressure_gradient,
                    )
            expected = full.get_velocity_field()[small_ids]
            full_coefficients = force.sample(full)["cylinder"]["coeffs"]
            for mode, small in solvers.items():
                error = small.get_velocity_field() - expected
                volumes = small.get_cell_volume()
                coefficients = force.sample(small)["cylinder"]["coeffs"]
                near_body = np.linalg.norm(full_centre[small_ids, :2], axis=1) < 1.0
                record = {
                    "mode": mode,
                    "elapsed_flow_time": full.time,
                    "step": step,
                    "velocity_rms_difference_over_Uinf": float(
                        np.sqrt(np.average(np.sum(error**2, axis=1), weights=volumes))
                    ),
                    "velocity_max_difference_over_Uinf": float(np.linalg.norm(error, axis=1).max()),
                    "near_body_velocity_rms_difference_over_Uinf": float(
                        np.sqrt(
                            np.average(
                                np.sum(error[near_body] ** 2, axis=1), weights=volumes[near_body]
                            )
                        )
                    ),
                    "full_drag_coefficient": float(full_coefficients["drag_coefficient"]),
                    "small_drag_coefficient": float(coefficients["drag_coefficient"]),
                    "full_lift_coefficient": float(full_coefficients["lift_coefficient"]),
                    "small_lift_coefficient": float(coefficients["lift_coefficient"]),
                    "drag_coefficient_difference": float(
                        coefficients["drag_coefficient"] - full_coefficients["drag_coefficient"]
                    ),
                    "lift_coefficient_difference": float(
                        coefficients["lift_coefficient"] - full_coefficients["lift_coefficient"]
                    ),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    report = {
        "schema": "openonda-cylinder-boundary-oracle/1",
        "reference": reference,
        "source_checkpoint_time": checkpoint_time,
        "source_global_cells": original["n_cells"],
        "full_planar_cells": full_mesh["n_cells"],
        "small_planar_cells": small_mesh["n_cells"],
        "boundary_max_skew_distance_over_normal_distance": float(
            (np.linalg.norm(skew, axis=1) / normal_distance).max()
        ),
        "span": span,
        "small_requested_xy_bounds": bounds.tolist(),
        "small_actual_bounds": np.array(
            [small_mesh["vertex_position"].min(axis=0), small_mesh["vertex_position"].max(axis=0)]
        )
        .T.reshape(-1)
        .tolist(),
        "dt": dt,
        "steps": steps,
        "numerics": {
            "time_scheme": "euler_implicit",
            "convection": "limitedLinear",
            "gradient": "lsq",
            "linear_solver": "spsolve",
            "outer_correctors": 3,
            "correctors": 2,
            "relaxation": 1.0,
        },
        "results": records,
        "sources": source,
        "limitations": [
            "A new initial-value experiment from checkpoint cell fields; flux histories rebuilt on both meshes.",
            "One inherited span layer with slip ends; no exact replay of the archived multilayer parallel run.",
            "Normal data uses the full FVM native face flux; tangential data uses local Taylor/interpolated gradients.",
            "mixed_discrete_trace uses a reference owner-to-face difference, not a pointwise physical normal derivative.",
            "mixed_pressure_gradient uses the opt-in vorticity_mixed_pressure_gradient coupler mode with reference pressure data.",
            "No VPM, transfer, macro-step interpolation, or long-time shedding validation.",
        ],
    }
    (output / "cylinder-boundary-oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, output)
    return report


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "dirichlet": "Dirichlet velocity + flux pressure",
        "vorticity_mixed": "Mixed velocity + flux pressure",
        "pressure_gradient": "Dirichlet velocity + reference pressure gradient",
        "mixed_discrete_trace": "Discrete mixed trace + flux pressure",
        "mixed_pressure_gradient": "Mixed velocity + reference pressure gradient",
    }
    metrics = (
        ("velocity_rms_difference_over_Uinf", "Whole small domain: velocity RMS / U∞"),
        ("near_body_velocity_rms_difference_over_Uinf", "r/D < 1: velocity RMS / U∞"),
        ("drag_coefficient_difference", "Small − full: drag coefficient"),
        ("lift_coefficient_difference", "Small − full: lift coefficient"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(bottom=0.22, top=0.87, hspace=0.4, wspace=0.25)
    for mode, label in labels.items():
        records = [row for row in report["results"] if row["mode"] == mode and row["step"] > 0]
        if not records:
            continue
        time = [row["elapsed_flow_time"] for row in records]
        for i, (axis, (metric, title)) in enumerate(zip(axes.flat, metrics, strict=True)):
            axis.plot(time, [row[metric] for row in records], label=label, linewidth=1.4)
            axis.set(title=title, xlabel="Elapsed flow time U∞/D")
            if i < 2:
                axis.set_yscale("log")
            axis.grid(alpha=0.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        f"Cylinder boundary oracle: {report['small_planar_cells']:,} inherited cells, dt = {report['dt']:g}\n"
        "Boundary data from full FVM; no VPM or particle transfer"
    )
    fig.savefig(output / "cylinder-boundary-oracle.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="medium")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results/cylinder-boundary-oracle",
    )
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    run(args.reference, args.output, args.dt, args.steps)
