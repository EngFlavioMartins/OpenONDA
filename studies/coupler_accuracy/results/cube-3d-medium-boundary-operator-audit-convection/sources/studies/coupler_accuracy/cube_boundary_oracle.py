#!/usr/bin/env python3
"""Fully 3D cube boundary experiment on identical inherited native cells.

The full mesh uses the cube reference tutorial's geometry and refinement
requests. Every cell in the small (-1.5, 1.5)^3 domain is retained verbatim
from that mesh. All six artificial faces exchange data with the full solve;
there are no empty directions, extruded slices, or spanwise approximations.

This isolates boundary closure. It is not a VPM run or a recovered reference.
Both solves start a new initial-value experiment from the same cell fields,
with freshly initialized face fluxes and time histories.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

import openonda.fvm as fvm
import openonda.fvm.mesher as msh
from source.coupler.boundary import apply_fvm_boundary, tangential_normal_velocity_gradient
from source.coupler.config.types import CouplerSetup
from source.coupler.interpolation import FVMVelocityInterpolator
from source.solvers.fvm.fields.gradients import (
    _resolve_gradient_fn,
    compute_lsq_geometry,
    compute_lsq_gradient,
)
from source.solvers.fvm.io.mesh_storage import load_native_mesh, save_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import extract_cell_subset_mesh, validate_topology
from studies.coupler_accuracy.native_face_trace_3d import NativeFaceTrace

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "tutorials/coupled_fvm_vpm/02_cube_flow"
FULL_BOUNDS = (-5.0, 10.0, -5.0, 5.0, -5.0, 5.0)
SMALL_BOUNDS = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
MODES = ("dirichlet", "vorticity_mixed", "pressure_gradient", "vorticity_mixed_pressure_gradient")


def build_reference_mesh(dx):
    """Use the current reference tutorial's native 3D meshing requests."""
    return msh.CartesianMesher(
        domain=msh.BoxDomain(
            bounds=FULL_BOUNDS,
            patches=msh.BoxPatches(
                xmin="inlet", xmax="outlet", ymin="ymin", ymax="ymax", zmin="zmin", zmax="zmax"
            ),
        ),
        surfaces=(msh.STLSurface(CASE / "reference_flow/assets/cube.stl", patch="cube"),),
        max_cell_size=0.5,
        refinements=(
            msh.BoxRefinement("nearBody", (-1.5, 2.5, -1.5, 1.5, -1.5, 1.5), 3 * dx),
            msh.BoxRefinement("wake", (0.0, 8.0, -2.0, 2.0, -2.0, 2.0), 6 * dx),
        ),
        patch_refinements=(msh.PatchRefinement("cube", dx),),
    ).build()


def restrict_mesh_3d(mesh, selected_ids):
    """Retain all native cells/faces and name every newly exposed face alike."""
    selected_ids = np.unique(np.asarray(selected_ids, dtype=np.int64))
    subset = extract_cell_subset_mesh(mesh, selected_ids)
    selected = np.zeros(mesh["n_cells"], dtype=bool)
    selected[selected_ids] = True
    owners, neighbours = np.asarray(mesh["owners"]), np.asarray(mesh["neighbours"])
    n_internal = mesh["n_interior_faces"]
    left, right = selected[owners[:n_internal]], selected[neighbours]
    retained = np.flatnonzero(left & right)
    exposed = np.flatnonzero(left ^ right)
    boundary = n_internal + np.flatnonzero(selected[owners[n_internal:]])
    source_faces = np.concatenate((retained, exposed, boundary))
    signs = np.ones(len(source_faces))
    signs[len(retained) : len(retained) + len(exposed)] = np.where(left[exposed], 1, -1)
    names = np.full(mesh["n_faces"], "numericalBoundary", dtype=object)
    types = {"numericalBoundary": "patch"}
    for patch in mesh["boundary"]:
        first = patch["start_face"]
        names[first : first + patch["n_faces"]] = patch["name"]
        types[patch["name"]] = patch.get("type", "patch")
    boundary_names = names[source_faces[len(retained) :]]
    order = list(range(len(retained)))
    patches = []
    for name in sorted(set(boundary_names)):
        rows = len(retained) + np.flatnonzero(boundary_names == name)
        patches.append(
            {"name": name, "type": types[name], "start_face": len(order), "n_faces": len(rows)}
        )
        order.extend(rows)
    subset["faces"] = [subset["faces"][row] for row in order]
    subset["owners"] = np.asarray(subset["owners"])[order]
    subset["boundary"] = patches
    validate_topology(subset)
    return subset, source_faces[order], signs[order]


def setup_for(mesh, name, dt, steps, *, turbulence=True, outer_correctors=3):
    boundaries = []
    for patch in mesh["boundary"]:
        name_ = patch["name"]
        if name_ == "cube":
            boundaries.append(fvm.BoundaryConfig.wall(name_))
        elif name_ == "inlet":
            boundaries.append(fvm.BoundaryConfig.inlet(name_, [1, 0, 0]))
        elif name_ == "outlet":
            boundaries.append(fvm.BoundaryConfig.outlet(name_, kinematic_pressure=0))
        elif name_ in {"ymin", "ymax", "zmin", "zmax"}:
            boundaries.append(fvm.BoundaryConfig.slip(name_))
        elif name_ == "numericalBoundary":
            boundaries.append(
                fvm.BoundaryConfig(
                    name=name_,
                    velocity_type="fixedValue",
                    velocity_value=[1, 0, 0],
                    pressure_type="fixedFluxPressure",
                )
            )
        else:
            raise ValueError(f"Unexpected boundary: {name_}")
    return fvm.FVMSetup(
        case_name=name,
        cores=1,
        time=fvm.TimeConfig(
            time_step_size=dt,
            end_time=dt * max(steps, 1),
            output_schedule=fvm.RunSchedule(every_n_steps=100000),
        ),
        execution=fvm.ComputeConfig(operator_backend="numba"),
        schemes=fvm.DiscretizationConfig(
            convection_scheme="linearUpwind", gradient_scheme="gauss", time_scheme="euler_implicit"
        ),
        linear=fvm.LinearSolverConfig(
            pressure_solver="amg",
            momentum_solver="bicgstab",
            pressure_tolerance=1e-11,
            momentum_tolerance=1e-11,
            pressure_relative_tolerance=0,
            momentum_relative_tolerance=0,
            pressure_max_iterations=2000,
            momentum_max_iterations=2000,
        ),
        pimple=fvm.PimpleControl(
            n_outer_correctors=outer_correctors,
            n_correctors=2,
            n_orthogonal_correctors=1,
            velocity_relaxation=1,
            pressure_relaxation=1,
        ),
        transport=fvm.TransportConfig(density=1, kinematic_viscosity=0.001),
        turbulence=(
            fvm.TurbulenceConfig.equilibrium_smagorinsky(
                subgrid_kinetic_energy_coefficient=0.094, subgrid_dissipation_coefficient=1.048
            )
            if turbulence
            else fvm.TurbulenceConfig()
        ),
        boundaries=boundaries,
        initial_velocity=[1, 0, 0],
    )


def field_rms(field, volumes):
    return float(np.sqrt(np.average(np.sum(np.asarray(field) ** 2, axis=1), weights=volumes)))


def hash_file(path):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source_records = [
        hash_file(path) for path in (
            Path(__file__),
            CASE / "reference_flow/setup.py",
            CASE / "reference_flow/assets/cube.stl",
            ROOT / "source/coupler/boundary.py",
            ROOT / "source/solvers/fvm/coupling/coupler_interface.py",
            ROOT / "source/solvers/fvm/core/solver.py",
            ROOT / "source/solvers/fvm/mesh/cartesian/mesher.py",
            ROOT / "source/solvers/fvm/assemble/convection.py",
            ROOT / "source/solvers/fvm/assemble/momentum.py",
            ROOT / "source/solvers/fvm/assemble/diffusion.py",
            ROOT / "source/solvers/fvm/solve/simple_solver.py",
            ROOT / "source/solvers/fvm/fields/gradients.py",
            ROOT / "source/solvers/fvm/fields/mixed_velocity_boundary.py",
            ROOT / "source/solvers/fvm/mesh/geometry.py",
            ROOT / "studies/coupler_accuracy/native_face_trace_3d.py",
        )
    ]
    archive = output / "sources"
    for record in source_records:
        source = ROOT / record["path"]
        destination = archive / record["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if hashlib.sha256(destination.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError("Study source changed while its provenance was being archived")
    mesh = load_native_mesh(args.mesh) if args.mesh else build_reference_mesh(args.dx)
    mesh_file = save_native_mesh(mesh, output / "full-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    centres = geometry["cell_centre"][: mesh["n_cells"]]
    bounds = np.asarray(SMALL_BOUNDS)
    ids = np.flatnonzero(np.all((centres > bounds[::2]) & (centres < bounds[1::2]), axis=1))
    small_mesh, source_faces, signs = restrict_mesh_3d(mesh, ids)
    small_file = save_native_mesh(small_mesh, output / "small-native-mesh.npz")
    actual_bounds = np.array(
        [small_mesh["vertex_position"].min(axis=0), small_mesh["vertex_position"].max(axis=0)]
    ).T.ravel()
    # Native surface/volume optimization slightly moves interior grid planes.
    # Keep those faces verbatim instead of snapping the cut and changing cells.
    # Bound the deviation by one wall spacing and report the actual envelope.
    if np.max(np.abs(actual_bounds - SMALL_BOUNDS)) > args.dx:
        raise ValueError("Inherited cut exceeds the nominal small box by more than one wall cell")
    if {p["name"] for p in small_mesh["boundary"]} != {"cube", "numericalBoundary"}:
        raise ValueError("Small 3D cube must have only the wall and six-sided exchange boundary")
    np.savez_compressed(
        output / "cell-and-face-map.npz", cell_ids=ids, face_ids=source_faces, signs=signs
    )
    counts = {"full_cells": mesh["n_cells"], "small_cells": len(ids)}
    print(
        json.dumps(
            {"event": "meshes_ready", **counts, "wall_time_seconds": time.perf_counter() - start}
        ),
        flush=True,
    )
    config = {"turbulence": not args.laminar, "outer_correctors": args.outer_correctors}
    if args.seed:
        with np.load(args.seed, allow_pickle=False) as seed:
            np.testing.assert_allclose(seed["centres"], centres, atol=1e-13, rtol=0)
            velocity, pressure = seed["velocity"].copy(), seed["pressure"].copy()
            seed_time = float(seed["physical_time"])
    else:
        with fvm.create_fvm_solver(
            setup_for(mesh, "warmup", args.dt, args.warmup_steps, **config),
            case_dir=output / "warmup",
            mesh=copy.deepcopy(mesh),
        ) as warm:
            for step in range(args.warmup_steps):
                warm.advance()
                if step % 10 == 0:
                    print(
                        json.dumps({"event": "warmup", "step": step + 1, "time": warm.time}),
                        flush=True,
                    )
            velocity = warm.get_velocity_field().copy()
            pressure = warm.kinematic_pressure[: len(centres)].copy()
            seed_time = float(warm.time)
    np.savez_compressed(
        output / "initial-cell-fields.npz",
        centres=centres,
        velocity=velocity,
        pressure=pressure,
        physical_time=seed_time,
    )
    force = fvm.ForceSampler(patch_names=["cube"], reference_area=1, reference_length=1)
    records, trace_records = [], []
    with ExitStack() as stack:
        full = stack.enter_context(
            fvm.create_fvm_solver(
                setup_for(mesh, "full", args.dt, args.steps, **config),
                case_dir=output / "full",
                mesh=copy.deepcopy(mesh),
            )
        )
        full.set_initial_state(velocity, pressure)
        full_volumes = full.get_cell_volume().copy()
        pressure_geometry = dict(full.geo_data)
        pressure_geometry.update(compute_lsq_geometry(full.mesh_data, full.geo_data))
        trace = FVMVelocityInterpolator(centres, cKDTree(centres))
        solvers = {}
        for mode in args.modes:
            small = stack.enter_context(
                fvm.create_fvm_solver(
                    setup_for(small_mesh, mode, args.dt, args.steps, **config),
                    case_dir=output / mode,
                    mesh=copy.deepcopy(small_mesh),
                )
            )
            for previous in solvers.values():
                assert not (
                    {id(p) for p in previous.boundaries} & {id(p) for p in small.boundaries}
                )
            solvers[mode] = small
            np.testing.assert_allclose(
                small.get_cell_centre_coordinates(), centres[ids], atol=1e-13, rtol=0
            )
            np.testing.assert_allclose(
                small.get_cell_volume(), full_volumes[ids], atol=1e-14, rtol=1e-13
            )
        patch = next(p for p in small_mesh["boundary"] if p["name"] == "numericalBoundary")
        rows = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
        face = small.get_boundary_face_centre_coordinates("numericalBoundary")
        normal = small.get_boundary_face_normal("numericalBoundary")
        area = small.get_boundary_face_area("numericalBoundary")
        np.testing.assert_allclose(
            small.geo_data["face_area_vector"][rows],
            full.geo_data["face_area_vector"][source_faces[rows]] * signs[rows, None],
            atol=1e-13,
            rtol=0,
        )
        normal_counts = [
            int(np.count_nonzero(normal[:, axis] * direction > 0.9))
            for axis in range(3)
            for direction in (-1, 1)
        ]
        assert all(normal_counts), "All six outer sides must exchange boundary data"
        native_trace = NativeFaceTrace(full.mesh_data, full.geo_data, source_faces[rows], signs[rows])
        np.testing.assert_allclose(native_trace.normal, normal, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(native_trace.owner, ids[small_mesh["owners"][rows]])
        for solver in (full, *solvers.values()):
            wall_rows = next(p for p in solver.mesh_data["boundary"] if p["name"] == "cube")
            first, number = wall_rows["start_face"], wall_rows["n_faces"]
            np.testing.assert_allclose(
                solver.geo_data["face_area"][first : first + number].sum(), 6, atol=1e-12, rtol=0
            )

        def boundary_data():
            full_gradient = full.get_velocity_gradient_field()
            vector = trace.sample(face, full.get_velocity_field(), full_gradient)
            jacobian = np.stack(
                [trace.sample_cell_field(face, full_gradient[:, i, :]) for i in range(3)], axis=1
            ).transpose(0, 2, 1)
            un = full.volumetric_face_flux[source_faces[rows]] * signs[rows] / area
            vector += (un - np.einsum("ij,ij->i", vector, normal))[:, None] * normal
            gradp = compute_lsq_gradient(full.kinematic_pressure, full.mesh_data, pressure_geometry)[
                : len(centres), :, 0
            ]
            tangent = tangential_normal_velocity_gradient(jacobian, normal)
            gradp = trace.sample_cell_field(face, gradp)
            solver_gradp = _resolve_gradient_fn(full.geo_data)(
                full.kinematic_pressure, full.mesh_data, full.geo_data
            )[: len(centres), :, 0]
            native = native_trace.evaluate(
                full.get_velocity_field(), full_gradient, full.kinematic_pressure, solver_gradp
            )
            native["interpolated_face_velocity_with_conservative_normal"] = vector.copy()
            native["interpolated_tangential_gradient"] = tangent.copy()
            native["lsq_interpolated_pressure_normal_gradient"] = np.sum(gradp * normal, axis=1)
            native["conservative_normal_velocity"] = un.copy()
            audit = {"physical_time": seed_time + float(full.time)}
            for first, second in (
                ("native_flux_tangential_gradient", "interpolated_tangential_gradient"),
                ("native_value_tangential_gradient", "native_flux_tangential_gradient"),
                ("native_flux_pressure_normal_gradient", "lsq_interpolated_pressure_normal_gradient"),
                ("native_value_pressure_normal_gradient", "native_flux_pressure_normal_gradient"),
            ):
                difference = native[first] - native[second]
                if difference.ndim == 1:
                    difference = difference[:, None]
                audit[first + "_minus_" + second + "_rms"] = field_rms(difference, area)
            audit["native_linear_face_normal_minus_conservative_normal_rms"] = field_rms(
                (np.sum(native["native_face_velocity"] * normal, axis=1) - un)[:, None], area
            )
            trace_records.append(audit)
            if not full.step or full.step == args.steps:
                name = "initial" if not full.step else "final"
                np.savez_compressed(output / f"{name}-cut-traces.npz", face=face, normal=normal,
                                    area=area, distance=native_trace.distance, **native)
            if args.velocity_trace != "interpolated":
                vector = native["native_face_velocity"].copy()
                vector += (un - np.sum(vector * normal, axis=1))[:, None] * normal
                tangent = native[args.velocity_trace + "_tangential_gradient"]
            if args.pressure_trace != "lsq":
                gradp = native[args.pressure_trace + "_pressure_normal_gradient"][:, None] * normal
            return vector, un, tangent, gradp

        vector, un, tangent, gradp = boundary_data()
        for mode, small in solvers.items():
            if mode in {"vorticity_mixed", "vorticity_mixed_pressure_gradient"}:
                small.set_normal_velocity_tangential_gradient_boundary_condition(
                    un, tangent, "numericalBoundary"
                )
            else:
                small.set_dirichlet_velocity_boundary_condition_vec(vector, "numericalBoundary")
            if mode in {"pressure_gradient", "vorticity_mixed_pressure_gradient"}:
                small.set_neumann_pressure_boundary_condition(gradp, "numericalBoundary")
            small.set_initial_state(velocity[ids], pressure[ids])
        near = np.max(np.abs(centres[ids]), axis=1) < 1
        for step in range(args.steps + 1):
            if step:
                full.advance()
                vector, un, tangent, gradp = boundary_data()
                for mode, small in solvers.items():
                    apply_fvm_boundary(
                        SimpleNamespace(
                            fvm_solver=small, setup=CouplerSetup(boundary_condition_mode=mode)
                        ),
                        "numericalBoundary",
                        vector,
                        normal_velocity=un,
                        tangential_gradient=tangent,
                        pressure_gradient=gradp,
                    )
            expected = full.get_velocity_field()[ids]
            expected_force = force.sample(full)["cube"]["coeffs"]
            for mode, small in solvers.items():
                error = small.get_velocity_field() - expected
                coefficients = force.sample(small)["cube"]["coeffs"]
                pressure_error = small.kinematic_pressure[: len(ids)] - full.kinematic_pressure[ids]
                pressure_error -= np.average(pressure_error, weights=full_volumes[ids])
                record = {
                    "mode": mode,
                    "step": step,
                    "elapsed_flow_time": float(full.time),
                    "physical_time": seed_time + float(full.time),
                    "velocity_rms_over_Uinf": field_rms(error, full_volumes[ids]),
                    "near_body_velocity_rms_over_Uinf": field_rms(
                        error[near], full_volumes[ids][near]
                    ),
                    "velocity_max_over_Uinf": float(np.linalg.norm(error, axis=1).max()),
                    "pressure_gauge_removed_rms_over_Uinf_squared": field_rms(
                        pressure_error[:, None], full_volumes[ids]
                    ),
                    "full_drag_coefficient": float(expected_force["drag_coefficient"]),
                    "small_drag_coefficient": float(coefficients["drag_coefficient"]),
                    "drag_coefficient_difference": float(
                        coefficients["drag_coefficient"] - expected_force["drag_coefficient"]
                    ),
                    "full_transverse_velocity_rms": field_rms(expected[:, 1:], full_volumes[ids]),
                    "boundary_net_flux": float(np.dot(un, area)),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
            (output / "history.json").write_text(json.dumps(records, indent=2) + "\n")
        np.savez_compressed(
            output / "final-full-cell-fields.npz",
            centres=centres,
            velocity=full.get_velocity_field(),
            pressure=full.kinematic_pressure[: len(centres)],
            vorticity=full.get_vorticity_field(),
            cell_volumes=full_volumes,
            physical_time=seed_time + float(full.time),
        )
        profiles = {"centres": centres[ids], "volumes": full_volumes[ids], "full": expected}
        for mode, small in solvers.items():
            profiles[mode] = small.get_velocity_field().copy()
        np.savez_compressed(output / "final-shared-cell-velocities.npz", **profiles)
    sources = source_records + [
        hash_file(path)
        for path in (
            mesh_file,
            small_file,
            output / "initial-cell-fields.npz",
        )
    ]
    report = {
        "schema": "openonda-cube-boundary-oracle/1",
        "valid_for_comparison": True,
        "spatial_dimensions": 3,
        "extruded_or_empty_directions": [],
        **counts,
        "small_bounds": actual_bounds.tolist(),
        "full_bounds": list(FULL_BOUNDS),
        "requested_wall_spacing": args.dx,
        "exchange_face_counts_xminus_xplus_yminus_yplus_zminus_zplus": normal_counts,
        "identical_native_shared_cells": True,
        "cube_wall_area": 6,
        "initial_physical_time": seed_time,
        "dt": args.dt,
        "steps": args.steps,
        "boundary_trace": {
            "velocity": args.velocity_trace,
            "pressure": args.pressure_trace,
            "normal_velocity": "native conservative face flux divided by area",
            "native_flux": "Unit-coefficient full interior operator; momentum and pressure use their own decompositions",
            "native_value": "Native linear face value minus full cropped-owner value, divided by normal distance; a discrete replay control, not a physical normal derivative on skew faces",
            "max_tangential_owner_to_face_displacement_over_normal_distance": float(np.max(
                np.linalg.norm(native_trace.tangential_displacement, axis=1) / native_trace.distance
            )),
        },
        "trace_audit": trace_records,
        "numerics": {
            "time_scheme": "euler_implicit",
            "convection": "linearUpwind",
            "gradient": "gauss",
            "linear_absolute_tolerances": 1e-11,
            "linear_relative_tolerances": 0,
            "outer_correctors": args.outer_correctors,
            "correctors": 2,
            "relaxation": 1,
            "kinematic_viscosity": 0.001,
            "sgs": "equilibrium_smagorinsky" if not args.laminar else "none",
        },
        "results": records,
        "sources": sources,
        "wall_time_seconds": time.perf_counter() - start,
        "limitations": [
            "Fresh fully 3D component experiment, not recovered reference data or a complete hybrid result.",
            "Both solves reinitialize flux and time history from identical cell velocities and pressures.",
            "Uses implicit Euler and tighter solves to isolate closure; does not reproduce the tutorial BDF2/adaptive-time trajectory.",
            "Trace variants are diagnostic; they do not replay every full-domain convection, diffusion, pressure and correction stencil.",
            "Native flux derivatives use unit scalar coefficients; variable SGS viscosity and component pressure coefficients need separate matching.",
            "No VPM, transfer, macro-step interpolation, or long-time force statistics.",
        ],
    }
    (output / "cube-boundary-oracle.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_report(report, output)
    return report


def plot_report(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "dirichlet": "Velocity + flux pressure",
        "vorticity_mixed": "Mixed + flux pressure",
        "pressure_gradient": "Velocity + reference ∇p",
        "vorticity_mixed_pressure_gradient": "Mixed + reference ∇p",
    }
    metrics = (
        ("velocity_rms_over_Uinf", "Shared volume: velocity RMS / U∞"),
        ("near_body_velocity_rms_over_Uinf", "Near cube: velocity RMS / U∞"),
        ("drag_coefficient_difference", "Small − full: drag coefficient"),
        ("pressure_gauge_removed_rms_over_Uinf_squared", "Pressure RMS / U∞² (gauge removed)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(bottom=0.19, top=0.86, hspace=0.4, wspace=0.3)
    for mode, label in labels.items():
        rows = [row for row in report["results"] if row["mode"] == mode and row["step"] > 0]
        if not rows:
            continue
        for i, (axis, (metric, title)) in enumerate(zip(axes.flat, metrics, strict=True)):
            axis.plot(
                [row["elapsed_flow_time"] for row in rows],
                [row[metric] for row in rows],
                label=label,
            )
            axis.set(title=title, xlabel="Elapsed flow time U∞/D")
            if i != 2:
                axis.set_yscale("log")
            axis.grid(alpha=0.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        f"Fully 3D cube: {report['small_cells']:,} inherited cells in a 3D × 3D × 3D box\n"
        "Reference FVM boundary data; no VPM or transfer"
    )
    fig.savefig(output / "cube-boundary-oracle.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dx", type=float, default=0.125)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--outer-correctors", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--laminar", action="store_true")
    parser.add_argument("--velocity-trace", choices=("interpolated", "native_flux", "native_value"),
                        default="interpolated", help="Diagnostic reference velocity trace")
    parser.add_argument("--pressure-trace", choices=("lsq", "native_flux", "native_value"),
                        default="lsq", help="Diagnostic reference pressure trace")
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0 or args.dt <= 0 or args.dx <= 0:
        parser.error("steps, dt and dx must be positive; warmup steps must be nonnegative")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must be distinct")
    run(args)
