#!/usr/bin/env python3
"""Measure the cropped/full momentum-operator difference at identical 3D fields.

No solve or time advance. The cropped face flux is copied exactly, including
interior faces. This isolates spatial operator changes from pressure correction,
nonlinear iteration and particle data. All cases use prescribed pressure traces.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
from source.solvers.fvm.assemble.convection import assemble_convection_term
from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
from source.solvers.fvm.assemble.momentum import compute_dev2_stress_source
from source.solvers.fvm.fields.gradients import _resolve_gradient_fn
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from studies.coupler_accuracy.cube_boundary_oracle import ROOT, field_rms, hash_file, setup_for
from studies.coupler_accuracy.native_face_trace_3d import tangential


def divergence(flux, mesh, geometry):
    """Signed owner/neighbour reduction of integrated face fluxes."""
    result = np.zeros((mesh["n_cells"], 3))
    np.add.at(result, mesh["owners"], flux)
    np.add.at(result, mesh["neighbours"], -flux[:mesh["n_interior_faces"]])
    return result / geometry["cell_volume"][:, None]


def observe(solver, face_flux):
    mesh, geometry = solver.mesh_data, solver.geo_data
    gradient_fn = _resolve_gradient_fn(geometry)
    gradient = gradient_fn(solver.velocity, mesh, geometry)
    pressure_gradient = gradient_fn(solver.kinematic_pressure, mesh, geometry)[:mesh["n_cells"], :, 0]
    convective, diffusive = [], []
    for component in range(3):
        convective.append(assemble_convection_term(
            solver.velocity[:, component], face_flux, mesh, geometry, solver.boundaries,
            scheme="linearUpwind", scalar_field_gradient=gradient[:, :, component],
            component=component,
        )["flux_tf"])
        diffusive.append(assemble_diffusion_term(
            solver.velocity[:, component], gradient[:, :, component], 0.001, mesh, geometry,
            solver.boundaries, volumetric_face_flux=face_flux, vector_field=solver.velocity,
            component=component,
        )["flux_tf"])
    convective, diffusive = np.column_stack(convective), np.column_stack(diffusive)
    terms = {
        "convection": -divergence(convective, mesh, geometry),
        "laplacian": -divergence(diffusive, mesh, geometry),
        "transpose_stress": compute_dev2_stress_source(gradient, 0.001, mesh, geometry),
        "pressure": -pressure_gradient,
    }
    terms["total"] = sum(terms.values())
    return terms, convective, diffusive


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    dependency_paths = [
        Path(__file__), ROOT / "studies/coupler_accuracy/cube_boundary_oracle.py",
        ROOT / "studies/coupler_accuracy/native_face_trace_3d.py",
        *[ROOT / f"source/solvers/fvm/{name}" for name in (
            "assemble/convection.py", "assemble/diffusion.py", "assemble/momentum.py",
            "fields/gradients.py", "fields/mixed_velocity_boundary.py",
            "coupling/coupler_interface.py",
        )],
    ]
    sources = [hash_file(path) for path in dependency_paths]
    for path in dependency_paths:
        target = args.output / "sources" / path.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    mesh, small_mesh = [load_native_mesh(args.oracle / name) for name in
                        ("full-native-mesh.npz", "small-native-mesh.npz")]
    with np.load(args.oracle / "initial-cell-fields.npz", allow_pickle=False) as data:
        seed = {key: data[key].copy() for key in data.files}
    with np.load(args.oracle / "cell-and-face-map.npz", allow_pickle=False) as data:
        ids, faces, signs = [data[key].copy() for key in ("cell_ids", "face_ids", "signs")]
    with np.load(args.traces, allow_pickle=False) as data:
        traces = {key: data[key].copy() for key in data.files}
    patch = next(b for b in small_mesh["boundary"] if b["name"] == "numericalBoundary")
    cut = slice(patch["start_face"], patch["start_face"] + patch["n_faces"])
    records, fields = [], {}
    with fvm.create_fvm_solver(setup_for(mesh, "operator-full", 0.01, 1, turbulence=False),
                               mesh=copy.deepcopy(mesh), case_dir=args.output / "full") as full:
        np.testing.assert_allclose(full.get_cell_centre_coordinates(), seed["centres"], atol=1e-13, rtol=0)
        full.set_initial_state(seed["velocity"], seed["pressure"])
        full_terms, full_convection, full_diffusion = observe(full, full.volumetric_face_flux)
        phi = full.volumetric_face_flux[faces] * signs
        area, normal = traces["area"], traces["normal"]
        np.testing.assert_allclose(phi[cut] / area, traces["conservative_normal_velocity"], atol=1e-13, rtol=0)
        expected_convection = full_convection[faces[cut]] * signs[cut, None]
        expected_diffusion = full_diffusion[faces[cut]] * signs[cut, None]
        for label, velocity_trace, pressure_trace in (
            ("interpolated_lsq", "interpolated", "lsq_interpolated"),
            ("native_flux_lsq", "native_flux", "lsq_interpolated"),
            ("native_flux_native_flux", "native_flux", "native_flux"),
            ("native_value_native_value", "native_value", "native_value"),
        ):
            with fvm.create_fvm_solver(setup_for(small_mesh, label, 0.01, 1, turbulence=False),
                                       mesh=copy.deepcopy(small_mesh), case_dir=args.output / label) as small:
                small.set_normal_velocity_tangential_gradient_boundary_condition(
                    traces["conservative_normal_velocity"], traces[velocity_trace + "_tangential_gradient"],
                    "numericalBoundary",
                )
                small.set_neumann_pressure_boundary_condition(
                    traces[pressure_trace + "_pressure_normal_gradient"][:, None] * normal,
                    "numericalBoundary",
                )
                small.set_initial_state(seed["velocity"][ids], seed["pressure"][ids])
                np.testing.assert_array_equal(small.get_velocity_field(), full.get_velocity_field()[ids])
                np.testing.assert_array_equal(small.kinematic_pressure[:len(ids)], full.kinematic_pressure[ids])
                np.testing.assert_allclose(small.geo_data["cell_volume"], full.geo_data["cell_volume"][ids], atol=1e-14, rtol=1e-13)
                actual, convective, diffusive = observe(small, phi)
                volume = small.geo_data["cell_volume"]
                owners = np.unique(small.mesh_data["owners"][cut])
                difference = {name: value - full_terms[name][ids] for name, value in actual.items()}
                row = {"name": label, "momentum_acceleration_rms_over_Uinf_squared_per_D": {
                    name: field_rms(value, volume) for name, value in difference.items()
                }, "cut_owner_acceleration_rms_over_Uinf_squared_per_D": {
                    name: field_rms(value[owners], volume[owners]) for name, value in difference.items()
                }}
                for name, flux, expected in (("convection", convective, expected_convection),
                                             ("laplacian", diffusive, expected_diffusion)):
                    delta = (flux[cut] - expected) / area[:, None]
                    row[name + "_cut_flux_density_difference_rms"] = field_rms(delta, area)
                    row[name + "_cut_tangential_flux_density_difference_rms"] = field_rms(tangential(delta, normal), area)
                if velocity_trace == "native_flux":
                    np.testing.assert_allclose(tangential(diffusive[cut] - expected_diffusion, normal), 0, atol=3e-17, rtol=0)
                if pressure_trace == "native_value":
                    np.testing.assert_allclose(difference["pressure"], 0, atol=2e-13, rtol=0)
                fields.update({label + "__" + name: value for name, value in difference.items()})
                records.append(row)
                print(json.dumps(row), flush=True)
    np.savez_compressed(args.output / "operator-differences.npz", centres=seed["centres"][ids], volumes=volume, **fields)
    input_paths = [args.traces, *[args.oracle / name for name in (
        "full-native-mesh.npz", "small-native-mesh.npz", "initial-cell-fields.npz", "cell-and-face-map.npz")]]
    report = {
        "schema": "openonda-frozen-boundary-operator-3d/1", "spatial_dimensions": 3,
        "physical_time": float(seed["physical_time"]), "full_cells": mesh["n_cells"],
        "small_cells": small_mesh["n_cells"], "results": records,
        "sources": sources + [hash_file(path) for path in input_paths],
        "limitations": [
            "Frozen identical primitive fields and face fluxes; no pressure correction, time advance or particles.",
            "Laminar constant viscosity 0.001; all rows prescribe the indicated pressure trace.",
            "Terms are momentum accelerations, not accumulated velocity/force errors or a complete nonlinear residual.",
        ],
    }
    (args.output / "boundary-operator-audit-3d.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
