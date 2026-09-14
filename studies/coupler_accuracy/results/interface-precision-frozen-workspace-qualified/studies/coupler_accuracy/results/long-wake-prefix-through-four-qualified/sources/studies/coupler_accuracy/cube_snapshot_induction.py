#!/usr/bin/env python3
"""Measure fully 3D omega*volume induction against native cube cell velocities.

Uses an immutable result from cube_boundary_oracle.py. Every particle has a
three-component strength and uses the free-space 3D Gaussian kernel. There
are no repeated spans, 2D kernels, time advancement, pruning or tree errors.
An optional finite-domain surface term and native cell divergence separate
boundary-completion and FVM kinematic consistency from particle quadrature.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.special import erf

import openonda.fvm as fvm
from source.solvers.fvm.io.mesh_storage import load_native_mesh
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from studies.coupler_accuracy.cube_boundary_oracle import field_rms, hash_file, setup_for


def volume_velocity(target, position, strength, sigma, divergence_strength=None):
    """Independent float64 direct Gaussian vorticity/potential quadrature."""
    target = np.asarray(target, dtype=np.float64)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (len(position),))
    velocity = np.zeros_like(target)
    potential = np.zeros_like(target)
    for first in range(0, len(position), 512):
        rows = slice(first, first + 512)
        delta = target[:, None] - position[None, rows]
        r2 = np.sum(delta**2, axis=-1)
        radius = sigma[None, rows]
        q = np.sqrt(r2) / radius
        safe = np.where(r2 > 0, r2, 1)
        factor = (erf(q) - 2 / np.sqrt(np.pi) * q * np.exp(-(q**2))) / (4 * np.pi * safe**1.5)
        small = q < 0.01
        origin = (1 - 0.6 * q**2 + 3 / 14 * q**4) / (3 * np.pi**1.5 * radius**3)
        factor[small] = origin[small]
        velocity += np.sum(np.cross(strength[None, rows], delta) * factor[..., None], axis=1)
        if divergence_strength is not None:
            potential += np.sum(
                divergence_strength[None, rows, None] * delta * factor[..., None], axis=1
            )
    return velocity, potential


def boundary_velocity(target, position, area_vector, velocity):
    """Midpoint quadrature of the bounded-domain Helmholtz surface term.

    r = target - source, n points out of the fluid. The contribution is
    -integral[((n.u) r + (n x u) x r)/(4 pi |r|^3)] dS.
    Targets must not lie on the source surface.
    """
    result = np.zeros_like(target, dtype=np.float64)
    for first in range(0, len(position), 512):
        rows = slice(first, first + 512)
        delta = target[:, None] - position[None, rows]
        r2 = np.sum(delta**2, axis=-1)
        if np.any(r2 == 0):
            raise ValueError("Boundary quadrature targets must be off the surface")
        normal_strength = np.einsum("ij,ij->i", area_vector[rows], velocity[rows])
        tangent_strength = np.cross(area_vector[rows], velocity[rows])
        numerator = normal_strength[None, :, None] * delta + np.cross(tangent_strength[None], delta)
        result -= np.sum(numerator / (4 * np.pi * r2[..., None] ** 1.5), axis=1)
    return result


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.oracle.resolve()
    source_report = json.loads((source / "cube-boundary-oracle.json").read_text())
    if source_report.get("spatial_dimensions") != 3 or not source_report.get(
        "valid_for_comparison"
    ):
        raise ValueError("Expected a qualified fully 3D cube oracle")
    mesh = load_native_mesh(source / "full-native-mesh.npz")
    geometry = compute_mesh_geometry(mesh, gradient_scheme="gauss", compute_lsq=False)
    with np.load(source / "final-full-cell-fields.npz", allow_pickle=False) as data:
        centres, volumes, velocity, pressure, omega = (
            data[k].copy() for k in ("centres", "cell_volumes", "velocity", "pressure", "vorticity")
        )
        physical_time = float(data["physical_time"])
    np.testing.assert_allclose(centres, geometry["cell_centre"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(volumes, geometry["cell_volume"], rtol=1e-13, atol=1e-14)
    with fvm.create_fvm_solver(
        setup_for(mesh, "snapshot", 0.01, 1, turbulence=source_report["numerics"]["sgs"] != "none"),
        case_dir=output / "field-reconstruction",
        mesh=copy.deepcopy(mesh),
    ) as solver:
        solver.set_initial_state(velocity, pressure)
        reconstructed_omega = solver.get_vorticity_field()
        omega_difference = float(np.max(np.abs(reconstructed_omega - omega)))
        gradient = solver.get_velocity_gradient_field()
        divergence = np.trace(gradient, axis1=1, axis2=2)
        boundary_position, boundary_area, boundary_values = [], [], []
        for patch in mesh["boundary"]:
            if patch["name"] == "cube":
                continue  # The no-slip wall has exactly zero surface velocity.
            first, count = patch["start_face"], patch["n_faces"]
            boundary_position.append(geometry["face_centre"][first : first + count])
            boundary_area.append(geometry["face_area_vector"][first : first + count])
            boundary_values.append(solver.get_boundary_face_velocity(patch["name"]))
    generator = np.random.default_rng(20260913)
    radius = np.max(np.abs(centres), axis=1)
    groups = {
        "near_cube": radius < 1,
        "outer_small_domain": (radius > 1.1) & (radius < 1.5),
        "wake": (centres[:, 0] > 1.5)
        & (centres[:, 0] < 5)
        & (np.max(np.abs(centres[:, 1:]), axis=1) < 1.5),
    }
    selected, labels = [], []
    for name, mask in groups.items():
        ids = np.flatnonzero(mask)
        ids = np.sort(generator.choice(ids, min(args.samples_per_region, len(ids)), replace=False))
        selected.extend(ids)
        labels.extend([name] * len(ids))
    ids = np.asarray(selected)
    labels = np.asarray(labels)
    target, expected = centres[ids], velocity[ids]
    boundary_position = np.concatenate(boundary_position)
    boundary_area = np.concatenate(boundary_area)
    boundary_values = np.concatenate(boundary_values)
    # The closed exterior box's constant-freestream contribution is exactly
    # Uinf at every interior target. Integrate only the small residual field.
    completion = boundary_velocity(
        target, boundary_position, boundary_area, boundary_values - [1, 0, 0]
    )
    outputs, records = {}, []
    for name, sigma in (
        ("sigma_0.0625", 0.0625),
        ("sigma_0.125", 0.125),
        ("sigma_0.25", 0.25),
        ("sigma_cell_volume_scale", np.cbrt(volumes)),
    ):
        induced, potential = volume_velocity(
            target, centres, volumes[:, None] * omega, sigma, volumes * divergence
        )
        variants = {
            "particles_and_freestream": induced + [1, 0, 0],
            "plus_outer_boundary": induced + [1, 0, 0] + completion,
            "plus_native_divergence": induced + [1, 0, 0] + completion + potential,
        }
        for variant, actual in variants.items():
            outputs[f"{name}__{variant}"] = actual
            for group in groups:
                mask = labels == group
                record = {
                    "core": name,
                    "representation": variant,
                    "region": group,
                    "velocity_rms_over_Uinf": field_rms(
                        (actual - expected)[mask], volumes[ids][mask]
                    ),
                    "velocity_max_over_Uinf": float(
                        np.linalg.norm((actual - expected)[mask], axis=1).max()
                    ),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    np.savez_compressed(
        output / "held-out-velocities.npz",
        cell_ids=ids,
        position=target,
        region=labels,
        expected=expected,
        boundary_completion=completion,
        **outputs,
    )
    source_paths = [
        Path(__file__),
        source / "cube-boundary-oracle.json",
        source / "full-native-mesh.npz",
        source / "final-full-cell-fields.npz",
    ]
    report = {
        "schema": "openonda-cube-snapshot-induction/1",
        "spatial_dimensions": 3,
        "physical_time": physical_time,
        "source_cells": len(centres),
        "held_out_cells": len(ids),
        "reconstructed_vorticity_max_difference": omega_difference,
        "native_velocity_divergence_rms": field_rms(divergence[:, None], volumes),
        "outer_boundary_completion_rms_over_Uinf": field_rms(completion, volumes[ids]),
        "records": records,
        "sources": [
            hash_file(p)
            for p in source_paths
        ],
        "limitations": [
            "Fixed native cell field; no VPM evolution, tree approximation, panel correction, transfer or pruning.",
            "Cell vorticity is a discrete FVM curl, and cell quadrature/core smoothing are approximations.",
            "Outer-boundary surface quadrature is an offline diagnostic, not a proposed live coupling boundary.",
            "Native divergence correction is diagnostic only; incompressible VPM velocity is solenoidal.",
        ],
    }
    (output / "cube-snapshot-induction.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-region", type=int, default=256)
    run(parser.parse_args())
