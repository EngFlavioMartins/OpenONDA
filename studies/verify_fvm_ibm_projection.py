"""Verify the actual cylinder IBM setup beyond its former startup divergence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

import openonda.fvm as fvm
from openonda.tutorial_support.fvm_cylinder_ibm.mesh_rectilinear import cylinder_ibm_mesh
from source.solvers.fvm.core.solver import FVMSolver


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "cylinder_ibm_setup", root / "tutorials/fvm/cylinder_ibm/setup.py"
    )
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    directory = root / "build/fvm_ibm_projection_verification"
    directory.mkdir(parents=True, exist_ok=True)
    spacing = case.SPACING
    viscosity = case.FREESTREAM_VELOCITY * case.DIAMETER / case.REYNOLDS_NUMBER
    maximum_step = min(case.MAX_TIME_STEP_SIZE, case.MAX_FORCING_FOURIER * spacing**2 / viscosity)
    setup = case.create_fvm_setup(
        case.REYNOLDS_NUMBER, 0.6, spacing, min(case.TIME_STEP_SIZE, maximum_step), maximum_step
    )
    mesh, _ = cylinder_ibm_mesh(grid_spacing=spacing, diameter=case.DIAMETER)
    rows = []
    with FVMSolver(setup, str(directory), mesh_data=mesh) as solver:
        solver.auto_write = False
        solver.set_immersed_bodies(
            fvm.ImmersedBody.cylinder_z(
                centre=[0, 0, spacing / 2],
                diameter=case.DIAMETER,
                grid_spacing=spacing,
                marker_spacing_ratio=case.MARKER_ALPHA,
                name="cylinder",
            ),
            grid_spacing=spacing,
        )
        n = mesh["n_cells"]
        volumes = solver.geo_data["cell_volume"][:n]
        energy0 = 0.5 * case.DENSITY * np.sum(volumes)
        while solver.time < 0.6 - 1e-12:
            solver.advance()
            velocity = solver.velocity[:n]
            phi = solver.volumetric_face_flux
            ni = mesh["n_interior_faces"]
            div = np.bincount(mesh["owners"], weights=phi, minlength=n)
            div -= np.bincount(mesh["neighbours"][:ni], weights=phi[:ni], minlength=n)
            rows.append(
                {
                    "time": float(solver.time),
                    "step": int(solver.step),
                    "max_velocity": float(np.max(np.linalg.norm(velocity, axis=1))),
                    "slip": float(solver.ibm.slip_error(solver.velocity)),
                    "max_divergence": float(np.max(np.abs(div / volumes))),
                    "energy_ratio": float(
                        np.sum(volumes * np.sum(velocity**2, axis=1)) * case.DENSITY / (2 * energy0)
                    ),
                    "drag_coefficient": float(
                        solver.ibm.body_forces(case.DENSITY)["cylinder"][0]
                        / (
                            0.5
                            * case.DENSITY
                            * case.FREESTREAM_VELOCITY**2
                            * case.DIAMETER
                            * spacing
                        )
                    ),
                }
            )
    result = {
        "initial_time_step_size": case.TIME_STEP_SIZE,
        "maximum_time_step_size": maximum_step,
        "mesh_cells": n,
        "end_time": rows[-1]["time"],
        "steps": len(rows),
        "history": rows,
        "max_velocity": max(row["max_velocity"] for row in rows),
        "max_divergence": max(row["max_divergence"] for row in rows),
        "final_slip": rows[-1]["slip"],
        "final_energy_ratio": rows[-1]["energy_ratio"],
    }
    (root / "studies/fvm_ibm_projection_verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    assert result["end_time"] >= 0.6 - 1e-12
    assert result["max_velocity"] < 3
    assert result["max_divergence"] < 1e-9
    assert result["final_slip"] < 0.01
    assert result["final_energy_ratio"] < 1.1
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
