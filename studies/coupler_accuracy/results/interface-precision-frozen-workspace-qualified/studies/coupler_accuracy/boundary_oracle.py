#!/usr/bin/env python3
"""Advance real FVM solves with exact unsteady outer boundary data.

An advected, viscously decaying Taylor-Green field provides independent
velocity, gradient and pressure. No VPM or transfer error is present. This
isolates the implemented mixed boundary and pressure projection on a small
box. It is not a cylinder or a full coupled validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from source.coupler.boundary import apply_fvm_boundary, tangential_normal_velocity_gradient
from source.coupler.config.types import CouplerSetup
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
from source.solvers.fvm.mesh.rectilinear import coupling_box_mesh


def exact(position, time, viscosity=0.01):
    x, y, _ = np.asarray(position).T
    k = np.pi
    x = k * (x - time)
    y = k * y
    decay = np.exp(-2.0 * viscosity * k**2 * time)
    velocity = np.column_stack(
        (1.0 - decay * np.cos(x) * np.sin(y), decay * np.sin(x) * np.cos(y), np.zeros(len(x)))
    )
    jacobian = np.zeros((len(x), 3, 3))
    jacobian[:, 0, 0] = k * decay * np.sin(x) * np.sin(y)
    jacobian[:, 0, 1] = -k * decay * np.cos(x) * np.cos(y)
    jacobian[:, 1, 0] = k * decay * np.cos(x) * np.cos(y)
    jacobian[:, 1, 1] = -k * decay * np.sin(x) * np.sin(y)
    pressure = -0.25 * decay**2 * (np.cos(2 * x) + np.cos(2 * y))
    return velocity, jacobian, pressure


def exact_pressure_gradient(position, time, viscosity=0.01):
    x, y, _ = np.asarray(position).T
    k = np.pi
    decay_squared = np.exp(-4.0 * viscosity * k**2 * time)
    return (
        0.5
        * k
        * decay_squared
        * np.column_stack((np.sin(2 * k * (x - time)), np.sin(2 * k * y), np.zeros(len(x))))
    )


def solve(n, mode, output, dt=0.0025, steps=20):
    axis = np.linspace(-1, 1, n + 1)
    mesh = coupling_box_mesh(
        (-1, 1, -1, 1, -0.125, 0.125),
        2 / n,
        nodes=(axis, axis, np.array([-0.125, 0.125])),
        empty_spanwise=True,
    )
    setup = FVMSetup(
        case_name=f"oracle_{mode}_{n}",
        time=TimeConfig(
            time_step_size=dt,
            end_time=steps * dt,
            output_schedule=RunSchedule(every_n_steps=100000),
        ),
        schemes=DiscretizationConfig(
            convection_scheme="linear", gradient_scheme="lsq", time_scheme="backward"
        ),
        linear=LinearSolverConfig(
            linear_solver="spsolve",
            pressure_solver="spsolve",
            pressure_tolerance=1e-12,
            momentum_tolerance=1e-12,
            pressure_relative_tolerance=0.0,
            momentum_relative_tolerance=0.0,
        ),
        pimple=PimpleControl(
            n_outer_correctors=3, n_correctors=2, velocity_relaxation=1.0, pressure_relaxation=1.0
        ),
        transport=TransportConfig(kinematic_viscosity=0.01),
        boundaries=[
            BoundaryConfig(
                name="numericalBoundary",
                velocity_type="fixedValue",
                velocity_value=[1, 0, 0],
                pressure_type="fixedFluxPressure",
            ),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
        initial_velocity=[1, 0, 0],
    )
    with create_fvm_solver(setup, case_dir=output / f"{mode}_{n}", mesh=mesh) as solver:
        coupler = SimpleNamespace(
            fvm_solver=solver, setup=CouplerSetup(boundary_condition_mode=mode)
        )
        centre = solver.get_cell_centre_coordinates()
        face = solver.get_boundary_face_centre_coordinates("numericalBoundary")
        normal = solver.get_boundary_face_normal("numericalBoundary")
        initial, _, pressure = exact(centre, 0)
        velocity, jacobian, _ = exact(face, 0)
        if mode in {"vorticity_mixed", "vorticity_mixed_pressure_gradient"}:
            solver.set_normal_velocity_tangential_gradient_boundary_condition(
                np.einsum("ij,ij->i", velocity, normal),
                tangential_normal_velocity_gradient(jacobian, normal),
                "numericalBoundary",
            )
        else:
            solver.set_dirichlet_velocity_boundary_condition_vec(velocity, "numericalBoundary")
        if mode == "vorticity_mixed_pressure_gradient":
            solver.set_neumann_pressure_boundary_condition(
                exact_pressure_gradient(face, 0), "numericalBoundary"
            )
        solver.kinematic_pressure[: len(centre)] = pressure
        solver.set_initial_velocity(initial)
        for step in range(1, steps + 1):
            velocity, jacobian, _ = exact(face, step * dt)
            apply_fvm_boundary(
                coupler,
                "numericalBoundary",
                velocity,
                normal_velocity=np.einsum("ij,ij->i", velocity, normal),
                tangential_gradient=tangential_normal_velocity_gradient(jacobian, normal),
                pressure_gradient=exact_pressure_gradient(face, step * dt),
            )
        expected, _, _ = exact(centre, solver.time)
        error = solver.get_velocity_field() - expected
        volumes = solver.get_cell_volume()
        rms = np.sqrt(np.sum(volumes * np.sum(error**2, axis=1)) / volumes.sum())
        return {
            "n": n,
            "h": 2 / n,
            "mode": mode,
            "dt": dt,
            "steps": steps,
            "time": solver.time,
            "velocity_rms_error_over_Uinf": float(rms),
            "velocity_max_error_over_Uinf": float(np.linalg.norm(error, axis=1).max()),
        }


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in ("dirichlet", "vorticity_mixed", "vorticity_mixed_pressure_gradient"):
        for n in (8, 16, 32):
            result = solve(n, mode, output)
            results.append(result)
            print(json.dumps(result), flush=True)
    for mode in ("dirichlet", "vorticity_mixed", "vorticity_mixed_pressure_gradient"):
        series = [row for row in results if row["mode"] == mode]
        for previous, row in zip(series, series[1:], strict=False):
            row["observed_spatial_order"] = float(
                np.log2(
                    previous["velocity_rms_error_over_Uinf"] / row["velocity_rms_error_over_Uinf"]
                )
            )
    report = {
        "schema": "openonda-coupler-boundary-oracle/1",
        "results": results,
        "method": "Advected Taylor-Green exact endpoint data at every FVM substep; direct linear solves; fixed dt.",
        "limitations": "Combined FVM space/time/boundary error; no wall, VPM, interpolation in time, or transfer.",
    }
    (output / "boundary-oracle.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "results/boundary-oracle"
    )
    main(parser.parse_args().output)
