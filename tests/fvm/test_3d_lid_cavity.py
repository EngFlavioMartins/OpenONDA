"""Mesh-refined cubic lid-cavity comparison with published 3D data."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box

# Albensoeder & Kuhlmann, JCP 206 (2005), 536–558,
# doi:10.1016/j.jcp.2004.12.024. Coordinates and velocities are normalized by
# cavity length and lid speed at Re=1000 on the z=0.5 symmetry plane.
REFERENCE = {
    "u_min": (-0.2803833, 0.12419),
    "v_min": (-0.4350186, 0.90957),
    "v_max": (0.2466511, 0.10913),
}


def _cosine_clustered_cube(level: int) -> tuple[dict, np.ndarray]:
    mesh = structured_box(level, level, level)
    nodes = 0.5 * (1.0 - np.cos(np.pi * np.arange(level + 1) / level))
    for axis in range(3):
        indices = np.rint(mesh["points"][:, axis] * level).astype(int)
        mesh["points"][:, axis] = nodes[indices]
    return mesh, 0.5 * (nodes[:-1] + nodes[1:])


def _run_cavity(level: int) -> tuple[np.ndarray, float, float]:
    mesh, coordinates = _cosine_clustered_cube(level)
    params_schemes = SchemesConfig(convection_scheme="limitedLinear")
    params_linear = LinearSolverConfig(linear_solver="spsolve")
    params_pimple = PimpleControl(algorithm="SIMPLE", alpha_u=0.7, alpha_p=0.3)
    config = FVMSetup(
        case_name=f"cubic-cavity-{level}",
        time=TimeConfig.transient(dt=0.01, duration=50.0, write_interval=10**9),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        transport=TransportConfig(density=1.0, nu=1.0e-3),
        boundaries=[
            BoundaryConfig.wall("xmin"),
            BoundaryConfig.wall("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig(
                "ymax",
                type_velocity="fixedValue",
                value_velocity=[1.0, 0.0, 0.0],
                type_p="zeroGradient",
            ),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.0, 0.0, 0.0],
        initial_p=0.0,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, mesh_data=mesh)
        solver.auto_write = False
        for _ in range(5000):
            solver.evolve()
            increment = solver.last_diagnostics.residuals["U_increment"]
            if increment < 2.0e-5:
                break
        else:
            raise AssertionError(f"cavity level {level} did not reach the nonlinear tolerance")

    velocity = solver.U[: mesh["n_elements"]].reshape(level, level, level, 3)
    middle = [level // 2] if level % 2 else [level // 2 - 1, level // 2]
    u_line = velocity[np.ix_(middle, np.arange(level), middle, [0])].mean(axis=(0, 2, 3))
    v_line = velocity[np.ix_(middle, middle, np.arange(level), [1])].mean(axis=(0, 1, 3))
    samples = np.asarray(
        [
            np.interp(REFERENCE["u_min"][1], coordinates, u_line),
            np.interp(REFERENCE["v_min"][1], coordinates, v_line),
            np.interp(REFERENCE["v_max"][1], coordinates, v_line),
        ]
    )
    return samples, increment, solver.last_diagnostics.continuity_max


@pytest.mark.verification
@pytest.mark.slow
def test_cubic_lid_cavity_converges_toward_published_centerline_data():
    reference = np.asarray([REFERENCE[name][0] for name in ("u_min", "v_min", "v_max")])
    levels = (6, 8, 12)
    results = [_run_cavity(level) for level in levels]
    errors = np.asarray([np.linalg.norm(samples - reference) for samples, _, _ in results])

    assert np.all(np.diff(errors) < 0.0), f"non-monotone cavity errors: {errors}"
    assert errors[-1] < 0.22, f"finest cavity reference error is too large: {errors[-1]:.3f}"
    assert max(result[1] for result in results) < 2.0e-5
    assert max(result[2] for result in results) < 1.0e-10
