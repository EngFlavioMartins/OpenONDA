"""Coarse-grid 3D WALE LES against the published Taylor--Green decay."""

from __future__ import annotations

import contextlib
import io
import tempfile

import numpy as np

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from source.solvers.fvm.assemble.convection import compute_volumetric_face_flux
from source.solvers.fvm.solve.simple_solver import update_scalar_boundaries

from ._structured_mesh import structured_box

TWO_PI = 2.0 * np.pi
DNS_PEAK_DISSIPATION = 0.01289
DNS_PEAK_TIME = 8.86


def _run_wale_decay(level: int) -> tuple[float, float, float]:
    """Return peak dissipation, peak time, and continuity for a coarse LES."""
    steps = 113
    time_step_size = 0.08
    mesh = structured_box(level, level, level, lx=TWO_PI, ly=TWO_PI, lz=TWO_PI)
    boundaries = [
        BoundaryConfig.cyclic("xmin", "xmax"),
        BoundaryConfig.cyclic("xmax", "xmin"),
        BoundaryConfig.cyclic("ymin", "ymax"),
        BoundaryConfig.cyclic("ymax", "ymin"),
        BoundaryConfig.cyclic("zmin", "zmax"),
        BoundaryConfig.cyclic("zmax", "zmin"),
    ]
    config = FVMSetup(
        case_name=f"tgv-wale-{level}",
        time=TimeConfig.transient(
            time_step_size=time_step_size,
            duration=steps * time_step_size,
            output_interval_steps=10**9,
        ),
        schemes=DiscretizationConfig(convection_scheme="central", time_scheme="backward"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="amg",
            momentum_tolerance=1e-8,
            pressure_tolerance=1e-9,
        ),
        pimple=PimpleControl(n_correctors=2, n_outer_correctors=1),
        transport=TransportConfig(density=1.0, kinematic_viscosity=1.0 / 1600.0),
        turbulence=TurbulenceConfig.wale(),
        boundaries=boundaries,
        initial_velocity=[0.0, 0.0, 0.0],
    )

    with tempfile.TemporaryDirectory() as case_dir, contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(config, case_dir=case_dir, mesh_data=mesh)
        solver.auto_write = False
        n_cells = mesh["n_cells"]
        centres = solver.geo_data["cell_centre"]
        x, y, z = centres.T
        velocity = np.column_stack(
            (
                np.sin(x) * np.cos(y) * np.cos(z),
                -np.cos(x) * np.sin(y) * np.cos(z),
                np.zeros_like(x),
            )
        )
        pressure = (np.cos(2.0 * x) + np.cos(2.0 * y)) * (np.cos(2.0 * z) + 2.0) / 16.0
        solver.set_initial_velocity(velocity)
        solver.kinematic_pressure[:n_cells] = pressure - np.mean(pressure)
        update_scalar_boundaries(
            solver.kinematic_pressure, mesh, solver.boundaries, field_name="kinematic_pressure"
        )
        solver.volumetric_face_flux = compute_volumetric_face_flux(
            solver.velocity, mesh, solver.geo_data
        )

        volumes = solver.geo_data["cell_volume"]
        total_volume = np.sum(volumes)
        total_kinetic_energy_history = [
            0.5 * np.sum(volumes * np.sum(solver.velocity[:n_cells] ** 2, axis=1)) / total_volume
        ]
        for _ in range(steps):
            solver.solve_pimple(time_step_size)
            solver.advance_time()
            total_kinetic_energy_history.append(
                0.5
                * np.sum(volumes * np.sum(solver.velocity[:n_cells] ** 2, axis=1))
                / total_volume
            )

    times = np.arange(steps + 1) * time_step_size
    dissipation = -np.gradient(np.asarray(total_kinetic_energy_history), times, edge_order=2)
    peak_index = int(np.argmax(dissipation))
    return (
        float(dissipation[peak_index]),
        float(times[peak_index]),
        solver.last_diagnostics.max_continuity_error,
    )


def test_wale_taylor_green_decay_moves_toward_published_dns_under_refinement():
    """A coarse LES must preserve the DNS decay event and improve with refinement.

    The reference is the Re=1600 pseudo-spectral Taylor--Green result in
    van Rees et al., JCP 230 (2011), DOI 10.1016/j.jcp.2010.11.031. The peak
    values are digitized from its dissipation history. These 12^3 and 16^3
    runs are deliberately treated as coarse LES, not as DNS.
    """
    coarse = _run_wale_decay(12)
    fine = _run_wale_decay(16)

    coarse_error = np.hypot(
        (coarse[0] - DNS_PEAK_DISSIPATION) / DNS_PEAK_DISSIPATION,
        (coarse[1] - DNS_PEAK_TIME) / DNS_PEAK_TIME,
    )
    fine_error = np.hypot(
        (fine[0] - DNS_PEAK_DISSIPATION) / DNS_PEAK_DISSIPATION,
        (fine[1] - DNS_PEAK_TIME) / DNS_PEAK_TIME,
    )

    assert 4.0 < coarse[1] < 10.0
    assert 4.0 < fine[1] < 10.0
    assert 0.0 < coarse[0] < 0.02
    assert 0.0 < fine[0] < 0.02
    assert fine_error < coarse_error
    assert abs(fine[0] - DNS_PEAK_DISSIPATION) / DNS_PEAK_DISSIPATION < 0.30
    assert abs(fine[1] - DNS_PEAK_TIME) / DNS_PEAK_TIME < 0.35
    assert max(coarse[2], fine[2]) < 1e-10
