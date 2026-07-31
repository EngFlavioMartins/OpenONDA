"""Coarse-grid 3D WALE LES against the published Taylor--Green decay."""

from __future__ import annotations

import contextlib
import io
import tempfile

import numpy as np

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
    TurbulenceConfig,
)
from source.solvers.FVM.assemble.convection import compute_volumetric_face_flux
from source.solvers.FVM.solve.simple_solver import update_scalar_boundaries

from ._structured_mesh import structured_box

TWO_PI = 2.0 * np.pi
DNS_PEAK_DISSIPATION = 0.01289
DNS_PEAK_TIME = 8.86


def _run_wale_decay(level: int) -> tuple[float, float, float]:
    """Return peak dissipation, peak time, and continuity for a coarse LES."""
    steps = 113
    dt = 0.08
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
        time=TimeConfig.transient(dt=dt, duration=steps * dt, write_interval=10**9),
        schemes=SchemesConfig(convection_scheme="central", time_scheme="backward"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="amg",
            momentum_tol=1e-8,
            pressure_tol=1e-9,
        ),
        pimple=PimpleControl(n_correctors=2, n_outer_correctors=1),
        transport=TransportConfig(density=1.0, nu=1.0 / 1600.0),
        turbulence=TurbulenceConfig.wale(),
        boundaries=boundaries,
        initial_U=[0.0, 0.0, 0.0],
    )

    with tempfile.TemporaryDirectory() as case_dir, contextlib.redirect_stdout(io.StringIO()):
        solver = Solver(config, case_dir=case_dir, mesh_data=mesh)
        solver.auto_write = False
        n_cells = mesh["n_elements"]
        centers = solver.geo_data["element_centroids"]
        x, y, z = centers.T
        velocity = np.column_stack(
            (
                np.sin(x) * np.cos(y) * np.cos(z),
                -np.cos(x) * np.sin(y) * np.cos(z),
                np.zeros_like(x),
            )
        )
        pressure = (np.cos(2.0 * x) + np.cos(2.0 * y)) * (np.cos(2.0 * z) + 2.0) / 16.0
        solver.set_initial_velocity(velocity)
        solver.p[:n_cells] = pressure - np.mean(pressure)
        update_scalar_boundaries(solver.p, mesh, solver.boundaries, field_name="p")
        solver.phi = compute_volumetric_face_flux(solver.U, mesh, solver.geo_data)

        volumes = solver.geo_data["element_volumes"]
        total_volume = np.sum(volumes)
        energy = [0.5 * np.sum(volumes * np.sum(solver.U[:n_cells] ** 2, axis=1)) / total_volume]
        for _ in range(steps):
            solver.solve_pimple(dt)
            solver.advance_time()
            energy.append(
                0.5 * np.sum(volumes * np.sum(solver.U[:n_cells] ** 2, axis=1)) / total_volume
            )

    times = np.arange(steps + 1) * dt
    dissipation = -np.gradient(np.asarray(energy), times, edge_order=2)
    peak_index = int(np.argmax(dissipation))
    return (
        float(dissipation[peak_index]),
        float(times[peak_index]),
        solver.last_diagnostics.continuity_max,
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
