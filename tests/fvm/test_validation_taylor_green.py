"""Full-solver 2D Taylor–Green vortex validation (decaying, analytic).

Drives PIMPLE (momentum, pressure, Rhie–Chow, and BDF2) on the periodic
two-dimensional solution using translational cyclic pairs. The analytic
solution on ``[0, 2π]²`` is

    u = e^{-2νt}( sin x cos y, −cos x sin y, 0 ),
    KE(t) = KE(0) · e^{-4νt}.

Checks the headline DNS/LES property — the energy-conserving central scheme is far
less diffusive than upwind and tracks the analytic kinetic-energy decay, whereas
upwind under-predicts KE (numerical dissipation).

This is also the integrated cyclic-boundary gate: opposite-face fluxes must pair,
the pressure solve must use its all-Neumann protocol, and no imposed outer value
may mask a periodic boundary error.
"""

import contextlib
import io
import tempfile

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box

TWO_PI = 2.0 * np.pi


def _tgv_U(x, y, t, kinematic_viscosity):
    F = np.exp(-2.0 * kinematic_viscosity * t)
    return np.column_stack(
        [F * np.sin(x) * np.cos(y), -F * np.cos(x) * np.sin(y), np.zeros_like(x)]
    )


def _run(N, scheme, kinematic_viscosity=0.1, time_step_size=0.005, nsteps=10):
    """Return (relative L2 velocity error at T, KE(T), analytic KE(T), KE(0))."""
    mesh = structured_box(N, N, 1, lx=TWO_PI, ly=TWO_PI, lz=TWO_PI / N)
    sp_schemes = DiscretizationConfig(convection_scheme=scheme, time_scheme="backward")
    sp_linear = LinearSolverConfig(linear_solver="spsolve")
    sp_pimple = PimpleControl(n_correctors=2, n_outer_correctors=1)
    bnds = [
        BoundaryConfig.cyclic("xmin", "xmax"),
        BoundaryConfig.cyclic("xmax", "xmin"),
        BoundaryConfig.cyclic("ymin", "ymax"),
        BoundaryConfig.cyclic("ymax", "ymin"),
    ]
    bnds += [BoundaryConfig.empty("zmin"), BoundaryConfig.empty("zmax")]
    cfg = FVMSetup(
        case_name="tgv",
        time=TimeConfig(
            time_step_size=time_step_size,
            end_time=time_step_size * nsteps,
            output_interval_steps=10**9,
        ),
        schemes=sp_schemes,
        linear=sp_linear,
        pimple=sp_pimple,
        transport=TransportConfig(density=1.0, kinematic_viscosity=kinematic_viscosity),
        boundaries=bnds,
        initial_velocity=[0, 0, 0],
    )
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        s = FVMSolver(cfg, case_dir=d, mesh_data=mesh)
        s.auto_write = False
        ne = mesh["n_cells"]
        cc, vol = (
            s.geo_data["cell_centre"],
            s.geo_data["cell_volume"],
        )

        s.set_initial_velocity(_tgv_U(cc[:, 0], cc[:, 1], 0.0, kinematic_viscosity))
        ke0 = 0.5 * float(np.sum(np.sum(s.velocity[:ne] ** 2, axis=1) * vol))

        t = 0.0
        for _ in range(nsteps):
            t += time_step_size
            s.solve_pimple(time_step_size)
            s.advance_time()

        Uex = _tgv_U(cc[:, 0], cc[:, 1], t, kinematic_viscosity)
        rel = float(
            np.sqrt(np.sum(vol[:, None] * (s.velocity[:ne] - Uex) ** 2) / np.sum(vol))
            / np.sqrt(np.sum(vol[:, None] * Uex**2) / np.sum(vol))
        )
        ke = 0.5 * float(np.sum(np.sum(s.velocity[:ne] ** 2, axis=1) * vol))
        return rel, ke, ke0 * np.exp(-4 * kinematic_viscosity * t), ke0


@pytest.mark.slow
class TestTaylorGreenValidation:
    def test_central_less_diffusive_and_tracks_analytic_KE(self):
        rel_c, ke_c, ke_a, ke0 = _run(32, "central")
        rel_u, ke_u, _, _ = _run(32, "upwind")

        # Central tracks the analytic KE decay to <0.2%; upwind under-predicts
        # (numerical dissipation removes kinetic energy).
        assert abs(ke_c - ke_a) / ke_a < 2e-3, f"central KE error {(ke_c - ke_a) / ke_a:.2e}"
        assert (ke_u - ke_a) / ke_a < -1e-3, "upwind should under-predict KE (dissipative)"

        # The "less diffusive" claim end-to-end: central is several× more accurate.
        assert rel_c < rel_u / 4.0, f"central relL2 {rel_c:.2e} not << upwind {rel_u:.2e}"
        assert rel_c < 2e-3, f"central relL2 {rel_c:.2e} too large"

    def test_solver_converges_under_refinement(self):
        # Level 12 is pre-asymptotic for the coupled pressure/velocity solve;
        # 16/24/32 exhibits the designed second-order regime.
        levels = (16, 24, 32)
        errors = [_run(n, "central")[0] for n in levels]
        orders = [
            np.log(errors[i] / errors[i + 1]) / np.log(levels[i + 1] / levels[i]) for i in range(2)
        ]
        assert min(orders) > 1.8, f"observed coupled orders {orders} are not second-order"
