"""T0a — full-solver Taylor–Green vortex validation (decaying, analytic).

Drives the *integrated* PIMPLE solver (momentum + pressure + Rhie–Chow + BDF2) on
the periodic 2-D decaying Taylor–Green vortex using translational cyclic pairs.
The
analytic solution on ``[0, 2π]²`` is

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

from source.solvers.FVM import (
    BoundaryConfig,
    FVMConfig,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)

from ._structured_mesh import structured_box

TWO_PI = 2.0 * np.pi


def _tgv_U(x, y, t, nu):
    F = np.exp(-2.0 * nu * t)
    return np.column_stack(
        [F * np.sin(x) * np.cos(y), -F * np.cos(x) * np.sin(y), np.zeros_like(x)]
    )


def _run(N, scheme, nu=0.1, dt=0.005, nsteps=10):
    """Return (relative L2 velocity error at T, KE(T), analytic KE(T), KE(0))."""
    mesh = structured_box(N, N, 1, lx=TWO_PI, ly=TWO_PI, lz=TWO_PI / N)
    sp = SolverParams.pimple(
        n_correctors=2, n_outer=1, linear_solver="spsolve", convection_scheme=scheme
    )
    sp.time_scheme = "backward"
    bnds = [
        BoundaryConfig.cyclic("xmin", "xmax"),
        BoundaryConfig.cyclic("xmax", "xmin"),
        BoundaryConfig.cyclic("ymin", "ymax"),
        BoundaryConfig.cyclic("ymax", "ymin"),
    ]
    bnds += [BoundaryConfig.empty("zmin"), BoundaryConfig.empty("zmax")]
    cfg = FVMConfig(
        case_name="tgv",
        time=TimeConfig(delta_t=dt, end_time=dt * nsteps, write_interval=10**9),
        solver=sp,
        transport=TransportConfig(density=1.0, nu=nu),
        boundaries=bnds,
        initial_U=[0, 0, 0],
    )
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        s = Solver(cfg, case_dir=d, mesh_data=mesh)
        s.auto_write = False
        ne = mesh["n_elements"]
        nint = mesh["n_interior_faces"]
        cc, vol = (
            s.geo_data["element_centroids"],
            s.geo_data["element_volumes"],
        )

        s.U[:ne] = _tgv_U(cc[:, 0], cc[:, 1], 0.0, nu)
        for face in range(nint, mesh["n_faces"]):
            ghost = ne + face - nint
            paired = mesh["boundary_neighbours"][face]
            if paired >= 0:
                s.U[ghost] = s.U[paired]
        s.U_old[:] = s.U
        s.U_old_old[:] = s.U
        ke0 = 0.5 * float(np.sum(np.sum(s.U[:ne] ** 2, axis=1) * vol))

        t = 0.0
        for _ in range(nsteps):
            t += dt
            s.solve_pimple(dt)
            s.advance_time()

        Uex = _tgv_U(cc[:, 0], cc[:, 1], t, nu)
        rel = float(
            np.sqrt(np.sum(vol[:, None] * (s.U[:ne] - Uex) ** 2) / np.sum(vol))
            / np.sqrt(np.sum(vol[:, None] * Uex**2) / np.sum(vol))
        )
        ke = 0.5 * float(np.sum(np.sum(s.U[:ne] ** 2, axis=1) * vol))
        return rel, ke, ke0 * np.exp(-4 * nu * t), ke0


@pytest.mark.slow
class TestTaylorGreenValidation:
    def test_central_less_diffusive_and_tracks_analytic_KE(self):
        rel_c, ke_c, ke_a, ke0 = _run(24, "central")
        rel_u, ke_u, _, _ = _run(24, "upwind")

        # Central tracks the analytic KE decay to <0.2%; upwind under-predicts
        # (numerical dissipation removes kinetic energy).
        assert abs(ke_c - ke_a) / ke_a < 2e-3, f"central KE error {(ke_c - ke_a) / ke_a:.2e}"
        assert (ke_u - ke_a) / ke_a < -1e-3, "upwind should under-predict KE (dissipative)"

        # The "less diffusive" claim end-to-end: central is several× more accurate.
        assert rel_c < rel_u / 4.0, f"central relL2 {rel_c:.2e} not << upwind {rel_u:.2e}"
        assert rel_c < 2e-3, f"central relL2 {rel_c:.2e} too large"

    def test_solver_converges_under_refinement(self):
        rel16, *_ = _run(16, "central")
        rel24, *_ = _run(24, "central")
        order = np.log(rel16 / rel24) / np.log(24.0 / 16.0)
        assert order > 0.7, f"observed coupled order {order:.2f} — not converging"
