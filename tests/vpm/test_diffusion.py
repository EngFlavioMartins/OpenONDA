"""
Viscous diffusion accuracy tests for the VPM solver.

These tests verify the Core Spreading (CS) scheme against exact analytical
results that are independent of 2-D / 3-D normalisation ambiguity.

Design rationale
----------------
A Lamb-Oseen comparison requires many z-layers to approximate the 2-D line
vortex as an O(N²) 3-D sum — impractical for fast unit tests.  Instead we
use tests where the analytical answer is EXACT by construction:

* Core radius growth rate: σ² += C_diff · nu · dt where C_diff = 4 for the
  Gaussian kernel.  Single particle, O(1).

* Vorticity field post-diffusion: for a single particle CS only changes σ,
  not α.  The regularised vorticity ω(r) = α · ζ_σ(r) is therefore exactly
  known once σ is known.

Tests
-----
test_cs_core_radius_grows_at_correct_rate
    Verify σ_final² = σ₀² + 4nu·N·dt for the Gaussian kernel (C_diff = 4).
    Failure → wrong diffusivity constant or viscosity not reaching the kernel.

test_cs_vorticity_field_matches_grown_kernel
    Verify the regularised z-vorticity at probe points matches the 3-D
    Gaussian kernel evaluated with the grown σ_final.  CS never alters α,
    so deviations mean σ grew at the wrong rate or the kernel is mis-scaled.
"""

import numpy as np

from source.solvers.VPM import Solver, SolverConfig
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig

# ── Shared parameters ─────────────────────────────────────────────────────────
_NU = 1e-3  # kinematic viscosity  [m²/s]
_DT = 0.02  # time step            [s]
_N_STEPS = 20  # number of steps
_SIGMA_0 = 0.05  # initial core radius  [m]
_ALPHA_Z = 1.0  # z-circulation strength


def _single_particle_cs_solver(tmp_path):
    """Return a solver with one z-circulation particle and CS enabled."""
    config = SolverConfig(
        time_step_size=_DT,
        processing_unit="CPU",
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="CS"),
        advection=AdvectionConfig(scheme="NONE"),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = Solver(config=config)
    volume = (4.0 / 3.0) * np.pi * _SIGMA_0**3
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, _ALPHA_Z]]),
        radius=np.array([_SIGMA_0]),
        volume=np.array([volume]),
        viscosity=np.array([_NU]),
    )
    return solver


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_cs_core_radius_grows_at_correct_rate(tmp_path):
    """
    After N Core Spreading steps the particle radius must satisfy:

        σ_final² = σ₀² + C_diff · nu · N · dt,   C_diff = 4  (Gaussian kernel)

    Physical basis
    --------------
    The Gaussian diffusion operator satisfies d(σ²)/dt = 4nu.  Replacing
    exp(−r²/σ²) with exp(−r²/(σ²+4nudt)) at each step reproduces the exact
    heat-equation solution for a Gaussian blob.

    This test fails when
    --------------------
    * The diffusivity constant is wrong (2nu instead of 4nu, or similar typo).
    * The viscosity stored in the particle is not propagated to the CS kernel.
    * The update is applied to σ (not σ²), breaking the quadratic growth law.
    """
    sigma_expected_sq = _SIGMA_0**2 + 4.0 * _NU * _N_STEPS * _DT
    sigma_expected = float(np.sqrt(sigma_expected_sq))

    solver = _single_particle_cs_solver(tmp_path)
    for _ in range(_N_STEPS):
        solver.update_state()

    sigma_actual = float(solver.particles_radii[0])
    rel_err = abs(sigma_actual - sigma_expected) / sigma_expected

    assert rel_err < 1e-4, (
        f"CS core radius growth rate is wrong.\n"
        f"  σ_expected : {sigma_expected:.6f} m  (σ₀²+4nu·N·dt = {sigma_expected_sq:.6e})\n"
        f"  σ_actual   : {sigma_actual:.6f} m\n"
        f"  rel. error : {rel_err:.2%}"
    )


def test_cs_vorticity_field_matches_grown_kernel(tmp_path):
    """
    After N Core Spreading steps the regularised z-vorticity at probe points
    must match the 3-D Gaussian kernel evaluated at the grown radius σ_final.

    Physical basis
    --------------
    CS grows σ but never modifies α.  Therefore the regularised vorticity
    produced by a single particle at the origin is EXACTLY:

        ωz(r) = α_z · ζ_{σ_final}(r)  =  α_z · (1/π^{3/2} σ_final³) · exp(−r²/σ_final²)

    Any discrepancy means either σ grew at the wrong rate (same root cause as
    the radius test) or the vorticity kernel is mis-normalised.

    This test fails when
    --------------------
    * The vorticity kernel normalisation constant 1/π^{3/2} has a typo.
    * σ inside compute_target_vorticities is read before the CS update.
    * CS changes α instead of (only) σ.
    """
    solver = _single_particle_cs_solver(tmp_path)
    for _ in range(_N_STEPS):
        solver.update_state()

    sigma_final = float(solver.particles_radii[0])

    # Probe at five radii spanning near- to far-field
    r_values = np.array([0.5, 1.0, 1.5, 2.0, 3.0]) * sigma_final
    probes = np.column_stack([r_values, np.zeros_like(r_values), np.zeros_like(r_values)])
    omega_numerical = solver.compute_target_vorticities(probes)[:, 2]

    # Analytical 3-D Gaussian ζ_{σ}(r) = (1/π^{3/2} σ³) exp(−r²/σ²)
    one_over_pi_15 = 1.0 / (np.pi**1.5)  # ≈ 0.17959
    omega_analytical = (
        _ALPHA_Z * one_over_pi_15 / sigma_final**3 * np.exp(-(r_values**2) / sigma_final**2)
    )

    l2_ref = float(np.sqrt(np.sum(omega_analytical**2)))
    l2_err = float(np.sqrt(np.sum((omega_numerical - omega_analytical) ** 2))) / l2_ref

    assert l2_err < 0.02, (
        f"CS vorticity field L₂ error = {l2_err:.1%} (threshold 2 %).\n"
        f"  σ_final  = {sigma_final:.4f} m\n"
        f"  r_probes = {r_values.tolist()}\n"
        f"  ω_analytical = {omega_analytical.tolist()}\n"
        f"  ω_numerical  = {omega_numerical.tolist()}"
    )
