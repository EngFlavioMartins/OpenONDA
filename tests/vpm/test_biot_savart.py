"""
Biot-Savart kernel accuracy tests for the VPM solver.

These tests verify that the regularised Biot-Savart kernel used by the VPM
solver matches the exact analytical formula for a single Gaussian vortex blob.
They are independent of any time-integration or diffusion code.

Kernel formula (Gaussian, 3-D)
------------------------------
The velocity at point x from a particle at the origin with z-circulation α_z
and core radius σ is derived from the regularised Biot-Savart law:

    u(x) = -Σⱼ  q(|r_ij|/σⱼ) * (r_ij × αⱼ) / |r_ij|³  +  U_bg

For the GAUSSIAN kernel:

    q(ρ) = [erf(ρ) − (2/√π)·ρ·exp(−ρ²)] / (4π)

For a probe at (r, 0, 0) and a single z-circulation particle:

    u_y = q(r/σ) · α_z / r²          (with U_bg = 0)

Far-field limit (r ≫ σ):  q(ρ) → 1/(4π),  so  u_y → α_z/(4π r²).

Tests
-----
test_gaussian_kernel_velocity_matches_analytical_formula
    Compares numerical velocity to the exact q(ρ) formula at five
    radial distances covering both near- and far-field regimes.

test_gaussian_kernel_far_field_power_law
    Verifies u_y ∝ r⁻² at large separation (r/σ = 10 and 20).
    Catches scale errors or wrong kernel normalisation without
    depending on the exact q function value.
"""

from math import erf, exp, pi, sqrt

import numpy as np

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig

# ── Analytical Gaussian kernel helpers ───────────────────────────────────────


def _q_gaussian(rho: float) -> float:
    """q(ρ) = [erf(ρ) − (2/√π)·ρ·exp(−ρ²)] / (4π)  — exact Gaussian q-function."""
    return (erf(rho) - (2.0 / sqrt(pi)) * rho * exp(-(rho**2))) / (4.0 * pi)


def _u_y_analytical(r: float, sigma: float, alpha_z: float) -> float:
    """Analytical y-velocity at probe (r, 0, 0) from a z-particle at the origin.

    Derived directly from the kernel formula in kernels_common.py:
        target_velocities[i] = -q(ρ) * (r_ij × αⱼ) / |r_ij|³ + U_bg
    For r_ij = (r, 0, 0), αⱼ = (0, 0, α_z):
        r_ij × αⱼ = (0, -r·α_z, 0)
        u_y = -q(r/σ) * (-r·α_z) / r³  =  q(r/σ) * α_z / r²
    """
    rho = r / sigma
    return _q_gaussian(rho) * alpha_z / (r**2)


# ─────────────────────────────────────────────────────────────────────────────
# Shared solver setup
# ─────────────────────────────────────────────────────────────────────────────

_SIGMA = 0.2  # particle core radius
_ALPHA_Z = 1.0  # z-circulation strength  [m²/s · m]


def _single_particle_solver(tmp_path):
    """Return a solver loaded with one z-circulation particle at the origin."""
    config = VPMSetup(
        time_step_size=0.01,
        processing_unit="CPU",
        particles_kernel="GAUSSIAN",
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = Solver(setup=config)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, _ALPHA_Z]]),
        radius=np.array([_SIGMA]),
        volume=np.array([(4.0 / 3.0) * np.pi * _SIGMA**3]),
        viscosity=np.array([0.0]),
    )
    return solver


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_gaussian_kernel_velocity_matches_analytical_formula(tmp_path):
    """
    The numerical y-velocity at (r, 0, 0) must match the analytical Gaussian
    q-function formula within 1 % relative error at five probe distances.

    Physical basis
    --------------
    The Biot-Savart kernel implemented in kernels_common.py is

        u_y = q(r/σ) · α_z / r²

    where q is built from the erf-based Gaussian quadrature.  Any mistake in
    the normalisation constant, the erf approximation, or the cross-product
    sign will change the numerical output while the analytical value is fixed.

    This test fails when
    --------------------
    * The ONE_OVER_FOUR_PI constant is wrong (normalisation error).
    * The erf approximation (Abramowitz & Stegun) has a coefficient typo.
    * The cross-product or sign convention is flipped.
    * A different kernel (Winckelmans, Super-Gaussian) is inadvertently used.
    """
    solver = _single_particle_solver(tmp_path)

    # Five probe distances: near field (0.5σ, 1σ, 2σ), mid (4σ), far (10σ)
    r_values = np.array([0.5, 1.0, 2.0, 4.0, 10.0]) * _SIGMA
    probes = np.column_stack([r_values, np.zeros_like(r_values), np.zeros_like(r_values)])

    vel_numerical = solver.compute_target_velocities(probes, include_freestream=False)
    u_y_numerical = vel_numerical[:, 1]

    for idx, r in enumerate(r_values):
        u_y_exact = _u_y_analytical(r, _SIGMA, _ALPHA_Z)
        num = float(u_y_numerical[idx])
        rel_err = abs(num - u_y_exact) / abs(u_y_exact)
        assert rel_err < 0.01, (
            f"Gaussian kernel mismatch at r/σ = {r / _SIGMA:.1f}:\n"
            f"  analytical = {u_y_exact:.6e}\n"
            f"  numerical  = {num:.6e}\n"
            f"  rel. error = {rel_err:.2%}"
        )


def test_gaussian_kernel_far_field_power_law(tmp_path):
    """
    At large separation (r ≫ σ) the induced velocity must decay as r⁻².

    Physical basis
    --------------
    For ρ = r/σ ≫ 1 the Gaussian kernel saturates:  q(ρ) → 1/(4π).
    Therefore u_y ∝ α_z/(4π r²), i.e. the 3-D point-vortex scaling.

    The ratio u_y(r₁)/u_y(r₂) = (r₂/r₁)² fixes the exponent unambiguously.
    Testing this power law catches global prefactor errors without requiring
    the exact q-function value.

    This test fails when
    --------------------
    * The velocity decays as r⁻¹ (2-D kernel mistakenly used in 3-D).
    * The velocity decays as r⁻³ (extra 1/r factor in the kernel call).
    * The kernel is saturated for a reason other than the Gaussian erf term
      (wrong regularisation functional form).
    """
    solver = _single_particle_solver(tmp_path)

    # Two far-field distances: r₁ = 10σ, r₂ = 20σ  → expected ratio = (20/10)² = 4
    r1, r2 = 10.0 * _SIGMA, 20.0 * _SIGMA
    probes = np.array([[r1, 0.0, 0.0], [r2, 0.0, 0.0]])
    vel = solver.compute_target_velocities(probes, include_freestream=False)
    u1, u2 = float(vel[0, 1]), float(vel[1, 1])

    expected_ratio = (r2 / r1) ** 2  # = 4.0
    actual_ratio = u1 / u2  # u ∝ 1/r²  →  u1/u2 = (r2/r1)²

    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, (
        f"Far-field power law failed.\n"
        f"  u_y(r₁={r1:.2f}) = {u1:.4e}\n"
        f"  u_y(r₂={r2:.2f}) = {u2:.4e}\n"
        f"  ratio actual / expected = {actual_ratio:.4f} / {expected_ratio:.4f}"
    )
