"""
Conservation-law tests for the VPM solver.

These tests check fundamental physics invariants that hold independently of
resolution or time-step size.  A failure here means core physics is broken —
not merely inaccurate.

Tests
-----
test_transposed_stretching_conserves_total_circulation
    The transposed stretching operator dΓ/dt = (Γ·∇ᵀ)u is anti-symmetric
    across particle pairs, so the vector sum ΣΓ is conserved exactly.
    Failure → sign error or wrong stretching mode in the kernel.

test_cs_diffusion_conserves_total_circulation
    Core Spreading grows particle core radii but does NOT modify circulation
    strengths.  The discrete sum ΣΓz must therefore stay constant.
    Failure → CS incorrectly alters particle strengths (α).
"""

import numpy as np

from source.solvers.VPM import ParticleDistributor, Solver, SolverConfig
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig
from source.solvers.VPM.utils.flow_models import LambOseenVPM

# ── Shared physics parameters (Lamb-Oseen benchmark) ─────────────────────────
_NU = 1.887e-3  # kinematic viscosity  [m²/s]
_GAMMA = 1.0  # total circulation    [m²/s]
_RC = 0.125  # initial core radius   [m]
_T0 = _RC**2 / (4.0 * _NU)  # vortex age so σ(t₀) = _RC  [s]
_H = 0.04  # particle spacing       [m]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _minimal_config(tmp_path, *, stretching, viscous, advection, dt=0.01):
    """Return a SolverConfig that writes nothing useful, only logs to tmp_path."""
    return SolverConfig(
        time_step_size=dt,
        processing_unit="CPU",
        stretching=stretching,
        viscous=viscous,
        advection=advection,
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )


def _load_lamb_oseen(solver, bounds, h):
    """Populate *solver* with a Lamb-Oseen vortex and return the circulations."""
    positions, volumes, radii = ParticleDistributor.hexagonal_distribution(bounds, h)
    velocities, viscosities, circulations = LambOseenVPM(
        viscosity=_NU,
        avg_particle_radius=float(radii.mean()),
        positions=positions,
        volumes=volumes,
        vortex_strength=_GAMMA,
        vortex_time=_T0,
    )
    solver.add_vortex_particles(
        position=positions,
        velocity=velocities,
        circulation=circulations,
        radius=radii,
        volume=volumes,
        viscosity=viscosities,
    )
    return circulations


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_transposed_stretching_conserves_total_circulation(tmp_path):
    """
    Transposed stretching must conserve the vector sum ΣΓ to near floating-point
    precision across one time step with 100 randomly oriented vortex particles.

    Physical basis
    --------------
    The discretised transposed operator

        dαᵢ/dt = Σⱼ K(xᵢ−xⱼ, σⱼ) ⊗ αⱼ   (anti-symmetric in i↔j)

    satisfies d/dt(Σᵢ αᵢ) = 0 algebraically.  Direct stretching lacks this
    anti-symmetry and does NOT conserve ΣΓ.

    This test fails when
    --------------------
    * The kernel uses DIRECT instead of TRANSPOSED mode.
    * There is a sign error in the stretching kernel.
    * Wrong pairs of indices contribute to the rate term.
    """
    rng = np.random.default_rng(42)
    N = 100
    positions = rng.uniform(-1.0, 1.0, (N, 3))
    # Non-trivial 3-D circulations (not aligned with z) to exercise all components.
    circulations = rng.uniform(-0.05, 0.05, (N, 3))
    sigma = 0.15
    radii = np.full(N, sigma)
    volumes = (4.0 / 3.0) * np.pi * radii**3
    velocities = np.zeros((N, 3))
    viscosities = np.zeros(N)

    config = _minimal_config(
        tmp_path,
        stretching=StretchingConfig(mode="TRANSPOSED", scheme="EULER"),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver = Solver(config=config)
    solver.add_vortex_particles(
        position=positions,
        velocity=velocities,
        circulation=circulations,
        radius=radii,
        volume=volumes,
        viscosity=viscosities,
    )

    gamma_before = solver.particles_strengths.sum(axis=0)
    solver.update_state()
    gamma_after = solver.particles_strengths.sum(axis=0)

    # Relative error per component; skip near-zero components to avoid division by ~0.
    scale = np.maximum(np.abs(gamma_before), 1e-12)
    rel_err = np.abs(gamma_after - gamma_before) / scale
    assert np.all(rel_err < 1e-4), (
        "TRANSPOSED stretching broke ΣΓ conservation.\n"
        f"  ΣΓ before : {gamma_before}\n"
        f"  ΣΓ after  : {gamma_after}\n"
        f"  Relative Δ: {rel_err}"
    )


def test_cs_diffusion_conserves_total_circulation(tmp_path):
    """
    Twenty steps of Core Spreading must leave the total z-circulation ΣΓz
    unchanged to round-off error (< 1 ppm relative change).

    Physical basis
    --------------
    CS grows each particle's core radius as σ²(t+dt) = σ²(t) + 4nu·dt
    but leaves the circulation strength vector αᵢ untouched.  The field
    vorticity at any point does change (the blobs spread), yet the global
    integral ∫ω dV = ΣΓz is conserved.

    This test fails when
    --------------------
    * CS accidentally modifies αᵢ instead of (only) σᵢ.
    * A volume update introduces spurious circulation gains/losses.
    """
    bounds = [-3 * _RC, 3 * _RC, -3 * _RC, 3 * _RC, -_H * 0.5, _H * 0.5]
    config = _minimal_config(
        tmp_path,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="CS"),
        advection=AdvectionConfig(scheme="NONE"),
        dt=0.02,
    )
    solver = Solver(config=config)
    _load_lamb_oseen(solver, bounds, _H)

    gamma_z_initial = solver.particles_strengths[:, 2].sum()

    for _ in range(20):
        solver.update_state()

    gamma_z_final = solver.particles_strengths[:, 2].sum()
    rel_err = abs(gamma_z_final - gamma_z_initial) / abs(gamma_z_initial)

    assert rel_err < 1e-6, (
        f"Core Spreading altered total circulation by {rel_err:.2e}.\n"
        f"  ΣΓz initial : {gamma_z_initial:.8f}\n"
        f"  ΣΓz final   : {gamma_z_final:.8f}"
    )
