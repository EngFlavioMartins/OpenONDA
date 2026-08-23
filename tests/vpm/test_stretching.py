"""
Vortex stretching tests for the VPM solver.

Tests
-----
test_disabled_stretching_leaves_circulation_invariant
    When stretching is disabled (StretchingConfig.disabled()), circulation
    vectors must remain exactly unchanged after any number of steps.
    Failure → the 'enabled=False' guard is missing or bypassed.

test_direct_stretching_changes_total_circulation
    The direct stretching operator dΓ/dt = (Γ·∇)u is NOT anti-symmetric
    across particle pairs, so the vector sum ΣΓ is NOT conserved.  For a
    generic set of randomly oriented vortex particles the relative change
    in |ΣΓ| after one step must exceed a small threshold.
    Failure → the DIRECT mode code path is not being reached, or the
    kernel has accidentally been made anti-symmetric.
"""

import numpy as np

from source.solvers.vpm import VPMSetup, VPMSolver
from source.solvers.vpm.config.types import AdvectionConfig, StretchingConfig, ViscousConfig

# ── Shared parameters ─────────────────────────────────────────────────────────
_RNG_SEED = 42
_N_PARTICLES = 20
_TIME_STEP_SIZE = 0.05  # time step  [s]
_SIGMA = 0.1  # particle core radius [m]
_GAMMA_SCALE = 1.0  # RMS circulation magnitude per component [m²/s]


def _stretching_solver(tmp_path, *, stretching_config, position, circulations):
    """Return a solver with only stretching active (advection=NONE, viscous=NONE)."""
    config = VPMSetup(
        time_step_size=_TIME_STEP_SIZE,
        compute_device="CPU",
        advection=AdvectionConfig(scheme="NONE"),
        stretching=stretching_config,
        viscous=ViscousConfig.inviscid(),
        checkpoint_interval_steps=0,
        logging_interval_steps=0,
        checkpoint_directory=str(tmp_path),
    )
    solver = VPMSolver(setup=config)
    n = len(position)
    particle_volume = (4.0 / 3.0) * np.pi * _SIGMA**3
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((n, 3)),
        vortex_strength=circulations,
        core_radius=np.full(n, _SIGMA),
        particle_volume=np.full(n, particle_volume),
        kinematic_viscosity=np.full(n, 1e-5),
    )
    return solver


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_disabled_stretching_leaves_circulation_invariant(tmp_path):
    """
    When stretching is disabled all circulation components must be exactly
    unchanged after multiple time steps.

    Physical basis
    --------------
    Setting StretchingConfig.disabled() sets stretching_enabled=False in the
    solver, causing the vortex_stretching call to be skipped entirely.  No
    other physics step modifies circulation when viscous diffusion is also
    disabled (NONE).

    This test fails when
    --------------------
    * The enabled=False guard is absent or bypassed in _update_strength.
    * A physics step (velocity gradient computation, LES, adaptation) has a
      side-effect that modifies particle circulations.
    """
    rng = np.random.default_rng(_RNG_SEED)
    position = rng.uniform(-1.0, 1.0, (_N_PARTICLES, 3))
    circulations = rng.normal(0.0, _GAMMA_SCALE, (_N_PARTICLES, 3))

    solver = _stretching_solver(
        tmp_path,
        stretching_config=StretchingConfig.disabled(),
        position=position,
        circulations=circulations,
    )

    circ_before = solver.particle_vortex_strength.copy()
    for _ in range(5):
        solver.advance()

    np.testing.assert_array_equal(
        solver.particle_vortex_strength,
        circ_before,
        err_msg="Disabled stretching must not modify any circulation component.",
    )


def test_direct_stretching_changes_total_circulation(tmp_path):
    """
    Direct stretching dΓ/dt = (Γ·∇)u is not anti-symmetric across particle
    pairs and therefore does NOT conserve the vector sum ΣΓ.

    Physical basis
    --------------
    The transposed operator satisfies ΣᵢΣⱼ K(xᵢ−xⱼ) ⊗ αⱼ = 0 by swapping i↔j.
    The direct operator lacks this anti-symmetry: the pair contributions
    (αⱼ·∇ᵢⱼ)uᵢⱼ and (αᵢ·∇ⱼᵢ)uⱼᵢ do not cancel.  For a generic random
    configuration of vortex particles the net change |ΔΣΓ| is therefore
    non-negligible after even a single Euler step.

    This test fails when
    --------------------
    * The DIRECT branch in vortex_stretching defaults back to TRANSPOSED.
    * The mode integer mapping (0=DIRECT, 1=TRANSPOSED) is swapped.
    * All particles happen to be co-planar in a configuration where direct
      accidentally conserves ΣΓ (extremely unlikely with random seed).
    """
    rng = np.random.default_rng(_RNG_SEED + 1)
    position = rng.uniform(-0.5, 0.5, (_N_PARTICLES, 3))
    circulations = rng.normal(0.0, _GAMMA_SCALE, (_N_PARTICLES, 3))

    solver = _stretching_solver(
        tmp_path,
        stretching_config=StretchingConfig.direct(scheme="EULER"),
        position=position,
        circulations=circulations,
    )

    sum_gamma_before = solver.particle_vortex_strength.sum(axis=0)
    solver.advance()
    sum_gamma_after = solver.particle_vortex_strength.sum(axis=0)

    delta = np.linalg.norm(sum_gamma_after - sum_gamma_before)
    total_gamma = np.linalg.norm(sum_gamma_before)
    relative_change = delta / (total_gamma + 1e-12)

    assert relative_change > 1e-4, (
        f"Direct stretching should break ΣΓ conservation but relative change "
        f"was only {relative_change:.2e}.  "
        f"Check that the DIRECT kernel mode (mode_int=0) is active."
    )
