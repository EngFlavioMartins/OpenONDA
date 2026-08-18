"""
Advection tests for the VPM solver.

Tests
-----
test_none_scheme_freezes_particle_positions
    With advection scheme set to 'NONE', particle positions must remain exactly
    unchanged across every time step, regardless of the assigned velocity.
    Failure → the NONE guard is missing in _update_positions.

test_uniform_background_all_schemes_exact_translation
    A zero-circulation particle in a uniform background flow U_bg has no
    self-induced Biot-Savart velocity.  Its advection velocity is therefore
    constant and equal to U_bg at every point and every RK stage, making
    all four integration schemes exactly equivalent:

        x_final = x_0 + N · dt · U_bg

    Failure → a scheme applies wrong Butcher-tableau weights, forgets to
    include background velocity at an RK intermediate stage, or leaves
    temporary position fields uninitialised between steps.

test_velocity_method_is_consistent_across_all_rk_stages
    Every velocity evaluation inside an RK step must use the SAME configured
    method (direct OR treecode) — never a mix.  Instruments the two branches of
    PhysicsBase.velocity_self and asserts that a single RK4 step performs all
    four stage evaluations through one branch only.
    Failure → the old bug where the treecode was used at k1 but direct (or
    nothing) at the remaining stages, breaking accuracy/consistency.

test_velocity_method_is_consistent_at_arbitrary_targets
    Target-point queries must honor the configured direct/treecode method too.
    Failure -> coupled boundary evaluation can silently fall back to one very
    large direct GPU dispatch and trigger an operating-system watchdog reset.
"""

import numpy as np
import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

# ── Shared parameters ─────────────────────────────────────────────────────────
_DT = 0.1  # time step            [s]
_N_STEPS = 10  # number of steps
_U_BG = [2.0, -1.0, 0.5]  # background velocity   [m/s]
_X0 = np.array([[0.3, -0.7, 1.2]])  # initial position      [m]
_SIGMA = 0.05  # particle core radius  [m]


def _advection_solver(tmp_path, *, scheme: str, background=None):
    """
    Return a solver with a single, zero-circulation particle.

    Stretching and viscous diffusion are both disabled so that the only
    physics active is the chosen advection scheme.  The zero-circulation
    particle has no self-induced velocity; its motion is determined solely
    by the background (free-stream) velocity.
    """
    config = VPMSetup(
        time_step_size=_DT,
        processing_unit="CPU",
        advection=AdvectionConfig(scheme=scheme),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        freestream_velocity=background if background is not None else [0.0, 0.0, 0.0],
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = Solver(setup=config)
    volume = (4.0 / 3.0) * np.pi * _SIGMA**3
    solver.add_vortex_particles(
        position=_X0.copy(),
        velocity=np.zeros((1, 3)),
        circulation=np.zeros((1, 3)),  # zero → no Biot-Savart self-induction
        radius=np.array([_SIGMA]),
        volume=np.array([volume]),
        viscosity=np.array([1e-5]),
    )
    return solver


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_none_scheme_freezes_particle_positions(tmp_path):
    """
    Advection scheme 'NONE' must leave particle positions exactly unchanged.

    Physical basis
    --------------
    The NONE branch in _update_positions returns immediately without calling
    the physics engine.  This is used for viscous-only benchmarks (e.g.,
    CS diffusion on a frozen particle cloud).

    This test fails when
    --------------------
    * The 'NONE' check is missing and the solver falls through to Euler.
    * Another physics step (velocity gradient, LES, adaptation) moves particles
      as an unintended side-effect.
    """
    solver = _advection_solver(tmp_path, scheme="NONE", background=_U_BG)
    pos_before = solver.particles_positions.copy()

    for _ in range(_N_STEPS):
        solver.update_state()

    np.testing.assert_array_equal(
        solver.particles_positions,
        pos_before,
        err_msg="NONE advection scheme must not modify particle positions.",
    )


@pytest.mark.parametrize("scheme", ["EULER", "RK2", "RK3", "RK4"])
def test_uniform_background_all_schemes_exact_translation(tmp_path, scheme):
    """
    For a zero-circulation particle in a uniform background flow all four
    integration schemes must produce the exact displacement x_0 + N·dt·U_bg.

    Physical basis
    --------------
    With zero circulation the Biot-Savart kernel returns zero induced velocity
    at every query point.  The total velocity therefore equals U_bg regardless
    of where the intermediate RK stages evaluate it.  All Butcher tableaux
    then integrate a constant vector field, which every consistent scheme
    integrates exactly in one step.

    This test fails when
    --------------------
    * A scheme applies wrong stage weights (e.g. 1/3 instead of 1/6 in RK4).
    * Background velocity is omitted at one or more RK intermediate stages.
    * Temporary position/velocity fields are not reset correctly between steps.
    """
    solver = _advection_solver(tmp_path, scheme=scheme, background=_U_BG)
    x0 = solver.particles_positions.copy()

    for _ in range(_N_STEPS):
        solver.update_state()

    x_expected = x0 + _N_STEPS * _DT * np.array(_U_BG)

    np.testing.assert_allclose(
        solver.particles_positions,
        x_expected,
        rtol=1e-5,
        err_msg=(
            f"{scheme} scheme gave incorrect position after {_N_STEPS} steps "
            f"in a uniform background flow."
        ),
    )


def _self_induced_solver(tmp_path, *, velocity_config):
    """Solver with a small cloud of finite-circulation particles (self-induced flow)."""
    config = VPMSetup(
        time_step_size=_DT,
        processing_unit="CPU",
        advection=AdvectionConfig(scheme="RK4"),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        velocity=velocity_config,
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = Solver(setup=config)
    rng = np.random.default_rng(0)
    n = 8
    volume = (4.0 / 3.0) * np.pi * _SIGMA**3
    solver.add_vortex_particles(
        position=rng.uniform(-0.5, 0.5, size=(n, 3)),
        velocity=np.zeros((n, 3)),
        circulation=rng.uniform(-1.0, 1.0, size=(n, 3)),  # finite → real self-induction
        radius=np.full(n, _SIGMA),
        volume=np.full(n, volume),
        viscosity=np.full(n, 1e-5),
    )
    return solver


@pytest.mark.parametrize(
    "velocity_config, expected_branch",
    [
        (VelocityConfig.direct(), "DIRECT"),
        (VelocityConfig.treecode(theta=0.5), "TREECODE"),
    ],
)
def test_velocity_method_is_consistent_across_all_rk_stages(
    tmp_path, velocity_config, expected_branch
):
    """A single RK4 step must evaluate velocity through ONE method at every stage.

    PhysicsBase.velocity_self is the single source of truth for direct-vs-treecode.
    Its two branches are instrumented with counters; one RK4 step makes four stage
    evaluations (k1..k4), so the configured branch must fire exactly four times and
    the other branch zero times.  This guards against the regression where the
    treecode was applied only at k1 while later stages silently fell back to direct.
    """
    solver = _self_induced_solver(tmp_path, velocity_config=velocity_config)
    physics = solver.physics

    counts = {"DIRECT": 0, "TREECODE": 0}

    # Wrap the direct branch (compute_velocities_kernel) and the treecode branch
    # (_copy_vec3, invoked once per treecode evaluation) with counters.
    direct_fn = physics.compute_velocities_kernel
    treecode_fn = physics._copy_vec3

    def counting_direct(*args, **kwargs):
        counts["DIRECT"] += 1
        return direct_fn(*args, **kwargs)

    def counting_treecode(*args, **kwargs):
        counts["TREECODE"] += 1
        return treecode_fn(*args, **kwargs)

    physics.compute_velocities_kernel = counting_direct
    physics._copy_vec3 = counting_treecode
    try:
        physics.update_positions(solver.particles, _DT, scheme="RK4")
    finally:
        physics.compute_velocities_kernel = direct_fn
        physics._copy_vec3 = treecode_fn

    other_branch = "TREECODE" if expected_branch == "DIRECT" else "DIRECT"
    assert counts[expected_branch] == 4, (
        f"{expected_branch} method should be used at all 4 RK4 stages, "
        f"got {counts[expected_branch]}."
    )
    assert counts[other_branch] == 0, (
        f"RK4 mixed velocity methods: {other_branch} branch fired "
        f"{counts[other_branch]} times when {expected_branch} was configured."
    )


@pytest.mark.parametrize(
    "velocity_config, expected_branch",
    [
        (VelocityConfig.direct(), "DIRECT"),
        (VelocityConfig.treecode(theta=0.5), "TREECODE"),
    ],
)
def test_velocity_method_is_consistent_at_arbitrary_targets(
    tmp_path, velocity_config, expected_branch
):
    """Target queries must use the velocity method selected in ``VPMSetup``."""
    solver = _self_induced_solver(tmp_path, velocity_config=velocity_config)
    physics = solver.physics
    targets = np.array([[0.7, 0.1, -0.2], [-0.4, 0.8, 0.3]], dtype=np.float32)
    counts = {"DIRECT": 0, "TREECODE": 0}

    direct_fn = physics.compute_target_velocity_kernel
    treecode_fn = physics.compute_target_velocities_hierarchical

    def counting_direct(*args, **kwargs):
        counts["DIRECT"] += 1
        return direct_fn(*args, **kwargs)

    def counting_treecode(*args, **kwargs):
        counts["TREECODE"] += 1
        return treecode_fn(*args, **kwargs)

    physics.compute_target_velocity_kernel = counting_direct
    physics.compute_target_velocities_hierarchical = counting_treecode
    try:
        velocity = solver.compute_target_velocities(targets, include_freestream=False)
    finally:
        physics.compute_target_velocity_kernel = direct_fn
        physics.compute_target_velocities_hierarchical = treecode_fn

    other_branch = "TREECODE" if expected_branch == "DIRECT" else "DIRECT"
    assert counts[expected_branch] == 1
    assert counts[other_branch] == 0
    assert velocity.shape == targets.shape
    assert np.isfinite(velocity).all()
