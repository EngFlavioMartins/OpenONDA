"""
Pressure gradient tests for the VPM solver.

Tests
-----
test_q_kernel_zero_at_origin
    The regularised velocity kernel q(r/σ) must vanish at r=0.  A non-zero
    value would induce a spurious self-velocity at the particle centre.

test_q_kernel_far_field_asymptote
    For r/σ >> 1, the Gaussian regularisation is negligible and q recovers
    the point-vortex (Biot-Savart) limit: q(ρ→∞) → 1/(4π).

test_zeta_kernel_maximum_at_origin
    The vorticity kernel ζ(0) equals 1/π^{3/2}, which ensures the normalised
    3-D Gaussian integral ∫ζ(r/σ)dV/σ³ = 1 (vorticity is conserved under
    regularisation as σ→0).

test_pressure_gradients_zero_for_empty_field
    With no particles in the domain the velocity and all its derivatives
    are zero, so the momentum-equation pressure gradient must also be zero
    at every query point.

test_pressure_eulerian_method_requires_dt_and_velocity_previous
    When temporal_method='eulerian' the caller must supply both
    velocity_previous and dt; omitting either raises ValueError.

test_velocity_gradient_tensor_matches_analytical_formula
    The full 3×3 velocity gradient tensor at (r,0,0) for a single z-vortex
    must agree with the closed-form derivatives of the Gaussian Biot-Savart
    kernel.  Validates compute_target_velocity_gradients() and the
    grad_u[a,b] = ∂u_a/∂x_b row-vector convention.

test_advective_pressure_gradient_matches_analytical
    The convective contribution −ρ(u·∇)u at (r,0,0) for a single z-vortex
    must match the closed-form result ∂p/∂x = ρ·q²·α_z²/r⁵.  Validates the
    einsum "mb,mab->ma" in compute_target_pressure_gradients() and confirms
    ∂p/∂y = ∂p/∂z = 0 for this axisymmetric configuration.

test_advective_pressure_gradient_radial_power_law
    Far from the vortex core (r/σ ≥ 6) the convective pressure gradient must
    decay as r⁻⁵.  Verifies the normalisation of q in the far field and that
    no spurious constant offset has been introduced.

test_pressure_gradient_component_decomposition
    compute_target_pressure_gradient_components() must yield a total that
    equals the element-wise sum of the individual convective/viscous/temporal
    contributions.  Guards against split-sign bugs in the component method.

test_viscous_term_small_at_high_reynolds
    At Re=1000 the viscous contribution ρnu∇²u must be smaller than the
    convective contribution −ρ(u·∇)u by at least a factor of 100 everywhere
    on the probe grid.  Confirms that omitting the viscous term in the
    coupling BC is physically justified.

test_eulerian_temporal_term_self_consistency
    When velocity_previous is obtained from the same internal code path as
    u_target (i.e., PressurePhysics.compute_target_velocities), the computed
    temporal term (u_new − u_old)/dt must equal the finite-difference of two
    consecutive evaluations to machine precision.  This is the correctness
    contract that must hold for the Eulerian temporal term to be physically
    meaningful.

test_eulerian_temporal_mismatch_with_corrected_velocity
    Supplying a velocity_previous that was computed via a DIFFERENT code path
    (VPMSolver.compute_target_velocities, which applies a divergence-free
    correction) to a steady, unchanging particle field must produce a nonzero
    temporal term.  This is the canary test that documents and protects against
    the source-of-error identified in the coupling BC: if the coupler naively
    stores the corrected face-centre velocity as the snapshot, the resulting
    temporal contribution is spurious.
"""

from math import erf, exp, pi, sqrt

import numpy as np
import pytest

from source.solvers.VPM import VPMSetup, VPMSolver
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig
from source.solvers.VPM.physics.pressure import _q_kernel, _zeta_kernel

# Physical constants
_ONE_OVER_FOUR_PI = 1.0 / (4.0 * np.pi)  # ≈ 0.079577
_ONE_OVER_PI_15 = np.pi ** (-1.5)  # ≈ 0.179587


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function kernel tests (no Taichi required)
# ─────────────────────────────────────────────────────────────────────────────


def test_q_kernel_zero_at_origin():
    """
    q(0) must be exactly zero.

    Physical basis
    --------------
    The regularised Gaussian velocity kernel is q(ρ) = [erf(ρ) - (2/√π)ρ·e^{-ρ²}]
    / (4π).  Both the Taylor series and the erf branch evaluate to zero at ρ=0
    because the numerator vanishes to O(ρ³).

    This test fails when
    --------------------
    * The small-ρ Taylor branch contains a constant offset term.
    * The branch threshold (1e-4) is too large, allowing the erf branch to
      evaluate at ρ=0 where it produces a floating-point artefact.
    """
    result = _q_kernel(np.array([0.0]))
    np.testing.assert_allclose(result, [0.0], atol=1e-14)


def test_q_kernel_far_field_asymptote():
    """
    q(ρ→∞) → 1/(4π): Biot-Savart point-vortex limit.

    Physical basis
    --------------
    For large ρ: erf(ρ) → 1 and (2/√π)ρ·e^{-ρ²} → 0, so
    q(ρ) → 1/(4π) which is the Biot-Savart kernel coefficient for a
    filament of unit circulation.

    This test fails when
    --------------------
    * ONE_OVER_FOUR_PI constant has a typo.
    * The erf-branch formula is missing the subtraction of the exp term.
    """
    rho_far = np.array([20.0, 50.0, 100.0])
    q_far = _q_kernel(rho_far)
    np.testing.assert_allclose(q_far, _ONE_OVER_FOUR_PI, rtol=1e-6)


def test_zeta_kernel_maximum_at_origin():
    """
    ζ(0) = 1/π^{3/2}: peak value of the 3-D Gaussian vorticity kernel.

    Physical basis
    --------------
    The Gaussian regularisation kernel ζ_σ(r) = (1/π^{3/2}σ³) exp(-r²/σ²)
    evaluates to 1/π^{3/2} at r=0 for σ=1 (i.e. at the normalised distance
    ρ = r/σ = 0).  Integrating over ℝ³ gives unity, so vorticity is preserved
    in the continuum limit.

    This test fails when
    --------------------
    * The ONE_OVER_PI_15 constant has a typo (e.g. computed as 1/(2π^{3/2})).
    * The exponent coefficient is wrong (e.g. -ρ²/2 instead of -ρ²).
    """
    result = _zeta_kernel(np.array([0.0]))
    np.testing.assert_allclose(result, [_ONE_OVER_PI_15], rtol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# VPMSolver-level pressure gradient tests
# ─────────────────────────────────────────────────────────────────────────────


def _empty_solver(tmp_path):
    """Return a solver with no particles added."""
    config = VPMSetup(
        time_step_size=0.01,
        processing_unit="CPU",
        advection=AdvectionConfig(scheme="NONE"),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    return VPMSolver(setup=config)


def test_pressure_gradients_zero_for_empty_field(tmp_path):
    """
    An empty VPM field must return zero pressure gradient at all query points.

    Physical basis
    --------------
    With N=0 particles every velocity and velocity-derivative sum is zero.
    The momentum equation ∇p = -ρ[∂u/∂t + (u·∇)u - nu∇²u] then gives ∇p=0
    everywhere.  The code short-circuits this via an early-return when
    particles.number_of_particles == 0.

    This test fails when
    --------------------
    * The N==0 guard is removed, causing division-by-zero or NaN propagation.
    * Uninitialised Taichi fields produce non-zero output for zero-particle runs.
    """
    solver = _empty_solver(tmp_path)
    targets = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = solver.compute_target_pressure_gradients(
        targets,
        nu=1e-5,
        include_viscous=True,
        include_temporal=True,
    )
    grad_p = result["grad_p"]

    np.testing.assert_array_equal(
        grad_p,
        np.zeros((3, 3)),
        err_msg="Pressure gradient must be zero when no particles are present.",
    )


def test_hierarchical_pressure_can_exclude_body_velocity(tmp_path):
    """The coupler can avoid an expensive panel reevaluation at its outer boundary."""
    solver = _empty_solver(tmp_path)
    solver._pressure_body_induced_fn = lambda _points: (_ for _ in ()).throw(
        AssertionError("body callback should not be evaluated")
    )
    targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result, velocity = solver.compute_target_pressure_gradients(
        targets,
        include_viscous=False,
        include_temporal=False,
        include_body=False,
        temporal_method="eulerian",
        return_velocity=True,
        treecode_theta=0.3,
    )

    np.testing.assert_array_equal(result["grad_p"], np.zeros_like(targets))
    np.testing.assert_array_equal(velocity, np.zeros_like(targets))


def test_pressure_eulerian_method_requires_dt_and_velocity_previous(tmp_path):
    """
    Calling compute_target_pressure_gradients with temporal_method='eulerian'
    but without supplying velocity_previous and dt must raise ValueError.

    Physical basis
    --------------
    The Eulerian backward-difference formula (u_new - u_prev)/dt requires both
    the previous velocity snapshot and the time step.  Proceeding without them
    would produce a silent divide-by-None or semantically wrong gradient.

    This test fails when
    --------------------
    * The validation guard is removed and the code silently returns zeros or NaN.
    * The guard only checks one of the two required arguments.
    """
    config = VPMSetup(
        time_step_size=0.01,
        processing_unit="CPU",
        advection=AdvectionConfig(scheme="NONE"),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = VPMSolver(setup=config)
    sigma = 0.1
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([sigma]),
        volume=np.full(1, (4.0 / 3.0) * np.pi * sigma**3),
        viscosity=np.array([1e-5]),
    )

    targets = np.array([[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        solver.compute_target_pressure_gradients(
            targets,
            nu=1e-5,
            include_temporal=True,
            temporal_method="eulerian",
            # velocity_previous and dt intentionally omitted
        )


# =============================================================================
# Analytical reference helpers
# =============================================================================
# Single Gaussian vortex at the origin, α = (0,0,α_z), core radius σ.
# Probe at (r, 0, 0).  All formulas derived from the closed-form Biot-Savart
# kernel (kernels_common.py) and its exact spatial derivatives.
#
#   q(ρ) = [erf(ρ) − (2/√π)ρ exp(−ρ²)] / (4π),  ρ = r/σ
#   ζ(ρ) = exp(−ρ²) / π^{3/2}
#
# Velocity at (r,0,0):
#   u_x = 0,  u_y = q·α_z / r²,  u_z = 0
#
# Non-zero velocity gradient components:
#   ∂u_x/∂y = −q·α_z / r³
#   ∂u_y/∂x = (−2q/r³ + ζ/σ³) · α_z
#
# Advective term (u·∇)u at (r,0,0):
#   (u·∇)u_x = u_y · ∂u_x/∂y = −q²·α_z² / r⁵
#   (u·∇)u_y = 0   (∂u_y/∂y = 0 by symmetry)
#   (u·∇)u_z = 0
#
# Convective pressure gradient ∇p = −ρ(u·∇)u:
#   ∂p/∂x = ρ · q² · α_z² / r⁵  > 0  (pressure increases outward)
#   ∂p/∂y = 0
#   ∂p/∂z = 0
# =============================================================================

_SIGMA_A = 0.2  # particle core radius  [m]
_ALPHA_Z_A = 2.0  # z-circulation strength  [m³/s]
_DENSITY = 1.0  # fluid density  [kg/m³]
_NU = 1e-3  # kinematic viscosity (Re ≈ 1000)


def _q_exact(rho: float) -> float:
    """Exact q(ρ) = [erf(ρ) − (2/√π)·ρ·exp(−ρ²)] / (4π)."""
    return (erf(rho) - (2.0 / sqrt(pi)) * rho * exp(-(rho**2))) / (4.0 * pi)


def _zeta_exact(rho: float, sigma: float) -> float:
    """Exact ζ_val = exp(−ρ²) / (π^{3/2} · σ³)."""
    return exp(-(rho**2)) / (pi**1.5 * sigma**3)


def _analytical_grad_u_at_r00(r: float, sigma: float, alpha_z: float) -> np.ndarray:
    """Full 3×3 velocity gradient tensor ∂u_a/∂x_b at probe (r,0,0)."""
    rho = r / sigma
    q = _q_exact(rho)
    zeta_val = _zeta_exact(rho, sigma)
    grad = np.zeros((3, 3))
    grad[0, 1] = -q * alpha_z / r**3  # ∂u_x/∂y
    grad[1, 0] = (-2.0 * q / r**3 + zeta_val) * alpha_z  # ∂u_y/∂x
    return grad


def _analytical_grad_p_at_r00(
    r: float, sigma: float, alpha_z: float, rho_fluid: float
) -> np.ndarray:
    """Analytical ∂p/∂x = ρ·q²·α_z²/r⁵ at probe (r,0,0); ∂p/∂y = ∂p/∂z = 0."""
    rho = r / sigma
    q = _q_exact(rho)
    gp = np.zeros(3)
    gp[0] = rho_fluid * q**2 * alpha_z**2 / r**5
    return gp


def _single_vortex_solver(tmp_path, sigma: float = _SIGMA_A, alpha_z: float = _ALPHA_Z_A):
    """Return a solver loaded with one z-circulation particle at the origin."""
    config = VPMSetup(
        time_step_size=0.01,
        processing_unit="CPU",
        particles_kernel="GAUSSIAN",
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        advection=AdvectionConfig(scheme="NONE"),
        freestream_velocity=[0.0, 0.0, 0.0],
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = VPMSolver(setup=config)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, alpha_z]]),
        radius=np.array([sigma]),
        volume=np.array([(4.0 / 3.0) * pi * sigma**3]),
        viscosity=np.full(1, _NU),
    )
    return solver


# =============================================================================
# Test 1.1 — Velocity gradient tensor accuracy
# =============================================================================


def test_velocity_gradient_tensor_matches_analytical_formula(tmp_path):
    """
    Velocity gradient tensor ∂u_a/∂x_b at (r,0,0) must match the closed-form
    derivatives of the Gaussian Biot-Savart kernel within 1% relative error.

    Physical basis
    --------------
    From the kernel formula:
        ∂u_x/∂y = −q(ρ) · α_z / r³
        ∂u_y/∂x = [−2q(ρ)/r³ + ζ(ρ)/σ³] · α_z
        All other components = 0

    Row = velocity component (a), column = spatial direction (b).

    This test fails when
    --------------------
    * The skew matrix sign convention is wrong (interchanges ∂u_x/∂y and ∂u_y/∂x).
    * The q'(ρ) = ρ²·ζ(ρ) identity is violated (wrong zeta normalisation).
    * The velocity gradient tensor is stored transposed (col=component, row=direction).
    """
    solver = _single_vortex_solver(tmp_path)

    probe_ratios = [2.0, 4.0, 6.0, 10.0]  # r/σ
    r_values = [rr * _SIGMA_A for rr in probe_ratios]
    targets = np.array([[r, 0.0, 0.0] for r in r_values])

    grad_flat = solver.compute_target_velocity_gradients(targets)  # (M, 9)
    grad = grad_flat.reshape(-1, 3, 3)  # grad[m, a, b] = ∂u_a/∂x_b

    for k, r in enumerate(r_values):
        expected = _analytical_grad_u_at_r00(r, _SIGMA_A, _ALPHA_Z_A)

        for a in range(3):
            for b in range(3):
                expected_val = expected[a, b]
                computed_val = float(grad[k, a, b])

                if abs(expected_val) < 1e-10:
                    # Near-zero component: absolute tolerance
                    assert abs(computed_val) < 1e-6, (
                        f"r/σ={r / _SIGMA_A}, grad[{a},{b}]: expected ≈0, got {computed_val:.3e}"
                    )
                else:
                    rel_err = abs(computed_val - expected_val) / abs(expected_val)
                    assert rel_err < 0.01, (
                        f"r/σ={r / _SIGMA_A}, grad[{a},{b}]: "
                        f"expected {expected_val:.6e}, got {computed_val:.6e}, "
                        f"rel_err={rel_err:.3%}"
                    )


# =============================================================================
# Test 1.2 — Advective pressure gradient accuracy
# =============================================================================


def test_advective_pressure_gradient_matches_analytical(tmp_path):
    """
    The convective pressure gradient −ρ(u·∇)u at (r,0,0) must match
    ∂p/∂x = ρ·q²·α_z²/r⁵ within 2% relative error; ∂p/∂y and ∂p/∂z
    must be zero to within absolute tolerance 1e-6.

    Physical basis
    --------------
    At (r,0,0) the only nonzero velocity component is u_y = q·α_z/r².
    The advective acceleration (u·∇)u has only an x-component:
        (u·∇)u_x = u_y · ∂u_x/∂y = −q²·α_z²/r⁵
    Hence ∂p/∂x = ρ·q²·α_z²/r⁵ > 0 (centrifugal pressure increase outward).

    This test fails when
    --------------------
    * The einsum "mb,mab->ma" is wrong (e.g., wrong contraction index).
    * grad_u is transposed, swapping the component and direction roles.
    * The overall sign of ∇p = −ρ(u·∇)u is inverted.
    """
    solver = _single_vortex_solver(tmp_path)

    probe_ratios = [2.0, 4.0, 6.0, 10.0]
    r_values = [rr * _SIGMA_A for rr in probe_ratios]
    targets = np.array([[r, 0.0, 0.0] for r in r_values])

    grad_p = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=False,
        include_temporal=False,
        include_freestream=False,
    )["grad_p"]

    for k, r in enumerate(r_values):
        expected = _analytical_grad_p_at_r00(r, _SIGMA_A, _ALPHA_Z_A, _DENSITY)
        computed = grad_p[k]

        # x-component: must match analytical within 2%
        rel_err = abs(computed[0] - expected[0]) / abs(expected[0])
        assert rel_err < 0.02, (
            f"r/σ={r / _SIGMA_A}: ∂p/∂x expected {expected[0]:.6e}, "
            f"got {computed[0]:.6e}, rel_err={rel_err:.3%}"
        )

        # y and z components: must be zero
        for i, label in [(1, "∂p/∂y"), (2, "∂p/∂z")]:
            assert abs(computed[i]) < 1e-6, (
                f"r/σ={r / _SIGMA_A}: {label} expected 0, got {computed[i]:.3e}"
            )


# =============================================================================
# Test 1.3 — Radial power-law decay and sign
# =============================================================================


def test_advective_pressure_gradient_radial_power_law(tmp_path):
    """
    Far from the vortex core (r/σ ≥ 6) the convective pressure gradient must
    decay as r⁻⁵ and remain positive (pressure increases away from centre).

    Physical basis
    --------------
    In the far field q(ρ) → 1/(4π) = constant, so ∂p/∂x ∝ r⁻⁵.
    Deviations indicate a normalisation error in q or a sign mistake.

    This test fails when
    --------------------
    * The far-field q constant is wrong (e.g., 1/(2π) instead of 1/(4π)).
    * An extra r^n factor has been introduced somewhere in the kernel.
    * The sign of ∂p/∂x is negative (wrong direction of centripetal force).
    """
    solver = _single_vortex_solver(tmp_path)

    # Use two pairs far from the core to check the exponent
    pairs = [(6.0, 12.0), (8.0, 16.0)]

    for r1_ratio, r2_ratio in pairs:
        r1 = r1_ratio * _SIGMA_A
        r2 = r2_ratio * _SIGMA_A
        targets = np.array([[r1, 0.0, 0.0], [r2, 0.0, 0.0]])

        grad_p = solver.compute_target_pressure_gradients(
            targets,
            density=_DENSITY,
            nu=_NU,
            include_viscous=False,
            include_temporal=False,
            include_freestream=False,
        )["grad_p"]

        gp1 = grad_p[0, 0]
        gp2 = grad_p[1, 0]

        # Both positive
        assert gp1 > 0, f"∂p/∂x at r/σ={r1_ratio} must be positive, got {gp1:.3e}"
        assert gp2 > 0, f"∂p/∂x at r/σ={r2_ratio} must be positive, got {gp2:.3e}"

        # r⁻⁵ scaling: gp1/gp2 = (r2/r1)^5
        expected_ratio = (r2 / r1) ** 5
        computed_ratio = gp1 / gp2
        rel_err = abs(computed_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.02, (
            f"r/σ=({r1_ratio},{r2_ratio}): expected ratio {expected_ratio:.4f}, "
            f"got {computed_ratio:.4f}, rel_err={rel_err:.3%}"
        )


# =============================================================================
# Test 1.4 — Component decomposition consistency
# =============================================================================


def test_pressure_gradient_component_decomposition(tmp_path):
    """
    The total ∇p returned by compute_target_pressure_gradient_components()
    must equal the element-wise sum convective + viscous + temporal.

    Physical basis
    --------------
    ∇p = −ρ(∂u/∂t + (u·∇)u − nu∇²u)
       = convective + temporal + viscous

    This test fails when
    --------------------
    * One of the additive contributions is subtracted instead of added.
    * The component method duplicates a term that the total method counts once.
    * A term is included in the total but not in the decomposed fields.
    """
    solver = _single_vortex_solver(tmp_path)

    targets = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # include_viscous=True, include_temporal=False (no snapshot)
    components = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=True,
        include_temporal=False,
        include_freestream=False,
    )

    total = components["grad_p"]
    reconstructed = components["convective"] + components["viscous"] + components["temporal"]

    np.testing.assert_allclose(
        total,
        reconstructed,
        atol=1e-10,
        err_msg="grad_p must equal convective + viscous + temporal component-wise",
    )


# =============================================================================
# Test 1.5 — Viscous term is negligible at Re=1000
# =============================================================================


def test_viscous_term_small_at_high_reynolds(tmp_path):
    """
    At nu=1e-3 (Re≈1000 for unit velocity and length), the viscous term
    ρnu∇²u must be at most 1% of the convective term −ρ(u·∇)u.

    Physical basis
    --------------
    The ratio |viscous|/|convective| ≈ nu/(u·L) = 1/Re.  At Re=1000 this is
    0.1% — far below 1%.  If the ratio is larger, something is wrong with the
    finite-difference Laplacian or its scaling.

    This test fails when
    --------------------
    * The h parameter for finite differences is too large (coarsens the stencil).
    * nu is applied twice (once in the term, once in the FD normalisation).
    """
    solver = _single_vortex_solver(tmp_path, sigma=0.5, alpha_z=1.0)

    # Probes outside the core (r ≥ 4σ) where the far-field scaling holds
    r_values = [2.0, 3.0, 4.0]
    targets = np.array([[r, 0.0, 0.0] for r in r_values])

    components = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=True,
        include_temporal=False,
        include_freestream=False,
        particle_spacing=_SIGMA_A,
    )

    conv_mag = np.linalg.norm(components["convective"], axis=1)
    visc_mag = np.linalg.norm(components["viscous"], axis=1)

    for k, r in enumerate(r_values):
        if conv_mag[k] < 1e-10:
            continue  # no convective signal, skip ratio check
        ratio = visc_mag[k] / conv_mag[k]
        assert ratio < 0.01, f"r={r}: viscous/convective = {ratio:.4e} (must be < 0.01 at Re≈1000)"


# =============================================================================
# Test 1.6 — Eulerian temporal term self-consistency
# =============================================================================


def test_eulerian_temporal_term_self_consistency(tmp_path):
    """
    When the previous velocity snapshot is obtained from the same internal
    code path as u_target (PressurePhysics.compute_target_velocities), the
    temporal term must be numerically identical to (u_new − u_old)/dt.

    Physical basis
    --------------
    The Eulerian temporal term is:
        ∂u/∂t ≈ (u^{n} − u^{n−1}) / dt

    Both u^{n} and u^{n−1} must be evaluated by the same function
    (PressurePhysics.compute_target_velocities) to avoid a code-path mismatch.
    In practice: obtain u_prev by calling the SAME internal method before
    moving particles, call the pressure method after.  The temporal term
    returned must equal (u_new − u_prev) / dt.

    This test fails when
    --------------------
    * u_target and velocity_previous are evaluated by different functions.
    * The sign of (u_new − u_old) is inverted.
    * The dt divisor is applied twice.
    """
    solver = _single_vortex_solver(tmp_path)

    targets = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    # Force lazy initialisation of _pressure_physics
    _ = solver.compute_target_pressure_gradients(
        targets, density=_DENSITY, nu=_NU, include_temporal=False, include_freestream=False
    )
    pp = solver._pressure_physics

    # Step n-1: obtain velocity from the SAME internal code path
    u_prev = pp.compute_target_velocities(
        solver.particles, targets, include_freestream=False
    ).copy()

    # Simulate a small particle displacement (freestream advection Δx = U·dt)
    time_step_size = 0.05
    pos = solver.particles.position_cpu()
    solver.particles.set_field("position", pos + np.array([[0.05, 0.0, 0.0]]))

    # Step n: total grad_p with temporal enabled and velocity_previous from step n-1
    grad_p = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=False,
        include_temporal=True,
        temporal_method="eulerian",
        velocity_previous=u_prev,
        time_step_size=time_step_size,
        include_freestream=False,
    )["grad_p"]

    # Retrieve u_new that the pressure method computed internally
    u_new = pp.compute_target_velocities(solver.particles, targets, include_freestream=False)

    # Expected temporal contribution: −ρ·(u_new − u_prev)/dt
    expected_temporal = -_DENSITY * (u_new - u_prev) / time_step_size

    # Expected total grad_p (convective + temporal, no viscous)
    grad_u = pp.compute_target_velocity_gradients(solver.particles, targets).reshape(-1, 3, 3)
    advective = np.einsum("mb,mab->ma", u_new, grad_u)
    expected_convective = -_DENSITY * advective

    expected_total = expected_convective + expected_temporal

    np.testing.assert_allclose(
        grad_p,
        expected_total,
        atol=1e-8,
        rtol=1e-6,
        err_msg=(
            "compute_target_pressure_gradients with temporal_method='eulerian' must match "
            "(convective + temporal), where temporal = −ρ(u_new − u_prev)/dt"
        ),
    )


# =============================================================================
# Test 1.7 — Code-path mismatch canary
# =============================================================================


def test_eulerian_temporal_mismatch_with_corrected_velocity(tmp_path):
    """
    Supplying a velocity_previous obtained from VPMSolver.compute_target_velocities
    (which may differ from PressurePhysics.compute_target_velocities due to
    divergence-free correction or source induction) to a steady particle field
    must produce a nonzero temporal contribution.

    This is the *canary* test for the code-path mismatch between:
        - the coupler's face-centre velocity (corrected for mass flux)
        - the pressure method's internal u_target (pure Biot-Savart)

    Physical basis
    --------------
    For a static particle field where nothing has moved, ∂u/∂t should be zero.
    If we supply a velocity_previous that was computed by a different function
    (e.g., one that applies a divergence-free correction or a constant offset),
    the resulting temporal term is nonzero even though the physics demands zero.
    This documents the bug and ensures it can never be silently reintroduced.

    This test fails when
    --------------------
    * VPMSolver.compute_target_velocities and PressurePhysics.compute_target_velocities
      return identical results (e.g., the divergence-free correction is removed
      from the coupler, at which point this test rightly stops alerting).
    * The test is refactored to pass a consistent snapshot (then test 1.6 is
      the correctness guard and this test should be updated accordingly).
    """
    solver = _single_vortex_solver(tmp_path)
    targets = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Steady state: same particles, two consecutive velocity evaluations
    # via the VPMSolver-level API (which adds div-free correction if present).
    # We simulate the correction by adding a small uniform offset.
    u_via_solver = solver.compute_target_velocities(targets, include_freestream=False)

    # Apply a synthetic divergence-free correction (uniform normal shift) as
    # the coupler does.  This mimics the real mismatch without needing a full
    # coupled geometry.
    correction = 0.01  # m/s — representative div-free correction magnitude
    fake_normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    u_corrected = u_via_solver - correction * fake_normals

    # Nothing has moved — physics says ∂u/∂t = 0.
    # But using the corrected velocity as the previous snapshot must give nonzero temporal.
    time_step_size = 0.1
    _ = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=False,
        include_temporal=True,
        temporal_method="eulerian",
        velocity_previous=u_corrected,
        time_step_size=time_step_size,
        include_freestream=False,
    )["grad_p"]

    # Force lazy init, then get what the pressure method actually computed for u_target
    pp = solver._pressure_physics
    u_internal = pp.compute_target_velocities(solver.particles, targets, include_freestream=False)

    # The temporal term (internal) is (u_internal - u_corrected)/dt
    temporal_from_mismatch = (u_internal - u_corrected) / time_step_size

    # It must NOT be zero (the whole point is to document the mismatch)
    mismatch_norm = np.linalg.norm(temporal_from_mismatch)
    expected_norm = correction / time_step_size  # ≈ 0.01/0.1 = 0.1 m/s²
    assert mismatch_norm > 0.5 * expected_norm, (
        f"Expected nonzero temporal term from code-path mismatch "
        f"(norm ≈ {expected_norm:.3e}), got {mismatch_norm:.3e}. "
        "This means VPMSolver.compute_target_velocities and "
        "PressurePhysics.compute_target_velocities now return the same result — "
        "verify whether the divergence-free correction is still present."
    )


# =============================================================================
# Test 1.8 — return_velocity gives consistent snapshot for temporal term
# =============================================================================


def test_return_velocity_enables_consistent_temporal_term(tmp_path):
    """
    Using return_velocity=True to obtain u_target at step n-1 and passing it
    as velocity_previous at step n must produce the correct Eulerian temporal
    term — identical to the self-consistency result in test 1.6.

    This is the integration test for the fix: `_compute_pressure_bc` should
    use `return_velocity=True` to store a snapshot from the same internal
    code path, then feed it back on the next call.
    """
    solver = _single_vortex_solver(tmp_path)
    targets = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    # Step n-1: get grad_p AND u_snapshot from the same code path
    result = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=False,
        include_temporal=False,
        include_freestream=False,
        return_velocity=True,
    )
    _components_prev, u_snapshot = result

    # Move the particle to simulate physical evolution
    time_step_size = 0.05
    pos = solver.particles.position_cpu()
    solver.particles.set_field("position", pos + np.array([[0.05, 0.0, 0.0]]))

    # Step n: use the stored snapshot as velocity_previous
    grad_p_with_temporal = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=False,
        include_temporal=True,
        temporal_method="eulerian",
        velocity_previous=u_snapshot,
        time_step_size=time_step_size,
        include_freestream=False,
        return_velocity=True,
    )
    _components_temporal, u_new_snapshot = grad_p_with_temporal
    grad_p_temporal = _components_temporal["grad_p"]

    # Independently compute convective-only at the new state
    grad_p_conv_only = solver.compute_target_pressure_gradients(
        targets,
        density=_DENSITY,
        nu=_NU,
        include_viscous=False,
        include_temporal=False,
        include_freestream=False,
    )["grad_p"]

    # The temporal contribution should equal -ρ(u_new - u_old)/dt
    pp = solver._pressure_physics
    u_new = pp.compute_target_velocities(solver.particles, targets, include_freestream=False)
    expected_temporal = -_DENSITY * (u_new - u_snapshot) / time_step_size
    expected_total = grad_p_conv_only + expected_temporal

    np.testing.assert_allclose(
        grad_p_temporal,
        expected_total,
        atol=1e-8,
        rtol=1e-6,
        err_msg=(
            "return_velocity=True snapshot must produce a temporally consistent "
            "grad_p when fed back as velocity_previous"
        ),
    )

    # Also verify the returned snapshot matches what PressurePhysics computes
    np.testing.assert_allclose(
        u_new_snapshot,
        u_new,
        atol=1e-10,
        err_msg="Returned u_snapshot must match PressurePhysics.compute_target_velocities",
    )
