"""
Flow integral evaluation tests — scaling, invariance, and empty-system edge cases.

The solver exposes five primary flow integrals:
    total_kinetic_energy, total_enstrophy, total_helicity,
    total_linear_impulse, total_angular_impulse

All tests here are algebraic consequences of the definitions; no time-stepping
is performed.
"""

import numpy as np
import pytest

from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

_SIGMA = 0.2
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


def _two_particle_solver(make_solver, kernel_name, gamma1, gamma2):
    """Create a solver with two particles at fixed positions."""
    solver = make_solver(
        time_step_size=0.01,
        particle_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver.add_vortex_particles(
        position=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([gamma1, gamma2]),
        core_radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        kinematic_viscosity=np.zeros(2),
    )
    return solver


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_energy_rate_matches_latest_nonuniform_diagnostic_interval(
    solver_for_backend,
):
    """dE/dt must retain the sign and slope of the latest sampled energies."""
    solver = solver_for_backend(
        time_step_size=0.01,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    diagnostics = solver.field_diagnostics
    diagnostics._flow_time_history = [
        (0.0, 27.0),
        (0.3, 19.0),
        (1.1, 18.0),
        (2.7, 17.8),
        (3.0, 17.79),
        (4.2, 17.78),
        (5.0, 17.77),
    ]

    expected = (17.77 - 17.78) / (5.0 - 4.2)
    assert diagnostics._compute_energy_dissipation_rate() == pytest.approx(expected)
    assert diagnostics._compute_energy_dissipation_rate() < 0.0


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_empty_system_zero_integrals(kernel_name, backend, solver_for_backend):
    """
    An empty solver must report exactly zero for every flow integral.

    Failure → uninitialised accumulator or missing N==0 guard.
    """
    solver = solver_for_backend(
        time_step_size=0.01,
        particle_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver.advance()
    solver._update_all_flow_integrals()
    assert solver.total_kinetic_energy == 0.0
    assert solver.total_enstrophy == 0.0
    assert solver.total_helicity == 0.0
    assert np.allclose(solver.total_linear_impulse, 0.0)
    assert np.allclose(solver.total_angular_impulse, 0.0)
    assert np.allclose(solver.total_strength, 0.0)


def test_angular_impulse_core_correction_is_per_particle(backend, solver_for_backend):
    """Unequal cores must use Σ sigma_i² Gamma_i, not mean(sigma²) Σ Gamma."""
    solver = solver_for_backend(
        time_step_size=0.01,
        particle_kernel="GAUSSIAN",
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver.add_vortex_particles(
        position=np.zeros((2, 3)),
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        core_radius=np.array([0.1, 0.3]),
        volume=np.full(2, _VOLUME),
        kinematic_viscosity=np.zeros(2),
    )

    integrals = solver.field_diagnostics.compute_flow_integrals(
        solver.particles, solver.time, record_history=False
    )
    # Gaussian second moment m2 = 3/2 (verified against 3-D quadrature of
    # int x*(x*omega) dV; it was 3.0 here and in the kernel, twice too large).
    # The raw position term is zero because both blobs sit at the origin.
    expected = -(2.0 / 9.0) * 1.5 * np.array([0.0, 0.0, 0.1**2 - 0.3**2])
    assert np.allclose(integrals["angular_impulse"], expected, atol=1e-6)


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_strength_linear_scaling(kernel_name, backend, solver_for_backend):
    """
    Total strength ΣΓ must scale linearly with circulation.

    Doubling all circulations doubles the total strength vector.

    Failure → wrong strength accumulation or missing particle.
    """
    solver1 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    solver2 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 2.0],
        gamma2=[0.0, 0.0, 2.0],
    )
    solver1.advance()
    solver1._update_all_flow_integrals()
    solver2.advance()
    solver2._update_all_flow_integrals()
    assert np.allclose(solver2.total_strength, 2.0 * solver1.total_strength), (
        f"{kernel_name}/{backend}: strength not linear in Γ"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_kinetic_energy_quadratic_scaling(kernel_name, backend, solver_for_backend):
    """
    Kinetic energy is bilinear in circulation: E ∝ Γ².

    Doubling all circulations must quadruple KE.

    Failure → wrong energy kernel or missing cross-terms.
    """
    solver1 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    solver2 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 2.0],
        gamma2=[0.0, 0.0, 2.0],
    )
    solver1.advance()
    solver1._update_all_flow_integrals()
    solver2.advance()
    solver2._update_all_flow_integrals()
    ratio = solver2.total_kinetic_energy / (solver1.total_kinetic_energy + 1e-15)
    assert abs(ratio - 4.0) < 0.02, (
        f"{kernel_name}/{backend}: KE ratio = {ratio:.4f} (expected 4.0)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_linear_impulse_linear_scaling(kernel_name, backend, solver_for_backend):
    """
    Linear impulse I = ½ Σ r × Γ scales linearly with circulation.

    Failure → wrong impulse formula.
    """
    solver1 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    solver2 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 2.0],
        gamma2=[0.0, 0.0, 2.0],
    )
    solver1.advance()
    solver1._update_all_flow_integrals()
    solver2.advance()
    solver2._update_all_flow_integrals()
    assert np.allclose(solver2.total_linear_impulse, 2.0 * solver1.total_linear_impulse), (
        f"{kernel_name}/{backend}: linear impulse not linear in Γ"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_translation_invariance_energy_enstrophy(kernel_name, backend, solver_for_backend):
    """
    KE and enstrophy are translation-invariant: shifting all particles by a
    constant vector must not change them.

    Failure → position-dependent kernel normalisation or boundary effects.
    """
    solver1 = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    shift = np.array([10.0, -5.0, 3.0])
    solver2 = solver_for_backend(
        time_step_size=0.01,
        particle_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver2.add_vortex_particles(
        position=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]) + shift,
        velocity=np.zeros((2, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        core_radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        kinematic_viscosity=np.zeros(2),
    )
    solver1.advance()
    solver1._update_all_flow_integrals()
    solver2.advance()
    solver2._update_all_flow_integrals()
    assert (
        abs(solver1.total_kinetic_energy - solver2.total_kinetic_energy)
        / (solver1.total_kinetic_energy + 1e-15)
        < 0.01
    ), f"{kernel_name}/{backend}: KE not translation-invariant"
    assert (
        abs(solver1.total_enstrophy - solver2.total_enstrophy) / (solver1.total_enstrophy + 1e-15)
        < 0.01
    ), f"{kernel_name}/{backend}: enstrophy not translation-invariant"


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cross_backend_consistency_flow_integrals(kernel_name, backend, solver_for_backend):
    """
    Flow integrals on CPU, CUDA, and Vulkan must agree within 1%.

    Failure → backend-specific divergence in evaluation kernel.
    """
    if not hasattr(test_cross_backend_consistency_flow_integrals, "_cpu_cache"):
        test_cross_backend_consistency_flow_integrals._cpu_cache = {}

    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, -1.0],
    )
    solver.advance()
    solver._update_all_flow_integrals()

    key = kernel_name
    if backend == "CPU":
        test_cross_backend_consistency_flow_integrals._cpu_cache[key] = {
            "ke": solver.total_kinetic_energy,
            "enstrophy": solver.total_enstrophy,
            "helicity": solver.total_helicity,
            "impulse": solver.total_linear_impulse.copy(),
        }
    else:
        ref = test_cross_backend_consistency_flow_integrals._cpu_cache.get(key)
        if ref is None:
            pytest.skip("CPU reference not yet computed — run CPU tests first")
        rel_ke = abs(solver.total_kinetic_energy - ref["ke"]) / (ref["ke"] + 1e-15)
        rel_ens = abs(solver.total_enstrophy - ref["enstrophy"]) / (ref["enstrophy"] + 1e-15)
        rel_imp = np.linalg.norm(solver.total_linear_impulse - ref["impulse"]) / (
            np.linalg.norm(ref["impulse"]) + 1e-15
        )
        assert rel_ke < 0.01, f"{kernel_name}/{backend}: KE mismatch vs CPU: {rel_ke:.3e}"
        assert rel_ens < 0.01, f"{kernel_name}/{backend}: enstrophy mismatch vs CPU: {rel_ens:.3e}"
        assert rel_imp < 0.01, f"{kernel_name}/{backend}: impulse mismatch vs CPU: {rel_imp:.3e}"
