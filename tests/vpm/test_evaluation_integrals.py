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
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver.add_vortex_particles(
        position=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        circulation=np.array([gamma1, gamma2]),
        radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        viscosity=np.zeros(2),
    )
    return solver


# ── Tests ─────────────────────────────────────────────────────────────────────


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
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver.update_state()
    assert solver.total_kinetic_energy == 0.0
    assert solver.total_enstrophy == 0.0
    assert solver.total_helicity == 0.0
    assert np.allclose(solver.total_linear_impulse, 0.0)
    assert np.allclose(solver.total_angular_impulse, 0.0)
    assert np.allclose(solver.total_strength, 0.0)


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
    solver1.update_state()
    solver2.update_state()
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
    solver1.update_state()
    solver2.update_state()
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
    solver1.update_state()
    solver2.update_state()
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
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    solver2.add_vortex_particles(
        position=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]) + shift,
        velocity=np.zeros((2, 3)),
        circulation=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        viscosity=np.zeros(2),
    )
    solver1.update_state()
    solver2.update_state()
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
    solver.update_state()

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
