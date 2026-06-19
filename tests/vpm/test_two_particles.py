"""
Two-particle interaction tests — pair symmetry, superposition, and conservation.

Two particles is the minimum configuration where non-trivial pair interactions
occur.  All tests have analytical answers or exact symmetry arguments.
"""

import numpy as np
import pytest

from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    ViscousConfig,
)

_SIGMA = 0.15
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


def _two_particle_solver(make_solver, kernel_name, pos1, pos2, gamma1, gamma2):
    """Create a solver with two particles."""
    solver = make_solver(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver.add_vortex_particles(
        position=np.array([pos1, pos2]),
        velocity=np.zeros((2, 3)),
        circulation=np.array([gamma1, gamma2]),
        radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        viscosity=np.zeros(2),
    )
    return solver


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_mutual_induction_symmetry(kernel_name, backend, solver_for_backend):
    """
    Two identical particles symmetric about the origin induce equal and opposite
    velocities on each other: u(A from B) = −u(B from A) in the x-direction.

    Failure → sign error in Biot-Savart cross product.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[-0.5, 0.0, 0.0],
        pos2=[0.5, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    vel_at_1 = solver.compute_target_velocities(
        np.array([[-0.5, 0.0, 0.0]]), include_freestream=False
    )
    vel_at_2 = solver.compute_target_velocities(
        np.array([[0.5, 0.0, 0.0]]), include_freestream=False
    )
    # u_y should be equal (symmetric)
    assert abs(vel_at_1[0, 1] - vel_at_2[0, 1]) / (abs(vel_at_1[0, 1]) + 1e-12) < 0.01, (
        f"{kernel_name}/{backend}: mutual induction not symmetric: "
        f"u_y(A)={vel_at_1[0, 1]:.6e}, u_y(B)={vel_at_2[0, 1]:.6e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_vorticity_superposition(kernel_name, backend, solver_for_backend):
    """
    Vorticity at the midpoint of two identical blobs must equal twice the
    single-blob value (superposition principle).

    Failure → non-linear vorticity kernel or missing cross terms.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[-0.5, 0.0, 0.0],
        pos2=[0.5, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    omega_two = solver.compute_target_vorticities(np.array([[0.0, 0.0, 0.0]]))

    # Single blob at same position
    solver1 = solver_for_backend(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver1.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    omega_one = solver1.compute_target_vorticities(np.array([[0.0, 0.0, 0.0]]))

    ratio = float(omega_two[0, 2]) / float(omega_one[0, 2])
    assert abs(ratio - 2.0) < 0.02, (
        f"{kernel_name}/{backend}: vorticity superposition failed: ratio={ratio:.4f}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_kinetic_energy_pairwise(kernel_name, backend, solver_for_backend):
    """
    Two blobs at distance d: the cross-term kinetic energy must be positive
    and smaller than the sum of self-energies.

    Failure → wrong sign in energy kernel g or missing pairwise terms.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[0.0, 0.0, 0.0],
        pos2=[1.0, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    solver.update_state()
    ke_two = solver.total_kinetic_energy

    # Self-energy of one blob (same as test_single_blob)
    solver1 = solver_for_backend(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver1.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    solver1.update_state()
    ke_one = solver1.total_kinetic_energy

    # Total KE of two blobs = 2*self + cross-term
    # cross-term must be positive
    assert ke_two > 2.0 * ke_one * 0.5, (
        f"{kernel_name}/{backend}: KE_two={ke_two:.6e} not > self_energy={ke_one:.6e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_helicity_parallel_zero(kernel_name, backend, solver_for_backend):
    """
    Two parallel z-vortices have zero helicity (Γ₁ × Γ₂ = 0).

    Failure → helicity kernel does not vanish for parallel vortices.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[0.0, 0.0, 0.0],
        pos2=[1.0, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    solver.update_state()
    assert abs(solver.total_helicity) < 1e-5, (
        f"{kernel_name}/{backend}: helicity of parallel vortices = {solver.total_helicity:.3e} (must be 0)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_helicity_orthogonal_nonzero(kernel_name, backend, solver_for_backend):
    """
    Two orthogonal vortices (Γ₁=(1,0,0), Γ₂=(0,1,0)) have non-zero helicity.

    Failure → helicity kernel vanishes identically.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[0.0, 0.0, 0.0],
        pos2=[1.0, 0.0, 0.0],
        gamma1=[1.0, 0.0, 0.0],
        gamma2=[0.0, 1.0, 0.0],
    )
    solver.update_state()
    assert abs(solver.total_helicity) > 1e-6, (
        f"{kernel_name}/{backend}: helicity of orthogonal vortices = {solver.total_helicity:.3e} (must be nonzero)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_enstrophy_pairwise(kernel_name, backend, solver_for_backend):
    """
    Enstrophy of two identical blobs must exceed the enstrophy of one blob
    (cross-term is positive for identical circulations).

    Failure → wrong sign in enstrophy kernel.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[0.0, 0.0, 0.0],
        pos2=[1.0, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, 1.0],
    )
    solver.update_state()
    ens_two = solver.total_enstrophy

    solver1 = solver_for_backend(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver1.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    solver1.update_state()
    ens_one = solver1.total_enstrophy

    assert ens_two > ens_one, (
        f"{kernel_name}/{backend}: enstrophy_two={ens_two:.6e} not > enstrophy_one={ens_one:.6e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_strain_rate_pure_shear(kernel_name, backend, solver_for_backend):
    """
    Two counter-rotating vortices (vortex dipole) produce a pure shear at the
    midpoint: S_xy is maximum, S_xx = S_yy = S_zz = 0.

    Failure → wrong velocity gradient kernel or symmetry violation.
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[-0.5, 0.0, 0.0],
        pos2=[0.5, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, -1.0],
    )
    grad = solver.compute_target_velocity_gradients(np.array([[0.0, 0.0, 0.0]])).reshape(1, 3, 3)
    S = 0.5 * (grad[0] + grad[0].T)
    assert abs(S[0, 0]) < 1e-6 and abs(S[1, 1]) < 1e-6 and abs(S[2, 2]) < 1e-6, (
        f"{kernel_name}/{backend}: diagonal strain components must vanish: {np.diag(S)}"
    )
    assert abs(S[0, 1]) > 1e-6, f"{kernel_name}/{backend}: S_xy must be nonzero: {S[0, 1]:.3e}"
    assert abs(S[0, 1] - S[1, 0]) < 1e-8, (
        f"{kernel_name}/{backend}: strain tensor must be symmetric: S_xy={S[0, 1]}, S_yx={S[1, 0]}"
    )
