"""Analytical checks for infinite-span velocity, diffusion, and coupled renewal."""

from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from openonda import vpm
from source.coupler.stable_renewal import (
    build_stable_renewal_lattice,
    gaussian_represented_vortex_strength,
    recover_planar_vortex_invariants,
    renew_stable_overlap,
    vortex_invariants,
    vortex_strength_from_velocity_trace,
)
from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.diffusion.planar import (
    diffuse_planar_grid,
    planar_gbd,
    scatter_planar,
)
from source.solvers.vpm.physics.induction.planar import PlanarInduction


@pytest.mark.parametrize("plane_z", [0.0, 0.137])
@pytest.mark.parametrize("amplitude", [1e-8, 1.0, 1e8])
def test_planar_renewal_is_continuous_at_particle_birth(plane_z, amplitude):
    """Shrinking input perturbations cannot produce a finite support jump."""
    h = 0.05
    threshold = amplitude * 0.05 * h**2
    lattice = build_stable_renewal_lattice(
        (-0.3, 0.3, -0.3, 0.3, -0.5, 0.5),
        h,
        buffer_length=0.1,
        authority_ramp_width=0.1,
        planar_span=1.0,
        plane_z=plane_z,
    )
    # Isolate the pruning/recovery map from the fixed geometric blend.
    n = len(lattice.positions)
    lattice = replace(
        lattice,
        fvm_authority=np.ones(n),
        fluid_weight=np.ones(n),
        mesh_weight=np.ones(n),
        solid_interior=np.zeros(n, dtype=bool),
    )
    target = np.zeros((n, 3))
    lookup = {}
    for i in range(7):
        for j in range(7):
            xy = np.array([i - 3, j - 3]) * h
            index = int(np.argmin(np.linalg.norm(lattice.positions[:, :2] - xy, axis=1)))
            lookup[i, j] = index
            target[index, 2] = 0.35 * threshold
    for i, j, strength in [(1, 1, 5), (1, 5, -4), (5, 1, -3), (5, 5, 6), (3, 4, 3)]:
        target[lookup[i, j], 2] = strength * threshold
    crossing = lookup[3, 3]

    def renew(epsilon):
        perturbed = target.copy()
        perturbed[crossing, 2] = threshold * (1 + epsilon)
        result = renew_stable_overlap(
            np.empty((0, 3)),
            np.empty((0, 3)),
            lattice,
            fvm_vortex_strength_at_node=lambda _: perturbed,
            prune_threshold=threshold,
            amplification_cap=1.0,
        )
        expected = vortex_invariants(lattice.positions, perturbed)
        actual = vortex_invariants(result.position, result.vortex_strength)
        scale = np.abs(perturbed[:, 2]).sum()
        np.testing.assert_allclose(
            actual.total_vortex_strength, expected.total_vortex_strength, rtol=0, atol=scale * 2e-14
        )
        np.testing.assert_allclose(
            actual.linear_impulse, expected.linear_impulse, rtol=0, atol=scale * 2e-14
        )
        np.testing.assert_array_equal(result.vortex_strength[:, :2], 0)
        np.testing.assert_array_equal(result.particle_volume, h**2)
        PlanarInduction(plane_z=plane_z).validate_source_arrays(
            result.position,
            result.vortex_strength,
        )
        dense = np.zeros_like(target)
        for position, strength in zip(result.position, result.vortex_strength, strict=True):
            index = int(np.argmin(np.linalg.norm(lattice.positions - position, axis=1)))
            dense[index] = strength
        return dense

    jumps = []
    for epsilon in [1e-5, 1e-8]:
        before, after = renew(-epsilon), renew(epsilon)
        assert before[crossing, 2] == 0
        assert after[crossing, 2] != 0
        jump = np.linalg.norm(after - before, axis=1).sum()
        assert jump < 10 * (2 * epsilon * threshold)
        jumps.append(jump)
    assert jumps[1] < 0.002 * jumps[0]


def test_planar_weighted_recovery_rejects_insufficient_support():
    position = np.array([[0, 0, 0.137], [1, 0, 0.137], [2, 0, 0.137]])
    strength = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=float)
    target = vortex_invariants(position, 2 * strength)
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        recover_planar_vortex_invariants(position, strength, target, volumes=np.ones(3))
    with pytest.raises(ValueError, match="without retained strength"):
        recover_planar_vortex_invariants(position, strength * 0, target, volumes=np.ones(3))


def test_planar_gaussian_velocity_curl_and_jacobian():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=2)
    try:
        n = 8
        physics = PhysicsBase(max_n_particles=n, max_evaluation_points=n, accumulator_dtype=ti.f64)
        backend = PlanarInduction(span=2.0).bind(physics)

        def field():
            return ti.Vector.field(3, ti.f64, shape=n)

        source, target, strength, velocity, rate = (field() for _ in range(5))
        radius = ti.field(ti.f64, shape=n)
        gradient = ti.Matrix.field(3, 3, ti.f64, shape=n)
        positions = np.array(
            [
                [0, 0, 0],
                [0.2, -0.1, 0],
                [0.2, -0.1, 50],
                [1, 0, 0],
                [1e-8, 0, 0],
                [0.7, 0.5, 0],
                [0, 0, 0],
                [0, 0, 0.0],
            ]
        )
        source.fill(0)
        target.from_numpy(positions)
        strength.fill(0)
        radius.fill(0.3)
        strength[0] = [0, 0, 6]  # circulation 3 m²/s, represented span 2 m

        def evaluate(points):
            target.from_numpy(points)
            backend.evaluate_targets(
                target_position=target,
                source_position=source,
                source_vortex_strength=strength,
                source_core_radius=radius,
                target_velocity=velocity,
                target_velocity_gradient=gradient,
                target_count=n,
                source_count=1,
                include_freestream=False,
                background_velocity=(0, 0, 0),
            )
            return velocity.to_numpy(), gradient.to_numpy()

        u, jac = evaluate(positions)
        r2 = (positions[:, :2] ** 2).sum(1)
        f = np.full(n, 1 / 0.3**2)
        np.divide(-np.expm1(-r2 / 0.3**2), r2, out=f, where=r2 > 0)
        expected = (
            3
            / (2 * np.pi)
            * np.column_stack((-positions[:, 1] * f, positions[:, 0] * f, np.zeros(n)))
        )
        np.testing.assert_allclose(u, expected, rtol=2e-12, atol=1e-14)
        np.testing.assert_allclose(
            jac[:, 1, 0] - jac[:, 0, 1], 3 / (np.pi * 0.3**2) * np.exp(-r2 / 0.3**2), atol=1e-12
        )
        np.testing.assert_allclose(np.trace(jac, axis1=1, axis2=2), 0, atol=1e-13)
        for axis in range(3):
            delta = np.zeros_like(positions)
            delta[:, axis] = 1e-6
            diff = (evaluate(positions + delta)[0] - evaluate(positions - delta)[0]) / (2e-6)
            np.testing.assert_allclose(jac[:, :, axis], diff, rtol=2e-8, atol=1e-9)
        backend.evaluate_stage(
            position=source,
            vortex_strength=strength,
            core_radius=radius,
            count=1,
            velocity_out=velocity,
            vortex_strength_rate_out=rate,
            velocity_gradient_out=gradient,
        )
        np.testing.assert_array_equal(rate.to_numpy()[0], 0)
    finally:
        ti.reset()


def test_planar_scatter_and_diffusion_preserve_circulation_and_heat_variance():
    h, nu, dt = 0.05, 0.01, 0.17
    x = np.array([[0.127, -0.041, 0.0]])
    strength = np.array([[0.0, 0.0, 2.0]])
    origin = np.array([-1.0, -1.0])
    grid = scatter_planar(x, strength, origin, h, 41, 41)
    after, count = diffuse_planar_grid(grid, nu, dt, h)
    assert count >= 3
    xx, yy = np.meshgrid(
        origin[0] + h * np.arange(41), origin[1] + h * np.arange(41), indexing="ij"
    )
    np.testing.assert_allclose(after.sum((0, 1)), strength[0], atol=1e-14)
    for coord, centre in zip((xx, yy), x[0, :2], strict=True):
        assert np.sum(coord * after[:, :, 2]) == pytest.approx(2 * centre, abs=1e-13)
        variance = np.sum((coord - centre) ** 2 * after[:, :, 2]) / 2
        assert variance == pytest.approx(2 * nu * dt, abs=1e-13)


def test_planar_renewal_uses_full_span_volume_without_end_taper():
    h, span = 0.1, 0.2
    lattice = build_stable_renewal_lattice(
        (-1, 1, -1, 1, -0.1, 0.1), h, buffer_length=0.2, authority_ramp_width=0.3, planar_span=span
    )
    assert lattice.shape[2] == 1
    middle = np.argmin(np.linalg.norm(lattice.positions[:, :2], axis=1))
    assert lattice.fvm_authority[middle] == 1

    def velocity(points):
        return np.column_stack((-points[:, 1], points[:, 0], np.zeros(len(points))))

    def target(points):
        return vortex_strength_from_velocity_trace(points, h, velocity, planar_span=span)

    np.testing.assert_allclose(target(lattice.positions)[:, 2], 2 * h * h * span, atol=1e-15)
    result = renew_stable_overlap(
        np.empty((0, 3)), np.empty((0, 3)), lattice, fvm_vortex_strength_at_node=target
    )
    np.testing.assert_allclose(result.particle_volume, h * h * span)
    np.testing.assert_array_equal(result.position[:, 2], 0)
    np.testing.assert_allclose(result.vortex_strength[:, :2], 0, atol=1e-15)
    # At interior constant strength, the represented 2D Gaussian is normalized;
    # an accidental third z convolution would multiply it by 1/sqrt(pi).
    represented = gaussian_represented_vortex_strength(
        np.tile([0.0, 0.0, 1.0], (len(lattice.positions), 1)),
        lattice.shape,
        h,
        core_radius=h,
        dimensions=2,
    )
    assert represented[middle, 2] == pytest.approx(1, rel=3e-4)


def test_planar_gbd_retains_moments_after_pruning():
    h = 0.05
    xx, yy = np.meshgrid(np.arange(-0.5, 0.51, h), np.arange(-0.5, 0.51, h), indexing="ij")
    position = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    strength = np.zeros_like(position)
    strength[:, 2] = h * h * np.exp(-np.sum(position[:, :2] ** 2, axis=1) / 0.1)
    particles = SimpleNamespace(
        n_particles_total=len(position),
        position_cpu=lambda: position,
        vortex_strength_cpu=lambda: strength,
    )
    config = vpm.ViscousConfig.gbd(
        particle_spacing=h,
        gbd_grid_spacing=h,
        kinematic_viscosity=0.01,
        threshold_mode="absolute",
        threshold=2e-4,
    )
    result, _ = planar_gbd(
        particles,
        config,
        PlanarInduction(span=1),
        dt=0.1,
        anchor=np.zeros(3),
        core_radius_ratio=1.0,
        max_particles=2000,
    )
    weights = result["vortex_strength"][:, 2]
    np.testing.assert_allclose(weights.sum(), strength[:, 2].sum(), atol=1e-14)
    for axis in range(2):
        expected = (
            np.sum(position[:, axis] ** 2 * strength[:, 2]) + 2 * 0.01 * 0.1 * strength[:, 2].sum()
        )
        assert np.sum(result["position"][:, axis] ** 2 * weights) == pytest.approx(
            expected, abs=1e-13
        )
    np.testing.assert_allclose(result["particle_volume"], h * h)


def test_planar_configuration_rejects_three_dimensional_physics():
    with pytest.raises(ValueError, match="supports GBD or NONE"):
        vpm.Numerics(induction=vpm.PlanarInduction())
    with pytest.raises(ValueError, match="zero spanwise"):
        vpm.Numerics(
            induction=vpm.PlanarInduction(),
            viscous=vpm.ViscousConfig(scheme="NONE"),
            freestream_velocity=(1, 0, 0.1),
        )


@pytest.mark.parametrize("particle_cap", [None, 50])
def test_pruned_nonbinary_plane_renewal_keeps_exactly_axial_strength(particle_cap):
    """A constrained correction cannot create transverse roundoff at z=.137."""
    lattice = build_stable_renewal_lattice(
        (-1, 1, -1, 1, -0.5, 0.5),
        0.1,
        buffer_length=0.2,
        authority_ramp_width=0.3,
        planar_span=1,
        plane_z=0.137,
    )

    def target(points):
        strengths = np.zeros_like(points)
        strengths[:, 2] = 0.007 * np.exp(
            -((points[:, 0] - 0.123) ** 2 + (points[:, 1] + 0.211) ** 2) / 0.13
        ) - 0.003 * np.exp(-((points[:, 0] + 0.21) ** 2 + (points[:, 1] - 0.15) ** 2) / 0.09)
        return strengths

    donors = np.empty((0, 3)) if particle_cap is None else np.array([[2.0, 0.3, np.float32(0.137)]])
    donor_strength = np.empty((0, 3)) if particle_cap is None else np.array([[0.0, 0.0, 0.002]])
    result = renew_stable_overlap(
        donors,
        donor_strength,
        lattice,
        fvm_vortex_strength_at_node=target,
        prune_threshold=0.001,
        maximum_particle_count=particle_cap,
    )
    assert result.conservation_raw_mismatch["total_vortex_strength"] > 1e-4
    assert result.conservation_raw_mismatch["linear_impulse"] > 1e-4
    np.testing.assert_array_equal(result.vortex_strength[:, :2], 0)
    np.testing.assert_array_equal(result.position[:, 2].astype(np.float32), np.float32(0.137))
    vpm.PlanarInduction(plane_z=0.137).validate_source_arrays(
        result.position, result.vortex_strength
    )
    for key in ("total_vortex_strength", "linear_impulse"):
        assert result.conservation_residual[key] < 1e-14
    if particle_cap is not None:
        assert len(result.position) == particle_cap
        for key in ("total_vortex_strength", "linear_impulse"):
            assert result.population_conservation_residual[key] < 1e-14


def test_planar_invariant_recovery_rejects_three_dimensional_input():
    position = np.array([[0.0, 0.0, 0.137], [1.0, 0.0, 0.137], [0.0, 1.0, 0.137]])
    strength = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    target = vortex_invariants(position, strength)
    invalid = strength.copy()
    invalid[0, 0] = 1e-30
    with pytest.raises(ValueError, match="only z"):
        recover_planar_vortex_invariants(position, invalid, target, volumes=np.ones(3))
    position[0, 2] += 1e-3
    with pytest.raises(ValueError, match="common source plane"):
        recover_planar_vortex_invariants(position, strength, target, volumes=np.ones(3))


@pytest.mark.parametrize("device", ["CPU", "METAL"])
def test_planar_solver_target_routes_and_evolution(tmp_path, device):
    """Real RK/GBD solver: sampler curl, filtered queries, restart diagnostics."""
    from scipy.special import exp1

    if device == "METAL" and os.environ.get("OPENONDA_TEST_METAL") != "1":
        pytest.skip("Set OPENONDA_TEST_METAL=1 for native Metal qualification")
    if device == "METAL" and not ti._lib.core.with_metal():
        pytest.skip("Metal backend unavailable")
    numerics = vpm.Numerics(
        induction=vpm.PlanarInduction(span=1),
        compute_device=device,
        time_step_size=0.01,
        max_n_particles=2000,
        max_evaluation_points=2000,
        freestream_velocity=(1, 0, 0),
        verbose=False,
        viscous=vpm.ViscousConfig.gbd(
            particle_spacing=0.1,
            gbd_grid_spacing=0.1,
            kinematic_viscosity=0.01,
            threshold_mode="absolute",
            threshold=1e-7,
            core_radius_ratio=1,
        ),
    )
    solver = vpm.VPMSolver(
        vpm.VPMCase(
            numerics=numerics,
            directory=tmp_path,
            run=vpm.RunPlan(steps=2, initial_samples=False, final_backup=False),
        )
    )
    try:
        positions = np.array([[-0.3, 0, 0], [0.3, 0, 0]], dtype=np.float32)
        strength = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]], dtype=np.float32)
        solver.add_vortex_particles(
            position=positions,
            velocity=np.zeros_like(positions),
            vortex_strength=strength,
            core_radius=np.full(2, 0.1),
            particle_volume=np.full(2, 0.01),
            kinematic_viscosity=np.full(2, 0.01),
        )
        with pytest.raises(ValueError, match="only a z"):
            solver.induction.validate_source_arrays(positions, np.ones_like(strength))
        with pytest.raises(ValueError, match="plane_z"):
            solver.induction.validate_source_arrays(positions + [0, 0, 0.1], strength)
        probes = np.array([[0.1, 0.2, 0], [0.1, 0.2, 2]], dtype=np.float32)
        velocity, gradient = solver.compute_velocity_and_gradient_at_points(
            probes, particle_spacing=0.1
        )
        omega = solver.compute_vorticity_at_points(probes)
        np.testing.assert_allclose(velocity[0], velocity[1])
        np.testing.assert_allclose(omega[:, 2], gradient[:, 1, 0] - gradient[:, 0, 1], atol=1e-7)
        np.testing.assert_allclose(
            solver.physics.compute_transport_target_velocity(solver.particles, probes, 0.1),
            velocity,
        )
        filtered = solver.physics.compute_target_velocity(
            solver.particles, probes, zone_mask=np.array([True, False])
        )
        assert np.isfinite(filtered).all()
        with pytest.raises(NotImplementedError, match="Planar pressure"):
            solver.compute_pressure_gradient_at_points(probes)
        integrals = solver.field_diagnostics.compute_flow_integrals(solver.particles, 0)
        combined_sigma2 = 0.02
        centre_potential = 0.5 * (np.log(combined_sigma2) - np.euler_gamma)
        off_potential = 0.5 * (np.log(0.6**2) + exp1(0.6**2 / combined_sigma2))
        expected_energy = 0.01**2 / (2 * np.pi) * (off_potential - centre_potential)
        assert integrals["total_kinetic_energy"] == pytest.approx(expected_energy, rel=2e-5)
        sampler = vpm.LineSampler(
            start=[0.1, 0.2, 0],
            end=[0.1, 0.2, 1],
            spacing=0.2,
            file_name="planar_probe",
            schedule=vpm.EverySteps(1),
        )
        sampled = sampler.sample(solver)
        np.testing.assert_allclose(sampled["velocity_x"], sampled["velocity_x"][0])
        np.testing.assert_allclose(
            sampled["vorticity_z"],
            sampled["velocity_gradient_yx"] - sampled["velocity_gradient_xy"],
            atol=1e-7,
        )
        solver.advance()
        solver.advance()
        assert np.isfinite(solver.particle_velocity).all()
        assert np.isfinite(solver.particle_velocity_gradient).all()
        np.testing.assert_array_equal(solver.particle_position[:, 2], 0)
        np.testing.assert_array_equal(solver.particle_vortex_strength[:, :2], 0)
        np.testing.assert_allclose(solver.particle_volume, 0.01)
    finally:
        solver.close()
