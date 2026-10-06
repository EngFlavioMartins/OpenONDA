"""Channel wall, Gaussian curl, energy and restart-configuration qualification."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from openonda import vpm
from source.solvers.vpm.config.configuration_values import numerical_configuration
from tests.support.cylinder.audit_saved_wall_circulation import gaussian_velocity_and_gradient
from tests.support.cylinder.audit_slip_channel_induction import channel_image_velocity_gradient


def test_channel_width_and_copy_are_part_of_native_restart_configuration():
    original = vpm.PlanarChannelInduction(half_width=10, span=2, plane_z=0.3)
    copy = original.build()
    assert copy is not original
    assert copy.channel_half_width == 10
    assert copy.planar_span == 2
    viscous = vpm.ViscousConfig.gbd(particle_spacing=0.04, kinematic_viscosity=0.01)
    first = numerical_configuration(vpm.Numerics(induction=original, viscous=viscous))
    second = numerical_configuration(
        vpm.Numerics(
            induction=vpm.PlanarChannelInduction(half_width=20, span=2, plane_z=0.3),
            viscous=viscous,
        )
    )
    assert first["induction"]["channel_half_width"] == 10
    assert first["induction"] != second["induction"]
    with pytest.raises(ValueError, match="half width"):
        vpm.PlanarChannelInduction(half_width=0)
    with pytest.raises(ValueError, match="full-3D"):
        vpm.SlipSlabInduction(original, z_min=-0.5, z_max=0.5)
    with pytest.raises(ValueError, match="f32 diffusion grid"):
        vpm.Numerics(induction=vpm.DirectInduction(), viscous=viscous, precision="f64")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_channel_velocity_jacobian_wall_condition_and_source_core_guard(device):
    if device == "cuda" and os.environ.get("OPENONDA_TEST_CUDA") != "1":
        pytest.skip("Set OPENONDA_TEST_CUDA=1 for native CUDA qualification")
    ti.reset()
    ti.init(arch=getattr(ti, device), default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
    assert ti.cfg.arch == getattr(ti, device)
    try:
        model = vpm.PlanarChannelInduction(half_width=10, span=2).bind(
            SimpleNamespace(accumulator_dtype=ti.f64, max_n_particles=8)
        )
        source = ti.Vector.field(3, ti.f32, shape=8)
        target = ti.Vector.field(3, ti.f64, shape=8)
        strength = ti.Vector.field(3, ti.f32, shape=8)
        radius = ti.field(ti.f64, shape=8)
        velocity = ti.Vector.field(3, ti.f64, shape=8)
        gradient = ti.Matrix.field(3, 3, ti.f64, shape=8)
        positions = np.array([[-1, 0.4, 0], [4, -0.7, 0]], dtype=np.float32)
        points = np.vstack((positions, [[0.2, 0.1, 3], [0.7, -10, 0], [0.7, 10, 0], [70, -5, 0]]))
        coefficients = np.array([2.6, -2], dtype=np.float32)
        source.from_numpy(np.vstack((positions, np.zeros((6, 3)))).astype(np.float32))
        strength.from_numpy(
            np.column_stack((np.zeros(8), np.zeros(8), np.r_[coefficients, np.zeros(6)])).astype(
                np.float32
            )
        )
        radius.fill(0.04)
        velocity.fill(2)
        gradient.fill(3)

        def evaluate(values):
            target.from_numpy(np.vstack((values, np.zeros((2, 3)))))
            model.evaluate_targets(
                target_position=target,
                source_position=source,
                source_vortex_strength=strength,
                source_core_radius=radius,
                target_velocity=velocity,
                target_velocity_gradient=gradient,
                target_count=6,
                source_count=2,
                include_freestream=True,
                background_velocity=(1, 0, 0),
            )
            return velocity.to_numpy()[:6], gradient.to_numpy()[:6]

        actual, jacobian = evaluate(points)
        circulation = coefficients.astype(float) / 2
        expected, expected_jacobian = gaussian_velocity_and_gradient(
            points, positions.astype(float), circulation, 0.04
        )
        correction, correction_jacobian = channel_image_velocity_gradient(
            points, positions.astype(float), circulation, 10
        )
        np.testing.assert_allclose(actual, expected + correction + [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(jacobian, expected_jacobian + correction_jacobian, atol=1e-10)
        np.testing.assert_allclose(actual[3:5, 1], 0, atol=1e-14)
        np.testing.assert_allclose(np.trace(jacobian, axis1=1, axis2=2), 0, atol=1e-12)
        np.testing.assert_allclose(
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
            expected_jacobian[:, 1, 0] - expected_jacobian[:, 0, 1],
            atol=1e-10,
        )
        np.testing.assert_array_equal(velocity.to_numpy()[6:], np.full((2, 3), 2))
        np.testing.assert_array_equal(gradient.to_numpy()[6:], np.full((2, 3, 3), 3))
        for axis in range(2):
            offset = np.zeros_like(points)
            offset[:3, axis] = 1e-6
            offset[5, axis] = 1e-6
            difference = (evaluate(points + offset)[0] - evaluate(points - offset)[0]) / 2e-6
            np.testing.assert_allclose(
                difference[[0, 1, 2, 5]], jacobian[[0, 1, 2, 5], :, axis], atol=1e-7
            )
        radius[0] = 2
        with pytest.raises(ValueError, match="nine Gaussian core radii"):
            evaluate(points)
        radius[0] = 0.04
        outside = points.copy()
        outside[2, 1] = 11
        with pytest.raises(ValueError, match="targets"):
            evaluate(outside)
    finally:
        ti.reset()


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_channel_solver_nonzero_circulation_energy_and_viscous_evolution(tmp_path, precision):
    numerics = vpm.Numerics(
        induction=vpm.PlanarChannelInduction(half_width=2),
        compute_device="CPU",
        time_step_size=0.01,
        max_n_particles=2000,
        precision=precision,
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
        position = np.array([[0, 0.4, 0]], dtype=np.float32)
        solver.add_vortex_particles(
            position=position,
            velocity=np.zeros_like(position),
            vortex_strength=np.array([[0, 0, 0.01]], dtype=np.float32),
            core_radius=np.array([0.1]),
            particle_volume=np.array([0.01]),
            kinematic_viscosity=np.array([0.01]),
        )
        initial = solver.field_diagnostics.compute_flow_integrals(solver.particles, 0)
        assert initial["energy_measurement"] == "planar_channel_energy"
        assert np.isfinite(initial["total_kinetic_energy"])
        assert initial["total_kinetic_energy"] > 0
        assert initial["net_vortex_strength"][2] > 0
        probes = np.array([[0.2, -2, 0], [0.2, 2, 0]], dtype=np.float32)
        velocity, gradient = solver.compute_velocity_and_gradient_at_points(
            probes, particle_spacing=0.1
        )
        np.testing.assert_allclose(velocity[:, 1], 0, atol=1e-7)
        np.testing.assert_allclose(np.trace(gradient, axis1=1, axis2=2), 0, atol=1e-7)
        solver.advance()
        solver.advance()
        assert solver.particle_position.dtype == np.dtype(precision.replace("f", "float"))
        assert solver.particle_vortex_strength.dtype == solver.particle_position.dtype
        assert np.isfinite(solver.particle_velocity).all()
        assert np.isfinite(solver.particle_velocity_gradient).all()
        np.testing.assert_array_equal(solver.particle_position[:, 2], 0)
        final = solver.field_diagnostics.compute_flow_integrals(solver.particles, 0.02)
        assert final["energy_measurement"] == "planar_channel_energy"
        assert np.isfinite(final["total_kinetic_energy"])
        np.testing.assert_allclose(
            final["net_vortex_strength"], initial["net_vortex_strength"], atol=1e-6
        )
    finally:
        solver.close()


def test_channel_energy_matches_independent_velocity_quadrature_for_nonzero_circulation():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False, cpu_max_num_threads=1)
    try:
        model = vpm.PlanarChannelInduction(half_width=2, span=2).bind(
            SimpleNamespace(accumulator_dtype=ti.f64, max_n_particles=1)
        )
        position = ti.Vector.field(3, ti.f64, shape=1)
        strength = ti.Vector.field(3, ti.f64, shape=1)
        radius = ti.field(ti.f64, shape=1)
        position[0], strength[0], radius[0] = [0, 0.4, 0], [0, 0, 2.6], 0.1
        model._integrals(position, strength, radius, 1)
        model._add_image_energy(position, strength, 1)
        energy = float(model._energy[0])
        assert energy > 0
        nodes, weights = np.polynomial.legendre.leggauss(256)
        y_nodes, y_weights = np.polynomial.legendre.leggauss(400)
        y, y_weights = 2 * y_nodes, 2 * y_weights
        numerical_energy = 0.0
        for lower, upper in ((-20, -0.6), (-0.6, 0.6), (0.6, 20)):
            x = (upper + lower) / 2 + (upper - lower) / 2 * nodes
            x_weights = (upper - lower) / 2 * weights
            for start in range(0, len(x), 16):
                xx, yy = np.meshgrid(x[start : start + 16], y, indexing="ij")
                points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
                velocity, _ = gaussian_velocity_and_gradient(
                    points, np.array([[0, 0.4, 0]]), np.array([1.3]), 0.1
                )
                correction, _ = channel_image_velocity_gradient(
                    points, np.array([[0, 0.4, 0]]), np.array([1.3]), 2
                )
                squared_velocity = np.sum((velocity + correction) ** 2, axis=1).reshape(xx.shape)
                numerical_energy += np.sum(
                    squared_velocity * x_weights[start : start + 16, None] * y_weights[None, :]
                )  # one-half times represented span = 1 m
        np.testing.assert_allclose(energy, numerical_energy, rtol=1e-9)
        np.testing.assert_allclose(
            model._enstrophy[0], 1.3**2 * 2 / (2 * np.pi * 0.1**2), rtol=1e-12
        )
    finally:
        ti.reset()
