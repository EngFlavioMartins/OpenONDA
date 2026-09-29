"""Body corrections retain every source and every target across device batches."""

import numpy as np
import pytest

from openonda import vpm


@pytest.fixture
def solver(tmp_path):
    result = vpm.VPMSolver(
        vpm.VPMCase(
            directory=tmp_path,
            numerics=vpm.Numerics(
                compute_device="CPU",
                max_n_particles=2,
                verbose=False,
                induction=vpm.DirectInduction(),
            ),
        )
    )
    yield result
    result.close()


def test_surface_sources_beyond_workspace_are_not_truncated(solver):
    count = 2 * solver._source_batch_size + 3
    strength = np.zeros(count)
    strength[-3:] = [1, 2, 3]
    solver.set_surface_sources(np.zeros((count, 3)), strength, np.full(count, 0.05))
    assert solver.n_sources == count
    points = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    initial = np.zeros_like(points)
    velocity = solver._add_target_velocity_corrections(points, initial, include_body=False)
    expected = np.array([[6 / (4 * np.pi), 0, 0], [0, 6 / (16 * np.pi), 0]])
    np.testing.assert_allclose(velocity, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_array_equal(initial, 0)
    with pytest.raises(ValueError, match="finite"):
        solver.set_surface_sources(np.array([[np.nan, 0, 0]]), np.ones(1), np.ones(1))
    assert solver.n_sources == count


def test_large_body_query_preserves_order_and_all_contributions(solver):
    count = solver.physics.max_evaluation_points + 3
    points = np.zeros((count, 3))
    points[:, 0] = np.linspace(1, 3, count)
    solver.set_surface_sources(np.zeros((1, 3)), np.ones(1), np.full(1, 0.05))
    calls = []

    class Body:
        _solved = True

        def add_stage_velocity(self, positions, velocities, n, time):
            calls.append(n)
            values = solver.physics.extract_target_velocity(n)
            values[:, 1] += 0.25
            solver.physics._upload_vector_array(values, velocities, n)

    solver.vlm_solver = Body()
    solver._body_induced_fn = lambda pts, time: np.tile([0, 0, 0.5], (len(pts), 1))
    actual = solver._add_target_velocity_corrections(
        points, np.zeros_like(points), include_body=True
    )
    expected = np.column_stack(
        (1 / (4 * np.pi * points[:, 0] ** 2), np.full(count, 0.25), np.full(count, 0.5))
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    assert calls == [solver.physics.max_evaluation_points, 3]
    solver.vlm_solver = None


def test_rk_source_batches_include_velocity_gradient_and_stretching(solver):
    import taichi as ti

    from source.solvers.vpm.physics.induction.base import StageRates, StageState
    from source.solvers.vpm.physics.stage_rhs import ParticleExternalStageContribution

    count = solver._source_batch_size + 1
    strengths = np.zeros(count)
    strengths[-1] = 2.0
    solver.set_surface_sources(np.zeros((count, 3)), strengths, np.full(count, 0.05))
    position = ti.Vector.field(3, ti.f32, shape=1)
    strength = ti.Vector.field(3, ti.f32, shape=1)
    radius = ti.field(ti.f32, shape=1)
    velocity = ti.Vector.field(3, ti.f32, shape=1)
    rate = ti.Vector.field(3, ti.f32, shape=1)
    gradient = ti.Matrix.field(3, 3, ti.f32, shape=1)
    position[0] = [1, 0, 0]
    strength[0] = [0, 1, 0]
    radius[0] = 0.1
    velocity.fill(0)
    rate.fill(0)
    gradient.fill(0)
    provider = ParticleExternalStageContribution(solver.particles, solver.physics, solver)
    provider.add_stage_rates(
        StageState(position, strength, radius, 1),
        0.0,
        StageRates(velocity, rate, gradient),
    )
    scale = 2 / (4 * np.pi)
    np.testing.assert_allclose(velocity.to_numpy()[0], [scale, 0, 0], rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        gradient.to_numpy()[0], np.diag([-2, 1, 1]) * scale, rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(rate.to_numpy()[0], [0, scale, 0], rtol=2e-6, atol=1e-7)
