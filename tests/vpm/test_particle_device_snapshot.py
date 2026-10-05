"""Device rollback preserves the existing host replacement conditions."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.stabilization.manager import StabilizationManager


@pytest.fixture(params=["f32", "f64"])
def solver(request):
    ti.init(arch=ti.cpu, cpu_max_num_threads=2)
    particles = Particles(max_n_particles=32, float_dtype=request.param)
    refinement_reference = SimpleNamespace(reference_vortex_strength=None, reference_lengths=None)
    refinement_reference.on_replacement = MethodType(
        StabilizationManager.on_replacement, refinement_reference
    )
    solver = SimpleNamespace(
        particles=particles, stabilization=refinement_reference, _axisymmetric_orbits_validated=True
    )
    solver.capture_particle_snapshot = MethodType(VPMSolver.capture_particle_snapshot, solver)
    solver.restore_particle_snapshot = MethodType(VPMSolver.restore_particle_snapshot, solver)
    yield solver
    for buffer in getattr(solver, "_particle_snapshot_buffers", {}).values():
        buffer.destroy()
    ti.reset()


def _particle_fields(count, seed=49):
    rng = np.random.default_rng(seed)
    return {
        "position": rng.normal(size=(count, 3)),
        "velocity": rng.normal(size=(count, 3)),
        "vortex_strength": rng.normal(size=(count, 3)),
        "core_radius": rng.uniform(0.1, 0.2, size=count),
        "particle_volume": rng.uniform(0.1, 0.2, size=count),
        "kinematic_viscosity": rng.uniform(0.001, 0.002, size=count),
        "eddy_viscosity": rng.uniform(0.002, 0.003, size=count),
        "group_id": np.arange(count, dtype=np.int32),
        "zone_id": -np.arange(count, dtype=np.int32),
        "velocity_gradient": rng.normal(size=(count, 3, 3)),
        "strain_rate": rng.normal(size=(count, 3, 3)),
    }


def _assert_particle_fields(solver, expected):
    expected = {
        name: np.asarray(
            values, dtype=(np.int32 if name.endswith("_id") else solver._np_float_dtype)
        )
        for name, values in expected.items()
    }
    count = len(expected["position"])
    assert solver.n_particles_total == count
    assert solver.device_n_particles[None] == count
    for name, values in expected.items():
        np.testing.assert_array_equal(getattr(solver, name).to_numpy()[:count], values)
    np.testing.assert_allclose(
        solver.vorticity.to_numpy()[:count],
        expected["vortex_strength"] / expected["particle_volume"][:, None],
        rtol=4 * np.finfo(solver._np_float_dtype).eps,
    )
    np.testing.assert_array_equal(
        solver.effective_viscosity.to_numpy()[:count],
        expected["kinematic_viscosity"] + expected["eddy_viscosity"],
    )


def test_device_snapshot_restores_all_fields_without_host_particle_reads(solver, monkeypatch):
    original = _particle_fields(7)
    solver.particles.replace_from_numpy(**original)
    for name in original:
        monkeypatch.setattr(
            solver.particles,
            name + "_cpu",
            lambda: pytest.fail("device rollback must not download a particle field"),
        )
    snapshot = solver.capture_particle_snapshot(slot="predictor")
    assert snapshot.refinement_reference is None
    solver.particles.replace_from_numpy(**_particle_fields(11, seed=81))
    revision = solver.particles.state_revision
    solver.restore_particle_snapshot(snapshot)
    assert solver.particles.state_revision == revision + 1
    assert not solver._axisymmetric_orbits_validated
    _assert_particle_fields(solver.particles, original)


def test_nested_slots_are_independent_and_reused_handles_fail_before_mutation(solver):
    first = _particle_fields(4)
    second = _particle_fields(6, seed=55)
    solver.particles.replace_from_numpy(**first)
    predictor = solver.capture_particle_snapshot(slot="predictor")
    solver.particles.replace_from_numpy(**second)
    trial = solver.capture_particle_snapshot(slot="trial")
    solver.restore_particle_snapshot(predictor)
    _assert_particle_fields(solver.particles, first)
    solver.restore_particle_snapshot(trial)
    _assert_particle_fields(solver.particles, second)
    reused = solver.capture_particle_snapshot(slot="trial")
    assert reused.buffer is trial.buffer
    revision = solver.particles.state_revision
    with pytest.raises(RuntimeError, match="reused or released"):
        solver.restore_particle_snapshot(trial)
    assert solver.particles.state_revision == revision
    _assert_particle_fields(solver.particles, second)


def test_grown_slot_releases_old_allocation_and_empty_snapshot_restores_count(solver):
    empty = solver.capture_particle_snapshot(slot="predictor")
    solver.particles.replace_from_numpy(**_particle_fields(4))
    solver.restore_particle_snapshot(empty)
    assert solver.particles.n_particles_total == 0
    assert solver.particles.device_n_particles[None] == 0
    solver.particles.replace_from_numpy(**_particle_fields(4))
    grown = solver.capture_particle_snapshot(slot="predictor")
    assert grown.buffer.capacity >= 4
    assert empty.buffer._tree is None
    with pytest.raises(RuntimeError, match="reused or released"):
        solver.restore_particle_snapshot(empty)


def test_snapshot_preserves_refinement_reference(solver):
    original = _particle_fields(6)
    solver.particles.replace_from_numpy(**original)
    solver.stabilization.reference_vortex_strength = np.full(6, 100.0)
    solver.stabilization.reference_lengths = np.full(6, 20.0)
    snapshot = solver.capture_particle_snapshot(slot="predictor")
    assert snapshot.refinement_reference is not None
    solver.particles.replace_from_numpy(**_particle_fields(9, seed=12))
    solver.restore_particle_snapshot(snapshot)
    magnitude = np.linalg.norm(
        original["vortex_strength"].astype(solver.particles._np_float_dtype).astype(np.float64),
        axis=1,
    )
    np.testing.assert_array_equal(solver.stabilization.reference_vortex_strength, magnitude)
    np.testing.assert_array_equal(
        solver.stabilization.reference_lengths,
        np.cbrt(
            original["particle_volume"].astype(solver.particles._np_float_dtype).astype(np.float64)
        ),
    )
    _assert_particle_fields(solver.particles, original)


def test_snapshot_cannot_restore_into_another_particle_container(solver):
    solver.particles.replace_from_numpy(**_particle_fields(4))
    snapshot = solver.capture_particle_snapshot(slot="predictor")
    other = Particles(max_n_particles=32, float_dtype="f64")
    with pytest.raises(ValueError, match="another particle container"):
        snapshot.restore(other)
    assert other.n_particles_total == 0


def test_invalid_snapshot_is_rejected_before_restoring_any_fields(solver):
    original = _particle_fields(4)
    solver.particles.replace_from_numpy(**original)
    solver.particles.velocity[2] = [float("nan"), 0.0, 0.0]
    snapshot = solver.capture_particle_snapshot(slot="predictor")
    solver.particles.replace_from_numpy(**original)
    revision = solver.particles.state_revision
    with pytest.raises(ValueError, match="non-finite"):
        solver.restore_particle_snapshot(snapshot)
    assert solver.particles.state_revision == revision
    _assert_particle_fields(solver.particles, original)
