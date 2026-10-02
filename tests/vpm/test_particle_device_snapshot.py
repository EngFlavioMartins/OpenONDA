"""Device rollback preserves the existing host replacement contract."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.particles.container import Particles
from source.solvers.vpm.stabilization.manager import StabilizationManager


@pytest.fixture(params=["f32", "f64"])
def owner(request):
    ti.init(arch=ti.cpu, cpu_max_num_threads=2)
    particles = Particles(max_n_particles=32, float_dtype=request.param)
    lineage = SimpleNamespace(reference_vortex_strength=None, reference_lengths=None)
    lineage.on_replacement = MethodType(StabilizationManager.on_replacement, lineage)
    solver = SimpleNamespace(
        particles=particles, stabilization=lineage, _axisymmetric_orbits_validated=True
    )
    solver.capture_particle_snapshot = MethodType(VPMSolver.capture_particle_snapshot, solver)
    solver.restore_particle_snapshot = MethodType(VPMSolver.restore_particle_snapshot, solver)
    yield solver
    for buffer in getattr(solver, "_particle_snapshot_buffers", {}).values():
        buffer.destroy()
    ti.reset()


def _payload(count, seed=49):
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


def _assert_payload(particles, expected):
    expected = {
        name: np.asarray(
            values, dtype=(np.int32 if name.endswith("_id") else particles._np_float_dtype)
        )
        for name, values in expected.items()
    }
    count = len(expected["position"])
    assert particles.n_particles_total == count
    assert particles.device_n_particles[None] == count
    for name, values in expected.items():
        np.testing.assert_array_equal(getattr(particles, name).to_numpy()[:count], values)
    np.testing.assert_allclose(
        particles.vorticity.to_numpy()[:count],
        expected["vortex_strength"] / expected["particle_volume"][:, None],
        rtol=4 * np.finfo(particles._np_float_dtype).eps,
    )
    np.testing.assert_array_equal(
        particles.effective_viscosity.to_numpy()[:count],
        expected["kinematic_viscosity"] + expected["eddy_viscosity"],
    )


def test_device_snapshot_restores_all_fields_without_host_particle_reads(owner, monkeypatch):
    original = _payload(7)
    owner.particles.replace_from_numpy(**original)
    for name in original:
        monkeypatch.setattr(
            owner.particles,
            name + "_cpu",
            lambda: pytest.fail("device rollback must not download a particle field"),
        )
    snapshot = owner.capture_particle_snapshot(slot="predictor")
    assert snapshot.lineage is None
    owner.particles.replace_from_numpy(**_payload(11, seed=81))
    revision = owner.particles.state_revision
    owner.restore_particle_snapshot(snapshot)
    assert owner.particles.state_revision == revision + 1
    assert not owner._axisymmetric_orbits_validated
    _assert_payload(owner.particles, original)


def test_nested_slots_are_independent_and_reused_handles_fail_before_mutation(owner):
    first = _payload(4)
    second = _payload(6, seed=55)
    owner.particles.replace_from_numpy(**first)
    predictor = owner.capture_particle_snapshot(slot="predictor")
    owner.particles.replace_from_numpy(**second)
    trial = owner.capture_particle_snapshot(slot="trial")
    owner.restore_particle_snapshot(predictor)
    _assert_payload(owner.particles, first)
    owner.restore_particle_snapshot(trial)
    _assert_payload(owner.particles, second)
    reused = owner.capture_particle_snapshot(slot="trial")
    assert reused.buffer is trial.buffer
    revision = owner.particles.state_revision
    with pytest.raises(RuntimeError, match="reused or released"):
        owner.restore_particle_snapshot(trial)
    assert owner.particles.state_revision == revision
    _assert_payload(owner.particles, second)


def test_grown_slot_releases_old_allocation_and_empty_snapshot_restores_count(owner):
    empty = owner.capture_particle_snapshot(slot="predictor")
    owner.particles.replace_from_numpy(**_payload(4))
    owner.restore_particle_snapshot(empty)
    assert owner.particles.n_particles_total == 0
    assert owner.particles.device_n_particles[None] == 0
    owner.particles.replace_from_numpy(**_payload(4))
    grown = owner.capture_particle_snapshot(slot="predictor")
    assert grown.buffer.capacity >= 4
    assert empty.buffer._tree is None
    with pytest.raises(RuntimeError, match="reused or released"):
        owner.restore_particle_snapshot(empty)


def test_snapshot_preserves_refinement_lineage(owner):
    original = _payload(6)
    owner.particles.replace_from_numpy(**original)
    owner.stabilization.reference_vortex_strength = np.full(6, 100.0)
    owner.stabilization.reference_lengths = np.full(6, 20.0)
    snapshot = owner.capture_particle_snapshot(slot="predictor")
    assert snapshot.lineage is not None
    owner.particles.replace_from_numpy(**_payload(9, seed=12))
    owner.restore_particle_snapshot(snapshot)
    magnitude = np.linalg.norm(
        original["vortex_strength"].astype(owner.particles._np_float_dtype).astype(np.float64),
        axis=1,
    )
    np.testing.assert_array_equal(owner.stabilization.reference_vortex_strength, magnitude)
    np.testing.assert_array_equal(
        owner.stabilization.reference_lengths,
        np.cbrt(
            original["particle_volume"].astype(owner.particles._np_float_dtype).astype(np.float64)
        ),
    )
    _assert_payload(owner.particles, original)


def test_snapshot_cannot_restore_into_another_owner(owner):
    owner.particles.replace_from_numpy(**_payload(4))
    snapshot = owner.capture_particle_snapshot(slot="predictor")
    other = Particles(max_n_particles=32, float_dtype="f64")
    with pytest.raises(ValueError, match="another solver"):
        snapshot.restore(other)
    assert other.n_particles_total == 0


def test_invalid_snapshot_is_rejected_before_restoring_any_fields(owner):
    original = _payload(4)
    owner.particles.replace_from_numpy(**original)
    owner.particles.velocity[2] = [float("nan"), 0.0, 0.0]
    snapshot = owner.capture_particle_snapshot(slot="predictor")
    owner.particles.replace_from_numpy(**original)
    revision = owner.particles.state_revision
    with pytest.raises(ValueError, match="non-finite"):
        owner.restore_particle_snapshot(snapshot)
    assert owner.particles.state_revision == revision
    _assert_payload(owner.particles, original)
