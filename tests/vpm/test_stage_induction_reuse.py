"""Production StageRHS dispatch and private-cache lifecycle qualification."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.core import solver as solver_module
from source.solvers.vpm.physics.induction.base import StageRates, StageState
from source.solvers.vpm.physics.stage_rhs import StageRHS
from tests.vpm.test_induction_reuse_backends import Harness


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


def _evaluate(rhs, h, time, *, rate=True):
    rhs.evaluate(
        StageState(h.position, h.strength, h.radius, h.count, time=time),
        time,
        StageRates(h.velocity, h.rate, h.gradient, strength_rate_enabled=rate),
    )


def test_default_dispatch_retains_backend_guard_and_all_providers():
    h = Harness()
    events = []

    class Provider:
        def __init__(self, index):
            self.index = index

        def add_stage_rates(self, state, time, rates):
            events.append((self.index, time))
            rates.velocity.from_numpy(rates.velocity.to_numpy() + time * self.index)
            rates.vortex_strength_rate.fill(21)

    rhs = StageRHS(h.backend, (Provider(1), Provider(2)), strength_enabled=False)

    def guard(state):
        events.append(("guard", state.time))
        if state.time == 3:
            h.position[0] = h.position[0] + ti.Vector([0.01, 0, 0])

    rhs.position_guard = guard
    try:
        pure = h.velocity.to_numpy()
        for time in (1.0, 2.0, 3.0):
            _evaluate(rhs, h, time, rate=False)
            if time < 3:
                np.testing.assert_array_equal(h.velocity.to_numpy(), (pure + time) + 2 * time)
            np.testing.assert_array_equal(h.rate.to_numpy(), 0)
        assert rhs.induction is h.backend
        assert rhs.induction_reuse.backend is h.backend
        assert rhs.induction_reuse_statistics.hits == 1
        assert rhs.induction_reuse_statistics.misses == 2
        assert events == [(who, time) for time in (1.0, 2.0, 3.0) for who in ("guard", 1, 2)]
        snapshot = rhs.induction_reuse_statistics
        snapshot.hits = 1000
        assert rhs.induction_reuse_statistics.hits == 1
        rhs.close()
        rhs.close()
        assert rhs.induction_reuse is None
        assert rhs.induction_reuse_statistics.hits == 1
        assert rhs.induction_reuse_statistics.storage_bytes == 0
        with pytest.raises(RuntimeError, match="closed"):
            _evaluate(rhs, h, 4.0)
        assert len(events) == 9
    finally:
        rhs.close()
        h.close()


def test_replacement_and_runtime_disable_release_old_owners():
    h = Harness()
    rhs = StageRHS(h.backend)

    class Unknown:
        def __init__(self):
            self.calls = 0

        def evaluate_stage(self, **args):
            self.calls += 1
            args["velocity_out"].fill(args["stage_time"])
            args["vortex_strength_rate_out"].fill(0)

    try:
        _evaluate(rhs, h, 1.0)
        old = rhs.induction_reuse
        unknown = Unknown()
        rhs.induction = unknown
        _evaluate(rhs, h, 2.0)
        assert old._storage is None
        assert rhs.induction_reuse is None
        assert unknown.calls == 1
        np.testing.assert_array_equal(h.velocity.to_numpy(), 2)
        rhs.induction = h.backend
        _evaluate(rhs, h, 3.0)
        current = rhs.induction_reuse
        assert current is not old
        assert rhs.induction_reuse_statistics.misses == 2
        rhs.reuse_induction = False
        _evaluate(rhs, h, 4.0)
        assert current._storage is None
        assert rhs.induction_reuse is None
        assert rhs.induction_reuse_statistics.storage_bytes == 0
        assert rhs.induction_reuse_statistics.misses == 2
    finally:
        rhs.close()
        h.close()


def test_opt_out_uses_original_backend_on_every_call():
    h = Harness()
    rhs = StageRHS(h.backend, reuse_induction=False)
    try:
        before = h.base.diagnostics.stage_evaluations
        _evaluate(rhs, h, 1.0)
        _evaluate(rhs, h, 2.0)
        assert h.base.diagnostics.stage_evaluations == before + 2
        assert rhs.induction_reuse is None
        assert rhs.induction_reuse_statistics is None
    finally:
        rhs.close()
        h.close()


def test_default_slab_dispatch_hits_and_guard_failure_never_publishes():
    h = Harness(slab=True)
    rhs = StageRHS(h.backend)
    try:
        _evaluate(rhs, h, 1.0)
        expected = h.velocity.to_numpy()
        _evaluate(rhs, h, 2.0)
        np.testing.assert_array_equal(h.velocity.to_numpy(), expected)
        assert rhs.induction_reuse_statistics.hits == 1
        h.position[0] = [0, 0, 3]
        h.velocity.fill(31)
        with pytest.raises(RuntimeError, match="escaped"):
            _evaluate(rhs, h, 3.0)
        np.testing.assert_array_equal(h.velocity.to_numpy(), 31)
        assert not rhs.induction_reuse._valid
    finally:
        rhs.close()
        h.close()


@pytest.mark.parametrize("fail", [None, "cache", "snapshot"])
def test_solver_closes_cache_before_backend_reset_even_on_cleanup_failure(monkeypatch, fail):
    events = []

    def cleanup(label):
        events.append(label)
        if fail == label:
            raise RuntimeError(f"{label} cleanup failed")

    owner = SimpleNamespace(
        _closed=False,
        stage_rhs=SimpleNamespace(close=lambda: cleanup("cache")),
        _particle_snapshot_buffers={"a": SimpleNamespace(destroy=lambda: cleanup("snapshot"))},
        _run_started=True,
        _initial_conditions_built=True,
        _backend_claimed=True,
        _restore_output_streams=lambda: cleanup("streams"),
    )
    monkeypatch.setattr(solver_module, "reset_taichi_backend", lambda **kwargs: cleanup("reset"))
    if fail is None:
        solver_module.VPMSolver.close(owner)
        solver_module.VPMSolver.close(owner)
        assert owner._closed
    else:
        with pytest.raises(RuntimeError, match=f"{fail} cleanup failed"):
            solver_module.VPMSolver.close(owner)
    assert events == ["cache", "snapshot", "streams", "reset"]
    assert owner._particle_snapshot_buffers == {}
    assert not owner._backend_claimed
