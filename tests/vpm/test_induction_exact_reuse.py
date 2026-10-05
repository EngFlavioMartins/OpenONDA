"""CPU qualification of unwired pure-induction exact-content reuse."""

from dataclasses import replace

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.physics.induction import reuse as reuse_module
from source.solvers.vpm.physics.induction.reuse import (
    ExactContentInductionReuse,
    InductionReuseConditions,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    ti.reset()


def _put(field, values, count):
    array = field.to_numpy()
    array[:count] = values
    field.from_numpy(array)


class PureBackend:
    def __init__(self):
        self.calls = 0
        self.key = ("test-kernel", 1)
        self.offset = 0.0
        self.time_dependent = False
        self.fail = False
        self.last_tail = {"shell": -1}
        self.seen_flags = []

    def conditions(self):
        return InductionReuseConditions(
            operator_key=self.key,
            autonomous=not self.time_dependent,
            sources_are_read_only=True,
            complete_outputs_are_equivalent=True,
            diagnostics_are_complete=True,
            capture_diagnostics=lambda: self.last_tail,
            restore_diagnostics=lambda value: setattr(self, "last_tail", value),
        )

    def evaluate_stage(self, **args):
        self.calls += 1
        self.seen_flags.append(
            (args["strength_rate_enabled"], args["velocity_gradient_out"] is not None)
        )
        count = args["count"]
        position = args["position"].to_numpy()[:count]
        strength = args["vortex_strength"].to_numpy()[:count]
        radius = args["core_radius"].to_numpy()[:count]
        extra = self.offset + (args["stage_time"] if self.time_dependent else 0)
        velocity = position + 2 * strength + radius[:, None] + extra
        gradient = (
            np.einsum("ni,nj->nij", position, strength) + np.eye(3)[None] * radius[:, None, None]
        )
        rate = np.einsum("nij,nj->ni", gradient, strength)
        _put(args["velocity_out"], velocity, count)
        _put(args["vortex_strength_rate_out"], rate if args["strength_rate_enabled"] else 0, count)
        if args["velocity_gradient_out"] is not None:
            _put(args["velocity_gradient_out"], gradient, count)
        self.last_tail = {"shell": 7, "value": float(velocity.sum())}
        if self.fail:
            raise RuntimeError("injected incomplete tail")


class Harness:
    def __init__(self, dtype=ti.f32, *, conditions=True):
        self.backend = PureBackend()
        self.count = 4
        self.position = ti.Vector.field(3, dtype=dtype, shape=8)
        self.strength = ti.Vector.field(3, dtype=dtype, shape=8)
        self.radius = ti.field(dtype=dtype, shape=8)
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=8)
        self.gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=8)
        self.rate = ti.Vector.field(3, dtype=dtype, shape=8)
        self.position.from_numpy(np.arange(24).reshape(8, 3) * 0.125)
        self.strength.from_numpy(np.arange(24).reshape(8, 3) * 0.25 - 1)
        self.radius.from_numpy(np.arange(8) * 0.01 + 0.2)
        self.reuse = ExactContentInductionReuse(
            self.backend,
            max_particles=8,
            source_dtype=dtype,
            result_dtype=dtype,
            conditions_provider=self.backend.conditions if conditions else None,
        )

    def run(self, *, gradient=True, rate=True, time=0.0, position=None):
        self.reuse.evaluate_stage(
            position=self.position if position is None else position,
            vortex_strength=self.strength,
            core_radius=self.radius,
            count=self.count,
            velocity_out=self.velocity,
            velocity_gradient_out=self.gradient if gradient else None,
            vortex_strength_rate_out=self.rate,
            strength_rate_enabled=rate,
            stage_time=time,
        )
        return tuple(
            field.to_numpy()[: self.count].copy()
            for field in (self.velocity, self.gradient, self.rate)
        )


@pytest.fixture
def harness():
    harness = Harness()
    yield harness
    harness.reuse.close()


@pytest.mark.parametrize("dtype", [ti.f32, ti.f64])
def test_exact_reuse_private_outputs_and_state_local_diagnostics(dtype):
    h = Harness(dtype)
    try:
        first = h.run()
        tail = h.backend.last_tail.copy()
        # Simulate arbitrary provider accumulation and later target-query
        # diagnostics. Neither may corrupt the cached pure output.
        h.velocity.fill(123)
        h.gradient.fill(456)
        h.rate.fill(789)
        h.backend.last_tail = {"shell": 200}
        second = h.run(time=2.0)
        assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
        assert h.backend.calls == 1
        assert h.backend.last_tail == tail
        assert h.reuse.statistics.hits == 1
        assert h.reuse.statistics.exact_checks == 1
        # The actual-work counter was NOT restored/advanced on the hit.
        assert h.backend.calls == h.reuse.statistics.misses
    finally:
        h.reuse.close()


@pytest.mark.parametrize("field", ["position", "strength", "radius"])
def test_same_object_mutation_without_revision_forces_miss(harness, field):
    h = harness
    h.run()
    storage = getattr(h, field)
    data = storage.to_numpy()
    data[0] += 0.125
    storage.from_numpy(data)
    h.run()
    assert h.backend.calls == 2
    assert h.reuse.statistics.hits == 0


def test_count_order_signed_zero_and_rollback_are_exact(harness):
    h = harness
    original = h.position.to_numpy()
    h.run()
    h.count = 5
    h.run()
    h.count = 4
    h.run()
    values = original.copy()
    values[:4] = values[:4][::-1]
    h.position.from_numpy(values)
    h.run()
    h.position.from_numpy(original)
    h.run()
    values = original.copy()
    values[0, 0] = -0.0
    h.position.from_numpy(values)
    h.run()
    assert h.backend.calls == 6
    h.run()
    assert h.reuse.statistics.hits == 1


def test_identical_stage_zero_copy_hits_but_equal_time_changed_stage_misses(harness):
    h = harness
    h.run()
    stage = ti.Vector.field(3, dtype=ti.f32, shape=8)
    stage.from_numpy(h.position.to_numpy())
    h.run(position=stage)
    assert h.backend.calls == 1
    values = stage.to_numpy()
    values[0, 2] += 0.5
    stage.from_numpy(values)
    h.run(position=stage)
    assert h.backend.calls == 2


def test_state_check_rate_disabled_then_full_stage_uses_complete_private_result(harness):
    h = harness
    h.gradient.fill(17)
    first = h.run(gradient=False, rate=False)
    assert np.all(first[1] == 17)
    assert np.all(first[2] == 0)
    full = h.run()
    assert h.backend.calls == 1
    assert h.backend.seen_flags == [(True, True)]
    assert np.any(full[1] != 0) and np.any(full[2] != 0)
    disabled = h.run(rate=False)
    assert np.all(disabled[2] == 0)
    assert h.reuse.statistics.hits == 2


def test_failed_miss_leaves_caller_unpublished_and_entry_invalid(harness):
    h = harness
    h.run()
    values = h.radius.to_numpy()
    values[0] *= 2
    h.radius.from_numpy(values)
    h.velocity.fill(17)
    h.gradient.fill(19)
    h.rate.fill(23)
    h.backend.fail = True
    with pytest.raises(RuntimeError, match="incomplete tail"):
        h.run()
    assert np.all(h.velocity.to_numpy() == 17)
    assert np.all(h.gradient.to_numpy() == 19)
    assert np.all(h.rate.to_numpy() == 23)
    assert not h.reuse._valid
    h.backend.fail = False
    h.run()
    assert h.backend.calls == 3


def test_operator_dependency_change_and_explicit_invalidation(harness):
    h = harness
    first = h.run()[0]
    h.backend.key = ("test-kernel", 2)
    h.backend.offset = 1
    second = h.run()[0]
    assert np.allclose(second - first, 1)
    assert h.backend.calls == 2
    h.reuse.invalidate()
    h.run()
    assert h.backend.calls == 3


@pytest.mark.parametrize("unsupported", ["unknown", "time", "subset", "diagnostics", "mutable_key"])
def test_unsupported_conditions_bypass_original_query_unchanged(harness, unsupported):
    h = harness
    provider = h.backend.conditions
    if unsupported == "unknown":
        h.reuse.conditions_provider = None
    elif unsupported == "time":
        h.backend.time_dependent = True
    elif unsupported == "subset":
        h.reuse.conditions_provider = lambda: replace(
            provider(), complete_outputs_are_equivalent=False
        )
    elif unsupported == "diagnostics":
        h.reuse.conditions_provider = lambda: replace(provider(), diagnostics_are_complete=False)
    else:
        h.reuse.conditions_provider = lambda: replace(provider(), operator_key=([],))
    a = h.run(gradient=False, rate=False, time=1)[0]
    b = h.run(gradient=False, rate=False, time=2)[0]
    assert h.backend.calls == 2
    assert h.backend.seen_flags == [(False, False), (False, False)]
    assert h.reuse.statistics.bypasses == 2
    assert h.reuse.statistics.storage_bytes == 0
    if unsupported == "time":
        assert np.allclose(b - a, 1)


def test_nonfinite_results_do_not_grant_source_cache_hit(harness):
    h = harness
    values = h.position.to_numpy()
    values[0, 0] = np.nan
    h.position.from_numpy(values)
    h.run()
    h.run()
    assert h.backend.calls == 2
    assert not h.reuse._valid


def test_external_provider_is_never_part_of_pure_cache(harness):
    h = harness
    provider_calls = []
    totals = []
    for external_state in (1.0, 9.0):
        # This represents the required caller order: guard -> pure reuse ->
        # providers. The module has no blend_weight to skip either outside call.
        pure = h.run()[0]
        provider_calls.append(external_state)
        totals.append(pure + external_state)
    assert h.backend.calls == 1
    assert provider_calls == [1, 9]
    assert np.allclose(totals[1] - totals[0], 8)


def test_operator_key_distinguishes_types_and_float_bits(harness):
    h = harness
    for dependency in (True, 1, 1.0, 0.0, -0.0):
        h.backend.key = (dependency,)
        h.run()
    assert h.backend.calls == 5
    h.run()
    assert h.reuse.statistics.hits == 1


def test_output_precision_mismatch_bypasses_without_rounding(harness):
    h = harness
    h.run()
    h.velocity = ti.Vector.field(3, dtype=ti.f64, shape=8)
    h.backend.offset = 1e-12
    result = h.run()[0]
    assert h.backend.calls == 2
    assert h.reuse.statistics.bypasses == 1
    assert not h.reuse._valid
    # The actual original backend writes the f64 destination directly.
    assert result.dtype == np.float64


def test_nonfinite_private_output_is_published_but_never_cached(harness):
    h = harness
    h.backend.offset = float("inf")
    for _ in range(2):
        assert np.isinf(h.run()[0]).all()
        assert not h.reuse._valid
    assert h.backend.calls == 2


def test_growth_rebinds_storage_and_matches_original_backend(harness):
    h = harness
    h.count = 1
    h.run()
    field_groups = [h.reuse._storage]
    for count in (3, 8):
        h.count = count
        result = h.run()
        assert h.reuse._storage not in field_groups
        field_groups.append(h.reuse._storage)
        direct = Harness()
        try:
            direct.count = count
            expected = direct.run()
            assert all(np.array_equal(a, b) for a, b in zip(result, expected, strict=True))
        finally:
            direct.reuse.close()
        assert h.backend.calls == len(field_groups)
    assert h.reuse.statistics.storage_bytes == 8 * 22 * 4 + 8


@pytest.mark.parametrize("failure", ["allocate", "place", "finalize"])
def test_partial_allocation_releases_acquired_field(monkeypatch, failure):
    field_groups = []
    error = MemoryError("injected cache allocation failure")

    class DeviceFields:
        def __init__(self):
            if failure == "allocate":
                raise error
            self.destroys = 0
            field_groups.append(self)

        def field(self, *args, **kwargs):
            if failure == "place":
                raise error
            return object()

        scalar = vector = matrix = field

        def finalize(self):
            raise error

        def destroy(self):
            self.destroys += 1

    monkeypatch.setattr(reuse_module, "_DeviceFields", DeviceFields)
    storage = reuse_module._ReuseStorage.__new__(reuse_module._ReuseStorage)
    with pytest.raises(MemoryError) as caught:
        storage.__init__(3, ti.f32, ti.f32)
    assert caught.value is error
    assert storage._fields is None
    assert all(solver.destroys == 1 for solver in field_groups)
    storage.destroy()
    assert all(solver.destroys == 1 for solver in field_groups)


def test_cache_growth_failure_invalidates_and_keeps_caller_fields(monkeypatch, harness):
    h = harness
    h.run()
    h.velocity.fill(17)
    h.count = 8

    def fail(*args):
        raise MemoryError("injected replacement allocation failure")

    monkeypatch.setattr(reuse_module, "_ReuseStorage", fail)
    with pytest.raises(MemoryError, match="replacement allocation"):
        h.run()
    assert h.reuse._storage is None
    assert not h.reuse._valid
    assert h.reuse.statistics.storage_bytes == 0
    assert np.all(h.velocity.to_numpy() == 17)


@pytest.mark.parametrize("alias", ["source", "output", "source_view"])
def test_aliased_fields_preserve_original_backend_call(harness, alias):
    h = harness
    h.run()
    if alias == "source":
        h.velocity = h.position
    elif alias == "output":
        h.rate = h.velocity
    else:
        # A distinct Python vector-field view can share all placed members.
        h.velocity = ti.lang.matrix.MatrixField(h.position._get_field_members(), 3, 1)
        assert h.velocity is not h.position
    calls = h.backend.calls
    h.run(gradient=False, rate=False)
    assert h.backend.calls == calls + 1
    assert h.backend.seen_flags[-1] == (False, False)
    assert h.reuse.statistics.bypasses == 1
    assert not h.reuse._valid
