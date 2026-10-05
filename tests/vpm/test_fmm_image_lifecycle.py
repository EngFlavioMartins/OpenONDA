"""Pure orchestration tests for image-FMM freshness and scratch ownership.

All workspaces are fake Python objects. No Taichi runtime, compilation or GPU
allocation is needed; image numerical qualification lives in separate tests.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.fmm import device


class _CapacityError(RuntimeError):
    def __init__(self, capacity, required):
        self.required_pairs = required
        super().__init__(f"{capacity} < {required}")


class _DeclinedBlockError(RuntimeError):
    def __init__(self, required_pairs, capacity, diagnostics=None):
        self.required_pairs = required_pairs
        self.capacity = capacity
        self.diagnostics = diagnostics


@pytest.fixture
def rig(monkeypatch):
    state = SimpleNamespace(
        events=[],
        targets=[],
        sources=[],
        source_alloc_failure=False,
        target_alloc_failure=False,
        required_pairs=0,
        always_overflow=False,
    )

    class Source:
        def __init__(self, count, *_args, max_pairs=None, _list_capacities=None, **_kwargs):
            if state.source_alloc_failure:
                raise MemoryError("source allocation failed")
            self.max_n_particles = count
            capacities = 32 * count if max_pairs is None else max_pairs
            self.list_capacities = device._normalize_list_capacities(
                capacities if _list_capacities is None else _list_capacities
            )
            self.max_pairs = max(self.list_capacities)
            self.target_batch_capacity = 16
            self.profile_passes = False
            self.destroyed = False
            self.builds = self.preparations = 0
            self.velocity = np.zeros(32)
            self.gradient = np.zeros(32)
            self.tree = SimpleNamespace(build=self.build)
            state.sources.append(self)
            state.events.append(("source_created", self))

        def build(self, position, strength, radius, count):
            assert not self.destroyed
            self.builds += 1
            self.strength = np.asarray(strength)[:count].copy()

        def prepare_source_multipoles(self, count):
            assert not self.destroyed
            self.preparations += 1
            self.moment = float(self.strength.sum())

        def destroy(self):
            assert not self.destroyed
            self.destroyed = True
            state.events.append(("source_destroyed", self))

        def _empty_target_pass(self, velocity, gradient, background, count, write_v, write_g):
            if write_v:
                velocity[:count] = 0
            if write_g:
                gradient[:count] = 0

    class Target:
        def __init__(self, source, targets, *, max_images, max_pairs=None):
            if state.target_alloc_failure:
                raise MemoryError("target allocation failed")
            self.source = source
            self.max_targets = targets
            self.max_images = max_images
            self.max_pairs = 4 if max_pairs is None else max_pairs
            self.preparations = self.evaluations = 0
            self.destroyed = False
            state.targets.append(self)
            state.events.append(("target_created", self))

        def prepare_targets(self, position, count, *, target_start):
            assert not self.destroyed and not self.source.destroyed
            self.preparations += 1
            self.points = np.asarray(position)[target_start : target_start + count].copy()

        def evaluate_image_block(self, images, velocity, gradient, background):
            assert not self.destroyed and not self.source.destroyed
            self.evaluations += 1
            if state.always_overflow or self.max_pairs < state.required_pairs:
                raise _CapacityError(self.max_pairs, max(state.required_pairs, self.max_pairs + 1))
            value = (
                self.source.moment + float(self.points.sum()) + sum(shift for shift, _ in images)
            )
            if velocity is not None:
                velocity[:] = value
            if gradient is not None:
                gradient[:] = value

        def destroy(self):
            assert not self.destroyed
            assert not self.source.destroyed, "dependent target must be destroyed first"
            self.destroyed = True
            state.events.append(("target_destroyed", self))

    monkeypatch.setattr(device.ti, "sync", lambda: None)
    monkeypatch.setattr(device.ti.lang.impl, "get_runtime", lambda: SimpleNamespace(prog=object()))
    monkeypatch.setattr(device.ti.lang.impl, "current_cfg", lambda: SimpleNamespace(arch=device.ti.cpu))
    monkeypatch.setattr(device, "FMMDeviceWorkspace", Source)
    monkeypatch.setitem(
        sys.modules,
        "source.solvers.vpm.physics.induction.fmm.targets",
        SimpleNamespace(
            FMMTargetEvaluator=Target,
            TargetInteractionCapacityError=_CapacityError,
            TargetBlockNotWorthwhile=_DeclinedBlockError,
        ),
    )
    physics = SimpleNamespace(
        accumulator_dtype=device.ti.f32,
        particle_kernel="GAUSSIAN",
        max_n_particles=128,
        max_evaluation_points=16,
        _kernel_functions={"radial_factors_": object()},
        _zero_velocity=np.zeros(3),
    )
    state.backend = device.FMMInduction().bind(physics)
    monkeypatch.setattr(state.backend, "estimated_workspace_bytes", lambda *_args, **_kwargs: 100)
    state.physics = physics
    state.position = np.arange(12, dtype=np.float32).reshape(4, 3)
    state.strength = np.ones((4, 3), dtype=np.float32)
    state.radius = np.ones(4, dtype=np.float32)
    state.points = np.arange(18, dtype=np.float32).reshape(6, 3)
    state.velocity = np.full((3, 3), -101.0)
    state.gradient = np.full((3, 3, 3), -102.0)
    return state


def _query(rig, **changes):
    arguments = {
        "source_position": rig.position,
        "source_vortex_strength": rig.strength,
        "source_core_radius": rig.radius,
        "source_count": 4,
        "target_position": rig.points,
        "target_start": 0,
        "target_count": 3,
        "images": ((1.0, False),),
        "target_velocity": rig.velocity,
        "target_velocity_gradient": rig.gradient,
    }
    arguments.update(changes)
    return rig.backend.evaluate_image_block(**arguments)


def _scope(rig, **changes):
    arguments = {
        "position": rig.position,
        "strength": rig.strength,
        "radius": rig.radius,
        "count": 4,
    }
    arguments.update(changes)
    return rig.backend.fixed_source_targets(**arguments)


def test_outside_scope_rebuilds_even_when_mutated_source_objects_are_identical(rig):
    _query(rig)
    first = rig.velocity.copy()
    source = rig.backend.workspace
    rig.strength *= 2
    _query(rig)
    assert source.builds == source.preparations == 2
    np.testing.assert_array_equal(rig.velocity - first, np.full_like(first, 12.0))
    assert rig.backend._fixed_source_key is None
    assert getattr(rig.backend, "_prepared_target_key", None) is None


def test_source_scope_reuses_moments_and_clears_target_tile_on_exit(rig):
    with _scope(rig):
        _query(rig)
        _query(rig, images=((2.0, True),))
        source = rig.backend.workspace
        target = rig.backend._target_workspace
        assert source.builds == source.preparations == 1
        assert target.preparations == 2
        assert target.evaluations == 2
    with _scope(rig):
        _query(rig)
    assert source.builds == source.preparations == 2
    assert target.preparations == 3


@pytest.mark.parametrize(
    "changed", ["source_position", "source_vortex_strength", "source_core_radius", "source_count"]
)
def test_source_scope_rejects_different_fields_and_count_before_publication(rig, changed):
    values = {
        "source_position": rig.position.copy(),
        "source_vortex_strength": rig.strength.copy(),
        "source_core_radius": rig.radius.copy(),
        "source_count": 3,
    }
    with _scope(rig), pytest.raises(RuntimeError, match="different source fields"):
        _query(rig, **{changed: values[changed]})
    assert not rig.targets
    assert np.all(rig.velocity == -101) and np.all(rig.gradient == -102)


def test_target_slice_and_object_changes_force_new_preparation(rig):
    with _scope(rig):
        _query(rig)
        target = rig.backend._target_workspace
        _query(rig, target_start=2)
        _query(rig, target_position=rig.points.copy(), target_start=2)
        _query(rig, target_start=2, target_count=2)
    assert target.preparations == 4


def test_target_field_mutation_is_not_hidden_by_source_only_scope(rig):
    with _scope(rig):
        _query(rig)
        first = rig.velocity.copy()
        rig.points += 1
        _query(rig)
    np.testing.assert_array_equal(rig.velocity - first, np.full_like(first, 9.0))


def test_exception_clears_scope_and_cached_target_geometry(rig):
    with pytest.raises(ValueError, match="caller failed"), _scope(rig):
        _query(rig)
        raise ValueError("caller failed")
    assert rig.backend._fixed_source_key is None
    assert getattr(rig.backend, "_prepared_target_key", None) is None


def test_capacity_growth_replaces_whole_target_and_reprepares_unmodified_input(rig):
    rig.required_pairs = 200
    with pytest.warns(RuntimeWarning, match="image-list storage"):
        _query(rig)
    assert len(rig.targets) == 2
    old, new = rig.targets
    assert old.destroyed and not new.destroyed
    assert old.preparations == new.preparations == 1
    np.testing.assert_array_equal(old.points, new.points)
    assert rig.backend.workspace.preparations == 1
    assert not np.any(rig.velocity == -101)


def test_bounded_overflow_publishes_no_partial_outputs(rig):
    rig.always_overflow = True
    with (
        pytest.warns(RuntimeWarning, match="image-list storage"),
        pytest.raises(RuntimeError, match="bounded growth"),
    ):
        _query(rig)
    assert len(rig.targets) == device._MAX_LIST_GROWTH_RETRIES + 1
    assert sum(target.evaluations for target in rig.targets) == len(rig.targets)
    assert all(target.destroyed for target in rig.targets[:-1])
    assert np.all(rig.velocity == -101) and np.all(rig.gradient == -102)
    assert rig.backend._fixed_source_key is None


def test_dense_image_block_is_declined_before_allocation_or_publication(rig):
    rig.required_pairs = device._MAX_IMAGE_PAIR_CAPACITY + 1
    with pytest.raises(_DeclinedBlockError) as caught:
        _query(rig)
    assert caught.value.required_pairs == rig.required_pairs
    assert caught.value.capacity == device._MAX_IMAGE_PAIR_CAPACITY
    assert len(rig.targets) == 1
    assert rig.targets[0].max_pairs <= device._MAX_IMAGE_PAIR_CAPACITY
    assert np.all(rig.velocity == -101) and np.all(rig.gradient == -102)
    assert rig.backend._fixed_source_key is None


def test_growing_image_scratch_never_exceeds_hard_bound(rig, monkeypatch):
    monkeypatch.setattr(device, "_MAX_IMAGE_PAIR_CAPACITY", 100)
    rig.required_pairs = 100
    with pytest.warns(RuntimeWarning, match="image-list storage"):
        _query(rig)
    assert [target.max_pairs for target in rig.targets] == [96, 100]
    assert not np.any(rig.velocity == -101)


def test_image_tile_cap_is_independent_of_user_query_capacity(rig):
    with pytest.raises(ValueError, match="bounded workspace capacity"):
        _query(rig, target_count=rig.backend.max_image_block_targets + 1)
    assert rig.targets == []
    assert rig.backend._fixed_source_key is None


def test_live_memory_diagnostic_includes_target_owned_scratch(rig):
    _query(rig)
    rig.targets[0].estimated_memory_bytes = lambda: 1234
    assert rig.backend._estimate_memory_bytes() == 1334


def test_target_allocation_failure_clears_released_handles(rig):
    _query(rig)
    old = rig.backend._target_workspace
    rig.target_alloc_failure = True
    with pytest.raises(MemoryError, match="target allocation"):
        _query(rig, target_count=4)
    assert old.destroyed
    assert rig.backend._target_workspace is None
    assert getattr(rig.backend, "_prepared_target_key", None) is None
    assert rig.backend._fixed_source_key is None


@pytest.mark.parametrize("operation", ["grow", "rebind"])
def test_source_replacement_destroys_dependent_target_first(rig, operation):
    _query(rig)
    source, target = rig.backend.workspace, rig.backend._target_workspace
    if operation == "grow":
        rig.backend._ensure_workspace(9)
    else:
        rig.backend.bind(rig.physics)
    assert source.destroyed and target.destroyed
    assert rig.events.index(("target_destroyed", target)) < rig.events.index(
        ("source_destroyed", source)
    )
    assert rig.backend._target_workspace is None
    assert getattr(rig.backend, "_prepared_target_key", None) is None
    assert not rig.backend._source_moments_ready


@pytest.mark.parametrize("operation", ["grow", "rebind"])
def test_failed_source_replacement_does_not_retain_destroyed_workspace(rig, operation):
    _query(rig)
    source, target = rig.backend.workspace, rig.backend._target_workspace
    rig.source_alloc_failure = True
    with pytest.raises(MemoryError, match="source allocation"):
        if operation == "grow":
            rig.backend._ensure_workspace(9)
        else:
            rig.backend.bind(rig.physics)
    assert source.destroyed and target.destroyed
    assert rig.backend.workspace is None
    assert rig.backend._target_workspace is None


def _grow_stage_lists(rig):
    capacity = rig.backend.workspace.max_pairs
    error = device._InteractionListCapacityError(
        capacity, m2l=capacity + 1, near=0, queue_a=0, queue_b=0
    )
    with pytest.warns(RuntimeWarning, match="FMM interaction-list storage"):
        rig.backend._grow_interaction_lists(error)


def test_warm_stage_workspace_keeps_sources_targets_and_cache(rig):
    _query(rig)
    source, target = rig.backend.workspace, rig.backend._target_workspace
    events = list(rig.events)

    def forbidden_reclaim():
        raise AssertionError("warm scratch must not discard the mesh cache")

    with rig.backend.stage_workspace(4, reclaim=forbidden_reclaim):
        assert rig.backend._reclaim_stage_cache is forbidden_reclaim
        assert rig.backend.workspace is source
        assert rig.backend._target_workspace is target
    assert getattr(rig.backend, "_reclaim_stage_cache", None) is None
    assert rig.events == events


@pytest.mark.parametrize("growth", ["sources", "lists"])
def test_stage_workspace_reclaims_before_replacing_dependent_fields(rig, growth):
    _query(rig)
    source, target = rig.backend.workspace, rig.backend._target_workspace
    calls = []

    def reclaim():
        assert rig.backend.workspace is source and not source.destroyed
        assert rig.backend._target_workspace is target and not target.destroyed
        calls.append("reclaim")
        rig.events.append(("reclaim", source))

    with rig.backend.stage_workspace(9 if growth == "sources" else 4, reclaim=reclaim):
        if growth == "lists":
            _grow_stage_lists(rig)
        replacement = rig.backend.workspace
        assert replacement is not source
        assert replacement.max_n_particles == (9 if growth == "sources" else 4)
        assert rig.backend._reclaim_stage_cache is reclaim
    assert calls == ["reclaim"]
    assert rig.events.index(("reclaim", source)) < rig.events.index(
        ("target_destroyed", target)
    ) < rig.events.index(("source_destroyed", source)) < rig.events.index(
        ("source_created", replacement)
    )
    assert getattr(rig.backend, "_reclaim_stage_cache", None) is None


@pytest.mark.parametrize("growth", ["sources", "lists"])
def test_failed_cache_reclaim_preserves_fmm_and_target_state(rig, growth):
    _query(rig)
    source, target = rig.backend.workspace, rig.backend._target_workspace
    events = list(rig.events)

    def previous():
        return None

    rig.backend._reclaim_stage_cache = previous
    last_tree_key = rig.backend._last_tree_key
    prepared_key = getattr(rig.backend, "_prepared_target_key", None)
    moments_ready = rig.backend._source_moments_ready

    def reclaim():
        raise ValueError("cache reclaim failed")

    with (
        pytest.raises(ValueError, match="cache reclaim failed"),
        rig.backend.stage_workspace(9 if growth == "sources" else 4, reclaim=reclaim),
    ):
        _grow_stage_lists(rig)
    assert rig.backend._reclaim_stage_cache is previous
    assert rig.backend.workspace is source and not source.destroyed
    assert rig.backend._target_workspace is target and not target.destroyed
    assert rig.backend._last_tree_key == last_tree_key
    assert getattr(rig.backend, "_prepared_target_key", None) == prepared_key
    assert rig.backend._source_moments_ready is moments_ready
    assert rig.events == events


def test_stage_workspace_restores_callback_after_body_failure(rig):
    _query(rig)

    def previous():
        return None

    def reclaim():
        return None

    rig.backend._reclaim_stage_cache = previous
    with (
        pytest.raises(ValueError, match="caller failed"),
        rig.backend.stage_workspace(4, reclaim=reclaim),
    ):
        assert rig.backend._reclaim_stage_cache is reclaim
        raise ValueError("caller failed")
    assert rig.backend._reclaim_stage_cache is previous


def test_stage_workspace_restores_callback_after_allocation_failure(rig):
    _query(rig)

    def previous():
        return None

    rig.backend._reclaim_stage_cache = previous
    calls = []
    rig.source_alloc_failure = True
    with (
        pytest.raises(MemoryError, match="source allocation"),
        rig.backend.stage_workspace(9, reclaim=lambda: calls.append("reclaim")),
    ):
        pytest.fail("failed scope entry must not execute its body")
    assert calls == ["reclaim"]
    assert rig.backend._reclaim_stage_cache is previous
    assert rig.backend.workspace is None and rig.backend._target_workspace is None


def test_nested_stage_workspaces_restore_the_enclosing_reclaim(rig):
    _query(rig)
    calls = []

    def previous():
        calls.append("previous")

    def outer():
        calls.append("outer")

    def inner():
        calls.append("inner")

    rig.backend._reclaim_stage_cache = previous
    with rig.backend.stage_workspace(4, reclaim=outer):
        with rig.backend.stage_workspace(4, reclaim=inner):
            _grow_stage_lists(rig)
            assert rig.backend._reclaim_stage_cache is inner
        assert rig.backend._reclaim_stage_cache is outer
        _grow_stage_lists(rig)
    assert calls == ["inner", "outer"]
    assert rig.backend._reclaim_stage_cache is previous


def test_empty_sources_publish_zero_without_target_allocation(rig):
    _query(rig, source_count=0)
    assert not rig.targets
    assert np.all(rig.velocity == 0) and np.all(rig.gradient == 0)


def test_zero_targets_perform_no_allocation_and_do_not_publish(rig):
    _query(rig, target_count=0)
    assert not rig.targets
    assert rig.backend.workspace.builds == 0
    assert np.all(rig.velocity == -101) and np.all(rig.gradient == -102)


@pytest.mark.parametrize("arch", [device.ti.cpu, device.ti.cuda, device.ti.metal, device.ti.vulkan])
@pytest.mark.parametrize("kernel", ["GAUSSIAN", "WINCKELMANS", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN"])
def test_image_capability_is_conservative_for_runtime_and_kernel(monkeypatch, arch, kernel):
    backend = device.FMMInduction()
    backend.kernel = SimpleNamespace(name=kernel)
    monkeypatch.setattr(device.ti.lang.impl, "get_runtime", lambda: SimpleNamespace(prog=object()))
    monkeypatch.setattr(device.ti.lang.impl, "current_cfg", lambda: SimpleNamespace(arch=arch))
    assert backend.supports_image_blocks == (
        arch in (device.ti.cpu, device.ti.cuda) and kernel in ("GAUSSIAN", "WINCKELMANS")
    )


def test_image_capability_is_false_before_runtime_initialization(monkeypatch):
    backend = device.FMMInduction()
    monkeypatch.setattr(device.ti.lang.impl, "get_runtime", lambda: SimpleNamespace(prog=None))

    def forbidden_configuration_read():
        raise AssertionError("an uninitialized runtime must not be queried for architecture")

    monkeypatch.setattr(device.ti.lang.impl, "current_cfg", forbidden_configuration_read)
    assert backend.supports_image_blocks is False
