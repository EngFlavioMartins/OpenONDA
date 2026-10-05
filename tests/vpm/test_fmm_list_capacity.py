"""Independent FMM scratch sizing, without a Taichi runtime or device."""

from types import SimpleNamespace

import pytest

from source.solvers.vpm.physics.induction.fmm import device


class _Field:
    def __init__(self, shape):
        self.shape = shape

    def from_numpy(self, values):
        pass


class _Fields:
    def scalar(self, *, dtype, shape):
        return _Field(shape)

    def vector(self, n, *, dtype, shape):
        return _Field(shape)

    def matrix(self, n, m, *, dtype, shape):
        return _Field(shape)

    def finalize(self):
        pass


@pytest.fixture
def make_workspace(monkeypatch):
    monkeypatch.setattr(device, "_DeviceFields", _Fields)
    monkeypatch.setattr(device, "TaichiTreecode", lambda **kwargs: SimpleNamespace(**kwargs))

    def make(**kwargs):
        return device.FMMDeviceWorkspace(4, None, "GAUSSIAN", 1.0, 1.0, 4, **kwargs)

    return make


def test_independent_buffers_keep_near_adjacency_storage(make_workspace):
    workspace = make_workspace(_list_capacities=(7, 23, 11))

    assert workspace.list_capacities == (7, 23, 11)
    assert workspace.max_pairs == 23
    assert workspace.m2l_batch_size == 7
    assert workspace.m2l_target.shape == 7
    assert workspace.m2l_source.shape == 23
    assert workspace.near_target.shape == workspace.near_source.shape == 23
    assert workspace._queue_target_a.shape == workspace._queue_source_a.shape == 11
    assert workspace._queue_target_b.shape == workspace._queue_source_b.shape == 11


def test_equal_capacity_control_and_default_remain_supported(make_workspace):
    assert make_workspace(max_pairs=513).list_capacities == (513, 513, 513)
    assert make_workspace().list_capacities == (128, 128, 128)
    with pytest.raises(ValueError, match="either equal or independent"):
        make_workspace(max_pairs=513, _list_capacities=(7, 23, 11))


@pytest.mark.parametrize("capacities", [(0, 3, 5), (3, -1, 5), (3, 5, 1 << 30), (3, 5)])
def test_invalid_capacity_is_rejected_before_allocation(capacities):
    with pytest.raises(ValueError):
        device._normalize_list_capacities(capacities)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [((100, 0, 0), (125, 80, 96)), ((0, 81, 0), (64, 120, 96)), ((0, 0, 200), (64, 80, 250))],
)
def test_growth_changes_only_overflowing_group(monkeypatch, counts, expected):
    induction = device.FMMInduction()
    induction.workspace = SimpleNamespace(max_n_particles=1, list_capacities=(64, 80, 96))
    replacements = []
    monkeypatch.setattr(induction, "_replace_workspace", lambda *args: replacements.append(args))
    error = device._InteractionListCapacityError(
        (64, 80, 96), m2l=counts[0], near=counts[1], queue_a=0, queue_b=0, queue_peak=counts[2]
    )

    with pytest.warns(RuntimeWarning, match="Growing FMM interaction-list storage"):
        induction._grow_interaction_lists(error)

    assert replacements == [(1, expected)]
    assert error.counts == counts
    assert induction.diagnostics.interaction_list_resizes == 1


def test_source_growth_preserves_independent_list_history(monkeypatch):
    induction = device.FMMInduction()
    induction.max_n_particles = 16
    induction.workspace = SimpleNamespace(max_n_particles=4, list_capacities=(513, 128, 400))
    replacements = []
    monkeypatch.setattr(induction, "_replace_workspace", lambda *args: replacements.append(args))

    induction._ensure_workspace(5)

    assert replacements == [(8, (513, 256, 400))]


def test_impossible_group_growth_never_replaces_live_field(monkeypatch):
    induction = device.FMMInduction()
    capacities = (64, 80, device._MAX_PAIR_CAPACITY)
    induction.workspace = SimpleNamespace(max_n_particles=1, list_capacities=capacities)
    replacements = []
    monkeypatch.setattr(induction, "_replace_workspace", lambda *args: replacements.append(args))
    error = device._InteractionListCapacityError(
        capacities, m2l=0, near=0, queue_a=0, queue_b=0, queue_peak=1 << 30
    )

    with pytest.raises(RuntimeError, match="safe i32 capacity"):
        induction._grow_interaction_lists(error)

    assert not replacements
    assert induction.diagnostics.interaction_list_resizes == 0


@pytest.mark.parametrize("capacities", [(7, 23, 11), (65536, 32768, 32768)])
def test_memory_estimate_counts_each_group_and_shared_near_scratch(capacities):
    induction = device.FMMInduction()
    equal = max(capacities)
    baseline = induction.estimated_workspace_bytes(64, max_pairs=equal)
    separate = induction.estimated_workspace_bytes(64, _list_capacities=capacities)
    pairs = device._normalize_list_capacities(capacities)
    pair_reduction = 8 * equal * 4 - pairs.storage_bytes
    derivative_reduction = (
        (min(device._M2L_BATCH_SIZE, equal) - min(device._M2L_BATCH_SIZE, pairs.m2l))
        * device._DERIVATIVE_COUNT
        * 4
    )

    assert baseline - separate == pair_reduction + derivative_reduction


def test_runtime_memory_estimate_uses_actual_separate_capacities():
    induction = device.FMMInduction()
    induction.workspace = SimpleNamespace(
        max_n_particles=64,
        target_batch_capacity=4,
        list_capacities=device._ListCapacities(65536, 32768, 32768),
        tree=SimpleNamespace(max_stack_depth=48),
    )

    assert induction._estimate_memory_bytes() == induction.estimated_workspace_bytes(
        64, 4, _list_capacities=induction.workspace.list_capacities
    )
