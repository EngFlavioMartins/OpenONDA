"""Live-memory admission with real source-index ownership and FFT budgets."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh import fields
from source.solvers.vpm.physics.induction.gaussian_mesh.correction import GaussianCoreCorrectionGPU
from source.solvers.vpm.physics.induction.gaussian_mesh.planning import (
    correction_query_reserve,
    device_field_execution_plan,
    field_execution_plan,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.runtime import FFTPlanPair


@pytest.mark.parametrize("itemsize", [4, 8])
def test_correction_reservation_covers_rounded_query_storage(itemsize):
    for count in (1, 17, 2665):
        reserve = correction_query_reserve(count, itemsize)
        required = 8192 + count * (24 + 12 * itemsize + 12)
        assert required <= reserve < required + 512
        assert reserve % 512 == 0
    assert correction_query_reserve(1, itemsize) == 8704


@pytest.mark.parametrize("count,size", [(0, 4), (-1, 8), (True, 4), (1, 2), (1.0, 8)])
def test_correction_reservation_rejects_invalid_dimensions(count, size):
    with pytest.raises(ValueError):
        correction_query_reserve(count, size)


def test_actual_index_free_memory_does_not_reserve_unused_correction_cap():
    options = ((20, 20, 54), (20, 20, 54), 1, 1, 10, 4)
    smooth, workspace, free = 2 * 1024**3, 128 * 1024**2, 372 * 1024**2
    with pytest.raises(MemoryError, match="insufficient free device memory"):
        device_field_execution_plan(*options, smooth, workspace, 256 * 1024**2, free)
    plan, cap = device_field_execution_plan(
        *options, smooth, workspace, correction_query_reserve(1, 4), free
    )
    assert plan.payload_bytes + workspace <= cap
    assert cap == free - 8704


def _field_runtime(monkeypatch, *, free, work=(2048, 4096), correction_failure=None):
    state = SimpleNamespace(
        events=[], index=4096, idle=8192, correction_live=0, workspace_live=0,
        correction=None, plans=None,
    )

    class Pool:
        def __init__(self, cap):
            self.cap = cap

        def set_limit(self, *, size):
            self.cap = size

        def total_bytes(self):
            return state.workspace_live

        def free_all_blocks(self):
            state.events.append("smooth_idle_release")

    class Owner:
        def __init__(self, cap):
            self.pool = Pool(cap)
            self.stream = SimpleNamespace(synchronize=lambda: None)
            self.cp = SimpleNamespace(
                cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=self.memory)),
                RawModule=lambda **kwargs: None,
            )

        def memory(self):
            state.events.append("memory_measurement")
            return free - state.correction_live - state.workspace_live, 6 * 1024**3

        def admit(self):
            pass

        def allocation_scope(self):
            return nullcontext()

        def close(self):
            state.events.append("smooth_close")

    class Correction:
        def __init__(self, *args, accumulation_dtype, max_scratch_bytes, **kwargs):
            state.correction = self
            self.dtype = np.dtype(accumulation_dtype)
            self.cap = max_scratch_bytes
            self.pool = SimpleNamespace(total_bytes=lambda: state.correction_live)
            state.correction_live = state.index + state.idle
            state.events.append("correction_build")

        def release_build_scratch(self):
            state.correction_live -= state.idle
            state.events.append("correction_idle_release")

        def close(self):
            state.events.append("correction_close")
            if correction_failure:
                raise correction_failure
            state.correction_live = 0

    class Plans:
        def __init__(self, owner, shape, dtype, cap, *, single_workspace):
            state.events.append("plan_build")
            state.plans = self
            self.single_workspace = single_workspace
            self.cap = cap
            state.workspace_live = work[0]

        def measure_directional_work(self):
            state.events.append("plan_measurement")
            state.workspace_live = work[1]
            if max(work) > self.cap:
                raise MemoryError("explicit FFT workspaces exceed plan cap")
            return work

        def retain_both_directions(self):
            self.single_workspace = False
            state.workspace_live = sum(work)
            state.events.append("retain_both")

        def close(self):
            state.workspace_live = 0
            state.events.append("plan_close")

    monkeypatch.setattr(fields, "DeviceOwner", Owner)
    monkeypatch.setattr(fields, "GaussianCoreCorrectionGPU", Correction)
    monkeypatch.setattr(fields, "FFTPlanPair", Plans)
    monkeypatch.setattr(fields, "cardinal_stencil_gpu", lambda *args, **kwargs: (None, None, {}))
    return state


def _case(**kwargs):
    return fields.GaussianImageFields(
        [[0.0, 0.0, 0.1]], [[1.0, 0.0, 0.0]], [0.1], [[0.2, 0.0, 0.1]],
        zmin=0.0, zmax=0.2, tau=0.2, spacing=0.1, cutoff=0.6, **kwargs,
    )


def test_correction_built_and_drained_before_live_admission_without_cap_shrink(monkeypatch):
    state = _field_runtime(monkeypatch, free=372 * 1024**2)
    with _case() as owner:
        assert state.events[:3] == [
            "correction_build", "correction_idle_release", "memory_measurement"
        ]
        report = owner.device_admission
        assert report["free_device_bytes"] == 372 * 1024**2 - state.index
        assert report["correction_owned_bytes"] == state.index
        assert report["correction_reserve_bytes"] == 8704
        assert report["configured_correction_pool_cap"] == state.correction.cap == 256 * 1024**2
        assert owner.effective_smooth_pool_cap == report["free_device_bytes"] - 8704
        assert "plan_build" not in state.events  # Preserve ordinary deferred plan creation.
    assert state.correction_live == 0


def test_measured_work_recovers_below_configured_workspace_cap_without_double_count(monkeypatch):
    state = _field_runtime(monkeypatch, free=16 * 1024**2)
    with _case() as owner:
        assert owner.execution_plan.mode == "all_channels"
        assert state.plans.single_workspace is False
        assert owner.device_admission["workspace_reserve_bytes"] == 6144
        assert owner.device_admission["measured_forward_workspace_bytes"] == 2048
        assert owner.effective_smooth_pool_cap == 16 * 1024**2 - state.index - 8704
        assert owner.execution_plan.payload_bytes + 6144 <= owner.effective_smooth_pool_cap
        assert state.events.count("memory_measurement") == 1
    assert state.workspace_live == state.correction_live == 0


def test_workspace_sum_not_peak_governs_simultaneous_plans(monkeypatch):
    # Each direction fits the unchanged cap, but their sum does not.
    state = _field_runtime(monkeypatch, free=7 * 1024**2, work=(5 * 1024**2, 5 * 1024**2))
    with _case(max_plan_bytes=8 * 1024**2) as owner:
        # This free budget must reject the conservative reserve to exercise
        # measurement, rather than the ordinary retained scheduling branch.
        assert "plan_measurement" in state.events
        assert owner.execution_plan.mode == "streamed"
        assert state.plans.single_workspace
        assert owner.device_admission["workspace_reserve_bytes"] == 5 * 1024**2
        assert "retain_both" not in state.events


def test_impossible_array_payload_rejects_without_fft_probe_and_cleans_index(monkeypatch):
    state_free = 32768
    state = _field_runtime(monkeypatch, free=state_free)
    with pytest.raises(MemoryError, match="insufficient free device memory"):
        _case()
    assert state_free > state.index + 8704
    assert "plan_build" not in state.events
    assert state.correction_live == state.workspace_live == 0


def test_admission_failure_retains_uncertain_resource_handle(monkeypatch):
    failure = RuntimeError("injected correction drain failure")
    state = _field_runtime(monkeypatch, free=32768, correction_failure=failure)
    owner = fields.GaussianImageFields.__new__(fields.GaussianImageFields)
    with pytest.raises(MemoryError) as error:
        owner.__init__(
            [[0.0, 0.0, 0.1]], [[1.0, 0.0, 0.0]], [0.1], [[0.2, 0.0, 0.1]],
            zmin=0.0, zmax=0.2, tau=0.2, spacing=0.1, cutoff=0.6,
        )
    assert error.value.__cause__ is failure
    assert owner._correction is state.correction
    with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
        owner.close()


def test_correction_close_cannot_forget_a_failed_drain():
    correction = GaussianCoreCorrectionGPU.__new__(GaussianCoreCorrectionGPU)
    correction.closed, correction._cleanup_failure = False, None
    retained = object()
    correction._owned = [retained]
    failure = RuntimeError("injected stream failure")
    correction._owner = SimpleNamespace(admit=lambda: None)
    correction.stream = SimpleNamespace(synchronize=lambda: (_ for _ in ()).throw(failure))
    with pytest.raises(RuntimeError, match="injected stream failure"):
        correction.close()
    assert correction._owned == [retained]
    with pytest.raises(RuntimeError, match="cleanup remains uncertain"):
        correction.close()


def test_directional_probe_releases_only_retired_owned_workspace():
    plan = FFTPlanPair.__new__(FFTPlanPair)
    events = []
    plan.single_workspace, plan._release_idle_workspaces = True, False
    plan.forward, plan.inverse, plan.closed = None, None, False
    plan.shape, plan.spectrum_shape = (20, 20, 54), (20, 20, 28)
    plan.dtype = np.dtype("float32")
    plan.max_plan_bytes, plan.work_bytes, plan.peak_work_bytes = 8192, 0, 0
    plan.plan_builds, plan.plan_build_seconds = 0, 0.0
    plan.owner = SimpleNamespace(
        stream=SimpleNamespace(synchronize=lambda: events.append("drain")),
        pool=SimpleNamespace(free_all_blocks=lambda: events.append("owned_idle_release")),
        allocation_scope=nullcontext, admit=lambda: None,
    )

    def make_plan(*args):
        events.append("forward" if args[7] == 1 else "inverse")
        return SimpleNamespace(work_area=SimpleNamespace(mem=SimpleNamespace(
            size=2048 if args[7] == 1 else 4096
        )))

    plan.cufft = SimpleNamespace(CUFFT_R2C=1, CUFFT_C2R=2, PlanNd=make_plan)
    assert plan.measure_directional_work() == (2048, 4096)
    assert events == ["drain", "owned_idle_release", "forward", "drain", "owned_idle_release", "inverse"]
    assert plan.forward is None and plan.inverse is not None
    plan.retain_both_directions()
    assert plan.work_bytes == 6144 and events[-1] == "forward"
    plan.close()


def test_peak_only_workspace_planner_never_schedules_two_simultaneous_directions():
    options = ((20, 20, 54), (20, 20, 54), 1, 1, 10, 4)
    plan = field_execution_plan(*options, 16 * 1024**2, 4096, allow_all_channels=False)
    assert plan.mode == "streamed"
