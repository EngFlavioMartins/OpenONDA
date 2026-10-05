"""Device-free conditions for the opt-in cylinder GBD component-profiler asset."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_profiler():
    path = (
        Path(__file__).resolve().parents[2] / "tests/support/cylinder/profile_solver_components.py"
    )
    spec = spec_from_file_location("component_profile_test_asset", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Physics:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.batch_calls = 0
        self._allocate_grid = lambda: "instance allocation"

    def gbd_diffusion(self):
        self._allocate_grid()
        return self._gbd_diffusion_impl()

    def _gbd_diffusion_impl(self):
        for _ in range(3):
            self._m4_scatter_gpu_kernel()
        if self.fail:
            raise ValueError("deliberate operator failure")
        return 17

    def _m4_scatter_gpu_kernel(self):
        self.batch_calls += 1


class _VPM:
    def __init__(self):
        self._physics = None
        self.physics_reads = 0
        self.stepper = SimpleNamespace(_apply_grid_diffusion=self._apply_grid_diffusion)
        self.stage_rhs = SimpleNamespace(evaluate=lambda: None, position_guard=None)
        self.induction = SimpleNamespace(evaluate_stage=lambda: None)
        self.output_manager = SimpleNamespace(dispatch=lambda: None)

    @property
    def physics(self):
        self.physics_reads += 1
        if self._physics is None:
            self._physics = _Physics()
        return self._physics

    def _apply_grid_diffusion(self):
        return self.physics.gbd_diffusion()


def _coupler():
    return SimpleNamespace(
        _is_master=True,
        vpm_solver=_VPM(),
        _transfer_vorticity_to_vpm=lambda: None,
    )


def test_gbd_timers_follow_lazy_and_replaced_field_once(monkeypatch):
    module = _load_profiler()
    sync_calls = []
    monkeypatch.setitem(
        __import__("sys").modules, "taichi", SimpleNamespace(sync=lambda: sync_calls.append(1))
    )
    coupler = _coupler()
    vpm = coupler.vpm_solver
    measurements = {}
    original_grid_call = vpm.stepper._apply_grid_diffusion
    with module.profile_components(coupler, measurements):
        assert vpm.physics_reads == 0
        assert vpm.stepper._apply_grid_diffusion() == 17
        first = vpm._physics
        first_wrapped = first.gbd_diffusion
        assert vpm.stepper._apply_grid_diffusion() == 17
        assert first.gbd_diffusion is first_wrapped
        second = _Physics()
        second_instance_allocation = second._allocate_grid
        vpm._physics = second
        assert vpm.stepper._apply_grid_diffusion() == 17
    assert first.batch_calls == 6
    assert second.batch_calls == 3
    assert vpm.stepper._apply_grid_diffusion is original_grid_call
    for solver in (first, second):
        assert "gbd_diffusion" not in vars(solver)
        assert "_gbd_diffusion_impl" not in vars(solver)
        assert "_m4_scatter_gpu_kernel" not in vars(solver)
    assert second._allocate_grid is second_instance_allocation
    assert measurements["gbd.gbd_diffusion"]["calls"] == 3
    assert measurements["gbd._gbd_diffusion_impl"]["calls"] == 3
    assert measurements["gbd._m4_scatter_gpu_kernel"]["calls"] == 9
    assert measurements["gbd._allocate_grid"]["calls"] == 3
    assert measurements["vpm._apply_grid_diffusion"]["calls"] == 3
    assert sync_calls
    assert all(row["seconds"] >= 0 for row in measurements.values())


def test_gbd_timers_restore_on_operator_exception(monkeypatch):
    module = _load_profiler()
    monkeypatch.setitem(__import__("sys").modules, "taichi", SimpleNamespace(sync=lambda: None))
    coupler = _coupler()
    physics = _Physics(fail=True)
    coupler.vpm_solver._physics = physics
    original_allocation = physics._allocate_grid
    measurements = {}
    with (
        pytest.raises(ValueError, match="deliberate operator failure"),
        module.profile_components(coupler, measurements),
    ):
        coupler.vpm_solver.stepper._apply_grid_diffusion()
    assert "gbd_diffusion" not in vars(physics)
    assert physics._allocate_grid is original_allocation
    assert measurements["gbd.gbd_diffusion"]["calls"] == 1
    assert measurements["gbd._m4_scatter_gpu_kernel"]["calls"] == 3


@pytest.mark.parametrize("master,details", [(False, True), (True, False)])
def test_gbd_timers_are_opt_in_and_master_only(monkeypatch, master, details):
    module = _load_profiler()
    monkeypatch.setitem(__import__("sys").modules, "taichi", SimpleNamespace(sync=lambda: None))
    coupler = _coupler()
    coupler._is_master = master
    measurements = {}
    with module.profile_components(coupler, measurements, gbd_detail=details):
        assert coupler.vpm_solver.stepper._apply_grid_diffusion() == 17
    assert not any(key.startswith("gbd.") for key in measurements)
    assert bool(measurements) is master


@pytest.mark.parametrize("fail", [False, True])
def test_private_diffusion_timers_restore_without_solver(monkeypatch, fail):
    module = _load_profiler()
    monkeypatch.setitem(__import__("sys").modules, "taichi", SimpleNamespace(sync=lambda: None))
    physics = _Physics(fail=fail)
    allocation = physics._allocate_grid
    measurements = {}
    try:
        with module.profile_grid_diffusion(physics, measurements):
            physics.gbd_diffusion()
    except ValueError:
        assert fail
    assert "gbd_diffusion" not in vars(physics)
    assert physics._allocate_grid is allocation
    assert measurements["gbd._m4_scatter_gpu_kernel"]["calls"] == 3
    assert measurements["gbd.gbd_diffusion"]["calls"] == 1
