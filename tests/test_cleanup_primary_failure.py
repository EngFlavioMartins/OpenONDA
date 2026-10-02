"""Primary solver failures survive secondary cleanup errors without a GPU."""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from source.coupler import solver as coupler_module
from source.solvers.vpm.core import solver as vpm_module
from source.solvers.vpm.physics.induction.slip_slab import SlipSlabInduction


def _vpm_owner(monkeypatch, *, mesh_failure=None):
    events = []
    slab = SlipSlabInduction.__new__(SlipSlabInduction)

    def close_mesh():
        events.append("mesh")
        if mesh_failure is not None:
            raise mesh_failure

    slab.close_mesh_session = close_mesh
    owner = SimpleNamespace(
        _closed=False, induction=slab, _run_started=True,
        _initial_conditions_built=False, _backend_claimed=True,
        _particle_snapshot_buffers={},
        stage_rhs=SimpleNamespace(close=lambda: events.append("stage")),
        _restore_output_streams=lambda: events.append("streams"),
    )
    monkeypatch.setattr(vpm_module, "reset_taichi_backend",
                        lambda **_: events.append("reset"))
    return owner, events


def test_vpm_preserves_primary_allocation_failure_and_blocks_uncertain_reset(monkeypatch):
    cleanup = RuntimeError("mesh drain remains uncertain")
    owner, events = _vpm_owner(monkeypatch, mesh_failure=cleanup)
    primary = MemoryError("original field allocation failed")
    vpm_module.VPMSolver.close(owner, failure=primary)
    assert owner._mesh_cleanup_failure is cleanup
    assert owner._backend_claimed and not owner._closed
    assert events == ["mesh", "stage", "streams"]
    assert any("mesh drain remains uncertain" in note for note in primary.__notes__)
    # A later explicit close still reports the uncertain resource, with no
    # second drain or unsafe runtime reset after the original unwind.
    with pytest.raises(RuntimeError) as captured:
        vpm_module.VPMSolver.close(owner)
    assert captured.value is cleanup and "reset" not in events
    assert events.count("mesh") == 1


def test_vpm_normal_cleanup_releases_runtime_during_primary_unwind(monkeypatch):
    owner, events = _vpm_owner(monkeypatch)
    primary = RuntimeError("original evolution failed")
    vpm_module.VPMSolver.close(owner, failure=primary)
    assert owner._closed and not owner._backend_claimed
    assert events == ["mesh", "stage", "streams", "reset"]
    assert not getattr(primary, "__notes__", ())


def test_vpm_restart_start_preserves_primary_failure(monkeypatch):
    primary = ValueError("invalid restart input")
    seen = []

    def fail_restart(_):
        raise primary

    def close(*, failure=None):
        seen.append(failure)
        failure.add_note("injected cleanup failure")

    owner = SimpleNamespace(_run_started=False, _start_run_from=fail_restart, close=close)
    with pytest.raises(ValueError, match="invalid restart input") as captured:
        vpm_module.VPMSolver.run(owner, start_from="checkpoint")
    assert captured.value is primary and seen == [primary]


def _coupled_owner(monkeypatch, *, resource_failure, log_failure=None):
    events = []
    driver = coupler_module.FVMVPMCoupler.__new__(coupler_module.FVMVPMCoupler)
    driver._closed = False
    driver._owned_resources = ExitStack()
    driver._owned_resources.callback(lambda: events.append("remaining resource"))

    def close_resource():
        events.append("failed resource")
        raise resource_failure

    def close_log():
        events.append("log close")
        if log_failure is not None:
            raise log_failure

    driver._owned_resources.callback(close_resource)
    driver._log_handler = SimpleNamespace(close=close_log)
    monkeypatch.setattr(coupler_module.logger, "removeHandler",
                        lambda _: events.append("log remove"))
    return driver, events


def test_coupled_context_preserves_primary_and_attempts_all_cleanup(monkeypatch):
    cleanup = RuntimeError("VPM drain remains uncertain")
    log_cleanup = OSError("log close failed")
    driver, events = _coupled_owner(monkeypatch, resource_failure=cleanup,
                                    log_failure=log_cleanup)
    primary = MemoryError("original field allocation failed")
    with pytest.raises(MemoryError) as captured, driver:
        raise primary
    assert captured.value is primary and driver._closed
    assert events == ["failed resource", "remaining resource", "log remove", "log close"]
    assert any("VPM drain remains uncertain" in note for note in primary.__notes__)
    assert any("log close failed" in note for note in primary.__notes__)
    driver.close()
    assert len(events) == 4


def test_coupled_normal_close_propagates_resource_failure_and_keeps_log_evidence(monkeypatch):
    cleanup = RuntimeError("VPM drain remains uncertain")
    driver, events = _coupled_owner(monkeypatch, resource_failure=cleanup,
                                    log_failure=OSError("log close failed"))
    with pytest.raises(RuntimeError) as captured:
        driver.close()
    assert captured.value is cleanup and driver._closed
    assert events == ["failed resource", "remaining resource", "log remove", "log close"]
    assert any("log close failed" in note for note in cleanup.__notes__)
