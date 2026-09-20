"""Retained field frames and coupled checkpoints share accepted physical times."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from source.coupler.interface_iteration import validate_output_schedules


@pytest.mark.parametrize(
    ("name", "end_time", "interval", "expected_steps"),
    [("cube", 30.0, 0.25, 5), ("cylinder", 100.0, 0.24, 6)],
)
def test_study_frames_align_with_checkpoint_and_exchange(
    name, end_time, interval, expected_steps, tmp_path
):
    module = importlib.import_module(f"studies.panel_removal.run_{name}")
    result = (
        module.configuration(tmp_path, False) if name == "cube" else module.configuration(tmp_path)
    )
    fvm, vpm, coupling = result[:3]
    macro_dt = vpm.numerics.time_step_size
    substeps = round(macro_dt / fvm.time.time_step_size)
    assert fvm.time.end_time == end_time
    assert vpm.run.steps * macro_dt == pytest.approx(end_time)
    assert coupling.backup_interval_steps == expected_steps
    assert fvm.time.output_schedule.every_time == pytest.approx(interval)
    assert not fvm.time.output_schedule.final_only
    validate_output_schedules(
        SimpleNamespace(
            fvm_solver=SimpleNamespace(
                _output_schedule=fvm.time.output_schedule,
                _backup_config=fvm.backup,
                _sampler_schedules={str(i): s.schedule for i, s in enumerate(fvm.samplers)},
                _samplers=fvm.samplers,
            ),
            n_fvm_substeps=substeps,
            vpm_time_step_size=macro_dt,
        )
    )
    # Exercise the real schedule at accepted endpoints, including intervals
    # where provisional FVM substeps must not publish a frame.
    written = [
        step
        for step in range(1, 3 * expected_steps + 1)
        if fvm.time.output_schedule.is_due(
            step * substeps, step * macro_dt, fvm.time.time_step_size
        )
    ]
    assert written == [expected_steps, 2 * expected_steps, 3 * expected_steps]
    assert vpm.numerics.panel_solver is None and not vpm.numerics.bodies


@pytest.mark.parametrize(
    "name,interval", [("cube", 0.24), ("cylinder", 0.25), ("cylinder", np.nan)]
)
def test_provisional_snapshot_times_are_rejected(name, interval, tmp_path):
    module = importlib.import_module(f"studies.panel_removal.run_{name}")
    args = (tmp_path, False) if name == "cube" else (tmp_path,)
    with pytest.raises(ValueError, match="snapshot_interval"):
        module.configuration(*args, snapshot_interval=interval)


def test_ordinary_coupled_tutorials_keep_their_panel_formulation():
    for case, end, interval in (
        ("02_cube_flow", 30.0, 0.25),
        ("01_cylinder_shedding_flow", 100.0, 0.24),
    ):
        module = importlib.import_module(f"tutorials.coupled_fvm_vpm.{case}.setup")
        assert end == module.END_TIME
        assert module.VPM_CASE.numerics.panel_solver is not None
        assert (
            module.FVM_SETUP.time.output_schedule.every_n_steps * module.FVM_TIME_STEP_SIZE
            == pytest.approx(interval)
        )
        assert (
            module.COUPLER_SETUP.backup_interval_steps * module.VPM_TIME_STEP_SIZE
            == pytest.approx(interval)
        )
