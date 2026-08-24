"""Cube-flow timing and output contracts."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[2] / "tutorials" / "coupled_fvm_vpm" / "cube_flow"


def _load_setup(path: Path, module_name: str):
    spec = spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cube_flow_uses_one_exact_sampling_cadence_and_native_substeps():
    assert not (CASE_DIR / "cube_flow_timing.py").exists()
    setup = _load_setup(CASE_DIR / "cube_flow_setup.py", "cube_flow_setup_contract")

    assert setup.VPM_TIME_STEP_SIZE / setup.FVM_TIME_STEP_SIZE == 2
    assert setup.SAMPLING_INTERVAL_TIME / setup.VPM_TIME_STEP_SIZE == 5
    assert setup.CHECKPOINT_INTERVAL_TIME / setup.VPM_TIME_STEP_SIZE == 100
    assert setup.END_TIME / setup.VPM_TIME_STEP_SIZE == 2000

    assert all(
        sampler.schedule.every_time == setup.SAMPLING_INTERVAL_TIME
        and sampler.schedule.every_n_steps is None
        for sampler in setup.FVM_SAMPLERS
    )
    assert all(
        sampler.schedule.every_n_steps == setup.VPM_SAMPLING_INTERVAL_STEPS
        for sampler in setup.VPM_SAMPLERS
    )
    assert setup.FVM_SETUP.time.output_interval_time == setup.CHECKPOINT_INTERVAL_TIME
    assert setup.VPM_SETUP.checkpoint_interval_steps == setup.VPM_CHECKPOINT_INTERVAL_STEPS
    assert setup.COUPLER_SETUP.checkpoint_interval_steps == setup.VPM_CHECKPOINT_INTERVAL_STEPS


def test_reference_flow_uses_the_same_sampling_and_checkpoint_cadence():
    reference = _load_setup(
        CASE_DIR / "reference_flow" / "reference_flow_setup.py",
        "reference_flow_setup_contract",
    )

    assert pytest.approx(0.010) == reference.FVM_TIME_STEP_SIZE
    assert pytest.approx(0.050) == reference.SAMPLING_INTERVAL_TIME
    assert all(
        sampler.schedule.every_time == reference.SAMPLING_INTERVAL_TIME
        for sampler in reference.SAMPLERS
    )
    assert reference.FVM_SETUP.time.output_interval_time == reference.CHECKPOINT_INTERVAL_TIME
