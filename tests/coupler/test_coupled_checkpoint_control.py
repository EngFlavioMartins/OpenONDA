"""Verification controls retain the native cylinder physics and accepted history."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openonda.tutorial_runner import load_case_module
from source.solvers.fvm.io.backup import config_hash
from source.solvers.fvm.sampling.forces import FORCES_HEADER, ForceSampler
from tests.support.cylinder.run_coupled_checkpoint_control import (
    HISTORY_FIELDS,
    json_value,
    old_target_predictor_refresh,
    quiet_setup,
    read_force_history,
)

CASE = Path(__file__).resolve().parents[2] / "tutorials/coupled_fvm_vpm/01_cylinder_shedding_flow"


def test_native_force_history_retains_patch_text_and_finite_numeric_fields(tmp_path):
    context = SimpleNamespace(time=46.04, step=5755, _accepted_time_step_size=0.008)
    sampler = ForceSampler(file_name="forces_history")
    forces = {
        "cylinder": {
            "pressure_force": [0.47, -0.13, 0.0],
            "viscous_force": [0.14, -0.007, 0.0],
            "total_force": [0.61, -0.137, 0.0],
            "moment": [0.0, 0.0, 0.002],
            "coeffs": {
                "drag_coefficient": 1.22,
                "lift_coefficient": -0.274,
                "side_force_coefficient": 0.0,
                "pitching_moment_coefficient": 0.004,
            },
        }
    }
    sampler.write_csv(context, str(tmp_path), forces)
    path = tmp_path / "forces_history.csv"
    values = read_force_history(path)
    assert values.dtype.names == tuple(FORCES_HEADER)
    assert values["patch"].tolist() == ["cylinder"]
    assert values["pressure_force_x"].tolist() == [0.47]
    rows = [{name: json_value(row[name]) for name in values.dtype.names} for row in values]
    assert json.loads(json.dumps(rows, allow_nan=False))[0]["patch"] == "cylinder"
    path.write_text(path.read_text().replace("0.47", "nan", 1))
    with pytest.raises(ValueError, match="Non-finite.*pressure_force_x"):
        read_force_history(path)


def test_private_output_reduction_preserves_native_equation_configuration():
    flow, particles, coupling, _mesh = load_case_module(CASE).build_case()
    reduced = quiet_setup(flow)
    assert config_hash(reduced) == config_hash(flow)
    assert len(reduced.samplers) == 1
    assert reduced.samplers[0] is flow.samplers[0]
    assert reduced.backup == flow.backup
    assert particles.numerics.max_n_particles == 200000
    assert particles.numerics.compute_device == "AUTO"
    assert coupling.interface_iterations == 6
    assert coupling.interface_normal_tolerance == coupling.interface_gradient_tolerance == 1e-5
    assert coupling.backup_interval_steps == 25


@pytest.mark.parametrize("fail", [False, True])
def test_optional_old_target_wrapper_preserves_step_history_and_instance_method(tmp_path, fail):
    position = np.array([[0.6, 0, 0], [1.0, 0, 0]])
    vpm = SimpleNamespace(
        particle_position=position,
        particle_vortex_strength=np.array([[0, 0, 1.0], [0, 0, -1.0]]),
        time=46.0,
        step=1150,
    )
    transfer = SimpleNamespace(step=1151)
    calls = []
    owner = SimpleNamespace(
        _is_master=True,
        _comm=None,
        vpm_solver=vpm,
        fvm_solver=SimpleNamespace(time=46.0),
        vorticity_transfer=transfer,
        fvm_box=np.array([-1.5, 2.5, -1.5, 1.5, -0.5, 0.5]),
    )
    for name in HISTORY_FIELDS:
        setattr(owner, name, np.array([1.0, 2.0]))
    owner.apply_vpm = lambda callback, *args: callback(vpm, *args)

    def advance(step, end):
        calls.append(("native_advance", step, end))
        vpm.time = end
        vpm.step = step
        return 0.1

    def renew(*geometry):
        calls.append(("old_target_renewal", owner.fvm_solver.time))
        transfer.step += 1
        if fail:
            raise RuntimeError("native transfer guard failed")
        return {"native_guard_passed": True}, 0.2

    owner._advance_vpm = advance
    owner._transfer_vorticity_to_vpm = renew
    history = {name: getattr(owner, name).copy() for name in HISTORY_FIELDS}
    report = {"predictor_phases": []}
    geometry = (np.zeros((2, 3)), np.ones((2, 3)), np.ones(2))
    if fail:
        with (
            pytest.raises(RuntimeError, match="native transfer guard"),
            old_target_predictor_refresh(owner, geometry, report, tmp_path / "report.json"),
        ):
            owner._advance_vpm(1151, 46.04)
    else:
        with old_target_predictor_refresh(owner, geometry, report, tmp_path / "report.json"):
            owner._advance_vpm(1151, 46.04)
        assert report["predictor_phases"][0]["accepted_old_boundary_history_preserved"]
    assert owner._advance_vpm is advance
    assert transfer.step == 1151
    assert owner.fvm_solver.time == 46.0
    for name, values in history.items():
        np.testing.assert_array_equal(getattr(owner, name), values)
    assert calls == [("native_advance", 1151, 46.04), ("old_target_renewal", 46.0)]
