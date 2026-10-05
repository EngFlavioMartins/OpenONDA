"""Real VPM HDF5 preflight precedes any coupled FVM/history mutation.

FVM is a recording test double; VPM writes, inspects and loads its native
format. The production validation function is never replaced by a fake here.
All files are private tmp_path checkpoints/histories, not tutorial outputs.
"""

from copy import deepcopy
import hashlib
import json
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from source.coupler.backup import (
    _validate_coupled_configuration,
    checkpoint_path_hash,
    config_mapping_digest,
    load_coupled_backup,
    save_coupled_backup,
)
from source.coupler.solver import FVMVPMCoupler
from source.solvers.vpm import DirectInduction
from source.solvers.vpm.config.restart_changes import MISSING_CONFIGURATION_VALUE
from source.solvers.vpm.io.backup import _BackupIO
from tests.coupler.test_coupled_backup import _make_coupler
from tests.vpm.test_backup_storage import _add_counter_rotating_pair, _solver


@pytest.fixture(scope="module")
def native_coupled_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-coupled-validation")
    writer = _solver(
        directory / "writer", induction=DirectInduction(stretching_scheme="TRANSPOSED")
    )
    try:
        _add_counter_rotating_pair(writer)
        coupler = _make_coupler()
        coupler.vpm_solver = writer
        coupler.fvm_solver.time = 0.0
        coupler.fvm_solver.step = 0
        backup = save_coupled_backup(coupler, directory / "backups", coupling_step=0)
        yield backup, writer
    finally:
        writer.close()


class BroadcastChannel:
    """Deterministic two-rank message channel, no production-path shortcuts."""

    def __init__(self, messages=None):
        self.messages = [] if messages is None else messages
        self.index = 0

    def Get_size(self):
        return 2

    def bcast(self, value, root):
        assert root == 0
        if value is not None:
            self.messages.append(deepcopy(value))
            return value
        message = self.messages[self.index]
        self.index += 1
        return message


def _reader(writer, tmp_path, monkeypatch):
    coupler = _make_coupler()
    coupler.vpm_solver = writer
    coupler.fvm_solver.step, coupler.fvm_solver.time = 999, 99.0
    coupler.solution_dir = tmp_path
    coupler.coupling_diagnostics = [{"step": 999, "time": 99.0}]
    history = tmp_path / "coupler_diagnostics.jsonl"
    history.write_text('{"step":999,"time":99.}\n', encoding="utf-8")
    monkeypatch.setattr(
        coupler.fvm_solver, "load_state", lambda *args: pytest.fail("premature FVM load")
    )
    monkeypatch.setattr(
        writer, "_load_backup_from", lambda *args, **kwargs: pytest.fail("premature VPM load")
    )
    return coupler, history


def _copied_bundle(backup, target):
    shutil.copytree(backup, target)
    checkpoint_info = json.loads((target / "checkpoint_info.json").read_text(encoding="utf-8"))
    return checkpoint_info, target / checkpoint_info["checkpoint_files"]["vpm"]


def _publish_test_checkpoint_info(target, checkpoint_info):
    checkpoint_info["config_sha256"] = config_mapping_digest(checkpoint_info["config"])
    checkpoint_info["file_sha256"] = {
        name: checkpoint_path_hash(target / filename)
        for name, filename in checkpoint_info["checkpoint_files"].items()
    }
    (target / "checkpoint_info.json").write_text(json.dumps(checkpoint_info), encoding="utf-8")


@pytest.mark.parametrize(
    "defect",
    [
        "core",
        "configuration_hash",
        "checkpoint_info_configuration",
        "checkpoint_info_step",
        "checkpoint_info_time",
        "checkpoint_info_subcycles",
        "dt_attribute",
    ],
)
def test_real_native_validation_failure_is_broadcast_before_fields_or_history(
    native_coupled_checkpoint, tmp_path, monkeypatch, defect
):
    backup, writer = native_coupled_checkpoint
    target = tmp_path / "case-copy"
    checkpoint_info, hdf5 = _copied_bundle(backup, target)
    with h5py.File(hdf5, "r+") as archive:
        if defect == "core":
            archive["particles/core_radius"][0] = 0.0
        elif defect == "configuration_hash":
            archive["solver"].attrs["numerical_configuration_sha256"] = "0" * 64
        elif defect == "checkpoint_info_configuration":
            # Each configuration is verified by its configuration hash and operationally
            # compatible, but they do not describe the same saved config.
            config = json.loads(archive["solver"].attrs["numerical_configuration"])
            config["compute_device"] = "CPU"
            encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
            archive["solver"].attrs["numerical_configuration"] = encoded
            archive["solver"].attrs["numerical_configuration_sha256"] = hashlib.sha256(
                encoded.encode()
            ).hexdigest()
        elif defect == "dt_attribute":
            archive["solver"].attrs["time_step_size"] = 0.02
    if defect == "checkpoint_info_step":
        checkpoint_info["coupling_step"] = 1
    elif defect == "checkpoint_info_time":
        checkpoint_info["time"] = 0.1
    elif defect == "checkpoint_info_subcycles":
        checkpoint_info["n_fvm_substeps"] = 3
    _publish_test_checkpoint_info(target, checkpoint_info)
    coupler, history = _reader(writer, tmp_path, monkeypatch)
    before = writer.particle_position.copy()
    history_before = history.read_bytes()
    root = BroadcastChannel()
    with pytest.raises(ValueError, match="Coupled restart validation failed") as error:
        load_coupled_backup(coupler, target, comm=root)
    assert len(root.messages) == 1 and str(error.value) == root.messages[0][0]
    assert (coupler.fvm_solver.step, coupler.fvm_solver.time) == (999, 99.0)
    assert history.read_bytes() == history_before
    assert coupler.coupling_diagnostics == [{"step": 999, "time": 99.0}]
    np.testing.assert_array_equal(writer.particle_position, before)
    coupler._is_master, coupler.vpm_solver = False, None
    with pytest.raises(ValueError, match="Coupled restart validation failed") as follower_error:
        load_coupled_backup(
            coupler, tmp_path / "not-read-on-follower", comm=BroadcastChannel(root.messages)
        )
    assert str(follower_error.value) == str(error.value)


def test_root_inspection_interrupt_enters_collective_error_channel(
    native_coupled_checkpoint, tmp_path, monkeypatch
):
    backup, writer = native_coupled_checkpoint
    coupler, history = _reader(writer, tmp_path, monkeypatch)
    inspect = _BackupIO.inspect

    def interrupted(*args, **kwargs):
        inspect(*args, **kwargs)  # The actual native reader completes first.
        raise KeyboardInterrupt("injected after native inspection")

    monkeypatch.setattr(_BackupIO, "inspect", interrupted)
    channel = BroadcastChannel()
    with pytest.raises(ValueError, match="KeyboardInterrupt"):
        load_coupled_backup(coupler, backup, comm=channel)
    assert "KeyboardInterrupt" in channel.messages[0][0]
    assert history.read_text() == '{"step":999,"time":99.}\n'


def test_real_allowed_vpm_change_is_forwarded_to_native_reader(native_coupled_checkpoint, tmp_path):
    backup, _writer = native_coupled_checkpoint
    reader = _solver(tmp_path / "reader", induction=DirectInduction(stretching_scheme="DIRECT"))
    try:
        coupler = _make_coupler()
        coupler.vpm_solver = reader
        coupler.interface_predictor = SimpleNamespace(reset=lambda: None)
        coupler._comm = None
        path = "vpm.induction.stretching_scheme"
        with pytest.raises(ValueError, match=path):
            FVMVPMCoupler.load_backup(coupler, backup)
        assert (
            FVMVPMCoupler.load_backup(
                coupler,
                backup,
                allowed_config_differences={path},
                expected_config_differences={path: ("TRANSPOSED", "DIRECT")},
            )
            == 0
        )
        assert (
            reader._restart_details["configuration_changes"][0]["path"]
            == "induction.stretching_scheme"
        )
        assert (reader.step, reader.time, coupler.fvm_solver.step, coupler.fvm_solver.time) == (
            0,
            0.0,
            0,
            0.0,
        )
        assert coupler._restart_loaded
    finally:
        reader.close()


@pytest.mark.parametrize("key", ["compute_device", "max_n_particles"])
def test_vpm_operational_exemptions_never_escape_their_namespace(key):
    old = {"vpm": {}, key: 1}
    new = {"vpm": {}, key: 2}
    with pytest.raises(ValueError, match=key):
        _validate_coupled_configuration(old, new, (), None)


def test_explicit_setting_changes_require_exact_values():
    old = {"vpm": {"induction": {}}, "coupler": {"settings": None}}
    new = {"vpm": {"induction": {"mesh": {"order": 10}}}, "coupler": {"settings": {"mode": "new"}}}
    paths = {"vpm.induction.mesh", "coupler.settings"}
    expectation = {
        "vpm.induction.mesh": (MISSING_CONFIGURATION_VALUE, {"order": 10}),
        "coupler.settings": (None, {"mode": "new"}),
    }
    changes, allowed, forwarded = _validate_coupled_configuration(old, new, paths, expectation)
    assert len(changes) == 2 and allowed == ("induction.mesh",)
    assert forwarded["induction.mesh"] == (MISSING_CONFIGURATION_VALUE, {"order": 10})
    new["coupler"]["settings"]["unrequested"] = True
    with pytest.raises(ValueError, match="expectation mismatch"):
        _validate_coupled_configuration(old, new, paths, expectation)
    assert forwarded["induction.mesh"][1] == {"order": 10}


@pytest.mark.parametrize(
    "permission", ["coupler", "coupler.*", "vpm", "unknown", "vpm.induction.*"]
)
def test_coupled_parent_wildcard_and_unused_grants_reject(permission):
    config = {"vpm": {"induction": {"theta": 0.1}}, "coupler": {"mode": "old"}}
    with pytest.raises(ValueError):
        _validate_coupled_configuration(config, config, {permission}, None)


def test_public_run_forwards_exact_expectations_without_enabling_any_transition(tmp_path):
    calls = []
    path = "vpm.induction.stretching_scheme"
    expected = {path: ("TRANSPOSED", "DIRECT")}

    def load(directory, **kwargs):
        calls.append((directory, kwargs))
        return 7

    holder = SimpleNamespace(
        vorticity_transfer=object(),
        load_backup=load,
        _comm=None,
        _update_freestream=lambda time: None,
        fvm_solver=SimpleNamespace(
            auto_write=False,
            time=0.0,
            _resolved_setup=SimpleNamespace(velocity_boundaries=(), boundaries=()),
        ),
        solve=lambda **kwargs: kwargs["start_step"],
    )
    assert (
        FVMVPMCoupler.run(
            holder,
            restart_from=tmp_path,
            restart_allowed_config_differences={path},
            restart_expected_config_differences=expected,
        )
        == 7
    )
    assert calls == [
        (tmp_path, {"allowed_config_differences": {path}, "expected_config_differences": expected})
    ]
    with pytest.raises(ValueError, match="require restart_from"):
        FVMVPMCoupler.run(holder, restart_expected_config_differences=expected)
