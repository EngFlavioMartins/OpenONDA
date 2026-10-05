"""Real native HDF5 validation and public solver propagation of exact changes.

No production checkpoint or numerical configuration is rewritten. Corruption
tests operate only on private copies of a freshly generated test checkpoint.
"""

from copy import deepcopy
import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest

from source.solvers.vpm import DirectInduction
from source.solvers.vpm.config.restart_changes import MISSING_CONFIGURATION_VALUE
from source.solvers.vpm.io.backup import _BackupIO
from tests.vpm.test_backup_storage import _add_counter_rotating_pair, _solver


@pytest.fixture(scope="module")
def native_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("explicit-vpm-restart")
    writer = _solver(
        directory / "writer", induction=DirectInduction(stretching_scheme="TRANSPOSED")
    )
    try:
        _add_counter_rotating_pair(writer)
        # A real accepted state needs no advance for a restart-identity test.
        writer.stabilization.reference_vortex_strength = np.array([0.1, 0.1], dtype=np.float32)
        writer.stabilization.reference_lengths = np.array([0.2, 0.3], dtype=np.float32)
        writer.stabilization.reference_moments = tuple(row.copy() for row in np.eye(3))
        writer.save_backup()
        path = directory / "writer/solution/vpm/vpm_000000.h5"
        with h5py.File(path, "r") as archive:
            config = json.loads(archive["solver"].attrs["numerical_configuration"])
            primary = {
                name: archive["particles"][name][:]
                for name in (
                    "position",
                    "vortex_strength",
                    "core_radius",
                    "particle_volume",
                    "kinematic_viscosity",
                    "group_id",
                    "zone_id",
                    "filament_reference_vortex_strength",
                    "filament_reference_length",
                )
            }
        yield path, config, primary
    finally:
        writer.close()


def _validate(path, config, **kwargs):
    return _BackupIO._validate_hdf5_structure(
        path, expected_float_dtype=np.dtype("float32"), expected_configuration=config, **kwargs
    )


def test_real_reader_strict_default_and_structured_expectations(native_checkpoint):
    path, saved, _ = native_checkpoint
    current = deepcopy(saved)
    current["induction"]["image_settings"] = {"order": 10, "spacing_ratio": 0.25}
    permission = "induction.image_settings"
    with pytest.raises(ValueError, match="mismatch at induction.image_settings"):
        _validate(path, current)
    with pytest.raises(ValueError, match="requires exact stored/current"):
        _validate(path, current, allowed_config_differences={permission})
    expectation = {
        permission: (MISSING_CONFIGURATION_VALUE, deepcopy(current["induction"]["image_settings"]))
    }
    changes = _validate(
        path,
        current,
        allowed_config_differences={permission},
        expected_config_differences=expectation,
    )
    assert changes[0]["stored"] == {"present": False}
    current["induction"]["image_settings"]["relax_state_limits"] = True
    with pytest.raises(ValueError, match="expectation mismatch"):
        _validate(
            path,
            current,
            allowed_config_differences={permission},
            expected_config_differences=expectation,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("time_step_size", 0.05),
        ("particle_kernel", "WINCKELMANS"),
        ("precision", "f64"),
        ("random_seed", 43),
    ],
)
def test_backend_permission_does_not_authorize_other_numerical_changes(
    native_checkpoint, field, value
):
    path, saved, _ = native_checkpoint
    current = deepcopy(saved)
    current["induction"]["stretching_scheme"] = "DIRECT"
    current[field] = value
    with pytest.raises(ValueError, match=field):
        _validate(path, current, allowed_config_differences={"induction.stretching_scheme"})


@pytest.mark.parametrize("corruption", ["config_hash", "core", "clock", "dtype"])
def test_config_permissions_never_override_native_integrity(
    native_checkpoint, tmp_path, corruption
):
    path, saved, _ = native_checkpoint
    damaged = tmp_path / "damaged.h5"
    shutil.copy2(path, damaged)
    with h5py.File(damaged, "r+") as archive:
        if corruption == "config_hash":
            archive["solver"].attrs["numerical_configuration_sha256"] = "0" * 64
        elif corruption == "core":
            archive["particles/core_radius"][0] = 0.0
        elif corruption == "clock":
            archive["solver"].attrs["time"] = float("nan")
        else:
            data = archive["particles/position"][:].astype(np.float64)
            del archive["particles/position"]
            archive["particles"].create_dataset("position", data=data)
    current = deepcopy(saved)
    current["induction"]["stretching_scheme"] = "DIRECT"
    with pytest.raises(ValueError):
        _validate(damaged, current, allowed_config_differences={"induction.stretching_scheme"})


def test_real_public_load_preserves_primary_state_and_writes_new_identity(
    native_checkpoint, tmp_path
):
    path, saved, primary = native_checkpoint
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = _solver(tmp_path / "reader", induction=DirectInduction(stretching_scheme="DIRECT"))
    permission = "induction.stretching_scheme"
    try:
        _add_counter_rotating_pair(reader)
        reader.particles.set_field("position", np.ones((2, 3), dtype=np.float32) * 0.15)
        before = reader.particle_position.copy()
        with pytest.raises(ValueError, match=permission):
            reader.load_backup(path)
        np.testing.assert_array_equal(reader.particle_position, before)
        assert (reader.step, reader.time) == (0, 0.0)
        with pytest.raises(ValueError, match="expectation mismatch"):
            reader.load_backup(
                path,
                allowed_config_differences={permission},
                expected_config_differences={permission: ("wrong", "DIRECT")},
            )
        np.testing.assert_array_equal(reader.particle_position, before)
        reader.load_backup(
            path,
            allowed_config_differences={permission},
            expected_config_differences={permission: ("TRANSPOSED", "DIRECT")},
        )
        for name in (
            "position",
            "vortex_strength",
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "group_id",
            "zone_id",
        ):
            np.testing.assert_array_equal(getattr(reader.particles, name + "_cpu")(), primary[name])
        np.testing.assert_array_equal(
            reader.stabilization.reference_vortex_strength,
            primary["filament_reference_vortex_strength"],
        )
        np.testing.assert_array_equal(
            reader.stabilization.reference_lengths, primary["filament_reference_length"]
        )
        np.testing.assert_array_equal(
            reader.stabilization.reference_moments, np.eye(3, dtype=np.float32)
        )
        assert (reader.step, reader.time, reader.time_step_size) == (0, 0.0, 0.01)
        assert reader._restart_details["configuration_changes"][0]["path"] == permission
        reader.save_backup()
        migrated = tmp_path / "reader/solution/vpm/vpm_000000.h5"
        with h5py.File(migrated, "r") as archive:
            new_config = json.loads(archive["solver"].attrs["numerical_configuration"])
            assert new_config["induction"]["stretching_scheme"] == "DIRECT"
        reader.load_backup(migrated)  # Ordinary strict restart now accepts its own configuration.
        assert reader._restart_details is None
    finally:
        reader.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert saved["induction"]["stretching_scheme"] == "TRANSPOSED"


def test_real_public_load_cannot_smuggle_changed_dt(native_checkpoint, tmp_path):
    path, _, _ = native_checkpoint
    reader = _solver(
        tmp_path / "reader",
        time_step_size=0.005,
        induction=DirectInduction(stretching_scheme="DIRECT"),
    )
    try:
        with pytest.raises(ValueError, match="explicit time_step_size"):
            reader.load_backup(
                path, allowed_config_differences={"time_step_size", "induction.stretching_scheme"}
            )
        reader.load_backup(
            path, time_step_size=0.005, allowed_config_differences={"induction.stretching_scheme"}
        )
        assert (reader.step, reader.time, reader.time_step_size) == (0, 0.0, 0.005)
        assert reader._restart_details["kind"] == "explicit_changed_time_step_continuation"
        assert (
            reader._restart_details["configuration_changes"][0]["path"]
            == "induction.stretching_scheme"
        )
    finally:
        reader.close()
