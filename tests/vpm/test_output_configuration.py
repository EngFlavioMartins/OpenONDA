"""The VPM output API has one explicit configuration path."""

from __future__ import annotations

import inspect

import pytest

import openonda.vpm as vpm


class _Sample:
    def sample(self, _solver):
        return None


def test_backup_defaults_and_custom_directories():
    assert vpm.Backup() == vpm.Backup(
        interval_steps=0,
        directory="solution",
        log_directory="solution",
    )
    assert (
        vpm.Backup(
            interval_steps=25,
            directory="results/state",
            log_directory="results/logs",
        ).interval_steps
        == 25
    )


@pytest.mark.parametrize("interval", [-1, True, 1.5])
def test_backup_rejects_invalid_intervals(interval):
    error = ValueError if interval == -1 else TypeError
    with pytest.raises(error):
        vpm.Backup(interval_steps=interval)


def test_samplers_take_samples_positionally_and_own_the_directory():
    first = _Sample()
    second = _Sample()
    configured = vpm.Samplers(first, second, directory="ring/run_a")

    assert configured.samples == (first, second)
    assert configured.directory == "ring/run_a"


@pytest.mark.parametrize("directory", ["", "/absolute", ".", "../escape"])
def test_sampler_directory_must_stay_below_samples(directory):
    with pytest.raises(ValueError):
        vpm.Samplers(_Sample(), directory=directory)


def test_setup_accepts_only_the_two_output_constructors():
    setup = vpm.VPMSetup(
        backup=vpm.Backup(interval_steps=10),
        samplers=vpm.Samplers(_Sample()),
    )

    assert setup.backup.interval_steps == 10
    assert len(setup.samplers.samples) == 1
    assert set(inspect.signature(vpm.VPMSetup).parameters).isdisjoint(
        {
            "checkpoint_interval_steps",
            "checkpoint_directory",
            "checkpoint_name",
            "backup_interval_steps",
            "backup_directory",
            "backup_name",
            "logging_interval_steps",
            "timing_interval_steps",
            "log_mode",
            "sample_subdirectory",
            "final_samplers",
            "export_flow_integrals",
            "export_discretization_health",
            "backup_store_velocity_gradient",
        }
    )


def test_setup_does_not_translate_old_output_configuration():
    with pytest.raises(TypeError):
        vpm.VPMSetup(backup_interval_steps=10)
    with pytest.raises(TypeError):
        vpm.VPMSetup(samplers=(_Sample(),))
    with pytest.raises(ValueError):
        vpm.VPMSetup.from_dict({"checkpoint_interval_steps": 10})


def test_setup_serializes_only_the_backup_value_object():
    serialized = vpm.VPMSetup(
        backup=vpm.Backup(
            interval_steps=8,
            directory="state",
            log_directory="logs",
        ),
        samplers=vpm.Samplers(_Sample()),
    ).to_dict()

    assert serialized["backup"] == {
        "interval_steps": 8,
        "directory": "state",
        "log_directory": "logs",
    }
    assert "samplers" not in serialized


def test_solver_has_one_public_backup_save_and_load_pair():
    public_methods = {
        name
        for name, value in inspect.getmembers(vpm.VPMSolver, inspect.isfunction)
        if not name.startswith("_")
    }

    assert {"save_backup", "load_backup"} <= public_methods
    assert public_methods.isdisjoint(
        {
            "save_state",
            "save_numerical_state",
            "load_numerical_state",
            "write_backup",
            "continue_from_checkpoint",
            "continue_from_backup",
        }
    )
