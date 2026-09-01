"""The VPM output API has one explicit configuration path."""

from __future__ import annotations

import inspect

import pytest

import openonda.vpm as vpm
from source.solvers.vpm.config.artifacts import Backup, Samplers


class _Sample:
    def sample(self, _solver):
        return None


def test_backup_defaults_and_custom_directories():
    assert Backup() == Backup(0, "solution", "solution")
    configured = Backup(25, "results/state", "results/logs")
    assert configured.interval_steps == 25
    assert configured.directory == "results/state"
    assert configured.log_directory == "results/logs"


@pytest.mark.parametrize("interval", [-1, True, 1.5])
def test_backup_rejects_invalid_intervals(interval):
    error = ValueError if interval == -1 else TypeError
    with pytest.raises(error):
        Backup(interval_steps=interval)


def test_samplers_accept_a_tuple_and_own_the_directory():
    first = _Sample()
    second = _Sample()
    configured = Samplers((first, second), "ring/run_a")

    assert configured.samples == (first, second)
    assert configured.directory == "ring/run_a"


@pytest.mark.parametrize("directory", ["", "/absolute", ".", "../escape"])
def test_sampler_directory_must_stay_below_samples(directory):
    with pytest.raises(ValueError):
        Samplers(samples=(_Sample(),), directory=directory)


def test_public_case_owns_backup_and_sampler_construction_objects():
    case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        backup=Backup(interval_steps=10),
        samplers=Samplers(samples=(_Sample(),)),
    )

    assert case.backup.interval_steps == 10
    assert len(case.samplers.samples) == 1
    assert {"backup", "samplers"} <= set(inspect.signature(vpm.VPMCase).parameters)


def test_internal_setup_type_is_not_public():
    assert not hasattr(vpm, "VPMSetup")


def test_numerics_has_no_legacy_serialization_hook():
    assert not hasattr(vpm.Numerics, "to_dict")
    assert not hasattr(vpm.Numerics, "from_dict")


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
            "continue_from_backup",
        }
    )
