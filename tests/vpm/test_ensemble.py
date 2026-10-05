"""Native ensembles retain common physics, exact clocks and independent samples."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from source.solvers.vpm.io.ensemble import (
    field_ensemble,
    mean_standard_error,
    realization_metadata,
    table_ensemble,
)


def metadata():
    record = {
        "schema_version": 1,
        "solver": "VPM",
        "case_name": "realization_0",
        "configuration": {
            "numerics": {
                "time_step_size": 0.1,
                "random_seed": 11,
                "compute_device": "CPU",
                "viscous": {"scheme": "RWM", "kinematic_viscosity": 0.01},
            },
            "initial_conditions": [{"type": "vortex", "circulation": 2.0}],
            "initial_weak_particle_percent": 0.0,
            "samplers": {"directory": "samples/a"},
            "backup": {"directory": "solution/a"},
            "run": {"steps": 2},
        },
        "state": {"initial_step": 0, "initial_time": 0.0, "step": 2, "time": 0.2},
    }
    other = deepcopy(record)
    other["case_name"] = "realization_1"
    other["configuration"]["numerics"]["random_seed"] = 12
    return [record, other]


def fields():
    x, y = np.meshgrid([0.0, 1.0], [0.0, 2.0], indexing="ij")
    return [
        {"step": 2, "time": 0.2, "x": x, "y": y, "velocity": x + y + offset}
        for offset in (0.0, 2.0, 4.0)
    ]


def tables():
    return [
        pd.DataFrame(
            {
                "step": [0, 2],
                "time": [0.0, 0.2],
                "energy": [1 + offset, 2 + offset],
                "energy_source": ["direct", "direct"],
            }
        )
        for offset in (0.0, 2.0, 4.0)
    ]


def test_realizations_admit_portable_output_and_execution_choices_without_mutation():
    records = metadata()
    records[1]["configuration"]["numerics"]["compute_device"] = "METAL"
    records[1]["configuration"]["samplers"] = {"directory": "another/machine"}
    records[1]["configuration"]["backup"] = {"directory": "another/backup"}
    original = deepcopy(records)
    assert realization_metadata(records) == tuple(original)
    assert records == original


def test_vlm_physics_uses_recorded_geometry_identity_instead_of_machine_paths():
    records = metadata()
    for index, record in enumerate(records):
        record["configuration"]["numerics"]["vlm"] = {
            "physics_identity": {"geometry": "loaded-surface", "velocity": [1, 0, 0]},
            "surfaces": [{"surface_file": f"machine-{index}/surface.json"}],
        }
    realization_metadata(records)
    records[1]["configuration"]["numerics"]["vlm"]["physics_identity"]["velocity"] = [2, 0, 0]
    with pytest.raises(ValueError, match="physical configurations"):
        realization_metadata(records)


@pytest.mark.parametrize(
    "corruption", ["identity", "seed", "physics", "initial_field", "clock", "schema"]
)
def test_realizations_reject_dependent_or_different_scientific_inputs(corruption):
    records = metadata()
    changed = records[1]
    if corruption == "identity":
        changed["case_name"] = records[0]["case_name"]
    elif corruption == "seed":
        changed["configuration"]["numerics"]["random_seed"] = 11
    elif corruption == "physics":
        changed["configuration"]["numerics"]["viscous"]["kinematic_viscosity"] = 0.02
    elif corruption == "initial_field":
        changed["configuration"]["initial_conditions"][0]["circulation"] = 3.0
    elif corruption == "clock":
        changed["state"].update(step=1, time=0.1)
    else:
        changed["schema_version"] = 0
    with pytest.raises(ValueError):
        realization_metadata(records)


def test_field_statistics_retain_sample_grid_and_native_clock():
    records = fields()
    arrays = field_ensemble(records, coordinates=("x", "y"), fields=("velocity",))
    mean, standard_error = mean_standard_error(arrays["velocity"])
    np.testing.assert_array_equal(mean, records[0]["velocity"] + 2)
    np.testing.assert_allclose(standard_error, 2 / np.sqrt(3))
    np.testing.assert_array_equal(arrays["x"], records[0]["x"])
    assert arrays["step"] == 2 and arrays["time"] == 0.2


@pytest.mark.parametrize(
    "corruption", ["time", "step", "grid", "coordinate_shape", "field_shape", "nonfinite"]
)
def test_field_ensemble_rejects_mixed_clocks_grids_and_fields(corruption):
    records = fields()
    if corruption == "time":
        records[1]["time"] = 0.3
    elif corruption == "step":
        records[1]["step"] = 3
    elif corruption == "grid":
        records[1]["x"] = records[1]["x"] + 1
    elif corruption == "coordinate_shape":
        for record in records:
            record["y"] = record["y"].ravel()
    elif corruption == "field_shape":
        records[1]["velocity"] = records[1]["velocity"].ravel()
    else:
        records[1]["velocity"][0, 0] = np.nan
    with pytest.raises(ValueError):
        field_ensemble(records, coordinates=("x", "y"), fields=("velocity",))


def test_table_mean_preserves_clock_roundoff_and_recorded_categories():
    frames = tables()
    frames[1].loc[1, "time"] += 1e-13
    clock, arrays = table_ensemble(frames)
    pd.testing.assert_frame_equal(clock, frames[0][["step", "time", "energy_source"]])
    assert not {"step", "time"}.intersection(arrays)
    np.testing.assert_array_equal(mean_standard_error(arrays["energy"])[0], [3, 4])


@pytest.mark.parametrize(
    "corruption", ["time", "step", "duplicate", "schema", "category", "nonfinite"]
)
def test_tables_reject_clock_or_schema_repair_and_invalid_scientific_values(corruption):
    frames = tables()
    if corruption == "time":
        frames[1].loc[1, "time"] = 0.3
    elif corruption == "step":
        frames[1].loc[1, "step"] = 3
    elif corruption == "duplicate":
        frames[1].loc[1, ["step", "time"]] = [0, 0]
    elif corruption == "schema":
        frames[1] = frames[1].drop(columns="energy")
    elif corruption == "category":
        frames[1].loc[1, "energy_source"] = "undefined"
    else:
        frames[1].loc[1, "energy"] = np.nan
    with pytest.raises(ValueError):
        table_ensemble(frames)


def test_leave_one_out_variance_requires_two_remaining_realizations():
    arrays = field_ensemble(fields(), coordinates=("x", "y"), fields=("velocity",))
    mean, standard_error = mean_standard_error(arrays["velocity"][1:])
    np.testing.assert_array_equal(mean, fields()[0]["velocity"] + 3)
    np.testing.assert_allclose(standard_error, 1)
    with pytest.raises(ValueError, match="two finite realizations"):
        mean_standard_error(arrays["velocity"][:1])
