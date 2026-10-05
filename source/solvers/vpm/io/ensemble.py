"""Admit independent native realizations before computing ensemble statistics."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd

from openonda.saved_times import match_saved_times, same_saved_time


def _clock(step, time):
    step_value, time_value = float(step), float(time)
    if (
        isinstance(step, bool)
        or not np.isfinite([step_value, time_value]).all()
        or step_value < 0
        or step_value != int(step_value)
        or time_value < 0
    ):
        raise ValueError("Ensemble states require finite nonnegative integer-step clocks")
    return int(step_value), time_value


def _physics(record):
    configuration = record["configuration"]
    numerics = deepcopy(configuration["numerics"])
    for name in (
        "random_seed",
        "compute_device",
        "write_precision",
        "debug_mode",
        "diagnostics",
        "verbose",
    ):
        numerics.pop(name, None)
    vlm = numerics.get("vlm")
    if vlm is not None:
        # Native VLM records carry the loaded geometry and physical motion
        # identity independently of source paths and output controls.
        numerics["vlm"] = {"physics_identity": vlm["physics_identity"]}
    physics = {
        "numerics": numerics,
        "initial_conditions": configuration["initial_conditions"],
        "initial_weak_particle_percent": configuration["initial_weak_particle_percent"],
    }
    return json.dumps(physics, sort_keys=True, allow_nan=False)


def realization_metadata(records) -> tuple[dict, ...]:
    """Read current native identities, common physics and accepted horizons.

    Seeds and case identities must be distinct. Output locations, schedules
    and execution devices can differ without changing the physical ensemble.
    """
    records = tuple(records)
    if len(records) < 2:
        raise ValueError("Ensemble statistics require at least two independent realizations")
    identities, seeds, physics, clocks = [], [], [], []
    for record in records:
        if record["schema_version"] != 1 or record["solver"] != "VPM":
            raise ValueError("Ensembles require current native VPM metadata")
        identity = record["case_name"]
        seed = record["configuration"]["numerics"]["random_seed"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("Ensemble realizations require named native identities")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Ensemble realizations require recorded integer random seeds")
        identities.append(identity)
        seeds.append(seed)
        physics.append(_physics(record))
        state = record["state"]
        initial = _clock(state["initial_step"], state["initial_time"])
        final = _clock(state["step"], state["time"])
        dt = float(record["configuration"]["numerics"]["time_step_size"])
        if not np.isfinite(dt) or dt <= 0 or final[0] < initial[0]:
            raise ValueError("Ensemble metadata has an invalid native accepted horizon")
        expected_time = initial[1] + (final[0] - initial[0]) * dt
        if not same_saved_time(final[1], expected_time):
            raise ValueError("Ensemble metadata step/time disagrees with its native clock")
        clocks.append((initial, final))
    if len(set(identities)) != len(records) or len(set(seeds)) != len(records):
        raise ValueError("Ensemble realization identities and random seeds must be independent")
    if len(set(physics)) != 1:
        raise ValueError("Ensemble realizations have different physical configurations")
    if any(
        current[0] != expected[0] or not same_saved_time(current[1], expected[1])
        for initial, final in clocks[1:]
        for current, expected in zip((initial, final), clocks[0], strict=True)
    ):
        raise ValueError("Ensemble realizations have different native accepted horizons")
    return records


def field_ensemble(records, *, coordinates, fields) -> dict:
    """Stack finite fields on one unchanged coordinate grid and physical clock.

    Coordinate names denote scalar coordinate-component arrays. Field arrays
    can carry additional component axes after those spatial dimensions.
    """
    records = tuple(records)
    if len(records) < 2:
        raise ValueError("Ensemble statistics require at least two independent realizations")
    clocks = [_clock(record["step"], record["time"]) for record in records]
    step, time = clocks[0]
    if any(
        other_step != step or not same_saved_time(other_time, time)
        for other_step, other_time in clocks[1:]
    ):
        raise ValueError("Ensemble fields have different native physical clocks")
    result = {"step": step, "time": time}
    spatial_shape = None
    for name in coordinates:
        arrays = [np.asarray(record[name], dtype=float) for record in records]
        reference = arrays[0]
        if (
            not reference.size
            or not np.isfinite(reference).all()
            or any(not np.array_equal(array, reference) for array in arrays[1:])
        ):
            raise ValueError("Ensemble fields have non-finite or different sample grids")
        if spatial_shape is not None and reference.shape != spatial_shape:
            raise ValueError("Ensemble coordinate components have inconsistent shapes")
        spatial_shape = reference.shape
        result[name] = reference.copy()
    if spatial_shape is None:
        raise ValueError("Ensemble fields require recorded sample coordinates")
    for name in fields:
        arrays = [np.asarray(record[name], dtype=float) for record in records]
        if any(
            array.shape[: len(spatial_shape)] != spatial_shape
            or array.shape != arrays[0].shape
            or not np.isfinite(array).all()
            for array in arrays
        ):
            raise ValueError(f"Ensemble field {name!r} has non-finite or inconsistent arrays")
        result[name] = np.stack(arrays)
    return result


def table_ensemble(frames):
    """Align current single-entity tables without averaging time or step.

    Return the first realization's clock and common categorical columns,
    plus numeric physical columns stacked along the realization dimension.
    """
    frames = tuple(frames)
    if len(frames) < 2:
        raise ValueError("Ensemble statistics require at least two independent realizations")
    columns = tuple(frames[0].columns)
    if not {"step", "time"}.issubset(columns) or any(
        set(frame.columns) != set(columns) or not len(frame) for frame in frames
    ):
        raise ValueError("Ensemble tables require the same nonempty native schema")
    clock = frames[0][["step", "time"]].reset_index(drop=True).copy()
    for frame in frames:
        steps = frame["step"].to_numpy(dtype=float)
        times = frame["time"].to_numpy(dtype=float)
        if (
            not np.isfinite(steps).all()
            or np.any(steps < 0)
            or np.any(steps != np.rint(steps))
            or np.any(np.diff(steps) <= 0)
        ):
            raise ValueError("Ensemble tables have non-finite or unordered native step clocks")
        match_saved_times(times)
        if (
            not np.array_equal(steps, clock["step"].to_numpy())
            or len(times) != len(clock)
            or any(
                not same_saved_time(time, reference)
                for time, reference in zip(times, clock["time"], strict=True)
            )
        ):
            raise ValueError("Ensemble tables have different native physical clocks")
    stacks = {}
    for name in columns:
        if name in {"step", "time"}:
            continue
        series = [frame[name] for frame in frames]
        if all(pd.api.types.is_numeric_dtype(values) for values in series):
            arrays = [values.to_numpy(dtype=float) for values in series]
            if any(not np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Ensemble table field {name!r} has non-finite values")
            stacks[name] = np.stack(arrays)
        else:
            values = series[0].to_numpy()
            if any(not np.array_equal(other.to_numpy(), values) for other in series[1:]):
                raise ValueError(f"Ensemble table field {name!r} has different recorded categories")
            clock[name] = values.copy()
    return clock, stacks


def mean_standard_error(values):
    """Mean and sample standard error across finite independent realizations."""
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or len(values) < 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Ensemble mean and standard error require two finite realizations")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
