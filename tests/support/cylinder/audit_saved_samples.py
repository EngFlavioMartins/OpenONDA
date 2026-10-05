"""Audit normal-directory observations against their recorded sampler comparison_settings.

This is read-only validation, not a solver or a periodicity/grid-convergence
test. An observed horizon need not coincide with a slower sampler's cadence.
"""

import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from openonda.saved_times import read_pvd_times, same_saved_time


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _fixed_time_step(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Recorded fixed time step must be finite and positive")
    return float(value)


def _cadence(schedule, dt, particle=False):
    if schedule.get("final_only", False) or schedule.get("is_final_only", False):
        raise ValueError("This audit requires an explicit periodic sampler schedule")
    if particle:
        if schedule.get("type") != "EverySteps" or any(
            schedule.get(key) is not None for key in ("first_step", "start_time")
        ):
            raise ValueError("Unsupported particle sample clock; no cadence will be inferred")
        interval = dt * schedule["interval"]
    else:
        steps, elapsed = schedule.get("every_n_steps"), schedule.get("every_time")
        if (steps is None) == (elapsed is None):
            raise ValueError("Expected exactly one recorded FVM sampler cadence")
        interval = elapsed if elapsed is not None else dt * steps
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampler cadence must be finite and positive")
    return float(interval)


def _expected_events(end, interval):
    last = math.floor(end / interval)
    if same_saved_time((last + 1) * interval, end):
        last += 1
    return set(range(1, last + 1))


def audit_csv(path, sampler, interval, end, particle=False, *, time_step_size):
    """Stream a history, retaining one event's geometry rather than all rows."""
    time_step_size = _fixed_time_step(time_step_size)
    kind = sampler["type"]
    count = 1 if kind == "ForceSampler" else int(sampler["n_points"])
    if count < 1:
        raise ValueError(f"Invalid recorded sample count for {path}")
    required = {"time", "step"}
    if kind == "ForceSampler":
        required |= {"drag_coefficient", "lift_coefficient", "side_force_coefficient"}
        required |= {
            f"{part}_force_{axis}" for part in ("total", "pressure", "viscous") for axis in "xyz"
        }
    else:
        quantities = ["position", "velocity"]
        if not particle or sampler.get("include_derivatives", True):
            quantities.append("vorticity")
        required |= {f"{quantity}_{axis}" for quantity in quantities for axis in "xyz"}
        if not particle:
            required.add("kinematic_pressure")
    events, geometry, group = set(), None, []
    previous, rows = None, 0

    def finish_group():
        nonlocal geometry
        if len(group) != count or len(set(group)) != count:
            raise ValueError(f"{path}: wrong point count or duplicate point at sample {previous}")
        current = tuple(group)
        if geometry is None:
            geometry = current
        elif current != geometry:
            raise ValueError(f"{path}: sample geometry/order changed at sample {previous}")

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(
                f"{path}: missing columns {sorted(required - set(reader.fieldnames or ()))}"
            )
        for raw in reader:
            values = {key: float(value) for key, value in raw.items() if key != "patch"}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}: non-finite saved value")
            time = values["time"]
            # Parse the original decimal token, not its rounded float: a tiny
            # fractional step must not become an apparently integral float.
            saved_step = Decimal(raw["step"])
            if saved_step < 0 or saved_step != saved_step.to_integral_value():
                raise ValueError(f"{path}: sample step must be a nonnegative integer")
            if not same_saved_time(time, int(saved_step) * time_step_size):
                raise ValueError(f"{path}: step/time mismatch for the recorded fixed time step")
            if "accepted_time_step_size" in values and not same_saved_time(
                values["accepted_time_step_size"], time_step_size
            ):
                raise ValueError(
                    f"{path}: accepted time step differs from the recorded fixed time step"
                )
            event = round(time / interval)
            if event < 0 or not same_saved_time(time, event * interval):
                raise ValueError(f"{path}: sample at {time:g} violates the recorded cadence")
            if previous is not None and event < previous:
                raise ValueError(f"{path}: reversed sample clock")
            if event != previous:
                if previous is not None:
                    finish_group()
                group = []
                events.add(event)
                previous = event
            if kind == "ForceSampler":
                if raw.get("patch") not in sampler["patch_names"]:
                    raise ValueError(f"{path}: force patch differs from its recorded definition")
                for axis in "xyz":
                    if not math.isclose(
                        values[f"total_force_{axis}"],
                        values[f"pressure_force_{axis}"] + values[f"viscous_force_{axis}"],
                        rel_tol=1e-10,
                        abs_tol=1e-9,
                    ):
                        raise ValueError(f"{path}: total force is inconsistent with its components")
                group.append(())
            else:
                group.append(tuple(values[f"position_{axis}"] for axis in "xyz"))
            rows += 1
    if previous is None:
        raise ValueError(f"{path}: empty history")
    finish_group()
    missing = _expected_events(end, interval) - events
    if missing:
        raise ValueError(
            f"{path}: missing {len(missing)} scheduled events through {end:g}; first={min(missing) * interval:g}"
        )
    # Particle line geometry is explicitly recorded with a hash in recorded metadata.
    recorded = sampler.get("line_points", {})
    if recorded.get("sha256"):
        points = np.asarray(geometry, dtype=recorded["dtype"])
        if (
            list(points.shape) != recorded["shape"]
            or hashlib.sha256(points.tobytes()).hexdigest() != recorded["sha256"]
        ):
            raise ValueError(f"{path}: coordinates differ from the recorded particle sampler")
    through = [
        event
        for event in events
        if event * interval <= end or same_saved_time(event * interval, end)
    ]
    return {
        "path": str(path),
        "sha256": _digest(path),
        "rows": rows,
        "points": count,
        "events": len(events),
        "cadence": interval,
        "start": min(events) * interval,
        "end": max(events) * interval,
        "checked_end": end,
        "time_step_size": time_step_size,
        "step_time_consistency": "validated",
        "last_required_event": max(_expected_events(end, interval), default=0) * interval,
        "events_after_checked_end": len(events) - len(through),
    }


def audit_collection(path, interval, end, watch):
    """Check every indexed surface, its finite valid fields and fixed grid."""
    import pyvista as pv

    times = read_pvd_times(path)
    events = []
    for time in times:
        event = round(time / interval)
        if not same_saved_time(time, event * interval):
            raise ValueError(f"{path}: field clock violates the recorded cadence")
        events.append(event)
    if _expected_events(end, interval) - set(events):
        raise ValueError(f"{path}: missing scheduled field frames through {end:g}")
    frames, geometry = [], None
    for item in sorted(
        ET.parse(path).iter("DataSet"), key=lambda item: float(item.attrib["timestep"])
    ):
        frame = path.parent / item.attrib["file"]
        watch[frame] = _stamp(frame)
        grid = pv.read(frame)
        points = np.asarray(grid.points)
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{frame}: non-finite grid")
        signature = (
            tuple(grid.dimensions),
            points.dtype.str,
            hashlib.sha256(points.tobytes()).hexdigest(),
        )
        if geometry is not None and signature != geometry:
            raise ValueError(f"{path}: moving or reordered surface grid")
        geometry = signature
        mask = np.asarray(grid.point_data.get("vtkValidPointMask", np.ones(len(points))))
        if mask.shape != (len(points),) or not np.all(np.isin(mask, [0, 1])) or not np.any(mask):
            raise ValueError(f"{frame}: invalid or empty fluid-point mask")
        valid = mask.astype(bool)
        if "velocity" not in grid.point_data or np.asarray(grid.point_data["velocity"]).shape != (
            len(points),
            3,
        ):
            raise ValueError(f"{frame}: missing vector velocity field")
        for name, values in grid.point_data.items():
            if not np.all(np.isfinite(np.asarray(values)[valid])):
                raise ValueError(f"{frame}: non-finite valid {name} samples")
        frames.append(
            {"file": frame.name, "time": float(item.attrib["timestep"]), "sha256": _digest(frame)}
        )
    if not frames:
        raise ValueError(f"{path}: empty surface collection")
    return {
        "path": str(path),
        "sha256": _digest(path),
        "cadence": interval,
        "frames": frames,
        "start": times[0],
        "end": times[-1],
        "checked_end": end,
        "grid_dimensions": list(geometry[0]),
        "grid_sha256": geometry[2],
    }


def audit_normal(case_directory, kind, end=None):
    """Audit the ordinary case/reference_flow directories; never discover parameter_studys."""
    if kind not in ("coupled", "reference"):
        raise ValueError("kind must be coupled or reference")
    case_directory = Path(case_directory).resolve()
    directory = case_directory / "reference_flow" if kind == "reference" else case_directory
    samples, solution = directory / "samples", directory / "solution"
    paths = [solution / "fvm_metadata.json"]
    if kind == "coupled":
        paths.append(solution / "vpm_metadata.json")
    watch = {path: _stamp(path) for path in paths}
    metadata = [json.loads(path.read_text()) for path in paths]
    definitions = []
    for index, item in enumerate(metadata):
        config = item["configuration"]
        particle = index == 1
        dt = _fixed_time_step(
            config["numerics"]["time_step_size"] if particle else config["time"]["time_step_size"]
        )
        if not particle and (
            config["time"].get("adjustment") is not None or config["time"].get("start_time", 0) != 0
        ):
            raise ValueError(
                "Normal audit currently requires the recorded fixed-step zero-origin clock"
            )
        samplers = config["samplers"]["items"] if particle else config["samplers"]
        for sampler in samplers:
            if sampler["type"] not in {"LineSampler", "ForceSampler", "SurfaceSampler"}:
                raise ValueError(
                    f"Unsupported recorded sampler {sampler['type']}; audit will not omit it"
                )
            filename = sampler["file_name"]
            if Path(filename).name != filename or not filename:
                raise ValueError("Sampler names must be simple filenames")
            path = samples / (
                filename + (".pvd" if sampler["type"] == "SurfaceSampler" else ".csv")
            )
            if path in watch:
                raise ValueError(f"Duplicate recorded sampler output: {path}")
            watch[path] = _stamp(path)
            definitions.append(
                (path, sampler, _cadence(sampler["schedule"], dt, particle), particle, dt)
            )
    if end is None:
        force = next(path for path, sampler, *_ in definitions if sampler["type"] == "ForceSampler")
        with force.open(newline="") as stream:
            for row in csv.DictReader(stream):
                end = float(row["time"])
    if end is None or not math.isfinite(end) or end <= 0:
        raise ValueError("Observed audit horizon must be finite and positive")
    results = []
    for path, sampler, interval, particle, dt in definitions:
        results.append(
            audit_collection(path, interval, end, watch)
            if path.suffix == ".pvd"
            else audit_csv(path, sampler, interval, end, particle, time_step_size=dt)
        )
    for path, before in watch.items():
        if _stamp(path) != before:
            raise RuntimeError(
                f"Observation changed during audit; retry after output settles: {path}"
            )
    csv_count = sum(path.suffix == ".csv" for path, *_ in definitions)
    return {
        "schema_version": 1,
        "kind": kind,
        "directory": str(directory),
        "checked_end": end,
        "csv_count": csv_count,
        "collection_count": len(definitions) - csv_count,
        "metadata": [{"path": str(path), "sha256": _digest(path)} for path in paths],
        "samples": results,
        "force_signal": {
            "status": "not assessed",
            "reason": "This audits sampling coverage and integrity, not settled periodicity or phase accuracy.",
        },
        "scope": "recorded observations through the requested horizon; no completion or grid-independence claim",
    }
