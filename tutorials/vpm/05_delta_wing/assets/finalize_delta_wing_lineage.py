#!/usr/bin/env python3
"""Finalize the canonical Delta-wing lineage from native run outputs.

The fresh case starts with a logical initial-value origin at ``t=0`` but its
first persisted coupled owner state is the scheduled backup at ``0.025 s``.
This helper is deliberately separate from the solver setup: it only promotes
the active manifest after the completed native output and dense owner clock
have passed the final-run checks. It supports both an ordinary fresh run and
a declared fresh-prefix/checkpoint-continuation pair.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyvista as pv
from defusedxml import ElementTree

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
MANIFEST_PATH = CASE_DIR / "assets" / "delta_wing_accepted_lineage.json"
CLOCK_ATOL = 1.0e-12
BACKUP_NAME = re.compile(r"vpm_(\d+)\.h5$")
XDMF_NAME = re.compile(r"vpm_(\d+)\.xdmf$")
VTP_NAME = re.compile(r"vlm_(\d+)\.vtp$")
REQUIRED_CSVS = (
    "flow_integrals.csv",
    "vlm_forces.csv",
    "vlm_surface_forces.csv",
    "vlm_chordwise_front_wing.csv",
    "vlm_chordwise_rear_wing.csv",
    "vlm_spanwise_front_wing.csv",
    "vlm_spanwise_rear_wing.csv",
)
LOADING_FILES = {
    "vlm_chordwise_front_wing.csv": "front_wing",
    "vlm_chordwise_rear_wing.csv": "rear_wing",
    "vlm_spanwise_front_wing.csv": "front_wing",
    "vlm_spanwise_rear_wing.csv": "rear_wing",
}
MAX_NATIVE_GAP = 1.0 / 30.0
CAMPAIGN_END_STEP = 4000
CAMPAIGN_END_TIME = 10.0
REQUIRED_CSV_COLUMNS = {
    "flow_integrals.csv": {
        "total_kinetic_energy",
        "total_enstrophy",
        "vortex_strength_magnitude_sum",
        "n_particles_total",
        "lagrangian_cfl",
    },
    "vlm_forces.csv": {"force_x", "force_y", "force_z", "lift", "drag", "power"},
    "vlm_surface_forces.csv": {
        "surface",
        "force_x",
        "force_y",
        "force_z",
        "moment_x",
        "moment_y",
        "moment_z",
        "power",
        "translation_velocity_z",
        "centroid_z",
    },
    "vlm_chordwise_front_wing.csv": {
        "surface",
        "station_id",
        "chord_index",
        "panel_circulation",
        "panel_force_x",
        "panel_force_y",
        "panel_force_z",
    },
    "vlm_chordwise_rear_wing.csv": {
        "surface",
        "station_id",
        "chord_index",
        "panel_circulation",
        "panel_force_x",
        "panel_force_y",
        "panel_force_z",
    },
    "vlm_spanwise_front_wing.csv": {
        "surface",
        "station_id",
        "span_index",
        "circulation",
        "section_force_x",
        "section_force_y",
        "section_force_z",
    },
    "vlm_spanwise_rear_wing.csv": {
        "surface",
        "station_id",
        "span_index",
        "circulation",
        "section_force_x",
        "section_force_y",
        "section_force_z",
    },
}
NUMERIC_CSV_COLUMNS = {
    name: required - {"surface", "station_id"} for name, required in REQUIRED_CSV_COLUMNS.items()
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clock(path: Path) -> tuple[int, float]:
    with h5py.File(path, "r") as archive:
        try:
            solver = archive["solver"]
            step = int(solver.attrs["step"])
            time = float(solver.attrs["time"])
            vlm = solver["vlm"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{path.name}: missing native coupled solver clock") from error
        missing = {"panel_corner_position", "circulation", "panel_force"} - set(vlm)
        if missing:
            raise RuntimeError(f"{path.name}: missing coupled VLM arrays {sorted(missing)}")
        try:
            vlm_time = float(vlm.attrs["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{path.name}: missing saved VLM time") from error
    if step < 0 or not np.isfinite(time):
        raise RuntimeError(f"{path.name}: invalid native coupled solver clock")
    if not np.isfinite(vlm_time) or not np.isclose(vlm_time, time, rtol=0.0, atol=CLOCK_ATOL):
        raise RuntimeError(f"{path.name}: saved VLM time does not match solver time")
    return step, time


def _text_attribute(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _native_identity(path: Path) -> dict[str, object]:
    """Read and validate the native identity carried by one coupled backup."""
    with h5py.File(path, "r") as archive:
        try:
            solver = archive["solver"]
            vlm = solver["vlm"]
            backup_format = _text_attribute(solver.attrs["backup_format_version"])
            numerical_configuration = _text_attribute(solver.attrs["numerical_configuration"])
            numerical_hash = _text_attribute(solver.attrs["numerical_configuration_sha256"])
            write_precision = _text_attribute(solver.attrs["write_precision"])
            time_step_size = float(solver.attrs["time_step_size"])
            version = int(vlm.attrs["version"])
            identity = _text_attribute(vlm.attrs["identity"])
            physics_identity = _text_attribute(vlm.attrs["physics_identity"])
            panel_count = len(vlm["panel_corner_position"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{path.name}: missing native identity attributes") from error
    if backup_format != "10.0":
        raise RuntimeError(f"{path.name}: unsupported native backup format {backup_format!r}")
    if hashlib.sha256(numerical_configuration.encode("utf-8")).hexdigest() != numerical_hash:
        raise RuntimeError(f"{path.name}: numerical configuration hash is invalid")
    if not np.isfinite(time_step_size) or time_step_size <= 0.0:
        raise RuntimeError(f"{path.name}: native time-step identity is invalid")
    return {
        "backup_format_version": backup_format,
        "numerical_configuration": numerical_configuration,
        "numerical_configuration_sha256": numerical_hash,
        "write_precision": write_precision,
        "time_step_size": time_step_size,
        "vlm_version": version,
        "vlm_identity": identity,
        "vlm_physics_identity": physics_identity,
        "vlm_panel_count": panel_count,
    }


def _compare_native_identities(
    expected: dict[str, object], found: dict[str, object], label: str
) -> None:
    for key in expected:
        if key == "time_step_size":
            equal = np.isclose(float(expected[key]), float(found[key]), rtol=0.0, atol=CLOCK_ATOL)
        else:
            equal = expected[key] == found[key]
        if not equal:
            raise RuntimeError(f"{label}: native identity mismatch in {key}")


def _strip_manifest_type_fields(value):
    if isinstance(value, dict):
        return {
            key: _strip_manifest_type_fields(item) for key, item in value.items() if key != "type"
        }
    if isinstance(value, list):
        return [_strip_manifest_type_fields(item) for item in value]
    return value


def _check_metadata_native_identity(
    metadata: dict, identity: dict[str, object], label: str
) -> None:
    try:
        numerics = metadata["configuration"]["numerics"]
        vlm = numerics["vlm"]
        restart_identity = vlm["restart_identity"]
        physics_identity = vlm["physics_identity"]
        metadata_time_step = float(numerics["time_step_size"])
        metadata_precision = str(numerics["write_precision"])
        native_numerics = json.loads(str(identity["numerical_configuration"]))
        expected_numerics = {key: deepcopy(numerics[key]) for key in native_numerics}
        expected_induction = expected_numerics.get("induction")
        if isinstance(expected_induction, dict) and "kernel" not in expected_induction:
            expected_induction["kernel"] = numerics["particle_kernel"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"{label}: metadata is missing VLM identity") from error
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label}: native numerical configuration is malformed") from error
    if restart_identity != identity["vlm_identity"]:
        raise RuntimeError(f"{label}: metadata VLM identity does not match native backups")
    if physics_identity != identity["vlm_physics_identity"]:
        raise RuntimeError(f"{label}: metadata VLM physics identity does not match native backups")
    if not np.isclose(
        metadata_time_step,
        float(identity["time_step_size"]),
        rtol=0.0,
        atol=CLOCK_ATOL,
    ):
        raise RuntimeError(f"{label}: metadata time-step identity does not match native backups")
    if metadata_precision != identity["write_precision"]:
        raise RuntimeError(f"{label}: metadata write precision does not match native backups")
    if _strip_manifest_type_fields(native_numerics) != _strip_manifest_type_fields(
        expected_numerics
    ):
        raise RuntimeError(
            f"{label}: metadata numerical configuration does not match native backups"
        )


def _named_steps(directory: Path, pattern: re.Pattern[str], suffix: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in found:
            raise RuntimeError(f"duplicate {suffix} owner for step {step}")
        found[step] = path
    return found


def _collection_rows(path: Path) -> list[tuple[float, str]]:
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise RuntimeError(f"cannot parse native collection {path}") from error
    rows = []
    for dataset in root.findall(".//DataSet"):
        try:
            time = float(dataset.attrib["timestep"])
            filename = dataset.attrib["file"]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{path}: malformed DataSet entry") from error
        if not np.isfinite(time) or not filename:
            raise RuntimeError(f"{path}: invalid DataSet entry")
        rows.append((time, filename))
    return rows


def _xdmf_time(path: Path) -> float:
    try:
        root = ElementTree.parse(path).getroot()
        value = root.find(".//Time").attrib["Value"]
        time = float(value)
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        ElementTree.ParseError,
    ) as error:
        raise RuntimeError(f"{path}: missing or malformed XDMF time") from error
    if not np.isfinite(time):
        raise RuntimeError(f"{path}: XDMF time is not finite")
    return time


def _vtp_time(path: Path) -> tuple[float, float]:
    try:
        dataset = pv.read(path)
        time = np.asarray(dataset.field_data["time"], dtype=float)
        time_value = np.asarray(dataset.field_data["TimeValue"], dtype=float)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"{path}: missing or malformed VTP time fields") from error
    if (
        time.size != 1
        or time_value.size != 1
        or not np.isfinite(time).all()
        or not np.isfinite(time_value).all()
    ):
        raise RuntimeError(f"{path}: VTP time fields must contain one finite value each")
    return float(time[0]), float(time_value[0])


def _check_owner_collections(expected: dict[int, float], solution_dir: Path | None = None) -> None:
    directory = SOLUTION_DIR if solution_dir is None else Path(solution_dir)
    vtp_by_step = _named_steps(directory, VTP_NAME, ".vtp")
    if set(vtp_by_step) != set(expected):
        raise RuntimeError("native VTP owners do not exactly match coupled H5 backups")
    for step, path in vtp_by_step.items():
        time, time_value = _vtp_time(path)
        if not np.isclose(time, expected[step], rtol=0.0, atol=CLOCK_ATOL):
            raise RuntimeError(f"native VTP time mismatch at step {step}")
        if not np.isclose(time_value, expected[step], rtol=0.0, atol=CLOCK_ATOL):
            raise RuntimeError(f"native VTP TimeValue mismatch at step {step}")
    pvd = directory / "vlm.pvd"
    if not pvd.is_file():
        raise RuntimeError(f"missing native VLM collection {pvd}")
    rows = _collection_rows(pvd)
    if len(rows) != len(expected):
        raise RuntimeError("native VLM collection does not contain one entry per coupled backup")
    row_steps = []
    for time, filename in rows:
        match = VTP_NAME.fullmatch(filename)
        if match is None or int(match.group(1)) not in expected:
            raise RuntimeError(f"native VLM collection references unexpected owner {filename!r}")
        step = int(match.group(1))
        row_steps.append(step)
        if not np.isclose(time, expected[step], rtol=0.0, atol=CLOCK_ATOL):
            raise RuntimeError(f"native VLM collection clock mismatch at step {step}")
    if row_steps != sorted(expected):
        raise RuntimeError("native VLM collection owners are not in ascending clock order")


def _check_csv(path: Path, end_step: int, initial_time: float, dt: float) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(f"missing required sampled output {path}")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read required sampled output {path}") from error
    required = {"step", "time", *REQUIRED_CSV_COLUMNS.get(path.name, set())}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        detail = f"missing columns {missing}" if missing else "required rows are missing"
        raise RuntimeError(
            f"{path.name}: required scientific columns or rows are missing ({detail})"
        )
    for column in NUMERIC_CSV_COLUMNS.get(path.name, set()):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise RuntimeError(f"{path.name}: required numeric column {column} must be numeric")
    steps = pd.to_numeric(frame["step"], errors="coerce")
    times = pd.to_numeric(frame["time"], errors="coerce")
    if (
        steps.isna().any()
        or times.isna().any()
        or not np.isfinite(steps.to_numpy(dtype=float)).all()
        or not np.isfinite(times.to_numpy(dtype=float)).all()
    ):
        raise RuntimeError(f"{path.name}: required step/time columns must be finite numeric values")
    if (steps % 1 != 0).any():
        raise RuntimeError(f"{path.name}: step samples must be integer values")
    expected_times = initial_time + steps.to_numpy(dtype=float) * dt
    if not np.isclose(times.to_numpy(dtype=float), expected_times, rtol=0.0, atol=CLOCK_ATOL).all():
        raise RuntimeError(f"{path.name}: time samples do not match step*dt")
    frame = frame.copy()
    frame["step"] = steps.astype(int)
    frame["time"] = times.astype(float)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError(f"{path.name}: non-finite numeric samples")
    if int(steps.max()) < end_step:
        raise RuntimeError(f"{path.name}: samples do not reach final step {end_step}")
    return frame


def _require_exact_steps(frame: pd.DataFrame, expected_steps: list[int], label: str) -> None:
    observed = np.unique(frame["step"].to_numpy(dtype=int))
    if not np.array_equal(observed, np.asarray(expected_steps, dtype=int)):
        raise RuntimeError(f"{label}: samples do not cover every required step")


def _require_surface_steps(
    frame: pd.DataFrame,
    expected_steps: list[int],
    expected_surface: str,
    label: str,
    expected_rows_per_step: int | None = None,
    expected_station_keys: set[tuple] | None = None,
) -> None:
    if "surface" not in frame:
        raise RuntimeError(f"{label}: missing surface membership column")
    if {str(surface) for surface in frame["surface"]} != {expected_surface}:
        raise RuntimeError(f"{label}: surface membership is incomplete or unexpected")
    station_columns = _station_identity_columns(frame, label)
    grouped = frame.groupby("step", sort=False, dropna=False)
    observed_steps = {int(step) for step in grouped.groups}
    if observed_steps != set(expected_steps):
        raise RuntimeError(f"{label}: surface membership is incomplete or unexpected")
    expected_stations = None
    for step, rows in grouped:
        stations = _station_keys(rows, station_columns, label)
        if len(stations) != len(rows):
            raise RuntimeError(f"{label}: duplicate station rows at step {int(step)}")
        if expected_rows_per_step is not None and len(stations) != expected_rows_per_step:
            raise RuntimeError(f"{label}: station membership is incomplete at step {int(step)}")
        if expected_station_keys is not None and stations != expected_station_keys:
            raise RuntimeError(f"{label}: station membership is incomplete at step {int(step)}")
        if expected_stations is None:
            expected_stations = stations
        elif stations != expected_stations:
            raise RuntimeError(f"{label}: station membership changes across steps")


def _station_identity_columns(frame: pd.DataFrame, label: str) -> list[str]:
    """Return stable station/panel identity columns for loading tables."""
    if "station_id" not in frame:
        raise RuntimeError(f"{label}: missing station identity column")
    columns = [
        column
        for column in ("wing_uid", "segment_uid", "station_id", "half", "span_index")
        if column in frame
    ]
    if "chord_index" in frame:
        columns.append("chord_index")
    return columns


def _station_keys(frame: pd.DataFrame, columns: list[str], label: str) -> set[tuple]:
    return set(_station_keys_by_row(frame, columns, label))


def _station_keys_by_row(frame: pd.DataFrame, columns: list[str], label: str) -> list[tuple]:
    keys = []
    for values in frame[columns].itertuples(index=False, name=None):
        if any(pd.isna(value) for value in values):
            raise RuntimeError(f"{label}: station identity contains missing values")
        keys.append(
            tuple(value.item() if isinstance(value, np.generic) else value for value in values)
        )
    return keys


def _declared_loading_count(metadata: dict, filename: str, surface: str, label: str) -> int:
    return len(_declared_loading_records(metadata, filename, surface, label))


def _declared_loading_records(
    metadata: dict, filename: str, surface: str, label: str
) -> list[dict[str, object]]:
    try:
        surfaces = metadata["configuration"]["numerics"]["vlm"]["surfaces"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"{label}: metadata is missing declared VLM geometry") from error
    try:
        multi_surface = len(surfaces) > 1
    except TypeError as error:
        raise RuntimeError(f"{label}: metadata VLM surfaces are malformed") from error
    for record in surfaces:
        try:
            if record.get("name") != surface:
                continue
            wings = record["geometry"]["wings"]
            loading_records: list[dict[str, object]] = []
            for wing in wings:
                wing_uid = str(wing["uid"])
                exported_wing_uid = f"{surface}_{wing_uid}" if multi_surface else wing_uid
                symmetry = int(wing.get("symmetry", 0))
                halves = ("orig", "mirror") if symmetry > 0 else ("orig",)
                for segment in wing["segments"]:
                    segment_uid = str(segment["uid"])
                    n_spanwise = int(segment["n_spanwise_panels"])
                    n_chordwise = int(segment["n_chordwise_panels"])
                    if n_spanwise <= 0 or n_chordwise <= 0:
                        raise ValueError("panel counts must be positive")
                    for half in halves:
                        for span_index in range(n_spanwise):
                            station_id = f"{exported_wing_uid}:{segment_uid}:{half}:{span_index}"
                            base = {
                                "wing_uid": exported_wing_uid,
                                "segment_uid": segment_uid,
                                "station_id": station_id,
                                "half": half,
                                "span_index": span_index,
                            }
                            if "chordwise" in filename:
                                loading_records.extend(
                                    {**base, "chord_index": chord_index}
                                    for chord_index in range(n_chordwise)
                                )
                            else:
                                loading_records.append(base)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{label}: metadata geometry is malformed") from error
        if not loading_records:
            raise RuntimeError(f"{label}: metadata geometry declares no loading stations")
        return loading_records
    raise RuntimeError(f"{label}: metadata has no VLM surface {surface!r}")


def _declared_loading_keys(
    metadata: dict,
    filename: str,
    surface: str,
    label: str,
    columns: list[str],
) -> set[tuple]:
    records = _declared_loading_records(metadata, filename, surface, label)
    try:
        return {tuple(record[column] for column in columns) for record in records}
    except KeyError as error:
        raise RuntimeError(f"{label}: metadata geometry cannot describe loading keys") from error


def _require_force_surface_steps(
    frame: pd.DataFrame, expected_steps: list[int], label: str
) -> None:
    if "surface" not in frame:
        raise RuntimeError(f"{label}: missing surface membership column")
    observed = {
        (int(step), str(surface))
        for step, surface in zip(frame["step"], frame["surface"], strict=True)
    }
    expected = {
        (step, surface) for step in expected_steps for surface in ("front_wing", "rear_wing")
    }
    if observed != expected or len(frame) != len(expected):
        raise RuntimeError(f"{label}: surface membership is incomplete or unexpected")


def _check_completed_native_run() -> tuple[int, float, dict[int, float]]:
    metadata_path = SOLUTION_DIR / "vpm_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"missing solver metadata {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        config = metadata["configuration"]
        state = metadata["state"]
        lifecycle = metadata["lifecycle"]
        initial_step = int(state["initial_step"])
        initial_time = float(state["initial_time"])
        requested_steps = int(config["run"]["steps"])
        dt = float(config["numerics"]["time_step_size"])
        interval = int(config["backup"]["interval_steps"])
        final_step = int(state["step"])
        final_time = float(state["time"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("solver metadata is malformed") from error
    if lifecycle.get("status") != "completed":
        raise RuntimeError(f"canonical run is not completed (status={lifecycle.get('status')!r})")
    expected_step = initial_step + requested_steps
    expected_time = initial_time + requested_steps * dt
    if initial_step != 0 or not np.isclose(initial_time, 0.0, rtol=0.0, atol=CLOCK_ATOL):
        raise RuntimeError("canonical finalizer requires a fresh run from step 0/time 0")
    if final_step != expected_step or not np.isclose(
        final_time, expected_time, rtol=0.0, atol=CLOCK_ATOL
    ):
        raise RuntimeError(
            f"canonical run is incomplete (state step/time={final_step}/{final_time:g}; "
            f"expected {expected_step}/{expected_time:g})"
        )
    if interval <= 0 or requested_steps <= 0 or requested_steps % interval:
        raise RuntimeError("canonical backup interval must divide the requested run")

    segment = {
        "id": "clean_dense_run",
        "accepted_interval": {
            "first_step": initial_step,
            "last_step": expected_step,
        },
        "solution_path": SOLUTION_DIR,
    }
    expected_steps = _segment_owner_steps(segment, interval)
    clocks, native_identity = _check_native_segment(
        segment, interval=interval, initial_time=initial_time, dt=dt
    )
    _check_metadata_native_identity(metadata, native_identity, "canonical run")
    loading_counts = {
        name: _declared_loading_count(metadata, name, surface, "canonical run")
        for name, surface in LOADING_FILES.items()
    }
    declared_panel_count = sum(
        count for name, count in loading_counts.items() if "chordwise" in name
    )
    if declared_panel_count != native_identity["vlm_panel_count"]:
        raise RuntimeError(
            "canonical run: declared VLM geometry does not match native panel topology"
        )

    frames = {
        name: _check_csv(SAMPLES_DIR / name, expected_step, initial_time, dt)
        for name in REQUIRED_CSVS
    }
    _require_exact_steps(frames["flow_integrals.csv"], expected_steps, "flow_integrals.csv")
    if len(frames["flow_integrals.csv"]) != len(expected_steps):
        raise RuntimeError("flow-integral samples contain duplicate owner rows")
    accepted_steps = list(range(initial_step + 1, expected_step + 1))
    _require_exact_steps(frames["vlm_forces.csv"], accepted_steps, "vlm_forces.csv")
    if len(frames["vlm_forces.csv"]) != len(accepted_steps):
        raise RuntimeError("vlm_forces.csv contains duplicate accepted-step rows")
    force = frames["vlm_surface_forces.csv"]
    _require_force_surface_steps(force, accepted_steps, "vlm_surface_forces.csv")
    for name, surface in LOADING_FILES.items():
        _require_exact_steps(frames[name], accepted_steps, name)
        station_columns = _station_identity_columns(frames[name], name)
        expected_station_keys = _declared_loading_keys(
            metadata, name, surface, "canonical run", station_columns
        )
        _require_surface_steps(
            frames[name],
            accepted_steps,
            surface,
            name,
            loading_counts[name],
            expected_station_keys,
        )
    return expected_step, expected_time, clocks


def _read_metadata(path: Path) -> dict:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read solver metadata {path}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError(f"solver metadata is malformed: {path}")
    return metadata


def _metadata_identity(metadata: dict, path: Path) -> str:
    """Return the configuration identity that must survive a restart."""
    try:
        configuration = metadata["configuration"]
        run = configuration["run"]
        backup = configuration["backup"]
        samplers = configuration["samplers"]
        identity = {
            "numerics": configuration["numerics"],
            "run": {key: value for key, value in run.items() if key != "steps"},
            "backup": {"interval_steps": backup["interval_steps"]},
            "samplers": {"items": samplers["items"]},
            "initial_conditions": configuration["initial_conditions"],
            "initial_weak_particle_percent": configuration["initial_weak_particle_percent"],
        }
        return json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"solver metadata identity is malformed: {path}") from error


def _segment_metadata(
    segment: dict,
    *,
    initial_time: float,
    dt: float,
    require_completed: bool,
) -> tuple[dict, int, float, int, float, int]:
    """Validate one declared lineage segment's solver-owned metadata."""
    solution = segment["solution_path"]
    metadata_path = solution / "vpm_metadata.json"
    metadata = _read_metadata(metadata_path)
    try:
        configuration = metadata["configuration"]
        state = metadata["state"]
        lifecycle = metadata["lifecycle"]
        initial_step = int(state["initial_step"])
        segment_initial_time = float(state["initial_time"])
        final_step = int(state["step"])
        final_time = float(state["time"])
        requested_steps = int(configuration["run"]["steps"])
        segment_dt = float(configuration["numerics"]["time_step_size"])
        interval = int(configuration["backup"]["interval_steps"])
        status = lifecycle["status"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"solver metadata is malformed: {metadata_path}") from error
    if not np.isfinite(segment_initial_time) or not np.isfinite(final_time):
        raise RuntimeError(f"solver metadata has non-finite clocks: {metadata_path}")
    if interval <= 0 or requested_steps < 0 or initial_step < 0 or final_step < 0:
        raise RuntimeError(f"solver metadata has invalid run controls: {metadata_path}")
    if not np.isclose(segment_dt, dt, rtol=0.0, atol=CLOCK_ATOL):
        raise RuntimeError(f"restart segment time-step identity mismatch: {metadata_path}")
    interval_value = segment["accepted_interval"]
    last_step = interval_value["last_step"]
    last_time = interval_value["last_time"]
    if last_step is None or last_time is None:
        raise RuntimeError(f"{segment['id']}: completed lineage segment has no endpoint")
    if final_step != last_step or not np.isclose(final_time, last_time, rtol=0.0, atol=CLOCK_ATOL):
        raise RuntimeError(
            f"{segment['id']}: metadata endpoint does not match declared lineage interval"
        )
    if segment["origin"] == "fresh_initial_value":
        if initial_step != 0 or not np.isclose(
            segment_initial_time, initial_time, rtol=0.0, atol=CLOCK_ATOL
        ):
            raise RuntimeError(f"{segment['id']}: fresh metadata does not start at step 0/time 0")
    else:
        expected_initial_step = interval_value["first_step"] - 1
        expected_initial_time = interval_value["first_time"] - dt
        if initial_step != expected_initial_step or not np.isclose(
            segment_initial_time, expected_initial_time, rtol=0.0, atol=CLOCK_ATOL
        ):
            raise RuntimeError(
                f"{segment['id']}: restart metadata does not start at its predecessor"
            )
    if require_completed:
        if status != "completed":
            raise RuntimeError(f"{segment['id']}: continuation metadata is not completed")
        if requested_steps != final_step - initial_step or not np.isclose(
            final_time, segment_initial_time + requested_steps * dt, rtol=0.0, atol=CLOCK_ATOL
        ):
            raise RuntimeError(f"{segment['id']}: completed metadata clock is inconsistent")
    return metadata, initial_step, segment_initial_time, final_step, final_time, interval


def _segment_owner_steps(segment: dict, interval: int) -> list[int]:
    first_step = segment["accepted_interval"]["first_step"]
    last_step = segment["accepted_interval"]["last_step"]
    if last_step is None:
        raise RuntimeError(f"{segment['id']}: finalizer requires a finite segment endpoint")
    predecessor = 0 if first_step == 0 else first_step - 1
    owner_steps = list(range(predecessor + interval, last_step + 1, interval))
    if not owner_steps or owner_steps[-1] != last_step:
        raise RuntimeError(f"{segment['id']}: endpoint is not on the scheduled owner clock")
    return owner_steps


def _check_native_segment(
    segment: dict, *, interval: int, initial_time: float, dt: float
) -> tuple[dict[int, float], dict[str, object]]:
    """Validate one namespace's complete coupled owner clock."""
    solution = segment["solution_path"]
    expected_steps = _segment_owner_steps(segment, interval)
    backups = _named_steps(solution, BACKUP_NAME, ".h5")
    if sorted(backups) != expected_steps:
        raise RuntimeError(
            f"{segment['id']}: native H5 backups are not the complete scheduled owner clock"
        )
    clocks: dict[int, float] = {}
    native_identity = None
    for step in expected_steps:
        identity = _native_identity(backups[step])
        if native_identity is None:
            native_identity = identity
        else:
            _compare_native_identities(native_identity, identity, f"{segment['id']} step {step}")
        actual_step, actual_time = _clock(backups[step])
        expected_time = initial_time + step * dt
        if actual_step != step or not np.isclose(
            actual_time, expected_time, rtol=0.0, atol=CLOCK_ATOL
        ):
            raise RuntimeError(f"{segment['id']}: native H5 clock mismatch at step {step}")
        clocks[step] = actual_time

    xdmf_by_step = _named_steps(solution, XDMF_NAME, ".xdmf")
    if set(xdmf_by_step) != set(expected_steps):
        raise RuntimeError(f"{segment['id']}: native XDMF owners do not match H5 backups")
    for step, path in xdmf_by_step.items():
        if not np.isclose(_xdmf_time(path), clocks[step], rtol=0.0, atol=CLOCK_ATOL):
            raise RuntimeError(f"{segment['id']}: native XDMF time mismatch at step {step}")
    _check_owner_collections(clocks, solution)
    return clocks, native_identity


def _select_segment_interval(frame: pd.DataFrame, segment: dict) -> pd.DataFrame:
    interval = segment["accepted_interval"]
    mask = frame["step"] >= interval["first_step"]
    if interval["last_step"] is not None:
        mask &= frame["step"] <= interval["last_step"]
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise RuntimeError(f"{segment['id']}: required samples contain no accepted rows")
    return selected


def _check_resumed_samples(
    segments: list[dict],
    *,
    owner_steps: dict[str, list[int]],
    initial_time: float,
    dt: float,
    loading_counts: dict[str, int],
    loading_keys: dict[str, set[tuple]],
) -> None:
    """Validate per-segment samples and their exact non-overlapping union."""
    selected: dict[str, dict[str, pd.DataFrame]] = {}
    for segment in segments:
        last_step = segment["accepted_interval"]["last_step"]
        segment_frames: dict[str, pd.DataFrame] = {}
        for name in REQUIRED_CSVS:
            frame = _check_csv(
                segment["samples_path"] / name,
                last_step,
                initial_time,
                dt,
            )
            segment_frames[name] = _select_segment_interval(frame, segment)
        selected[segment["id"]] = segment_frames

    expected_by_name: dict[str, set] = {}
    station_columns_by_name: dict[str, list[str]] = {}
    observed_inventory_by_name: dict[str, set[tuple]] = {}
    for segment in segments:
        interval = segment["accepted_interval"]
        first_step = interval["first_step"]
        last_step = interval["last_step"]
        accepted_start = 1 if first_step == 0 else first_step
        accepted_steps = list(range(accepted_start, last_step + 1))
        segment_frames = selected[segment["id"]]
        flow = segment_frames["flow_integrals.csv"]
        _require_exact_steps(
            flow, owner_steps[segment["id"]], f"{segment['id']}: flow_integrals.csv"
        )
        if len(flow) != len(owner_steps[segment["id"]]):
            raise RuntimeError(
                f"{segment['id']}: flow-integral samples contain duplicate owner rows"
            )
        forces = segment_frames["vlm_forces.csv"]
        _require_exact_steps(forces, accepted_steps, f"{segment['id']}: vlm_forces.csv")
        if len(forces) != len(accepted_steps):
            raise RuntimeError(
                f"{segment['id']}: vlm_forces.csv contains duplicate accepted-step rows"
            )
        surface_forces = segment_frames["vlm_surface_forces.csv"]
        _require_force_surface_steps(
            surface_forces, accepted_steps, f"{segment['id']}: vlm_surface_forces.csv"
        )
        for name, surface in LOADING_FILES.items():
            loading = segment_frames[name]
            _require_exact_steps(loading, accepted_steps, f"{segment['id']}: {name}")
            _require_surface_steps(
                loading,
                accepted_steps,
                surface,
                f"{segment['id']}: {name}",
                loading_counts[name],
                loading_keys[name],
            )
            station_columns = _station_identity_columns(loading, f"{segment['id']}: {name}")
            previous_columns = station_columns_by_name.setdefault(name, station_columns)
            if station_columns != previous_columns:
                raise RuntimeError(f"{name}: station identity schema changes across segments")
            first_step = int(loading["step"].min())
            inventory = _station_keys(
                loading.loc[loading["step"] == first_step],
                station_columns,
                f"{segment['id']}: {name}",
            )
            # Geometry/H5 topology supplies the required cardinality. The CSV
            # identities are only observations whose stability is checked; a
            # first-row inventory is never used as the expected coverage set.
            previous_inventory = observed_inventory_by_name.setdefault(name, inventory)
            if inventory != previous_inventory:
                raise RuntimeError(f"{name}: station membership changes across segments")

        expected_by_name.setdefault("flow_integrals.csv", set()).update(owner_steps[segment["id"]])
        expected_by_name.setdefault("vlm_forces.csv", set()).update(accepted_steps)
        expected_by_name.setdefault("vlm_surface_forces.csv", set()).update(
            (step, surface) for step in accepted_steps for surface in ("front_wing", "rear_wing")
        )
        for name in LOADING_FILES:
            expected_by_name.setdefault(name, set()).update(accepted_steps)

    for name, expected in expected_by_name.items():
        frames = [selected[segment["id"]][name] for segment in segments]
        if name in {"flow_integrals.csv", "vlm_forces.csv"}:
            observed = {int(step) for frame in frames for step in frame["step"]}
            row_count = sum(len(frame) for frame in frames)
        elif name in LOADING_FILES:
            observed = {int(step) for frame in frames for step in frame["step"]}
            row_count = sum(len(frame) for frame in frames)
            expected_row_count = len(expected) * loading_counts[name]
        else:
            observed = {
                (int(step), str(surface))
                for frame in frames
                for step, surface in zip(frame["step"], frame["surface"], strict=True)
            }
            row_count = sum(len(frame) for frame in frames)
        if name in LOADING_FILES:
            if observed != expected or row_count != expected_row_count:
                raise RuntimeError(
                    f"{name}: combined segments do not cover the exact accepted history"
                )
            continue
        if observed != expected or row_count != len(expected):
            raise RuntimeError(f"{name}: combined segments do not cover the exact accepted history")


def _check_declared_resume_lineage(manifest_path: Path) -> tuple[dict, list[dict], int, float]:
    """Validate a declared fresh-prefix/restart pair and return its endpoint."""
    try:
        from tutorials.vpm.delta_wing.assets._delta_wing_plots import load_accepted_lineage
    except ModuleNotFoundError:
        from _delta_wing_plots import load_accepted_lineage

    payload = _read_metadata(manifest_path)
    try:
        segments = load_accepted_lineage(manifest_path, include_active=True)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(f"declared lineage is invalid: {manifest_path}") from error
    if len(segments) != 2:
        raise RuntimeError("declared restart lineage must contain exactly two segments")
    prefix, continuation = segments
    if prefix["status"] != "accepted" or prefix.get("origin") != "fresh_initial_value":
        raise RuntimeError("restart lineage must begin with an accepted fresh-origin prefix")
    if continuation.get("origin") != "restart_checkpoint":
        raise RuntimeError("restart lineage must end with a checkpoint-origin continuation")
    if continuation["status"] not in {"active", "accepted"}:
        raise RuntimeError("restart continuation has an unsupported status")
    if continuation["status"] == "active" and any(
        boundary.get("role") == "end" for boundary in continuation.get("boundary_checkpoints", [])
    ):
        raise RuntimeError("active restart continuation cannot claim an end boundary")
    if (
        prefix["accepted_interval"]["last_step"] + 1
        != continuation["accepted_interval"]["first_step"]
    ):
        raise RuntimeError("restart lineage intervals are not contiguous at the predecessor")

    source = payload.get("animation_source")
    expected_ids = [prefix["id"], continuation["id"]]
    if not isinstance(source, dict) or source.get("segment_ids") != expected_ids:
        raise RuntimeError("restart animation source must select prefix and continuation in order")

    try:
        root_metadata = _read_metadata(prefix["solution_path"] / "vpm_metadata.json")
        dt = float(root_metadata["configuration"]["numerics"]["time_step_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("fresh prefix metadata is missing its time step") from error
    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError("restart lineage has an invalid time step")
    # Re-read with the authoritative metadata time step after the initial
    # prefix-clock sanity check above.
    root_metadata, _, initial_time, _, _, root_interval = _segment_metadata(
        prefix, initial_time=0.0, dt=dt, require_completed=False
    )
    try:
        campaign_steps = int(root_metadata["configuration"]["run"]["steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("fresh prefix metadata is missing the campaign endpoint") from error
    if campaign_steps != CAMPAIGN_END_STEP or not np.isclose(
        initial_time + campaign_steps * dt,
        CAMPAIGN_END_TIME,
        rtol=0.0,
        atol=CLOCK_ATOL,
    ):
        raise RuntimeError("restart lineage does not declare the step-4000 / t=10 s campaign")
    continuation_interval = continuation["accepted_interval"]
    if continuation_interval["last_step"] is None:
        try:
            continuation_metadata = _read_metadata(
                continuation["solution_path"] / "vpm_metadata.json"
            )
            endpoint_step = int(continuation_metadata["state"]["step"])
            endpoint_time = float(continuation_metadata["state"]["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("continuation metadata is missing its endpoint") from error
        effective_continuation = deepcopy(continuation)
        effective_continuation["accepted_interval"] = {
            **continuation_interval,
            "last_step": endpoint_step,
            "last_time": endpoint_time,
        }
    else:
        effective_continuation = continuation
        endpoint_step = continuation_interval["last_step"]
        endpoint_time = continuation_interval["last_time"]
    if endpoint_step <= prefix["accepted_interval"]["last_step"]:
        raise RuntimeError("restart continuation endpoint does not advance the prefix")
    if endpoint_step != CAMPAIGN_END_STEP or not np.isclose(
        endpoint_time, CAMPAIGN_END_TIME, rtol=0.0, atol=CLOCK_ATOL
    ):
        raise RuntimeError("continuation does not reach the declared step-4000 / t=10 s endpoint")
    effective_segments = [prefix, effective_continuation]
    continuation_metadata, _, _, _, _, continuation_interval_steps = _segment_metadata(
        effective_continuation,
        initial_time=initial_time,
        dt=dt,
        require_completed=True,
    )
    if root_interval != continuation_interval_steps:
        raise RuntimeError("restart segments do not share the backup interval")
    if _metadata_identity(
        root_metadata, prefix["solution_path"] / "vpm_metadata.json"
    ) != _metadata_identity(
        continuation_metadata, effective_continuation["solution_path"] / "vpm_metadata.json"
    ):
        raise RuntimeError("restart segments do not share the numerical metadata identity")

    def boundary(segment: dict, role: str) -> dict:
        matches = [
            item for item in segment.get("boundary_checkpoints", []) if item.get("role") == role
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{segment['id']}: expected exactly one {role} boundary")
        return matches[0]

    predecessor = boundary(prefix, "end")
    continuation_start = boundary(continuation, "start")
    if any(
        predecessor.get(key) != continuation_start.get(key)
        for key in ("path", "sha256", "step", "time")
    ):
        raise RuntimeError("restart continuation start boundary does not match prefix end boundary")

    owner_steps = {
        segment["id"]: _segment_owner_steps(segment, root_interval)
        for segment in effective_segments
    }
    native_results = {
        segment["id"]: _check_native_segment(
            segment, interval=root_interval, initial_time=initial_time, dt=dt
        )
        for segment in effective_segments
    }
    root_clocks, root_native_identity = native_results[prefix["id"]]
    continuation_clocks, continuation_native_identity = native_results[continuation["id"]]
    _compare_native_identities(
        root_native_identity,
        continuation_native_identity,
        "restart segments",
    )
    _check_metadata_native_identity(root_metadata, root_native_identity, prefix["id"])
    _check_metadata_native_identity(
        continuation_metadata, continuation_native_identity, continuation["id"]
    )
    loading_counts = {
        name: _declared_loading_count(root_metadata, name, surface, "restart lineage")
        for name, surface in LOADING_FILES.items()
    }
    declared_panel_count = sum(
        count for name, count in loading_counts.items() if "chordwise" in name
    )
    if declared_panel_count != root_native_identity["vlm_panel_count"]:
        raise RuntimeError(
            "restart lineage: declared VLM geometry does not match native panel topology"
        )
    loading_keys = {
        name: _declared_loading_keys(
            root_metadata,
            name,
            surface,
            "restart lineage",
            [
                "wing_uid",
                "segment_uid",
                "station_id",
                "half",
                "span_index",
                *(["chord_index"] if "chordwise" in name else []),
            ],
        )
        for name, surface in LOADING_FILES.items()
    }
    clocks = {
        prefix["id"]: root_clocks,
        continuation["id"]: continuation_clocks,
    }

    def require_owner_boundary(
        segment: dict, boundary_record: dict, owner_path: Path, expected_clock: tuple[int, float]
    ) -> None:
        resolved_boundary = (segment["case_root"] / boundary_record["path"]).resolve()
        if resolved_boundary != owner_path.resolve():
            raise RuntimeError(
                f"{segment['id']}: {boundary_record['role']} boundary is not its selected owner H5"
            )
        if boundary_record["sha256"] != _sha256(owner_path):
            raise RuntimeError(f"{segment['id']}: boundary hash is not its selected owner H5 hash")
        if int(boundary_record["step"]) != expected_clock[0] or not np.isclose(
            float(boundary_record["time"]), expected_clock[1], rtol=0.0, atol=CLOCK_ATOL
        ):
            raise RuntimeError(
                f"{segment['id']}: boundary clock is not its selected owner H5 clock"
            )

    prefix_end = boundary(prefix, "end")
    continuation_start = boundary(continuation, "start")
    prefix_owner = (
        prefix["solution_path"] / f"vpm_{prefix['accepted_interval']['last_step']:06d}.h5"
    )
    prefix_clock = (
        prefix["accepted_interval"]["last_step"],
        root_clocks[prefix["accepted_interval"]["last_step"]],
    )
    require_owner_boundary(prefix, prefix_end, prefix_owner, prefix_clock)
    require_owner_boundary(continuation, continuation_start, prefix_owner, prefix_clock)
    if continuation["status"] == "accepted":
        continuation_end = boundary(continuation, "end")
        continuation_owner = continuation["solution_path"] / f"vpm_{endpoint_step:06d}.h5"
        require_owner_boundary(
            continuation,
            continuation_end,
            continuation_owner,
            (endpoint_step, continuation_clocks[endpoint_step]),
        )
    combined_clock = {
        step: time for segment_clock in clocks.values() for step, time in segment_clock.items()
    }
    ordered_times = [combined_clock[step] for step in sorted(combined_clock)]
    if len(ordered_times) < 2 or np.max(np.diff(ordered_times)) > MAX_NATIVE_GAP + CLOCK_ATOL:
        raise RuntimeError("restart native owner clock is too sparse for dense animation")
    _check_resumed_samples(
        effective_segments,
        owner_steps=owner_steps,
        initial_time=initial_time,
        dt=dt,
        loading_counts=loading_counts,
        loading_keys=loading_keys,
    )
    return payload, effective_segments, endpoint_step, endpoint_time


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def finalize_lineage(case_dir: Path = CASE_DIR) -> Path:
    """Validate a fresh run or declared restart pair and atomically accept it."""
    global CASE_DIR, SOLUTION_DIR, SAMPLES_DIR, MANIFEST_PATH
    CASE_DIR = Path(case_dir).resolve()
    SOLUTION_DIR = CASE_DIR / "solution"
    SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
    MANIFEST_PATH = CASE_DIR / "assets" / "delta_wing_accepted_lineage.json"
    manifest_payload = _read_metadata(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None
    declared_segments = manifest_payload.get("segments") if manifest_payload is not None else None
    if isinstance(declared_segments, list) and len(declared_segments) == 2:
        _, effective_segments, end_step, end_time = _check_declared_resume_lineage(MANIFEST_PATH)
        payload = deepcopy(manifest_payload)
        by_id = {segment["id"]: segment for segment in payload["segments"]}
        continuation = by_id[effective_segments[-1]["id"]]
        continuation["status"] = "accepted"
        continuation["accepted_interval"]["last_step"] = end_step
        continuation["accepted_interval"]["last_time"] = end_time
        end_path = effective_segments[-1]["solution_path"] / f"vpm_{end_step:06d}.h5"
        boundaries = continuation.setdefault("boundary_checkpoints", [])
        if not any(boundary.get("role") == "end" for boundary in boundaries):
            boundaries.append(
                {
                    "role": "end",
                    "path": str(end_path.relative_to(CASE_DIR)),
                    "sha256": _sha256(end_path),
                    "step": end_step,
                    "time": end_time,
                }
            )
        source = payload.setdefault("animation_source", {})
        source["status"] = "accepted"
        _atomic_write_json(MANIFEST_PATH, payload)
        print(f"finalized {MANIFEST_PATH} at step {end_step} / t={end_time:g} s")
        return MANIFEST_PATH
    end_step, end_time, _ = _check_completed_native_run()
    end_path = SOLUTION_DIR / f"vpm_{end_step:06d}.h5"
    payload = {
        "schema_version": 1,
        "case_root": "..",
        "purpose": "Canonical clean Delta-wing source lineage for plotting and native animation inputs.",
        "default_statuses": ["accepted"],
        "animation_source": {
            "status": "accepted",
            "segment_ids": ["clean_dense_run"],
            "selection": "single_dense_run",
            "note": "Finalized from the completed canonical native output and dense owner clock.",
        },
        "segments": [
            {
                "id": "clean_dense_run",
                "status": "accepted",
                "origin": "fresh_initial_value",
                "solution": "solution",
                "samples": "samples/delta_wing",
                "accepted_interval": {
                    "first_step": 0,
                    "last_step": end_step,
                    "first_time": 0.0,
                    "last_time": end_time,
                },
                "boundary_checkpoints": [
                    {
                        "role": "end",
                        "path": f"solution/vpm_{end_step:06d}.h5",
                        "sha256": _sha256(end_path),
                        "step": end_step,
                        "time": end_time,
                    }
                ],
            }
        ],
        "overlap_policy": "No cross-segment merges; the canonical fresh run has one owner clock.",
        "native_wake_policy": "Use only coupled owner-clocked wake frames.",
    }
    _atomic_write_json(MANIFEST_PATH, payload)
    print(f"finalized {MANIFEST_PATH} at step {end_step} / t={end_time:g} s")
    return MANIFEST_PATH


if __name__ == "__main__":
    finalize_lineage()
