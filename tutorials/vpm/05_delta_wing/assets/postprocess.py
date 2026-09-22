"""Shared scripted native-data processing for the delta-wing tutorial.

This module owns the explicit source lineage, its finalization, the CSV/sample
readers, wake-plane loading and period averaging, figure-layout helpers,
completed-run validation and the coupled native-backup GIF renderer. It exposes
no figure; the ``plot_<figure>.py`` scripts hold one thesis figure each and call
this module for data, validation and layout. ``generate_surface.py`` builds the
VLM geometry and ``setup.py`` owns the physics.
"""

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd
import pyvista as pv
from defusedxml import ElementTree
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from scipy.signal import find_peaks

from openonda import plotting as _theme

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION_DIR = CASE_DIR / "solution"
SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
FIGURES_DIR = CASE_DIR / "figures"
LINEAGE_MANIFEST = CASE_DIR / "assets" / "delta_wing_accepted_lineage.json"
DEFAULT_GIF_OUTPUT = CASE_DIR / "figures" / "delta_wing_30fps.gif"
CSV_SERIALIZATION_RTOL = 5.0e-6
CSV_SERIALIZATION_ATOL = 1.0e-7
CSV_CLOCK_ATOL = 1.0e-12
GIF_FPS = 30
MAX_NATIVE_GAP = 1.0 / GIF_FPS
BACKUP_NAME = re.compile(r"vpm_(\d+)\.h5$")
VTU_NAME = re.compile(r"vpm_(\d+)\.vtu$")
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
# Fixed screen-space projection for the moving panels. A plain x-z view is
# nearly edge-on for this geometry because the wing span is mostly y. This
# oblique projection retains chord (x), span (y), and heave/pitch (z) in one
# consistent 2-D view without inventing intermediate solver states.
OBLIQUE_PROJECTION = np.array([[1.0, 0.28, 0.0], [0.0, 0.72, 1.0]])
_SOURCE_COLUMNS = {
    "source_segment",
    "source_samples_directory",
    "source_lineage_manifest",
    "source_lineage_status",
}


@dataclass(frozen=True)
class PlotSources:
    """Resolved native sample sources and destination for one plotting run.

    Attributes
    ----------
    samples_arg : object
        ``None`` to plot the accepted lineage, otherwise a ``Path`` or list of
        sample directories to read directly.
    solution_dirs : list[Path] | None
        Matching solver directories for the wake plotters; ``None`` for the
        accepted lineage, a one-element list following an active continuation.
    destination : Path
        ``figures/`` for a completed run, otherwise ``figures/partial/``.
    complete : bool
        True only when the native lifecycle status is ``completed``.
    status : str
        Native lifecycle status read from solver metadata.
    state : dict
        Native saved-state clock with ``step`` and ``time`` keys.
    """

    samples_arg: object
    solution_dirs: list[Path] | None
    destination: Path
    complete: bool
    status: str
    state: dict


def resolve_plot_sources(case_dir: Path = CASE_DIR, figures_dir: Path | None = None) -> PlotSources:
    """Resolve native sample sources and the destination for figure scripts.

    Reads ``solution/vpm_metadata.json`` from the case root, or from the final
    active segment when the lineage manifest declares a sparse-to-dense pair.
    A completed native lifecycle targets ``figures/`` with the accepted lineage;
    interrupted or failed runs target ``figures/partial/`` with the currently
    written samples. Plot scripts never promote lineage or render the animation
    through this resolver.

    Parameters
    ----------
    case_dir : Path
        Case root containing ``solution/`` and ``assets/``.
    figures_dir : Path | None
        Target figures directory; defaults to ``case_dir / "figures"``.

    Returns
    -------
    PlotSources
        The resolved sample argument, matching solution directories, the write
        destination and completion flag.

    Notes
    -----
    Prints the partial-run diagnostic for unfinished native output; a completed
    run stays silent so full-run results are free of status noise.
    """
    if figures_dir is None:
        figures_dir = case_dir / "figures"
    segments = None
    metadata_path = case_dir / "solution" / "vpm_metadata.json"
    manifest = case_dir / "assets" / "delta_wing_accepted_lineage.json"
    if manifest.is_file() and len(json.loads(manifest.read_text()).get("segments", [])) == 2:
        segments = load_accepted_lineage(manifest, include_active=True)
        metadata_path = segments[-1]["solution_path"] / "vpm_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    status = metadata["lifecycle"]["status"]
    complete = status == "completed"
    if complete:
        return PlotSources(None, None, figures_dir, True, status, metadata["state"])
    state = metadata["state"]
    samples = (
        segments[-1]["samples_path"]
        if segments is not None
        else case_dir / "samples" / "delta_wing"
    )
    print(f"Run status: {status}; saved state step {state['step']}, t={state['time']:g} s.")
    print(
        "Writing partial diagnostics; cycle-mean wake and final animation require a completed run."
    )
    return PlotSources(
        samples, [metadata_path.parent], figures_dir / "partial", False, status, state
    )


def _integer_step(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label}: step must be a finite integer")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label}: step must be a finite integer")
    return int(numeric)


def _finite_time(value, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{label}: time must be finite numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"{label}: time must be finite numeric")
    return numeric


def _validate_selected_clock(step, time, interval: dict, label: str) -> tuple[int, float]:
    step_value = _integer_step(step, label)
    time_value = _finite_time(time, label)
    if step_value < interval["first_step"] or (
        interval["last_step"] is not None and step_value > interval["last_step"]
    ):
        raise ValueError(f"{label}: step {step_value} is outside its lineage interval")
    if time_value < interval["first_time"] - CSV_CLOCK_ATOL or (
        interval["last_time"] is not None and time_value > interval["last_time"] + CSV_CLOCK_ATOL
    ):
        raise ValueError(f"{label}: time {time_value:g} is outside its lineage interval")
    return step_value, time_value


def _validate_selected_csv_clocks(frame: pd.DataFrame, filename: str, label: str) -> None:
    """Validate ordered step clocks while allowing agreeing surface rows."""
    steps = []
    times = []
    for step, time in zip(frame["step"], frame["time"], strict=True):
        steps.append(_integer_step(step, f"{label} {filename}"))
        times.append(_finite_time(time, f"{label} {filename}"))
    if len(steps) < 2:
        return
    if np.any(np.diff(steps) < 0):
        raise ValueError(f"{label}: {filename} step clocks must be nondecreasing")
    previous_step = steps[0]
    previous_time = times[0]
    for step, time in zip(steps[1:], times[1:], strict=True):
        if step == previous_step:
            if not np.isclose(time, previous_time, rtol=0.0, atol=CSV_CLOCK_ATOL):
                raise ValueError(f"{label}: {filename} has multiple timestamps for step {step}")
            continue
        if time <= previous_time + CSV_CLOCK_ATOL:
            raise ValueError(f"{label}: {filename} timestamps must increase between distinct steps")
        previous_step = step
        previous_time = time


def _resolve_case_relative(value: str | Path, case_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"lineage path must be repository-relative, got {value!r}")
    resolved_root = case_root.resolve()
    resolved = (resolved_root / path).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"lineage path escapes case root: {value!r}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_boundary_clock(path: Path) -> tuple[int, float]:
    with h5py.File(path, "r") as archive:
        if "solver" not in archive:
            raise ValueError(f"{path}: missing solver group")
        solver = archive["solver"]
        try:
            step = _integer_step(solver.attrs["step"], f"{path}: solver step")
            time = _finite_time(solver.attrs["time"], f"{path}: solver time")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: missing native solver clock") from error
    if step < 0:
        raise ValueError(f"{path}: invalid native solver clock")
    return step, time


def _validate_boundary(boundary: dict, case_root: Path, interval: dict, segment_id: str) -> None:
    try:
        path_value = boundary["path"]
        expected_hash = boundary["sha256"]
        expected_step = _integer_step(boundary["step"], f"{segment_id}: boundary step")
        expected_time = _finite_time(boundary["time"], f"{segment_id}: boundary time")
        role = boundary["role"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{segment_id}: malformed boundary checkpoint") from error
    if role not in {"start", "end"}:
        raise ValueError(f"{segment_id}: unsupported boundary role {role!r}")
    path = _resolve_case_relative(path_value, case_root)
    if not path.is_file():
        raise FileNotFoundError(f"{segment_id}: missing boundary checkpoint {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{segment_id}: boundary checkpoint {path_value!r} hash mismatch "
            f"(expected {expected_hash}, got {actual_hash})"
        )
    actual_step, actual_time = _native_boundary_clock(path)
    if actual_step != expected_step or not np.isclose(
        actual_time, expected_time, rtol=0.0, atol=CSV_CLOCK_ATOL
    ):
        raise ValueError(
            f"{segment_id}: boundary checkpoint {path_value!r} clock mismatch "
            f"(expected step={expected_step}, time={expected_time:g}; "
            f"got step={actual_step}, time={actual_time:g})"
        )
    if role == "end" and expected_step != interval["last_step"]:
        raise ValueError(f"{segment_id}: end boundary does not match accepted interval")
    if role == "end" and not np.isclose(
        expected_time, interval["last_time"], rtol=0.0, atol=CSV_CLOCK_ATOL
    ):
        raise ValueError(f"{segment_id}: end boundary time does not match accepted interval")
    if role == "start" and expected_step != interval["first_step"] - 1:
        raise ValueError(
            f"{segment_id}: start boundary is not immediately before accepted interval"
        )
    if role == "start" and expected_time > interval["first_time"] + CSV_CLOCK_ATOL:
        raise ValueError(f"{segment_id}: start boundary is after accepted interval start")


def load_accepted_lineage(
    manifest_path: Path = LINEAGE_MANIFEST, *, include_active: bool = False
) -> list[dict]:
    """Load and validate explicit case-relative plotting/animation lineage."""
    manifest_path = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"lineage manifest not found: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"malformed lineage manifest {manifest_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported lineage schema in {manifest_path}")
    try:
        case_root_value = Path(payload["case_root"])
        if case_root_value.is_absolute():
            raise ValueError("case_root must be relative to the manifest")
        case_root = (manifest_path.parent / case_root_value).resolve()
        raw_segments = payload["segments"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"malformed lineage manifest {manifest_path}") from error
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("lineage manifest must contain at least one segment")

    validated = []
    seen_ids = set()
    previous_end = None
    previous_time = None
    active_seen = False
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise ValueError("lineage segments must be objects")
        segment = dict(raw_segment)
        segment_id = segment.get("id")
        status = segment.get("status")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError("lineage segment ids must be non-empty strings")
        if segment_id in seen_ids:
            raise ValueError(f"duplicate lineage segment id {segment_id!r}")
        seen_ids.add(segment_id)
        if status not in {"accepted", "active"}:
            raise ValueError(f"{segment_id}: unsupported lineage status {status!r}")
        if active_seen:
            raise ValueError("active lineage segment must be the final segment")
        if status == "active":
            active_seen = True
        origin = segment.get("origin")
        if origin is not None and origin not in {"fresh_initial_value", "restart_checkpoint"}:
            raise ValueError(f"{segment_id}: unsupported lineage origin {origin!r}")
        try:
            interval = dict(segment["accepted_interval"])
            first_step = _integer_step(interval["first_step"], f"{segment_id}: first_step")
            last_step_value = interval.get("last_step")
            last_step = (
                None
                if last_step_value is None
                else _integer_step(last_step_value, f"{segment_id}: last_step")
            )
            first_time = _finite_time(interval["first_time"], f"{segment_id}: first_time")
            last_time_value = interval.get("last_time")
            last_time = (
                None
                if last_time_value is None
                else _finite_time(last_time_value, f"{segment_id}: last_time")
            )
            solution_value = segment["solution"]
            samples_value = segment["samples"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{segment_id}: malformed accepted interval or path") from error
        if first_step < 0 or (last_step is not None and last_step < first_step):
            raise ValueError(f"{segment_id}: invalid accepted step interval")
        if not np.isfinite(first_time) or (last_time is not None and not np.isfinite(last_time)):
            raise ValueError(f"{segment_id}: invalid accepted time interval")
        if last_time is not None and last_time < first_time:
            raise ValueError(f"{segment_id}: accepted times are reversed")
        if last_step is None and status != "active":
            raise ValueError(f"{segment_id}: accepted segments need a finite last step")
        if last_time is None and status != "active":
            raise ValueError(f"{segment_id}: accepted segments need a finite last time")
        fresh_origin = origin == "fresh_initial_value"
        if fresh_origin and (
            first_step != 0 or not np.isclose(first_time, 0.0, rtol=0.0, atol=CSV_CLOCK_ATOL)
        ):
            raise ValueError(
                f"{segment_id}: fresh initial-value origin must begin at step 0 and time 0"
            )
        if previous_end is not None and first_step <= previous_end:
            raise ValueError(f"{segment_id}: accepted intervals overlap or are unordered")
        if previous_time is not None and first_time <= previous_time:
            raise ValueError(f"{segment_id}: accepted clocks are unordered or overlapping")
        previous_end = last_step
        if last_step is None:
            previous_end = first_step - 1
        previous_time = last_time if last_time is not None else first_time

        solution_path = _resolve_case_relative(solution_value, case_root)
        samples_path = _resolve_case_relative(samples_value, case_root)
        if status == "accepted" and (not solution_path.is_dir() or not samples_path.is_dir()):
            raise FileNotFoundError(
                f"{segment_id}: accepted source directory missing "
                f"(solution={solution_path}, samples={samples_path})"
            )
        boundaries = segment.get("boundary_checkpoints", [])
        if not isinstance(boundaries, list):
            raise ValueError(f"{segment_id}: boundary_checkpoints must be a list")
        interval_for_validation = {
            "first_step": first_step,
            "last_step": last_step,
            "first_time": first_time,
            "last_time": last_time,
        }
        for boundary in boundaries:
            if not isinstance(boundary, dict):
                raise ValueError(f"{segment_id}: malformed boundary checkpoint")
            _validate_boundary(boundary, case_root, interval_for_validation, segment_id)
        if fresh_origin and any(boundary.get("role") == "start" for boundary in boundaries):
            raise ValueError(
                f"{segment_id}: fresh initial-value origin cannot have a start boundary"
            )
        if status == "accepted" and not any(
            boundary.get("role") == "end" for boundary in boundaries
        ):
            raise ValueError(f"{segment_id}: accepted segment needs an end boundary checkpoint")
        if (
            status == "active"
            and not fresh_origin
            and not any(boundary.get("role") == "start" for boundary in boundaries)
        ):
            raise ValueError(f"{segment_id}: active segment needs a start boundary checkpoint")
        segment["accepted_interval"] = {
            "first_step": first_step,
            "last_step": last_step,
            "first_time": first_time,
            "last_time": last_time,
        }
        segment["solution_path"] = solution_path
        segment["samples_path"] = samples_path
        segment["manifest_path"] = manifest_path
        segment["case_root"] = case_root
        validated.append(segment)

    selected_statuses = {"accepted"}
    if include_active:
        selected_statuses.add("active")
    return [segment for segment in validated if segment["status"] in selected_statuses]


def load_animation_lineage(
    manifest_path: Path = LINEAGE_MANIFEST,
) -> list[dict]:
    """Return the finalized dense segment(s) selected for the final GIF."""
    manifest_path = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"lineage manifest not found: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"malformed lineage manifest {manifest_path}")
    source = payload.get("animation_source")
    if not isinstance(source, dict) or source.get("status") != "accepted":
        raise ValueError(
            "dense animation source is not finalized; keep the active continuation out of the final GIF"
        )
    segment_ids = source.get("segment_ids")
    if (
        not isinstance(segment_ids, list)
        or not segment_ids
        or not all(isinstance(segment_id, str) and segment_id for segment_id in segment_ids)
    ):
        raise ValueError("animation_source.segment_ids must name at least one segment")
    segments = load_accepted_lineage(manifest_path)
    by_id = {segment["id"]: segment for segment in segments}
    if any(segment_id not in by_id for segment_id in segment_ids):
        raise ValueError("animation source must name finalized accepted lineage segments")
    selected = [by_id[segment_id] for segment_id in segment_ids]
    for left, right in zip(selected, selected[1:], strict=False):
        if right["accepted_interval"]["first_step"] <= left["accepted_interval"]["last_step"]:
            raise ValueError("animation source segments overlap or are unordered")
        if right["accepted_interval"]["first_time"] <= left["accepted_interval"]["last_time"]:
            raise ValueError("animation source clocks overlap or are unordered")
    return selected


def _read_lineage_csv(filename: str, manifest_path: Path = LINEAGE_MANIFEST) -> pd.DataFrame:
    segments = load_accepted_lineage(manifest_path)
    frames = []
    for segment in segments:
        path = segment["samples_path"] / filename
        if not path.is_file():
            raise FileNotFoundError(f"{segment['id']}: missing {filename} at {path}")
        frame = pd.read_csv(path)
        _validate_selected_csv_clocks(frame, filename, segment["id"])
        if "step" not in frame or "time" not in frame:
            raise ValueError(f"{segment['id']}: {filename} must contain step and time columns")
        steps = pd.to_numeric(frame["step"], errors="coerce")
        if steps.isna().any() or not np.isfinite(steps).all():
            raise ValueError(f"{segment['id']}: {filename} contains invalid step values")
        interval = segment["accepted_interval"]
        if (steps % 1 != 0).any():
            raise ValueError(f"{segment['id']}: {filename} contains non-integer step values")
        upper = True if interval["last_step"] is None else steps <= interval["last_step"]
        frame = frame.loc[(steps >= interval["first_step"]) & upper].copy()
        if frame.empty:
            continue
        for step, time in zip(steps.loc[frame.index], frame["time"], strict=True):
            _validate_selected_clock(step, time, interval, f"{segment['id']} {filename}")
        frame["source_segment"] = segment["id"]
        frame["source_samples_directory"] = segment["samples"]
        frame["source_lineage_manifest"] = str(
            Path(segment["manifest_path"]).relative_to(segment["case_root"])
        )
        frame["source_lineage_status"] = segment["status"]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no accepted {filename} rows found")
    key = ["step", "surface"] if "surface" in frames[0] else ["step"]
    _validate_duplicate_csv_rows(filename, frames, key)
    combined = pd.concat(frames, ignore_index=True)
    _validate_selected_csv_clocks(combined, filename, "accepted lineage")
    if combined.duplicated(key, keep=False).any():
        raise ValueError(f"{filename}: duplicate accepted lineage key(s) after interval selection")
    return combined.sort_values(
        ["time", "step"] + (["surface"] if "surface" in combined else [])
    ).reset_index(drop=True)


def _directories(value) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    directories = [Path(directory) for directory in value]
    if not directories:
        raise ValueError("at least one output directory is required")
    return directories


def _segment_name(directory: Path) -> str:
    return directory.name or str(directory)


def _csv_values_match(left, right, *, clock: bool = False) -> bool:
    numeric = isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    )
    if clock:
        # Accepted clocks must be finite numeric values. Missing or textual
        # clocks are malformed even when both segments carry the same value.
        if (
            not numeric
            or isinstance(left, (bool, np.bool_))
            or isinstance(right, (bool, np.bool_))
            or not np.isfinite(left)
            or not np.isfinite(right)
        ):
            return False
        if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
            return int(left) == int(right)
        return bool(np.isclose(left, right, rtol=0.0, atol=CSV_CLOCK_ATOL, equal_nan=False))

    left_missing, right_missing = pd.isna(left), pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    if numeric:
        if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
            return int(left) == int(right)
        return bool(
            np.isfinite(left)
            and np.isfinite(right)
            and np.isclose(
                left,
                right,
                rtol=CSV_SERIALIZATION_RTOL,
                atol=CSV_SERIALIZATION_ATOL,
                equal_nan=False,
            )
        )
    return type(left) is type(right) and left == right


def _compare_csv_rows(filename: str, key_value, left, right, columns) -> None:
    for column in columns:
        clock = column == "time"
        if not _csv_values_match(left[column], right[column], clock=clock):
            tolerance = (
                f"clock atol={CSV_CLOCK_ATOL:g}"
                if clock
                else f"serialization tolerance rtol={CSV_SERIALIZATION_RTOL:g}, "
                f"atol={CSV_SERIALIZATION_ATOL:g}"
            )
            raise ValueError(
                f"{filename}: overlapping sample key {key_value!r} conflicts in {column} "
                f"({tolerance})"
            )


def _rows_by_key(frame: pd.DataFrame, key: list[str]) -> dict[tuple, pd.DataFrame]:
    rows = {}
    for key_value, group in frame.groupby(key, sort=False, dropna=False):
        if not isinstance(key_value, tuple):
            key_value = (key_value,)
        rows[key_value] = group
    return rows


def _validate_duplicate_csv_rows(filename: str, frames: list[pd.DataFrame], key: list[str]) -> None:
    """Validate common physical columns before dropping repeated handoff rows."""
    for frame in frames:
        frame_rows = _rows_by_key(frame, key)
        physical_columns = [column for column in frame.columns if column not in _SOURCE_COLUMNS]
        for key_value, rows in frame_rows.items():
            reference = rows.iloc[0]
            for index in range(1, len(rows)):
                _compare_csv_rows(
                    filename, key_value, reference, rows.iloc[index], physical_columns
                )

    for left_index, left_frame in enumerate(frames[:-1]):
        left_rows = _rows_by_key(left_frame, key)
        for right_frame in frames[left_index + 1 :]:
            right_rows = _rows_by_key(right_frame, key)
            common_keys = left_rows.keys() & right_rows.keys()
            columns = [
                column
                for column in left_frame.columns.intersection(right_frame.columns)
                if column not in _SOURCE_COLUMNS
            ]
            for key_value in common_keys:
                _compare_csv_rows(
                    filename,
                    key_value,
                    left_rows[key_value].iloc[0],
                    right_rows[key_value].iloc[0],
                    columns,
                )


def _read_segment_csv(filename: str, samples_dirs) -> pd.DataFrame:
    frames = []
    for directory in _directories(samples_dirs):
        path = directory / filename
        frame = pd.read_csv(path)
        frame["source_segment"] = _segment_name(directory)
        frame["source_samples_directory"] = str(directory)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no {filename} files found")
    key = ["step", "surface"] if "surface" in frames[0] else ["step"]
    _validate_duplicate_csv_rows(filename, frames, key)
    combined = pd.concat(frames, ignore_index=True)
    # A resumed run may emit an initial sample at the handoff step. Prefer the
    # final (dense continuation) segment for that one duplicate clock while
    # retaining the source label for every surviving row.
    combined = combined.drop_duplicates(key, keep="last")
    return combined.sort_values(
        ["time", "step"] + (["surface"] if "surface" in combined else [])
    ).reset_index(drop=True)


def force_history(samples_dir=None):
    if samples_dir is None:
        return _read_lineage_csv("vlm_surface_forces.csv")
    return _read_segment_csv("vlm_surface_forces.csv", samples_dir)


def flow_integrals(samples_dir=None):
    if samples_dir is None:
        return _read_lineage_csv("flow_integrals.csv")
    return _read_segment_csv("flow_integrals.csv", samples_dir)


def _wake_values_match(left, right) -> bool:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    numeric = left_array.dtype.kind in "biufc" and right_array.dtype.kind in "biufc"
    if not numeric:
        return np.array_equal(left_array, right_array)
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        return False
    if left_array.dtype.kind in "biu" or right_array.dtype.kind in "biu":
        return np.array_equal(left_array, right_array)
    return bool(
        np.allclose(
            left_array,
            right_array,
            rtol=CSV_SERIALIZATION_RTOL,
            atol=CSV_SERIALIZATION_ATOL,
            equal_nan=False,
        )
    )


def motion_period(data):
    """Measure the prescribed period from the solver's sampled heave velocity."""
    front = data[data.surface == "front_wing"].sort_values("time")
    peaks, _ = find_peaks(front.translation_velocity_z)
    if len(peaks) < 2:
        raise ValueError("At least two sampled heave-velocity peaks are needed to measure a period")
    return float(np.median(np.diff(front.time.to_numpy()[peaks])))


def last_cycles(data, period, count=3):
    """Yield complete cycles; a sample exactly on the endpoint belongs to the prior cycle."""
    times = np.sort(data.time.unique())
    cadence = np.median(np.diff(times))
    first = max(0, int(np.ceil((times[0] - cadence - 1e-9) / period)))
    last = int(np.floor((data.time.max() + 1e-9) / period))
    for cycle in range(max(first, last - count), last):
        rows = data[
            (data.time > cycle * period + 1e-9) & (data.time <= (cycle + 1) * period + 1e-9)
        ]
        yield cycle, (rows.time.to_numpy() - cycle * period) / period, rows


def _save_figure(fig, axes, path, figure_format, *, fit=True):
    """Export the fixed thesis canvas without automatic cropping or relayout."""
    if fit:
        _theme.fit_thesis_y_label_margins(fig, axes)
    _theme.validate_thesis_figure(fig, axes)
    path = Path(path).with_suffix(f".{figure_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_theme.DEFAULT_DPI, bbox_inches=None)
    plt.close(fig)
    print(f"wrote {path}")


def _available_period(data):
    front = data[data.surface == "front_wing"].sort_values("time")
    peaks, _ = find_peaks(front.translation_velocity_z)
    return motion_period(data) if len(peaks) >= 2 else None


def _solution_directories_for_samples(samples_dirs, solution_dirs=None) -> list[Path]:
    if solution_dirs is not None:
        solutions = _directories(solution_dirs)
        if len(solutions) != len(_directories(samples_dirs)):
            raise ValueError("one solution directory is required for each sample directory")
        return solutions
    return [CASE_DIR / "solution"]


def _wake_frame_step(path: Path) -> int:
    match = re.search(r"_(\d+)\.vts$", path.name)
    if match is None:
        raise ValueError(f"invalid native wake frame name: {path.name}")
    return int(match.group(1))


def _wake_frames(
    samples_dirs, plane_name: str, lineage_segments: list[dict] | None = None
) -> list[tuple[float, Path, str]]:
    records = []
    intervals = {}
    if lineage_segments is not None:
        intervals = {
            segment["samples_path"].resolve(): (
                segment["accepted_interval"],
                segment["id"],
            )
            for segment in lineage_segments
        }
    for directory in _directories(samples_dirs):
        pvd = directory / plane_name
        if not pvd.is_file():
            continue
        lineage = intervals.get(directory.resolve())
        for frame in ElementTree.parse(pvd).findall(".//DataSet"):
            frame_path = pvd.parent / frame.attrib["file"]
            frame_time = _finite_time(
                float(frame.attrib["timestep"]), f"{plane_name} {frame_path.name}"
            )
            if lineage is not None:
                interval, segment = lineage
                step = _wake_frame_step(frame_path)
                if step < interval["first_step"] or (
                    interval["last_step"] is not None and step > interval["last_step"]
                ):
                    continue
                _validate_selected_clock(step, frame_time, interval, f"{segment} {plane_name}")
            else:
                segment = _segment_name(directory)
            records.append(
                (
                    frame_time,
                    frame_path,
                    segment,
                )
            )
    if not records:
        raise FileNotFoundError(f"no {plane_name} files found")
    records.sort(key=lambda item: item[0])
    if lineage_segments is not None:
        wake_steps = np.asarray([_wake_frame_step(path) for _, path, _ in records])
        wake_times = np.asarray([time for time, _, _ in records])
        if np.any(np.diff(wake_steps) <= 0) or np.any(np.diff(wake_times) <= 0):
            raise ValueError(f"{plane_name}: accepted wake steps or timestamps are unordered")
    deduplicated = {}
    for time, path, segment in records:
        matching_time = next(
            (
                existing_time
                for existing_time in deduplicated
                if np.isclose(existing_time, time, rtol=0.0, atol=1.0e-12)
            ),
            None,
        )
        if matching_time is not None:
            if lineage_segments is not None:
                raise ValueError(
                    f"{plane_name}: accepted lineage contains overlapping wake frame at t={time:g}"
                )
            previous_time, previous_path, _ = deduplicated[matching_time]
            previous_grid = pv.read(previous_path)
            current_grid = pv.read(path)
            if not _wake_values_match(previous_grid.points, current_grid.points):
                raise ValueError(
                    f"{plane_name}: overlapping wake frame at t={time:g} conflicts in points"
                )
            for field_name in set(previous_grid.point_data) & set(current_grid.point_data):
                if not _wake_values_match(
                    previous_grid.point_data[field_name], current_grid.point_data[field_name]
                ):
                    raise ValueError(
                        f"{plane_name}: overlapping wake frame at t={time:g} "
                        f"conflicts in point field {field_name}"
                    )
            for field_name in set(previous_grid.cell_data) & set(current_grid.cell_data):
                if not _wake_values_match(
                    previous_grid.cell_data[field_name], current_grid.cell_data[field_name]
                ):
                    raise ValueError(
                        f"{plane_name}: overlapping wake frame at t={time:g} "
                        f"conflicts in cell field {field_name}"
                    )
            time = matching_time
        deduplicated[time] = (time, path, segment)
    return [deduplicated[time] for time in sorted(deduplicated)]


def _read_freestream_velocity(solution_dirs) -> np.ndarray:
    velocities = []
    for directory in _directories(solution_dirs):
        metadata_path = directory / "vpm_metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text())
        velocities.append(np.asarray(metadata["configuration"]["numerics"]["freestream_velocity"]))
    if not velocities:
        raise FileNotFoundError("no solution metadata found for wake plot")
    if any(not np.array_equal(velocity, velocities[0]) for velocity in velocities[1:]):
        raise ValueError("sparse and dense segments use different freestream velocities")
    return velocities[0]


def _wake_average(collections, end, period):
    """Integrate exactly one sampled period, including its boundary values."""
    start = end - period
    records = []
    for frames in collections:
        times = np.asarray([time for time, _, _ in frames])
        if times[0] > start + CSV_CLOCK_ATOL or times[-1] < end - CSV_CLOCK_ATOL:
            raise ValueError("wake samples do not cover a full measured heave period")
        first = max(0, np.searchsorted(times, start, side="right") - 1)
        last = np.searchsorted(times, end, side="left")
        selected = frames[first : last + 1]
        grids = [pv.read(path) for _, path, _ in selected]
        points = grids[0].points
        if any(not np.array_equal(grid.points, points) for grid in grids[1:]):
            raise ValueError("wake sampling grid changed during the averaging interval")
        fields = np.asarray([grid["velocity"] for grid in grids])
        selected_times = times[first : last + 1].copy()
        # Interpolate only the integration endpoints between adjacent native samples.
        # No extrapolation or synthetic solver states are used.
        for index, boundary, neighbour in ((0, start, 1), (-1, end, -2)):
            fraction = (boundary - selected_times[index]) / (
                selected_times[neighbour] - selected_times[index]
            )
            fields[index] = fields[index] + fraction * (fields[neighbour] - fields[index])
            selected_times[index] = boundary
        mean = trapezoid(fields, selected_times, axis=0) / period
        records.append((points, mean))
    return records


def _plot_wake_field(
    samples_arg,
    destination: Path,
    figure_format: str,
    field_name: str,
    *,
    solution_dirs=None,
    partial: bool = False,
) -> None:
    """Export one thesis wake-plane figure from native stational field samples.

    Parameters
    ----------
    samples_arg : object
        ``None`` to plot the accepted lineage, otherwise sample directories to
        read directly.
    destination : Path
        ``figures/`` or ``figures/partial/`` directory for the exported figure.
    figure_format : str
        Export extension, ``png`` or ``pdf``.
    field_name : str
        Wake field to draw: ``streamwise`` or ``vertical``.
    solution_dirs : list[Path] | None
        Matching solver directories when ``samples_arg`` is explicit; ``None``
        uses the accepted-lineage solution directories.
    partial : bool
        True to plot the latest common native wake timestamp instead of the
        measured-period mean. Partial output never certifies completion.

    Notes
    -----
    Completed-run fields are integrated over one full measured heave period
    with trapezoidal weights and linear interpolation only at the integration
    endpoints. Insufficient temporal coverage is rejected rather than plotted.
    """

    _theme.set_thesis_style()
    lineage_segments = None
    if samples_arg is None:
        lineage_segments = load_accepted_lineage()
        samples_dirs = [segment["samples_path"] for segment in lineage_segments]
        solution_dirs = (
            [segment["solution_path"] for segment in lineage_segments]
            if solution_dirs is None
            else solution_dirs
        )
    else:
        samples_dirs = _directories(samples_arg)
        solution_dirs = _solution_directories_for_samples(samples_dirs, solution_dirs)
    velocity = _read_freestream_velocity(solution_dirs)
    speed = np.linalg.norm(velocity)
    direction = velocity / speed
    data = force_history(None if lineage_segments is not None else samples_dirs)
    period = _available_period(data)
    plane_names = sorted(
        {path.name for directory in samples_dirs for path in directory.glob("wake_*span.pvd")}
    )
    if len(plane_names) != 3:
        raise ValueError(f"expected three wake planes, found {plane_names}")
    collections = [_wake_frames(samples_dirs, name, lineage_segments) for name in plane_names]
    end = min(frames[-1][0] for frames in collections)
    if partial:
        records = []
        for frames in collections:
            matching = [path for time, path, _ in frames if abs(time - end) <= CSV_CLOCK_ATOL]
            if len(matching) != 1:
                raise ValueError("no common native wake timestamp across all three planes")
            grid = pv.read(matching[0])
            records.append((grid.points, np.asarray(grid["velocity"])))
        title = f"Partial run: $t = {end:g}$ s"
    else:
        if period is None:
            raise ValueError("two sampled heave-velocity peaks are required for a mean wake")
        records = _wake_average(collections, end, period)
        title = f"Mean: $t = {end - period:g}$--${end:g}$ s"
    records.sort(key=lambda row: row[0][0] @ direction)
    axial = [field @ direction / speed for _, field in records]
    vertical = [field[:, 2] / speed for _, field in records]
    sequential = LinearSegmentedColormap.from_list(
        "thesis_teal", ["white", _theme.COLORS["TUDcyan"]]
    )
    diverging = LinearSegmentedColormap.from_list(
        "thesis_signed", [_theme.COLORS["TUDcyan"], "white", _theme.COLORS["VPMpurple"]]
    )
    if field_name == "streamwise":
        values, cmap, label = axial, sequential, r"$u_{\parallel}/U_\infty$"
    elif field_name == "vertical":
        values, cmap, label = vertical, diverging, r"$u_z/U_\infty$"
    else:
        raise ValueError(f"unsupported wake field {field_name!r}")
    limits = (min(v.min() for v in values), max(v.max() for v in values))
    if field_name == "vertical":
        limit = max(abs(limits[0]), abs(limits[1]), 1e-8)
        limits = (-limit, limit)
    elif limits[1] - limits[0] < 1e-8:
        limits = (limits[0] - 1e-8, limits[1] + 1e-8)
    fig, axes = plt.subplots(
        1, 3, sharex=True, sharey=True, figsize=(12.5 * _theme.CM, 7.6 * _theme.CM)
    )
    _theme.centered_subplots_adjust(fig, outer=0.15, bottom=0.40, top=0.83, wspace=0.13)
    for ax, (points, _), field in zip(axes, records, values, strict=True):
        artist = ax.tricontourf(
            points[:, 1], points[:, 2], field, levels=np.linspace(*limits, 25), cmap=cmap
        )
        ax.set(xlabel="$y$ [m]", title=f"$x={points[0, 0]:g}$ m")
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-1, 0])
    axes[0].set_ylabel("$z$ [m]")
    # Measure the y labels first; preserve equal physical aspect by choosing
    # the height from the resulting panel width, then place the colorbar.
    _theme.fit_thesis_y_label_margins(fig, axes)
    outer = axes[0].get_position().x0
    panel_width_cm = axes[0].get_position().width * 12.5
    aspect = np.ptp(records[0][0][:, 2]) / np.ptp(records[0][0][:, 1])
    panel_height_cm = panel_width_cm * aspect
    height_cm = panel_height_cm + 4.1
    fig.set_size_inches(12.5 * _theme.CM, height_cm * _theme.CM, forward=False)
    fig.subplots_adjust(bottom=2.8 / height_cm, top=1 - 1.3 / height_cm)
    for ax in axes:
        ax.set_aspect("equal")
    cax = fig.add_axes([outer, 1.3 / height_cm, 1 - 2 * outer, 0.22 / height_cm])
    fig.colorbar(
        artist,
        cax=cax,
        orientation="horizontal",
        ticks=np.linspace(*limits, 3),
        format="%.2f",
        label=label,
    )
    fig.suptitle(title, y=1 - 0.22 / height_cm)
    _save_figure(
        fig,
        (*axes, cax),
        destination / f"delta_wing_wake_{field_name}.png",
        figure_format,
        fit=False,
    )


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
    if not np.isfinite(vlm_time) or not np.isclose(vlm_time, time, rtol=0.0, atol=CSV_CLOCK_ATOL):
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
    if backup_format not in {"10.0", "10.1"}:
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
            equal = np.isclose(
                float(expected[key]), float(found[key]), rtol=0.0, atol=CSV_CLOCK_ATOL
            )
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
        atol=CSV_CLOCK_ATOL,
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


def _vtu_time(path: Path) -> float:
    try:
        dataset = pv.read(path)
        time_value = np.asarray(dataset.field_data["TimeValue"], dtype=float)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"{path}: missing or malformed VTU TimeValue") from error
    if time_value.size != 1 or not np.isfinite(time_value).all():
        raise RuntimeError(f"{path}: VTU TimeValue must contain one finite value")
    return float(time_value[0])


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
    vtp_by_step = _named_steps(directory / "vlm", VTP_NAME, ".vtp")
    if set(vtp_by_step) != set(expected):
        raise RuntimeError("native VTP owners do not exactly match coupled H5 backups")
    for step, path in vtp_by_step.items():
        time, time_value = _vtp_time(path)
        if not np.isclose(time, expected[step], rtol=0.0, atol=CSV_CLOCK_ATOL):
            raise RuntimeError(f"native VTP time mismatch at step {step}")
        if not np.isclose(time_value, expected[step], rtol=0.0, atol=CSV_CLOCK_ATOL):
            raise RuntimeError(f"native VTP TimeValue mismatch at step {step}")
    pvd = directory / "vlm.pvd"
    if not pvd.is_file():
        raise RuntimeError(f"missing native VLM collection {pvd}")
    rows = _collection_rows(pvd)
    if len(rows) != len(expected):
        raise RuntimeError("native VLM collection does not contain one entry per coupled backup")
    row_steps = []
    for time, filename in rows:
        match = VTP_NAME.fullmatch(Path(filename).name)
        if match is None or int(match.group(1)) not in expected:
            raise RuntimeError(f"native VLM collection references unexpected owner {filename!r}")
        step = int(match.group(1))
        row_steps.append(step)
        if not np.isclose(time, expected[step], rtol=0.0, atol=CSV_CLOCK_ATOL):
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
    if not np.isclose(
        times.to_numpy(dtype=float), expected_times, rtol=0.0, atol=CSV_CLOCK_ATOL
    ).all():
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
    if initial_step != 0 or not np.isclose(initial_time, 0.0, rtol=0.0, atol=CSV_CLOCK_ATOL):
        raise RuntimeError("canonical finalizer requires a fresh run from step 0/time 0")
    if final_step != expected_step or not np.isclose(
        final_time, expected_time, rtol=0.0, atol=CSV_CLOCK_ATOL
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


def _segment_run_metadata(
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
    if not np.isclose(segment_dt, dt, rtol=0.0, atol=CSV_CLOCK_ATOL):
        raise RuntimeError(f"restart segment time-step identity mismatch: {metadata_path}")
    interval_value = segment["accepted_interval"]
    last_step = interval_value["last_step"]
    last_time = interval_value["last_time"]
    if last_step is None or last_time is None:
        raise RuntimeError(f"{segment['id']}: completed lineage segment has no endpoint")
    if final_step != last_step or not np.isclose(
        final_time, last_time, rtol=0.0, atol=CSV_CLOCK_ATOL
    ):
        raise RuntimeError(
            f"{segment['id']}: metadata endpoint does not match declared lineage interval"
        )
    if segment["origin"] == "fresh_initial_value":
        if initial_step != 0 or not np.isclose(
            segment_initial_time, initial_time, rtol=0.0, atol=CSV_CLOCK_ATOL
        ):
            raise RuntimeError(f"{segment['id']}: fresh metadata does not start at step 0/time 0")
    else:
        expected_initial_step = interval_value["first_step"] - 1
        expected_initial_time = interval_value["first_time"] - dt
        if initial_step != expected_initial_step or not np.isclose(
            segment_initial_time, expected_initial_time, rtol=0.0, atol=CSV_CLOCK_ATOL
        ):
            raise RuntimeError(
                f"{segment['id']}: restart metadata does not start at its predecessor"
            )
    if require_completed:
        if status != "completed":
            raise RuntimeError(f"{segment['id']}: continuation metadata is not completed")
        if requested_steps != final_step - initial_step or not np.isclose(
            final_time, segment_initial_time + requested_steps * dt, rtol=0.0, atol=CSV_CLOCK_ATOL
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
    backups = _named_steps(solution / "vpm", BACKUP_NAME, ".h5")
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
            actual_time, expected_time, rtol=0.0, atol=CSV_CLOCK_ATOL
        ):
            raise RuntimeError(f"{segment['id']}: native H5 clock mismatch at step {step}")
        clocks[step] = actual_time

    vtu_by_step = _named_steps(solution / "vpm", VTU_NAME, ".vtu")
    if set(vtu_by_step) != set(expected_steps):
        raise RuntimeError(f"{segment['id']}: native VTU owners do not match H5 backups")
    for step, path in vtu_by_step.items():
        if not np.isclose(_vtu_time(path), clocks[step], rtol=0.0, atol=CSV_CLOCK_ATOL):
            raise RuntimeError(f"{segment['id']}: native VTU time mismatch at step {step}")
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
    root_metadata, _, initial_time, _, _, root_interval = _segment_run_metadata(
        prefix, initial_time=0.0, dt=dt, require_completed=False
    )
    try:
        campaign_steps = int(root_metadata["configuration"]["run"]["steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("fresh prefix metadata is missing the campaign endpoint") from error
    campaign_end_time = initial_time + campaign_steps * dt
    if campaign_steps <= prefix["accepted_interval"]["last_step"] or campaign_steps % root_interval:
        raise RuntimeError("restart campaign endpoint must follow the prefix on the backup cadence")
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
    if endpoint_step != campaign_steps or not np.isclose(
        endpoint_time, campaign_end_time, rtol=0.0, atol=CSV_CLOCK_ATOL
    ):
        raise RuntimeError(
            f"continuation does not reach the declared step-{campaign_steps} / "
            f"t={campaign_end_time:g} s endpoint"
        )
    effective_segments = [prefix, effective_continuation]
    continuation_metadata, _, _, _, _, continuation_interval_steps = _segment_run_metadata(
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
            float(boundary_record["time"]), expected_clock[1], rtol=0.0, atol=CSV_CLOCK_ATOL
        ):
            raise RuntimeError(
                f"{segment['id']}: boundary clock is not its selected owner H5 clock"
            )

    prefix_end = boundary(prefix, "end")
    continuation_start = boundary(continuation, "start")
    prefix_owner = (
        prefix["solution_path"] / "vpm" / f"vpm_{prefix['accepted_interval']['last_step']:06d}.h5"
    )
    prefix_clock = (
        prefix["accepted_interval"]["last_step"],
        root_clocks[prefix["accepted_interval"]["last_step"]],
    )
    require_owner_boundary(prefix, prefix_end, prefix_owner, prefix_clock)
    require_owner_boundary(continuation, continuation_start, prefix_owner, prefix_clock)
    if continuation["status"] == "accepted":
        continuation_end = boundary(continuation, "end")
        continuation_owner = continuation["solution_path"] / "vpm" / f"vpm_{endpoint_step:06d}.h5"
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
    if len(ordered_times) < 2 or np.max(np.diff(ordered_times)) > MAX_NATIVE_GAP + CSV_CLOCK_ATOL:
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
    global CASE_DIR, SOLUTION_DIR, SAMPLES_DIR, LINEAGE_MANIFEST
    CASE_DIR = Path(case_dir).resolve()
    SOLUTION_DIR = CASE_DIR / "solution"
    SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
    LINEAGE_MANIFEST = CASE_DIR / "assets" / "delta_wing_accepted_lineage.json"
    manifest_payload = _read_metadata(LINEAGE_MANIFEST) if LINEAGE_MANIFEST.is_file() else None
    declared_segments = manifest_payload.get("segments") if manifest_payload is not None else None
    if isinstance(declared_segments, list) and len(declared_segments) == 2:
        _, effective_segments, end_step, end_time = _check_declared_resume_lineage(LINEAGE_MANIFEST)
        payload = deepcopy(manifest_payload)
        by_id = {segment["id"]: segment for segment in payload["segments"]}
        continuation = by_id[effective_segments[-1]["id"]]
        continuation["status"] = "accepted"
        continuation["accepted_interval"]["last_step"] = end_step
        continuation["accepted_interval"]["last_time"] = end_time
        end_path = effective_segments[-1]["solution_path"] / "vpm" / f"vpm_{end_step:06d}.h5"
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
        _atomic_write_json(LINEAGE_MANIFEST, payload)
        print(f"finalized {LINEAGE_MANIFEST} at step {end_step} / t={end_time:g} s")
        return LINEAGE_MANIFEST
    end_step, end_time, _ = _check_completed_native_run()
    end_path = SOLUTION_DIR / "vpm" / f"vpm_{end_step:06d}.h5"
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
                        "path": f"solution/vpm/vpm_{end_step:06d}.h5",
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
    _atomic_write_json(LINEAGE_MANIFEST, payload)
    print(f"finalized {LINEAGE_MANIFEST} at step {end_step} / t={end_time:g} s")
    return LINEAGE_MANIFEST


def oblique_projection(points: np.ndarray) -> np.ndarray:
    """Project native x/y/z points to fixed oblique screen coordinates."""
    return np.asarray(points) @ OBLIQUE_PROJECTION.T


def _backup_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid coupled backup name: {path.name}") from error


def _read_backup_clock(path: Path) -> tuple[int, float]:
    with h5py.File(path, "r") as archive:
        if "solver" not in archive or "vlm" not in archive["solver"]:
            raise ValueError(f"{path.name}: missing coupled solver/vlm state")
        solver = archive["solver"]
        try:
            step = _integer_step(solver.attrs["step"], f"{path}: solver step")
            time = _finite_time(solver.attrs["time"], f"{path}: solver time")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path.name}: missing coupled solver clock") from error
        vlm = solver["vlm"]
        required = {"panel_corner_position", "circulation", "panel_force"}
        missing = required - set(vlm)
        if missing:
            raise ValueError(f"{path.name}: missing coupled VLM arrays {sorted(missing)}")
    if step < 0:
        raise ValueError(f"{path.name}: non-finite or invalid coupled solver clock")
    return step, time


def _compare_serialized_values(label: str, left, right) -> None:
    """Reject a conflicting common HDF5 value before handoff deduplication."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        raise ValueError(
            f"overlapping coupled backups conflict in {label}: "
            f"shapes {left_array.shape} and {right_array.shape}"
        )
    numeric = left_array.dtype.kind in "biufc" and right_array.dtype.kind in "biufc"
    if numeric:
        if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
            raise ValueError(f"overlapping coupled backups contain non-finite {label}")
        if left_array.dtype.kind in "biu" or right_array.dtype.kind in "biu":
            # IDs, counts and flags are schema values, not f32 payloads.
            equal = np.array_equal(left_array, right_array)
        else:
            equal = np.allclose(
                left_array,
                right_array,
                rtol=CSV_SERIALIZATION_RTOL,
                atol=CSV_SERIALIZATION_ATOL,
                equal_nan=False,
            )
    else:
        equal = np.array_equal(left_array, right_array)
    if not equal:
        raise ValueError(
            f"overlapping coupled backups conflict in {label} "
            f"(serialization tolerance rtol={CSV_SERIALIZATION_RTOL:g}, "
            f"atol={CSV_SERIALIZATION_ATOL:g})"
        )


def _compare_common_hdf5_datasets(left, right, prefix: str) -> None:
    """Compare common native datasets while allowing additive new fields."""
    for name in sorted(set(left) & set(right)):
        left_item = left[name]
        right_item = right[name]
        label = f"{prefix}/{name}"
        if isinstance(left_item, h5py.Dataset) != isinstance(right_item, h5py.Dataset):
            raise ValueError(
                f"overlapping coupled backups conflict at {label}: object type differs"
            )
        if isinstance(left_item, h5py.Dataset):
            _compare_serialized_values(label, left_item[()], right_item[()])
        else:
            _compare_common_hdf5_datasets(left_item, right_item, label)


def _compare_overlapping_backups(first: Path, second: Path) -> None:
    """Check a repeated accepted state before choosing the dense source row."""
    with h5py.File(first, "r") as left_archive, h5py.File(second, "r") as right_archive:
        left_solver = left_archive["solver"]
        right_solver = right_archive["solver"]
        required_solver_attrs = {
            "backup_format_version",
            "n_particles_total",
            "numerical_configuration",
            "numerical_configuration_sha256",
            "freestream_velocity",
            "step",
            "time",
            "time_step_size",
        }
        missing = sorted(
            name
            for name in required_solver_attrs
            if name not in left_solver.attrs or name not in right_solver.attrs
        )
        if missing:
            raise ValueError(
                f"overlapping coupled backups missing required solver fields {missing}"
            )
        for name in sorted(set(left_solver.attrs) & set(right_solver.attrs)):
            if name in {"step", "time", "time_step_size"}:
                # The accepted clock is a double-precision contract, not a
                # relaxed comparison of physical f32 payloads.
                if name == "step":
                    equal = int(left_solver.attrs[name]) == int(right_solver.attrs[name])
                else:
                    equal = bool(
                        np.isclose(
                            float(left_solver.attrs[name]),
                            float(right_solver.attrs[name]),
                            rtol=0.0,
                            atol=CSV_CLOCK_ATOL,
                        )
                    )
                if not equal:
                    raise ValueError(
                        f"overlapping coupled backups conflict in solver attribute {name}"
                    )
            else:
                _compare_serialized_values(
                    f"solver attribute {name}", left_solver.attrs[name], right_solver.attrs[name]
                )

        left_vlm = left_solver["vlm"]
        right_vlm = right_solver["vlm"]
        for name in ("coupled_mode", "reference_speed", "reference_velocity", "solved", "time"):
            if name not in left_vlm.attrs or name not in right_vlm.attrs:
                raise ValueError(f"overlapping coupled backups missing VLM attribute {name}")
        left_version = int(left_vlm.attrs.get("version", -1))
        right_version = int(right_vlm.attrs.get("version", -1))
        if left_version != right_version and {left_version, right_version} != {6, 7}:
            raise ValueError(
                f"overlapping coupled backups conflict in VLM version: "
                f"{left_version} versus {right_version}"
            )
        for name in sorted(set(left_vlm.attrs) & set(right_vlm.attrs)):
            if name in {"identity", "version"}:
                continue
            if name == "time":
                if not np.isclose(
                    float(left_vlm.attrs[name]),
                    float(right_vlm.attrs[name]),
                    rtol=0.0,
                    atol=CSV_CLOCK_ATOL,
                ):
                    raise ValueError(
                        "overlapping coupled backups conflict in solver/vlm attribute time"
                    )
            else:
                _compare_serialized_values(
                    f"solver/vlm attribute {name}", left_vlm.attrs[name], right_vlm.attrs[name]
                )
        left_physics_identity = left_vlm.attrs.get("physics_identity")
        right_physics_identity = right_vlm.attrs.get("physics_identity")
        if left_physics_identity is not None and right_physics_identity is not None:
            _compare_serialized_values(
                "solver/vlm attribute physics_identity",
                left_physics_identity,
                right_physics_identity,
            )
        left_identity = left_vlm.attrs.get("identity")
        right_identity = right_vlm.attrs.get("identity")
        if left_identity is None or right_identity is None:
            if left_version == right_version and left_identity != right_identity:
                raise ValueError("overlapping coupled backups have incomplete VLM identity")
        elif left_version == right_version:
            _compare_serialized_values(
                "solver/vlm attribute identity", left_identity, right_identity
            )
        # A v6 -> v7 continuation may legitimately change the legacy full
        # identity as output-only fields migrate. The common physics identity,
        # numerical identity and clocks above still have to agree.

        _compare_common_hdf5_datasets(
            left_archive["particles"], right_archive["particles"], "particles"
        )
        _compare_common_hdf5_datasets(left_vlm, right_vlm, "solver/vlm")


def _solution_directories(value: Path | list[Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    directories = [Path(directory) for directory in value]
    if not directories:
        raise ValueError("at least one solution directory is required")
    return directories


def coupled_frames(
    solution_dir: Path | list[Path] | None = None,
    lineage_manifest: Path = LINEAGE_MANIFEST,
) -> list[tuple[float, Path, str]]:
    """Return ordered coupled backups from one or more named source segments."""
    lineage = load_animation_lineage(lineage_manifest) if solution_dir is None else None
    if lineage is None:
        sources = [(directory, None) for directory in _solution_directories(solution_dir)]
    else:
        sources = [(segment["solution_path"], segment) for segment in lineage]
    by_step: dict[int, tuple[float, Path, str]] = {}
    for directory, segment_info in sources:
        paths = sorted((directory / "vpm").glob("vpm_*.h5"), key=_backup_step)
        if segment_info is not None:
            interval = segment_info["accepted_interval"]
            paths = [
                path
                for path in paths
                if _backup_step(path) >= interval["first_step"]
                and (interval["last_step"] is None or _backup_step(path) <= interval["last_step"])
            ]
        if not paths:
            raise FileNotFoundError(f"no coupled VPM backups found under {directory}")
        segment = (
            segment_info["id"] if segment_info is not None else directory.name or str(directory)
        )
        for path in paths:
            step, time = _read_backup_clock(path)
            if segment_info is not None:
                interval = segment_info["accepted_interval"]
                _validate_selected_clock(step, time, interval, f"{segment} {path.name}")
            if step in by_step:
                previous_time, previous_path, _ = by_step[step]
                if not np.isclose(previous_time, time, rtol=0.0, atol=CSV_CLOCK_ATOL):
                    raise ValueError(f"handoff step {step} has inconsistent source timestamps")
                _compare_overlapping_backups(previous_path, path)
            # If a continuation repeats the handoff backup, its later segment
            # owns the merged record while both source directories remain in
            # the output manifest.
            by_step[step] = (time, path, segment)
    records = [by_step[step] for step in sorted(by_step)]
    steps = np.asarray([_backup_step(path) for _, path, _ in records])
    times = np.asarray([time for time, _, _ in records])
    if np.any(np.diff(steps) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("coupled VPM backup steps or timestamps are unordered")
    return records


def select_frames(records: list[tuple[float, Path, str]], fps: int = GIF_FPS):
    if fps != GIF_FPS:
        raise ValueError(f"GIF output is fixed at {GIF_FPS} fps")
    source_times = np.asarray([time for time, _, _ in records])
    count = max(1, int(round((source_times[-1] - source_times[0]) * fps)) + 1)
    targets = np.linspace(source_times[0], source_times[-1], count)
    indexes = np.searchsorted(source_times, targets).clip(0, len(source_times) - 1)
    left = np.maximum(indexes - 1, 0)
    choose_left = np.abs(source_times[left] - targets) < np.abs(source_times[indexes] - targets)
    indexes[choose_left] = left[choose_left]
    # Keep every presentation target. A sparse native frame is held across
    # several targets; dropping those targets would make the GIF far shorter
    # than its physical source interval while still claiming the requested fps.
    return targets, [records[int(index)] for index in indexes]


def _require_dense_native_source(records: list[tuple[float, Path, str]]) -> float:
    """Reject sparse native sources before a public dense animation is written."""
    if len(records) < 2:
        raise ValueError("dense animation requires at least two native backups")
    source_times = np.asarray([time for time, _, _ in records], dtype=float)
    maximum_gap = float(np.max(np.diff(source_times)))
    if maximum_gap > MAX_NATIVE_GAP + CSV_CLOCK_ATOL:
        raise ValueError(
            "dense animation requires native backup gaps at most "
            f"1/{GIF_FPS} s (observed {maximum_gap:g} s)"
        )
    return maximum_gap


def _require_unique_presentation_sources(
    selected: list[tuple[float, Path, str]],
) -> None:
    """Reject a presentation sequence that would hold one native state twice."""
    source_paths = [path.resolve() for _, path, _ in selected]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("dense animation would repeat a native state across presentation frames")


def _read_vlm_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as archive:
        vlm = archive["solver/vlm"]
        corners = np.asarray(vlm["panel_corner_position"], dtype=float)
        circulation = np.asarray(vlm["circulation"], dtype=float)
        panel_force = np.asarray(vlm["panel_force"], dtype=float)
    if corners.ndim != 3 or corners.shape[1:] != (4, 3):
        raise ValueError(f"{path.name}: invalid coupled VLM panel corners")
    if circulation.shape != (len(corners),) or panel_force.shape != (len(corners), 3):
        raise ValueError(f"{path.name}: inconsistent coupled VLM panel arrays")
    if (
        not np.isfinite(corners).all()
        or not np.isfinite(circulation).all()
        or not np.isfinite(panel_force).all()
    ):
        raise ValueError(f"{path.name}: non-finite coupled VLM geometry or loads")
    return corners, circulation


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(CASE_DIR))
    except ValueError:
        return str(path)


def _source_segments(records: list[tuple[float, Path, str]]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for time, path, segment in records:
        if not segments or segments[-1]["namespace"] != segment:
            segments.append(
                {
                    "namespace": segment,
                    "directory": _display_path(path.parent),
                    "first_step": _backup_step(path),
                    "last_step": _backup_step(path),
                    "first_time": float(time),
                    "last_time": float(time),
                    "frame_count": 1,
                }
            )
        else:
            segments[-1]["last_step"] = _backup_step(path)
            segments[-1]["last_time"] = float(time)
            segments[-1]["frame_count"] = int(segments[-1]["frame_count"]) + 1
    return segments


def _segment_transitions(records: list[tuple[float, Path, str]]) -> list[dict[str, object]]:
    transitions = []
    for previous, current in zip(records[:-1], records[1:], strict=True):
        if previous[2] == current[2]:
            continue
        transitions.append(
            {
                "from_namespace": previous[2],
                "to_namespace": current[2],
                "from_step": _backup_step(previous[1]),
                "to_step": _backup_step(current[1]),
                "accepted_step_gap": _backup_step(current[1]) - _backup_step(previous[1]),
                "time_gap": float(current[0] - previous[0]),
            }
        )
    return transitions


def render(
    output: Path,
    solution_dir: Path | list[Path] | None = None,
    fps: int = GIF_FPS,
    lineage_manifest: Path = LINEAGE_MANIFEST,
) -> None:
    if fps != GIF_FPS:
        raise ValueError(f"GIF output is fixed at {GIF_FPS} fps")
    lineage = load_animation_lineage(lineage_manifest) if solution_dir is None else None
    records = coupled_frames(solution_dir, lineage_manifest)
    maximum_native_gap = _require_dense_native_source(records)
    target_times, selected = select_frames(records, fps)
    _require_unique_presentation_sources(selected)
    native_frames = [
        (time, path, segment, *_read_vlm_frame(path)) for time, path, segment in selected
    ]

    vertices = np.concatenate(
        [oblique_projection(corners) for _, _, _, corners, _ in native_frames]
    ).reshape(-1, 2)
    circulation = np.concatenate([values for _, _, _, _, values in native_frames])
    horizontal_limits = (vertices[:, 0].min() - 0.15, vertices[:, 0].max() + 0.15)
    vertical_limits = (vertices[:, 1].min() - 0.15, vertices[:, 1].max() + 0.15)
    scale = max(float(np.max(np.abs(circulation))), 1e-12)
    norm = Normalize(-scale, scale)
    _theme.set_thesis_style()
    cmap = LinearSegmentedColormap.from_list(
        "thesis_signed", [_theme.COLORS["TUDcyan"], "white", _theme.COLORS["VPMpurple"]]
    )
    frames = []
    for target_time, (source_time, _, source_segment), (_, path, segment, corners, values) in zip(
        target_times, selected, native_frames, strict=True
    ):
        fig, ax = plt.subplots(figsize=(12.5 * _theme.CM, 7.5 * _theme.CM), dpi=160)
        polygons = oblique_projection(corners)
        ax.add_collection(
            PolyCollection(
                polygons,
                array=values,
                cmap=cmap,
                norm=norm,
                edgecolors=_theme.COLORS["DarkText"],
                linewidths=0.15,
            )
        )
        ax.set(
            xlim=horizontal_limits,
            ylim=vertical_limits,
            xlabel="$s$ [m]",
            ylabel="$h$ [m]",
            title=f"$t = {source_time:.3f}$ s",
        )
        ax.grid(True, color=_theme.COLORS["LightBG"], linewidth=0.5)
        ax.locator_params(axis="y", nbins=3)
        _theme.centered_subplots_adjust(fig, outer=0.15, bottom=0.30, top=0.87)
        _theme.fit_thesis_y_label_margins(fig, (ax,))
        outer = ax.get_position().x0
        height_cm = (1 - 2 * outer) * 12.5 * np.ptp(vertical_limits) / np.ptp(
            horizontal_limits
        ) + 3.8
        fig.set_size_inches(12.5 * _theme.CM, height_cm * _theme.CM, forward=False)
        fig.subplots_adjust(bottom=2.8 / height_cm, top=1 - 1.0 / height_cm)
        ax.set_aspect("equal")
        cax = fig.add_axes([outer, 1.3 / height_cm, 1 - 2 * outer, 0.22 / height_cm])
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="horizontal",
            ticks=[-scale, 0, scale],
            format="%.2f",
            label=r"$\Gamma$ [m$^2$/s]",
        )
        _theme.validate_thesis_figure(fig, (ax, cax))
        fig.canvas.draw()
        pixels = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(pixels))
        plt.close(fig)

    output.parent.mkdir(parents=True, exist_ok=True)
    # GIF stores durations in centiseconds. The repeating 30/30/40 ms
    # pattern has an exact 100 ms / 3-frame average, i.e. 30 fps.
    durations = [30 if index % 3 < 2 else 40 for index in range(len(frames))]
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2
    )
    manifest = output.with_suffix(".json")
    sparse_hold = len({path for _, path, _ in selected}) < len(selected)
    manifest.write_text(
        json.dumps(
            {
                "gif": _display_path(output),
                "fps": fps,
                "frame_duration_pattern_ms": [30, 30, 40],
                "source_format": "coupled_vpm_hdf5",
                "source_selection": (
                    "finalized_dense_lineage" if lineage is not None else "explicit_directories"
                ),
                "source_lineage_manifest": (
                    _display_path(Path(lineage_manifest).resolve()) if lineage is not None else None
                ),
                "source_lineage_segments": (
                    [segment["id"] for segment in lineage] if lineage is not None else None
                ),
                "overlap_validation": {
                    "clock_atol": CSV_CLOCK_ATOL,
                    "serialization_rtol": CSV_SERIALIZATION_RTOL,
                    "serialization_atol": CSV_SERIALIZATION_ATOL,
                    "additive_fields_allowed": True,
                },
                "sampling_mode": "nearest_native_hold" if sparse_hold else "nearest_native_backup",
                "sparse_hold": sparse_hold,
                "maximum_native_gap_seconds": maximum_native_gap,
                "presentation_repeated_native_state": sparse_hold,
                "native_source_count": len(records),
                "native_source_backups": [_display_path(path) for _, path, _ in records],
                "native_source_timestamps": [float(time) for time, _, _ in records],
                "source_segments": _source_segments(records),
                "segment_transitions": _segment_transitions(records),
                "physical_start_time": float(records[0][0]),
                "physical_end_time": float(records[-1][0]),
                "frame_count": len(frames),
                "source_backups": [_display_path(path) for _, path, _ in selected],
                "source_backup_segments": [segment for _, _, segment in selected],
                "accepted_source_timestamps": [
                    float(source_time) for source_time, _, _ in selected
                ],
                "target_timestamps": [float(target_time) for target_time in target_times],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output} ({len(frames)} frames at {fps} fps)")


def _render_main(argv) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solution",
        type=Path,
        action="append",
        help="dense solution directory; repeat in sparse-to-dense order for a continuation",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_GIF_OUTPUT)
    parser.add_argument("--fps", type=int, default=GIF_FPS)
    args = parser.parse_args(argv)
    if args.fps != GIF_FPS:
        parser.error(f"--fps must be exactly {GIF_FPS}")
    if not args.solution and not resolve_plot_sources().complete:
        print("GIF skipped: completed-run animation requires finalized native output.")
        return
    render(args.output, args.solution, args.fps)


def _cli_directories(values, default: Path) -> list[Path]:
    paths = [Path(value) for value in values] if values else [default]
    return [path if path.is_absolute() else CASE_DIR / path for path in paths]


def _configured_flow_interval(metadata: dict) -> int | None:
    items = metadata["configuration"].get("samplers", {}).get("items", [])
    for item in items:
        if item.get("type") == "FlowIntegralsSampler":
            return item.get("schedule", {}).get("interval")
    return None


def _segment_metadata(solution_dir: Path) -> dict:
    path = solution_dir / "vpm_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _observed_cadence(steps: np.ndarray) -> int | None:
    unique = np.unique(steps.astype(int))
    if len(unique) < 2:
        return None
    differences = np.diff(unique)
    return int(np.gcd.reduce(differences))


def _check_segment(solution_dir: Path, samples_dir: Path, *, require_complete: bool) -> list[str]:
    """Check one segment while treating source lifecycle markers as archival."""
    failures = []
    metadata = _segment_metadata(solution_dir)
    config, state = metadata["configuration"], metadata["state"]
    expected_step = state["initial_step"] + config["run"]["steps"]
    label = solution_dir.name or str(solution_dir)
    status = metadata.get("lifecycle", {}).get("status")
    if status != "completed":
        print(
            f"[INFO] {label}: lifecycle marker {status!r} is archival metadata; "
            "this validator does not infer live-process state from it"
        )
    if require_complete:
        if status != "completed" or state["step"] != expected_step:
            failures.append(
                f"{label}: final segment incomplete ({status=}; step={state['step']}/{expected_step})"
            )

    force_path = samples_dir / "vlm_surface_forces.csv"
    force = pd.read_csv(force_path)
    duplicate_times = any(rows.time.duplicated().any() for _, rows in force.groupby("surface"))
    if duplicate_times or not np.isfinite(force.select_dtypes("number")).all().all():
        failures.append(f"{label}: duplicate or non-finite force samples")
    logging_interval = config["numerics"]["vlm"]["logging_interval_steps"]
    observed_force_cadence = _observed_cadence(force.step.to_numpy())
    print(
        f"[INFO] {label}: force cadence observed every {observed_force_cadence} steps; "
        f"configured {logging_interval}"
    )
    if require_complete and force.step.max() < expected_step - logging_interval:
        failures.append(f"{label}: incomplete force history")
    if require_complete and logging_interval == 1 and observed_force_cadence != 1:
        failures.append(f"{label}: dense continuation force cadence is not every accepted step")

    flow_path = samples_dir / "flow_integrals.csv"
    flow = pd.read_csv(flow_path)
    if not np.isfinite(flow.select_dtypes("number")).all().all():
        failures.append(f"{label}: non-finite flow integrals")
    flow_interval = _configured_flow_interval(metadata)
    observed_flow_cadence = _observed_cadence(flow.step.to_numpy())
    print(
        f"[INFO] {label}: flow cadence observed every {observed_flow_cadence} steps; "
        f"configured {flow_interval}"
    )
    if require_complete and flow_interval and observed_flow_cadence != flow_interval:
        failures.append(f"{label}: flow-integral cadence differs from its configured schedule")

    wake_planes = list(samples_dir.glob("wake_*span.pvd"))
    if require_complete and len(wake_planes) != 3:
        failures.append(f"{label}: expected three published wake planes")
    return failures


def _validate_main(argv) -> int:
    parser = argparse.ArgumentParser(
        description="Check native delta-wing samples, including a sparse-to-dense continuation."
    )
    parser.add_argument("--pre-plot", action="store_true")
    parser.add_argument(
        "--solution",
        type=Path,
        action="append",
        help="solution directory; repeat in sparse-to-dense order",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        action="append",
        help="sample directory matching each --solution directory",
    )
    args = parser.parse_args(argv)
    solution_dirs = _cli_directories(args.solution, CASE_DIR / "solution")
    samples_dirs = _cli_directories(args.samples, SAMPLES_DIR)
    if len(solution_dirs) != len(samples_dirs):
        parser.error("repeat --solution and --samples the same number of times")

    failures = []
    for index, (solution_dir, samples_dir) in enumerate(
        zip(solution_dirs, samples_dirs, strict=True)
    ):
        failures.extend(
            _check_segment(
                solution_dir,
                samples_dir,
                require_complete=index == len(solution_dirs) - 1,
            )
        )

    final_metadata = _segment_metadata(solution_dirs[-1])
    final_config, final_state = final_metadata["configuration"], final_metadata["state"]
    data = force_history(samples_dirs)
    period = motion_period(data)
    for surface, rows in data.groupby("surface"):
        if (
            rows.time.duplicated().any()
            or not np.isfinite(rows.select_dtypes("number")).all().all()
        ):
            failures.append(f"{surface}: duplicate or non-finite merged samples")
        cycles = list(last_cycles(rows, period))
        if len(cycles) < 3:
            failures.append(f"{surface}: fewer than three complete sampled cycles")
            continue
        phase = np.linspace(0.01, 1, 100)
        profiles = np.array([np.interp(phase, x, cycle.force_z) for _, x, cycle in cycles])
        scale = max(np.sqrt(np.mean(profiles**2)), 1e-12)
        drift = np.max(np.sqrt(np.mean(np.diff(profiles, axis=0) ** 2, axis=1))) / scale
        print(f"{surface}: period={period:.6g}s, phase-resolved force RMS drift={drift:.2%}")
        if not np.isfinite(drift) or drift > 0.05:
            failures.append(f"{surface}: final-cycle force drift exceeds5%; run longer")

    integrals = flow_integrals(samples_dirs)
    if not np.isfinite(integrals.select_dtypes("number")).all().all():
        failures.append("non-finite merged flow integrals")
    dt = final_config["numerics"]["time_step_size"]
    end = final_state["initial_time"] + final_config["run"]["steps"] * dt
    if final_state["step"] == final_state["initial_step"] + final_config["run"]["steps"]:
        if integrals.time.max() < end - 0.05:
            failures.append("incomplete merged sampled flow history")
    if not args.pre_plot:
        for name in (
            "delta_wing_forces",
            "delta_wing_force_cycles",
            "delta_wing_circulation_history",
            "delta_wing_wake_streamwise",
            "delta_wing_wake_vertical",
        ):
            for extension in ("png", "pdf"):
                if not (FIGURES_DIR / f"{name}.{extension}").is_file():
                    failures.append(f"missing {name}.{extension}")
    print("\n".join(f"[FAIL] {item}" for item in failures) or "[OK] Delta-wing checks passed")
    return bool(failures)


def main(argv=None) -> int:
    """Dispatch the tutorial data commands: ``finalize``, ``validate``, ``render-gif``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["finalize"]:
        if len(argv) > 1:
            raise SystemExit("finalize takes no arguments")
        finalize_lineage()
        return 0
    if argv[:1] == ["render-gif"]:
        _render_main(argv[1:])
        return 0
    if argv[:1] == ["validate"]:
        return int(bool(_validate_main(argv[1:])))
    parser = argparse.ArgumentParser(
        prog="assets/postprocess.py",
        description="Finalize, validate or render delta-wing native data.",
    )
    parser.add_argument("command", choices=("finalize", "validate", "render-gif"))
    parser.parse_args(argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
