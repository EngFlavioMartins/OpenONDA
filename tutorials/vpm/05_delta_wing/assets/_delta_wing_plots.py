"""Delta-wing figures from native force, motion and velocity samples."""

import json
import hashlib
import re
from pathlib import Path

from defusedxml import ElementTree
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.signal import find_peaks

from openonda import plotting as _theme

CASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = CASE_DIR / "samples" / "delta_wing"
FIGURES_DIR = CASE_DIR / "figures"
LINEAGE_MANIFEST = CASE_DIR / "assets" / "delta_wing_accepted_lineage.json"
_COLORS = _theme.COLORS
CSV_SERIALIZATION_RTOL = 5.0e-6
CSV_SERIALIZATION_ATOL = 1.0e-7
CSV_CLOCK_ATOL = 1.0e-12
_SOURCE_COLUMNS = {
    "source_segment",
    "source_samples_directory",
    "source_lineage_manifest",
    "source_lineage_status",
}


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


def _lineage_source_label(segments: list[dict]) -> str:
    return " → ".join(segment["id"] for segment in segments)


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


def _source_label(directories) -> str:
    return " → ".join(_segment_name(directory) for directory in _directories(directories))


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


def plot_forces(samples_dir=None, figures_dir=FIGURES_DIR, figure_format="png", *, partial=False):
    _theme.set_thesis_style()
    data = force_history(samples_dir)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12.5 * _theme.CM, 10.5 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.16, bottom=0.13, top=0.89, hspace=0.20)
    for surface, color, label in (
        ("front_wing", _COLORS["TUDcyan"], "Front"),
        ("rear_wing", _COLORS["VPMpurple"], "Rear"),
    ):
        rows = data[data.surface == surface]
        for ax, values in zip(axes, (rows.force_z, rows.centroid_z, -rows.power), strict=True):
            ax.plot(rows.time, values, color=color, label=label)
    for ax, label in zip(axes, (r"$F_z$ [N]", r"$z_c$ [m]", r"$P_{\mathrm{in}}$ [W]"), strict=True):
        ax.set_ylabel(label)
        ax.axhline(0, color=_COLORS["RefGray"], lw=0.4)
        ax.locator_params(axis="y", nbins=3)
    axes[-1].set_xlabel("Time [s]")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0 if not partial else 0.96),
    )
    if partial:
        fig.suptitle("Partial run", y=0.98)
        # Leave a separate line for the shared legend.
        fig.subplots_adjust(top=0.83)
    _save_figure(fig, axes, figures_dir / "delta_wing_forces.png", figure_format)
    period = _available_period(data)
    if period is None:
        print("Cycle comparison skipped: fewer than two sampled heave-velocity peaks.")
        return
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12.5 * _theme.CM, 8.3 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.16, bottom=0.16, top=0.80, hspace=0.27)
    for ax, surface, color, label in zip(
        axes,
        ("front_wing", "rear_wing"),
        (_COLORS["TUDcyan"], _COLORS["VPMpurple"]),
        ("Front", "Rear"),
        strict=True,
    ):
        for index, (cycle, phase, tail) in enumerate(
            last_cycles(data[data.surface == surface], period)
        ):
            ax.plot(
                phase,
                tail.force_z,
                color=color,
                ls=(":", "--", "-")[index],
                label=f"Cycle {cycle + 1}",
            )
        ax.set_ylabel(r"$F_z$ [N]")
        ax.set_title(label, loc="left", pad=2)
        ax.locator_params(axis="y", nbins=3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
    axes[-1].set_xlabel("Cycle phase")
    if partial:
        axes[0].set_title("Front (partial run)", loc="left", pad=2)
    _save_figure(fig, axes, figures_dir / "delta_wing_force_cycles.png", figure_format)


def _solution_directories_for_samples(samples_dirs, solution_dirs=None) -> list[Path]:
    if solution_dirs is not None:
        solutions = _directories(solution_dirs)
        if len(solutions) != len(_directories(samples_dirs)):
            raise ValueError("one solution directory is required for each sample directory")
        return solutions
    return [CASE_DIR / "solution"]


def plot_circulation(
    samples_dir=None, figures_dir=FIGURES_DIR, figure_format="png", *, partial=False
):
    _theme.set_thesis_style()
    data = flow_integrals(samples_dir)
    fig, ax = plt.subplots(figsize=(12.5 * _theme.CM, 7.0 * _theme.CM))
    _theme.centered_subplots_adjust(fig, outer=0.16, bottom=0.20, top=0.88)
    ax.plot(data.time, data.vortex_strength_magnitude_sum, color=_COLORS["VPMpurple"])
    ax.set(
        xlabel="Time [s]",
        ylabel=r"$\sum_p |\boldsymbol{\Gamma}_p|$ [m$^3$/s]",
        title="Partial run" if partial else "",
    )
    _save_figure(fig, (ax,), figures_dir / "delta_wing_circulation_history.png", figure_format)


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
        mean = np.trapezoid(fields, selected_times, axis=0) / period
        records.append((points, mean))
    return records


def plot_wake(
    samples_dir=None,
    figures_dir=FIGURES_DIR,
    figure_format="png",
    solution_dirs=None,
    *,
    partial=False,
):
    from matplotlib.colors import LinearSegmentedColormap

    _theme.set_thesis_style()
    lineage_segments = None
    if samples_dir is None:
        lineage_segments = load_accepted_lineage()
        samples_dirs = [segment["samples_path"] for segment in lineage_segments]
        solution_dirs = (
            [segment["solution_path"] for segment in lineage_segments]
            if solution_dirs is None
            else solution_dirs
        )
    else:
        samples_dirs = _directories(samples_dir)
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
    sequential = LinearSegmentedColormap.from_list("thesis_teal", ["white", _COLORS["TUDcyan"]])
    diverging = LinearSegmentedColormap.from_list(
        "thesis_signed", [_COLORS["TUDcyan"], "white", _COLORS["VPMpurple"]]
    )
    for values, cmap, field_name, label in (
        (axial, sequential, "streamwise", r"$u_{\parallel}/U_\infty$"),
        (vertical, diverging, "vertical", r"$u_z/U_\infty$"),
    ):
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
            figures_dir / f"delta_wing_wake_{field_name}.png",
            figure_format,
            fit=False,
        )
