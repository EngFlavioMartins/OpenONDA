"""Physical histories and common saved states for cylinder figures."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from openonda.plotting import export_figure, fit_thesis_y_label_margins, prepare_figure
from openonda.results import (
    read_csv_frame,
    read_csv_table,
    read_history_table,
    write_json as write_record,
)
from openonda.saved_times import match_saved_times

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
AUXILIARY = FIGURES / "auxiliary"
REFERENCE_ROOT = CASE_DIR / "reference_flow"
VELOCITY_COLUMNS = tuple(f"velocity_{axis}" for axis in "xyz")


def reference_directory(root: Path = REFERENCE_ROOT) -> Path:
    return root / "samples"


def history(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(read_history_table(path))[["time", *columns]]


def common_history(candidate, reference, columns):
    """Compare forces by interpolation within their common physical interval."""
    start = max(float(candidate.time.iloc[0]), float(reference.time.iloc[0]))
    end = min(float(candidate.time.iloc[-1]), float(reference.time.iloc[-1]))
    times = np.unique(np.r_[start, end, candidate.time, reference.time])
    times = times[(times >= start) & (times <= end)]
    left = np.column_stack([np.interp(times, candidate.time, candidate[name]) for name in columns])
    right = np.column_stack([np.interp(times, reference.time, reference[name]) for name in columns])
    errors = {}
    for index, name in enumerate(columns):
        difference = left[:, index] - right[:, index]
        errors[name] = {
            "rms": float(np.sqrt(trapezoid(difference**2, times) / (end - start))),
            "maximum": float(np.abs(difference).max()),
        }
    return times, left, right, errors


def history_coverage(candidate, reference):
    intervals = {
        name: [float(frame.time.iloc[0]), float(frame.time.iloc[-1])]
        for name, frame in (("coupled", candidate), ("reference", reference))
    }
    return {
        "available_time_intervals": intervals,
        "comparison_time_interval": [
            max(interval[0] for interval in intervals.values()),
            min(interval[1] for interval in intervals.values()),
        ],
        "time_alignment": "piecewise-linear interpolation on the common interval; no time shift",
    }


def coincident_profiles(paths):
    """Select each transverse profile at the same native physical clock."""
    times = {path: np.unique(read_csv_table(path)["time"]) for path in dict.fromkeys(paths)}
    common = match_saved_times(*times.values())
    for index, time in enumerate(common.times):
        samples = {
            path: pd.DataFrame(
                read_csv_frame(path, native_times[indices[index]], coordinates=("position_y",))
            )
            for (path, native_times), indices in zip(times.items(), common.indices, strict=True)
        }
        yield time, samples


def write_json(name: str, record: dict) -> None:
    write_record(AUXILIARY / name, record)


def save_figure(fig, axes, name: str, figure_format: str) -> None:
    prepare_figure(fig)
    fit_thesis_y_label_margins(fig, tuple(axes))
    export_figure(fig, FIGURES / name, figure_format=figure_format)
