"""Shared data loading and analysis for the cylinder figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import pandas as pd

from openonda.plotting import DEFAULT_DPI, validate_thesis_figure

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
AUXILIARY = FIGURES / "auxiliary"
REFERENCE_ROOT = CASE_DIR / "reference_flow"
VELOCITY_COLUMNS = tuple(f"velocity_{axis}" for axis in "xyz")


def reference_directory(root: Path = REFERENCE_ROOT) -> Path:
    """Resolve the reference from the post-processed selection metadata."""
    selection_path = root / "reference_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"reference selection is missing: run the complete reference campaign ({selection_path})"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("force_grid_qualified") is not True:
        raise ValueError("selected reference has not passed its force-grid qualification")
    relative = Path(str(selection["samples_relative"]))
    candidate = (root / relative).resolve()
    if candidate.parent != (root / "samples").resolve() or not candidate.is_dir():
        raise ValueError(
            f"reference selection escapes the reference samples directory: {candidate}"
        )
    metadata = candidate / "grid_run.json"
    if (
        not metadata.is_file()
        or json.loads(metadata.read_text(encoding="utf-8")).get("case") != candidate.name
    ):
        raise ValueError(f"reference selection does not identify a valid grid run: {candidate}")
    return candidate


def history(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Load a finite, strictly increasing sampled history."""
    frame = pd.read_csv(path)
    missing = {"time", *columns} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    values = frame[["time", *columns]].to_numpy(dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} must contain at least two finite samples")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"{path} times must be strictly increasing")
    return frame


def common_history(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Interpolate histories only over their common physical time interval."""
    start = max(float(candidate.time.iloc[0]), float(reference.time.iloc[0]))
    end = min(float(candidate.time.iloc[-1]), float(reference.time.iloc[-1]))
    if end <= start:
        raise ValueError("Force histories have no common physical time interval")
    times = np.unique(np.r_[start, end, candidate.time, reference.time])
    times = times[(times >= start) & (times <= end)]
    left = np.column_stack(
        [np.interp(times, candidate.time, candidate[column]) for column in columns]
    )
    right = np.column_stack(
        [np.interp(times, reference.time, reference[column]) for column in columns]
    )
    errors = {}
    for index, column in enumerate(columns):
        difference = left[:, index] - right[:, index]
        errors[column] = {
            "rms": float(np.sqrt(trapezoid(difference**2, times) / (end - start))),
            "maximum": float(np.abs(difference).max()),
        }
    return times, left, right, errors


def profile(path: Path, time: float) -> pd.DataFrame:
    """Load one native line-sampler state at an exact saved time."""
    columns = ("position_x", "position_y", "position_z", *VELOCITY_COLUMNS)
    frame = profile_history(path, columns)
    selected = frame[np.isclose(frame.time, time, rtol=0.0, atol=1.0e-7)].copy()
    if selected.empty:
        raise ValueError(f"{path} has no profile at t={time:g} s")
    return selected.sort_values("position_y")


def profile_history(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Load finite line profiles whose time repeats once per spatial point."""
    frame = pd.read_csv(path)
    missing = {"time", *columns} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    values = frame[["time", *columns]].to_numpy(dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} must contain finite profile samples")
    times = np.unique(values[:, 0])
    if len(times) == 0 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} profile times must be ordered")
    return frame


def latest_common_profile_time(paths: tuple[Path, ...]) -> float:
    """Return the latest physical time stored by every requested profile."""
    columns = ("position_x", "position_y", "position_z", *VELOCITY_COLUMNS)
    common: set[float] | None = None
    for path in paths:
        frame = profile_history(path, columns)
        available = set(np.round(frame.time.to_numpy(dtype=float), 7))
        common = available if common is None else common & available
    if not common:
        raise ValueError("Velocity profiles have no common physical sample time")
    return float(max(common))


def write_json(name: str, payload: dict) -> None:
    """Write non-figure results below the figure auxiliary directory."""
    AUXILIARY.mkdir(parents=True, exist_ok=True)
    (AUXILIARY / name).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_figure(fig, axes, name: str, figure_format: str) -> None:
    """Validate and save one fixed-canvas thesis figure."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    validate_thesis_figure(fig, axes)
    fig.savefig(FIGURES / f"{name}.{figure_format}", dpi=DEFAULT_DPI, bbox_inches=None)
    plt.close(fig)
