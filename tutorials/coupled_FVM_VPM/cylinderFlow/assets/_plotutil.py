"""Load matched reference/hybrid cylinder samples for plotting and metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
SOLUTION = CASE_DIR / "solution"
SAMPLES = CASE_DIR / "samples"
REFERENCE_SAMPLES = CASE_DIR / "referenceFlow" / "samples"
FIGURES = CASE_DIR / "figures"
TIME_TOL = 1.0e-3
SHEDDING_START = 8.0

_THEME_PATH = Path(__file__).resolve().parents[4] / "docs" / "themes" / "matplotlib_setup.py"
_SPEC = importlib.util.spec_from_file_location("openonda_matplotlib_setup", _THEME_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load plotting theme from {_THEME_PATH}")
_THEME = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_THEME)
_THEME.set_style()
COLORS = dict(_THEME.COLORS)
COLORMAPS = dict(_THEME.COLORMAPS)

SOURCES = {
    "reference": {"dir": REFERENCE_SAMPLES, "prefix": "", "label": "Reference FVM"},
    "fvm": {"dir": SAMPLES, "prefix": "fvm_", "label": "Hybrid FVM"},
    "vpm": {"dir": SAMPLES, "prefix": "vpm_", "label": "Hybrid VPM"},
}


def save(fig, name: str, fmt: str, dpi: int = 240) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.{fmt}"
    _THEME.save_fig(fig, path, figure_format=fmt, dpi=dpi)
    print(f"  wrote {path.relative_to(CASE_DIR)}")
    return path


def _path(source: str, name: str, suffix: str) -> Path:
    entry = SOURCES[source]
    return entry["dir"] / f"{entry['prefix']}{name}{suffix}"


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
    tagged = re.fullmatch(r"#\s*flow_time\s*=\s*([^\s]+)\s*", first)
    rows = np.atleast_1d(
        np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            skip_header=1 if tagged else 0,
        )
    )
    table = {name: np.asarray(rows[name]) for name in rows.dtype.names}
    if "flow_time" not in table:
        if tagged is None:
            raise ValueError(f"{path} has no flow_time data")
        table["flow_time"] = np.full(rows.size, float(tagged.group(1)))
    return table


def line_times(source: str, name: str) -> np.ndarray:
    path = _path(source, name, ".csv")
    return np.unique(_read_csv(path)["flow_time"]) if path.exists() else np.empty(0)


def load_line(source: str, name: str, time: float) -> dict[str, np.ndarray] | None:
    path = _path(source, name, ".csv")
    if not path.exists():
        return None
    table = _read_csv(path)
    times = np.unique(table["flow_time"])
    if not len(times):
        return None
    picked = float(times[np.argmin(np.abs(times - time))])
    if abs(picked - time) > TIME_TOL:
        return None
    mask = np.isclose(table["flow_time"], picked, atol=TIME_TOL)
    frame = {key: values[mask] for key, values in table.items()}
    frame["time"] = np.asarray(picked)
    return frame


def slice_frames(source: str) -> list[tuple[float, Path]]:
    pvd = _path(source, "slice_z0", ".pvd")
    if not pvd.exists():
        return []
    matches = re.finditer(r'timestep="([^"]+)"\s+[^>]*file="([^"]+)"', pvd.read_text())
    return sorted((float(match.group(1)), pvd.parent / match.group(2)) for match in matches)


def slice_times(source: str) -> np.ndarray:
    return np.asarray([time for time, _ in slice_frames(source)])


def load_slice(source: str, time: float) -> dict[str, np.ndarray] | None:
    import pyvista as pv

    frames = slice_frames(source)
    if not frames:
        return None
    picked, path = min(frames, key=lambda item: abs(item[0] - time))
    if abs(picked - time) > TIME_TOL:
        return None
    grid = pv.read(path)
    ni, nj, _ = grid.dimensions
    shape = (nj, ni)
    legacy = source != "vpm" and "OpenONDASurfaceOrdering" not in grid.field_data

    def field(name: str, component: int) -> np.ndarray:
        values = np.asarray(grid.point_data[name], dtype=float)[:, component]
        return values.reshape((ni, nj)).T if legacy else values.reshape(shape)

    points = np.asarray(grid.points, dtype=float)
    return {
        "x": points[:, 0].reshape(shape),
        "y": points[:, 1].reshape(shape),
        "Ux": field("Velocity", 0),
        "Uy": field("Velocity", 1),
        "omega_z": field("Vorticity", 2),
    }


def load_forces(source: str) -> dict[str, np.ndarray] | None:
    directory = REFERENCE_SAMPLES if source == "reference" else SAMPLES
    path = directory / "ibm_forces_history.csv"
    if not path.exists():
        return None
    rows = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    if not rows.size:
        return None
    resets = (
        np.flatnonzero(np.diff(rows["step"].astype(int)) <= 0) if "step" in rows.dtype.names else []
    )
    if len(resets):
        rows = rows[resets[-1] + 1 :]
    return {name: np.asarray(rows[name]) for name in rows.dtype.names}


def common_times(*series: np.ndarray) -> np.ndarray:
    if not series or any(not np.asarray(values).size for values in series):
        return np.empty(0)
    base = np.unique(np.asarray(series[0], dtype=float))
    return np.asarray(
        [
            time
            for time in base
            if time > TIME_TOL
            and all(np.min(np.abs(np.asarray(values) - time)) <= TIME_TOL for values in series[1:])
        ]
    )


def plot_times(available: np.ndarray, interval: float = 4.0) -> np.ndarray:
    if not len(available):
        return available
    targets = np.arange(interval, available[-1] + 0.5 * interval, interval)
    selected = [available[np.argmin(np.abs(available - target))] for target in targets]
    selected.append(available[-1])
    return np.unique(np.asarray(selected))


def settled_force_metrics(data: dict[str, np.ndarray], start: float | None = None) -> dict:
    time = np.asarray(data["time"], dtype=float)
    if start is None:
        start = (
            SHEDDING_START
            if SHEDDING_START < time[-1]
            else 0.5 * (float(time[0]) + float(time[-1]))
        )
    start = float(start)
    mask = time >= start
    t, cd, cl = time[mask], np.asarray(data["Cd"])[mask], np.asarray(data["Cl"])[mask]
    result = {
        "start_time": start,
        "mean_cd": float(np.mean(cd)),
        "rms_cl": float(np.sqrt(np.mean((cl - np.mean(cl)) ** 2))),
        "mean_cl": float(np.mean(cl)),
        "strouhal": float("nan"),
    }
    if len(t) >= 16:
        uniform = np.linspace(t[0], t[-1], len(t))
        signal = np.interp(uniform, t, cl)
        signal -= np.mean(signal)
        spectrum = np.abs(np.fft.rfft(signal))
        frequencies = np.fft.rfftfreq(len(signal), d=float(uniform[1] - uniform[0]))
        if len(spectrum) > 1:
            peak = 1 + int(np.argmax(spectrum[1:]))
            result["strouhal"] = float(frequencies[peak])
    return result


def write_metrics(payload: dict) -> Path:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    path = SAMPLES / "comparison_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
