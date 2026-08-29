"""Shared data loading and plot styles for the two-ring cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
SOLUTION_DIR = SCRIPT_DIR / "solution"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

RING_RADIUS = 1.0
RING_CIRCULATION = np.pi
REFERENCE_TIME = RING_RADIUS**2 / RING_CIRCULATION
FAMILIES = ("leapfrog", "collide")
FAMILY_LABELS = {"leapfrog": "Leapfrogging", "collide": "Collision"}
FAMILY_FILE_STEMS = {"leapfrog": "leapfrogging", "collide": "collision"}
INTENDED_CASE_ORDER = {
    name: order
    for order, name in enumerate(
        (
            "leapfrog_les",
            "leapfrog_les_splitting",
            "leapfrog_les_realignment",
            "collide_les",
            "collide_les_realignment",
        )
    )
}

_THEME_MODULE = None


def _theme():
    """Return the shared OpenONDA Matplotlib theme."""
    global _THEME_MODULE
    if _THEME_MODULE is None:
        if not THEME_PATH.exists():
            raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
        spec = importlib.util.spec_from_file_location("openonda_matplotlib_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        _THEME_MODULE = theme
    return _THEME_MODULE


def load_theme() -> tuple[dict[str, str], object | None]:
    """Apply the shared OpenONDA Matplotlib style."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size from the shared plot theme."""
    return _theme().figure_size(name)


def reference_style() -> dict:
    """Return the shared reference-line style."""
    return dict(_theme().REFERENCE_STYLE)


def reference_fill_style(kind: str = "normal") -> dict:
    """Return the shared reference-band style."""
    if kind == "strong":
        return dict(_theme().REFERENCE_STRONG_FILL_STYLE)
    return dict(_theme().REFERENCE_FILL_STYLE)


def legend_handle_style(style: dict) -> dict:
    """Return the shared legend-handle style."""
    return _theme().legend_handle_style(style)


def case_legend_handles(cases: list[str]) -> list:
    """Build one legend entry per plotted method."""
    from matplotlib.lines import Line2D

    handles = []
    labels = set()
    for name in cases:
        style = case_style(name)
        if style["label"] in labels:
            continue
        labels.add(style["label"])
        handles.append(Line2D([0], [0], **legend_handle_style(style)))
    return handles


def mark_every(name: str = "default") -> int:
    """Return the shared marker cadence for a plot kind."""
    return _theme().MARK_EVERY[name]


def secondary_line_style() -> dict:
    """Return the shared style for a secondary line."""
    theme = _theme()
    return {
        "linestyle": theme.SECONDARY_LINESTYLE,
        "linewidth": theme.SECONDARY_LINE_WIDTH,
    }


def build_arg_parser(description: str):
    """Build the argument parser shared by all plot scripts."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--solution-dir",
        default=str(SOLUTION_DIR),
        help="Root solution directory.",
    )
    parser.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI.")
    return parser


def _case_parts(name: str) -> tuple[str, str]:
    """Split ``<family>_<variant>`` into its two parts."""
    family, _, variant = name.partition("_")
    return family, variant


def case_style(name: str) -> dict:
    """Return a consistent style dictionary from the shared plot theme."""
    theme = _theme()
    _, variant = _case_parts(name)
    styles = {
        "les": {"label": "LES", "color": theme.COLORS["TUDcyan"], "marker": "s"},
        "les_splitting": {
            "label": "LES + splitting",
            "color": theme.COLORS["VPMpurple"],
            "marker": "D",
        },
        "les_realignment": {
            "label": "LES + realignment",
            "color": theme.COLORS["FVMorange"],
            "marker": "^",
        },
    }
    style = styles[variant]
    return {
        "color": style["color"],
        "linestyle": "-",
        "linewidth": theme.LINE_WIDTH,
        "marker": style["marker"],
        "markersize": theme.MARKER_SIZE,
        "markeredgewidth": theme.MARKER_EDGE_WIDTH,
        "label": style["label"],
    }


def discover_cases(solution_dir, family: str | None = None) -> list[Path]:
    """Return available cases in plotting order."""
    solution = Path(solution_dir)
    if not solution.is_dir():
        return []
    cases = []
    intended = INTENDED_CASE_ORDER
    for case_dir in solution.iterdir():
        if not case_dir.is_dir() or case_dir.name not in intended:
            continue
        case_family, _ = _case_parts(case_dir.name)
        if family and case_family != family:
            continue
        if (case_dir / "run_manifest.json").exists() or (
            _samples_dir(case_dir) / "flow_integrals.csv"
        ).exists():
            cases.append(case_dir)
    return sorted(cases, key=lambda path: intended[path.name])


def _samples_dir(case_dir: str | Path) -> Path:
    """Return the sample directory for one case."""
    case = Path(case_dir)
    return case.parent.parent / "samples" / case.name


def _trim_to_last_monotone_segment(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Keep the latest monotone segment after a restart."""
    if time_column not in df.columns or len(df) <= 1:
        return df
    times = df[time_column].to_numpy(float)
    last_restart = 0
    for index in range(1, len(times)):
        if np.isfinite(times[index]) and times[index] < times[index - 1]:
            last_restart = index
    if last_restart:
        df = df.iloc[last_restart:].reset_index(drop=True)
    return df


def read_integrals(case_dir) -> pd.DataFrame | None:
    """Return the VPM flow-integral history."""
    path = _samples_dir(case_dir) / "flow_integrals.csv"
    if not path.is_file():
        return None
    diagnostics = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "time" not in diagnostics or "total_kinetic_energy" not in diagnostics:
        return None
    diagnostics = diagnostics.dropna(subset=["time", "total_kinetic_energy"])
    if diagnostics.empty:
        return None
    return _trim_to_last_monotone_segment(diagnostics.reset_index(drop=True), "time")


def read_metric(case_dir, column: str):
    """Return nondimensional time and one global diagnostic."""
    diagnostics = read_integrals(case_dir)
    if diagnostics is None or column not in diagnostics:
        return np.array([]), np.array([])
    diagnostics = diagnostics.dropna(subset=["time", column])
    if diagnostics.empty:
        return np.array([]), np.array([])
    time = diagnostics["time"].to_numpy(float) / REFERENCE_TIME
    return time, diagnostics[column].to_numpy(float)


def read_ring_diagnostics(case_dir) -> pd.DataFrame | None:
    """Return the grouped vortex-ring history."""
    path = _samples_dir(case_dir) / "ring_diagnostics.csv"
    if not path.is_file():
        return None
    diagnostics = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = ["time", "step", "group_id", "vortex_centroid_x", "major_radius"]
    if diagnostics.empty or not set(required).issubset(diagnostics.columns):
        return None
    diagnostics = diagnostics.dropna(subset=required)
    diagnostics = _trim_to_last_monotone_segment(diagnostics, "time")
    return (
        diagnostics.sort_values(["step", "group_id"], kind="stable")
        .drop_duplicates(["step", "group_id"], keep="last")
        .reset_index(drop=True)
    )


def save_fig(
    fig,
    path,
    dpi: int | None = None,
    figure_format: str = "png",
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Save a figure through the shared theme."""
    _theme().save_fig(
        fig,
        path,
        figure_format=figure_format,
        dpi=dpi,
        tight_rect=tight_rect,
    )
