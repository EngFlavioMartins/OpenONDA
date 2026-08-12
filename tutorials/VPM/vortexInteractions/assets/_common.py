"""Shared plotting utilities backed by VPM-managed diagnostic CSV files."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = ASSETS_DIR.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
SOLUTION_DIR = SCRIPT_DIR / "solution"
THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

R0 = 1.0
GAMMA = np.pi
T_REF = R0**2 / GAMMA

_THEME_MODULE = None


def _theme():
    """Return the shared OpenONDA matplotlib theme module."""
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
    """Load the OpenONDA matplotlib theme."""
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
    """Return the shared legend-handle style for a case style."""
    return _theme().legend_handle_style(style)


def compact_case_legend_handles(include_families: bool = True) -> list:
    """Build one compact method key and, optionally, one family key."""
    from matplotlib.lines import Line2D

    theme = _theme()
    handles = []
    for variant in theme.VARIANT_ORDER:
        style = case_style(f"leapfrog_{variant}", include_family=False)
        style["linestyle"] = "-"
        handles.append(Line2D([0], [0], **legend_handle_style(style)))
    if include_families:
        for family in ("leapfrog", "collide"):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=theme.PALETTE["black"],
                    linestyle=theme.FAMILY_LINESTYLE[family],
                    linewidth=theme.LINE_WIDTH,
                    label=theme.FAMILY_LABEL[family],
                )
            )
    return handles


def mark_every(name: str = "default") -> int:
    """Return the shared marker cadence for a plot kind."""
    return _theme().MARK_EVERY[name]


def secondary_line_style() -> dict:
    """Return the shared style for secondary lines related to a primary case."""
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
    parser.add_argument(
        "--figures-dir",
        default=str(FIGURES_DIR),
        help="Output directory for figures.",
    )
    parser.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI.")
    return parser


def _case_parts(name: str) -> tuple[str, str]:
    family, _, variant = name.partition("_")
    return family, variant


def case_style(
    name: str,
    include_family: bool = True,
    colors: dict[str, str] | None = None,
) -> dict:
    """Return a consistent style dictionary from the shared plot theme."""
    del colors
    return _theme().case_style(name, include_family=include_family)


def discover_cases(solution_dir, family: str | None = None) -> list[Path]:
    """Return cases carrying solver-managed diagnostics or a run manifest."""
    solution = Path(solution_dir)
    if not solution.is_dir():
        return []
    cases = []
    intended = _theme().INTENDED_CASE_ORDER
    for case_dir in solution.iterdir():
        if not case_dir.is_dir() or case_dir.name not in intended:
            continue
        case_family, _ = _case_parts(case_dir.name)
        if family and case_family != family:
            continue
        if (case_dir / "run_manifest.json").exists() or (
            case_dir / "samples" / "flow_integrals.csv"
        ).exists():
            cases.append(case_dir)
    return sorted(cases, key=lambda path: intended[path.name])


def _trim_to_last_monotone_segment(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
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
    """Return the complete built-in VPM flow-integral sample history."""
    path = Path(case_dir) / "samples" / "flow_integrals.csv"
    if not path.is_file():
        return None
    diagnostics = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    if "time" not in diagnostics or "kinetic_energy" not in diagnostics:
        return None
    diagnostics = diagnostics.dropna(subset=["time", "kinetic_energy"])
    if diagnostics.empty:
        return None
    return _trim_to_last_monotone_segment(diagnostics.reset_index(drop=True), "time")


def read_metric(case_dir, column: str, truncate_blowup: bool = True):
    """Return normalized time and one built-in global diagnostic."""
    del truncate_blowup
    diagnostics = read_integrals(case_dir)
    csv_column = "strength_magnitude" if column == "sum_gamma_magnitude" else column
    if diagnostics is None or csv_column not in diagnostics:
        return np.array([]), np.array([])
    diagnostics = diagnostics.dropna(subset=["time", csv_column])
    if diagnostics.empty:
        return np.array([]), np.array([])
    time = diagnostics["time"].to_numpy(float) / T_REF
    return time, diagnostics[csv_column].to_numpy(float)


def read_ring_diagnostics(case_dir) -> pd.DataFrame | None:
    """Return the built-in grouped vortex-ring sampler history."""
    path = Path(case_dir) / "samples" / "ring_diagnostics.csv"
    if not path.is_file():
        return None
    diagnostics = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    required = ["flow_time", "time_step", "group_id", "x_centroid", "major_radius"]
    if diagnostics.empty or not set(required).issubset(diagnostics.columns):
        return None
    diagnostics = diagnostics.dropna(subset=required)
    diagnostics = _trim_to_last_monotone_segment(diagnostics, "flow_time")
    return (
        diagnostics.sort_values(["time_step", "group_id"], kind="stable")
        .drop_duplicates(["time_step", "group_id"], keep="last")
        .reset_index(drop=True)
    )


def save_fig(
    fig,
    path,
    dpi: int | None = None,
    figure_format: str = "png",
    tight_rect: tuple[float, float, float, float] | None = None,
) -> None:
    """Save a figure through the shared OpenONDA theme."""
    _theme().save_fig(
        fig,
        path,
        figure_format=figure_format,
        dpi=dpi,
        tight_rect=tight_rect,
    )
