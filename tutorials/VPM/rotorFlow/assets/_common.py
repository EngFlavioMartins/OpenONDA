"""Shared utilities for rotorFlow plot scripts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

# -- Directory layout --------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
FIGURES_DIR = CASE_DIR / "figures"
SOLUTION_DIR = CASE_DIR / "solution" / "rotor"
THEME_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"
FONT_PATH = CASE_DIR.parents[2] / "docs" / "themes" / "DejaVuSerif.ttf"


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme and return (COLORS, theme module)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    theme = None
    if THEME_PATH.exists():
        spec = importlib.util.spec_from_file_location("mpl_setup", THEME_PATH)
        theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(theme)
        try:
            theme.set_style()
        except Exception:
            pass

    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "DejaVu Serif"

    if theme is not None and hasattr(theme, "COLORS"):
        return dict(theme.COLORS), theme
    return {}, theme


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Base argument parser shared by rotorFlow plot scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--solution-dir", default=str(SOLUTION_DIR), help="Rotor solution directory.")
    p.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output figure directory.")
    p.add_argument("--format", choices=["png", "svg"], default="png")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (PNG only).")
    return p


def build_rotor_style_map(colors: dict[str, str]) -> dict[str, dict[str, object]]:
    """Map rotor-data sources to plot style dicts (color, marker, linestyle, label)."""
    gray_kw = {"color": "gray", "linestyle": "--", "linewidth": 1.0}
    return {
        "vpm": {
            "color": colors.get("vpm", "#5C3D9B"),
            "marker": "o",
            "markersize": 1.5,
            "linewidth": 1.0,
            "label": "VLM-VPM",
        },
        "bem": dict(gray_kw, label="BEM"),
        "theory": dict(gray_kw),
        "reference": dict(gray_kw),
        "ct": {
            "color": colors.get("vpm", "#5C3D9B"),
            "marker": "o",
            "markersize": 1.5,
            "linewidth": 1.0,
            "label": r"$C_T$",
        },
        "cp": {
            "color": colors.get("vpm", "#5C3D9B"),
            "marker": "s",
            "markersize": 1.5,
            "linewidth": 1.0,
            "label": r"$C_P$",
        },
        "plane_0": {"color": colors.get("vpm", "#5C3D9B"), "linewidth": 1.0},
        "plane_1": {"color": colors.get("vpm", "#5C3D9B"), "linewidth": 1.0},
        "plane_2": {"color": colors.get("vpm", "#5C3D9B"), "linewidth": 1.0},
        "plane_3": {"color": colors.get("vpm", "#5C3D9B"), "linewidth": 1.0},
        "plane_4": {"color": colors.get("vpm", "#5C3D9B"), "linewidth": 1.0},
    }
