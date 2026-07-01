"""Shared utilities for rotorFlow plot scripts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

CM = 1.0 / 2.54

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


def rotor_styles(colors: dict[str, str]) -> dict[str, dict[str, object]]:
    """Return consistent line styles for rotorFlow figures."""
    return {
        "vpm": {
            "color": colors.get("vpm", "#5C3D9B"),
            "marker": "o",
            "markersize": 2.2,
            "linewidth": 1.0,
        },
        "bem": {
            "color": colors.get("hybrid", "#772953"),
            "linestyle": "--",
            "linewidth": 1.0,
        },
        "theory": {
            "color": colors.get("DarkText", "#2E3D46"),
            "linestyle": "-",
            "linewidth": 1.0,
        },
        "reference": {
            "color": colors.get("RefGray", "#6E8898"),
            "linestyle": ":",
            "linewidth": 0.8,
        },
    }


def save_figure(fig, path: Path, dpi: int, fmt: str) -> None:
    """Save a matplotlib figure with repository-standard export settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, object] = {"bbox_inches": "tight"}
    if fmt == "png":
        save_kw["dpi"] = dpi
    fig.savefig(path, **save_kw)
