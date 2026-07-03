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

_THEME_MODULE = None


def _theme():
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
    """Load the OpenONDA matplotlib theme and return (COLORS, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


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
    return {name: dict(style) for name, style in _theme().ROTOR_STYLE.items()}
