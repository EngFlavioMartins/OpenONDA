"""Shared OpenONDA matplotlib theme for flat-plate plot scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
SAMPLES_DIR = CASE_DIR / "samples"
FIG_DIR = CASE_DIR / "figures"
REPO_ROOT = CASE_DIR.parents[2]
THEME_PATH = REPO_ROOT / "docs" / "themes" / "matplotlib_setup.py"

_theme = None


def _load():
    global _theme
    if _theme is None:
        if not THEME_PATH.exists():
            raise FileNotFoundError(f"OpenONDA matplotlib theme not found: {THEME_PATH}")
        spec = importlib.util.spec_from_file_location("matplotlib_setup", str(THEME_PATH))
        _theme = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_theme)
        _theme.set_style()
    return _theme


def colors() -> dict[str, str]:
    """Return the theme colour palette."""
    return dict(_load().COLORS)


def color(key: str) -> str:
    """Return a single theme colour by key."""
    return _load().COLORS[key]


def cm() -> float:
    """Return the centimetre-to-inch conversion factor from the theme."""
    return _load().CM


def save_fig(fig, path, *, figure_format: str = "png", dpi: int = 300) -> None:
    """Save a figure through the shared OpenONDA theme."""
    _load().save_fig(fig, path, figure_format=figure_format, dpi=dpi)


def export_formats() -> tuple[str, ...]:
    """Return the supported export format strings."""
    return tuple(_load().EXPORT_FORMATS)
