"""Shared plotting style, figure export, and CLI argument setup for the
Lamb--Oseen tutorial plot scripts.

Wraps the shared OpenONDA theme (``docs/themes/matplotlib_setup.py``) rather
than duplicating it — except ``save_fig``, which deliberately skips the
theme's ``tight_layout()``/``bbox_inches="tight"`` defaults: every plot in
this tutorial places its legend with a manual ``fig.subplots_adjust(...)``,
and the theme's auto-layout would fight that.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

if __package__:
    from .vortex_diagnostics import (
        CORE_RADIUS,
        FIGURES_DIR,
        REFERENCE_CIRCULATION,
        REYNOLDS_NUMBER,
        SAMPLES_DIR,
        SCRIPT_DIR,
        SEPARATION,
    )
else:
    from vortex_diagnostics import (
        CORE_RADIUS,
        FIGURES_DIR,
        REFERENCE_CIRCULATION,
        REYNOLDS_NUMBER,
        SAMPLES_DIR,
        SCRIPT_DIR,
        SEPARATION,
    )

THEME_PATH = SCRIPT_DIR.parents[2] / "docs" / "themes" / "matplotlib_setup.py"

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
    """Load the OpenONDA matplotlib theme and return (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def build_style_map(colors: dict[str, str]) -> dict[str, dict]:
    """Map scheme names to plot style dicts (color, marker, label)."""
    return {name: dict(style) for name, style in _theme().LAMB_OSEEN_SCHEME_STYLE.items()}


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named figure size in inches from the shared theme."""
    return _theme().figure_size(name)


def save_fig(fig, path: Path, dpi: int) -> None:
    """Save without tight layout or cropping; manual subplots_adjust() takes precedence."""
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {out}")


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI (PNG only).")
    p.add_argument(
        "--format",
        choices=_theme().EXPORT_FORMATS,
        default="png",
        help="Output figure format (default: png).",
    )
    kinematic_viscosity = REFERENCE_CIRCULATION / REYNOLDS_NUMBER
    p.set_defaults(
        samples_dir=SAMPLES_DIR,
        figures_dir=FIGURES_DIR,
        circulation=REFERENCE_CIRCULATION,
        kinematic_viscosity=kinematic_viscosity,
        b0=SEPARATION,
        a0_over_b0=CORE_RADIUS / SEPARATION,
    )
    return p
