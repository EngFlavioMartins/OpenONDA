"""Shared plotting style, figure export, and CLI argument setup for the
vortex_ring plot scripts.

Each plot script lives in assets/ and imports from here via::

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from plot_style import load_theme, build_arg_parser, ...
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ring_metrics import SCRIPT_DIR

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


VARIANT_STYLE = _theme().VORTEX_RING_VARIANT_STYLE
VARIANT_LABEL = _theme().VORTEX_RING_VARIANT_LABEL


def load_theme() -> tuple[dict[str, str], object | None]:
    """Load the OpenONDA matplotlib theme. Returns (COLORS dict, theme module)."""
    theme = _theme()
    theme.set_style()
    return dict(theme.COLORS), theme


def figure_size(name: str = "single") -> tuple[float, float]:
    return _theme().figure_size(name)


def mark_every(name: str = "default") -> int:
    return _theme().MARK_EVERY[name]


def reference_style() -> dict:
    return dict(_theme().REFERENCE_STYLE)


def build_arg_parser(description: str):
    """Base argument parser shared by all plot scripts."""
    import argparse

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--format", choices=_theme().EXPORT_FORMATS, default="png")
    p.add_argument("--dpi", type=int, default=_theme().DEFAULT_DPI, help="Figure DPI.")
    return p


def save_fig(
    fig,
    path,
    dpi: int | None = None,
    tight_rect: tuple[float, float, float, float] | None = None,
    figure_format: str = "png",
) -> None:
    """Save without tight layout or cropping; manual subplots_adjust() takes precedence.

    Matches the Lamb--Oseen tutorial convention: ``bbox_inches=None`` and no
    ``tight_layout``, so the ``left/right/top/bottom/wspace/hspace`` controls
    set by each plot script are respected exactly.
    """
    import matplotlib.pyplot as plt

    out = Path(path)
    fmt = figure_format or "png"
    if fmt not in _theme().EXPORT_FORMATS:
        raise ValueError(f"Unsupported figure format: {fmt!r}")
    out = out.with_suffix(f".{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_theme().DEFAULT_DPI if dpi is None else dpi, bbox_inches=None)
    plt.close(fig)
    print(f"  Saved: {out}")
