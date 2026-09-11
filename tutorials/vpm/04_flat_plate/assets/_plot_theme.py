"""Shared OpenONDA matplotlib theme for flat-plate plot scripts."""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent
CASE_DIR = ASSETS_DIR.parent
SAMPLES_DIR = CASE_DIR / "samples"
FIG_DIR = CASE_DIR / "figures"

_theme = None


def _load():
    global _theme
    if _theme is None:
        from openonda import plotting as _theme

        _theme.set_thesis_style()
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


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a named thesis figure size in inches."""
    return _load().figure_size(name)


def centered_subplots_adjust(fig, *, outer: float, **kwargs) -> None:
    """Apply fixed layout with equal left and right plotting-area margins."""
    _load().centered_subplots_adjust(fig, outer=outer, **kwargs)


def save_fig(fig, path, *, figure_format: str = "png", dpi: int | None = None) -> None:
    """Validate and save a fixed-layout thesis figure without recropping it."""
    theme = _load()
    out = theme.figure_path(path, figure_format)
    out.parent.mkdir(parents=True, exist_ok=True)
    theme.validate_thesis_figure(fig, fig.axes)
    fig.savefig(out, dpi=theme.DEFAULT_DPI if dpi is None else dpi, bbox_inches=None)
    theme.plt.close(fig)
    print(f"  Saved: {out}")


def export_formats() -> tuple[str, ...]:
    """Return the supported export format strings."""
    return tuple(_load().EXPORT_FORMATS)
