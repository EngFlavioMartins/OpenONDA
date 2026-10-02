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
    _load().save_validation_figure(fig, path, figure_format=figure_format, dpi=dpi)


def validation_subplots(*args, **kwargs):
    return _load().validation_subplots(*args, **kwargs)


def validation_legend(fig, axis, *, ncol=2, outside=False):
    """Keep the rounded legend frame; reserve exterior spacing in each plotter."""
    _load()
    if outside:
        handles, labels = axis.get_legend_handles_labels()
        return fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=ncol,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            facecolor="white",
            edgecolor="0.8",
            borderaxespad=0,
            handlelength=1.5,
            columnspacing=0.8,
            handletextpad=0.4,
        )
    legend = _load().validation_legend(fig, axis, ncol=ncol)
    legend.get_frame().set_edgecolor("0.8")
    legend.get_frame().set_facecolor("white")
    return legend


def export_formats() -> tuple[str, ...]:
    """Return the supported export format strings."""
    return tuple(_load().FORMAT_CHOICES)
