"""Thin study-specific layer over the installed OpenONDA thesis plot style."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from openonda import plotting as project_plotting

THESIS_WIDTH = project_plotting.MAX_FIGURE_WIDTH_CM * project_plotting.CM
EXPORT_DPI = project_plotting.DEFAULT_DPI
PALETTE = project_plotting.PALETTE

METHOD_STYLE = {
    "coupled_ssprk3": {
        "label": "Coupled SSPRK3",
        "color": PALETTE["purple"],
        "marker": "o",
        "linestyle": "-",
    },
    "historical_lie": {
        "label": "Sequential RK3",
        "color": PALETTE["orange"],
        "marker": "s",
        "linestyle": "--",
    },
    "symmetric_strang": {
        "label": "Symmetric split",
        "color": PALETTE["teal"],
        "marker": "^",
        "linestyle": "-.",
    },
    "strang_rk4": {
        "label": "Symmetric split",
        "color": PALETTE["teal"],
        "marker": "^",
        "linestyle": "-.",
    },
}
METHOD_STYLE["ssp3_coupled"] = METHOD_STYLE["coupled_ssprk3"]


def apply_style() -> None:
    """Apply the repository's LaTeX/newpx thesis style without substitution."""
    project_plotting.set_thesis_style()
    plt.rcParams["svg.hashsalt"] = "openonda-vpm-time-integration"


def finish_layout(
    fig: plt.Figure,
    axes: Iterable[Axes] | Axes,
    **tight_layout_options,
) -> None:
    """Compact the layout, centre its plotting area, and enforce thesis rules."""
    axes = (axes,) if isinstance(axes, Axes) else tuple(axes)
    fig.tight_layout(**tight_layout_options)
    left = min(axis.get_position().x0 for axis in axes)
    right = 1.0 - max(axis.get_position().x1 for axis in axes)
    project_plotting.centered_subplots_adjust(fig, outer=max(left, right))
    project_plotting.validate_thesis_figure(fig, axes)


def save_all(fig: plt.Figure, stem: Path) -> None:
    """Export one final-size figure in the three requested formats."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        creator = "OpenONDA VPM time-integration study"
        if suffix == "pdf":
            fixed_date = datetime(2026, 9, 10, tzinfo=UTC)
            metadata = {"Creator": creator, "CreationDate": fixed_date, "ModDate": fixed_date}
        elif suffix == "svg":
            metadata = {"Creator": creator, "Date": "2026-09-10"}
        else:
            metadata = {"Software": creator}
        options = {"metadata": metadata}
        if suffix == "png":
            options["dpi"] = EXPORT_DPI
        fig.savefig(stem.with_suffix(f".{suffix}"), **options)
    plt.close(fig)
