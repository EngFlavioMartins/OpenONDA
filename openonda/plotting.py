"""Shared figure style for OpenONDA plots: palette, sizes, line styles and export."""

from collections.abc import Iterable
from pathlib import Path
import shutil

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import numpy as np

# Figures are written to files. The non-interactive backend keeps a long series of
# figures from accumulating window-system memory after each figure is closed.
plt.switch_backend("Agg")

# Plot style.
# All plot presentation lives here: palette, font sizes, figure sizes,
# markers, line widths, reference styles, and export defaults.
CM = 1 / 2.54
THESIS_FONT_SIZE_PT = 10.95  # \normalsize in the thesis's 11pt class
FONT_SIZE_PT = THESIS_FONT_SIZE_PT
DEFAULT_DPI = 400
EXPORT_FORMATS = ("png", "pdf")
FORMAT_CHOICES = (*EXPORT_FORMATS, "both")
MAX_FIGURE_WIDTH_CM = 12.5
WIDE_FIGURE_WIDTH_CM = 12.5
MIN_TEXT_CANVAS_PADDING_PT = 2.0
MIN_LINE_WIDTH_PT = 0.6  # At the 125 mm printed figure width.
FONT_PATH = Path(__file__).parent / "_resources" / "DejaVuSerif.ttf"

FIGURE_SIZES_CM = {
    "single": (MAX_FIGURE_WIDTH_CM, 7.0),
    "single_short": (MAX_FIGURE_WIDTH_CM, 6.2),
    "single_tall": (MAX_FIGURE_WIDTH_CM, 8.0),
    "trajectory": (MAX_FIGURE_WIDTH_CM, 7.5),
    "stacked": (MAX_FIGURE_WIDTH_CM, 12.5),
    "stacked_tall": (MAX_FIGURE_WIDTH_CM, 12.5),
    "wide": (WIDE_FIGURE_WIDTH_CM, 9.0),
    "wide_short": (WIDE_FIGURE_WIDTH_CM, 8.0),
    "wide_stacked": (WIDE_FIGURE_WIDTH_CM, 12.5),
}

LINE_WIDTH = 1.0
REFERENCE_LINE_WIDTH = 1.0
MARKER_SIZE = 3.0
MARKER_EDGE_WIDTH = MIN_LINE_WIDTH_PT
MARK_EVERY = {
    "default": 3,
    "total_kinetic_energy": 4,
    "trajectory": 5,
}

# Approved Thesis/thesis_visuals palette (2026-10-02).
PALETTE = {
    "dark": "#0D3D45",
    "teal": "#156074",
    "blue": "#41A6C4",
    "amber": "#EF9C1F",
    "green": "#629B42",
    "red": "#BF2326",
    "gray": "#60686C",
    "text": "#28353A",
    "light_gray": "#C0C0C0",
    "strong_gray": "#8B8B8B",
    "white": "#ffffff",
    "black": "#000000",
}
COLOR_CYCLE = tuple(
    PALETTE[name] for name in ("teal", "blue", "amber", "red", "green", "gray", "dark")
)
BACKGROUND_LIGHT = PALETTE["light_gray"]
BACKGROUND_STRONG = PALETTE["strong_gray"]
REFERENCE_GRAY = PALETTE["gray"]

COLORS = {
    **PALETTE,
    "background_light": BACKGROUND_LIGHT,
    "background_strong": BACKGROUND_STRONG,
    "reference": REFERENCE_GRAY,
    "reference_fill": BACKGROUND_LIGHT,
    "vpm": PALETTE["blue"],
    "hybrid": PALETTE["amber"],
    "fvm": PALETTE["teal"],
    "dvh": PALETTE["green"],
    "dvhr": PALETTE["teal"],
    "dns": PALETTE["dark"],
}

# Sequential maps for magnitudes; a zero-centred diverging map for signed fields.
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

for _name, _stops in {
    "thesis_blue": ["#0D3D45", "#156074", "#41A6C4", "#C8EDF5"],
    "thesis_amber": ["#6B3019", "#B45D1E", "#EF9C1F", "#FFF0CA"],
    "thesis_signed": ["#156074", "#F5F6F5", "#BF2326"],
}.items():
    if _name not in colormaps:
        colormaps.register(LinearSegmentedColormap.from_list(_name, _stops, N=256))

COLORMAPS = dict.fromkeys(
    (
        "field_speed",
        "field_vorticity",
        "vorticity_magnitude",
        "velocity",
        "error",
        "vortex_speed",
        "vortex_vorticity",
    ),
    "thesis_blue",
)
COLORMAPS.update(
    vorticity="thesis_signed", error_diverging="thesis_signed", streamlines="thesis_amber"
)

REFERENCE_STYLE = {
    "color": COLORS["reference"],
    "linestyle": "--",
    "linewidth": REFERENCE_LINE_WIDTH,
}


def get_color(name: str, fallback: str | None = None) -> str:
    """Return a named color from the central palette."""
    if fallback is None:
        fallback = COLORS["reference"]
    return COLORS.get(name, fallback)


def get_colormap(name: str) -> str:
    """Return a named colormap from the central palette."""
    return COLORMAPS[name]


def figure_path(path, figure_format: str = "png") -> Path:
    """Return a figure path with one of the supported export suffixes."""
    if figure_format not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported figure format: {figure_format!r}")
    return Path(path).with_suffix(f".{figure_format}")


def figure_size(name: str = "single") -> tuple[float, float]:
    """Return a figure size in inches from a named centimetre preset."""
    width_cm, height_cm = FIGURE_SIZES_CM[name]
    return width_cm * CM, height_cm * CM


def centered_subplots_adjust(fig, *, outer: float, **kwargs) -> None:
    """Apply manual spacing with equal left and right plotting-area margins."""
    if not 0.0 < outer < 0.5:
        raise ValueError("outer must lie between 0 and 0.5")
    with plt.rc_context({"figure.constrained_layout.use": False, "figure.autolayout": False}):
        fig.set_layout_engine(None)
    fig.subplots_adjust(left=outer, right=1.0 - outer, **kwargs)


def figure_renderer(fig) -> RendererAgg:
    """Render the authored canvas without changing its selected backend."""
    renderer = RendererAgg(fig.bbox.width, fig.bbox.height, fig.dpi)
    fig.draw(renderer)
    return renderer


def thesis_y_label_margin(fig, axes: Iterable[Axes] | Axes) -> float:
    """Measure a symmetric plot margin with outer y text just inside the canvas.

    Measure rendered labels at their final font size. The returned fraction
    is used for both left and right margins, not for asymmetric tight cropping.
    """
    axes = (axes,) if isinstance(axes, Axes) else tuple(axes)
    renderer = figure_renderer(fig)
    left = min(axis.get_position().x0 for axis in axes)
    text_left = float("inf")
    for axis in axes:
        if abs(axis.get_position().x0 - left) > 1e-6:
            continue
        labels = [axis.yaxis.label, axis.yaxis.get_offset_text()]
        lower, upper = sorted(axis.get_ylim())
        labels += [
            tick.label1 for tick in axis.yaxis.get_major_ticks() if lower <= tick.get_loc() <= upper
        ]
        for label in labels:
            if label.get_visible() and label.get_text().strip():
                text_left = min(text_left, label.get_window_extent(renderer).x0)
    if not np.isfinite(text_left):
        return left
    # A half-point reserve absorbs small raster/PDF text metric differences.
    padding = (MIN_TEXT_CANVAS_PADDING_PT + 0.5) * fig.dpi / 72.0
    return left + (fig.bbox.x0 + padding - text_left) / fig.bbox.width


def fit_thesis_y_label_margins(fig, axes: Iterable[Axes] | Axes) -> None:
    """Place y labels near the edge, retaining an x-centred subplot grid."""
    axes = (axes,) if isinstance(axes, Axes) else tuple(axes)
    for _ in range(3):
        _separate_corner_tick_labels(fig, axes)
        _separate_title_from_y_ticks(fig, axes)
        outer = thesis_y_label_margin(fig, axes)
        centered_subplots_adjust(fig, outer=outer)


def _separate_corner_tick_labels(fig, axes: tuple[Axes, ...]) -> None:
    """Give intersecting x/y tick labels enough horizontal room to clear."""
    renderer = figure_renderer(fig)
    gap = fig.dpi / 72.0
    for axis in axes:
        xlow, xhigh = sorted(axis.get_xlim())
        ylow, yhigh = sorted(axis.get_ylim())
        for major in (True, False):
            x_ticks = axis.xaxis.get_major_ticks() if major else axis.xaxis.get_minor_ticks()
            y_ticks = axis.yaxis.get_major_ticks() if major else axis.yaxis.get_minor_ticks()
            x_labels = [
                tick.label1.get_window_extent(renderer)
                for tick in x_ticks
                if xlow <= tick.get_loc() <= xhigh
                and tick.label1.get_visible()
                and tick.label1.get_text().strip()
            ]
            needed = 0.0
            for tick in y_ticks:
                label = tick.label1
                if not (
                    ylow <= tick.get_loc() <= yhigh
                    and label.get_visible()
                    and label.get_text().strip()
                ):
                    continue
                bounds = label.get_window_extent(renderer)
                for x_bounds in x_labels:
                    if bounds.overlaps(x_bounds):
                        needed = max(needed, bounds.x1 - x_bounds.x0 + gap)
            if needed > 0.0 and y_ticks:
                axis.yaxis.set_tick_params(
                    which="major" if major else "minor",
                    pad=y_ticks[0].get_pad() + needed * 72.0 / fig.dpi,
                )


def _separate_title_from_y_ticks(fig, axes: tuple[Axes, ...]) -> None:
    """Raise a title only when its rendered bounds touch a y tick."""
    renderer = figure_renderer(fig)
    gap = fig.dpi / 72.0
    for axis in axes:
        title = axis.title
        if not title.get_visible() or not title.get_text().strip():
            continue
        title_bounds = title.get_window_extent(renderer)
        lower, upper = sorted(axis.get_ylim())
        needed = 0.0
        for tick in (*axis.yaxis.get_major_ticks(), *axis.yaxis.get_minor_ticks()):
            if not lower <= tick.get_loc() <= upper:
                continue
            for label in (tick.label1, tick.label2):
                if not label.get_visible() or not label.get_text().strip():
                    continue
                bounds = label.get_window_extent(renderer)
                if bounds.overlaps(title_bounds):
                    needed = max(needed, bounds.y1 - title_bounds.y0 + gap)
        if needed > 0.0:
            _, y = title.get_position()
            axis.set_title(
                title.get_text(),
                loc="center",
                y=y + needed / axis.bbox.height,
                fontproperties=title.get_fontproperties(),
                color=title.get_color(),
            )


class TwoSignificantDigitsFormatter(ScalarFormatter):
    """Two significant digits, with a shared exponent/offset for close values."""

    def _set_format(self):
        template = r"$\mathdefault{%1.2g}$" if self._usetex or self._useMathText else "%1.2g"
        if hasattr(self, "_format"):
            self._format = template
        else:
            self.format = template


def prepare_figure(fig):
    """Apply print precision and stroke floors without changing authored layout.

    Zero-width/absent edges stay absent. Full data precision, locators, limits,
    normalisation and explicitly authored textual tick labels are retained.
    """
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            formatter = axis.get_major_formatter()
            if isinstance(formatter, ScalarFormatter) and not isinstance(
                formatter, TwoSignificantDigitsFormatter
            ):
                new = TwoSignificantDigitsFormatter(useMathText=True)
                new.set_powerlimits(
                    formatter.get_powerlimits()
                    if hasattr(formatter, "get_powerlimits")
                    else (-3, 2)
                )
                lo, hi = sorted(axis.get_view_interval())
                # A compact additive offset preserves visible variation near 1,
                # e.g. ring-circulation values 0.996--1.000, without long labels.
                if lo * hi > 0 and hi - lo < 0.1 * min(abs(lo), abs(hi)):
                    new.set_useOffset(float(format((lo + hi) / 2, ".2g")))
                axis.set_major_formatter(new)
            elif isinstance(formatter, FormatStrFormatter):
                axis.set_major_formatter(FormatStrFormatter("%.2g"))
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                tick.tick1line.set_markeredgewidth(MIN_LINE_WIDTH_PT)
                tick.tick2line.set_markeredgewidth(MIN_LINE_WIDTH_PT)
        for spine in ax.spines.values():
            spine.set_linewidth(max(MIN_LINE_WIDTH_PT, spine.get_linewidth()))
    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        if hasattr(artist, "get_linewidth") and hasattr(artist, "set_linewidth"):
            width = np.asarray(artist.get_linewidth(), dtype=float)
            if width.size:
                adjusted = np.where(width > 0, np.maximum(width, MIN_LINE_WIDTH_PT), width)
                artist.set_linewidth(float(adjusted) if adjusted.ndim == 0 else adjusted)
        if isinstance(artist, Line2D) and artist.get_marker() not in (None, "None", "", " "):
            artist.set_markeredgewidth(max(MIN_LINE_WIDTH_PT, artist.get_markeredgewidth()))


def validate_thesis_figure(fig, axes: Iterable[Axes] | Axes) -> None:
    """Check thesis figure size, centring, and consistent fonts."""
    prepare_figure(fig)
    axes = (axes,) if isinstance(axes, Axes) else tuple(axes)
    if not axes:
        raise ValueError("at least one plotting axis is required")

    width_cm = fig.get_figwidth() / CM
    # GUI backends quantise the initial canvas to whole pixels and can turn a
    # requested 12.5 cm width into 12.4968 cm.  Snap near-limit figures back
    # to the exact physical width without asking the GUI manager to resize.
    if abs(width_cm - MAX_FIGURE_WIDTH_CM) < 0.01:
        fig.set_size_inches(
            MAX_FIGURE_WIDTH_CM * CM,
            fig.get_figheight(),
            forward=False,
        )
        width_cm = fig.get_figwidth() / CM
    if width_cm > MAX_FIGURE_WIDTH_CM + 1e-3:
        raise RuntimeError(
            f"figure width is {width_cm:.3f} cm; limit is {MAX_FIGURE_WIDTH_CM:g} cm"
        )
    left = min(axis.get_position().x0 for axis in axes)
    right = 1.0 - max(axis.get_position().x1 for axis in axes)
    if abs(left - right) > 1e-6:
        raise RuntimeError(
            f"plotting area is not centred: left margin={left:.6f}, right margin={right:.6f}"
        )

    renderer = figure_renderer(fig)
    canvas = fig.bbox
    minimum_padding = MIN_TEXT_CANVAS_PADDING_PT * fig.dpi / 72.0
    in_range_tick_label: dict[Text, bool] = {}
    for axis in fig.axes:
        for axis_object, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            lower, upper = sorted(limits)
            tolerance = (
                16
                * np.finfo(float).eps
                * max(abs(lower), abs(upper), abs(upper - lower), np.finfo(float).tiny)
            )
            for tick in (*axis_object.get_major_ticks(), *axis_object.get_minor_ticks()):
                in_range = lower - tolerance <= tick.get_loc() <= upper + tolerance
                in_range_tick_label[tick.label1] = in_range
                in_range_tick_label[tick.label2] = in_range
    painted_text: list[tuple[Text, object]] = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        if item in in_range_tick_label and not in_range_tick_label[item]:
            continue
        if abs(float(item.get_fontsize()) - THESIS_FONT_SIZE_PT) > 1e-6:
            raise RuntimeError(
                f"text {item.get_text()!r} uses {item.get_fontsize():g} pt; "
                f"expected {THESIS_FONT_SIZE_PT:g} pt"
            )
        bounds = item.get_window_extent(renderer=renderer)
        if (
            bounds.x1 <= canvas.x0
            or bounds.y1 <= canvas.y0
            or bounds.x0 >= canvas.x1
            or bounds.y0 >= canvas.y1
        ):
            # Matplotlib creates tick-label artists just outside fixed data
            # limits; they are not painted and therefore cannot be clipped.
            continue
        clearances = (
            bounds.x0 - canvas.x0,
            canvas.x1 - bounds.x1,
            bounds.y0 - canvas.y0,
            canvas.y1 - bounds.y1,
        )
        if min(clearances) < minimum_padding:
            raise RuntimeError(
                f"text {item.get_text()!r} has only "
                f"{min(clearances) * 72.0 / fig.dpi:.2f} pt clearance from the "
                f"fixed figure canvas; expected at least {MIN_TEXT_CANVAS_PADDING_PT:g} pt"
            )
        for previous, previous_bounds in painted_text:
            overlap = bounds.intersection(bounds, previous_bounds)
            if overlap is not None and overlap.width > 1.0 and overlap.height > 1.0:
                raise RuntimeError(
                    f"text overlap between {previous.get_text()!r} and {item.get_text()!r}"
                )
        painted_text.append((item, bounds))


def set_style(*, use_tex: bool = False):
    """Apply the OpenONDA thesis plotting style.

    Matplotlib's built-in math renderer is the portable default. Pass
    ``use_tex=True`` only with a working LaTeX and dvipng installation.
    """
    if FONT_PATH.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(FONT_PATH))

    tex_fonts = {
        "text.usetex": use_tex,
        "text.latex.preamble": r"\usepackage{amsmath}" if use_tex else "",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "figure.titlesize": FONT_SIZE_PT,
        "figure.labelsize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
        "legend.title_fontsize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT,
        "ytick.labelsize": FONT_SIZE_PT,
        "figure.dpi": DEFAULT_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.major.width": MIN_LINE_WIDTH_PT,
        "ytick.major.width": MIN_LINE_WIDTH_PT,
        "xtick.minor.width": MIN_LINE_WIDTH_PT,
        "ytick.minor.width": MIN_LINE_WIDTH_PT,
        "axes.grid": False,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.grid.which": "both",
        "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
        "axes.edgecolor": "black",
        "axes.linewidth": MIN_LINE_WIDTH_PT,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
    }

    plt.rcParams.update(tex_fonts)


def requested_formats(figure_format="both"):
    """Expand the CLI format without treating 'both' as a filename extension."""
    if figure_format == "both":
        return ("pdf", "png")
    if figure_format not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported figure format: {figure_format!r}")
    return (figure_format,)


def export_figure(fig, path, *, figure_format=None, dpi=None, close=True):
    """Export an already laid-out canvas at native size, without cropping.

    PDF is saved first so both formats use identical physical geometry.
    """
    prepare_figure(fig)
    validate_thesis_figure(fig, fig.axes)
    path = Path(path)
    suffix_format = path.suffix.lstrip(".")
    formats = requested_formats(
        figure_format or (suffix_format if suffix_format in FORMAT_CHOICES else "both")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        output = (
            path.with_suffix(f".{fmt}")
            if suffix_format in FORMAT_CHOICES
            else Path(f"{path}.{fmt}")
        )
        fig.savefig(output, dpi=DEFAULT_DPI if dpi is None else dpi, bbox_inches=None)
        outputs.append(output)
        print(f"Saved: {output}")
    if close:
        plt.close(fig)
    return tuple(outputs)


def save_fig(fig, path, figure_format="both", dpi=None):
    """Lay out, validate and export a simple thesis figure in PNG and PDF."""
    prepare_figure(fig)
    validate_thesis_figure(fig, fig.axes)
    return export_figure(fig, path, figure_format=figure_format, dpi=dpi)


_THESIS_LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{newpxtext}\usepackage{amsmath}\usepackage{newpxmath}\newcommand{\vect}[1]{\boldsymbol{#1}}\newcommand{\dd}{\mathrm{d}}\newcommand{\zetaeps}{\zeta_{\sigma}}\newcommand{\Reyn}{\mathrm{Re}}"
# Serif families close to Palatino for figures drawn without LaTeX; DejaVu
# Serif, bundled with Matplotlib, always resolves.
_PALATINO_LIKE_FONTS = ["TeX Gyre Pagella", "Palatino Linotype", "URW Palladio L", "DejaVu Serif"]


def set_thesis_style():
    """Use the thesis body fonts at final physical size.

    With LaTeX and dvipng on PATH, LaTeX typesets the text in Palatino
    (newpx). Without them, Matplotlib's math text renders the same labels in a
    Palatino-like serif, so figures keep their sizes and layout.
    """
    use_tex = all(shutil.which(tool) for tool in ("latex", "dvipng"))
    set_style()
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "text.latex.preamble": _THESIS_LATEX_PREAMBLE if use_tex else "",
            "mathtext.fontset": "custom",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            "font.family": "serif",
            "font.serif": ["Palatino"] if use_tex else _PALATINO_LIKE_FONTS,
            **dict.fromkeys(
                (
                    "font.size",
                    "axes.labelsize",
                    "axes.titlesize",
                    "figure.titlesize",
                    "figure.labelsize",
                    "legend.fontsize",
                    "legend.title_fontsize",
                    "xtick.labelsize",
                    "ytick.labelsize",
                ),
                THESIS_FONT_SIZE_PT,
            ),
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
        }
    )


def validation_subplots(
    nrows, *, height_cm=None, sharex=False, outer=0.17, top_padding_cm=0.65, bottom_padding_cm=1.35
):
    """Compact boxed panels at the thesis width, with no caption gutters."""
    set_thesis_style()
    height = min(height_cm or (2 + 3.6 * nrows), 2 + 3.6 * nrows)
    fig, axes = plt.subplots(
        nrows, 1, squeeze=False, sharex=sharex, figsize=(MAX_FIGURE_WIDTH_CM * CM, height * CM)
    )
    centered_subplots_adjust(
        fig,
        outer=outer,
        bottom=bottom_padding_cm / height,
        top=1 - top_padding_cm / height,
        hspace=0.12 if sharex else 0.48,
    )
    return fig, axes[:, 0]


def validation_legend(fig, axis, *, ncol=2):
    """Use one compact legend inside a panel."""
    return axis.legend(
        loc="best",
        ncol=ncol,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        fancybox=True,
        facecolor="white",
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )


def save_validation_figure(fig, path, *, figure_format="both", dpi=None):
    """Check text extents and collisions before saving a comparison figure."""
    prepare_figure(fig)
    validate_thesis_figure(fig, fig.axes)
    return export_figure(fig, path, figure_format=figure_format, dpi=dpi)
