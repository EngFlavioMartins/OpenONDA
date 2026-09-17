"""Shared plotting style and figure helpers for installed OpenONDA tutorials."""

# =================================================
# Standard library imports
# =================================================
from collections.abc import Iterable
from pathlib import Path
import shutil

from matplotlib.axes import Axes

# =================================================
# Third-party library imports
# =================================================
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

from source.solution_layout import collection_path

# =================================================
# OpenONDA project-specific imports moved to functions
# =================================================


# ==================================================


def read_vlm_surface(surface_record: dict, geometry_directory: str | Path) -> dict:
    """Read the run's native geometry snapshot, including older saved cases.

    Current VPM metadata embeds the geometry loaded by the solver, so plotting
    remains valid after input files change or disappear. Older records retain
    a filename; resolve those within the copied tutorial's geometry directory.
    """
    import json

    if "geometry" in surface_record:
        return surface_record["geometry"]
    path = Path(geometry_directory) / Path(surface_record["surface"]).name
    return json.loads(path.read_text())


def latest_fvm_snapshot(solution_directory: str | Path) -> Path | None:
    """Return the latest field snapshot in the solver's recorded time series.

    Parameters
    ----------
    solution_directory : str or pathlib.Path
        FVM output directory containing ``fvm_metadata.json`` and the case's
        ``fvm.pvd`` index. Geometry-only ``fvm/mesh.vtu`` files are never selected.

    Returns
    -------
    pathlib.Path or None
        Snapshot with the greatest recorded physical time, or ``None`` when
        the solver has not yet published a field time series.

    Raises
    ------
    ValueError, KeyError
        If a published metadata or time-series record is malformed.
    """
    import json

    from defusedxml import ElementTree

    directory = Path(solution_directory)
    metadata_path = directory / "fvm_metadata.json"
    if not metadata_path.is_file():
        return None
    json.loads(metadata_path.read_text())
    series = collection_path(directory, "fvm")
    if not series.is_file():
        return None
    frames = ElementTree.parse(series).findall(".//DataSet")
    if not frames:
        return None
    latest = max(frames, key=lambda frame: float(frame.attrib["timestep"]))
    return directory / latest.attrib["file"]


def theoretical_ring_trajectory(
    kinematic_viscosity: float,
    initial_core_radius: float,
    initial_ring_radius: float,
    tube_circulation: float,
    time: np.ndarray,
):
    """
    Computes the theoretical trajectory and speed of a vortex ring over time.

    Arguments:
    ----------
    kinematic_viscosity : float
        The kinematic viscosity of the fluid.
    initial_core_radius : float
        The initial core radius of the vortex ring.
    initial_ring_radius : float
        The initial radius of the vortex ring.
    tube_circulation : float
        The vortex-ring tube circulation.
    time : np.ndarray
        Array of time points at which to compute the trajectory.

    Returns:
    --------
    theoretical_ring_position : np.ndarray
        Theoretical cumulative distance traveled by the ring over time.
    theoretical_ring_velocity : np.ndarray
        Theoretical speed of the ring at each time point.
    """
    # Calculate ring thickness over time
    ring_core_radius = np.sqrt(4 * kinematic_viscosity * time + initial_core_radius**2)

    # Core thickness ratio
    core_to_ring_radius_ratio = ring_core_radius / initial_ring_radius

    # Empirical correction factor for finite core thickness
    correction_coefficient = (
        -0.558 - 1.12 * core_to_ring_radius_ratio**2 - 5.0 * core_to_ring_radius_ratio**4
    )

    # Term A based on initial circulation and radius
    circulation_velocity_scale = tube_circulation / (4 * np.pi * initial_ring_radius)

    # Term B includes the logarithmic factor and correction coefficient
    logarithmic_correction = np.log(8 / core_to_ring_radius_ratio) + correction_coefficient

    # Compute theoretical ring speed
    theoretical_ring_velocity = circulation_velocity_scale * logarithmic_correction

    # Calculate time step from the time array
    time_increment = np.gradient(time)

    # Compute theoretical ring location by cumulative sum of speed * time_step
    theoretical_ring_position = np.cumsum(theoretical_ring_velocity * time_increment)

    return theoretical_ring_position, theoretical_ring_velocity


# -- Plot style ---------------------------------------------------------------
# All tutorial plot presentation lives here: palette, font sizes, figure sizes,
# markers, line widths, reference styles, and export defaults.
CM = 1 / 2.54
THESIS_FONT_SIZE_PT = 10.95  # \normalsize in the thesis's 11pt class
FONT_SIZE_PT = THESIS_FONT_SIZE_PT
DEFAULT_DPI = 400
EXPORT_FORMATS = ("png", "pdf")
MAX_FIGURE_WIDTH_CM = 12.5
WIDE_FIGURE_WIDTH_CM = 12.5
MIN_TEXT_CANVAS_PADDING_PT = 5.0
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

LINE_WIDTH = 1.1
SECONDARY_LINE_WIDTH = 1.0
REFERENCE_LINE_WIDTH = 1.0
MARKER_SIZE = 3.0
LEGEND_MARKER_SIZE = 4.0
MARKER_EDGE_WIDTH = 0.4
SECONDARY_LINESTYLE = ":"
MARK_EVERY = {
    "default": 3,
    "total_kinetic_energy": 4,
    "trajectory": 5,
}

# Match Thesis/thesis.tex (the rendered document is authoritative where the
# older thesis_visuals/styles/colors.py differs). Keep legacy palette names
# for compatibility: "orange" is the thesis's FVMorange aubergine.
PALETTE = {
    "dark": "#0C2340",
    "teal": "#0E8A85",
    "purple": "#5C3D9B",
    "orange": "#772953",
    "green": "#2B7A4E",
    "red": "#9C2F50",
    "gray": "#6E8898",
    "text": "#2E3D46",
    "light_gray": "#C0C0C0",
    "strong_gray": "#8B8B8B",
    "white": "#ffffff",
    "black": "#000000",
}
COLOR_CYCLE = (
    PALETTE["teal"],
    PALETTE["purple"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["gray"],
    PALETTE["dark"],
)
BACKGROUND_LIGHT = PALETTE["light_gray"]
BACKGROUND_STRONG = PALETTE["strong_gray"]
REFERENCE_GRAY = PALETTE["gray"]

COLORS = {
    # Named thesis colours and compatibility aliases.
    "TUDdark": PALETTE["dark"],
    "TUDcyan": PALETTE["teal"],
    "TUDred": PALETTE["red"],
    "VPMpurple": PALETTE["purple"],
    "FVMorange": PALETTE["orange"],
    "AccentGreen": PALETTE["green"],
    "AccentRed": PALETTE["red"],
    "BackgroundLight": BACKGROUND_LIGHT,
    "BackgroundGray": BACKGROUND_STRONG,
    "ReferenceGray": REFERENCE_GRAY,
    "RefGray": REFERENCE_GRAY,
    "DarkText": PALETTE["text"],
    "LightBG": "#EDF3F5",
    "LightCyan": "#CBE8E7",
    "LightPurple": "#E5E0F5",
    "LightOrange": "#F5E8D3",
    "LightGreen": "#D6EDE2",
    "LightText": PALETTE["white"],
    "AxisBlack": PALETTE["black"],
    "MaskGray": PALETTE["light_gray"],
    "DNSblue": PALETTE["dark"],
    "DNSorange": PALETTE["orange"],
    "LESteal": PALETTE["teal"],
    "LESpurple": PALETTE["purple"],
    "LBMgray": REFERENCE_GRAY,
    "TheoryGray": REFERENCE_GRAY,
    "background": BACKGROUND_LIGHT,
    "background_light": BACKGROUND_LIGHT,
    "background_strong": BACKGROUND_STRONG,
    "decor_light": BACKGROUND_LIGHT,
    "reference": REFERENCE_GRAY,
    "reference_fill": BACKGROUND_LIGHT,
    # Semantic aliases used by existing tutorials.
    "vpm": PALETTE["purple"],
    "hybrid": PALETTE["orange"],
    "fvm": PALETTE["text"],
    "of": PALETTE["green"],
    "ref": REFERENCE_GRAY,
    "literature": REFERENCE_GRAY,
    "dvh": PALETTE["green"],
    "dvhr": PALETTE["teal"],
    "dns": PALETTE["dark"],
}

COLORMAPS = {
    "field_speed": "viridis",
    "field_vorticity": "magma",
    "vorticity_magnitude": "hot",
    "velocity": "Spectral_r",
    "vorticity": "RdBu_r",
    "error": "inferno",
    "error_diverging": "seismic",
    "vortex_speed": "plasma",
    "vortex_vorticity": "inferno",
}

# Vortex-interaction ladder. The two interaction families are plotted in
# separate panels, so each method keeps the same style in both panels.
VORTEX_INTERACTION_VARIANT_STYLE = {
    "dns": {"label": "DNS", "color": COLORS["RefGray"], "marker": "o"},
    "les": {"label": "LES", "color": COLORS["TUDcyan"], "marker": "s"},
    "les_stabilized": {
        "label": "LES + stabilization",
        "color": COLORS["VPMpurple"],
        "marker": "D",
    },
}
INTENDED_CASE_ORDER = {
    f"{family}_{variant}": order
    for order, (family, variant) in enumerate(
        (family, variant)
        for family in ("leapfrog", "collide")
        for variant in VORTEX_INTERACTION_VARIANT_STYLE
    )
}

VORTEX_RING_VARIANT_STYLE = {
    "dns_direct": {"color": COLORS["DNSblue"], "marker": "o", "linestyle": "--"},
    "dns_transposed": {"color": COLORS["VPMpurple"], "marker": "s", "linestyle": "--"},
    "dns_mixed": {"color": PALETTE["orange"], "marker": "^", "linestyle": "--"},
    "les_transposed": {"color": COLORS["TUDcyan"], "marker": "v", "linestyle": "-"},
    # Schema-2 compatibility while users replace the earlier two-case campaign.
    "dns_treecode": {"color": COLORS["VPMpurple"], "marker": "s", "linestyle": "--"},
    "les_treecode": {"color": COLORS["TUDcyan"], "marker": "v", "linestyle": "-"},
}
for _style in VORTEX_RING_VARIANT_STYLE.values():
    _style["linewidth"] = LINE_WIDTH
    _style["markersize"] = MARKER_SIZE
    _style["markeredgewidth"] = MARKER_EDGE_WIDTH
VORTEX_RING_VARIANT_LABEL = {
    "dns_direct": "DNS Direct",
    "dns_transposed": "DNS Transposed",
    "dns_mixed": "DNS Mixed",
    "les_transposed": "LES Transposed",
    "dns_treecode": "DNS Transposed",
    "les_treecode": "LES Transposed",
}

LAMB_OSEEN_SCHEME_STYLE = {
    "cs": {"label": "CS", "color": COLORS["FVMorange"], "marker": "o"},
    "rwm": {"label": "RWM", "color": COLORS["RefGray"], "marker": "^"},
    "dvh": {"label": "DVH", "color": COLORS["TUDcyan"], "marker": "v"},
    "gbd": {"label": "GBD", "color": COLORS["VPMpurple"], "marker": "D"},
}

ROTOR_STYLE = {
    "vpm": {
        "color": COLORS["vpm"],
        "marker": "o",
        "markersize": 1.5,
        "linewidth": 1.0,
        "label": "VLM-VPM",
    },
    "bem": {"color": COLORS["reference"], "linestyle": "--", "linewidth": 1.0, "label": "BEM"},
    "theory": {"color": COLORS["reference"], "linestyle": "--", "linewidth": 1.0},
    "reference": {"color": COLORS["reference"], "linestyle": "--", "linewidth": 1.0},
    "thrust_coefficient": {
        "color": COLORS["vpm"],
        "marker": "o",
        "markersize": 1.5,
        "linewidth": 1.0,
        "label": r"$C_T$",
    },
    "power_coefficient": {
        "color": COLORS["vpm"],
        "marker": "s",
        "markersize": 1.5,
        "linewidth": 1.0,
        "label": r"$C_P$",
    },
    "plane_0": {"color": COLORS["vpm"], "linewidth": 1.0},
    "plane_1": {"color": COLORS["vpm"], "linewidth": 1.0},
    "plane_2": {"color": COLORS["vpm"], "linewidth": 1.0},
    "plane_3": {"color": COLORS["vpm"], "linewidth": 1.0},
    "plane_4": {"color": COLORS["vpm"], "linewidth": 1.0},
}

REFERENCE_STYLE = {
    "color": COLORS["reference"],
    "linestyle": "--",
    "linewidth": REFERENCE_LINE_WIDTH,
}
REFERENCE_FILL_STYLE = {
    "facecolor": COLORS["reference_fill"],
    "alpha": 0.25,
    "zorder": 0,
}
REFERENCE_STRONG_FILL_STYLE = {
    "facecolor": COLORS["reference_fill"],
    "alpha": 0.50,
    "zorder": 0,
}


def get_color(name: str, fallback: str | None = None) -> str:
    """Return a named color from the central tutorial palette."""
    if fallback is None:
        fallback = COLORS["reference"]
    return COLORS.get(name, fallback)


def get_colormap(name: str) -> str:
    """Return a named colormap from the central tutorial palette."""
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


def thesis_y_label_margin(fig, axes: Iterable[Axes] | Axes) -> float:
    """Measure a symmetric plot margin with outer y text just inside the canvas.

    Measure rendered labels at their final font size. The returned fraction
    is used for both left and right margins, not for asymmetric tight cropping.
    """
    axes = (axes,) if isinstance(axes, Axes) else tuple(axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
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
        outer = thesis_y_label_margin(fig, axes)
        centered_subplots_adjust(fig, outer=outer)


def validate_thesis_figure(fig, axes: Iterable[Axes] | Axes) -> None:
    """Validate the fixed-size, centred, single-font thesis plot contract."""
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

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    minimum_padding = MIN_TEXT_CANVAS_PADDING_PT * fig.dpi / 72.0
    in_range_tick_label: dict[Text, bool] = {}
    for axis in fig.axes:
        for axis_object, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
            lower, upper = sorted(limits)
            tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
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


def case_style(name: str) -> dict:
    """Return the shared style for a vortex-interaction case name."""
    _, _, variant = name.partition("_")
    style = VORTEX_INTERACTION_VARIANT_STYLE.get(
        variant,
        {"label": variant.replace("_", " ").title(), "color": COLORS["reference"], "marker": "o"},
    )
    return {
        "color": style["color"],
        "linestyle": "-",
        "linewidth": LINE_WIDTH,
        "marker": style["marker"],
        "markersize": MARKER_SIZE,
        "markeredgewidth": MARKER_EDGE_WIDTH,
        "label": style["label"],
    }


def legend_handle_style(style: dict) -> dict:
    """Return line style values sized for legend-only handles."""
    return {
        "color": style["color"],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "markersize": LEGEND_MARKER_SIZE,
        "linewidth": style["linewidth"],
        "label": style["label"],
    }


def set_style(*, use_tex: bool = False):
    """Apply the OpenONDA publication plotting style.

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
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
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
        "axes.linewidth": 0.5,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
    }

    plt.rcParams.update(tex_fonts)


def save_fig(
    fig,
    path,
    figure_format: str | None = None,
    dpi: int | None = None,
    tight_rect: tuple[float, float, float, float] | None = None,
    bbox_inches: str | None = None,
) -> None:
    """Save a Matplotlib figure with the shared export defaults."""
    out = Path(path)
    if figure_format is not None:
        out = figure_path(out, figure_format)
    out.parent.mkdir(parents=True, exist_ok=True)
    layout_engine = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if layout_engine is None:
        if tight_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_rect)
    fig.savefig(out, dpi=DEFAULT_DPI if dpi is None else dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    from source import log_style

    print(log_style.block_section("figure output", [("saved", str(out))]))


def set_thesis_style():
    """Use the thesis body fonts at final physical size; never silently substitute."""
    if not all(shutil.which(tool) for tool in ("latex", "dvipng")):
        raise RuntimeError("Thesis plots require LaTeX, dvipng and newpxtext/newpxmath on PATH.")
    set_style()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{newpxtext}\usepackage{amsmath}\usepackage{newpxmath}",
            "font.family": "serif",
            "font.serif": ["Palatino"],
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
