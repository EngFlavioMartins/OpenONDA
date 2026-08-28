# =================================================
# Standard library imports
# =================================================
from pathlib import Path

# =================================================
# Third-party library imports
# =================================================
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# OpenONDA project-specific imports moved to functions
# =================================================


# ==================================================


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
FONT_SIZE_PT = 10
DEFAULT_DPI = 400
EXPORT_FORMATS = ("png", "pdf")
MAX_FIGURE_WIDTH_CM = 12.5
WIDE_FIGURE_WIDTH_CM = 12.5
FONT_PATH = Path(__file__).with_name("DejaVuSerif.ttf")

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

PALETTE = {
    "dark": "#0C2340",
    "teal": "#0E8A85",
    "purple": "#5C3D9B",
    "orange": "#C76D24",
    "green": "#2B7A4E",
    "red": "#9C2F50",
    "gray": "#5A6972",
    "light_gray": "#C0C0C0",
    "strong_gray": "#8B8B8B",
    "white": "#ffffff",
    "black": "#000000",
}
COLOR_CYCLE = (
    PALETTE["dark"],
    PALETTE["purple"],
    PALETTE["orange"],
    PALETTE["teal"],
    PALETTE["green"],
    PALETTE["gray"],
)
BACKGROUND_LIGHT = PALETTE["light_gray"]
BACKGROUND_STRONG = PALETTE["strong_gray"]
REFERENCE_GRAY = PALETTE["gray"]

COLORS = {
    # Semantic aliases over the 10-color PALETTE above.
    "TUDdark": PALETTE["dark"],
    "TUDcyan": PALETTE["teal"],
    "TUDred": PALETTE["orange"],
    "VPMpurple": PALETTE["purple"],
    "FVMorange": PALETTE["orange"],
    "AccentGreen": PALETTE["green"],
    "AccentRed": PALETTE["teal"],
    "BackgroundLight": BACKGROUND_LIGHT,
    "BackgroundGray": BACKGROUND_STRONG,
    "ReferenceGray": REFERENCE_GRAY,
    "RefGray": REFERENCE_GRAY,
    "DarkText": PALETTE["dark"],
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
    "hybrid": PALETTE["teal"],
    "fvm": PALETTE["orange"],
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
    "les_direct": {"color": PALETTE["dark"], "marker": "D", "linestyle": "-"},
    "les_transposed": {"color": COLORS["TUDcyan"], "marker": "v", "linestyle": "-"},
    "les_mixed": {"color": PALETTE["gray"], "marker": "p", "linestyle": "-"},
}
for _style in VORTEX_RING_VARIANT_STYLE.values():
    _style["linewidth"] = LINE_WIDTH
    _style["markersize"] = MARKER_SIZE
    _style["markeredgewidth"] = MARKER_EDGE_WIDTH
VORTEX_RING_VARIANT_LABEL = {
    "dns_direct": "DNS Direct",
    "dns_transposed": "DNS Transposed",
    "dns_mixed": "DNS Mixed",
    "les_direct": "LES Direct",
    "les_transposed": "LES Transposed",
    "les_mixed": "LES Mixed",
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


def set_style():
    """Apply the OpenONDA publication plotting style."""
    if FONT_PATH.exists():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(FONT_PATH))

    tex_fonts = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "figure.titlesize": FONT_SIZE_PT,
        "legend.fontsize": FONT_SIZE_PT,
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
    }

    plt.rcParams.update(tex_fonts)


def save_fig(
    fig,
    path,
    figure_format: str | None = None,
    dpi: int | None = None,
    tight_rect: tuple[float, float, float, float] | None = None,
    bbox_inches: str | None = "tight",
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
    print(f"  Saved: {out}")
