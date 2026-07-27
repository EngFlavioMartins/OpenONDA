# =================================================
# Standard library imports
# =================================================
import os
from pathlib import Path

import defusedxml.ElementTree as ET  # noqa: N817

# =================================================
# Third-party library imports
# =================================================
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# OpenONDA project-specific imports moved to functions
# =================================================


# ==================================================


def compute_ring_trajectory(pvd_path: str, extra_stride: int = 0):
    """
    Compute vortex ring axial location and speed using the PVD time-series file.

    Parameters
    ----------
    pvd_path : str
        Path to the .pvd file indexing the VTP time series.
    extra_stride : int
        Optional extra stride to skip frames (default = 0).

    Returns
    -------
    ring_location : ndarray
        X-position of the ring centroid at each sampled time.
    ring_speed : ndarray
        Axial speed of the ring estimated from finite differences.
    ring_time : ndarray
        Time stamps corresponding to each sampled position.
    """
    # Parse the PVD file
    from OpenONDA.solvers.VPM import vpmModule as vpm  # noqa: N813

    tree = ET.parse(pvd_path)
    root = tree.getroot()
    data_set = root.find("Collection")

    # Extract file paths and time values
    entries = data_set.findall("DataSet")
    entries = entries[:: 1 + extra_stride]  # apply stride

    ring_location = []
    ring_time = []

    for entry in entries:
        vtp_path = Path(pvd_path).parent / entry.attrib["file"]
        time = float(entry.attrib["timestep"])

        psys = vpm.ParticleSystem(processing_unit="CPU")
        psys.load_particle_field_from_backup(str(vtp_path))

        centroid = psys.get_centroids_of_circulation()[0]
        ring_location.append(centroid[0])
        ring_time.append(time)

    ring_location = np.array(ring_location)
    ring_time = np.array(ring_time)
    ring_speed = np.gradient(ring_location, ring_time)

    return ring_location, ring_speed, ring_time


# ==================================================


def theoretical_ring_trajectory(
    kinematic_viscosity: float,
    ring_thickness_0: float,
    ring_radius_0: float,
    ring_strength_0: float,
    time: np.ndarray,
):
    """
    Computes the theoretical trajectory and speed of a vortex ring over time.

    Arguments:
    ----------
    kinematic_viscosity : float
        The kinematic viscosity of the fluid.
    ring_thickness_0 : float
        The initial core thickness of the vortex ring.
    ring_radius_0 : float
        The initial radius of the vortex ring.
    ring_strength_0 : float
        The initial strength (circulation) of the vortex ring.
    time : np.ndarray
        Array of time points at which to compute the trajectory.

    Returns:
    --------
    ring_location_theo : np.ndarray
        Theoretical cumulative distance traveled by the ring over time.
    ring_speed_theo : np.ndarray
        Theoretical speed of the ring at each time point.
    """
    # Calculate ring thickness over time
    ring_thickness = np.sqrt(4 * kinematic_viscosity * time + ring_thickness_0**2)

    # Core thickness ratio
    eps = ring_thickness / ring_radius_0

    # Empirical correction factor for finite core thickness
    C = -0.558 - 1.12 * eps**2 - 5.0 * eps**4

    # Term A based on initial circulation and radius
    term_a = ring_strength_0 / (4 * np.pi * ring_radius_0)

    # Term B includes the logarithmic factor and correction coefficient
    term_b = np.log(8 / eps) + C

    # Compute theoretical ring speed
    ring_speed_theo = term_a * term_b

    # Calculate time step from the time array
    time_step = np.gradient(time)

    # Compute theoretical ring location by cumulative sum of speed * time_step
    ring_location_theo = np.cumsum(ring_speed_theo * time_step)

    return ring_location_theo, ring_speed_theo


def get_particles_diagnostics(
    backup_file_name: str, pvd_path: str, extra_stride: int, processing_unit: str = "GPU"
) -> dict[str, np.ndarray]:
    """
    Loads diagnostics from backup if available; otherwise computes and saves them.

    Returns:
        A dictionary with the following keys:
        - "total_strength"
        - "total_enstrophy"
        - "linear_impulse_magnitude"
        - "total_kinetic_energy"
        - "total_vorticity_magnitude"
        - "total_helicity"
        - "simulation_time"
    """

    if os.path.exists(backup_file_name):
        print(f"Loading diagnostics from backup: {backup_file_name}")
        backup = np.load(backup_file_name)
        data = {key: backup[key] for key in backup.files}

    else:
        print(f"No backup found. Computing diagnostics and saving to: {backup_file_name}")
        (
            total_strength,
            total_enstrophy,
            linear_impulse_magnitude,
            total_kinetic_energy,
            total_vorticity_magnitude,
            total_helicity,
            simulation_time,
        ) = compute_particle_field_diagnostics(pvd_path, extra_stride, processing_unit)

        np.savez_compressed(
            backup_file_name,
            total_strength=total_strength,
            total_enstrophy=total_enstrophy,
            linear_impulse_magnitude=linear_impulse_magnitude,
            total_kinetic_energy=total_kinetic_energy,
            total_vorticity_magnitude=total_vorticity_magnitude,
            total_helicity=total_helicity,
            simulation_time=simulation_time,
        )

        data = {
            "total_strength": total_strength,
            "total_enstrophy": total_enstrophy,
            "linear_impulse_magnitude": linear_impulse_magnitude,
            "total_kinetic_energy": total_kinetic_energy,
            "total_vorticity_magnitude": total_vorticity_magnitude,
            "total_helicity": total_helicity,
            "simulation_time": simulation_time,
        }

    return data


def compute_particle_field_diagnostics(
    pvd_path: str, diagnostic_stride: int, processing_unit: str = "GPU"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute diagnostics using a PVD time-series file for consistent and efficient access.

    Parameters
    ----------
    pvd_path : str
        Path to the .pvd file indexing the VTP time series.
    diagnostic_stride : int
        Subsampling interval for diagnostics (e.g., 1 = all, 2 = every second).
    processing_unit : str
        Processing backend: 'CPU' or 'GPU'.

    Returns
    -------
    Tuple of diagnostic arrays (same as original function)
    """
    from OpenONDA.solvers.VPM import vpmModule as vpm  # noqa: N813

    tree = ET.parse(pvd_path)
    root = tree.getroot()
    collection = root.find("Collection")
    entries = collection.findall("DataSet")
    entries = entries[::diagnostic_stride]

    num_snapshots = len(entries)

    # Preallocate
    total_strength = np.zeros(num_snapshots)
    total_enstrophy = np.zeros(num_snapshots)
    linear_impulse_magnitude = np.zeros(num_snapshots)
    total_kinetic_energy = np.zeros(num_snapshots)
    total_vorticity_magnitude = np.zeros(num_snapshots)
    total_helicity = np.zeros(num_snapshots)
    simulation_time = np.zeros(num_snapshots)

    for i, entry in enumerate(entries):
        vtp_path = Path(pvd_path).parent / entry.attrib["file"]
        time = float(entry.attrib["timestep"])

        print(f"[{i + 1}/{num_snapshots}] Processing {vtp_path.name} at t = {time:.4f}")
        psys = vpm.ParticleSystem(processing_unit=processing_unit)
        psys.load_particle_field_from_backup(str(vtp_path))

        total_strength[i] = psys.get_total_strength_magnitude()
        total_enstrophy[i] = psys.get_total_enstrophy()
        linear_impulse_magnitude[i] = np.linalg.norm(psys.get_total_linear_impulse())
        total_kinetic_energy[i] = psys.get_total_kinetic_energy()
        total_vorticity_magnitude[i] = np.sum(
            np.linalg.norm(psys.get_particles_vorticities(), axis=1)
        )
        total_helicity[i] = psys.get_total_helicity()
        simulation_time[i] = time

    return (
        total_strength,
        total_enstrophy,
        linear_impulse_magnitude,
        total_kinetic_energy,
        total_vorticity_magnitude,
        total_helicity,
        simulation_time,
    )


def get_leapfrogging_rings_shapes(
    pvd_path: str, backup_file_name: str, extra_stride: int
) -> dict[str, np.ndarray]:
    """
    Computes the centroid positions and radii of leapfrogging vortex rings
    using a PVD time-series file. Caches results to disk.

    Parameters
    ----------
    pvd_path : str
        Path to the .pvd file listing the VTP snapshots.
    backup_file_name : str
        Path to the .npz file where results will be cached.
    extra_stride : int
        Interval at which to sample time snapshots.

    Returns
    -------
    Dictionary with:
        - 'rings_positions': (T, 2, 3) array of [x, y, z] centroids
        - 'rings_radii': (T, 2) array of radii
        - 'simulation_time': (T,) array of simulation times
    """
    backup_dir = os.path.dirname(backup_file_name)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)

    if os.path.exists(backup_file_name):
        print(f"Loading rings shapes from backup: {backup_file_name}")
        return dict(np.load(backup_file_name))

    print("No backup found. Computing shapes across timesteps...")

    # Parse the PVD file
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    entries = root.find("Collection").findall("DataSet")
    entries = entries[::extra_stride]
    T = len(entries)

    # Initialize first frame to determine number of rings
    from OpenONDA.solvers.VPM import vpmModule as vpm  # noqa: N813

    first_vtp = Path(pvd_path).parent / entries[0].attrib["file"]
    psys = vpm.ParticleSystem(processing_unit="CPU")
    psys.load_particle_field_from_backup(str(first_vtp), remove_current_particles=True)
    num_rings = len(np.unique([p.group_id for p in psys.particles]))

    # Preallocate
    rings_positions = np.zeros((T, num_rings, 3), dtype=np.float64)
    rings_radii = np.zeros((T, num_rings), dtype=np.float64)
    simulation_time = np.zeros(T, dtype=np.float64)

    for i, entry in enumerate(entries):
        vtp_path = Path(pvd_path).parent / entry.attrib["file"]
        time = float(entry.attrib["timestep"])
        print(f"[{i + 1}/{T}] Processing {vtp_path.name} at t = {time:.4f}")

        psys = vpm.ParticleSystem(processing_unit="CPU")
        psys.load_particle_field_from_backup(str(vtp_path), remove_current_particles=True)

        centroids = psys.get_centroids_of_circulation()
        rings_positions[i] = centroids

        positions = psys.get_particles_positions()
        strengths = (
            psys.get_particles_strengths()
        )  # Assuming this is a 1D array of scalar magnitudes or (N,3) array of vectors
        group_ids = np.array([p.group_id for p in psys.particles], dtype=np.int64)

        for j, group_id in enumerate(np.unique(group_ids)):
            mask = group_ids == group_id
            pos_group = positions[mask]
            str_group = strengths[mask]  # Get strengths for this group

            # Ensure strengths are magnitudes if they are vectors
            if (
                str_group.ndim > 1 and str_group.shape[1] > 1
            ):  # Check if it's a vector array (e.g., (N,3))
                str_group_magnitudes = np.linalg.norm(str_group, axis=1)
            else:  # Assume it's already a 1D array of scalar magnitudes
                str_group_magnitudes = np.abs(
                    str_group
                )  # Use abs for safety if strengths can be negative

            centroid = centroids[j]
            deltas = pos_group - centroid

            squared_radial_dists = np.sum(
                deltas**2, axis=1
            )  # Keep as is for now for simplicity, assuming centroid works well.

            # Compute weighted mean square distance
            weighted_squared_dists_sum = np.sum(str_group_magnitudes * squared_radial_dists)
            sum_of_weights = np.sum(str_group_magnitudes)

            if (
                sum_of_weights > 0
            ):  # Avoid division by zero if all strengths are zero (unlikely but good practice)
                rings_radii[i, j] = np.sqrt(weighted_squared_dists_sum / sum_of_weights)
            else:
                rings_radii[i, j] = (
                    0.0  # Or np.nan, depending on how you want to handle empty/zero-strength rings
                )

        simulation_time[i] = time

    np.savez_compressed(
        backup_file_name,
        rings_positions=rings_positions,
        rings_radii=rings_radii,
        simulation_time=simulation_time,
    )

    return {
        "rings_positions": rings_positions,
        "rings_radii": rings_radii,
        "simulation_time": simulation_time,
    }


def get_leapfrogging_rings_shapes2(
    pvd_path: str, backup_file_name: str, extra_stride: int
) -> dict[str, np.ndarray]:

    backup_dir = os.path.dirname(backup_file_name)
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)

    if os.path.exists(backup_file_name):
        print(f"Loading rings shapes from backup: {backup_file_name}")
        return dict(np.load(backup_file_name))

    print("No backup found. Computing shapes across timesteps...")

    tree = ET.parse(pvd_path)
    root = tree.getroot()
    entries = root.find("Collection").findall("DataSet")
    entries = entries[::extra_stride]
    T = len(entries)

    # Initialize
    from OpenONDA.solvers.VPM import vpmModule as vpm  # noqa: N813

    first_vtp = Path(pvd_path).parent / entries[0].attrib["file"]
    psys = vpm.ParticleSystem(processing_unit="CPU")
    psys.load_particle_field_from_backup(str(first_vtp), remove_current_particles=True)
    num_rings = len(np.unique([p.group_id for p in psys.particles]))

    rings_positions = np.zeros((T, num_rings, 3), dtype=np.float64)
    rings_radii = np.zeros((T, num_rings), dtype=np.float64)
    simulation_time = np.zeros(T, dtype=np.float64)

    for i, entry in enumerate(entries):
        vtp_path = Path(pvd_path).parent / entry.attrib["file"]
        time = float(entry.attrib["timestep"])
        print(f"[{i + 1}/{T}] Processing {vtp_path.name} at t = {time:.4f}")

        psys = vpm.ParticleSystem(processing_unit="CPU")
        psys.load_particle_field_from_backup(str(vtp_path), remove_current_particles=True)

        centroids = psys.get_centroids_of_circulation()
        rings_positions[i] = centroids

        positions = psys.get_particles_positions()
        strengths = psys.get_particles_strengths()
        group_ids = np.array([p.group_id for p in psys.particles], dtype=np.int64)

        for j, group_id in enumerate(np.unique(group_ids)):
            mask = group_ids == group_id
            pos_group = positions[mask]
            str_group = strengths[mask]

            if str_group.ndim > 1 and str_group.shape[1] > 1:
                str_group_magnitudes = np.linalg.norm(str_group, axis=1)
            else:
                str_group_magnitudes = np.abs(str_group)

            centroid = centroids[j]
            deltas = pos_group - centroid
            squared_radial_dists = np.sum(deltas**2, axis=1)

            weights = str_group_magnitudes
            total_weight = np.sum(weights)
            if total_weight == 0:
                rings_radii[i, j] = 0.0
                continue

            # Step 1: Initial radius estimate
            initial_radius = np.sqrt(np.sum(weights * squared_radial_dists) / total_weight)

            # Step 2: Strength threshold (keep top 50%)
            strength_threshold = np.percentile(str_group_magnitudes, 50)
            strong_mask = str_group_magnitudes >= strength_threshold

            # Step 3: Spatial mask (within 1.5 * initial_radius)
            spatial_mask = squared_radial_dists < (1.5 * initial_radius) ** 2

            # Combined mask
            refined_mask = strong_mask & spatial_mask
            if not np.any(refined_mask):
                rings_radii[i, j] = initial_radius
                continue

            pos_core = pos_group[refined_mask]
            str_core = str_group_magnitudes[refined_mask]
            deltas_core = pos_core - centroid
            squared_core_dists = np.sum(deltas_core**2, axis=1)

            refined_weight_sum = np.sum(str_core)
            if refined_weight_sum > 0:
                refined_radius = np.sqrt(np.sum(str_core * squared_core_dists) / refined_weight_sum)
                rings_radii[i, j] = refined_radius
            else:
                rings_radii[i, j] = initial_radius

        simulation_time[i] = time

    np.savez_compressed(
        backup_file_name,
        rings_positions=rings_positions,
        rings_radii=rings_radii,
        simulation_time=simulation_time,
    )

    return {
        "rings_positions": rings_positions,
        "rings_radii": rings_radii,
        "simulation_time": simulation_time,
    }


# ==================================================

# -- Plot style ---------------------------------------------------------------
# All tutorial plot presentation lives here: palette, font sizes, figure sizes,
# markers, line widths, reference styles, and export defaults.
CM = 1 / 2.54
FONT_SIZE_PT = 10
DEFAULT_DPI = 400
EXPORT_FORMATS = ("png", "pdf")
MAX_FIGURE_WIDTH_CM = 12.8
WIDE_FIGURE_WIDTH_CM = 17.2
FONT_PATH = Path(__file__).with_name("DejaVuSerif.ttf")

FIGURE_SIZES_CM = {
    "single": (MAX_FIGURE_WIDTH_CM, 7.0),
    "single_tall": (MAX_FIGURE_WIDTH_CM, 8.0),
    "trajectory": (MAX_FIGURE_WIDTH_CM, 7.5),
    "stacked": (MAX_FIGURE_WIDTH_CM, 11.0),
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
    "energy": 4,
    "trajectory": 5,
}

PALETTE = {
    "dark": "#0C2340",
    "teal": "#0E8A85",
    "purple": "#5C3D9B",
    "orange": "#C76D24",
    "green": "#2B7A4E",
    "red": "#9C2F50",
    "gray": "#6E8898",
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
    # Vortex-interaction baseline/LES/stabilized-LES comparison.
    "case_baseline": PALETTE["strong_gray"],
    "case_les": PALETTE["dark"],
    "case_les_stabilized": PALETTE["black"],
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

FAMILY_LINESTYLE = {"leapfrog": "-", "collide": "--"}
FAMILY_LABEL = {"leapfrog": "Leapfrog", "collide": "Collision"}
VARIANT_LABEL = {
    "baseline": "Baseline (DNS)",
    "les": "LES",
    "les_stabilized": "LES + stabilized",
}
VARIANT_ORDER = tuple(VARIANT_LABEL)
VARIANT_STYLE = {
    "baseline": {"color": COLORS["case_baseline"], "marker": "8"},
    "les": {"color": COLORS["case_les"], "marker": "s"},
    "les_stabilized": {"color": COLORS["case_les_stabilized"], "marker": "o"},
}
INTENDED_CASE_ORDER = {
    f"{family}_{variant}": family_i * len(VARIANT_ORDER) + variant_i
    for family_i, family in enumerate(("leapfrog", "collide"))
    for variant_i, variant in enumerate(VARIANT_ORDER)
}

VORTEX_RING_VARIANT_STYLE = {
    "DNS_direct": {"color": COLORS["DNSblue"], "marker": "o", "linestyle": "--"},
    "DNS_transposed": {"color": COLORS["VPMpurple"], "marker": "s", "linestyle": "--"},
    "DNS_mixed": {"color": PALETTE["orange"], "marker": "^", "linestyle": "--"},
    "LES_direct": {"color": PALETTE["dark"], "marker": "D", "linestyle": "-"},
    "LES_transposed": {"color": COLORS["TUDcyan"], "marker": "v", "linestyle": "-"},
    "LES_mixed": {"color": PALETTE["gray"], "marker": "p", "linestyle": "-"},
}
for _style in VORTEX_RING_VARIANT_STYLE.values():
    _style["linewidth"] = LINE_WIDTH
    _style["markersize"] = MARKER_SIZE
    _style["markeredgewidth"] = MARKER_EDGE_WIDTH
VORTEX_RING_VARIANT_LABEL = {
    **{name: name.replace("_", " ") for name in VORTEX_RING_VARIANT_STYLE},
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
    "ct": {
        "color": COLORS["vpm"],
        "marker": "o",
        "markersize": 1.5,
        "linewidth": 1.0,
        "label": r"$C_T$",
    },
    "cp": {
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


def case_style(name: str, include_family: bool = True) -> dict:
    """Return the shared style for a vortex-interaction case name."""
    family, _, variant = name.partition("_")
    variant = variant or "les"
    variant_style = VARIANT_STYLE.get(variant, {"color": COLORS["reference"], "marker": "o"})
    variant_label = VARIANT_LABEL.get(variant, variant.replace("_", " ").title())
    label = variant_label
    if include_family:
        label = f"{FAMILY_LABEL.get(family, family.title())} - {variant_label}"
    return {
        "color": variant_style["color"],
        "linestyle": FAMILY_LINESTYLE.get(family, "-"),
        "linewidth": LINE_WIDTH,
        "marker": variant_style["marker"],
        "markersize": MARKER_SIZE,
        "markeredgewidth": MARKER_EDGE_WIDTH,
        "label": label,
        "family": family,
        "variant": variant,
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
    fig.savefig(out, dpi=DEFAULT_DPI if dpi is None else dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
