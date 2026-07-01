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

# -- Colour palette -- aligned with thesis/LaTeX (thesis_visuals/styles/colors.py)
# Canonical names match the LaTeX \definecolor entries in thesis.tex exactly.
COLORS = {
    # -- TU Delft canonical names -------------------------------------------------
    "TUDdark":     "#0C2340",   # Deep navy
    "TUDcyan":     "#0E8A85",   # Aquamarine teal
    "TUDred":      "#C8102E",   # TU Delft red (also used as TUD accent)
    "VPMpurple":   "#5C3D9B",   # Amethyst purple
    "FVMorange":   "#772953",   # Aubergine / deep rose
    "AccentGreen": "#2B7A4E",   # Forest emerald
    "AccentRed":   "#9C2F50",   # Deep rose
    "RefGray":     "#6E8898",   # Steel blue-gray
    "DarkText":    "#2E3D46",   # Dark blue-gray
    "DNSblue":     "#1A6B9A",   # DNS / high-fidelity blue
    "DNSorange":   "#B85C2A",   # DNS mixed variant orange
    "LESteal":     "#1A8C88",   # LES transposed teal
    "LESpurple":   "#7B2969",   # LES rVPM purple
    "LBMgray":     "#505050",   # LBM reference gray
    "TheoryGray":  "#808080",   # Standard gray for theory/reference curves
    # -- Semantic aliases (backward-compatible keys for existing plot scripts) -----
    "vpm":         "#5C3D9B",   # -> VPMpurple  (VPM-only results)
    "hybrid":      "#772953",   # -> FVMorange  (hybrid solver results)
    "fvm":         "#2E3D46",   # -> DarkText   (pure FVM results)
    "of":          "#6E8898",   # -> RefGray    (OpenFOAM / reference)
    "ref":         "#6E8898",   # -> RefGray    (literature / reference data)
    "literature":  "#2E3D46",   # -> DarkText   (other literature)
    "dvh":         "#2B7A4E",   # -> AccentGreen (DVH scheme)
    "dvhr":        "#0E8A85",   # -> TUDcyan    (DVH-R scheme)
    "dns":         "#1A6B9A",   # -> DNSblue    (direct numerical simulation)
    "les":         "#0E8A85",   # -> TUDcyan    (large-eddy simulation)
}


def set_style():
    """Apply publication-quality matplotlib style settings."""
    # Define the style dictionary
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.titlesize": 10,
        # Ticks settings
        "xtick.major.size": 6,  # Major ticks length
        "ytick.major.size": 6,  # Major ticks length
        "xtick.minor.size": 4,  # Minor ticks length
        "ytick.minor.size": 4,  # Minor ticks length
        "xtick.major.width": 0.5,  # Major ticks width
        "ytick.major.width": 0.5,  # Major ticks width
        "xtick.minor.width": 0.5,  # Minor ticks width
        "ytick.minor.width": 0.5,  # Minor ticks width
        # Grid settings
        "axes.grid": False,  # Enable grid
        "grid.linestyle": "--",  # Style for grid lines
        "grid.linewidth": 0.5,  # Line width for grid
        "grid.alpha": 0.3,  # Make the grid lines semi-transparent
        # Minor gridlines always visible
        "axes.grid.which": "both",  # Show both major and minor gridlines
        "xtick.minor.visible": True,  # Make minor ticks visible
        "ytick.minor.visible": True,  # Make minor ticks visible
        # Enable ticks on all four sides
        "xtick.top": True,
        "ytick.right": True,
        "axes.edgecolor": "black",  # Set color of axis box (edges of the plot area)
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }

    # Apply the style
    plt.rcParams.update(tex_fonts)
