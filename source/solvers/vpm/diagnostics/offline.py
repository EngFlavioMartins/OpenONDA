"""
Offline post-processing diagnostics: flow integrals (energy, total_enstrophy,
vortex strength, impulse) computed from saved particle states.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import taichi as ti

from ..config.constants import EPSILON

# Use VPM logger
from ..io.logging import Logging
from ..physics.evaluation import ParticleFieldEvaluation

# =========================================================
# Taichi Kernel Functions (copied from physics module for standalone operation)
# =========================================================

# Constants for Abramowitz and Stegun approximation of erf
# =========================================================
# Helper Classes
# =========================================================


@dataclass
class FlowIntegrals:
    """Container for flow integral quantities at a single timestep."""

    time: float
    total_kinetic_energy: float
    total_helicity: float
    total_enstrophy: float
    kinetic_energy_rate: float
    kinetic_energy_rate_source: str
    viscous_kinetic_energy_rate: float
    vortex_strength_magnitude_sum: float
    net_vortex_strength: np.ndarray  # (3,) [m³/s]
    linear_impulse: np.ndarray  # (3,)
    angular_impulse: np.ndarray  # (3,)
    n_particles_total: int


class ParticleContainerWrapper:
    """Minimal particle-field interface used by offline diagnostics."""

    def __init__(
        self,
        position,
        vortex_strength,
        core_radius,
        effective_viscosity,
        count,
    ):
        """Wrap saved particle arrays behind the diagnostics field interface.

        Parameters
        ----------
        position : array_like, shape (N, 3)
            Particle centres in metres.
        vortex_strength : array_like, shape (N, 3)
            Particle-strength vectors ``Gamma = omega * V`` in m³/s.
        core_radius : array_like, shape (N,)
            Particle core radii in metres.
        effective_viscosity : array_like, shape (N,)
            Effective kinematic viscosities in m²/s.
        count : int
            Active prefix length. The wrapper stores the supplied arrays and
            count without copying or validating them; callers must provide
            mutually compatible prefixes.

        Notes
        -----
        This is an adapter for offline kernels, not a particle owner. The
        arrays are treated as read-only by this class.
        """
        self.position = position
        self.vortex_strength = vortex_strength
        self.core_radius = core_radius
        self.effective_viscosity = effective_viscosity
        self._count = count

    def __len__(self):
        return self._count


class OfflineFlowDiagnostics:
    """
    Offline flow diagnostics processor for native VPM backup frames.

    Reads particle data directly from the canonical ``vpm_<step>.h5`` backup
    series in one solution directory and computes all integral flow quantities
    using GPU-accelerated Taichi kernels. No visualization intermediate is
    involved.

    Attributes:
        h5_directory: Directory containing the canonical ``vpm_*.h5`` frames.
        h5_files: List of HDF5 frame files, sorted by timestep.
        results: List of FlowIntegrals for each timestep.

    Example:
        >>> diagnostics = OfflineFlowDiagnostics('solution/vpm')
        >>> diagnostics.compute_all()
        >>> diagnostics.save()  # Saves to 'diagnostics.log'
        >>> diagnostics.save('custom_output.csv')
    """

    def __init__(self, h5_directory: str | Path, h5_glob: str = "vpm_*.h5"):
        """
        Initialize the offline diagnostics processor.

        Args:
            h5_directory: Directory containing the canonical HDF5 backup
                frames (``vpm_<step>.h5`` by default).
            h5_glob: Filename pattern used to select the frames below
                ``h5_directory``.

        Raises:
            FileNotFoundError: If the directory does not exist or contains no
                matching HDF5 frames.
        """
        self.base_dir = Path(h5_directory)
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"HDF5 backup directory not found: {h5_directory}")

        self.h5_files: list[Path] = sorted(self.base_dir.glob(h5_glob))
        if not self.h5_files:
            raise FileNotFoundError(f"No HDF5 backup files found below {self.base_dir}")

        self.results: list[FlowIntegrals] = []

        Logging.info(
            f"Offline diagnostics: Loaded {len(self.h5_files)} timesteps from {self.base_dir}"
        )

    def _load_particle_data(self, h5_path: Path) -> dict[str, Any]:
        """
        Load particle data from a single HDF5 backup file.

        Args:
            h5_path: Path to HDF5 file.

        Returns:
            Dictionary with particle arrays and metadata.
        """
        data: dict[str, Any] = {}
        with h5py.File(h5_path, "r") as f:
            solver = f["solver"]
            particles = f["particles"]

            data["time"] = float(solver.attrs["time"])
            data["n_particles_total"] = int(solver.attrs["n_particles_total"])

            n: int = data["n_particles_total"]
            if n > 0:
                data["position"] = particles["position"][:n]
                data["vortex_strength"] = particles["vortex_strength"][:n]
                data["core_radius"] = particles["core_radius"][:n]
                data["kinematic_viscosity"] = particles["kinematic_viscosity"][:n]
                data["eddy_viscosity"] = particles["eddy_viscosity"][:n]
                data["effective_viscosity"] = particles["effective_viscosity"][:n]
            else:
                data["position"] = np.empty((0, 3), dtype=np.float32)
                data["vortex_strength"] = np.empty((0, 3), dtype=np.float32)
                data["core_radius"] = np.empty((0,), dtype=np.float32)
                data["effective_viscosity"] = np.empty((0,), dtype=np.float32)

        return data

    def _compute_single_time_step(self, h5_path: Path) -> FlowIntegrals:
        """Compute flow integrals for a single timestep."""
        data = self._load_particle_data(h5_path)
        n = data["n_particles_total"]

        if n == 0:
            return FlowIntegrals(
                time=data["time"],
                total_kinetic_energy=0.0,
                total_helicity=0.0,
                total_enstrophy=0.0,
                kinetic_energy_rate=0.0,
                kinetic_energy_rate_source="empty_particle_field",
                viscous_kinetic_energy_rate=0.0,
                vortex_strength_magnitude_sum=0.0,
                net_vortex_strength=np.zeros(3),
                linear_impulse=np.zeros(3),
                angular_impulse=np.zeros(3),
                n_particles_total=0,
            )

        # Allocate fields for this step
        position_field = ti.Vector.field(3, ti.f32, shape=n)
        str_field = ti.Vector.field(3, ti.f32, shape=n)
        core_radius_field = ti.field(ti.f32, shape=n)
        visc_field = ti.field(ti.f32, shape=n)

        # Copy data to Taichi
        position_field.from_numpy(data["position"].astype(np.float32))
        str_field.from_numpy(data["vortex_strength"].astype(np.float32))
        core_radius_field.from_numpy(data["core_radius"].astype(np.float32))
        visc_field.from_numpy(data["effective_viscosity"].astype(np.float32))

        # Create wrapper mimicking Particles class
        particles_wrapper = ParticleContainerWrapper(
            position=position_field,
            vortex_strength=str_field,
            core_radius=core_radius_field,
            effective_viscosity=visc_field,
            count=n,
        )

        results_dict = self.evaluator.compute_flow_integrals(particles_wrapper, data["time"])

        return FlowIntegrals(
            time=data["time"],
            total_kinetic_energy=results_dict["total_kinetic_energy"],
            total_helicity=results_dict["total_helicity"],
            total_enstrophy=results_dict["total_enstrophy"],
            kinetic_energy_rate=results_dict["kinetic_energy_rate"],
            kinetic_energy_rate_source=results_dict["kinetic_energy_rate_source"],
            viscous_kinetic_energy_rate=results_dict["viscous_kinetic_energy_rate"],
            vortex_strength_magnitude_sum=results_dict["vortex_strength_magnitude_sum"],
            net_vortex_strength=results_dict["net_vortex_strength"],
            linear_impulse=results_dict["linear_impulse"],
            angular_impulse=results_dict["angular_impulse"],
            n_particles_total=n,
        )

    # Precision choice: f32 is the default throughout OpenONDA. Precision should
    # be selected at the Solver class constructor and propagated consistently.
    # Ensure no other code uses f64 as default.
    def compute_all(self, verbose: bool = True) -> None:
        """
        Compute flow integrals for all timesteps.

        Args:
            verbose: If True, print progress to console.
        """
        # Initialize evaluator (reuse across steps to maintain energy history for dE/dt)
        self.evaluator = ParticleFieldEvaluation(
            particle_kernel="GAUSSIAN",
            max_n_particles=self._estimate_max_particles(),
            accumulator_dtype=ti.f64,
        )
        self.results = []
        n_files = len(self.h5_files)

        if verbose:
            Logging.info(
                f"Offline diagnostics: Computing flow integrals for {n_files} timesteps..."
            )

        for i, h5_path in enumerate(self.h5_files):
            if verbose and (i % max(1, n_files // 10) == 0 or i == n_files - 1):
                Logging.info(f"  Processing {i + 1}/{n_files} ({100 * (i + 1) / n_files:.0f}%)")

            integrals = self._compute_single_time_step(h5_path)
            self.results.append(integrals)

        if verbose:
            Logging.info(
                f"Offline diagnostics: Completed. {len(self.results)} timesteps processed."
            )

    def _estimate_max_particles(self) -> int:
        """Estimate maximum number of particles across all files (lightweight check)."""
        # Just use a heuristic or check the last file which usually has most particles
        if not self.h5_files:
            return 1000
        # Check last file
        try:
            with h5py.File(self.h5_files[-1], "r") as f:
                attrs = f["solver"].attrs
                return int(attrs["n_particles_total"]) * 2  # Safety factor
        except Exception:
            return 10000

    def _compute_energy_dissipation_rate(self) -> np.ndarray:
        """Return only rates formed from one consistent direct-energy measure."""
        return np.asarray([result.kinetic_energy_rate for result in self.results], dtype=float)

    def save(self, output_path: str | Path | None = None) -> Path:
        """
        Save diagnostics to a log file.

        Args:
            output_path: Output file path. Defaults to 'diagnostics.log' in
                ``h5_directory``.

        Returns:
            Path to the created output file.
        """
        if output_path is None:
            output_path = self.base_dir / "diagnostics.log"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        kinetic_energy_rate = self._compute_energy_dissipation_rate()

        with open(output_path, "w") as f:
            # Header
            f.write("# OpenONDA Offline Flow Diagnostics\n")
            f.write(f"# Source: {self.base_dir}\n")
            f.write(f"# Timesteps: {len(self.results)}\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write("#   time            - Flow time [s]\n")
            f.write("#   n_particles_total - Number of particles\n")
            f.write("#   total_kinetic_energy  - Total kinetic energy [m²/s²]\n")
            f.write("#   total_enstrophy       - Total total_enstrophy [1/s²]\n")
            f.write("#   total_helicity        - Total total_helicity [m³/s²]\n")
            f.write("#   vortex_strength_magnitude_sum - Sum of particle |Gamma| [m³/s]\n")
            f.write("#   kinetic_energy_rate - Signed energy rate [m²/s³]\n")
            f.write("#   impulse_x/y/z   - Linear impulse components [m³/s]\n")
            f.write("#\n")

            # Column header
            f.write(
                f"{'time':>12} {'n_particles_total':>18} {'total_kinetic_energy':>16} {'total_enstrophy':>16} "
                f"{'total_helicity':>16} {'vortex_strength_magnitude_sum':>26} {'kinetic_energy_rate':>22} "
                f"{'impulse_x':>14} {'impulse_y':>14} {'impulse_z':>14}\n"
            )

            # Data rows
            for i, r in enumerate(self.results):
                f.write(
                    f"{r.time:12.6f} {r.n_particles_total:18d} {r.total_kinetic_energy:16.8e} "
                    f"{r.total_enstrophy:16.8e} {r.total_helicity:16.8e} "
                    f"{r.vortex_strength_magnitude_sum:26.8e} "
                    f"{kinetic_energy_rate[i]:22.8e} {r.linear_impulse[0]:14.6e} "
                    f"{r.linear_impulse[1]:14.6e} {r.linear_impulse[2]:14.6e}\n"
                )

        Logging.info(f"Offline diagnostics: Saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print a summary of the computed diagnostics."""
        if not self.results:
            Logging.warning("No results computed. Call compute_all() first.")
            return

        first = self.results[0]
        last = self.results[-1]

        Logging.section(
            "offline flow diagnostics",
            ("source", str(self.base_dir)),
            ("recorded states", len(self.results)),
            ("initial time", first.time, "s"),
            ("final time", last.time, "s"),
            ("initial particles", first.n_particles_total),
            ("final particles", last.n_particles_total),
            ("initial energy / density", first.total_kinetic_energy, "m^5/s^2"),
            ("final energy / density", last.total_kinetic_energy, "m^5/s^2"),
            (
                "energy ratio",
                last.total_kinetic_energy / first.total_kinetic_energy
                if first.total_kinetic_energy > EPSILON
                else None,
            ),
            flush=True,
        )


def compute_offline_diagnostics(
    h5_directory: str | Path,
    h5_glob: str = "vpm_*.h5",
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    Compute and save offline flow diagnostics from native VPM backup frames.

    Args:
        h5_directory: Directory containing the canonical HDF5 backup frames
            (``vpm_<step>.h5`` by default).
        h5_glob: Filename pattern used to select the frames below
            ``h5_directory``.
        output_path: Output file path. Defaults to 'diagnostics.log' in
            ``h5_directory``.
        verbose: If True, print a summary to the shared log.

    Returns:
        Path to the created output file.

    Example:
        >>> compute_offline_diagnostics('solution/vpm')
    """
    diagnostics = OfflineFlowDiagnostics(h5_directory, h5_glob=h5_glob)
    diagnostics.compute_all(verbose=verbose)

    if verbose:
        diagnostics.print_summary()

    return diagnostics.save(output_path)
