"""
Offline post-processing diagnostics: flow integrals (energy, enstrophy,
circulation, impulse) computed from saved particle states.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

# defusedxml for safe parsing; Element type only from stdlib (defusedxml has none).
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET  # noqa: N817
import h5py
import numpy as np
import taichi as ti

from ..config.constants import EPSILON

# Use VPM logger
from ..io.logging import logger
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
    kinetic_energy: float
    helicity: float
    enstrophy: float
    vorticity_dissipation_rate: float
    strength_magnitude: float
    total_strength: np.ndarray  # (3,)
    linear_impulse: np.ndarray  # (3,)
    angular_impulse: np.ndarray  # (3,)
    n_particles: int

class ParticleContainerWrapper:
    """
    Lightweight wrapper to mimic Particles class interface for ParticleFieldEvaluation.

    Wraps Taichi fields to provide .position, .circulation, etc. access.
    """

    def __init__(self, position, circulation, radius, viscosity_effective, count):
        self.position = position
        self.circulation = circulation
        self.radius = radius
        self.viscosity_effective = viscosity_effective
        self._count = count

    def __len__(self):
        return self._count

class OfflineFlowDiagnostics:
    """
    Offline flow diagnostics processor for VPM backup files.

    Reads particle data from HDF5 backup files referenced by a temporal XDMF file
    and computes all integral flow quantities using GPU-accelerated Taichi kernels.

    Attributes:
        xdmf_path: Path to the temporal XDMF file.
        h5_files: List of HDF5 files extracted from XDMF.
        results: List of FlowIntegrals for each timestep.

    Example:
        >>> diagnostics = OfflineFlowDiagnostics('solution/vpm_temporal.xdmf')
        >>> diagnostics.compute_all()
        >>> diagnostics.save()  # Saves to 'diagnostics.log'
        >>> diagnostics.save('custom_output.csv')
    """

    def __init__(self, xdmf_path: str | Path):
        """
        Initialize the offline diagnostics processor.

        Args:
            xdmf_path: Path to the temporal XDMF file containing references
                       to all timestep HDF5 files.

        Raises:
            FileNotFoundError: If the XDMF file does not exist.
            ValueError: If the XDMF file is not a valid temporal collection.
        """
        self.xdmf_path = Path(xdmf_path)
        if not self.xdmf_path.exists():
            raise FileNotFoundError(f"XDMF file not found: {xdmf_path}")

        self.base_dir = self.xdmf_path.parent
        self.h5_files: list[Path] = []
        self.results: list[FlowIntegrals] = []

        self._parse_xdmf()

        logger.info(
            f"[OfflineFlowDiagnostics] Loaded {len(self.h5_files)} timesteps from {xdmf_path}"
        )

    def _parse_xdmf(self) -> None:
        """Parse the temporal XDMF file to extract HDF5 file references."""
        tree = ET.parse(self.xdmf_path)
        root = tree.getroot()

        # Extract unique HDF5 file references
        h5_refs = self._extract_h5_references(root)
        if not h5_refs:
            raise ValueError(f"No HDF5 references found in {self.xdmf_path}")

        # Convert to validated paths and sort by timestep
        self.h5_files = self._resolve_and_sort_h5_paths(h5_refs)

    def _extract_h5_references(self, root: Element) -> set:
        """Extract unique HDF5 file references from XDMF DataItem elements."""
        h5_refs = set()
        for data_item in root.iter("DataItem"):
            if data_item.get("Format") != "HDF":
                continue
            text = (data_item.text or "").strip()
            if ":" in text:
                h5_refs.add(text.split(":")[0])
        return h5_refs

    def _resolve_and_sort_h5_paths(self, h5_refs: set) -> list[Path]:
        """Convert HDF5 references to validated paths, sorted by timestep."""
        h5_paths = []
        for ref in h5_refs:
            path = self.base_dir / ref
            if path.exists():
                h5_paths.append(path)
            else:
                logger.warning(f"  Warning: HDF5 file not found: {path}")

        # Sort by timestep number extracted from filename
        return sorted(h5_paths, key=self._extract_timestep_from_path)

    @staticmethod
    def _extract_timestep_from_path(path: Path) -> int:
        """Extract timestep number from HDF5 filename (e.g., _000123.h5 -> 123)."""
        match = re.search(r"_(\d{6})\.h5$", path.name)
        return int(match.group(1)) if match else 0

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

            data["time"] = float(solver.attrs["flow_time"])
            data["n_particles"] = int(solver.attrs["number_of_particles"])

            n: int = data["n_particles"]
            if n > 0:
                data["position"] = particles["position"][:n]
                data["circulation"] = particles["circulation"][:n]
                data["radius"] = particles["radius"][:n]
                data["viscosity"] = particles["viscosity"][:n]
                data["viscosity_turbulent"] = particles["viscosity_turbulent"][:n]
                data["viscosity_eff"] = data["viscosity"] + data["viscosity_turbulent"]
            else:
                data["position"] = np.empty((0, 3), dtype=np.float32)
                data["circulation"] = np.empty((0, 3), dtype=np.float32)
                data["radius"] = np.empty((0,), dtype=np.float32)
                data["viscosity_eff"] = np.empty((0,), dtype=np.float32)

        return data

    def _compute_single_timestep(self, h5_path: Path) -> FlowIntegrals:
        """Compute flow integrals for a single timestep."""
        data = self._load_particle_data(h5_path)
        n = data["n_particles"]

        if n == 0:
            return FlowIntegrals(
                time=data["time"],
                kinetic_energy=0.0,
                helicity=0.0,
                enstrophy=0.0,
                vorticity_dissipation_rate=0.0,
                strength_magnitude=0.0,
                total_strength=np.zeros(3),
                linear_impulse=np.zeros(3),
                angular_impulse=np.zeros(3),
                n_particles=0,
            )

        # Allocate fields for this step
        pos_field = ti.Vector.field(3, ti.f32, shape=n)
        str_field = ti.Vector.field(3, ti.f32, shape=n)
        rad_field = ti.field(ti.f32, shape=n)
        visc_field = ti.field(ti.f32, shape=n)

        # Copy data to Taichi
        pos_field.from_numpy(data["position"].astype(np.float32))
        str_field.from_numpy(data["circulation"].astype(np.float32))
        rad_field.from_numpy(data["radius"].astype(np.float32))
        visc_field.from_numpy(data["viscosity_eff"].astype(np.float32))

        # Create wrapper mimicking Particles class
        particles_wrapper = ParticleContainerWrapper(
            position=pos_field,
            circulation=str_field,
            radius=rad_field,
            viscosity_effective=visc_field,
            count=n,
        )

        results_dict = self.evaluator.compute_flow_integrals(particles_wrapper, data["time"])

        return FlowIntegrals(
            time=data["time"],
            kinetic_energy=results_dict["kinetic_energy"],
            helicity=results_dict["helicity"],
            enstrophy=results_dict["enstrophy"],
            vorticity_dissipation_rate=results_dict["vorticity_dissipation_rate"],
            strength_magnitude=results_dict["strength_magnitude"],
            total_strength=results_dict["strength"],
            linear_impulse=results_dict["linear_impulse"],
            angular_impulse=results_dict["angular_impulse"],
            n_particles=n,
        )

    # Precision policy: f32 is the default throughout OpenONDA. Precision should
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
            particles_kernel="GAUSSIAN",
            max_particles=self._estimate_max_particles(),
            accumulator_dtype=ti.f64,
        )
        self.results = []
        n_files = len(self.h5_files)

        if verbose:
            logger.info(
                f"[OfflineFlowDiagnostics] Computing flow integrals for {n_files} timesteps..."
            )

        for i, h5_path in enumerate(self.h5_files):
            if verbose and (i % max(1, n_files // 10) == 0 or i == n_files - 1):
                logger.info(f"  Processing {i + 1}/{n_files} ({100 * (i + 1) / n_files:.0f}%)")

            integrals = self._compute_single_timestep(h5_path)
            self.results.append(integrals)

        if verbose:
            logger.info(
                f"[OfflineFlowDiagnostics] Completed. {len(self.results)} timesteps processed."
            )

    def _estimate_max_particles(self) -> int:
        """Estimate maximum number of particles across all files (lightweight check)."""
        # Just use a heuristic or check the last file which usually has most particles
        if not self.h5_files:
            return 1000
        # Check last file
        try:
            with h5py.File(self.h5_files[-1], "r") as f:
                return int(f["solver"].attrs["number_of_particles"]) * 2  # Safety factor
        except Exception:
            return 10000

    def _compute_energy_dissipation_rate(self) -> np.ndarray:
        """Compute dE/dt using centred finite differences (np.gradient).

        ``np.gradient`` uses second-order accurate centred differences for all
        interior points and one-sided stencils at the boundaries, with the
        actual (possibly non-uniform) time spacing as the coordinate.  This is
        more accurate than the previous backward-only BDF4 scheme, which (a)
        assumed uniform spacing and (b) had no smoothing at all.
        """
        times = np.array([r.time for r in self.results])
        energies = np.array([r.kinetic_energy for r in self.results])

        if len(times) < 2:
            return np.zeros(len(times))

        return np.gradient(energies, times)

    def save(self, output_path: str | Path | None = None) -> Path:
        """
        Save diagnostics to a log file.

        Args:
            output_path: Output file path. Defaults to 'diagnostics.log' in the
                         same directory as the XDMF file.

        Returns:
            Path to the created output file.
        """
        if output_path is None:
            output_path = self.base_dir / "diagnostics.log"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        dE_dt = self._compute_energy_dissipation_rate()

        with open(output_path, "w") as f:
            # Header
            f.write("# OpenONDA Offline Flow Diagnostics\n")
            f.write(f"# Source: {self.xdmf_path}\n")
            f.write(f"# Timesteps: {len(self.results)}\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write("#   time            - Flow time [s]\n")
            f.write("#   n_particles     - Number of particles\n")
            f.write("#   kinetic_energy  - Total kinetic energy [m²/s²]\n")
            f.write("#   enstrophy       - Total enstrophy [1/s²]\n")
            f.write("#   helicity        - Total helicity [m³/s²]\n")
            f.write("#   strength_mag    - Total circulation magnitude [m²/s]\n")
            f.write("#   dE_dt           - Energy dissipation rate [m²/s³]\n")
            f.write("#   impulse_x/y/z   - Linear impulse components [m³/s]\n")
            f.write("#\n")

            # Column header
            f.write(
                f"{'time':>12} {'n_particles':>12} {'kinetic_energy':>16} {'enstrophy':>16} "
                f"{'helicity':>16} {'strength_mag':>16} {'dE_dt':>16} "
                f"{'impulse_x':>14} {'impulse_y':>14} {'impulse_z':>14}\n"
            )

            # Data rows
            for i, r in enumerate(self.results):
                f.write(
                    f"{r.time:12.6f} {r.n_particles:12d} {r.kinetic_energy:16.8e} "
                    f"{r.enstrophy:16.8e} {r.helicity:16.8e} {r.strength_magnitude:16.8e} "
                    f"{dE_dt[i]:16.8e} {r.linear_impulse[0]:14.6e} "
                    f"{r.linear_impulse[1]:14.6e} {r.linear_impulse[2]:14.6e}\n"
                )

        logger.info(f"[OfflineFlowDiagnostics] Saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print a summary of the computed diagnostics."""
        if not self.results:
            print("No results computed. Call compute_all() first.")
            return

        first = self.results[0]
        last = self.results[-1]

        print("\n" + "=" * 70)
        print("OFFLINE FLOW DIAGNOSTICS SUMMARY")
        print("=" * 70)
        print(f"  Source:      {self.xdmf_path}")
        print(f"  Timesteps:   {len(self.results)}")
        print(f"  Time range:  {first.time:.4f} - {last.time:.4f} s")
        print("-" * 70)
        print(f"  Initial particles:   {first.n_particles:,}")
        print(f"  Final particles:     {last.n_particles:,}")
        print(f"  Initial energy:      {first.kinetic_energy:.6e}")
        print(f"  Final energy:        {last.kinetic_energy:.6e}")
        print(
            f"  Energy ratio:        {last.kinetic_energy / first.kinetic_energy:.4f}"
            if first.kinetic_energy > EPSILON
            else "  Energy ratio:        N/A"
        )
        print("=" * 70 + "\n")

def ComputeOfflineDiagnostics(
    backup_pattern: str | None = None,
    xdmf_path: str | Path | None = None,
    output_path: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    Compute and save offline flow diagnostics from VPM backup files.

    This function can accept either a glob pattern for backup files or a
    pre-existing temporal XDMF file. If backup_pattern is provided, it will
    automatically create the temporal XDMF file.

    Args:
        backup_pattern: Glob pattern for backup files (e.g., 'solution/case/case_*').
                        If provided, a temporal XDMF is generated automatically.
        xdmf_path: Path to existing temporal XDMF file. Ignored if backup_pattern is set.
        output_path: Output file path. Defaults to 'diagnostics.log' in the
                     same directory as the backup files.
        verbose: If True, print progress to console.

    Returns:
        Path to the created output file.

    Example:
        >>> # From backup pattern (recommended)
        >>> ComputeOfflineDiagnostics('solution/lamb_oseen/lamb_oseen_*')

        >>> # From existing XDMF file
        >>> ComputeOfflineDiagnostics(xdmf_path='solution/vpm_temporal.xdmf')
    """
    # Import BackupSystem internally to minimize user-facing imports
    from ..io import BackupSystem

    # Resolve XDMF path
    if backup_pattern is not None:
        xdmf_path = BackupSystem.create_temporal_xdmf(backup_pattern=backup_pattern)
    elif xdmf_path is None:
        raise ValueError("Either backup_pattern or xdmf_path must be provided")

    diagnostics = OfflineFlowDiagnostics(xdmf_path)
    diagnostics.compute_all(verbose=verbose)

    if verbose:
        diagnostics.print_summary()

    return diagnostics.save(output_path)
