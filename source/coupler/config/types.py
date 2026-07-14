"""
Configuration for the FVM-VPM Coupler.
======================================
A single flat, physics-driven dataclass configures the coupled solver.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CouplerSetup:
    """Coupling and Eulerian-case parameters for an FVM-VPM run.

    VPM-specific physics remains in the injected VPM ``SolverConfig``. The
    coupler derives its VPM step from that solver and requires an integer ratio
    between the VPM step and ``dt``.
    """

    # ── Physics ───────────────────────────────────────────────────────────
    u_inf: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    """Freestream velocity [m/s]."""

    nu: float = 1e-5
    """Kinematic viscosity [m²/s]."""

    rho: float = 1.0
    """Fluid density [kg/m³]."""

    dt: float = 0.05
    """FVM sub-step [s].  The VPM/coupling step is read from the injected VPM
    solver and the integer sub-cycle count is derived internally."""

    t_end: float = 1.0
    """Simulated end time [s]."""

    backup_period: int = 1
    """Steps between solution snapshots.  ``backup_period * dt`` is the write
    interval in physical time, applied to both solvers."""

    log_period: int = 1
    """Steps between sampler outputs."""

    # ── Near field (FVM / OpenFOAM) ───────────────────────────────────────
    backend: str = "ofw"
    """Eulerian backend: wrapped OpenFOAM (``ofw``) or native Python (``fvm``)."""

    fvm_box: tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    """Eulerian near-field box bounds [x0, x1, y0, y1, z0, z1] [m]."""

    patch_name: str = "numericalBoundary"
    """Name of the coupling (outer) boundary patch."""

    wall_patch_name: str | None = "cube"
    """Name of the body wall patch."""

    grid_spacing: float = 0.05
    """FVM cell size [m]."""

    case_dir: str = "."
    """OpenFOAM case directory or native-FVM output root."""

    surface: dict = field(default_factory=dict)
    """Body geometry consumed by the OFW mesh-generation path."""

    # ── Far field (VPM) ───────────────────────────────────────────────────
    h: float = 0.05
    """Particle and hand-off lattice spacing [m]."""

    # ── Coupling ──────────────────────────────────────────────────────────
    buffer_thickness: float = 0.10
    """Width [m] of the η cosine ramp (fringe band) inside the FVM box."""

    dead_zone_h: float = 3.0
    """η=0 band at every box face, in units of h (keeps the FVM boundary
    pressure artifact out of exiting particles)."""

    prune_vorticity_min: float = 0.01
    """Vorticity threshold [1/s] used to prune hand-off lattice nodes."""

    overlap_velocity_forcing: bool = True
    """Advect overlap particles with the η-blended FVM velocity instead of pure
    Biot–Savart (velocity forcing).  Disable to recover reconstruction-only
    transport (A/B comparisons)."""

    fringe_strength: float = 4.0
    """Fringe relaxation strength A (see fvm_fringe.py)."""

    blend_relaxation: float = 1.0
    """Under-relaxation of the overlap blend toward the FVM target."""

    strength_correction_iterations: int = 3
    """Iterations of the regularized particle-strength correction; zero disables it."""

    strength_correction_relax: float = 1.0
    """Under-relaxation λ ∈ (0, 1] of the strength-correction iteration."""

    bc_coupling_iterations: int = 1
    """Donor-boundary/PIMPLE Picard iterations per FVM substep."""

    donor_interior_source: str = "particles"
    """Interior donor representation: hand-off particles or live FVM vorticity."""

    donor_bc_mode: str = "dirichlet"
    """Donor velocity condition: full Dirichlet or normal-Dirichlet mixed mode."""

    overlap_radius_ratio: float = 1.5
    """Overlap-particle core radius divided by ``h``."""

    def __post_init__(self) -> None:
        """Validate enum-like config fields."""
        _valid_backends = ("ofw", "fvm")
        if self.backend not in _valid_backends:
            raise ValueError(f"backend must be one of {_valid_backends!r}, got {self.backend!r}.")
        _valid_donor_interior_sources = ("particles", "fvm")
        if self.donor_interior_source not in _valid_donor_interior_sources:
            raise ValueError(
                f"donor_interior_source must be one of "
                f"{_valid_donor_interior_sources!r}, got "
                f"{self.donor_interior_source!r}."
            )
        _valid_donor_bc_modes = ("dirichlet", "mixed")
        if self.donor_bc_mode not in _valid_donor_bc_modes:
            raise ValueError(
                f"donor_bc_mode must be one of {_valid_donor_bc_modes!r}, "
                f"got {self.donor_bc_mode!r}."
            )

    # ── Derived properties ────────────────────────────────────────────────
    @property
    def U_inf(self) -> np.ndarray:
        return np.array(self.u_inf, dtype=np.float64)

    @property
    def is_ramping(self) -> bool:
        # No velocity ramp in the coupled cases; kept because SetupHandler
        # reads it when writing the initial 0/U field.
        return False

    # Namespacing aliases: SetupHandler addresses the flat setup as
    # ``cfg.physics.*``, ``cfg.fvm_solver.*`` and ``cfg.fvm_domain.*``; the flat
    # object answers for all three.  (Not a solver — despite the name,
    # ``fvm_solver`` here is the FVM-case *parameters*, distinct from the
    # injected FVM solver object.)
    @property
    def physics(self):
        return self

    @property
    def fvm_solver(self):
        return self

    @property
    def fvm_domain(self):
        return self

    def to_dict(self) -> dict:
        return {
            "physics": {
                "u_inf": self.u_inf,
                "nu": self.nu,
                "rho": self.rho,
                "dt": self.dt,
                "t_end": self.t_end,
                "backup_period": self.backup_period,
                "log_period": self.log_period,
            },
            "fvm_solver": {
                "backend": self.backend,
                "patch_name": self.patch_name,
                "wall_patch_name": self.wall_patch_name,
                "grid_spacing": self.grid_spacing,
                "fvm_domain": {
                    "xmin": self.fvm_box[0],
                    "xmax": self.fvm_box[1],
                    "ymin": self.fvm_box[2],
                    "ymax": self.fvm_box[3],
                    "zmin": self.fvm_box[4],
                    "zmax": self.fvm_box[5],
                },
                "surface": self.surface,
            },
            # Coupling-interface geometry (the injected VPM's physics lives in
            # its own SolverConfig, not here).  The acceptance scripts read
            # h / buffer_thickness / dead_zone_h from this block.
            "vpm_solver": {
                "particle_spacing": self.h,
                "buffer_thickness": self.buffer_thickness,
                "dead_zone_h": self.dead_zone_h,
                "h": self.h,
            },
            "coupler": {
                "blend_relaxation": self.blend_relaxation,
                "prune_vorticity_min": self.prune_vorticity_min,
                "strength_correction_iterations": self.strength_correction_iterations,
                "strength_correction_relax": self.strength_correction_relax,
                "overlap_radius_ratio": self.overlap_radius_ratio,
                "overlap_velocity_forcing": self.overlap_velocity_forcing,
                "bc_coupling_iterations": self.bc_coupling_iterations,
                "donor_interior_source": self.donor_interior_source,
                "donor_bc_mode": self.donor_bc_mode,
            },
        }


# Public compatibility alias used by older setup scripts.
CouplerConfig = CouplerSetup
