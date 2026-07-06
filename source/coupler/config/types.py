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
class CouplerConfig:
    """Configuration for the FVM-VPM coupled solver.

    One flat dataclass, one coupling path.  Defaults reflect the validated
    cubeFlow setup; case scripts override only what differs.
    """

    # ── Physics ───────────────────────────────────────────────────────────
    u_inf: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    """Freestream velocity [m/s]."""

    nu: float = 1e-5
    """Kinematic viscosity [m²/s]."""

    rho: float = 1.0
    """Fluid density [kg/m³]."""

    dt: float = 0.05
    """Coupled time step [s] (shared by FVM and VPM)."""

    t_end: float = 1.0
    """Simulated end time [s]."""

    backup_period: int = 1
    """Steps between solution snapshots.  ``backup_period * dt`` is the write
    interval in physical time, applied to both solvers."""

    log_period: int = 1
    """Steps between sampler outputs."""

    # ── Near field (FVM / OpenFOAM) ───────────────────────────────────────
    # The near-body solver (OpenFOAM ``fvm_solver(case_dir)`` or the native
    # ``FVM.Solver.from_case``) is built by the caller and injected, so the
    # backend choice lives in the setup script, not here.
    fvm_box: tuple[float, float, float, float, float, float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    """OpenFOAM domain bounds [x0, x1, y0, y1, z0, z1] [m]."""

    patch_name: str = "numericalBoundary"
    """Name of the coupling (outer) boundary patch."""

    wall_patch_name: str | None = "cube"
    """Name of the body wall patch."""

    grid_spacing: float = 0.05
    """FVM cell size [m]."""

    case_dir: str = "."
    """Path to the OpenFOAM case directory."""

    surface: dict = field(default_factory=dict)
    """Body geometry definition consumed by ``assets/mesh_helper.py``, e.g.
    ``{"cube": {"side_length": 1.0, "center": [0,0,0], "refinement": 25}}`` or
    ``{"cylinder": {"diameter": 1.0, "center": [0,0,0], "refinement": 60}}``."""

    # ── Far field (VPM) ───────────────────────────────────────────────────
    # The VPM is built by the caller with its OWN native ``SolverConfig`` and
    # injected (``FVMVPMCoupler.from_solvers``).  Its physics — viscous scheme,
    # stretching, advection, turbulence, treecode θ, stabilization, particle
    # kernel, max_particles, precision, domain bounds — therefore lives in that
    # ``SolverConfig``, NOT here.  The coupler reads the domain/kernel/dtype it
    # needs back from the injected solver and validates them against ``fvm_box``
    # (see ``_validate_injected_vpm``).  Only the particle SPACING, which must
    # equal the hand-off lattice spacing, is a coupling parameter:
    h: float = 0.05
    """Particle spacing [m] — the hand-off lattice spacing; must equal the
    injected VPM's particle spacing.  (Typically also matches the FVM
    ``grid_spacing`` so the interface resolutions agree.)"""

    # ── Coupling ──────────────────────────────────────────────────────────
    buffer_thickness: float = 0.10
    """Width [m] of the η cosine ramp (fringe band) inside the FVM box."""

    dead_zone_h: float = 3.0
    """η=0 band at every box face, in units of h (keeps the FVM boundary
    pressure artifact out of exiting particles)."""

    prune_vorticity_min: float = 0.01
    """Hand-off prune floor expressed as a VORTICITY [1/s]: lattice nodes with
    |Γ| < prune_vorticity_min · h³ are pruned (then moment-redistributed).
    Scale-correct under h-refinement (Γ_node ∝ h³).  Choose ~1e-2 · U∞/L_char;
    the tutorials (U∞ = L = 1) use 0.01."""

    overlap_velocity_forcing: bool = True
    """Advect overlap particles with the η-blended FVM velocity instead of pure
    Biot–Savart (velocity forcing).  Disable to recover reconstruction-only
    transport (A/B comparisons)."""

    fringe_strength: float = 4.0
    """Fringe relaxation strength A (see fvm_fringe.py)."""

    blend_relaxation: float = 1.0
    """Under-relaxation α ∈ (0, 1] of the η-blend toward the FVM target:
    Γ ← Γ + α·η·(target − Γ).  α = 1 is the hard overwrite (validated)."""

    strength_correction_iterations: int = 3
    """Beale/Picard iterated strength assignment (Beale 1988): after the
    hand-off, lattice strengths are iterated so the *mollified* particle
    vorticity (Gaussian core, σ = 1.5h) matches ω_FVM at the nodes — a
    regularized deconvolution of the kernel smoothing on resolved scales.
    The correction is body-guarded (never acts across the wall).  0 = off."""

    strength_correction_relax: float = 1.0
    """Under-relaxation λ ∈ (0, 1] of the strength-correction iteration."""

    period_multiplier: int = 1
    """FVM sub-cycles per VPM (coupling) step — the multi-rate control.

    The configured ``dt`` is the **FVM** step ``dt_fvm`` (small, accurate).  The
    VPM cloud advances once per ``dt_vpm = period_multiplier · dt_fvm``, and the
    coupling cadence (donor BC, hand-off, viscous regen, samplers, backups) is the
    VPM step.  Each VPM step the FVM is sub-cycled ``period_multiplier`` times,
    its boundary velocity **linearly interpolated** between the previous cycle's
    donor BC (held at the cycle start) and the freshly-advanced VPM's donor BC
    (the "future" value at the cycle end), re-projected solenoidal each sub-step.

    ``1`` = synchronous FVM/VPM stepping (no sub-cycling). Use ``>1`` when the
    coupling/VPM step gives
    the FVM an inaccurate Courant number (e.g. cube: dt=0.01, period_multiplier=10
    → dt_vpm=0.1, FVM Co≈0.8 like the reference instead of Co≈11)."""

    bc_coupling_iterations: int = 1
    """Outer Picard iterations of the donor-velocity BC against the FVM pressure
    solve, per time step (Weymouth & Lauber, JCP 2025, arXiv:2404.09034).

    The Biot–Savart donor BC is non-locally coupled to the pressure: the pressure
    field generated inside the FVM box during the solve changes the interior
    vorticity, which changes the BS velocity the box boundary should see.  A
    single one-shot donor BC ignores
    this and leaves the boundary inconsistent with the interior — which damps the
    global shedding feedback for sustained bluff-body wakes.

    With >1, each step iterates:  recompute donor = U∞ + BS(exterior particles)
    + BS(FVM interior vorticity)  →  ``solve_pimple()`` (no time advance)  →
    repeat, then ``advance_time()``.  The FVM-interior term is re-evaluated from
    the freshly-solved field each iteration, closing the BC↔pressure coupling
    without re-remeshing the cloud (which would inject extra diffusion).  2–3 is
    typically enough; the residual is bounded by the per-step pressure change."""

    donor_interior_source: str = "particles"
    """Which representation of the FVM-box INTERIOR feeds the donor BC trace.

    ``"particles"`` (legacy) — the donor is one full-cloud Biot–Savart per
    coupling window: U∞ + BS(all particles, in-box included), linearly
    interpolated across the FVM sub-steps.  The in-box particles are a
    mollified copy of the *previous* window's FVM interior advected by the
    VPM, so the interior term the boundary sees lags by up to one full
    coupling step dt_vpm, and the linear interpolation of the trace smears
    vortices that translate past a face within the window.

    ``"fvm"`` (Weymouth–Lauber-consistent, arXiv:2404.09034) — the donor is
    decomposed per sub-step:

        u_bc(t) = U∞ + BS(exterior particles, interpolated in t)
                     + BS(FVM interior ω, evaluated LIVE at t)

    The fast near-field term is re-evaluated from the freshly solved FVM
    vorticity at EVERY sub-step (no time interpolation, no stale in-box
    representation, no BC-family jump at the final sub-step); only the slow,
    distant exterior-wake term is interpolated.  ``bc_coupling_iterations``
    then counts Picard iterations of BC↔pressure per sub-step (1 = one-shot
    with the interior at the sub-step's OLD time level, lag dt_fvm; 2 closes
    the pressure coupling at the new time level).  Collective-safe under MPI:
    the vorticity gather runs on all ranks, the Biot–Savart on rank 0 only."""

    donor_bc_mode: str = "dirichlet"
    """Type of donor velocity BC imposed on the FVM coupling patch
    (Billuart et al., JCP 2023, §3.1, Eqs. 11–14).

    ``"dirichlet"`` imposes the full
    donor velocity vector as a Dirichlet condition
    (``set_dirichlet_velocity_boundary_condition_vec``).  Any velocity mismatch
    between the VPM donor and the FVM interior is converted into spurious
    vorticity at the boundary that advects into the wake — the cubeFlow
    interface-noise defect (onset t > 14 s).

    ``"mixed"`` imposes a Robin / mixed condition
    (``set_robin_velocity_boundary_condition``): the *normal* component is
    Dirichlet (``u·n̂ = u_VPM·n̂``) and the *tangential* component is a
    vorticity-matched Neumann condition (``∂u_t/∂n = ω_VPM × n̂``) via an
    OpenFOAM ``directionMixed`` patch whose ``valueFraction`` tensor is
    ``n̂⊗n̂``.  This conserves the vorticity flux through the boundary, so a
    velocity mismatch no longer generates spurious vorticity.  Requires the
    ``0/U`` BC type of the coupling patch to be ``directionMixed`` (the
    ``setup`` handler switches it automatically when this is ``"mixed"``);
    otherwise the C++ wrapper silently falls back to fixedValue.

    The solenoidal projection of ``u_donor`` (Gresho–Sani compatibility) is
    applied in both modes — it is still needed for the normal component /
    ``fixedFluxPressure`` pairing."""

    handoff_target_mode: str = "vorticity"
    """How the FVM target circulation is computed for the hand-off blend
    (Billuart et al., JCP 2023, §3.3).

    ``"vorticity"`` scatters ``ω_FVM · V_cell`` directly onto
    the VPM lattice via M4′.  Fast, but interpolating ω (a derivative) is
    inaccurate in high-gradient regions (boundary layers, shear layers) and
    non-conservative.

    ``"velocity"`` (FIX B) scatters ``u_FVM · V_cell`` and ``V_cell`` separately,
    divides to get the interpolated velocity on the lattice (with the configured
    freestream ``u_inf`` subtracted first), then computes ``ω = ∇×u`` via
    central differences on the **data-extent sub-grid** — the bounding box of
    lattice nodes that received FVM data.  Because u is smoother than ω (one
    integration lower), the interpolation is more accurate.  Conservation holds
    by the **discrete Stokes theorem on the sub-grid**: the curl is not taken on
    the full lattice (where ``u_lat = 0`` in the buffer would telescope the sum
    to zero — the original FIX B defect), nor are the sub-grid corner holes
    (from the non-rectangular M4′-support data region) left as zero
    discontinuities; they are axis-wise edge-padded (Neumann) before the curl so
    ``Σ_node Γ = ∮_∂data (n̂ × u_lat) dS ≈ ∫_box ω_FVM dV``.  The vorticity path
    is exactly conservative (M4′ partition of unity); the velocity path is
    conservative to the ~2h data-extent smear, converging as the lattice
    refines.  The default is ``"vorticity"``; the
    velocity path is exercised by ``tests/coupler/test_continuous_overlap.py``."""

    body_panel_enabled: bool = False
    """Add a boundary-element (Hess-Smith panel) body model to the VPM so its
    induced field carries the body's IRROTATIONAL blockage — the no-penetration
    completion that free vortex particles structurally cannot represent (the
    diagnosed root cause of the wall-origin near-body error). Requires
    ``panel_mesh`` (a coarse closed STL)."""

    panel_bc_type: str = "DIRICHLET"
    """Panel boundary condition: ``"DIRICHLET"`` (Morino, validated for closed
    bluff bodies) or ``"NEUMANN"``.  Use DIRICHLET — NEUMANN is not calibrated
    for closed bodies in the current panel solver."""

    panel_mesh: str | None = None
    """Path to the coarse closed-surface panel STL for the body model (e.g.
    ``constant/triSurface/cube_panels.stl``, written by ``mesh_helper.py``).
    Keep it coarse (~1e3 panels): the dense AIC solve is O(n_panels^3)/step."""

    overlap_radius_ratio: float = 1.5
    """Particle core radius σ in units of h for overlap-region particles.
    σ sets the deconvolution bandwidth of the Beale correction: thin vortex
    sheets (1–2h, e.g. separated shear layers at moderate Re) are reconstructed
    at ~56 % peak with σ=1.5h/2 iters, ~81 % with σ=1.2h/4 iters, ~93 % with
    σ=1.0h/4 iters.  σ/h < 1 breaks particle overlap (BS field ripple) —
    do not go below 1.0.  Default 1.5 is the smoothest/safest."""

    def __post_init__(self) -> None:
        """Validate enum-like config fields."""
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
        _valid_handoff_modes = ("vorticity", "velocity")
        if self.handoff_target_mode not in _valid_handoff_modes:
            raise ValueError(
                f"handoff_target_mode must be one of {_valid_handoff_modes!r}, "
                f"got {self.handoff_target_mode!r}."
            )

    # ── Derived properties ────────────────────────────────────────────────
    @property
    def U_inf(self) -> np.ndarray:
        return np.array(self.u_inf, dtype=np.float64)

    @property
    def U_mag(self) -> float:
        return float(np.linalg.norm(self.U_inf))

    @property
    def is_ramping(self) -> bool:
        return False

    @property
    def velocity_ramp_period(self) -> float:
        return 0.0

    # Compatibility aliases: helpers address the config as cfg.physics.*,
    # cfg.fvm_solver.* and cfg.vpm_solver.*; the flat config answers for all.
    @property
    def physics(self):
        return self

    @property
    def vpm_solver(self):
        return self

    @property
    def fvm_solver(self):
        return self

    @property
    def particle_spacing(self):
        return self.h

    @property
    def fvm_domain(self):
        return self

    def as_list(self):
        return list(self.fvm_box)

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
                "body_panel_enabled": self.body_panel_enabled,
                "panel_bc_type": self.panel_bc_type,
                "panel_mesh": self.panel_mesh,
                "bc_coupling_iterations": self.bc_coupling_iterations,
                "donor_interior_source": self.donor_interior_source,
                "donor_bc_mode": self.donor_bc_mode,
                "handoff_target_mode": self.handoff_target_mode,
                "period_multiplier": self.period_multiplier,
            },
        }
