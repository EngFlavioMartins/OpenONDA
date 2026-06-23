"""
Configuration for the FVM-VPM Coupler.
======================================
A single flat, physics-driven dataclass configures the coupled solver.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: June 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from source.solvers.VPM.config.types import ViscousConfig


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
    eulerian_backend: str = "OFW"
    """Eulerian (near-body) backend: ``"OFW"`` (OpenFOAM Cython wrapper, default)
    or ``"FVM"`` (OpenONDA native finite-volume solver).  Both expose the same
    duck-typed contract; ``"FVM"`` builds via ``source.solvers.FVM.Solver.from_case``
    and runs serially (``n_procs()==1``)."""

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
    h: float = 0.05
    """Particle spacing [m] (match ``grid_spacing``)."""

    vpm_domain: tuple[float, float, float, float, float, float] = (-2.0, 5.0, -2.0, 2.0, -2.0, 2.0)
    """VPM domain bounds [m]; particles outside are removed."""

    max_particles: int = 500_000

    particles_kernel: str = "GAUSSIAN"
    """VPM regularization kernel.  The strength correction assumes GAUSSIAN."""

    treecode_theta: float = 0.3
    """Barnes-Hut opening angle (lower = more accurate, slower).  Each VPM step
    does ~6 treecode evaluations; ~0.4 is ~1.3-1.7x faster per eval at <1%
    velocity error and is a good default for coupled runs."""

    advection_scheme: str = "RK2"
    """Time integrator for particle advection: "RK2" | "RK4" | "RK3" | "EULER".
    RK4 costs 4 velocity (treecode) evaluations per step, RK2 only 2 — and since
    the overlap is remeshed every coupling step (the hand-off, not the time
    integrator, sets accuracy), RK2 is sufficient and ~25-30% cheaper on the
    dominant VPM stage.  Use "RK4" to recover the higher-order integrator."""

    viscous_scheme: ViscousConfig | None = None
    """VPM viscous diffusion scheme (``None`` = Core Spreading)."""

    les_smagorinsky_cs: float = 0.17
    """Smagorinsky constant Cs for the VPM sub-grid model (ν_t = (Cs·Δ)²|S|).

    0.17 is the standard value and matches OpenFOAM's Smagorinsky default
    (Ck=0.094 ⇒ Cs≈0.17) used by the reference/hybrid FVM, so the VPM far-wake
    is not damped harder than the grid it couples to.  The previous 0.3 put
    ν_t on the order of the molecular ν at Re=1000, roughly halving the
    effective wake Reynolds number and suppressing bluff-body shedding.
    0 disables the sub-grid model entirely."""

    # ── Stretching stabilization (ISR — Implicit Strain Relaxation) ────────
    isr_enabled: bool = False
    """Enable VPM strength-relaxation stabilization of the vortex-stretching
    term (``source/solvers/VPM/stabilization/strength_relaxation.py``).  When
    True the (ω·∇)u update is sub-stepped so σ_eff·dt_sub ≤ ``isr_cfl`` AND the
    kernel-unrepresentable strength residual (the grid-scale checkerboard that
    explicit stretching amplifies) is filtered each sub-step.  Without it the
    free-wake stretching term blows up for long runs.  Default False (passes
    straight through to the VPM ``SolverConfig``, which also defaults False)."""

    isr_mode: str = "blend"
    """ISR relaxation mode: ``"blend"`` (residual filter, damps the
    unrepresentable magnitude — the right choice for a magnitude runaway) or
    ``"pedrizzetti"`` (direction realignment, |Γ|-preserving)."""

    isr_cfl: float = 0.2
    """Target σ_eff·dt per ISR sub-step (sets the sub-step count
    k = ceil(max σ_eff·dt / isr_cfl))."""

    isr_C: float = 1.0
    """ISR strain-gate rate constant: r = 1 − exp(−C·σ_eff·dt_sub) per sub-step."""

    isr_k_max: int = 8
    """Hard cap on ISR sub-steps per macro step."""

    precision: Literal["f32", "f64"] = "f32"
    """Floating-point precision shared by the VPM solver and every coupler
    data hand-off (donor BC, particle injection, body-panel RHS).  ``'f32'``
    (default) or ``'f64'``.  Propagated to the VPM ``SolverConfig`` and used to
    pick the NumPy dtype of all arrays written into the VPM Taichi fields so
    no f32←f64 precision-loss warnings are emitted and a future f64 run is
    end-to-end consistent.  ``'f64'`` requires the CPU backend (Vulkan/CUDA do
    not support f64 well)."""

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

    ``1`` (default) = synchronous FVM/VPM stepping (no sub-cycling), byte-identical
    to the legacy single-rate path.  Use ``>1`` when the coupling/VPM step gives
    the FVM an inaccurate Courant number (e.g. cube: dt=0.01, period_multiplier=10
    → dt_vpm=0.1, FVM Co≈0.8 like the reference instead of Co≈11)."""

    bc_coupling_iterations: int = 1
    """Outer Picard iterations of the donor-velocity BC against the FVM pressure
    solve, per time step (Weymouth & Lauber, JCP 2025, arXiv:2404.09034).

    The Biot–Savart donor BC is non-locally coupled to the pressure: the pressure
    field generated inside the FVM box during the solve changes the interior
    vorticity, which changes the BS velocity the box boundary should see.  A
    single one-shot donor BC (the default, =1, the validated legacy path) ignores
    this and leaves the boundary inconsistent with the interior — which damps the
    global shedding feedback for sustained bluff-body wakes.

    With >1, each step iterates:  recompute donor = U∞ + BS(exterior particles)
    + BS(FVM interior vorticity)  →  ``solve_pimple()`` (no time advance)  →
    repeat, then ``advance_time()``.  The FVM-interior term is re-evaluated from
    the freshly-solved field each iteration, closing the BC↔pressure coupling
    without re-remeshing the cloud (which would inject extra diffusion).  2–3 is
    typically enough; the residual is bounded by the per-step pressure change."""

    donor_bc_mode: str = "dirichlet"
    """Type of donor velocity BC imposed on the FVM coupling patch
    (Billuart et al., JCP 2023, §3.1, Eqs. 11–14).

    ``"dirichlet"`` (default, the validated legacy path) imposes the full
    donor velocity vector as a Dirichlet condition
    (``set_dirichlet_velocity_boundary_condition``).  Any velocity mismatch
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

    ``"vorticity"`` (default, legacy) scatters ``ω_FVM · V_cell`` directly onto
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
    refines.  Default stays ``"vorticity"`` (the validated cubeFlow path); the
    velocity path is exercised by ``tests/coupler/test_continuous_overlap.py``."""

    body_panel_enabled: bool = False
    """Add a boundary-element (Hess-Smith panel) body model to the VPM so its
    induced field carries the body's IRROTATIONAL blockage — the no-penetration
    completion that free vortex particles structurally cannot represent (the
    diagnosed root cause of the wall-origin near-body error).  Opt-in; default
    off keeps the legacy path.  Requires ``panel_mesh`` (a coarse closed STL)."""

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

    # ── Samplers (optional) ───────────────────────────────────────────────
    samplers: list | None = None
    """List of (sampler, name) tuples written at ``log_period``."""

    def __post_init__(self) -> None:
        """Validate enum-like config fields."""
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
        if self.precision not in ("f32", "f64"):
            raise ValueError(f"precision must be 'f32' or 'f64', got {self.precision!r}.")

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
            "vpm_solver": {
                "particle_spacing": self.h,
                "buffer_thickness": self.buffer_thickness,
                "dead_zone_h": self.dead_zone_h,
                "h": self.h,
                "max_particles": self.max_particles,
                "particles_kernel": self.particles_kernel,
                "treecode_theta": self.treecode_theta,
                "advection_scheme": self.advection_scheme,
                "isr_enabled": self.isr_enabled,
                "isr_mode": self.isr_mode,
                "isr_cfl": self.isr_cfl,
                "isr_C": self.isr_C,
                "isr_k_max": self.isr_k_max,
                "vpm_domain": {
                    "xmin": self.vpm_domain[0],
                    "xmax": self.vpm_domain[1],
                    "ymin": self.vpm_domain[2],
                    "ymax": self.vpm_domain[3],
                    "zmin": self.vpm_domain[4],
                    "zmax": self.vpm_domain[5],
                },
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
                "donor_bc_mode": self.donor_bc_mode,
                "handoff_target_mode": self.handoff_target_mode,
                "les_smagorinsky_cs": self.les_smagorinsky_cs,
                "period_multiplier": self.period_multiplier,
                "precision": self.precision,
            },
        }
