"""
Types module for VPM solver.
==================
Types module for VPM solver. module.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

# =============================
# PRECISION CONFIGURATION
# =============================
from dataclasses import dataclass, field
import json

# Set traceback limit to 0 to avoid excessive output
import sys
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import constants as constants_module

sys.tracebacklimit = 0

# Import global constants
from .constants import (
    DEFAULT_BACKUP_FILENAME,
    DEFAULT_CUTOFF_RADIUS_FACTOR,
    DEFAULT_TIME_STEP,
    MAX_PARTICLES,
)


# =====================================================================================
# ADVECTION CONFIGURATION
# =====================================================================================
@dataclass
class AdvectionConfig:
    """
    Configuration for the advection term (dx/dt = u).
    """

    scheme: Literal["NONE", "EULER", "RK2", "RK3", "RK4"] = "RK2"
    """Time integration scheme for the advection substep (dx/dt = u).

    Options:
      - 'NONE':  freeze particle positions (useful for stationary-flow viscous tests)
      - 'EULER': forward Euler (1st order)
      - 'RK2':   Heun's method, 2nd-order Runge–Kutta (default)
      - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving
      - 'RK4':   classical 4th-order Runge–Kutta

    Advection advances the configured scheme over the macro time-step set by the
    solver (the DVH-pinned dt for DVH runs)."""


# =====================================================================================
# VISCOUS CONFIGURATION
# =====================================================================================


@dataclass
class ViscousConfig:
    """
    Configuration for viscous diffusion schemes.
    """

    scheme: Literal["CS", "RWM", "NONE", "DVH", "GBD"] = "CS"
    """Viscous diffusion scheme.
    'DVH' — DVH (Diffused Vortex Hydrodynamics, Durante et al. 2024):
                  each particle's circulation is spread to grid nodes via the
                  exact heat-kernel Gaussian; Shepard normalization enforces
                  per-particle Γ conservation.  Particles are then replaced
                  by surviving grid nodes (simultaneous diffusion + regularisation).
    'GBD' — Grid-Based Diffusion (Cottet & Koumoutsakos 2000):
                  CIC scatter to grid, explicit 7-point Laplacian diffusion,
                  threshold pruning + particle regeneration.  No inherent
                  lower bound on Δt (only an upper CFL bound h²/(6nu)).
    """

    def __post_init__(self) -> None:
        v_upper = self.scheme.upper()
        if v_upper == "RMW":
            v_upper = "RWM"
        if v_upper == "DVH_REGEN":
            v_upper = "DVH"
        if v_upper == "GBD_REGEN":
            v_upper = "GBD"
        if v_upper not in ("CS", "RWM", "NONE", "DVH", "GBD"):
            raise ValueError(f"Invalid viscous scheme: {self.scheme}")
        if v_upper != self.scheme:
            object.__setattr__(self, "scheme", v_upper)
        if v_upper == "DVH" and self.dvh_rd_ratio not in (3, 4, 5):
            raise ValueError(f"dvh_rd_ratio must be 3, 4, or 5 (got {self.dvh_rd_ratio})")
        _valid_modes = ("budget", "relative_max", "absolute")
        if self.dvh_threshold_mode not in _valid_modes:
            raise ValueError(
                f"dvh_threshold_mode must be one of {_valid_modes}, got {self.dvh_threshold_mode!r}"
            )
        if self.gbd_threshold_mode not in _valid_modes:
            raise ValueError(
                f"gbd_threshold_mode must be one of {_valid_modes}, got {self.gbd_threshold_mode!r}"
            )

    rwm_noise_amplitude: float = 1.0
    """Scaling factor for Random Walk noise."""

    regen_radius_ratio: float = 2.5
    """Core radius σ assigned to regenerated particles, in units of the regen
    grid spacing h (DVH and GBD schemes).  This SILENTLY OVERRIDES whatever
    radius the particles carried before the regen — in coupled FVM-VPM runs it
    MUST match the hand-off's ``overlap_radius_ratio``, otherwise the Beale
    strength correction deconvolves with the wrong kernel width and the
    reconstructed velocity field is over-smoothed (measured: σ=2.5h vs the
    corrected-for 1.5h costs ~4× in-box velocity error).  The coupler syncs
    this automatically.  Default 2.5 preserves legacy standalone behaviour."""

    # ---- Grid-Based Diffusion (DVH) parameters --------------------------------
    dvh_grid_spacing: float | None = None
    """Grid spacing h [m] for the DVH (DVH) scheme. Falls back to characteristic_distance
    if None."""

    dvh_domain_padding: float = 3.0
    """Grid extends this many grid-spacings beyond the particle bounding box on each
    side, so that vorticity decays to ~0 before reaching the Dirichlet boundary."""

    # ---- Grid-Based Diffusion / DVH (DVH) parameters -------
    dvh_threshold: float = 0.01
    """Circulation threshold for particle regeneration (DVH only).

    Interpretation depends on ``dvh_threshold_mode``:

    *  ``'budget'`` (default): keep the minimal set of grid nodes whose
       circulations collectively sum to at least ``(1 - threshold)`` of the
       total |Γ| in the domain.  With ``threshold=0.01`` at most 1% of total
       vorticity is discarded, regardless of the ratio between FVM-core and
       far-wake particle strengths.  This prevents the far wake from being
       silently wiped in coupled simulations.

    *  ``'relative_max'``: discard nodes where
       |Γ| < threshold × max|Γ|.  Problematic in coupled runs because the
       FVM-core maximum dominates and all weak far-wake nodes are pruned.

    *  ``'absolute'``: discard nodes where |Γ| < threshold (absolute value
       in [m³/s]).  This is the preferred mode for controlling particle count
       when the dynamic range of circulation spans many orders of magnitude.
       Typical values: 1e-4 to 1e-2 for unit-circulation problems."""

    dvh_threshold_mode: str = "budget"
    """How ``dvh_threshold`` is interpreted in DVH mode.
    ``'budget'``       — keep the top-(1-threshold) fraction of total |Γ| sum
                         (recommended for coupled FVM-VPM simulations).
    ``'relative_max'`` — keep nodes above threshold × global max|Γ|.
    ``'absolute'``     — keep nodes above the absolute circulation value
                         ``dvh_threshold`` [m³/s]."""

    # ---- Grid-Based Diffusion (GBD) parameters --------------------------------
    gbd_grid_spacing: float | None = None
    """Grid spacing h [m] for GBD.  Falls back to characteristic_distance if None."""

    gbd_domain_padding: float = 3.0
    """Grid extends this many grid-spacings beyond the particle bounding box."""

    gbd_threshold: float = 0.01
    """Circulation threshold for GBD particle regeneration.
    Interpretation depends on ``gbd_threshold_mode`` (same semantics as DVH)."""

    gbd_threshold_mode: str = "budget"
    """How ``gbd_threshold`` is interpreted in GBD mode.
    ``'budget'``       — keep the top-(1-threshold) fraction of total |Γ| sum.
    ``'relative_max'`` — keep nodes above threshold × global max|Γ|.
    ``'absolute'``     — keep nodes above the absolute circulation value
                         ``gbd_threshold`` [m³/s]."""

    dvh_rd_ratio: int = 4
    """Ratio R_d/h for the DVH scatter kernel (DVH only).  Integer in [3, 5].

    R_d is the compact-support radius within which a particle's circulation
    is redistributed to grid nodes.  Durante et al. (2024, Section 4.2)
    show that R_d/h = 4 gives optimal convergence (monotone, ≈1% error at
    h/σ = 0.8).  Smaller values (3) may converge non-monotonically; larger
    values (5) are more conservative but increase the scatter stencil.

    Acceptable range: 3, 4, or 5.  Default: 4.

    Because the DVH Gaussian width is tied to R_d (via β·R_d²), changing
    this ratio also changes the required time-step size Δt_d.
    """

    dvh_max_nodes: int | None = None
    """Hard cap on surviving grid nodes per DVH regen (DVH only).

    Bounds the budget-mode halo growth: with threshold_mode='budget' the
    heat-kernel tail keeps adding weak nodes every firing (measured: +45k
    nodes/firing on the ring tutorial).  Capping at the strongest N nodes is
    a budget-by-count — the dropped tail carries a negligible circulation
    fraction while the particle count (and treecode cost) stays bounded.
    None (default) keeps only the built-in safety cap (3×N, ≤ MAX_PARTICLES)."""

    viscosity: float | None = None
    """Molecular kinematic viscosity nu [m²/s].

    When set, every new particle added to the solver automatically receives
    this molecular viscosity.  For DVH, together with ``dvh_grid_spacing``
    and ``dvh_rd_ratio``, determines the DVH-required time-step
    Δt_d = β·R_d²/(4nu).  The solver pins dt = Δt_d (DVH fires exactly once
    per step); a user/coupler dt differing from Δt_d is overridden to Δt_d
    so the diffusion operator acts on every step with the correct increment.

    For GBD, the CFL upper bound is dt ≤ h²/(6nu).  The solver warns
    if the user-supplied dt exceeds this limit."""
    # ---------------------------------------------------------------------------

    characteristic_distance: float | None = None
    """Average inter-particle spacing h [m].

    When set together with ``viscosity``, the solver checks whether the
    configured time-step satisfies the stability criterion for the active
    viscous scheme and prints a warning if not.  Required for the Core
    Spreading (CS) and Random Walk Method (RWM) stability checks."""

    @field_validator("characteristic_distance")
    @classmethod
    def validate_characteristic_distance(cls, v):
        if v is None:
            return v
        if v <= 0:
            raise ValueError("characteristic_distance must be positive")
        return v

    def cs_max_dt(self) -> float:
        """Stability upper bound for Core Spreading: Δt ≤ h²/(4nu).

        The CS diffusion step integrates the parabolic heat equation with a
        Gaussian kernel of width √(4nuΔt).  The criterion h²/(4nu) ensures the
        diffusive step does not advance vortex cores by more than one
        inter-particle spacing per step (parabolic CFL condition).
        Larger Δt leads to over-diffusion and loss of spatial resolution.

        Returns
        -------
        float  Maximum stable time-step [s].

        Raises
        ------
        ValueError  If ``characteristic_distance`` or ``viscosity`` is not set.
        """
        h = self.characteristic_distance
        if h is None or h <= 0:
            raise ValueError(
                "characteristic_distance must be set to a positive value for CS stability check."
            )
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for CS stability check.")
        return h * h / (4.0 * nu)

    def rwm_accuracy_dt(self) -> float:
        """Accuracy upper bound for the Random Walk Method: Δt ≤ h²/(4nu).

        The RWM is unconditionally stable (stochastic Monte Carlo), but accuracy
        degrades when each particle's random displacement √(2nuΔt) exceeds the
        inter-particle spacing h.  The bound h²/(4nu) keeps displacements within
        ≈ h/√2, ensuring diffused vorticity remains well-resolved on the particle
        grid.  Beyond this limit the cloud diffuses faster than the grid can
        represent, causing artificial smoothing of vorticity gradients.

        Returns
        -------
        float  Recommended maximum time-step for RWM accuracy [s].

        Raises
        ------
        ValueError  If ``characteristic_distance`` or ``viscosity`` is not set.
        """
        h = self.characteristic_distance
        if h is None or h <= 0:
            raise ValueError(
                "characteristic_distance must be set to a positive value for RWM accuracy check."
            )
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for RWM accuracy check.")
        return h * h / (4.0 * nu)

    def dvh_required_dt(self) -> float:
        """Compute the DVH-required time-step size Δt_d = β·R_d²/(4nu).

        The DVH Gaussian width 4nu·Δt_d = β·R_d² must span several grid
        cells so that the scatter distributes circulation meaningfully
        across ~R_d³ nodes.  The solver pins dt = Δt_d so each step applies
        exactly one full diffusive increment (DVH fires once per step).

        Uses ``self.viscosity``, ``self.dvh_grid_spacing``, and
        ``self.dvh_rd_ratio``.

        Returns
        -------
        float  Required time-step size [s].
        """
        from ..physics.diffusion import _DVH_BETA

        h = self.dvh_grid_spacing
        if h is None or h <= 0:
            raise ValueError("dvh_grid_spacing must be set to a positive value.")
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for DVH.")
        R_d = self.dvh_rd_ratio * h
        return _DVH_BETA * R_d * R_d / (4.0 * nu)

    def gbd_max_dt(self) -> float:
        """CFL upper bound for GBD: dt_max = h² / (6nu).

        Returns
        -------
        float  Maximum stable time-step [s].
        """
        h = self.gbd_grid_spacing
        if h is None or h <= 0:
            raise ValueError("gbd_grid_spacing must be set to a positive value.")
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for GBD.")
        return h * h / (6.0 * nu)

    @staticmethod
    def cs(
        viscosity: float | None = None,
        characteristic_distance: float | None = None,
    ) -> "ViscousConfig":
        """Core Spreading method (deterministic, varying core size).

        Args:
            viscosity: Molecular kinematic viscosity nu [m²/s].  When set,
                every new particle automatically receives this viscosity.
            characteristic_distance: Average inter-particle spacing h [m].
                When set together with ``viscosity``, the solver prints the
                CS stability limit Δt ≤ h²/(4nu) in the initialisation header
                and warns if the configured time-step exceeds it.
        """
        return ViscousConfig(
            scheme="CS", viscosity=viscosity, characteristic_distance=characteristic_distance
        )

    @staticmethod
    def rwm(
        viscosity: float | None = None,
        characteristic_distance: float | None = None,
    ) -> "ViscousConfig":
        """Random Walk Method (stochastic).

        Args:
            viscosity: Molecular kinematic viscosity nu [m²/s].  When set,
                every new particle automatically receives this viscosity.
            characteristic_distance: Average inter-particle spacing h [m].
                When set, the solver prints the RWM accuracy limit Δt ≤ h²/(4nu)
                in the initialisation header.
        """
        return ViscousConfig(
            scheme="RWM", viscosity=viscosity, characteristic_distance=characteristic_distance
        )

    @staticmethod
    def inviscid():
        """No viscous diffusion."""
        return ViscousConfig(scheme="NONE")

    @staticmethod
    def dvh(
        h: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        dvh_rd_ratio: int = 4,
        viscosity: float | None = None,
        max_nodes: int | None = None,
    ) -> "ViscousConfig":
        """DVH (Diffused Vortex Hydrodynamics) with particle regeneration.

        Implements Durante et al. (2024) DVH algorithm: each particle's
        circulation is spread to nearby grid nodes using the exact heat-kernel
        Gaussian w = exp(-r²/(4nuΔt)).  Shepard normalization enforces exact
        per-particle Γ conservation.  The particle set is then replaced by
        surviving grid nodes (diffusion + regularisation in one step).  No
        finite-difference solve or CFL constraint is involved.

        The DVH Gaussian width is β·R_d² = 4nu·Δt_d, which constrains the
        simulation time-step to Δt = Δt_d.  If the user-supplied Δt differs,
        the solver will override it and print a warning.

        Args:
            h: Grid spacing [m].  Should match the desired inter-particle
                spacing after regeneration.  If None, falls back to
                characteristic_distance.
            padding: Cell-widths of padding beyond the bounding box.
                Default 20.0 — ensures vorticity at the boundary is
                negligible as the particle cloud evolves.
            threshold: Circulation threshold for pruning (see threshold_mode).
                Default 1e-5 — calibrated on the Lamb-Oseen analytic benchmark
                (tutorials/VPM/lambOseenVortex/assets/sweep_viscous.py):
                budget 1e-5 gives <0.001% circulation drift and <0.5% effective
                diffusion-rate error; budget 1e-3 loses ~5% of Σ|Γ| and 13% of
                the diffusion rate over ~150 firings.
            threshold_mode: How ``threshold`` is applied.
                ``'budget'`` (default) — keep the minimal set of nodes whose
                  collective |Γ| sum ≥ (1 − threshold) of total.  The loss per
                  firing is bounded by construction.
                ``'relative_max'`` — discard nodes below threshold × max|Γ|.
                ``'absolute'`` — discard nodes below threshold [m³/s].
                  WARNING: unbounded loss per firing — measured to destroy
                  ~1.2%/firing of the vortex-ring tutorial's circulation
                  (total evaporation) and −26% on the Lamb-Oseen benchmark.
                  Only justified where the far field must be truncated
                  aggressively (e.g. coupled FVM-VPM with a small VPM box)
                  and the loss is monitored.
            dvh_rd_ratio: R_d/h compact-support radius ratio for the DVH
                heat-kernel.  Integer in [3, 5].  Default 4
                (optimal, Durante 2024 Sec. 4.2).
            viscosity: Molecular kinematic viscosity nu [m²/s].  Required
                for DVH — automatically assigned to every new particle and
                determines the DVH time-step Δt_d = β·R_d²/(4nu).
            max_nodes: Hard cap on surviving regen nodes (budget-by-count) —
                bounds the budget-mode halo growth.  None = built-in cap only.
        """
        if not isinstance(dvh_rd_ratio, int) or dvh_rd_ratio not in (3, 4, 5):
            raise ValueError(
                f"dvh_rd_ratio must be an integer in {{3, 4, 5}}, got {dvh_rd_ratio!r}"
            )
        return ViscousConfig(
            scheme="DVH",
            dvh_grid_spacing=h,
            dvh_domain_padding=padding,
            dvh_threshold=threshold,
            dvh_threshold_mode=threshold_mode,
            dvh_rd_ratio=dvh_rd_ratio,
            viscosity=viscosity,
            dvh_max_nodes=max_nodes,
        )

    @staticmethod
    def gbd(
        h: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        viscosity: float | None = None,
    ) -> "ViscousConfig":
        """Grid-Based Diffusion (Cottet & Koumoutsakos 2000).

        M4' (Monaghan 1985) remeshing scatter → explicit 7-point Laplacian →
        threshold pruning with Γ-conservation → particle regeneration.
        No inherent lower bound on Δt; only an upper CFL bound dt ≤ h²/(6nu).

        Threshold default 1e-5 (budget) calibrated on the Lamb-Oseen analytic
        benchmark: −0.18% circulation drift, −0.33% diffusion-rate error; the
        previous 1e-4 lost ~1% of Σ|Γ| and 4% of the diffusion rate
        (assets/sweep_viscous.py, tutorials/VPM/lambOseenVortex).

        Because GBD fires every step (unlike DVH which uses a large Δt_d),
        the budget threshold must be much tighter (default 0.0001 = 0.01%)
        to avoid compound circulation loss over hundreds of steps.

        Args:
            h: Grid spacing [m].
            padding: Cell-widths of padding beyond the bounding box.
            threshold: Circulation threshold for pruning (default 0.0001).
            threshold_mode: ``'budget'``, ``'relative_max'``, or ``'absolute'``.
                ``'absolute'`` uses ``threshold`` as a raw circulation magnitude
                in [m³/s] and is the preferred mode for controlling particle
                count in simulations with large dynamic range.
            viscosity: Molecular kinematic viscosity nu [m²/s].
        """
        return ViscousConfig(
            scheme="GBD",
            gbd_grid_spacing=h,
            gbd_domain_padding=padding,
            gbd_threshold=threshold,
            gbd_threshold_mode=threshold_mode,
            viscosity=viscosity,
        )


@dataclass
class StretchingConfig:
    """
    Configuration for vortex stretching schemes.

    Uses direct pair-wise computation (O(N²)) for guaranteed circulation conservation,
    or a local gradU-based approach (O(N)) that reuses the pre-computed velocity
    gradient tensor.

    Modes:
        - TRANSPOSED: dΓ/dt = (Γ·∇')u - direct O(N²), conserves ΣΓ
        - CLASSICAL: dΓ/dt = (Γ·∇)u - direct O(N²), standard formulation
        - MIXED: Strain-based symmetric formulation - direct O(N²)
        - GRADU: dΓ/dt = (∇u)ᵀ·Γ using pre-computed gradU - local O(N)
          Requires velocity gradients computed beforehand (automatic in the
          standard solver loop).  Much cheaper than direct modes but can be
          less stable for large dt or highly strained flows because gradU is
          frozen at t_n for the entire RK sub-stepping.

    Examples:
          # Transposed stretching (conserves circulation, recommended)
          stretching = StretchingConfig.transposed()

          # Classical stretching
          stretching = StretchingConfig.classical()

          # Mixed/strain stretching
          stretching = StretchingConfig.mixed()

          # GradU-based stretching (O(N), experimental)
          stretching = StretchingConfig.gradu()

          # Disabled stretching
          stretching = StretchingConfig.disabled()
    """

    mode: Literal["CLASSICAL", "TRANSPOSED", "MIXED", "GRADU", "RVPM"] = "TRANSPOSED"
    """Stretching formulation mode: CLASSICAL, TRANSPOSED, MIXED, GRADU, or RVPM."""

    scheme: Literal["EULER", "RK2", "RK3", "RK4"] = "RK2"
    """Time integration scheme for the stretching substep (dΓ/dt).

    Options:
      - 'EULER': forward Euler (1st order)
      - 'RK2':   Heun's method, 2nd-order Runge–Kutta (default)
      - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving
      - 'RK4':   classical 4th-order Runge–Kutta
    """

    enabled: bool = True
    """Whether vortex stretching is enabled."""

    rvpm_f: float = 0.0
    """rVPM reformulation parameter f (mode='RVPM' only, Alvarez & Ning).
    Together with g sets c_r = (g+f)/(1/3+f) — the fraction of the parallel
    stretching rate removed from the strength magnitude — and
    c_σ = (g+f)/(1+3f) — the rate at which the core size absorbs it."""

    rvpm_g: float = 0.2
    """rVPM reformulation parameter g (mode='RVPM' only).  Default 1/5 (the
    Alvarez & Ning value, giving c_r = 3/5 with f=0): |Γ| then grows at
    (1−c_r)·S_∥ = (2/5)·S_∥ while σ shrinks at (1/5)·S_∥, conserving the
    element volume measure σ²·|Γ|.  Set g=1/3, f=0 (c_r=1) to fully suppress
    parallel growth (maximum stability)."""

    @staticmethod
    def classical(scheme: str = "RK2"):
        """Classical scheme: dΓ/dt = (Γ·∇)u"""
        return StretchingConfig(mode="CLASSICAL", scheme=scheme)

    @staticmethod
    def transposed(scheme: str = "RK2"):
        """Transposed scheme: dΓ/dt = (Γ·∇')u - conserves ΣΓ"""
        return StretchingConfig(mode="TRANSPOSED", scheme=scheme)

    @staticmethod
    def mixed(scheme: str = "RK2"):
        """Mixed/strain scheme: symmetric formulation"""
        return StretchingConfig(mode="MIXED", scheme=scheme)

    @staticmethod
    def gradu(scheme: str = "RK2"):
        """GradU-based stretching: dΓ/dt = (∇u)ᵀ·Γ using pre-computed velocity gradients.

        O(N) per sub-step (vs O(N²) for direct modes).  Requires that
        ``compute_velocity_gradients`` has been called before the stretching step.

        .. warning::
            Experimental — can be less stable than direct modes when dt is large
            or when the flow has strong velocity gradients, because ∇u is frozen
            at the beginning of the time step during RK sub-stepping.
        """
        return StretchingConfig(mode="GRADU", scheme=scheme)

    @staticmethod
    def rvpm(scheme: str = "RK2", f: float = 0.0, g: float = 0.2):
        """Reformulated-VPM stretching (Alvarez & Ning):

            dΓ/dt = (∇u)ᵀΓ − c_r·(Γ̂·(∇u)ᵀΓ)·Γ̂,    c_r = (g+f)/(1/3+f)
            dσ/dt = −c_σ·σ·S_∥,                      c_σ = (g+f)/(1+3f)

        The parallel stretching growth of |Γ| is reduced to (1−c_r)·S_∥ and the
        removed growth is absorbed by core-size contraction so that the vortex
        element's volume measure σ²·|Γ| is conserved.  Defaults (f=0, g=1/5)
        give c_r = 3/5, c_σ = 1/5 — the Alvarez & Ning values.  Like GRADU,
        this is a local O(N) operator on the pre-computed velocity gradients.
        """
        return StretchingConfig(mode="RVPM", scheme=scheme, rvpm_f=f, rvpm_g=g)

    @staticmethod
    def disabled():
        return StretchingConfig(enabled=False)


# ForceConfig is imported from vlm_solver to ensure consistency and avoid duplication
# as it is primarily a property of the VLM-VPM interaction.
from ..boundary_elements.vlm.solver.vlm_solver import ForceConfig


# =====================================================================================
# TURBULENCE CONFIGURATION
# =====================================================================================
@dataclass
class TurbulenceConfig:
    """
    Configuration for turbulence modeling in VPM.

    Supports multiple turbulence models for different simulation requirements:

    **Models:**
    - DNS: Direct Numerical Simulation (no subgrid scale modeling)
    - LES_SMAGORINSKY: Classical static Smagorinsky eddy viscosity model
    - INVISCID: No viscous diffusion or SGS model — pure stretching

    **Usage Examples:**
          # DNS (inviscid or with molecular viscosity only)
          config = SolverConfig(turbulence=TurbulenceConfig.dns())

          # Static Smagorinsky LES
          config = SolverConfig(turbulence=TurbulenceConfig.les_smagorinsky(cs=0.17))

          # Inviscid — pure stretching, stabilised with ISR
          config = SolverConfig(
              turbulence=TurbulenceConfig.inviscid(),
              viscous=ViscousConfig.inviscid(),
              stabilization=StabilizationConfig.strength_relaxation(),
          )
    """

    model: Literal["DNS", "LES_SMAGORINSKY", "INVISCID"] = "DNS"
    """
      Turbulence model selection.

      Options:
            - 'DNS': Direct Numerical Simulation (viscous only, no subgrid modeling)
            - 'LES_SMAGORINSKY': Static Smagorinsky eddy viscosity model
            - 'INVISCID': No viscous diffusion or SGS model — pure stretching only
      """

    cs: float = constants_module.SMAGORINSKY_CONSTANT
    """
      Classical Smagorinsky constant C_s (dimensionless).

      This is the standard user-facing constant known from grid-based LES.
      The k-equilibrium model derives the internal kinetic-energy coefficient
      C_k from C_s via:  C_k = (C_s² · √C_e)^(2/3)

      Used by: LES_SMAGORINSKY k-equilibrium model

      Typical values:
            - 0.17:  Classical Lilly (1966) value (default)
            - 0.10:  Low-dissipation / near-wall flows
            - 0.20:  High-dissipation / coarse grids

      Relationship to C_k:  C_s² = C_k^(3/2) / √C_e
      """

    ce: float = 1.048
    """
      Kolmogorov dissipation constant C_e (dimensionless).

      Used by: LES_SMAGORINSKY k-equilibrium model

      Default: 1.048 (OpenFOAM value, derived from Lilly 1966 spectral analysis)

      Physics: SGS dissipation rate ε = C_e * k^(3/2) / Δ
      Together with C_k determines C_s² = C_k * √(C_k / C_e).
      """

    flow_model: str = "DNS"
    """
      Associated flow physics model.

      Automatically set by model selection:
            - DNS model → 'DNS' flow
            - LES_* models → 'LES' flow
            - INVISCID model → 'INVISCID' flow

      Not manually configured (read-only reference).
      """

    @staticmethod
    def dns() -> "TurbulenceConfig":
        """
        Create DNS (Direct Numerical Simulation) configuration.

        **Physics:** Computes only molecular viscosity diffusion.
        No subgrid-scale turbulence modeling applied.

        **Use when:**
        - Sufficient grid resolution to capture all scales
        - Studying laminar or transitional flows
        - Molecular viscosity is the only dissipation mechanism

        **Note:** Can still use viscous schemes (CS, RWM) for diffusion.

        Returns:
            TurbulenceConfig: DNS configuration instance

        Example:
            >>> config = SolverConfig(
            ...     turbulence=TurbulenceConfig.dns(),
            ...     viscous=ViscousConfig.cs()  # With core spreading diffusion
            ... )
        """
        return TurbulenceConfig(model="DNS", flow_model="DNS")

    @staticmethod
    def les_smagorinsky(
        cs: float = constants_module.SMAGORINSKY_CONSTANT,
        ce: float = 1.048,
    ) -> "TurbulenceConfig":
        """
        Create kinetic-energy equilibrium Smagorinsky LES configuration.

        **Physics:** Eddy viscosity from SGS kinetic energy equilibrium.
        Local production–dissipation balance gives:

            k_eq = C_k · Δ² · |S|² / C_e,    nu_t = C_k · Δ · √k_eq

        where the internal coefficient C_k is derived from the user-supplied
        classical Smagorinsky constant C_s:  C_k = (C_s² · √C_e)^(2/3)

        This ensures C_s² = C_k^(3/2) / √C_e, i.e. the model is exactly
        equivalent to the standard nu_t = (C_s Δ)² |S| formulation.

        **Args:**
            cs: Classical Smagorinsky constant C_s (dimensionless, default 0.17).
                Typical: 0.10 (low dissipation) – 0.20 (high dissipation).
            ce: Kolmogorov dissipation constant C_e (default 1.048, Lilly 1966).

        **Notes:**
            - cs=0.17, ce=1.048 → C_k ≈ 0.096
            - cs=0.16, ce=1.048 → C_k ≈ 0.088
            - OpenFOAM-equivalent: C_k=0.094 → C_s ≈ 0.168

        Returns:
            TurbulenceConfig: LES_SMAGORINSKY configuration instance
        """
        return TurbulenceConfig(
            model="LES_SMAGORINSKY",
            cs=cs,
            ce=ce,
            flow_model="LES",
        )

    @staticmethod
    def inviscid() -> "TurbulenceConfig":
        """
        Create INVISCID configuration — pure stretching only.

        **Physics:** Only the vortex-stretching term (ω·∇)u is solved.
        No SGS eddy viscosity, no molecular diffusion, no turbulence model.

        **Use when:**
        - Testing stretching stabilisation schemes in isolation.
        - Running inviscid validation sweeps.
        - Comparing damping/regularisation methods without LES or CS.

        **Note:** Pass ``StabilizationConfig.strength_relaxation()`` to
        ``SolverConfig.stabilization`` to stabilize long inviscid runs.

        Returns:
            TurbulenceConfig: INVISCID configuration instance
        """
        return TurbulenceConfig(model="INVISCID", flow_model="INVISCID")


# =====================================================================================
# STABILIZATION CONFIGURATION
# =====================================================================================


@dataclass
class StabilizationConfig:
    """
    Unified configuration for all solution-stabilization mechanisms.

    Every mechanism that modifies the particle field to maintain accuracy
    or prevent instability lives in this single class.

    Particle splitting
    ------------------
    Splits particles whose core radius exceeds ``max_core_radius`` into two
    daughters offset perpendicular to the vorticity direction (transverse
    redistribution conserves the mollified field's kinetic energy through
    the split; an axial offset cannot).  Children inherit the parent's
    group and zone IDs exactly, so group-based diagnostics remain correct.

    Conservative remeshing
    ----------------------
    Periodically remeshes the particle field onto a regular grid to maintain
    spatial overlap.  Impulse correction enforces exact conservation of linear
    (and optionally angular) impulse across every remesh cycle.

    Examples
    --------
    .. code-block:: python

        # Default: no active stabilisation
        stab = StabilizationConfig()

        # Particle splitting
        stab = StabilizationConfig(max_core_radius=0.12)

        # Periodic remeshing
        stab = StabilizationConfig.conservative_remeshing(frequency=20, spacing=0.03)

        # Splitting + remeshing
        stab = StabilizationConfig(max_core_radius=0.12,
                                   remeshing_frequency=20, remeshing_spacing=0.03)
    """

    # ── Particle splitting ─────────────────────────────────────────────────────
    max_core_radius: float | None = None
    """Split particles whose core radius exceeds this value [m].  None = disabled."""

    # ── Wake / bounds cutoff ──────────────────────────────────────────────────
    remove_particles_by_bounds: list[float] | None = None
    """[xmin, xmax, ymin, ymax, zmin, zmax] — remove particles outside box.  None = disabled."""

    # ── Weak-particle removal ─────────────────────────────────────────────────
    weak_threshold_percent: float | None = None
    """Remove particles with |Γ| < weak_threshold_percent% of the group maximum.  None = disabled."""

    per_group: bool = True
    """Apply the weak-particle threshold independently per group (True, default) or globally."""

    # ── Conservative remeshing ────────────────────────────────────────────────
    remeshing_frequency: int | None = None
    """Remesh every N steps.  None = disabled."""

    remeshing_spacing: float | None = None
    """Grid spacing [m] for remeshing.  None → uses mean particle spacing."""

    remeshing_bounds: list[float] | None = None
    """[xmin, xmax, ymin, ymax, zmin, zmax] for the remeshing domain.  None = auto."""

    remeshing_relative_threshold: float = 0.05
    """Relative vorticity threshold for delta injection (5% default)."""

    remeshing_absolute_threshold: float = 1e-6
    """Absolute vorticity threshold [1/s] for delta injection."""

    remeshing_conserve_impulse: bool = True
    """Apply post-remesh correction to conserve linear/angular impulse."""

    remeshing_delta_correction: bool = False
    """Inject delta-correction particles for sub-grid residuals.  Disabled by default."""

    remeshing_impulse_constraint: str = "3d"
    """"3d" — full 3-component correction; "z" — z-only (for quasi-2-D flows)."""

    remeshing_radius: float | None = None
    """Core radius [m] for remeshed particles.  None → spacing × RADIUS_RATIO."""

    remeshing_project_solenoidal: bool = False
    """Apply a Helmholtz projection to grid vorticity during conservative remeshing."""

    remeshing_projection_padding: int = 4
    """Zero-padding cells used by the isolated-domain FFT projection."""

    # ── Strength relaxation ──────────────────────────────────────────────
    relaxation_enabled: bool = False
    """Enable strength-relaxation stabilization of vortex stretching."""

    relaxation_mode: Literal["blend", "pedrizzetti"] = "blend"
    """Relaxation update: residual filtering (``blend``) or direction
    realignment at fixed |Gamma| (``pedrizzetti``)."""

    relaxation_deconv: int = 1
    """Van Cittert approximate-deconvolution iterations (0 through 3)."""

    relaxation_gate: Literal["strain", "constant"] = "strain"
    """Use a local strain-rate gate or a constant relaxation factor."""

    relaxation_factor: float = 0.3
    """Constant relaxation factor used with the constant gate."""

    relaxation_conserve: bool = True
    """Restore configured circulation/impulse invariants after relaxation."""

    relaxation_constraint: Literal["both", "sum", "linear"] = "both"
    """Invariants restored when conservation is enabled."""

    relaxation_rate: float = 1.0
    """Strain-gate rate constant."""

    relaxation_seff_min: float = 1e-4
    """Skip strain-gated corrections below this effective-strain increment."""

    relaxation_verbose: bool = False
    """Collect per-step strength-relaxation diagnostics."""

    relaxation_cfl: float = 0.2
    """Target effective-strain increment per relaxation sub-step."""

    relaxation_max_substeps: int = 8
    """Maximum number of relaxation sub-steps per VPM step."""

    def __post_init__(self) -> None:
        """Validate direct construction as well as factory-created configs."""
        if self.max_core_radius is not None and self.max_core_radius <= 0:
            raise ValueError("max_core_radius must be positive")
        if self.remove_particles_by_bounds is not None and len(self.remove_particles_by_bounds) != 6:
            raise ValueError("remove_particles_by_bounds must have 6 elements")
        if self.weak_threshold_percent is not None and not 0 <= self.weak_threshold_percent <= 100:
            raise ValueError("weak_threshold_percent must be between 0 and 100")
        if self.remeshing_frequency is not None and self.remeshing_frequency <= 0:
            raise ValueError("remeshing_frequency must be positive")
        if self.remeshing_spacing is not None and self.remeshing_spacing <= 0:
            raise ValueError("remeshing_spacing must be positive")
        if self.remeshing_bounds is not None and len(self.remeshing_bounds) != 6:
            raise ValueError("remeshing_bounds must have 6 elements")
        if not 0 <= self.remeshing_relative_threshold <= 1:
            raise ValueError("remeshing_relative_threshold must be in [0, 1]")
        if self.remeshing_impulse_constraint not in ("3d", "z"):
            raise ValueError("remeshing_impulse_constraint must be '3d' or 'z'")
        if self.remeshing_radius is not None and self.remeshing_radius <= 0:
            raise ValueError("remeshing_radius must be positive")
        if self.remeshing_projection_padding < 0:
            raise ValueError("remeshing_projection_padding must be non-negative")
        if self.relaxation_mode not in ("blend", "pedrizzetti"):
            raise ValueError("relaxation_mode must be 'blend' or 'pedrizzetti'")
        if self.relaxation_gate not in ("strain", "constant"):
            raise ValueError("relaxation_gate must be 'strain' or 'constant'")
        if self.relaxation_constraint not in ("both", "sum", "linear"):
            raise ValueError("relaxation_constraint must be 'both', 'sum', or 'linear'")
        if not 0 <= self.relaxation_deconv <= 3:
            raise ValueError("relaxation_deconv must be between 0 and 3")
        if not 0 <= self.relaxation_factor <= 1:
            raise ValueError("relaxation_factor must be between 0 and 1")
        if self.relaxation_rate < 0:
            raise ValueError("relaxation_rate must be non-negative")
        if self.relaxation_seff_min < 0:
            raise ValueError("relaxation_seff_min must be non-negative")
        if self.relaxation_cfl <= 0:
            raise ValueError("relaxation_cfl must be positive")
        if self.relaxation_max_substeps < 1:
            raise ValueError("relaxation_max_substeps must be at least 1")

    # ── Factory methods ───────────────────────────────────────────────────────
    @staticmethod
    def disabled() -> "StabilizationConfig":
        """No stabilisation mechanisms active."""
        return StabilizationConfig()

    @staticmethod
    def particle_splitting(
        radius: float,
        weak_threshold_percent: float = 1.0,
    ) -> "StabilizationConfig":
        """Particle splitting when core radius exceeds ``radius``."""
        if radius <= 0:
            raise ValueError("max_core_radius must be positive")
        return StabilizationConfig(
            max_core_radius=radius,
            weak_threshold_percent=weak_threshold_percent,
        )

    @staticmethod
    def conservative_remeshing(
        frequency: int = 20,
        spacing: float | None = None,
        bounds: list[float] | None = None,
        relative_threshold: float = 0.01,
        absolute_threshold: float = 1e-6,
        conserve_impulse: bool = True,
        delta_correction: bool = False,
        impulse_constraint: str = "3d",
        radius: float | None = None,
        project_solenoidal: bool = False,
        projection_padding: int = 4,
    ) -> "StabilizationConfig":
        """Periodic conservative remeshing."""
        if frequency <= 0:
            raise ValueError("remeshing frequency must be positive")
        if spacing is not None and spacing <= 0:
            raise ValueError("remeshing spacing must be positive")
        if bounds is not None and len(bounds) != 6:
            raise ValueError("remeshing bounds must have 6 elements")
        if not 0 <= relative_threshold <= 1:
            raise ValueError("relative_threshold must be in [0, 1]")
        if impulse_constraint not in ("3d", "z"):
            raise ValueError("impulse_constraint must be '3d' or 'z'")
        if radius is not None and radius <= 0:
            raise ValueError("remeshing_radius must be positive")
        if projection_padding < 0:
            raise ValueError("projection_padding must be non-negative")
        return StabilizationConfig(
            remeshing_frequency=frequency,
            remeshing_spacing=spacing,
            remeshing_bounds=bounds,
            remeshing_relative_threshold=relative_threshold,
            remeshing_absolute_threshold=absolute_threshold,
            remeshing_conserve_impulse=conserve_impulse,
            remeshing_delta_correction=delta_correction,
            remeshing_impulse_constraint=impulse_constraint,
            remeshing_radius=radius,
            remeshing_project_solenoidal=project_solenoidal,
            remeshing_projection_padding=projection_padding,
        )

    @staticmethod
    def strength_relaxation(
        mode: Literal["blend", "pedrizzetti"] = "blend",
        *,
        deconv: int = 1,
        gate: Literal["strain", "constant"] = "strain",
        factor: float = 0.3,
        conserve: bool = True,
        constraint: Literal["both", "sum", "linear"] = "both",
        rate: float = 1.0,
        seff_min: float = 1e-4,
        verbose: bool = False,
        cfl: float = 0.2,
        max_substeps: int = 8,
    ) -> "StabilizationConfig":
        """Enable strength relaxation with its complete set of controls."""
        return StabilizationConfig(
            relaxation_enabled=True,
            relaxation_mode=mode,
            relaxation_deconv=deconv,
            relaxation_gate=gate,
            relaxation_factor=factor,
            relaxation_conserve=conserve,
            relaxation_constraint=constraint,
            relaxation_rate=rate,
            relaxation_seff_min=seff_min,
            relaxation_verbose=verbose,
            relaxation_cfl=cfl,
            relaxation_max_substeps=max_substeps,
        )


# =====================================================================================
# VELOCITY CONFIGURATION
# =====================================================================================
@dataclass
class VelocityConfig:
    """
    Configuration for velocity field computation.

    Controls whether to use direct O(N²) summation or hierarchical O(N log N)
    treecode acceleration for computing particle-induced velocities.

    Examples:
          # Direct computation (default, exact but O(N²))
          velocity = VelocityConfig.direct()

          # Barnes-Hut treecode (approximate, O(N log N))
          velocity = VelocityConfig.treecode(theta=0.5)  # ~5% error, 100x+ speedup
          velocity = VelocityConfig.treecode(theta=0.3)  # ~2% error, moderate speedup

    Performance Guide:
          - N < 5000: Direct method is typically faster (GPU parallelism)
          - N > 5000: Treecode provides significant speedup
          - N > 50000: Treecode is essential for reasonable runtimes
    """

    method: Literal["DIRECT", "TREECODE"] = "DIRECT"
    """Velocity computation method: 'DIRECT' (O(N²)) or 'TREECODE' (O(N log N))."""

    theta: float = 0.5
    """Opening angle for Barnes-Hut treecode. Only used when method='TREECODE'.
      Smaller = more accurate but slower.
      - θ = 0.3: ~2% error, slower
      - θ = 0.5: ~5% error (recommended)
      - θ = 0.7: ~15% error, faster"""

    @staticmethod
    def direct() -> "VelocityConfig":
        """Direct O(N²) pairwise velocity computation (exact)."""
        return VelocityConfig(method="DIRECT")

    @staticmethod
    def treecode(theta: float = 0.5) -> "VelocityConfig":
        """Barnes-Hut treecode O(N log N) velocity computation (approximate).

        Args:
              theta: Opening angle parameter (0.3-0.7). Smaller = more accurate.
        """
        return VelocityConfig(method="TREECODE", theta=theta)


# =====================================================================================
# VLM SOLVER CONFIGURATION
# =====================================================================================
@dataclass
class VLMSolverConfig:
    """Configuration for VLM coupling."""

    enabled: bool = False
    wake_shedding: bool = True
    regularization: float = 1e-8

    @staticmethod
    def create(wake: bool = True):
        return VLMSolverConfig(enabled=True, wake_shedding=wake)

    @staticmethod
    def disabled():
        return VLMSolverConfig(enabled=False)


# =====================================================================================
# SOLVER CONFIGURATION DATACLASS
# =====================================================================================


@dataclass
class SolverConfig:
    """
    Configuration dataclass for the VPM solver.

    This dataclass provides a clean, type-safe, and user-friendly interface
    for configuring all aspects of the VPM simulation. It includes validation,
    sensible defaults, and comprehensive documentation.

    Key benefits:
    - Type hints and validation for all parameters
    - Clear documentation with physical units
    - Sensible defaults for common use cases
    - Easy serialization/deserialization
    - IDE autocompletion support
    """

    # ===== TIME CONTROL =====
    time_step_size: float = DEFAULT_TIME_STEP
    """Time increment per simulation step [s]"""

    flow_time: float = 0.0
    """Initial simulation time [s]"""

    time_step: int = 0
    """Initial time step number"""

    physics_substeps: int = 1
    """Minimum number of complete VPM physics microsteps per output step.

    Unlike ISR substeps, a physics microstep advances positions, recomputes
    velocity gradients and LES state, then advances stretching and diffusion.
    The default of one preserves the legacy update path exactly.
    """

    deformation_cfl: float | None = None
    """Optional adaptive deformation limit ``max(||S||_F) * dt_sub``.

    When set, the solver increases ``physics_substeps`` so the limit is met.
    This controls the complete VPM update rather than repeatedly integrating a
    frozen velocity-gradient tensor.
    """

    max_physics_substeps: int = 64
    """Safety ceiling for complete physics microsteps. Exceeding it aborts the
    step instead of silently violating ``deformation_cfl``.
    """

    # ===== PHYSICS CONFIGURATION =====
    advection: AdvectionConfig | None = None
    """Configuration for advection term (scheme, etc.)."""

    stretching: StretchingConfig | None = None
    """Configuration for vortex stretching term."""

    viscous: ViscousConfig | None = None
    """Configuration for viscous diffusion term."""

    turbulence: TurbulenceConfig | None = None
    """Configuration for turbulence modeling (DNS/LES/INVISCID)."""

    stabilization: "StabilizationConfig | None" = None
    """Unified configuration for all stabilization mechanisms (stretching regularisation,
    particle splitting, wake cutoff, weak removal, and conservative remeshing)."""

    vlm: VLMSolverConfig | None = None
    """Configuration for VLM coupling."""

    force: ForceConfig | None = None
    """Configuration for force evaluation method (Kutta-Joukowski vs Impulse-based)."""

    particles_kernel: Literal[
        "GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"
    ] = "GAUSSIAN"
    """Particle interaction kernel function type.

      ``HIGH_ORDER_GAUSSIAN`` — recommended production kernel: Gaussian with 2nd-order
      polynomial correction (2.5 − ρ²), improves far-field accuracy over plain Gaussian.
      """

    max_particles: int = MAX_PARTICLES
    """Maximum number of particles allowed in the simulation. Default: 500000"""

    # ===== COMPUTATIONAL SETTINGS =====
    processing_unit: Literal["CPU", "GPU", "GPU_VULKAN", "VULKAN", "CUDA", "GPU_METAL", "METAL"] = (
        "GPU"
    )
    """Compute backend.  Default ``'GPU'`` selects the best available GPU automatically:

    * **macOS**             → Metal  (Apple's native GPU API)
    * **Linux / Windows**   → CUDA   (if an NVIDIA GPU + driver is found)
    * **Linux / Windows**   → Vulkan (universal fallback for AMD, Intel, NVIDIA)

    Override the automatic selection via the ``OPENONDA_PROCESSING_UNIT`` environment
    variable or by passing an explicit value here:

    * ``'GPU'``        — best GPU for this platform (recommended)
    * ``'GPU_VULKAN'`` / ``'VULKAN'`` — force Vulkan (Linux / Windows)
    * ``'CUDA'``       — force CUDA (requires NVIDIA GPU + driver)
    * ``'GPU_METAL'``  / ``'METAL'`` — force Metal (macOS only)
    * ``'CPU'``        — software rendering (no GPU required)
    """

    # ===== MONITORING AND DIAGNOSTICS  =====
    logging_frequency: int = 0
    """Log flow diagnostics every N time steps (0 = disabled)."""

    solution_name: str = "solution"
    """Name of the solution directory where output files will be saved."""

    # ===== BACKUP AND OUTPUT =====
    backup_frequency: int = 0
    """Save simulation state every N time steps (0 = disabled)."""

    backup_file_name: str = DEFAULT_BACKUP_FILENAME
    """Optional infix in backup names, producing prefixes like 'vpm_<name>_*'."""

    backup_directory: str = "solution"
    """Output directory for backups and sampled files."""

    clean: bool = False
    """If True, delete the backup_directory before starting the simulation."""

    # ===== PHYSICS PARAMETERS =====
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    """Cutoff radius multiplier for particle interactions (performance optimization)"""

    precision: Literal["f32", "f64"] = "f32"
    """Floating-point precision for compute operations.

      - 'f32' (default): Single precision. Fast on GPU (Vulkan/CUDA). Minor precision loss in atomic adds.
      - 'f64': Double precision. Higher accuracy but REQUIRES CPU backend (Vulkan/CUDA don't support f64 well).

      The 'Atomic add may lose precision' warnings with f32 are generally acceptable for most CFD simulations.
      For high-accuracy requirements, use precision='f64' with processing_unit='CPU'."""

    device_memory_fraction: float = 0.5
    """Fraction of GPU VRAM reserved for Taichi's internal memory pool (default 0.5).

      The remaining VRAM is available for external-array staging buffers used to
      transfer numpy arrays to/from the GPU.  If your simulation crashes with
      ``Failed to allocate ext arr buffer``, **lower** this value (e.g. 0.3–0.4)
      so that more VRAM is left for staging.

      Clamped to the range [0.1, 0.7].  Values above ~0.7 almost always cause
      allocation failures on Vulkan backends."""

    # TODO: Consider converting to numpy array with correct precision in __post_init__
    background_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """Free-stream background velocity vector [ux, uy, uz] in m/s. Default: [0.0, 0.0, 0.0]"""

    verbose: bool = True
    """Enable verbose output (print particle shedding info, etc.)."""

    # ===== VELOCITY COMPUTATION =====
    velocity: VelocityConfig | None = None
    """Configuration for velocity field computation (direct vs treecode)."""

    # ===== SOLVER INSTANCES (Dependency Injection) =====
    panel_solver: Any | None = None
    """Panel solver instance for hybrid simulations."""

    vlm_solver: Any | None = None
    """VLM solver instance for VLM-VPM coupling."""

    # ===== FIELD SAMPLERS =====
    samplers: list[Any] | None = None
    """List of field samplers (SurfaceSampler, LineSampler) called at logging_frequency intervals."""

    body_stl: str | None = None
    """Path to body STL file for field sampler masking and geometry-aware output.
      Used by field samplers to mask interior points and project near-wall velocities.
      Can be absolute or relative to case directory."""

    vpm_domain_bounds: list[float] | None = None
    """Optional ``[xmin, xmax, ymin, ymax, zmin, zmax]`` of the VPM domain.
      When set, the DVH diffusion grid is pre-allocated to cover this domain
      (if the memory cost is acceptable) and re-allocation is capped at these
      bounds.  Passed automatically by the coupler."""

    def __post_init__(self):
        """Post-initialization validation and setup."""

        # Set defaults for None fields
        if self.advection is None:
            object.__setattr__(self, "advection", AdvectionConfig())

        if self.stretching is None:
            object.__setattr__(self, "stretching", StretchingConfig.transposed())

        if self.viscous is None:
            object.__setattr__(self, "viscous", ViscousConfig.cs())

        if self.turbulence is None:
            object.__setattr__(self, "turbulence", TurbulenceConfig.dns())

        if self.vlm is None:
            object.__setattr__(self, "vlm", VLMSolverConfig.disabled())

        if self.stabilization is None:
            object.__setattr__(self, "stabilization", StabilizationConfig.disabled())

        if self.force is None:
            object.__setattr__(self, "force", ForceConfig.kutta_joukowski())

        if self.velocity is None:
            object.__setattr__(self, "velocity", VelocityConfig.treecode(theta=0.5))

        # Validate precision
        if self.precision not in ("f32", "f64"):
            raise ValueError(f"precision must be 'f32' or 'f64', got '{self.precision}'")

        # Validation
        self._validate_config()

    def _validate_config(self):
        """Comprehensive validation of configuration parameters."""
        # Time step validation
        if self.time_step_size <= 0:
            raise ValueError("time_step_size must be positive")
        if self.physics_substeps < 1:
            raise ValueError("physics_substeps must be at least 1")
        if self.deformation_cfl is not None and self.deformation_cfl <= 0.0:
            raise ValueError("deformation_cfl must be positive when provided")
        if self.max_physics_substeps < self.physics_substeps:
            raise ValueError("max_physics_substeps must be >= physics_substeps")

        # Frequency validation
        if self.logging_frequency < 0:
            raise ValueError("logging_frequency must be non-negative")

        # Backup frequency validation
        if self.backup_frequency < 0:
            raise ValueError("backup_frequency must be non-negative")

        # Store backup frequency for solver use
        object.__setattr__(self, "_backup_frequency_internal", self.backup_frequency)

        # Processing unit validation
        valid_processing_units = [
            "CPU",
            "GPU",
            "GPU_VULKAN",
            "VULKAN",
            "CUDA",
            "GPU_METAL",
            "METAL",
        ]
        if self.processing_unit.upper() not in valid_processing_units:
            raise ValueError(
                f"processing_unit must be one of {valid_processing_units}, got '{self.processing_unit}'"
            )

        # Particles kernel validation
        _kernel_up = self.particles_kernel.upper()
        valid_kernels = ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
        if _kernel_up not in valid_kernels:
            raise ValueError(
                f"particles_kernel must be one of {valid_kernels}, got '{self.particles_kernel}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        # Serialize generic logic (assuming dataclasses.asdict or manual)
        # Since we have nested dataclasses, let's use a helper or manual construct

        def _as_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _as_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
            return obj

        return {
            "time_step_size": self.time_step_size,
            "flow_time": self.flow_time,
            "time_step": self.time_step,
            "physics_substeps": self.physics_substeps,
            "deformation_cfl": self.deformation_cfl,
            "max_physics_substeps": self.max_physics_substeps,
            "advection": _as_dict(self.advection) if self.advection else None,
            "stretching": _as_dict(self.stretching) if self.stretching else None,
            "viscous": _as_dict(self.viscous) if self.viscous else None,
            "turbulence": _as_dict(self.turbulence) if self.turbulence else None,
            "stabilization": _as_dict(self.stabilization) if self.stabilization else None,
            "vlm": _as_dict(self.vlm) if self.vlm else None,
            "particles_kernel": self.particles_kernel,
            "logging_frequency": self.logging_frequency,
            "backup_frequency": self.backup_frequency,
            "backup_file_name": self.backup_file_name,
            "backup_directory": self.backup_directory,
            "cutoff_radius_factor": self.cutoff_radius_factor,
            "precision": self.precision,
            "background_velocity": list(self.background_velocity)
            if self.background_velocity
            else [0.0, 0.0, 0.0],
            "verbose": self.verbose,
            "velocity": _as_dict(self.velocity) if self.velocity else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolverConfig":
        """Create configuration from dictionary."""
        values = dict(data)
        nested_types = {
            "advection": AdvectionConfig,
            "stretching": StretchingConfig,
            "viscous": ViscousConfig,
            "turbulence": TurbulenceConfig,
            "stabilization": StabilizationConfig,
            "velocity": VelocityConfig,
            "vlm": VLMSolverConfig,
        }
        for name, config_type in nested_types.items():
            if isinstance(values.get(name), dict):
                values[name] = config_type(**values[name])
        return cls(**values)

    @staticmethod
    def viscous_flow_simulation(
        time_step_size: float = 0.01,
        background_velocity: Any = (0.0, 0.0, 0.0),
        viscous: Optional["ViscousConfig"] = None,
        **kwargs,
    ) -> "SolverConfig":
        """
        Create a standard viscous flow simulation configuration.

        **Default physics:**
          - Stretching: disabled
          - Viscous:    Core Spreading (CS)
          - Turbulence: DNS (no subgrid model)
          - Advection:  RK2 (Heun's method)
          - Velocity:   Treecode (θ = 0.5)

        Args:
              time_step_size: Simulation time step [s]
              background_velocity: Free-stream velocity [m/s]
              viscous: Viscous scheme configuration (None = CS)
              **kwargs: Additional parameters to override defaults

        Returns:
              SolverConfig: Initialized configuration object
        """
        if viscous is None:
            viscous = ViscousConfig.cs()

        stretching = kwargs.pop("stretching", StretchingConfig.disabled())

        return SolverConfig(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            background_velocity=list(background_velocity),
            **kwargs,
        )

    @staticmethod
    def dns_simulation(time_step_size: float = 0.01, **kwargs) -> "SolverConfig":
        """
        Create a Direct Numerical Simulation (DNS) configuration.

        **Default physics:**
          - Advection:  RK2 (Heun's method)
          - Stretching: TRANSPOSED mode, RK2 (Heun's method)
          - Viscous:    Core Spreading (CS)
          - Turbulence: DNS (molecular viscosity only, no SGS model)
          - Velocity:   Treecode (θ = 0.5)
          - Stabilization: disabled

        Args:
              time_step_size: Simulation time step [s]
              **kwargs: Additional parameters to override defaults

        Returns:
              SolverConfig: Initialized configuration object
        """
        # Extract common kwargs if provided to avoid multiple values for the same argument
        stretching = kwargs.pop("stretching", StretchingConfig.transposed())
        viscous = kwargs.pop("viscous", ViscousConfig.cs())
        turbulence = kwargs.pop("turbulence", TurbulenceConfig.dns())

        return SolverConfig(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    @staticmethod
    def les_simulation(time_step_size: float = 0.01, cs: float = 0.17, **kwargs) -> "SolverConfig":
        """
        Create a Large Eddy Simulation (LES) configuration with the static
        Smagorinsky SGS model (k-equilibrium formulation).

        **Default physics:**
          - Advection:  RK2 (Heun's method)
          - Stretching: TRANSPOSED mode, RK2 (Heun's method)
          - Viscous:    Core Spreading (CS)
          - Turbulence: LES_SMAGORINSKY with C_s = 0.17, C_e = 1.048
          - Velocity:   Treecode (θ = 0.5)
          - Stabilization: disabled

        Args:
              time_step_size: Simulation time step [s]
              cs: Smagorinsky coefficient (default 0.17)
              **kwargs: Additional parameters to override defaults

        Returns:
              SolverConfig: Initialized configuration object
        """
        stretching = kwargs.pop("stretching", StretchingConfig.transposed())
        viscous = kwargs.pop("viscous", ViscousConfig.cs())
        # Use les_smagorinsky or similar if available, else fallback
        turbulence = kwargs.pop("turbulence", TurbulenceConfig.les_smagorinsky(cs=cs))

        return SolverConfig(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    def save_to_file(self, filename: str):
        """Save configuration to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> "SolverConfig":
        """Load configuration from JSON file."""
        with open(filename) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """Human-readable string representation."""
        flow_model = (
            self.turbulence.flow_model
            if self.turbulence and hasattr(self.turbulence, "flow_model")
            else "Inferred"
        )

        lines = ["VPM Solver Configuration:"]
        lines.append(f"  Flow Model: {flow_model}")
        lines.append("  Time Integration:")
        lines.append(f"    Advection:  {self.advection}")
        lines.append(f"    Stretching: {self.stretching}")
        lines.append(f"    Diffusion:  {self.viscous}")
        lines.append(f"  Processing Unit: {self.processing_unit}")
        lines.append(f"  Time Step Size: {self.time_step_size:.2e} s")
        lines.append(f"  Viscous Scheme: {self.viscous.scheme if self.viscous else 'None'}")
        lines.append(f"  Particle Kernel: {self.particles_kernel}")
        lines.append(f"  Cutoff Factor: {self.cutoff_radius_factor}")
        lines.append(f"  Logging Frequency: {self.logging_frequency} steps")
        lines.append(f"  Backup Frequency: {self.backup_frequency} steps")
        lines.append(f"  Background Velocity: {self.background_velocity} m/s")
        return "\n".join(lines)


# =====================================================================================
# UTILITY DECORATORS AND HELPERS
# =====================================================================================
def CachedParticleProperty(func):
    """
    Decorator for caching expensive particle property calculations.

    Caches results based on the current time step to avoid redundant computations.
    Uses per-property timestamp to ensure independence.
    """
    cache_name = f"_{func.__name__}_cache"
    step_name = f"_{func.__name__}_step"

    def wrapper(self, use_cache: bool = True):
        # Check if cache is valid and should be used
        cache_valid = (
            use_cache
            and hasattr(self, step_name)
            and getattr(self, step_name) == self.time_step
            and hasattr(self, cache_name)
        )

        # Also check global invalidation signal (-1)
        if hasattr(self, "_cached_step") and self._cached_step == -1:
            cache_valid = False

        if not cache_valid:
            result = func(self)
            setattr(self, cache_name, result)
            setattr(self, step_name, self.time_step)

        return getattr(self, cache_name)

    return wrapper


# =====================================================================================
# PYDANTIC MODELS FOR SERIALIZATION AND STATE MANAGEMENT
# =====================================================================================
class SolverState(BaseModel):
    """
    Pydantic model for serializing and deserializing solver state.

    This model handles the complete simulation state including parameters,
    timing information, and configuration settings. It supports robust
    backup/restore operations with validation.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core simulation parameters (required for initialization)
    time_step_size: float = Field(gt=0.0, description="Time step size in seconds")
    flow_time: float = Field(ge=0.0, default=0.0, description="Current simulation time")
    time_step: int = Field(ge=0, default=0, description="Current time step number")

    # Method and model configuration
    # Method and model configuration
    # time_integration_scheme: removed in favor of specific schemes
    advection_scheme: str = Field(default="RK2", description="Advection time integration scheme")
    stretching_scheme: str = Field(default="RK2", description="Stretching time integration scheme")

    processing_unit: str = Field(default="GPU_VULKAN", description="Computation backend")
    flow_model: str = Field(default="DNS", description="Flow physics model")
    viscous_scheme: str = Field(default="CS", description="Viscous modeling scheme")

    # Additional config fields for full reconstruction
    stretching_enabled: bool = Field(default=True, description="Whether stretching is enabled")
    stretching_mode: str = Field(default="TRANSPOSED", description="Stretching mode")
    particles_kernel: str = Field(default="GAUSSIAN", description="Particle kernel function")

    # Simulation control parameters
    backup_file_name: str = Field(default="", description="Optional backup file name infix")
    backup_directory: str = Field(default="solution", description="Output directory for backups")
    logging_frequency: int = Field(default=0, description="Log frequency in steps")
    backup_frequency: int = Field(default=0, description="Backup frequency in steps")

    # Runtime state (optional, set during execution)
    simulation_time: float | None = Field(default=0.0, ge=0.0, description="Total wall-clock time")
    cached_step: int | None = Field(default=0, description="Last cached computation step")
    E_previous: float | None = Field(
        default=0.0, description="Previous energy for decay calculation"
    )
    E_previous2: float | None = Field(default=0.0, description="Energy from 2 steps ago")

    @field_validator("processing_unit")
    @classmethod
    def validate_processing_unit(cls, v: str) -> str:
        """Validate processing unit."""
        valid_units = {"CPU", "GPU", "GPU_VULKAN", "VULKAN", "CUDA", "GPU_METAL", "METAL"}
        v_upper = v.upper()
        if v_upper not in valid_units:
            raise ValueError(f"Invalid processing unit: {v}. Must be one of {valid_units}")
        return v_upper

    @field_validator("flow_model")
    @classmethod
    def validate_flow_model(cls, v: str) -> str:
        """Validate flow model."""
        valid_models = {"DNS", "LES"}
        v_upper = v.upper()
        if v_upper not in valid_models:
            raise ValueError(f"Invalid flow model: {v}. Must be one of {valid_models}")
        return v_upper

    @classmethod
    def from_solver(cls, solver) -> "SolverState":
        """
        Convert solver object to SolverState for serialization.

        Args:
              solver: The solver instance to convert

        Returns:
              SolverState: Serializable state representation

        Raises:
              ValueError: If solver has invalid or missing required attributes
        """
        try:
            # Extract core attributes, handling missing values gracefully
            solver_dict = {}
            for key, value in solver.__dict__.items():
                # Skip non-serializable attributes and private attributes
                if key in ["particles", "physics", "turbulence", "config", "io"] or key.startswith(
                    "_"
                ):
                    continue

                # Include only serializable types
                if isinstance(value, int | float | str | list | bool | type(None)):
                    solver_dict[key] = value

            return cls(**solver_dict)
        except Exception as e:
            raise ValueError(f"Failed to convert solver to state: {e}") from e

    def to_solver(self):
        """
        Convert SolverState back to solver object.

        Returns:
              Solver: Fully initialized solver instance

        Raises:
              ValueError: If state contains invalid parameters
        """
        try:
            # Import locally to avoid circular dependencies
            from source.solvers.VPM.config.types import (
                AdvectionConfig,
                SolverConfig,
                StretchingConfig,
                TurbulenceConfig,
                ViscousConfig,
            )
            from source.solvers.VPM.core.solver import Solver

            # Reconstruct Configuration Objects
            advection = AdvectionConfig(scheme=self.advection_scheme)

            stretching = StretchingConfig(
                mode=self.stretching_mode,
                scheme=self.stretching_scheme,
                enabled=self.stretching_enabled,
            )

            viscous = ViscousConfig(
                scheme=self.viscous_scheme,
            )

            # Turbulence config reconstruction
            # Same issue for cs, etc.
            turbulence = TurbulenceConfig.dns()
            if self.flow_model == "LES":
                turbulence = TurbulenceConfig.les_smagorinsky()  # Default/Placeholder

            # Reconstruct full SolverConfig
            config = SolverConfig(
                time_step_size=self.time_step_size,
                flow_time=self.flow_time,
                time_step=self.time_step,
                advection=advection,
                stretching=stretching,
                viscous=viscous,
                turbulence=turbulence,
                particles_kernel=self.particles_kernel,
                logging_frequency=self.logging_frequency,
                backup_frequency=self.backup_frequency,
                backup_file_name=self.backup_file_name,
                backup_directory=self.backup_directory,
                processing_unit=self.processing_unit,
                # precision?
            )

            # Create new solver instance with config
            new_solver = Solver(config=config)

            # Restore additional attributes that aren't constructor parameters
            for key, value in self.__dict__.items():
                if not hasattr(new_solver, key) and not key.startswith("_"):
                    setattr(new_solver, key, value)

            return new_solver

        except Exception as e:
            raise ValueError(f"Failed to create solver from state: {e}") from e


class ParticlesState(BaseModel):
    """
    Pydantic model for serializing complete particle state.

    This model handles all particle data including positions, velocities, strengths,
    and computed fields. It provides robust validation and efficient conversion
    to/from the Particles class.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Core particle data (always present)
    positions: list[list[float]] = Field(description="Particle positions [N, 3]")
    velocities: list[list[float]] = Field(description="Particle velocities [N, 3]")
    strengths: list[list[float]] = Field(description="Particle strengths [N, 3]")
    radii: list[float] = Field(description="Particle radii [N]")
    volumes: list[float] = Field(description="Particle volumes [N]")
    viscosities: list[float] = Field(description="Particle molecular viscosities [N]")
    viscosities_t: list[float] = Field(description="Particle turbulent viscosities [N]")
    group_ids: list[int] = Field(description="Particle group identifiers [N]")

    # Optional computed fields (may not always be present)
    grad_u: list[list[list[float]]] | None = Field(
        default=None, description="Velocity gradient tensors [N, 3, 3]"
    )
    vorticities: list[list[float]] | None = Field(
        default=None, description="Particle vorticities [N, 3]"
    )

    @field_validator("positions", "velocities", "strengths")
    @classmethod
    def validate_vector_fields(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate vector fields have consistent 3D structure."""
        if not v:
            return v
        for i, vec in enumerate(v):
            if len(vec) != 3:
                raise ValueError(f"Vector at index {i} must have 3 components, got {len(vec)}")
        return v

    @field_validator("radii", "volumes", "viscosities", "viscosities_t")
    @classmethod
    def validate_positive_scalars(cls, v: list[float]) -> list[float]:
        """Validate scalar fields are positive."""
        for i, val in enumerate(v):
            if val < 0:
                raise ValueError(f"Value at index {i} must be non-negative, got {val}")
        return v

    def validate_consistency(self) -> None:
        """Validate that all fields have consistent sizes."""
        n_particles = len(self.positions)

        # Check all required fields have same length
        fields_to_check = [
            ("velocities", self.velocities),
            ("strengths", self.strengths),
            ("radii", self.radii),
            ("volumes", self.volumes),
            ("viscosities", self.viscosities),
            ("viscosities_t", self.viscosities_t),
            ("group_ids", self.group_ids),
        ]

        for field_name, field_data in fields_to_check:
            if len(field_data) != n_particles:
                raise ValueError(
                    f"Field '{field_name}' has {len(field_data)} elements, "
                    f"expected {n_particles} to match positions"
                )

        # Check optional fields if present
        if self.grad_u is not None and len(self.grad_u) != n_particles:
            raise ValueError(f"grad_u field size mismatch: {len(self.grad_u)} != {n_particles}")
        if self.vorticities is not None and len(self.vorticities) != n_particles:
            raise ValueError(
                f"vorticities field size mismatch: {len(self.vorticities)} != {n_particles}"
            )

    @classmethod
    def from_particles(cls, particles) -> "ParticlesState":
        """
        Convert Particles object to ParticlesState for serialization.

        Args:
              particles: The particles instance to convert

        Returns:
              ParticlesState: Serializable particle state

        Raises:
              ValueError: If particles object is invalid or conversion fails
        """
        try:
            # Convert Taichi fields to numpy arrays, then to lists
            data = {
                "positions": particles.positions.to_numpy().tolist(),
                "velocities": particles.velocities.to_numpy().tolist(),
                "strengths": particles.strengths.to_numpy().tolist(),
                "radii": particles.radii.to_numpy().tolist(),
                "volumes": particles.volumes.to_numpy().tolist(),
                "viscosities": particles.viscosities.to_numpy().tolist(),
                "viscosities_t": particles.viscosities_t.to_numpy().tolist(),
                "group_ids": particles.group_ids.to_numpy().tolist(),
                "vorticities": particles.vorticities.to_numpy().tolist(),
            }

            # Add optional fields if they exist and are properly initialized
            if hasattr(particles, "grad_u") and particles.grad_u is not None:
                try:
                    grad_u_array = particles.grad_u.to_numpy()
                    if grad_u_array.size > 0:
                        data["grad_u"] = grad_u_array.tolist()
                except Exception:
                    pass  # Skip if conversion fails

            state = cls(**data)
            state.validate_consistency()
            return state

        except Exception as e:
            raise ValueError(f"Failed to convert particles to state: {e}") from e

    def to_particles(self):
        """
        Convert ParticlesState back to a fully initialized Particles object.

        Returns:
              Particles: Fully initialized particles instance

        Raises:
              ValueError: If state is invalid or conversion fails
        """
        try:
            # Import locally to avoid circular dependencies
            from source.solvers.VPM.particles.container import Particles

            # Validate consistency before conversion
            self.validate_consistency()

            # Determine the number of particles
            n_particles = len(self.positions)
            if n_particles == 0:
                raise ValueError("Cannot create particles from empty state")

            # Create a new Particles object with sufficient capacity
            particles = Particles(max_particles=max(n_particles, 100))

            # Convert all lists to numpy arrays with proper dtypes
            positions = np.array(self.positions, dtype=np.float32)
            velocities = np.array(self.velocities, dtype=np.float32)
            strengths = np.array(self.strengths, dtype=np.float32)
            radii = np.array(self.radii, dtype=np.float32)
            volumes = np.array(self.volumes, dtype=np.float32)
            viscosities = np.array(self.viscosities, dtype=np.float32)
            viscosities_t = np.array(self.viscosities_t, dtype=np.float32)
            group_ids = np.array(self.group_ids, dtype=np.int32)

            # Handle optional fields safely
            grad_u = None
            if self.grad_u is not None:
                grad_u = np.array(self.grad_u, dtype=np.float32)

            # Use add_vortex_particles for robust initialization
            particles.add_vortex_particles(
                positions=positions,
                velocities=velocities,
                strengths=strengths,
                radii=radii,
                volumes=volumes,
                viscosities=viscosities,
                viscosities_t=viscosities_t,
                group_id=group_ids,
                grad_u=grad_u,
            )

            # Ensure particle count is set correctly
            particles.number_of_particles = n_particles

            return particles

        except Exception as e:
            raise ValueError(f"Failed to create particles from state: {e}") from e


# =====================================================================================
# UTILITY FUNCTIONS FOR FLOW MODEL SETTING
# =====================================================================================
def SetFlowModel(psys, flow_model: str):
    """
    Set flow model and configure associated parameters.
    Note: Validation is already done in SolverConfig, so this just sets the model.
    """
    if flow_model == "DNS":
        psys.flow_model_description = "DNS ::: (ω.∇)u + (v)(∇²)ω"

    elif flow_model == "LES":
        # LES model description will be set based on smagorinsky type in ParticlesLES
        psys.flow_model_description = "LES ::: (ω.∇)u + (v+vt)(∇²)ω"

    elif flow_model == "INVISCID":
        psys.flow_model_description = "INV ::: (ω.∇)u (stretching only)"

    psys.flow_model = flow_model


__all__ = [
    "SolverConfig",
    "SolverState",
    "ParticlesState",
    "CachedParticleProperty",
    "SetFlowModel",
    "ViscousConfig",
    "StabilizationConfig",
]
