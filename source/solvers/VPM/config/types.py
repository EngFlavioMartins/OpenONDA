"""
Configuration dataclasses for the VPM solver: advection, viscous, stretching,
turbulence, stabilization, velocity, and the top-level VPMSetup.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

# =========================================================
# PRECISION CONFIGURATION
# =========================================================
from dataclasses import dataclass, field
import json
import re

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
    TREECODE_SUPPORTED_KERNELS,
)


# =========================================================
# ADVECTION CONFIGURATION
# =========================================================
@dataclass(frozen=True)
class AdvectionConfig:
    """
    Configuration for the advection substep: dx/dt = u.

    Advection advances each particle's position by solving the material-derivative
    equation dx/dt = u( x(t), t ) over the configured time-integration scheme.
    The substep is nested inside the solver's macro time-step (the DVH-pinned
    dt for DVH runs).

    Available schemes
    -----------------
    ``NONE``
        Freeze particle positions.  Useful for stationary-flow viscous tests
        where particle motion is undesirable.
    ``EULER``
        Forward / explicit Euler — 1st order.  Lowest accuracy per step;
        generally not recommended for production runs.
    ``RK2``
        Heun's method — 2nd-order Runge–Kutta.  Good balance of accuracy
        and cost for moderate advection-dominated applications.
    ``RK3``
        Strong-Stability-Preserving Runge–Kutta — 3rd order (Gottlieb, Shu &
        Tadmor 2001).  Default for production VPM runs.
    ``RK4``
        Classical 4th-order Runge–Kutta.  Highest per-step accuracy but
        4 evaluations per step versus 2 for RK2.

    Examples
    --------
    .. code-block:: python

        # Default (RK3)
        adv = AdvectionConfig()

        # 4th-order advection
        adv = AdvectionConfig(scheme="RK4")

        # Freeze particles (stationary-flow test)
        adv = AdvectionConfig(scheme="NONE")

    References
    ----------
    - Gottlieb, Shu & Tadmor (2001) SIAM J. Numer. Anal. 39(5), 1984–2012.
    """

    scheme: Literal["NONE", "EULER", "RK2", "RK3", "RK4"] = "RK3"
    """Time integration scheme for the advection substep (dx/dt = u).

    Options:
      - 'NONE':  freeze particle positions (useful for stationary-flow viscous tests)
      - 'EULER': forward Euler (1st order)
      - 'RK2':   Heun's method, 2nd-order Runge–Kutta
      - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
      - 'RK4':   classical 4th-order Runge–Kutta

    Advection advances the configured scheme over the macro time-step set by the
    solver (the DVH-pinned dt for DVH runs)."""


# =========================================================
# VISCOUS CONFIGURATION
# =========================================================


@dataclass(frozen=True)
class ViscousConfig:
    """
    Configuration for viscous diffusion in the VPM solver.

    Viscous diffusion models the physical term ν∇²ω in the vorticity transport
    equation.  Five fundamentally different schemes are available, each with
    distinct trade-offs in accuracy, stability, cost, and suitability for
    coupled FVM-VPM simulations.

    Available schemes
    -----------------
    ``CS``
        Core Spreading — deterministic, O(N).  Each particle's core radius
        grows as σ(t) = √(σ₀² + 4νt).  Simple and cheap, but the varying
        core size degrades spatial resolution over time.

    ``RWM``
        Random Walk Method — stochastic, O(N).  Particles are displaced by a
        Gaussian random vector with variance 2νΔt.  Unconditionally stable,
        but accuracy degrades when the displacement exceeds h/√2.
        Accuracy bound: Δt ≤ h²/(4ν).

    ``NONE``
        No viscous diffusion.  Use for inviscid validation or when only
        stretching (and possibly LES) is active.

    ``DVH``
        Diffused Vortex Hydrodynamics (Durante et al. 2024) — grid-based,
        deterministic.  Each particle's circulation is scattered to nearby
        grid nodes via the exact heat-kernel Gaussian; Shepard normalisation
        enforces per-particle Γ conservation.  Particles are then replaced
        by surviving grid nodes (simultaneous diffusion + regularisation).
        No CFL lower bound — the time-step is pinned to Δt_d = β·R_d²/(4ν).

    ``GBD``
        Grid-Based Diffusion (Cottet & Koumoutsakos 2000) — grid-based,
        deterministic.  M4' remeshing scatter → explicit 7-point Laplacian
        → threshold pruning with Γ conservation → particle regeneration.
        Upper CFL bound: Δt ≤ h²/(6ν).

    Guidance
    --------
    - **CS** is the simplest choice for standalone VPM runs where moderate
      over-diffusion is acceptable.
    - **RWM** is useful as a stochastic alternative; run ensemble averages
      to recover smooth fields.
    - **DVH** is recommended for coupled FVM-VPM simulations: the
      simultaneous regen controls particle count and the Gaussian heat-kernel
      respects the exact diffusion increment.
    - **GBD** is a lighter grid-based alternative when DVH's scatter stencil
      is too expensive or when a standard finite-difference Laplacian is
      preferred.

    Examples
    --------
    .. code-block:: python

        # Core Spreading
        visc = ViscousConfig.cs(viscosity=1e-5, characteristic_distance=0.01)

        # Random Walk
        visc = ViscousConfig.rwm(viscosity=1e-5, characteristic_distance=0.01)

        # DVH with default threshold
        visc = ViscousConfig.dvh(h=0.01, viscosity=1e-5, dvh_rd_ratio=4)

        # GBD
        visc = ViscousConfig.gbd(h=0.01, viscosity=1e-5)

        # Inviscid
        visc = ViscousConfig.inviscid()

    References
    ----------
    - Cottet & Koumoutsakos (2000) "Vortex Methods: Theory and Practice",
      Cambridge University Press.
    - Durante et al. (2024) "Diffused Vortex Hydrodynamics", J. Comput. Phys.
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
        _valid_modes = ("budget", "relative_max", "absolute", "relative_local")
        if self.dvh_threshold_mode not in _valid_modes:
            raise ValueError(
                f"dvh_threshold_mode must be one of {_valid_modes}, got {self.dvh_threshold_mode!r}"
            )
        if self.gbd_threshold_mode not in _valid_modes:
            raise ValueError(
                f"gbd_threshold_mode must be one of {_valid_modes}, got {self.gbd_threshold_mode!r}"
            )
        if not 0.0 < self.regen_cap_abs_fraction <= 1.0:
            raise ValueError("regen_cap_abs_fraction must be in (0, 1].")

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
    this automatically.  Default 2.5 is the standard standalone overlap."""

    # ---- Grid-Based Diffusion (DVH) parameters ----
    dvh_grid_spacing: float | None = None
    """Grid spacing h [m] for the DVH (DVH) scheme. Falls back to characteristic_distance
    if None."""

    dvh_domain_padding: float = 3.0
    """Grid extends this many grid-spacings beyond the particle bounding box on each
    side, so that vorticity decays to ~0 before reaching the Dirichlet boundary."""

    # ---- Grid-Based Diffusion / DVH (DVH) parameters ----
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

    # ---- Grid-Based Diffusion (GBD) parameters ----
    gbd_grid_spacing: float | None = None
    """Grid spacing h [m] for GBD.  Falls back to characteristic_distance if None."""

    gbd_domain_padding: float = 3.0
    """Grid extends this many grid-spacings beyond the particle bounding box."""

    gbd_threshold: float = 0.01
    """Circulation threshold for GBD particle regeneration.
    Interpretation depends on ``gbd_threshold_mode`` (same semantics as DVH)."""

    gbd_threshold_mode: str = "budget"
    """How ``gbd_threshold`` is interpreted in GBD mode.
    ``'budget'``         — keep the top-(1-threshold) fraction of total |Γ| sum.
    ``'relative_max'``   — keep nodes above threshold × global max|Γ|.
    ``'absolute'``       — keep nodes above the absolute circulation value
                           ``gbd_threshold`` [m³/s].
    ``'relative_local'`` — keep nodes above threshold × the MEAN |Γ| over a
                           (2w+1)³ window (w = ``regen_threshold_window``).
                           Recommended for coupled FVM-VPM runs: see
                           ``regen_threshold_window``."""

    regen_threshold_window: int = 3
    """Half-width w (in grid cells) of the ``'relative_local'`` reference window.

    ``'relative_local'`` thresholds each regen node against the mean |Γ| within
    w cells instead of against the global max|Γ|, which is what makes it safe on
    fields with a wide dynamic range.  In a coupled FVM-VPM run the
    global maximum sits in the wall vortex sheet while the wake one body-length
    downstream is ~10⁻³ of it, so every global-reference mode deletes the far
    wake to keep the boundary layer — and because the reference lives on the
    body, the cut level in the wake jitters with the near-wall solution.  A
    local reference decouples the two.

    w ≈ 3 (a 7³ window, i.e. ±3h ≈ 3 core radii) resolves individual wake
    structures; larger w averages unrelated structures together and drifts back
    toward global behaviour.  Applies to both DVH and GBD."""

    regen_cap_abs_fraction: float = 0.99
    """Minimum fraction of surviving Σ|Γ| protected by a particle cap."""

    gbd_max_nodes: int | None = None
    """Hard cap on surviving grid nodes per GBD regen (GBD only).

    Bounds particle-count growth in long grid-diffusion runs.  None (default)
    keeps only the built-in safety cap (3×N, ≤ MAX_PARTICLES)."""

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
        threshold_window: int = 3,
        dvh_rd_ratio: int = 4,
        viscosity: float | None = None,
        max_nodes: int | None = None,
        cap_abs_fraction: float = 0.99,
        regen_radius_ratio: float = 2.5,
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
                ``'relative_local'`` — discard nodes below threshold × the mean
                  |Γ| over a (2w+1)³ window.  Recommended for coupled
                  FVM-VPM runs; see ``regen_threshold_window``.
            threshold_window: Half-width w in cells of the ``'relative_local'``
                reference window (ignored by the other modes).
            dvh_rd_ratio: R_d/h compact-support radius ratio for the DVH
                heat-kernel.  Integer in [3, 5].  Default 4
                (optimal, Durante 2024 Sec. 4.2).
            viscosity: Molecular kinematic viscosity nu [m²/s].  Required
                for DVH — automatically assigned to every new particle and
                determines the DVH time-step Δt_d = β·R_d²/(4nu).
            max_nodes: Hard cap on surviving regen nodes (budget-by-count) —
                bounds the budget-mode halo growth.  None = built-in cap only.
            cap_abs_fraction: Minimum surviving Σ|Γ| protected by the cap.
            regen_radius_ratio: Core radius σ = ratio·h assigned to regenerated
                particles.  Default 2.5. Lower toward 1.5 to avoid
                over-smearing the reconstructed field (see the field docstring).
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
            regen_threshold_window=threshold_window,
            dvh_rd_ratio=dvh_rd_ratio,
            viscosity=viscosity,
            dvh_max_nodes=max_nodes,
            regen_cap_abs_fraction=cap_abs_fraction,
            regen_radius_ratio=regen_radius_ratio,
        )

    @staticmethod
    def gbd(
        h: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        threshold_window: int = 3,
        viscosity: float | None = None,
        max_nodes: int | None = None,
        cap_abs_fraction: float = 0.99,
        regen_radius_ratio: float = 2.5,
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
            threshold_mode: ``'budget'``, ``'relative_max'``, ``'absolute'``, or
                ``'relative_local'``.
                ``'absolute'`` uses ``threshold`` as a raw circulation magnitude
                in [m³/s] and is the preferred mode for controlling particle
                count in simulations with large dynamic range.
                ``'relative_local'`` thresholds against the local (not global)
                |Γ| level and is the recommended mode for coupled FVM-VPM runs,
                where a global reference deletes the far wake to keep the wall
                vortex sheet.
            threshold_window: Half-width w in cells of the ``'relative_local'``
                reference window (ignored by the other modes).
            viscosity: Molecular kinematic viscosity nu [m²/s].
            max_nodes: Hard cap on surviving regen nodes (budget-by-count).
            cap_abs_fraction: Minimum surviving Σ|Γ| protected by the cap.
            regen_radius_ratio: Core radius σ = ratio·h assigned to regenerated
                particles.  Default 2.5. Lower toward 1.5 to avoid
                over-smearing the reconstructed field (see the field docstring).
        """
        return ViscousConfig(
            scheme="GBD",
            gbd_grid_spacing=h,
            gbd_domain_padding=padding,
            gbd_threshold=threshold,
            gbd_threshold_mode=threshold_mode,
            regen_threshold_window=threshold_window,
            viscosity=viscosity,
            gbd_max_nodes=max_nodes,
            regen_cap_abs_fraction=cap_abs_fraction,
            regen_radius_ratio=regen_radius_ratio,
        )


@dataclass(frozen=True)
class StretchingConfig:
    """
    Configuration for vortex stretching schemes.

    The formulations may use either direct evaluation or a controlled treecode
    approximation.  ``StabilizationConfig`` is only a particle-retention policy
    and never modifies vortex strength.

    Modes:
        - DIRECT: dΓ/dt = (Γ·∇)u, the standard direct formulation
        - TRANSPOSED: dΓ/dt = (∇u)ᵀΓ, conserves vector circulation
        - MIXED: Strain-based symmetric direct formulation

    Examples:
          # Transposed stretching (conserves circulation, recommended)
          stretching = StretchingConfig.transposed()

          # Direct stretching
          stretching = StretchingConfig.direct()

          # Mixed/strain stretching
          stretching = StretchingConfig.mixed()

          # Disabled stretching
          stretching = StretchingConfig.disabled()
    """

    mode: Literal["DIRECT", "TRANSPOSED", "MIXED"] = "TRANSPOSED"
    """Stretching formulation mode."""

    scheme: Literal["EULER", "RK2", "RK3", "RK4"] = "RK3"
    """Time integration scheme for the stretching substep (dΓ/dt).

    Options:
      - 'EULER': forward Euler (1st order)
      - 'RK2':   Heun's method, 2nd-order Runge–Kutta
      - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
      - 'RK4':   classical 4th-order Runge–Kutta
    """

    enabled: bool = True
    """Whether vortex stretching is enabled."""

    use_treecode: bool = False
    """Compute the stretching rate from the treecode velocity gradient instead
    of the direct O(N²) pairwise kernel.

    The stretching rate dΓ/dt = (Γ·∇)u (DIRECT), (∇u)ᵀ·Γ (TRANSPOSED) or the
    symmetric S·Γ (MIXED) is an exact local contraction of the velocity-gradient
    tensor ∇u.  The direct kernel forms ∇u·Γ implicitly with an O(N²) pair sum;
    with ``use_treecode=True`` the same ∇u is evaluated by the Barnes–Hut
    treecode (O(N log N)) and contracted locally.  The contraction itself is
    exact (relL2 ≈ 1e-6 against the direct rate when contracted with the
    directly-computed gradient); through the treecode gradient the rate differs
    from direct by the Barnes–Hut opening-angle tolerance (measured relL2
    ≈ 4e-2 at θ=0.2 on a random cloud — the same error class the advection
    velocities already carry).

    IMPORTANT — measured tradeoff (RTX 3060, 2026-07): the treecode *gradient*
    traversal (9 tensor components, deep walks) is intrinsically expensive, so
    the O(N²) direct kernel is actually FASTER up to at least N ≈ 250k (direct
    0.80× the treecode wall time at 249k).  The crossover is beyond a 6 GB card.
    The physics is preserved either way (circulation matches to ~4e-5 after
    several steps).  Enable this only above the crossover, or once the treecode
    traversal itself is made cheaper (higher-order multipoles → larger θ).
    Default False keeps the faster exact-direct rate."""

    treecode_theta: float = 0.3
    """Barnes–Hut opening angle for the treecode stretching gradient
    (only used when ``use_treecode=True``).  Smaller = more accurate/slower."""

    def __post_init__(self) -> None:
        mode = self.mode.upper()
        scheme = self.scheme.upper()
        if mode not in ("DIRECT", "TRANSPOSED", "MIXED"):
            raise ValueError(
                f"stretching mode must be DIRECT, TRANSPOSED, or MIXED, got {self.mode!r}"
            )
        if scheme not in ("EULER", "RK2", "RK3", "RK4"):
            raise ValueError(
                f"stretching scheme must be EULER, RK2, RK3, or RK4, got {self.scheme!r}"
            )
        if not 0.0 < self.treecode_theta < 2.0:
            raise ValueError(f"treecode_theta must be in (0, 2), got {self.treecode_theta!r}")
        if mode != self.mode:
            object.__setattr__(self, "mode", mode)
        if scheme != self.scheme:
            object.__setattr__(self, "scheme", scheme)

    @staticmethod
    def direct(scheme: str = "RK3", use_treecode: bool = False, treecode_theta: float = 0.3):
        """Direct scheme: dΓ/dt = (Γ·∇)u

        Options for `scheme`:
          - 'EULER': forward Euler (1st order)
          - 'RK2':   Heun's method, 2nd-order Runge–Kutta
          - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
          - 'RK4':   classical 4th-order Runge–Kutta

        Set ``use_treecode=True`` to evaluate the rate from the O(N log N)
        treecode gradient instead of the O(N²) pairwise kernel (large N).
        """
        return StretchingConfig(
            mode="DIRECT", scheme=scheme, use_treecode=use_treecode, treecode_theta=treecode_theta
        )

    @staticmethod
    def transposed(scheme: str = "RK3", use_treecode: bool = False, treecode_theta: float = 0.3):
        """Transposed scheme: dΓ/dt = (Γ·∇')u - conserves ΣΓ

        Options for `scheme`:
          - 'EULER': forward Euler (1st order)
          - 'RK2':   Heun's method, 2nd-order Runge–Kutta
          - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
          - 'RK4':   classical 4th-order Runge–Kutta

        Set ``use_treecode=True`` to evaluate the rate from the O(N log N)
        treecode gradient instead of the O(N²) pairwise kernel (large N).
        """
        return StretchingConfig(
            mode="TRANSPOSED",
            scheme=scheme,
            use_treecode=use_treecode,
            treecode_theta=treecode_theta,
        )

    @staticmethod
    def mixed(scheme: str = "RK3", use_treecode: bool = False, treecode_theta: float = 0.3):
        """Mixed/strain scheme: symmetric formulation

        Options for `scheme`:
          - 'EULER': forward Euler (1st order)
          - 'RK2':   Heun's method, 2nd-order Runge–Kutta
          - 'RK3':   SSP-RK3, 3rd-order strong-stability-preserving (default)
          - 'RK4':   classical 4th-order Runge–Kutta

        Set ``use_treecode=True`` to evaluate the rate from the O(N log N)
        treecode gradient instead of the O(N²) pairwise kernel (large N).
        """
        return StretchingConfig(
            mode="MIXED", scheme=scheme, use_treecode=use_treecode, treecode_theta=treecode_theta
        )

    @staticmethod
    def disabled():
        return StretchingConfig(enabled=False)


# VLM definitions are independent of the runtime solver.
from ..boundary_elements.vlm.config import VLMSetup


# =========================================================
# TURBULENCE CONFIGURATION
# =========================================================
@dataclass(frozen=True)
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
          config = VPMSetup(turbulence=TurbulenceConfig.dns())

          # Static Smagorinsky LES
          config = VPMSetup(turbulence=TurbulenceConfig.les_smagorinsky(cs=0.17))

          # Inviscid — pure stretching
          config = VPMSetup(
              turbulence=TurbulenceConfig.inviscid(),
              viscous=ViscousConfig.inviscid(),
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

      Default: 1.048 (derived from Lilly 1966 spectral analysis)

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
            >>> config = VPMSetup(
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
            - Equilibrium coefficients C_k=0.094 → C_s ≈ 0.168

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
    def equilibrium_smagorinsky(ck: float = 0.094, ce: float = 1.048) -> "TurbulenceConfig":
        r"""Configure equilibrium Smagorinsky coefficients.

        The VPM solver is incompressible, so the algebraic SGS-energy model
        reduces exactly to

        ``nu_t = (C_s Delta)^2 |S|`` with
        ``C_s = C_k^(3/4) / C_e^(1/4)``.

        This factory accepts equilibrium coefficients and converts them to the
        existing particle model's ``C_s`` representation. The
        defaults ``C_k=0.094`` and ``C_e=1.048`` give ``C_s≈0.168``.

        Args:
            ck: SGS kinetic-energy coefficient ``C_k``.
            ce: SGS dissipation coefficient ``C_e``.

        Returns:
            An LES configuration using the equilibrium Smagorinsky model and
            supplied coefficients.

        Example:
            >>> cfg = TurbulenceConfig.equilibrium_smagorinsky()
            >>> cfg.flow_model
            'LES'
            >>> round((cfg.cs**2 * cfg.ce**0.5) ** (2.0 / 3.0), 3)
            0.094
        """
        if not np.isfinite(ck) or ck < 0.0:
            raise ValueError("Equilibrium Smagorinsky ck must be finite and non-negative")
        if not np.isfinite(ce) or ce <= 0.0:
            raise ValueError("Equilibrium Smagorinsky ce must be finite and positive")
        cs = ck**0.75 / ce**0.25
        return TurbulenceConfig.les_smagorinsky(cs=cs, ce=ce)

    @staticmethod
    def inviscid() -> "TurbulenceConfig":
        """
        Create INVISCID configuration — pure stretching only.

        **Physics:** Only the vortex-stretching term (ω·∇)u is solved.
        No SGS eddy viscosity, no molecular diffusion, no turbulence model.

        **Use when:**
        - Testing stretching formulations in isolation.
        - Running inviscid validation and convergence studies.

        Returns:
            TurbulenceConfig: INVISCID configuration instance
        """
        return TurbulenceConfig(model="INVISCID", flow_model="INVISCID")


# =========================================================
# STABILIZATION CONFIGURATION
# =========================================================


@dataclass(frozen=True)
class StabilizationConfig:
    """Particle-retention policy for a VPM domain.

    This class intentionally contains no strength filters, limiters,
    projections, splitting, or remeshing controls.  Stability is obtained by
    resolving the coupled equations and rejecting inadmissible time steps,
    while this policy only removes particles that have left a declared
    computational domain.
    """

    remove_particles_by_bounds: tuple[float, ...] | None = None
    """[xmin, xmax, ymin, ymax, zmin, zmax] — remove particles outside box.  None = disabled."""

    def __post_init__(self) -> None:
        """Normalize and validate the optional retention domain."""
        if self.remove_particles_by_bounds is not None:
            object.__setattr__(
                self,
                "remove_particles_by_bounds",
                tuple(float(value) for value in self.remove_particles_by_bounds),
            )
        if (
            self.remove_particles_by_bounds is not None
            and len(self.remove_particles_by_bounds) != 6
        ):
            raise ValueError("remove_particles_by_bounds must have 6 elements")

    # -- Factory methods -------------------------------------------------------
    @staticmethod
    def disabled() -> "StabilizationConfig":
        """Do not remove particles automatically."""
        return StabilizationConfig()

    @staticmethod
    def bounded_domain(bounds: list[float] | tuple[float, ...]) -> "StabilizationConfig":
        """Remove particles outside one declared six-coordinate domain."""
        return StabilizationConfig(remove_particles_by_bounds=tuple(bounds))


@dataclass(frozen=True)
class FilamentRefinementConfig:
    """Adaptive resolution for stretched Lagrangian vortex-line elements.

    At each refinement event, a particle whose strength exceeds
    ``max_strength_factor`` times its own lineage reference is bisected along
    its circulation direction.  Each child starts a new lineage reference.
    The transformation preserves circulation, total strength variation,
    volume, linear impulse, and the Gaussian kernel-corrected angular impulse
    without resetting the viscous core radius.
    """

    frequency: int = 0
    max_strength_factor: float = 2.0
    offset_fraction: float = 0.25
    max_particles: int | None = None
    energy_injection_tolerance: float = 1e-4
    energy_dissipation_tolerance: float = 1e-4
    enstrophy_transfer_tolerance: float = 1e-4
    helicity_transfer_tolerance: float = 1e-4
    cumulative_energy_tolerance: float = 1e-3
    cumulative_enstrophy_tolerance: float = 2e-2
    cumulative_helicity_tolerance: float = 2e-2
    cumulative_moment_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.frequency < 0:
            raise ValueError("filament-refinement frequency must be non-negative")
        if self.max_strength_factor <= 1.0:
            raise ValueError("max_strength_factor must be greater than one")
        if not 0.0 <= self.offset_fraction <= 0.5:
            raise ValueError("offset_fraction must be in [0, 0.5]")
        if self.max_particles is not None and self.max_particles <= 0:
            raise ValueError("max_particles must be positive or None")
        if self.energy_injection_tolerance < 0.0:
            raise ValueError("energy_injection_tolerance must be non-negative")
        if self.energy_dissipation_tolerance < 0.0:
            raise ValueError("energy_dissipation_tolerance must be non-negative")
        if self.enstrophy_transfer_tolerance < 0.0:
            raise ValueError("enstrophy_transfer_tolerance must be non-negative")
        if self.helicity_transfer_tolerance < 0.0:
            raise ValueError("helicity_transfer_tolerance must be non-negative")
        if self.cumulative_energy_tolerance < 0.0:
            raise ValueError("cumulative_energy_tolerance must be non-negative")
        if self.cumulative_enstrophy_tolerance < 0.0:
            raise ValueError("cumulative_enstrophy_tolerance must be non-negative")
        if self.cumulative_helicity_tolerance < 0.0:
            raise ValueError("cumulative_helicity_tolerance must be non-negative")
        if self.cumulative_moment_tolerance < 0.0:
            raise ValueError("cumulative_moment_tolerance must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.frequency > 0

    @staticmethod
    def disabled() -> "FilamentRefinementConfig":
        return FilamentRefinementConfig()

    @staticmethod
    def adaptive(
        *,
        frequency: int,
        max_strength_factor: float = 2.0,
        offset_fraction: float = 0.25,
        max_particles: int | None = None,
        energy_injection_tolerance: float = 1e-4,
        energy_dissipation_tolerance: float = 1e-4,
        enstrophy_transfer_tolerance: float = 1e-4,
        helicity_transfer_tolerance: float = 1e-4,
        cumulative_energy_tolerance: float = 1e-3,
        cumulative_enstrophy_tolerance: float = 2e-2,
        cumulative_helicity_tolerance: float = 2e-2,
        cumulative_moment_tolerance: float = 1e-4,
    ) -> "FilamentRefinementConfig":
        return FilamentRefinementConfig(
            frequency=frequency,
            max_strength_factor=max_strength_factor,
            offset_fraction=offset_fraction,
            max_particles=max_particles,
            energy_injection_tolerance=energy_injection_tolerance,
            energy_dissipation_tolerance=energy_dissipation_tolerance,
            enstrophy_transfer_tolerance=enstrophy_transfer_tolerance,
            helicity_transfer_tolerance=helicity_transfer_tolerance,
            cumulative_energy_tolerance=cumulative_energy_tolerance,
            cumulative_enstrophy_tolerance=cumulative_enstrophy_tolerance,
            cumulative_helicity_tolerance=cumulative_helicity_tolerance,
            cumulative_moment_tolerance=cumulative_moment_tolerance,
        )


@dataclass(frozen=True)
class DivergenceRelaxationConfig:
    """Atomic iterated Winckelmans projection with hard physics gates."""

    frequency: int = 0
    start_step: int = 0
    grid_spacing: float | None = None
    regularization: float = 0.1
    solver_rtol: float = 1e-5
    max_iterations: int = 30
    max_projection_sweeps: int = 3
    max_grid_nodes: int = 8_000_000
    max_correction_norm: float = 2e-2
    max_residual_ratio: float = 0.9
    max_direct_divergence_ratio: float = 0.98
    energy_tolerance: float = 1e-6
    enstrophy_tolerance: float = 1e-4
    helicity_tolerance: float = 1e-4
    variation_tolerance: float = 1e-3
    circulation_reference_scale: float | None = None
    linear_impulse_reference_scale: float | None = None
    angular_impulse_reference_scale: float | None = None
    circulation_reference_tolerance: float = 1e-3
    linear_impulse_reference_tolerance: float = 1e-2
    angular_impulse_reference_tolerance: float = 1e-2
    spectral_convergence_fraction: float = 0.1
    cumulative_energy_tolerance: float = 1e-3
    cumulative_enstrophy_tolerance: float = 2e-2
    cumulative_helicity_tolerance: float = 2e-2
    cumulative_variation_tolerance: float = 2e-2
    cumulative_moment_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.frequency < 0:
            raise ValueError("divergence-relaxation frequency must be non-negative")
        if self.start_step < 0:
            raise ValueError("divergence-relaxation start_step must be non-negative")
        if self.frequency > 0 and (self.grid_spacing is None or self.grid_spacing <= 0.0):
            raise ValueError("enabled divergence relaxation requires a positive grid_spacing")
        if self.regularization <= 0.0 or self.solver_rtol <= 0.0:
            raise ValueError("regularization and solver_rtol must be positive")
        if self.max_iterations < 1 or self.max_projection_sweeps < 1 or self.max_grid_nodes < 1:
            raise ValueError("iteration, projection-sweep, and grid-node limits must be positive")
        if self.max_correction_norm <= 0.0:
            raise ValueError("max_correction_norm must be positive")
        if not 0.0 < self.max_residual_ratio < 1.0:
            raise ValueError("max_residual_ratio must lie in (0, 1)")
        if not 0.0 < self.max_direct_divergence_ratio < 1.0:
            raise ValueError("max_direct_divergence_ratio must lie in (0, 1)")
        if not 0.0 < self.spectral_convergence_fraction <= 1.0:
            raise ValueError("spectral_convergence_fraction must lie in (0, 1]")
        for name in (
            "circulation_reference_scale",
            "linear_impulse_reference_scale",
            "angular_impulse_reference_scale",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided")
        reference_scales = (
            self.circulation_reference_scale,
            self.linear_impulse_reference_scale,
            self.angular_impulse_reference_scale,
        )
        if any(value is not None for value in reference_scales) and not all(
            value is not None for value in reference_scales
        ):
            raise ValueError("all three reference scales must be provided together")
        for name in (
            "energy_tolerance",
            "enstrophy_tolerance",
            "helicity_tolerance",
            "variation_tolerance",
            "circulation_reference_tolerance",
            "linear_impulse_reference_tolerance",
            "angular_impulse_reference_tolerance",
            "cumulative_energy_tolerance",
            "cumulative_enstrophy_tolerance",
            "cumulative_helicity_tolerance",
            "cumulative_variation_tolerance",
            "cumulative_moment_tolerance",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.frequency > 0

    @staticmethod
    def disabled() -> "DivergenceRelaxationConfig":
        return DivergenceRelaxationConfig()

    @staticmethod
    def constrained(
        *,
        frequency: int,
        grid_spacing: float,
        start_step: int = 0,
        regularization: float = 0.1,
        solver_rtol: float = 1e-5,
        max_iterations: int = 30,
        max_projection_sweeps: int = 3,
        max_grid_nodes: int = 8_000_000,
        max_correction_norm: float = 2e-2,
        max_residual_ratio: float = 0.9,
        max_direct_divergence_ratio: float = 0.98,
        energy_tolerance: float = 1e-6,
        enstrophy_tolerance: float = 1e-4,
        helicity_tolerance: float = 1e-4,
        variation_tolerance: float = 1e-3,
        circulation_reference_scale: float | None = None,
        linear_impulse_reference_scale: float | None = None,
        angular_impulse_reference_scale: float | None = None,
        circulation_reference_tolerance: float = 1e-3,
        linear_impulse_reference_tolerance: float = 1e-2,
        angular_impulse_reference_tolerance: float = 1e-2,
        spectral_convergence_fraction: float = 0.1,
        cumulative_energy_tolerance: float = 1e-3,
        cumulative_enstrophy_tolerance: float = 2e-2,
        cumulative_helicity_tolerance: float = 2e-2,
        cumulative_variation_tolerance: float = 2e-2,
        cumulative_moment_tolerance: float = 1e-4,
    ) -> "DivergenceRelaxationConfig":
        return DivergenceRelaxationConfig(
            frequency=frequency,
            start_step=start_step,
            grid_spacing=grid_spacing,
            regularization=regularization,
            solver_rtol=solver_rtol,
            max_iterations=max_iterations,
            max_projection_sweeps=max_projection_sweeps,
            max_grid_nodes=max_grid_nodes,
            max_correction_norm=max_correction_norm,
            max_residual_ratio=max_residual_ratio,
            max_direct_divergence_ratio=max_direct_divergence_ratio,
            energy_tolerance=energy_tolerance,
            enstrophy_tolerance=enstrophy_tolerance,
            helicity_tolerance=helicity_tolerance,
            variation_tolerance=variation_tolerance,
            circulation_reference_scale=circulation_reference_scale,
            linear_impulse_reference_scale=linear_impulse_reference_scale,
            angular_impulse_reference_scale=angular_impulse_reference_scale,
            circulation_reference_tolerance=circulation_reference_tolerance,
            linear_impulse_reference_tolerance=linear_impulse_reference_tolerance,
            angular_impulse_reference_tolerance=angular_impulse_reference_tolerance,
            spectral_convergence_fraction=spectral_convergence_fraction,
            cumulative_energy_tolerance=cumulative_energy_tolerance,
            cumulative_enstrophy_tolerance=cumulative_enstrophy_tolerance,
            cumulative_helicity_tolerance=cumulative_helicity_tolerance,
            cumulative_variation_tolerance=cumulative_variation_tolerance,
            cumulative_moment_tolerance=cumulative_moment_tolerance,
        )


# =========================================================
# VELOCITY CONFIGURATION
# =========================================================
@dataclass(frozen=True)
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

    theta: float = 0.3
    """Opening angle for Barnes-Hut treecode. Only used when method='TREECODE'.
      Smaller = more accurate but slower.
      - θ = 0.3: ~2% error, slower
      - θ = 0.5: ~5% error
      - θ = 0.7: ~15% error, faster"""

    multipole_order: Literal[1, 2, 3] = 1
    """Treecode far-field expansion order. 1 is the base monopole. 2 adds a
    circulation dipole moment per node, improving far-field gradient accuracy at
    extra per-node cost. 3 adds the quadrupole moment, allowing a larger theta
    at the same accuracy (usually the fastest accuracy/cost trade-off)."""

    sort_particle_targets: bool = False
    """Evaluate particle targets in Morton order for better GPU traversal
    coherence. Results are written back to original particle indices."""

    traversal_block_dim: int = 128
    """Taichi ``loop_config(block_dim=...)`` for tree traversal kernels.
    Set to 0 to leave Taichi's backend default untouched."""

    def __post_init__(self) -> None:
        method = self.method.upper()
        if method not in ("DIRECT", "TREECODE"):
            raise ValueError(f"velocity method must be DIRECT or TREECODE, got {self.method!r}")
        if not 0.0 < self.theta < 2.0:
            raise ValueError(f"velocity theta must be in (0, 2), got {self.theta!r}")
        if self.multipole_order not in (1, 2, 3):
            raise ValueError(
                f"velocity multipole_order must be 1, 2 or 3, got {self.multipole_order!r}"
            )
        if self.traversal_block_dim < 0:
            raise ValueError(
                f"velocity traversal_block_dim must be >= 0, got {self.traversal_block_dim!r}"
            )
        if method != self.method:
            object.__setattr__(self, "method", method)

    @staticmethod
    def direct() -> "VelocityConfig":
        """Direct O(N²) pairwise velocity computation (exact).

        Returns
        -------
        VelocityConfig
            Configuration with ``method`` set to ``'DIRECT'``.

        Examples
        --------
        .. code-block:: python

            >>> velocity = VelocityConfig.direct()
            >>> velocity.method
            'DIRECT'
        """
        return VelocityConfig(method="DIRECT")

    @staticmethod
    def treecode(
        theta: float = 0.3,
        multipole_order: Literal[1, 2, 3] = 1,
        sort_particle_targets: bool = False,
        traversal_block_dim: int = 128,
    ) -> "VelocityConfig":
        """Barnes-Hut treecode O(N log N) velocity computation (approximate).

        Args:
              theta: Opening angle parameter (0.3-0.7). Smaller = more accurate.
              multipole_order: 1 monopole, 2 dipole, 3 quadrupole far-node expansion.
              sort_particle_targets: Evaluate particle targets in Morton order.
              traversal_block_dim: Taichi traversal kernel block size; 0 = backend default.

        Returns
        -------
        VelocityConfig
            Configuration with ``method`` set to ``'TREECODE'`` and the given
            opening angle.

        Examples
        --------
        .. code-block:: python

            >>> velocity = VelocityConfig.treecode()
            >>> velocity.theta
            0.3

            >>> velocity = VelocityConfig.treecode(theta=0.5)
            >>> velocity.theta
            0.5
        """
        return VelocityConfig(
            method="TREECODE",
            theta=theta,
            multipole_order=multipole_order,
            sort_particle_targets=sort_particle_targets,
            traversal_block_dim=traversal_block_dim,
        )


# =========================================================
# SOLVER CONFIGURATION DATACLASS
# =========================================================


@dataclass(frozen=True)
class VPMSetup:
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

    # ---- TIME CONTROL ----
    time_step_size: float = DEFAULT_TIME_STEP
    """Time increment per simulation step [s]"""

    flow_time: float = 0.0
    """Initial simulation time [s]"""

    time_step: int = 0
    """Initial time step number"""

    # ---- PHYSICS CONFIGURATION ----
    time_integration: Literal["FRACTIONAL", "COUPLED"] = "FRACTIONAL"
    """How advection and vortex stretching are time-integrated.

    ``FRACTIONAL`` retains the legacy position-then-strength update.
    ``COUPLED`` treats ``(x, Gamma)`` as one ODE state and evaluates both
    right-hand sides at common RK stages.  It currently supports matching RK2
    or RK3 advection/stretching schemes and core-spreading (or no) viscous
    diffusion.
    """

    coupled_max_strain_increment: float = 0.08
    """Maximum ``dt_sub * ||S||_2`` used by automatic coupled subcycling."""

    coupled_max_advection_fraction: float = 0.25
    """Maximum particle displacement per substep as a fraction of spacing."""

    coupled_max_substeps: int = 128
    """Fail instead of silently filtering physics when a macro step needs more substeps."""

    advection: AdvectionConfig = field(default_factory=AdvectionConfig)
    """Configuration for advection term (scheme, etc.)."""

    stretching: StretchingConfig = field(default_factory=StretchingConfig.transposed)
    """Configuration for vortex stretching term."""

    viscous: ViscousConfig = field(default_factory=ViscousConfig.cs)
    """Configuration for viscous diffusion term."""

    turbulence: TurbulenceConfig = field(default_factory=TurbulenceConfig.dns)
    """Configuration for turbulence modeling (DNS/LES/INVISCID)."""

    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig.disabled)
    """Optional particle-retention domain used by wake and coupled cases."""

    filament_refinement: FilamentRefinementConfig = field(
        default_factory=FilamentRefinementConfig.disabled
    )
    """Adaptive conservative subdivision of stretched vortex-line elements."""

    divergence_relaxation: DivergenceRelaxationConfig = field(
        default_factory=DivergenceRelaxationConfig.disabled
    )
    """Atomic iterated constrained projection of the Gaussian particle field."""

    vlm: VLMSetup | None = None
    """Complete VLM definition. ``None`` selects a pure VPM simulation."""

    particles_kernel: Literal[
        "GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"
    ] = "GAUSSIAN"
    """Particle interaction kernel function type.

      ``HIGH_ORDER_GAUSSIAN`` — recommended production kernel: Gaussian with 2nd-order
      polynomial correction (2.5 − ρ²), improves far-field accuracy over plain Gaussian.
      """

    max_particles: int = MAX_PARTICLES
    """Maximum number of particles allowed in the simulation. Default: 500000"""

    max_targets: int = 200000
    """Maximum points in one velocity, vorticity, gradient, or sampler query.

    Taichi fields cannot be released independently. Allocate this capacity once
    when the solver starts so changing query sizes cannot accumulate abandoned
    device fields. Increase it before construction for larger sampler grids.
    """

    # ---- COMPUTATIONAL SETTINGS ----
    processing_unit: Literal["AUTO", "CPU", "VULKAN", "CUDA", "METAL"] = "AUTO"
    """Compute backend. Default ``'AUTO'`` selects the platform GPU and fails
    clearly when no GPU is available; CPU execution must be requested explicitly.

    * **macOS**             → Metal  (Apple's native GPU API)
    * **Linux / Windows**   → CUDA   (if an NVIDIA GPU + driver is found)
    * **Linux / Windows**   → Vulkan (universal fallback for AMD, Intel, NVIDIA)

    Override the automatic selection via the ``OPENONDA_PROCESSING_UNIT`` environment
    variable or by passing an explicit value here:

    * ``'AUTO'``       — best GPU for this platform, without CPU fallback
    * ``'VULKAN'``     — force Vulkan (Linux / Windows)
    * ``'CUDA'``       — force CUDA (requires NVIDIA GPU + driver)
    * ``'METAL'``      — force Metal (macOS only)
    * ``'CPU'``        — software rendering (no GPU required)
    """

    # ---- MONITORING AND DIAGNOSTICS ----
    logging_frequency: int = 0
    """Log flow diagnostics every N time steps (0 = disabled)."""

    export_flow_integrals: bool = True
    """Append computed global diagnostics to ``samples/flow_integrals.csv``."""

    export_discretization_health: bool = True
    """Also record the resolution metrics (particle overlap h/sigma, vorticity
    divergence error, Gamma-omega misalignment) with each diagnostics row.

    Invariant drift cannot detect a conservative-but-unresolved solution; these
    can.  Cost is one k-d tree plus a neighbour sum per diagnostics event, so it
    scales with ``logging_frequency`` and not with the step count."""

    log_mode: Literal["file", "tee", "console"] = "tee"
    """Solver output destination.

      - ``'tee'`` (default) — write to ``<backup_directory>/vpm*.log`` *and* keep
        the caller's console output working.
      - ``'file'`` — redirect to the log file only.  **This rebinds process-wide
        ``sys.stdout``/``sys.stderr`` for the lifetime of the solver**, so every
        ``print`` in the host program and in unrelated libraries is captured too.
        Only choose it for a batch run that owns the whole process; it was the
        default until the 2026-08 audit, where it silently swallowed diagnostics
        that had nothing to do with the solver.
      - ``'console'`` — no log file, streams untouched.

      All three restore the original streams via an ``atexit`` hook."""

    timing_frequency: int = 0
    """Print the cumulative runtime-profiling report every N time steps
    (0 = disabled). The per-step time line is always shown; this only controls
    the periodic ``RuntimeProfiler`` summary. ``Solver.print_timing()`` can be
    called manually at any time regardless of this setting."""

    # ---- BACKUP AND OUTPUT ----
    backup_frequency: int = 0
    """Save simulation state every N time steps (0 = disabled)."""

    backup_file_name: str = DEFAULT_BACKUP_FILENAME
    """Optional infix in backup names, producing prefixes like 'vpm_<name>_*'."""

    backup_directory: str = "solution"
    """Output directory for backups and sampled files."""

    clean: bool = False
    """If True, delete the backup_directory before starting the simulation."""

    # ---- PHYSICS PARAMETERS ----
    cutoff_radius_factor: float = DEFAULT_CUTOFF_RADIUS_FACTOR
    """Cutoff radius multiplier for particle interactions (performance optimization)"""

    precision: Literal["f32", "f64"] = "f32"
    """Floating-point precision of the particle fields.

      - ``'f32'`` (default) — **the supported production precision.**  Fast on
        GPU (Vulkan/CUDA); the 'Atomic add may lose precision' warnings are
        expected and acceptable.
      - ``'f64'`` — **EXPERIMENTAL and not end-to-end.**  It widens the particle
        container and the direct velocity summation, but three parts of the
        pipeline stay f32 regardless, so the delivered accuracy is well short of
        double precision:

          * the LBVH treecode is f32 throughout — fields, multipoles and host
            buffers — so ``velocity=VelocityConfig.treecode(...)`` is rejected
            with ``precision='f64'`` rather than silently downgraded;
          * the velocity-gradient and energy/helicity accumulators in
            ``numerics/kernels_common.py`` are hardcoded ``ti.f32``;
          * the Gaussian ``q``/``g`` use the Abramowitz & Stegun 7.1.26 erf
            approximation, whose max absolute error is 1.4e-7 — above f32 eps
            and nine orders above f64 eps.

        Requires ``processing_unit='CPU'`` (Vulkan/Metal have limited f64
        support).  Treat results as f32-accurate until the remaining paths are
        widened; see docs/reviews/2026-08-vpm-audit.md finding N-3."""

    random_seed: int = 42
    """Seed for Taichi's RNG (default 42).

      Only the Random Walk Method (``ViscousConfig(scheme='RWM')``) consumes it:
      its Brownian displacements are drawn from this stream.  A fixed seed makes
      a single run reproducible, but it also makes every run identical, so an
      RWM ensemble must vary this across members for the stochastic diffusion to
      average out.  Ignored on the Metal backend."""

    device_memory_fraction: float = 0.5
    """Fraction of GPU VRAM reserved for Taichi's internal memory pool (default 0.5).

      The remaining VRAM is available for external-array staging buffers used to
      transfer numpy arrays to/from the GPU.  If your simulation crashes with
      ``Failed to allocate ext arr buffer``, **lower** this value (e.g. 0.3–0.4)
      so that more VRAM is left for staging.

      Clamped to the range [0.1, 0.7].  Values above ~0.7 almost always cause
      allocation failures on Vulkan backends."""

    debug_mode: bool = False
    """Enable Taichi debug checks. This must be selected before solver construction."""

    background_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Free-stream background velocity vector [ux, uy, uz] in m/s."""

    verbose: bool = True
    """Enable verbose output (print particle shedding info, etc.)."""

    # ---- VELOCITY COMPUTATION ----
    velocity: VelocityConfig | None = None
    """Configuration for velocity field computation (direct vs treecode)."""

    # ---- SOLVER INSTANCES (Dependency Injection) ----
    panel_solver: Any | None = None
    """Panel solver instance for hybrid simulations."""

    # ---- FIELD SAMPLERS ----
    samplers: tuple[Any, ...] | None = None
    """List of field samplers (SurfaceSampler, LineSampler) called at logging_frequency intervals."""

    final_samplers: tuple[Any, ...] | None = None
    """Field samplers executed only when ``Solver.execute_final_samplers()`` is called."""

    body_stl: str | None = None
    """Path to body STL file for field sampler masking and geometry-aware output.
      Used by field samplers to mask interior points and project near-wall velocities.
      Can be absolute or relative to case directory."""

    vpm_domain_bounds: tuple[float, ...] | None = None
    """Optional ``[xmin, xmax, ymin, ymax, zmin, zmax]`` of the VPM domain.
      When set, the DVH diffusion grid is pre-allocated to cover this domain
      (if the memory cost is acceptable) and re-allocation is capped at these
      bounds.  Passed automatically by the coupler."""

    def __post_init__(self):
        """Post-initialization validation and setup."""

        if len(self.background_velocity) != 3:
            raise ValueError("background_velocity must contain three coordinates")
        object.__setattr__(
            self,
            "background_velocity",
            tuple(float(value) for value in self.background_velocity),
        )
        if self.samplers is not None:
            object.__setattr__(self, "samplers", tuple(self.samplers))
        if self.final_samplers is not None:
            object.__setattr__(self, "final_samplers", tuple(self.final_samplers))
        if self.vpm_domain_bounds is not None:
            object.__setattr__(self, "vpm_domain_bounds", tuple(self.vpm_domain_bounds))
        output_name = self.backup_file_name.strip()
        if output_name and (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", output_name) is None
            or output_name.startswith(("vpm_", "vlm_"))
        ):
            raise ValueError(
                "backup_file_name must be a filename-safe infix without a path, "
                "extension, or solver prefix"
            )
        object.__setattr__(self, "backup_file_name", output_name)

        if self.velocity is None:
            # The GPU LBVH currently implements Gaussian and Winckelmans
            # regularisation. Corrected/super-Gaussian solvers remain fully
            # GPU-capable through exact direct summation instead of failing
            # later during the first update_state().
            if self.particles_kernel.upper() in ("GAUSSIAN", "WINCKELMANS"):
                velocity = VelocityConfig.treecode(theta=0.3)
            else:
                velocity = VelocityConfig.direct()
            object.__setattr__(self, "velocity", velocity)

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

        integration = self.time_integration.upper()
        if integration not in {"FRACTIONAL", "COUPLED"}:
            raise ValueError("time_integration must be 'FRACTIONAL' or 'COUPLED'")
        if integration == "COUPLED":
            advection_scheme = self.advection.scheme.upper()
            stretching_scheme = self.stretching.scheme.upper()
            if not self.stretching.enabled:
                raise ValueError("COUPLED time integration requires stretching to be enabled")
            if advection_scheme != stretching_scheme or advection_scheme not in {"RK2", "RK3"}:
                raise ValueError(
                    "COUPLED time integration requires matching RK2 or RK3 "
                    "advection and stretching schemes"
                )
            if self.viscous.scheme.upper() not in {"NONE", "CS"}:
                raise ValueError(
                    "COUPLED time integration currently supports only NONE or CS diffusion"
                )
        if self.coupled_max_strain_increment <= 0.0:
            raise ValueError("coupled_max_strain_increment must be positive")
        if self.coupled_max_advection_fraction <= 0.0:
            raise ValueError("coupled_max_advection_fraction must be positive")
        if self.coupled_max_substeps < 1:
            raise ValueError("coupled_max_substeps must be at least one")

        # Frequency validation
        if self.logging_frequency < 0:
            raise ValueError("logging_frequency must be non-negative")

        if self.timing_frequency < 0:
            raise ValueError("timing_frequency must be non-negative")

        if self.max_particles < 1:
            raise ValueError("max_particles must be at least 1")

        if self.max_targets < 1:
            raise ValueError("max_targets must be at least 1")

        if self.log_mode not in {"file", "tee", "console"}:
            raise ValueError("log_mode must be 'file', 'tee', or 'console'")

        # Backup frequency validation
        if self.backup_frequency < 0:
            raise ValueError("backup_frequency must be non-negative")

        # Processing unit validation
        valid_processing_units = [
            "AUTO",
            "CPU",
            "VULKAN",
            "CUDA",
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
        # The treecode implements only a subset of the regularization kernels
        # (it carries its own Taichi q/zeta).
        _treecode = self.velocity is not None and self.velocity.method == "TREECODE"
        if _treecode and _kernel_up not in TREECODE_SUPPORTED_KERNELS:
            raise ValueError(
                f"particles_kernel='{_kernel_up}' cannot be used with "
                f"velocity method 'TREECODE': the treecode implements only "
                f"{list(TREECODE_SUPPORTED_KERNELS)}. Either switch to "
                f"VelocityConfig.direct() or choose a supported kernel."
            )
        # The treecode's fields, multipoles and host buffers are f32 throughout,
        # so this pairing delivers f32 accuracy under an f64 label.
        if _treecode and self.precision == "f64":
            raise ValueError(
                "precision='f64' cannot be used with velocity method 'TREECODE': "
                "the treecode is f32 end-to-end, so the velocity field would be "
                "single precision regardless. Use VelocityConfig.direct() for an "
                "f64 run, or precision='f32' with the treecode."
            )
        if self.filament_refinement.enabled and _kernel_up != "GAUSSIAN":
            raise ValueError(
                "filament refinement currently requires the GAUSSIAN kernel so its "
                "angular-impulse correction and exact transfer audit use the "
                "same declared regularization"
            )
        if self.divergence_relaxation.enabled and _kernel_up != "GAUSSIAN":
            raise ValueError("divergence relaxation currently requires the GAUSSIAN kernel")
        if self.divergence_relaxation.enabled and not self.filament_refinement.enabled:
            raise ValueError(
                "divergence relaxation requires filament refinement so the "
                "interpolation solve cannot hide inadequate vortex-line resolution"
            )
        if (
            self.filament_refinement.max_particles is not None
            and self.filament_refinement.max_particles > self.max_particles
        ):
            raise ValueError(
                "filament-refinement max_particles cannot exceed VPMSetup.max_particles"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the solver configuration, including
            all nested dataclass fields serialised as nested dictionaries.
        """
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
            "time_integration": self.time_integration,
            "coupled_max_strain_increment": self.coupled_max_strain_increment,
            "coupled_max_advection_fraction": self.coupled_max_advection_fraction,
            "coupled_max_substeps": self.coupled_max_substeps,
            "advection": _as_dict(self.advection),
            "stretching": _as_dict(self.stretching),
            "viscous": _as_dict(self.viscous),
            "turbulence": _as_dict(self.turbulence),
            "stabilization": _as_dict(self.stabilization),
            "filament_refinement": _as_dict(self.filament_refinement),
            "divergence_relaxation": _as_dict(self.divergence_relaxation),
            # Runtime geometry and kinematics are not particle-state backup data.
            "vlm": None,
            "particles_kernel": self.particles_kernel,
            "max_particles": self.max_particles,
            "max_targets": self.max_targets,
            "logging_frequency": self.logging_frequency,
            "export_flow_integrals": self.export_flow_integrals,
            "export_discretization_health": self.export_discretization_health,
            "log_mode": self.log_mode,
            "timing_frequency": self.timing_frequency,
            "backup_frequency": self.backup_frequency,
            "backup_file_name": self.backup_file_name,
            "backup_directory": self.backup_directory,
            "cutoff_radius_factor": self.cutoff_radius_factor,
            "precision": self.precision,
            "random_seed": self.random_seed,
            "device_memory_fraction": self.device_memory_fraction,
            "processing_unit": self.processing_unit,
            "debug_mode": self.debug_mode,
            "clean": self.clean,
            "background_velocity": list(self.background_velocity)
            if self.background_velocity
            else [0.0, 0.0, 0.0],
            "verbose": self.verbose,
            "velocity": _as_dict(self.velocity) if self.velocity else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VPMSetup":
        """Create configuration from dictionary.

        Returns
        -------
        VPMSetup
            Configuration object reconstructed from the dictionary.
        """
        values = dict(data)
        nested_types = {
            "advection": AdvectionConfig,
            "stretching": StretchingConfig,
            "viscous": ViscousConfig,
            "turbulence": TurbulenceConfig,
            "stabilization": StabilizationConfig,
            "filament_refinement": FilamentRefinementConfig,
            "divergence_relaxation": DivergenceRelaxationConfig,
            "velocity": VelocityConfig,
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
    ) -> "VPMSetup":
        """
        Create a standard viscous flow simulation configuration.

        **Default physics:**
          - Stretching: disabled
          - Viscous:    Core Spreading (CS)
          - Turbulence: DNS (no subgrid model)
          - Advection:  RK3 (SSP-RK3)
          - Velocity:   Treecode (θ = 0.5)

        Args:
              time_step_size: Simulation time step [s]
              background_velocity: Free-stream velocity [m/s]
              viscous: Viscous scheme configuration (None = CS)
              **kwargs: Additional parameters to override defaults

        Returns:
              VPMSetup: Initialized configuration object

        Examples:
              >>> # Default viscous flow
              >>> config = VPMSetup.viscous_flow_simulation()
              >>> config.time_step_size
              0.01

              >>> config = VPMSetup.viscous_flow_simulation(
              ...     time_step_size=0.005, background_velocity=(10.0, 0.0, 0.0)
              ... )
              >>> config.time_step_size
              0.005
        """
        if viscous is None:
            viscous = ViscousConfig.cs()

        stretching = kwargs.pop("stretching", StretchingConfig.disabled())

        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            background_velocity=list(background_velocity),
            **kwargs,
        )

    @staticmethod
    def dns_simulation(time_step_size: float = 0.01, **kwargs) -> "VPMSetup":
        """
        Create a Direct Numerical Simulation (DNS) configuration.

        **Default physics:**
          - Advection:  RK3 (SSP-RK3)
          - Stretching: TRANSPOSED mode, RK3 (SSP-RK3)
          - Viscous:    Core Spreading (CS)
          - Turbulence: DNS (molecular viscosity only, no SGS model)
          - Velocity:   Treecode (θ = 0.5)
          - Stabilization: disabled

        Args:
              time_step_size: Simulation time step [s]
              **kwargs: Additional parameters to override defaults

        Returns:
              VPMSetup: Initialized configuration object

        Examples:
              >>> # Default DNS
              >>> config = VPMSetup.dns_simulation()
              >>> config.time_step_size
              0.01

              >>> config = VPMSetup.dns_simulation(
              ...     time_step_size=0.001, processing_unit="CPU"
              ... )
              >>> config.time_step_size
              0.001
        """
        # Extract common kwargs if provided to avoid multiple values for the same argument
        stretching = kwargs.pop("stretching", StretchingConfig.transposed())
        viscous = kwargs.pop("viscous", ViscousConfig.cs())
        turbulence = kwargs.pop("turbulence", TurbulenceConfig.dns())

        return VPMSetup(
            time_step_size=time_step_size,
            stretching=stretching,
            viscous=viscous,
            turbulence=turbulence,
            **kwargs,
        )

    @staticmethod
    def les_simulation(time_step_size: float = 0.01, cs: float = 0.17, **kwargs) -> "VPMSetup":
        """
        Create a Large Eddy Simulation (LES) configuration with the static
        Smagorinsky SGS model (k-equilibrium formulation).

        **Default physics:**
          - Advection:  RK3 (SSP-RK3)
          - Stretching: TRANSPOSED mode, RK3 (SSP-RK3)
          - Viscous:    Core Spreading (CS)
          - Turbulence: LES_SMAGORINSKY with C_s = 0.17, C_e = 1.048
          - Velocity:   Treecode (θ = 0.5)
          - Stabilization: disabled

        Args:
              time_step_size: Simulation time step [s]
              cs: Smagorinsky coefficient (default 0.17)
              **kwargs: Additional parameters to override defaults

        Returns:
              VPMSetup: Initialized configuration object

        Examples:
              >>> # Default LES
              >>> config = VPMSetup.les_simulation()
              >>> config.time_step_size
              0.01

              >>> config = VPMSetup.les_simulation(
              ...     time_step_size=0.005, cs=0.15
              ... )
              >>> config.time_step_size
              0.005
        """
        stretching = kwargs.pop("stretching", StretchingConfig.transposed())
        viscous = kwargs.pop("viscous", ViscousConfig.cs())
        # Use les_smagorinsky or similar if available, else fallback
        turbulence = kwargs.pop("turbulence", TurbulenceConfig.les_smagorinsky(cs=cs))

        return VPMSetup(
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
    def load_from_file(cls, filename: str) -> "VPMSetup":
        """Load configuration from JSON file.

        Returns
        -------
        VPMSetup
            Configuration object deserialised from the JSON file.
        """
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


# =========================================================
# UTILITY DECORATORS AND HELPERS
# =========================================================
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


# =========================================================
# PYDANTIC MODELS FOR SERIALIZATION AND STATE MANAGEMENT
# =========================================================
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
    advection_scheme: str = Field(default="RK3", description="Advection time integration scheme")
    stretching_scheme: str = Field(default="RK3", description="Stretching time integration scheme")

    processing_unit: str = Field(default="AUTO", description="Computation backend")
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
    timing_frequency: int = Field(
        default=0, description="Runtime-profile report frequency in steps"
    )
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
        valid_units = {"AUTO", "CPU", "VULKAN", "CUDA", "METAL"}
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

        Examples:
              >>> state = SolverState.from_solver(solver)
              >>> state.time_step_size
              0.01
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
                StretchingConfig,
                TurbulenceConfig,
                ViscousConfig,
                VPMSetup,
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

            # Reconstruct full VPMSetup
            config = VPMSetup(
                time_step_size=self.time_step_size,
                flow_time=self.flow_time,
                time_step=self.time_step,
                advection=advection,
                stretching=stretching,
                viscous=viscous,
                turbulence=turbulence,
                particles_kernel=self.particles_kernel,
                logging_frequency=self.logging_frequency,
                timing_frequency=self.timing_frequency,
                backup_frequency=self.backup_frequency,
                backup_file_name=self.backup_file_name,
                backup_directory=self.backup_directory,
                processing_unit=self.processing_unit,
                # precision?
            )

            # Create new solver instance with config
            new_solver = Solver(setup=config)

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

        Examples:
              >>> state = ParticlesState.from_particles(particles)
              >>> len(state.positions)
              10000
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


# =========================================================
# UTILITY FUNCTIONS FOR FLOW MODEL SETTING
# =========================================================
def SetFlowModel(psys, flow_model: str):
    """
    Set flow model and configure associated parameters.
    Note: Validation is already done in VPMSetup, so this just sets the model.
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
    "VPMSetup",
    "SolverState",
    "ParticlesState",
    "CachedParticleProperty",
    "SetFlowModel",
    "ViscousConfig",
    "StabilizationConfig",
]
