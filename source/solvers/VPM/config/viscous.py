"""Viscous-diffusion configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import field_validator


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
        visc = ViscousConfig.dvh(particle_spacing=0.01, viscosity=1e-5, dvh_rd_ratio=4)

        # GBD
        visc = ViscousConfig.gbd(particle_spacing=0.01, viscosity=1e-5)

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

    core_radius_ratio: float = 2.5
    """Core radius σ assigned to regenerated particles, in units of the regen
    grid spacing h (DVH and GBD schemes).  This SILENTLY OVERRIDES whatever
    radius the particles carried before the regen — in coupled FVM-VPM runs it
    MUST match the coupler's ``vpm_core_radius_ratio``, otherwise the Beale
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

    def rwm_accuracy_time_step_size(self) -> float:
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
        particle_spacing = self.characteristic_distance
        if particle_spacing is None or particle_spacing <= 0:
            raise ValueError(
                "characteristic_distance must be set to a positive value for RWM accuracy check."
            )
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for RWM accuracy check.")
        return particle_spacing * particle_spacing / (4.0 * nu)

    def dvh_required_time_step_size(self) -> float:
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
        from .constants import _DVH_BETA

        particle_spacing = self.dvh_grid_spacing
        if particle_spacing is None or particle_spacing <= 0:
            raise ValueError("dvh_grid_spacing must be set to a positive value.")
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for DVH.")
        R_d = self.dvh_rd_ratio * particle_spacing
        return _DVH_BETA * R_d * R_d / (4.0 * nu)

    def gbd_max_time_step_size(self) -> float:
        """CFL upper bound for GBD: dt_max = h² / (6nu).

        Returns
        -------
        float  Maximum stable time-step [s].
        """
        particle_spacing = self.gbd_grid_spacing
        if particle_spacing is None or particle_spacing <= 0:
            raise ValueError("gbd_grid_spacing must be set to a positive value.")
        nu = self.viscosity
        if nu is None or nu <= 0:
            raise ValueError("viscosity must be set to a positive value for GBD.")
        return particle_spacing * particle_spacing / (6.0 * nu)

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
        particle_spacing: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        threshold_window: int = 3,
        dvh_rd_ratio: int = 4,
        viscosity: float | None = None,
        max_nodes: int | None = None,
        cap_abs_fraction: float = 0.99,
        core_radius_ratio: float = 2.5,
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
        particle_spacing: Grid spacing [m].  Should match the desired inter-particle
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
                  core_radius_ratio: Core radius σ = ratio·h assigned to regenerated
                      particles.  Default 2.5. Lower toward 1.5 to avoid
                      over-smearing the reconstructed field (see the field docstring).
        """
        if not isinstance(dvh_rd_ratio, int) or dvh_rd_ratio not in (3, 4, 5):
            raise ValueError(
                f"dvh_rd_ratio must be an integer in {{3, 4, 5}}, got {dvh_rd_ratio!r}"
            )
        return ViscousConfig(
            scheme="DVH",
            dvh_grid_spacing=particle_spacing,
            dvh_domain_padding=padding,
            dvh_threshold=threshold,
            dvh_threshold_mode=threshold_mode,
            regen_threshold_window=threshold_window,
            dvh_rd_ratio=dvh_rd_ratio,
            viscosity=viscosity,
            dvh_max_nodes=max_nodes,
            regen_cap_abs_fraction=cap_abs_fraction,
            core_radius_ratio=core_radius_ratio,
        )

    @staticmethod
    def gbd(
        particle_spacing: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        threshold_window: int = 3,
        viscosity: float | None = None,
        max_nodes: int | None = None,
        cap_abs_fraction: float = 0.99,
        core_radius_ratio: float = 2.5,
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
        particle_spacing: Grid spacing [m].
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
                  core_radius_ratio: Core radius σ = ratio·h assigned to regenerated
                      particles.  Default 2.5. Lower toward 1.5 to avoid
                      over-smearing the reconstructed field (see the field docstring).
        """
        return ViscousConfig(
            scheme="GBD",
            gbd_grid_spacing=particle_spacing,
            gbd_domain_padding=padding,
            gbd_threshold=threshold,
            gbd_threshold_mode=threshold_mode,
            regen_threshold_window=threshold_window,
            viscosity=viscosity,
            gbd_max_nodes=max_nodes,
            regen_cap_abs_fraction=cap_abs_fraction,
            core_radius_ratio=core_radius_ratio,
        )
