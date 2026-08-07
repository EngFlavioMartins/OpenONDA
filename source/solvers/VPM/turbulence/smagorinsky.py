"""
Smagorinsky Subgrid-Scale Turbulence Model for VPM Solver.
==========================================================
Kinetic-energy equilibrium Smagorinsky model: nu_t = C_k Δ √k_eq
where k_eq = C_k Δ² |S|² / C_e  and  Δ = V^(1/3) (particle volume cube root).

Key design choices:
  - Filter width  Δ = V^(1/3): pins the LES scale to the Lagrangian resolution
    (not to σ, which grows artificially during Core Spreading).
  - Strain-only measure  |S|² = 2 S_ij S_ij: excludes rotation so the model
    does not activate inside laminar vortex cores where |Ω| >> |S|.
  - No separate circulation damping: Core Spreading with nu_eff = nu + nu_t
    already models the full SGS diffusion; a second drain would double-count.

The turbulence model does not modify vortex stretching or particle strengths.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026 / Refactored May 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import MAX_PARTICLES, SMAGORINSKY_CONSTANT


@ti.data_oriented
class SmagorinskyModel:
    """
    Kinetic-energy equilibrium Smagorinsky subgrid-scale model.

    Eddy viscosity: nu_t = C_k · Δ · √k_eq
    Equilibrium SGS energy: k_eq = C_k · Δ² · |S|² / C_e

    Derivation (local production–dissipation balance):
        P = nu_t |S|² = C_e k^(3/2) / Δ  →  k_eq = C_k Δ² |S|² / C_e
        nu_t = C_k Δ √k_eq  = (C_k^{3/2} / √C_e) Δ² |S|

    This is equivalent to the classical model nu_t = (C_s Δ)² |S| with:
        C_s² = C_k √(C_k / C_e)

    With OpenFOAM defaults C_k=0.094, C_e=1.048 this recovers C_s ≈ 0.168.

    Filter width:  Δ = V^(1/3)  (cube root of particle volume).
    Strain measure:  |S|² = 2 S_ij S_ij  (symmetric part of ∇u only).
    """

    def __init__(
        self,
        max_particles: int = MAX_PARTICLES,
        kernel_type: str = "GAUSSIAN",
        cs: float = SMAGORINSKY_CONSTANT,
        ce: float = 1.048,
    ):
        """
        Initialize the Smagorinsky SGS eddy-viscosity model.

        Stores model constants, computes the derived ``ck`` parameter from
        ``cs`` and ``ce`` (see class docstring for the relation), and
        pre-allocates Taichi fields for the filter width and strain-rate
        magnitude.

        Args:
            max_particles: Maximum number of particles for field allocation.
            kernel_type: Base regularization kernel type (e.g. ``"GAUSSIAN"``).
            cs: Smagorinsky constant (default from ``SMAGORINSKY_CONSTANT``).
            ce: Model dissipation constant (default 1.048, OpenFOAM standard).
        """
        self.LES_filter_type = "SMAGORINSKY"
        self.max_particles = max_particles
        self.cs = cs
        self.ce = ce
        self.ck = (cs**2 * ce**0.5) ** (2.0 / 3.0)

        # Pre-allocate fields needed for eddy-viscosity computation
        self._delta_field = ti.field(dtype=ti.f32, shape=max_particles)
        self._Sij_norm_field = ti.field(dtype=ti.f32, shape=max_particles)

    def initialize(self, particles):
        pass

    def compute(self, particles, dt: float = None):
        """
        Main compute step for Smagorinsky model.

        Computes turbulent eddy viscosity: nu_t = C_k · Δ · √k_eq
        where k_eq = C_k · Δ² · |S|² / C_e  and  Δ = V^(1/3).

        Args:
            particles: Particle system (particles.strain_rate must be current).
            dt: Unused; retained for API compatibility.
        """
        N = len(particles)
        if N == 0:
            return

        self._compute_filter_sizes_kernel(particles.volume, self._delta_field, N)
        self._compute_strain_rate_magnitude_kernel(particles.strain_rate, self._Sij_norm_field, N)
        self._compute_smagorinsky_kernel(
            self._delta_field,
            self._Sij_norm_field,
            N,
            particles.viscosity,
            particles.viscosity_turbulent,
            particles.viscosity_effective,
            self.ck,
            self.ce,
        )

    def __str__(self):
        lines = [
            "  Model Type    : Smagorinsky",
            f"  C_s          : {self.cs:.4f}",
            f"  C_k          : {self.ck:.6f}  [C_k = (C_s^2 (C_e)^(1/2))^(2/3)]",
            f"  C_e          : {self.ce:.4f}",
            "  Filter width  : Vp^(1/3)",
        ]
        return "\n".join(lines)

    @ti.kernel
    def _compute_filter_sizes_kernel(
        self, volumes: ti.template(), deltas: ti.template(), N: ti.i32
    ):
        """Compute LES filter width as the cube root of particle volume: Δ = V^(1/3)."""
        for i in range(N):
            v = volumes[i]
            deltas[i] = ti.pow(v, 1.0 / 3.0) if v > 0.0 else 0.0

    @ti.kernel
    def _compute_strain_rate_magnitude_kernel(
        self, strain_rate: ti.template(), Sij_norm: ti.template(), N: ti.i32
    ):
        """
        Compute strain-rate magnitude |S| = √(2 S_ij S_ij) from the
        symmetric strain tensor S_ij = ½(∂_i u_j + ∂_j u_i).
        """
        for i in range(N):
            magnitude = 0.0
            for a in ti.static(range(3)):
                for b in ti.static(range(3)):
                    S_ab = strain_rate[i][a, b]
                    magnitude += S_ab * S_ab
            Sij_norm[i] = ti.sqrt(2.0 * magnitude)

    @ti.kernel
    def _compute_smagorinsky_kernel(
        self,
        deltas: ti.template(),
        Sij_norm: ti.template(),
        N: ti.i32,
        viscosity: ti.template(),
        viscosities_t: ti.template(),
        viscosities_eff: ti.template(),
        ck: ti.f32,
        ce: ti.f32,
    ):
        """Kinetic-energy equilibrium Smagorinsky model: nu_t = C_k·Δ·√k_eq."""
        for i in range(N):
            delta = deltas[i]
            s_mag = Sij_norm[i]
            k_eq = ck * delta * delta * s_mag * s_mag / ce
            nu_t = ck * delta * ti.sqrt(ti.max(k_eq, 0.0))

            if ti.math.isnan(nu_t) or ti.math.isinf(nu_t):
                nu_t = 0.0

            viscosities_t[i] = nu_t
            viscosities_eff[i] = viscosity[i] + nu_t
