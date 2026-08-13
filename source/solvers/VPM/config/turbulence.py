"""Turbulence/sub-grid-scale configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import constants as constants_module


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
