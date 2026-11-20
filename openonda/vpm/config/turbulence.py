"""Turbulence-model configuration for the VPM solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from . import constants as constants_module


@dataclass(frozen=True, slots=True)
class TurbulenceConfig:
    """Configure the VPM turbulence/LES closure.

    Parameters
    ----------
    model : {'dns', 'smagorinsky', 'sigma', 'spectral_cusp', 'inviscid'}, default='dns'
        ``dns`` applies no subgrid closure, ``smagorinsky`` computes an
        equilibrium eddy viscosity, ``sigma`` the sigma-model eddy
        viscosity, ``spectral_cusp`` the spectral eddy viscosity with the
        Kraichnan cusp on the grid-based-diffusion lattice, and ``inviscid``
        disables viscous/SGS modeling at this layer.
    smagorinsky_coefficient : float, default=0.17
        Non-negative dimensionless ``C_s`` in the filter-width eddy-viscosity
        model.
    filter_width : float or None, default=None
        Fixed LES filter width in metres. None retains the local particle
        volume^(1/3) rule. A fixed width separates the closure scale from
        quadrature changes during spatial refinement or redistribution.
    sigma_coefficient : float, default=1.35
        Non-negative sigma-model constant (Nicoud et al. 2011).
    consistency_angle : float or None, default=None
        Reference angle in degrees, in ``(0, 90)``, of the consistency switch
        ``sin^2(theta) / (sin^2(theta) + sin^2(theta_0))``: ``theta`` is the angle
        between a particle's strength and the resolved vorticity at the particle.
        None applies the eddy viscosity everywhere.
    subgrid_transport : {'core_spreading', 'strength_exchange'}, default='core_spreading'
        ``core_spreading`` adds the eddy viscosity to the viscous scheme;
        ``strength_exchange`` diffuses strengths between neighbouring particles
        after each step and leaves the cores to the molecular viscosity.
    kolmogorov_constant : float, default=1.4
        Positive Kolmogorov constant ``C_K`` of the ``spectral_cusp`` model.
    cutoff_wavenumber : float or None, default=None
        Positive cut-off ``k_c`` of the ``spectral_cusp`` model in rad/m; None
        uses the lattice Nyquist wavenumber ``pi/h``.

    Attributes
    ----------
    flow_model : {'dns', 'les', 'inviscid'}
        Derived solver category, excluded from the constructor.

    Raises
    ------
    ValueError
        If ``model`` is unsupported or a coefficient violates its range.

    Notes
    -----
    LES updates per-particle eddy viscosity from the accepted velocity-gradient
    field. With core-spreading transport it does not modify vortex
    strength directly; strength exchange conserves the total strength exactly.
    """

    model: Literal["dns", "smagorinsky", "sigma", "spectral_cusp", "inviscid"] = "dns"
    """Turbulence-model identifier."""

    smagorinsky_coefficient: float = constants_module.SMAGORINSKY_CONSTANT
    """Smagorinsky coefficient ``C_s``."""

    filter_width: float | None = None
    """Optional fixed LES filter width in metres."""

    sigma_coefficient: float = 1.35
    """Sigma-model constant ``C``."""

    consistency_angle: float | None = None
    """Consistency-switch reference angle in degrees, or None."""

    subgrid_transport: Literal["core_spreading", "strength_exchange"] = "core_spreading"
    """How the eddy viscosity acts on the particles."""

    kolmogorov_constant: float = 1.4
    """Kolmogorov constant of the spectral cusp model."""

    cutoff_wavenumber: float | None = None
    """Cut-off wavenumber of the spectral cusp model in rad/m, or None for ``pi/h``."""

    flow_model: str = field(default="dns", init=False)
    """Solver flow-model category derived from ``model``."""

    def __post_init__(self) -> None:
        model = self.model
        valid_models = {"dns", "smagorinsky", "sigma", "spectral_cusp", "inviscid"}
        if model not in valid_models:
            raise ValueError(
                f"Invalid turbulence model {self.model!r}; expected one of {sorted(valid_models)}"
            )
        if not math.isfinite(self.smagorinsky_coefficient) or self.smagorinsky_coefficient < 0.0:
            raise ValueError("smagorinsky_coefficient must be finite and non-negative")
        if self.filter_width is not None and (
            not math.isfinite(self.filter_width) or self.filter_width <= 0.0
        ):
            raise ValueError("filter_width must be finite and positive or None")
        if not math.isfinite(self.sigma_coefficient) or self.sigma_coefficient < 0.0:
            raise ValueError("sigma_coefficient must be finite and non-negative")
        if not math.isfinite(self.kolmogorov_constant) or self.kolmogorov_constant <= 0.0:
            raise ValueError("kolmogorov_constant must be finite and positive")
        if self.cutoff_wavenumber is not None and (
            not math.isfinite(self.cutoff_wavenumber) or self.cutoff_wavenumber <= 0.0
        ):
            raise ValueError("cutoff_wavenumber must be finite and positive or None")
        if self.consistency_angle is not None and not 0.0 < self.consistency_angle < 90.0:
            raise ValueError("consistency_angle must lie in (0, 90) degrees or be None")
        transport = self.subgrid_transport
        if transport not in {"core_spreading", "strength_exchange"}:
            raise ValueError(
                f"Invalid subgrid_transport {self.subgrid_transport!r}; expected "
                "'core_spreading' or 'strength_exchange'"
            )
        flow_model = {
            "dns": "dns",
            "smagorinsky": "les",
            "sigma": "les",
            "spectral_cusp": "les",
            "inviscid": "inviscid",
        }[model]

        object.__setattr__(self, "flow_model", flow_model)

    @staticmethod
    def dns() -> TurbulenceConfig:
        """Return DNS configuration without an SGS closure."""
        return TurbulenceConfig(model="dns")

    @staticmethod
    def smagorinsky(
        smagorinsky_coefficient: float = constants_module.SMAGORINSKY_CONSTANT,
        filter_width: float | None = None,
        consistency_angle: float | None = None,
        subgrid_transport: str = "core_spreading",
    ) -> TurbulenceConfig:
        """Return the equilibrium Smagorinsky LES configuration."""
        return TurbulenceConfig(
            model="smagorinsky",
            smagorinsky_coefficient=smagorinsky_coefficient,
            filter_width=filter_width,
            consistency_angle=consistency_angle,
            subgrid_transport=subgrid_transport,
        )

    @staticmethod
    def sigma(
        sigma_coefficient: float = 1.35,
        consistency_angle: float | None = 3.0,
        subgrid_transport: str = "strength_exchange",
        filter_width: float | None = None,
    ) -> TurbulenceConfig:
        """Return the sigma-model LES, by default switched and applied by strength exchange.

        The consistency switch keeps the eddy viscosity off particles whose strength
        is aligned with the resolved vorticity, as in resolved laminar vortices, and
        turns it on where the particle field stops representing the vorticity.
        """
        return TurbulenceConfig(
            model="sigma",
            sigma_coefficient=sigma_coefficient,
            consistency_angle=consistency_angle,
            subgrid_transport=subgrid_transport,
            filter_width=filter_width,
        )

    @staticmethod
    def spectral_cusp(
        kolmogorov_constant: float = 1.4,
        cutoff_wavenumber: float | None = None,
    ) -> TurbulenceConfig:
        """Return the spectral eddy viscosity with the Kraichnan cusp, for grid-based diffusion.

        Every GBD step multiplies each Fourier mode of the lattice vorticity by
        ``exp(-nu_t(k|k_c) k^2 dt)``, with ``nu_t(k|k_c) = C_K^(-3/2) [0.441 +
        15.2 exp(-3.03 k_c/k)] sqrt(E(k_c)/k_c)`` (Chollet & Lesieur 1981). The
        amplitude follows the energy that has reached the cut-off, so laminar
        vortices are left alone. The particles carry no eddy viscosity.
        """
        return TurbulenceConfig(
            model="spectral_cusp",
            kolmogorov_constant=kolmogorov_constant,
            cutoff_wavenumber=cutoff_wavenumber,
        )

    @property
    def particle_eddy_viscosity(self) -> bool:
        """Whether the model computes an eddy viscosity on every particle."""
        return self.model in {"smagorinsky", "sigma"}

    @staticmethod
    def equilibrium_smagorinsky(
        subgrid_kinetic_energy_coefficient: float = 0.094,
        subgrid_dissipation_coefficient: float = 1.048,
    ) -> TurbulenceConfig:
        """Configure Smagorinsky LES from equilibrium coefficients.

        The current implementation uses

        ``C_s = C_k**(3/4) / C_e**(1/4)``.
        """
        if (
            not math.isfinite(subgrid_kinetic_energy_coefficient)
            or subgrid_kinetic_energy_coefficient < 0.0
        ):
            raise ValueError("subgrid_kinetic_energy_coefficient must be finite and non-negative")
        if (
            not math.isfinite(subgrid_dissipation_coefficient)
            or subgrid_dissipation_coefficient <= 0.0
        ):
            raise ValueError("subgrid_dissipation_coefficient must be finite and positive")

        smagorinsky_coefficient = (
            subgrid_kinetic_energy_coefficient**0.75 / subgrid_dissipation_coefficient**0.25
        )
        return TurbulenceConfig.smagorinsky(
            smagorinsky_coefficient=smagorinsky_coefficient,
        )

    @staticmethod
    def inviscid() -> TurbulenceConfig:
        """Return inviscid turbulence-model configuration."""
        return TurbulenceConfig(model="inviscid")
