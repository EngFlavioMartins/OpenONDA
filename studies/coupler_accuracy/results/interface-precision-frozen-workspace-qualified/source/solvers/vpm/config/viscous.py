"""Viscous-diffusion configuration for the VPM solver."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

_THRESHOLD_MODES = {
    "budget",
    "relative_max",
    "absolute",
}


@dataclass(frozen=True)
class ViscousConfig:
    """Configure molecular viscous diffusion and particle regeneration.

    Parameters
    ----------
    scheme : {'CS', 'RWM', 'NONE', 'DVH', 'GBD'}, default='CS'
        Core spreading, random walk, no diffusion, Diffused Vortex
        Hydrodynamics, or Grid-Based Diffusion. Names are normalized uppercase.
    core_radius_ratio : float, default=2.5
        Positive dimensionless regenerated-particle ratio ``sigma/h``.
    dvh_grid_spacing, gbd_grid_spacing : float or None
        Positive regeneration-grid spacings ``h`` in m. Factory methods default
        these to ``particle_spacing``.
    dvh_domain_padding, gbd_domain_padding : float, default=3.0
        Number of grid cells padded beyond the active particle bounds.
    dvh_threshold, gbd_threshold : float, default=0.01
        Node-pruning controls interpreted by the corresponding threshold mode.
        ``budget`` is the allowed discarded L1 fraction, ``relative_max`` is a
        fraction of peak node strength, and ``absolute`` is a threshold in m³/s.
    dvh_threshold_mode, gbd_threshold_mode : {'budget', 'relative_max', 'absolute'}
        Interpretation of the associated threshold.
    gbd_max_nodes, dvh_max_nodes : int or None
        Optional positive caps on regenerated active nodes, additionally
        bounded by the solver particle capacity.
    gbd_remeshing_kernel : {'M4_PRIME', 'LAGRANGE6'}, default='M4_PRIME'
        Particle-to-grid scatter. The six-point Lagrange option is experimental,
        non-positivity-preserving, and requires at least four padding cells.
    dvh_support_radius_ratio : {3, 4, 5}, default=4
        Compact DVH heat-kernel support radius divided by grid spacing.
    kinematic_viscosity : float or None
        Non-negative molecular viscosity ``nu`` in m²/s. A positive value is
        required by active diffusion schemes.
    particle_spacing : float or None
        Positive representative particle spacing in m, used for accuracy and
        regeneration defaults.

    Raises
    ------
    ValueError
        If a choice, physical scale, support, padding, or node cap is invalid.

    Notes
    -----
    ``CS`` grows particle cores deterministically; ``RWM`` applies stochastic
    Brownian displacement; ``DVH`` and ``GBD`` replace the accepted particle
    cloud on a regular grid. RWM and DVH require spatially uniform effective
    viscosity; use GBD for LES variable-viscosity diffusion. Configuration is
    immutable and does not itself mutate particles.

    Examples
    --------
    >>> viscous = ViscousConfig.gbd(
    ...     particle_spacing=0.05, kinematic_viscosity=1e-5,
    ...     threshold=1e-4, threshold_mode="budget",
    ... )
    """

    scheme: Literal["CS", "RWM", "NONE", "DVH", "GBD"] = "CS"

    core_radius_ratio: float = 2.5
    """Regenerated-particle core radius divided by regeneration-grid spacing."""

    dvh_grid_spacing: float | None = None
    """DVH regeneration-grid spacing [m]."""

    dvh_domain_padding: float = 3.0
    """DVH domain padding in grid cells."""

    dvh_threshold: float = 0.01
    """DVH pruning threshold interpreted by ``dvh_threshold_mode``."""

    dvh_threshold_mode: str = "budget"
    """DVH pruning mode."""

    gbd_grid_spacing: float | None = None
    """GBD regeneration-grid spacing [m]."""

    gbd_domain_padding: float = 3.0
    """GBD domain padding in grid cells."""

    gbd_threshold: float = 0.01
    """GBD pruning threshold interpreted by ``gbd_threshold_mode``."""

    gbd_threshold_mode: str = "budget"
    """GBD pruning mode."""

    gbd_max_nodes: int | None = None
    """Optional cap on surviving GBD grid nodes."""

    gbd_remeshing_kernel: str = "M4_PRIME"
    """Scatter kernel: M4_PRIME or experimental six-point LAGRANGE6."""

    dvh_support_radius_ratio: int = 4
    """DVH compact-support radius ``R_d / h``; allowed values are 3, 4, and 5."""

    dvh_max_nodes: int | None = None
    """Optional cap on surviving DVH grid nodes."""

    kinematic_viscosity: float | None = None
    """Molecular kinematic viscosity ``kinematic_viscosity`` [m²/s]."""

    particle_spacing: float | None = None
    """Representative inter-particle spacing ``h`` [m]."""

    def __post_init__(self) -> None:
        scheme = self.scheme.upper()
        if scheme not in {"CS", "RWM", "NONE", "DVH", "GBD"}:
            raise ValueError(f"Invalid viscous scheme: {self.scheme!r}")
        object.__setattr__(self, "scheme", scheme)

        if self.core_radius_ratio <= 0.0 or not math.isfinite(self.core_radius_ratio):
            raise ValueError("core_radius_ratio must be finite and positive")

        for name in ("particle_spacing", "dvh_grid_spacing", "gbd_grid_spacing"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when set")

        if self.kinematic_viscosity is not None and (
            not math.isfinite(self.kinematic_viscosity) or self.kinematic_viscosity < 0.0
        ):
            raise ValueError("kinematic_viscosity must be finite and non-negative when set")

        if self.dvh_support_radius_ratio not in {3, 4, 5}:
            raise ValueError(
                f"dvh_support_radius_ratio must be 3, 4, or 5 (got {self.dvh_support_radius_ratio})"
            )

        if self.dvh_threshold_mode not in _THRESHOLD_MODES:
            raise ValueError(f"dvh_threshold_mode must be one of {sorted(_THRESHOLD_MODES)}")
        if self.gbd_threshold_mode not in _THRESHOLD_MODES:
            raise ValueError(f"gbd_threshold_mode must be one of {sorted(_THRESHOLD_MODES)}")
        if self.gbd_remeshing_kernel not in {"M4_PRIME", "LAGRANGE6"}:
            raise ValueError("gbd_remeshing_kernel must be M4_PRIME or LAGRANGE6")
        if self.gbd_remeshing_kernel == "LAGRANGE6" and self.gbd_domain_padding < 4:
            raise ValueError("LAGRANGE6 requires at least four grid cells of padding")
        if self.gbd_max_nodes is not None and self.gbd_max_nodes < 1:
            raise ValueError("gbd_max_nodes must be positive when set")
        if self.dvh_max_nodes is not None and self.dvh_max_nodes < 1:
            raise ValueError("dvh_max_nodes must be positive when set")

    def rwm_accuracy_time_step_size(self) -> float:
        """Return the RWM accuracy bound ``h²/(4*nu)`` in s.

        Raises
        ------
        ValueError
            If particle spacing is unset or viscosity is unset/non-positive.
        """
        if self.particle_spacing is None:
            raise ValueError("particle_spacing must be set for the RWM accuracy check")
        if self.kinematic_viscosity is None or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive for the RWM accuracy check")
        return self.particle_spacing**2 / (4.0 * self.kinematic_viscosity)

    def dvh_required_time_step_size(self) -> float:
        """Return required DVH increment ``beta*R_d²/(4*nu)`` in s.

        ``R_d = dvh_support_radius_ratio * dvh_grid_spacing``. Raises
        :class:`ValueError` when spacing or positive viscosity is unavailable.
        """
        from .constants import _DVH_BETA

        if self.dvh_grid_spacing is None:
            raise ValueError("dvh_grid_spacing must be set to a positive value")
        if self.kinematic_viscosity is None or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive for DVH")

        support_radius = self.dvh_support_radius_ratio * self.dvh_grid_spacing
        return _DVH_BETA * support_radius * support_radius / (4.0 * self.kinematic_viscosity)

    def gbd_max_time_step_size(self) -> float:
        """Return the explicit GBD stability bound ``h²/(6*nu)`` in s.

        Raises :class:`ValueError` when GBD spacing or positive viscosity is
        unavailable.
        """
        if self.gbd_grid_spacing is None:
            raise ValueError("gbd_grid_spacing must be set to a positive value")
        if self.kinematic_viscosity is None or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be positive for GBD")
        return self.gbd_grid_spacing**2 / (6.0 * self.kinematic_viscosity)

    @staticmethod
    def cs(
        kinematic_viscosity: float | None = None,
        particle_spacing: float | None = None,
        core_radius_ratio: float = 2.5,
    ) -> ViscousConfig:
        """Configure deterministic Gaussian core spreading.

        ``kinematic_viscosity`` is in m²/s, ``particle_spacing`` in m, and
        ``core_radius_ratio`` is the dimensionless regenerated ``sigma/h``.
        """
        return ViscousConfig(
            scheme="CS",
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=particle_spacing,
            core_radius_ratio=core_radius_ratio,
        )

    @staticmethod
    def rwm(
        kinematic_viscosity: float | None = None,
        particle_spacing: float | None = None,
        core_radius_ratio: float = 2.5,
    ) -> ViscousConfig:
        """Configure stochastic random-walk molecular diffusion.

        ``kinematic_viscosity`` is in m²/s, ``particle_spacing`` in m, and
        ``core_radius_ratio`` is dimensionless. RWM is incompatible with LES
        variable effective viscosity.
        """
        return ViscousConfig(
            scheme="RWM",
            kinematic_viscosity=kinematic_viscosity,
            particle_spacing=particle_spacing,
            core_radius_ratio=core_radius_ratio,
        )

    @staticmethod
    def inviscid(
        particle_spacing: float | None = None,
        core_radius_ratio: float = 2.5,
    ) -> ViscousConfig:
        """Configure inviscid particle evolution without molecular diffusion."""
        return ViscousConfig(
            scheme="NONE",
            particle_spacing=particle_spacing,
            core_radius_ratio=core_radius_ratio,
        )

    @staticmethod
    def dvh(
        particle_spacing: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        dvh_support_radius_ratio: int = 4,
        kinematic_viscosity: float | None = None,
        max_nodes: int | None = None,
        core_radius_ratio: float = 2.5,
    ) -> ViscousConfig:
        """Configure heat-kernel DVH regeneration.

        ``particle_spacing`` sets both nominal particle and DVH grid spacing in
        m; ``kinematic_viscosity`` is m²/s; ``padding`` and ``threshold`` follow
        the constructor conventions. DVH may accumulate macro-steps until its
        required diffusion interval is reached.
        """
        return ViscousConfig(
            scheme="DVH",
            particle_spacing=particle_spacing,
            dvh_grid_spacing=particle_spacing,
            dvh_domain_padding=padding,
            dvh_threshold=threshold,
            dvh_threshold_mode=threshold_mode,
            dvh_support_radius_ratio=dvh_support_radius_ratio,
            kinematic_viscosity=kinematic_viscosity,
            dvh_max_nodes=max_nodes,
            core_radius_ratio=core_radius_ratio,
        )

    @staticmethod
    def gbd(
        particle_spacing: float | None = None,
        padding: float = 20.0,
        threshold: float = 1e-5,
        threshold_mode: str = "budget",
        kinematic_viscosity: float | None = None,
        max_nodes: int | None = None,
        core_radius_ratio: float = 2.5,
        remeshing_kernel: str = "M4_PRIME",
    ) -> ViscousConfig:
        """Return Grid-Based Diffusion configuration.

        ``LAGRANGE6`` is an experimental six-point, degree-five Lagrange
        scatter added for the vortex-interaction core-transport controls.
        It reduces remapping damping compared with M4' on those controls;
        it is not positivity-preserving and still requires refinement.
        The molecular Laplacian and viscosity coefficient are unchanged.
        M4' remains the default. LAGRANGE6 needs at least four padding cells.
        """
        return ViscousConfig(
            scheme="GBD",
            particle_spacing=particle_spacing,
            gbd_grid_spacing=particle_spacing,
            gbd_domain_padding=padding,
            gbd_threshold=threshold,
            gbd_threshold_mode=threshold_mode,
            kinematic_viscosity=kinematic_viscosity,
            gbd_max_nodes=max_nodes,
            gbd_remeshing_kernel=remeshing_kernel,
            core_radius_ratio=core_radius_ratio,
        )
