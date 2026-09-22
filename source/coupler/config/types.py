"""Configuration of the FVM--VPM coupling operations."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CouplerSetup:
    """Configure FVM--VPM exchange without duplicating solver-owned physics.

    Users normally construct one instance after configuring the two native
    solvers and pass it to :class:`~source.coupler.solver.FVMVPMCoupler`.
    Fluid properties, mesh geometry, time-step sizes, particle spacing, and
    wall definitions remain owned by the FVM/VPM cases and are cross-checked
    during coupler initialization.

    Parameters
    ----------
    freestream_velocity : list[float], default=[1.0, 0.0, 0.0]
        Finite Cartesian background velocity ``[u, v, w]`` in m/s. The VPM
        freestream must match this value.
    transfer_method : {'buffered_m4_renewal', 'common_lattice', 'projected_renewal'}
        Algorithm used to replace the FVM-authoritative portion of the VPM
        particle cloud. ``common_lattice`` is the default general path;
        ``buffered_m4_renewal`` currently requires VPM GBD diffusion.
    transfer_region_bounds : tuple[float, float, float, float, float, float] or None
        Cartesian ``(xmin, xmax, ymin, ymax, zmin, zmax)`` bounds in m. The
        box must be contained by the FVM boundary; ``None`` uses that complete
        box except where an explicit region is required by the selected path.
    eta_blend_width : float, default=0.0
        Width in m of the C1 FVM-authority ramp measured inward from the
        transfer faces. Zero requests a sharp authority transition.
    vpm_only_width : float, default=0.0
        Inner face band in m retained entirely by the VPM. A positive value
        requires ``0 < vpm_only_width < eta_blend_width``.
    transfer_vorticity_cutoff : float, default=0.05
        Interior soft-pruning threshold in 1/s. The stable-renewal path
        converts this to particle strength with volume ``h**3`` in 3D or
        ``h**2 * L`` for planar span ``L``, and blends it to the VPM GBD
        vorticity floor at the release surface.
    transfer_amplification_cap : float, default=1.8
        Dimensionless upper gain, at least one, for represented-state
        corrections in stable renewal.
    transfer_diagnostic_interval_steps : int, default=1
        Positive number of accepted coupling steps between expensive transfer
        diagnostics and their log records.
    transfer_discretization_error_limit : float, default=0.08
        Maximum accepted relative discretization/closure error in ``(0, 1]``.
    renewal_vorticity_error_limit : float, default=5e-3
        Positive relative tolerance for independent projected-vorticity
        verification.
    renewal_velocity_error_limit : float, default=1e-3
        Positive relative tolerance for normal velocity at the ownership
        boundary.
    renewal_gaussian_tail_cutoff : float, default=1e-8
        Relative Gaussian basis weight in ``(0, 1)`` below which sparse
        projection entries are omitted.
    renewal_solver_tolerance : float, default=1e-9
        Positive relative LSMR tolerance for projected absolute strengths.
    coupling_patch : str, default='numericalBoundary'
        Name of the outer FVM boundary patch sampled from the VPM. Face
        centres are in m and outward face normals follow FVM owner orientation.
    boundary_condition_mode : str
        Boundary trace applied to ``coupling_patch`` during FVM subcycling.
        ``vorticity_mixed_pressure_gradient`` combines normal velocity and
        tangential normal derivative with a prescribed pressure gradient.
        This opt-in mode requires accurate VPM acceleration/viscous data.
    fvm_consistency_width : float, default=0.0
        Width in m of the optional resolved-scale consistency band between the
        transfer region and outer FVM boundary. Zero disables it.
    backup_interval_steps : int, default=1
        Accepted coupling steps between atomic coupled backups. Zero disables
        scheduled backups.

    Raises
    ------
    TypeError
        If NumPy cannot interpret the freestream or bounds as numeric values.
    ValueError
        If a vector, bound, choice, count, or tolerance violates the contracts
        above. Containment in the FVM box is checked later by
        :meth:`validate_transfer_region_box`.

    Notes
    -----
    The dataclass is frozen, so policy cannot be reassigned after construction.
    The caller-provided freestream list is validated but retained; do not mutate
    that list after construction. Coupling transfers particle-strength vectors
    ``Gamma = omega * V`` in m³/s, not pointwise vorticity in 1/s.

    Examples
    --------
    >>> setup = CouplerSetup(
    ...     freestream_velocity=[1.0, 0.0, 0.0],
    ...     transfer_method="common_lattice",
    ...     transfer_region_bounds=(-2.0, 4.0, -2.0, 2.0, -1.0, 1.0),
    ...     coupling_patch="numericalBoundary",
    ... )
    """

    # FLOW STATE
    freestream_velocity: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    """Freestream velocity (u, v, w) in m/s; must be a finite three-component vector."""

    # VORTICITY TRANSFER (FVM -> VPM)
    transfer_method: Literal[
        "buffered_m4_renewal",
        "common_lattice",
        "projected_renewal",
    ] = "common_lattice"
    """FVM-to-VPM state transfer. ``buffered_m4_renewal`` is the whole-belt
    M4' renewal method with a VPM-owned release buffer.
    ``projected_renewal`` and ``common_lattice`` remain experimental paths."""
    transfer_region_bounds: tuple[float, float, float, float, float, float] | None = None
    """FVM-authoritative replacement region ``(xmin, xmax, ymin, ymax, zmin,
    zmax)``. It must lie inside the FVM domain; ``None`` uses the full domain."""
    eta_blend_width: float = 0.0
    """Width (m) of the C1 FVM-authority ramp measured inward from the transfer
    faces. The stable renewal leaves the face and its release buffer under VPM
    authority; zero selects a hard interior authority profile."""
    vpm_only_width: float = 0.0
    """Width (m) just inside the transfer faces where stable renewal keeps
    FVM authority exactly zero. It must be smaller than ``eta_blend_width``."""
    transfer_vorticity_cutoff: float = 0.05
    """Interior stable-renewal soft-prune threshold in vorticity units (1/s)."""
    transfer_amplification_cap: float = 1.8
    """Maximum gain used by the stable represented-state correction."""
    transfer_diagnostic_interval_steps: int = 1
    """Replacement steps between transfer diagnostics; at least one."""
    transfer_discretization_error_limit: float = 0.08
    """Maximum Gaussian-particle vorticity-divergence error admitted before transfer."""
    renewal_vorticity_error_limit: float = 5.0e-3
    """Maximum independent relative vorticity mismatch for projected renewal."""
    renewal_velocity_error_limit: float = 1.0e-3
    """Maximum relative normal-velocity mismatch at the ownership boundary."""
    renewal_gaussian_tail_cutoff: float = 1.0e-8
    """Relative Gaussian kernel weight omitted from the production sparse operator."""
    renewal_solver_tolerance: float = 1.0e-9
    """Relative LSMR tolerance for the sparse absolute-strength solve."""

    # VPM BOUNDARY-CONDITION TRACE ON THE FVM
    coupling_patch: str = "numericalBoundary"
    """Name of the FVM patch on which the VPM boundary condition is imposed."""
    boundary_condition_mode: Literal[
        "dirichlet",
        "characteristic",
        "directional_outflow",
        "pressure_gradient",
        "vorticity_mixed",
        "vorticity_mixed_pressure_gradient",
    ] = "dirichlet"
    """VPM boundary-condition mode: dirichlet, characteristic, directional_outflow,
    pressure_gradient, vorticity_mixed, or vorticity_mixed_pressure_gradient."""
    fvm_consistency_width: float = 0.0
    """Width (m) of the resolved-scale VPM-to-FVM consistency band measured
    inward from the outer FVM boundary. Zero disables the band. A positive
    value must fit entirely outside ``transfer_region_bounds``."""
    interface_iterations: int = 1
    """Maximum fixed-predictor FVM/renewal sweeps; one keeps explicit exchange.
    Iteration requires mixed vorticity boundaries, buffered M4 renewal, no
    consistency band, and FVM output schedules aligned with coupling times."""
    interface_acceleration: Literal["none", "aitken"] = "none"
    """Experimental safeguarded trace acceleration; does not relax convergence gates."""
    interface_normal_tolerance: float = 1.0e-6
    """Area-weighted RMS normal-velocity residual tolerance in m/s."""
    interface_gradient_tolerance: float = 1.0e-6
    """Area-weighted RMS tangential-gradient residual tolerance in 1/s."""
    # RUN-LEVEL OPERATIONAL
    backup_interval_steps: int = 1
    """Coupling steps between automatic backups; non-negative (0 disables backups)."""

    def __post_init__(self) -> None:
        if (
            isinstance(self.interface_iterations, bool)
            or not isinstance(self.interface_iterations, int)
            or self.interface_iterations < 1
        ):
            raise ValueError("interface_iterations must be a positive integer")
        if self.interface_acceleration not in {"none", "aitken"}:
            raise ValueError("interface_acceleration must be 'none' or 'aitken'")
        if self.interface_acceleration != "none" and self.interface_iterations < 3:
            raise ValueError("Interface acceleration requires at least three interface sweeps")
        for value in (self.interface_normal_tolerance, self.interface_gradient_tolerance):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("Interface tolerances must be finite and positive")
        if self.interface_iterations > 1 and (
            self.boundary_condition_mode != "vorticity_mixed"
            or self.transfer_method != "buffered_m4_renewal"
            or self.fvm_consistency_width != 0.0
        ):
            raise ValueError(
                "Interface iteration requires vorticity_mixed, buffered_m4_renewal, and no consistency band"
            )
        freestream_velocity = np.asarray(self.freestream_velocity, dtype=np.float64)
        if freestream_velocity.shape != (3,) or not np.all(np.isfinite(freestream_velocity)):
            raise ValueError("freestream_velocity must be a finite three-component vector")

        if self.boundary_condition_mode not in {
            "dirichlet",
            "characteristic",
            "directional_outflow",
            "pressure_gradient",
            "vorticity_mixed",
            "vorticity_mixed_pressure_gradient",
        }:
            raise ValueError(
                "boundary_condition_mode must be 'dirichlet', 'characteristic', "
                "'directional_outflow', 'pressure_gradient', 'vorticity_mixed', "
                "or 'vorticity_mixed_pressure_gradient'"
            )

        if self.transfer_method not in {
            "buffered_m4_renewal",
            "common_lattice",
            "projected_renewal",
        }:
            raise ValueError(
                "transfer_method must be 'buffered_m4_renewal', 'common_lattice', "
                "or 'projected_renewal'"
            )

        if self.transfer_region_bounds is not None:
            transfer_region_bounds = np.asarray(self.transfer_region_bounds, dtype=np.float64)
            if transfer_region_bounds.shape != (6,) or not np.all(
                np.isfinite(transfer_region_bounds)
            ):
                raise ValueError("transfer_region_bounds must contain six finite bounds")
            if np.any(transfer_region_bounds[1::2] <= transfer_region_bounds[::2]):
                raise ValueError(
                    "Each transfer_region_bounds upper bound must exceed its lower bound"
                )

        if self.backup_interval_steps < 0:
            raise ValueError("backup_interval_steps must be non-negative")
        if not np.isfinite(self.fvm_consistency_width) or self.fvm_consistency_width < 0.0:
            raise ValueError("fvm_consistency_width must be finite and non-negative")
        if not np.isfinite(self.eta_blend_width) or self.eta_blend_width < 0.0:
            raise ValueError("eta_blend_width must be finite and non-negative")
        if not np.isfinite(self.vpm_only_width) or self.vpm_only_width < 0.0:
            raise ValueError("vpm_only_width must be finite and non-negative")
        if self.eta_blend_width == 0.0 and self.vpm_only_width != 0.0:
            raise ValueError("vpm_only_width requires a positive eta_blend_width")
        if self.eta_blend_width > 0.0 and self.vpm_only_width >= self.eta_blend_width:
            raise ValueError("vpm_only_width must be smaller than eta_blend_width")
        if not np.isfinite(self.transfer_vorticity_cutoff) or self.transfer_vorticity_cutoff < 0.0:
            raise ValueError("transfer_vorticity_cutoff must be finite and non-negative")
        if (
            not np.isfinite(self.transfer_amplification_cap)
            or self.transfer_amplification_cap < 1.0
        ):
            raise ValueError("transfer_amplification_cap must be at least one")
        if self.transfer_diagnostic_interval_steps < 1:
            raise ValueError("transfer_diagnostic_interval must be at least one")
        if (
            not np.isfinite(self.transfer_discretization_error_limit)
            or not 0.0 < self.transfer_discretization_error_limit <= 1.0
        ):
            raise ValueError("transfer_discretization_error_limit must lie in (0, 1]")
        if (
            not np.isfinite(self.renewal_vorticity_error_limit)
            or self.renewal_vorticity_error_limit <= 0.0
        ):
            raise ValueError("renewal_vorticity_error_limit must be finite and positive")
        if (
            not np.isfinite(self.renewal_velocity_error_limit)
            or self.renewal_velocity_error_limit <= 0.0
        ):
            raise ValueError("renewal_velocity_error_limit must be finite and positive")
        if (
            not np.isfinite(self.renewal_gaussian_tail_cutoff)
            or not 0.0 < self.renewal_gaussian_tail_cutoff < 1.0
        ):
            raise ValueError("renewal_gaussian_tail_cutoff must lie between zero and one")
        if not np.isfinite(self.renewal_solver_tolerance) or self.renewal_solver_tolerance <= 0.0:
            raise ValueError("renewal_solver_tolerance must be finite and positive")
        if self.transfer_method == "projected_renewal" and self.eta_blend_width != 0.0:
            raise ValueError("projected_renewal requires eta_blend_width=0")

    @property
    def freestream_velocity_vector(self) -> np.ndarray:
        """Return a new ``float64`` Cartesian freestream vector in m/s.

        The returned array has shape ``(3,)`` and never aliases the mutable
        list supplied to the constructor.
        """
        return np.asarray(self.freestream_velocity, dtype=np.float64)

    def validate_transfer_region_box(
        self,
        fvm_box: tuple[float, float, float, float, float, float] | np.ndarray,
    ) -> None:
        """Validate transfer and consistency regions against the FVM box.

        Parameters
        ----------
        fvm_box : tuple[float, float, float, float, float, float] or ndarray
            Outer FVM bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in m.

        Raises
        ------
        ValueError
            If the configured transfer region extends outside ``fvm_box``, if
            projected renewal has no explicit region, or if the consistency
            band does not fit between every pair of corresponding faces.

        Notes
        -----
        This method reads configuration only and does not build a lattice or
        mutate either solver.
        """
        outer = np.asarray(fvm_box, dtype=np.float64)
        if self.transfer_method == "projected_renewal" and self.transfer_region_bounds is None:
            raise ValueError(
                "projected_renewal requires explicit transfer_region_bounds with room "
                "inside the FVM domain for its runtime GBD guard"
            )
        inner = (
            outer
            if self.transfer_region_bounds is None
            else np.asarray(self.transfer_region_bounds, dtype=np.float64)
        )
        if np.any(inner[::2] < outer[::2]) or np.any(inner[1::2] > outer[1::2]):
            raise ValueError("transfer_region_bounds must be contained within the FVM domain")
        if self.fvm_consistency_width > 0.0:
            if self.transfer_region_bounds is None:
                raise ValueError("fvm_consistency_width requires explicit transfer_region_bounds")
            margins = np.column_stack((inner[::2] - outer[::2], outer[1::2] - inner[1::2]))
            if np.any(margins + 1.0e-14 < self.fvm_consistency_width):
                raise ValueError(
                    "fvm_consistency_width must fit between every transfer-region face "
                    "and the outer FVM boundary"
                )

    def to_dict(self) -> dict[str, object]:
        """Return coupling-owned settings for restart identity checks.

        Returns
        -------
        dict[str, object]
            New nested mapping under the ``"coupler"`` key. Bounds are
            labelled by axis rather than stored positionally. No solver state,
            particle arrays, or derived runtime values are included.
        """
        transfer_region_bounds = None
        if self.transfer_region_bounds is not None:
            transfer_region_bounds = dict(
                zip(
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"),
                    self.transfer_region_bounds,
                    strict=True,
                )
            )
        return {
            "coupler": {
                "freestream_velocity": self.freestream_velocity,
                "backup_interval_steps": self.backup_interval_steps,
                "coupling_patch": self.coupling_patch,
                "boundary_condition_mode": self.boundary_condition_mode,
                "fvm_consistency_width": self.fvm_consistency_width,
                "interface_iterations": self.interface_iterations,
                "interface_acceleration": self.interface_acceleration,
                "interface_normal_tolerance": self.interface_normal_tolerance,
                "interface_gradient_tolerance": self.interface_gradient_tolerance,
                "transfer_method": self.transfer_method,
                "transfer_region_bounds": transfer_region_bounds,
                "eta_blend_width": self.eta_blend_width,
                "vpm_only_width": self.vpm_only_width,
                "transfer_vorticity_cutoff": self.transfer_vorticity_cutoff,
                "transfer_amplification_cap": self.transfer_amplification_cap,
                "transfer_diagnostic_interval_steps": self.transfer_diagnostic_interval_steps,
                "transfer_discretization_error_limit": (self.transfer_discretization_error_limit),
                "renewal_vorticity_error_limit": self.renewal_vorticity_error_limit,
                "renewal_velocity_error_limit": self.renewal_velocity_error_limit,
                "renewal_gaussian_tail_cutoff": self.renewal_gaussian_tail_cutoff,
                "renewal_solver_tolerance": self.renewal_solver_tolerance,
            }
        }
