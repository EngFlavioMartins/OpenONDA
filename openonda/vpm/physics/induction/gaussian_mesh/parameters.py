"""Dimensionless controls for the optional Gaussian field mesh.

The auxiliary broadening is an algorithmic split, not a physical core change:
local correction restores actual source cores. These controls contain no body
size or case name. Resolving them is NOT an accuracy tail error bound. Finite
interpolation, omitted local correction and omitted images require separate
checks; no mesh ratio substitutes for those checks or solver particle state limits.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


def _ratio(value: object, name: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number (got {value!r})")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number (got {value})")
    return value


@dataclass(frozen=True, slots=True)
class ResolvedGaussianMesh:
    """Auxiliary lengths derived from an actual core snapshot."""

    tau: float
    spacing: float
    correction_cutoff: float
    order: int


@dataclass(frozen=True, slots=True)
class GaussianMeshParameters:
    """Case-independent, explicit discretization controls.

    Four auxiliary cells per broadened core is the conservative resolution
    used in finite-field refinement qualification. Increasing spacing/tau
    changes accuracy, not just performance. Correction radius alone is not
    validation of its omitted tail. No automatic tolerance changes occur here.

    Parameters
    ----------
    broadening_ratio : float, default=3.0
        Broadened Gaussian width ``tau`` divided by the largest particle core;
        at least 1, and at least the kernel's Gaussian partner width.
    spacing_over_tau : float, default=0.25
        Auxiliary mesh spacing divided by ``tau`` (four cells per ``tau``).
    correction_radius_over_tau : float, default=5.0
        Radius of the local core correction divided by ``tau``.
    order : {4, 6, 8, 10}, default=10
        Order of the mesh interpolation.
    """

    broadening_ratio: float = 3.0
    spacing_over_tau: float = 0.25
    correction_radius_over_tau: float = 5.0
    order: int = 10

    def __post_init__(self) -> None:
        for name in ("broadening_ratio", "spacing_over_tau", "correction_radius_over_tau"):
            object.__setattr__(self, name, _ratio(getattr(self, name), name))
        if self.broadening_ratio < 1:
            raise ValueError(
                "broadening_ratio must be at least 1 to cover every source core "
                f"(got {self.broadening_ratio})"
            )
        if (
            isinstance(self.order, bool)
            or not isinstance(self.order, Integral)
            or self.order not in (4, 6, 8, 10)
        ):
            raise ValueError(f"order must be 4, 6, 8 or 10 (got {self.order!r})")
        object.__setattr__(self, "order", int(self.order))

    def resolve(self, source_core: np.ndarray) -> "ResolvedGaussianMesh":
        """Derive lengths from ALL cores without retaining or modifying them.

        Resolve again after core mutation. This result is not an identity for
        a mutable source set. Empty fields need no mesh; the solver handles
        them before resolution.
        """
        original = np.asarray(source_core)
        if original.ndim != 1 or not len(original) or original.dtype.kind not in "fiu":
            raise ValueError("a nonempty vector of positive finite source cores is required")
        core = np.asarray(original, dtype=np.float64)
        if not np.isfinite(core).all() or np.any(core <= 0):
            raise ValueError("a nonempty vector of positive finite source cores is required")
        largest = float(core.max())
        tau = largest * self.broadening_ratio
        spacing = tau * self.spacing_over_tau
        cutoff = tau * self.correction_radius_over_tau
        if not all(math.isfinite(value) and value > 0 for value in (tau, spacing, cutoff)):
            raise ValueError("resolved auxiliary lengths are not finite and positive")
        return ResolvedGaussianMesh(tau, spacing, cutoff, self.order)
