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


def _ratio(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return value


@dataclass(frozen=True)
class ResolvedGaussianMesh:
    """Auxiliary lengths derived from an actual core snapshot."""

    tau: float
    spacing: float
    correction_cutoff: float
    order: int
    maximum_source_core: float


@dataclass(frozen=True)
class GaussianMeshParameters:
    """Case-independent, explicit discretization controls.

    Four auxiliary cells per broadened core is the conservative resolution
    used in finite-field refinement qualification. Increasing spacing/tau
    changes accuracy, not just performance. Correction radius alone is not
    validation of its omitted tail. No automatic tolerance changes occur here.
    """

    broadening_ratio: float = 3.0
    spacing_over_tau: float = 0.25
    correction_radius_over_tau: float = 5.0
    order: int = 10

    def __post_init__(self):
        for name in ("broadening_ratio", "spacing_over_tau", "correction_radius_over_tau"):
            object.__setattr__(self, name, _ratio(getattr(self, name), name))
        if self.broadening_ratio < 1:
            raise ValueError("broadening_ratio must cover every actual source core")
        if isinstance(self.order, bool) or not isinstance(self.order, Integral):
            raise ValueError("order must be an even integer in [4, 10]")
        if self.order not in (4, 6, 8, 10):
            raise ValueError("order must be an even integer in [4, 10]")
        object.__setattr__(self, "order", int(self.order))

    def resolve(self, source_core):
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
        return ResolvedGaussianMesh(tau, spacing, cutoff, self.order, largest)
