"""Numerical definition of the coupled VLM bound-surface field.

The VLM and VPM operators use different target filters.  A
bound filament is a singular geometric source and therefore needs a small
numerical regularization length at a point/trace target, while a VPM particle is a finite
volume target and carries its own core radius.  Keeping those scales explicit
prevents a particle radius from silently becoming the boundary condition, or
an almost-singular point evaluation from being used as a material transport
operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BoundSurfaceFieldSettings:
    """Describe the representation and evaluation settings of the bound field.

    Parameters
    ----------
    filament_core_radius : float
        Positive regularization length of the singular bound line-integral
        operator, used at collocation points, probes and bound midpoints. It is
        not a physical core radius.
    particle_core_rule : str
        Human-readable description of the finite-target rule. The current
        coupled model uses ``max(particle_core_radius,
        filament_core_radius)`` for particle-centre transport and stretching.
    free_wake_operator : str
        The configured VPM source/target backend. It is part of the complete
        transport definition even though this object owns only the bound field.
    bound_representation, bound_trace_operator, jacobian_operator : str
        Explicit names for the geometric source and its point/Jacobian
        evaluation.  They are persisted as part of the coupling record so
        a restart cannot silently mix a different source representation.
    transport_target_operator : str
        Name of the finite-target transport rule used for the free wake.
    near_wake_treatment : str
        Accepted/newborn wake rule used by the stage-responsive solve.
    """

    filament_core_radius: float
    particle_core_rule: str = "max(particle_core_radius, filament_core_radius)"
    free_wake_operator: str = "configured VPM induction backend"
    bound_representation: str = "finite_segment_global_horseshoe"
    bound_trace_operator: str = "finite_segment_rosenhead_point_trace"
    jacobian_operator: str = "finite_segment_rosenhead_same_source_kernel"
    transport_target_operator: str = "symmetric_pair_radius"
    near_wake_treatment: str = "partial_newborn_row_stage_responsive"

    def __post_init__(self) -> None:
        """Require a finite positive filament core radius."""
        value = float(self.filament_core_radius)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("filament_core_radius must be finite and positive")

    def as_dict(self) -> dict[str, object]:
        """Return the complete explicit field settings for metadata and restarts."""
        return {
            "filament_core_radius": float(self.filament_core_radius),
            "particle_core_rule": self.particle_core_rule,
            "free_wake_operator": self.free_wake_operator,
            "bound_representation": self.bound_representation,
            "bound_trace_operator": self.bound_trace_operator,
            "jacobian_operator": self.jacobian_operator,
            "transport_target_operator": self.transport_target_operator,
            "near_wake_treatment": self.near_wake_treatment,
        }


__all__ = ["BoundSurfaceFieldSettings"]
