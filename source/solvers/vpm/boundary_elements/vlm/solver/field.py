"""The numerical contract for the coupled VLM bound-surface field.

The VLM and VPM operators intentionally use different target filters.  A
bound filament is a singular geometric source and therefore needs a small
numerical safeguard at a point/trace target, while a VPM particle is a finite
volume target and carries its own core radius.  Keeping those scales explicit
prevents a particle radius from silently becoming the boundary condition, or
an almost-singular point evaluation from being used as a material transport
operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BoundSurfaceFieldContract:
    """Describe the representation and evaluation policy of the bound field.

    Parameters
    ----------
    numerical_epsilon : float
        Positive singularity safeguard used by the bound line-integral
        operator. It is not a physical core radius.
    bound_source_radius : float
        Radius used for point/probe and boundary-trace evaluation of the
        bound source. It may equal ``numerical_epsilon`` for the
        point-trace model, but is named separately so a resolved study can
        change it deliberately.
    particle_target_policy : str
        Human-readable description of the finite-target rule. The current
        coupled model uses ``max(particle_core_radius,
        numerical_epsilon)`` for particle-centre transport and stretching.
    free_wake_operator : str
        The configured VPM source/target backend. It is part of the complete
        transport contract even though this object owns only the bound field.
    bound_representation, bound_trace_operator, jacobian_operator : str
        Explicit names for the geometric source and its point/Jacobian
        evaluation.  They are persisted as part of the coupling contract so
        a restart cannot silently mix a different source representation.
    transport_target_operator : str
        Name of the finite-target transport rule used for the free wake.
    near_wake_policy : str
        Accepted/newborn wake rule used by the stage-responsive solve.
    """

    numerical_epsilon: float
    bound_source_radius: float
    particle_target_policy: str = "max(particle_core_radius, numerical_epsilon)"
    free_wake_operator: str = "configured VPM induction backend"
    bound_representation: str = "finite_segment_global_horseshoe"
    bound_trace_operator: str = "finite_segment_rosenhead_point_trace"
    jacobian_operator: str = "finite_segment_rosenhead_same_source_kernel"
    transport_target_operator: str = "symmetric_pair_radius"
    near_wake_policy: str = "partial_newborn_row_stage_responsive"

    def __post_init__(self) -> None:
        """Require finite positive numerical and bound-source radii."""
        for name in ("numerical_epsilon", "bound_source_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def particle_target_radius(self, particle_core_radius: float) -> float:
        """Return the target radius used by particle-centre transport."""
        return max(float(particle_core_radius), self.numerical_epsilon)

    def as_dict(self) -> dict[str, object]:
        """Return the complete explicit contract for manifests and restarts."""
        return {
            "numerical_epsilon": float(self.numerical_epsilon),
            "bound_source_radius": float(self.bound_source_radius),
            "particle_target_policy": self.particle_target_policy,
            "free_wake_operator": self.free_wake_operator,
            "bound_representation": self.bound_representation,
            "bound_trace_operator": self.bound_trace_operator,
            "jacobian_operator": self.jacobian_operator,
            "transport_target_operator": self.transport_target_operator,
            "near_wake_policy": self.near_wake_policy,
        }


__all__ = ["BoundSurfaceFieldContract"]
