"""Mesh-quality limits checked when an FVM solver is constructed."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import finite_real


@dataclass(frozen=True, slots=True)
class MeshQualityConfig:
    """Optional hard mesh-quality limits.

    ``None`` disables an individual check. The limits are acceptance
    thresholds, not mesh-generation targets; a passing mesh still requires a
    resolution study.

    Parameters
    ----------
    max_non_orthogonality_deg : float or None
        Maximum face non-orthogonality in degrees.
    max_skewness : float or None
        Maximum dimensionless skewness.
    max_aspect_ratio : float or None
        Maximum dimensionless cell aspect ratio.
    max_lsq_condition : float or None
        Maximum condition number of the least-squares gradient stencils.
    """

    max_non_orthogonality_deg: float | None = None
    max_skewness: float | None = None
    max_aspect_ratio: float | None = None
    max_lsq_condition: float | None = None

    def __post_init__(self) -> None:
        for name, strict in (
            ("max_non_orthogonality_deg", False),
            ("max_skewness", True),
            ("max_aspect_ratio", True),
            ("max_lsq_condition", True),
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    finite_real(
                        f"MeshQualityConfig.{name}", value, minimum=0.0, strict_minimum=strict
                    ),
                )
