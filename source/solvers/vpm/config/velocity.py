"""Induced-velocity evaluation configuration for the VPM solver."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VelocityConfig:
    """Configure particle-induced velocity evaluation.

    ``DIRECT`` evaluates all particle pairs. ``TREECODE`` uses the Barnes-Hut
    hierarchy and is controlled by its opening angle and multipole order.
    """

    method: Literal["DIRECT", "TREECODE"] = "DIRECT"
    """Induced-velocity evaluation method."""

    theta: float = 0.3
    """Barnes-Hut opening angle; smaller values are more accurate and costly."""

    multipole_order: Literal[1, 2, 3] = 1
    """Far-field expansion order for treecode nodes."""

    sort_particle_targets: bool = False
    """Evaluate particle targets in Morton order to improve traversal coherence."""

    traversal_block_dim: int = 128
    """Taichi tree-traversal block size; ``0`` leaves the backend default."""

    def __post_init__(self) -> None:
        method = self.method.upper()
        if method not in {"DIRECT", "TREECODE"}:
            raise ValueError(f"velocity method must be DIRECT or TREECODE, got {self.method!r}")
        if not 0.0 < self.theta < 2.0:
            raise ValueError(f"velocity theta must be in (0, 2), got {self.theta!r}")
        if self.multipole_order not in {1, 2, 3}:
            raise ValueError(
                f"velocity multipole_order must be 1, 2 or 3, got {self.multipole_order!r}"
            )
        if self.traversal_block_dim < 0:
            raise ValueError(
                f"velocity traversal_block_dim must be >= 0, got {self.traversal_block_dim!r}"
            )
        if method != self.method:
            object.__setattr__(self, "method", method)

    @staticmethod
    def direct() -> "VelocityConfig":
        """Return exact direct pairwise velocity evaluation."""
        return VelocityConfig(method="DIRECT")

    @staticmethod
    def treecode(
        theta: float = 0.3,
        multipole_order: Literal[1, 2, 3] = 1,
        sort_particle_targets: bool = False,
        traversal_block_dim: int = 128,
    ) -> "VelocityConfig":
        """Return Barnes-Hut treecode velocity evaluation."""
        return VelocityConfig(
            method="TREECODE",
            theta=theta,
            multipole_order=multipole_order,
            sort_particle_targets=sort_particle_targets,
            traversal_block_dim=traversal_block_dim,
        )
