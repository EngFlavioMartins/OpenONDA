"""Velocity-evaluation configuration for the VPM solver.
Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VelocityConfig:
    """
    Configuration for velocity field computation.

    Controls whether to use direct O(N²) summation or hierarchical O(N log N)
    treecode acceleration for computing particle-induced velocities.

    Examples:
          # Direct computation (default, exact but O(N²))
          velocity = VelocityConfig.direct()

          # Barnes-Hut treecode (approximate, O(N log N))
          velocity = VelocityConfig.treecode(theta=0.5)  # ~5% error, 100x+ speedup
          velocity = VelocityConfig.treecode(theta=0.3)  # ~2% error, moderate speedup

    Performance Guide:
          - N < 5000: Direct method is typically faster (GPU parallelism)
          - N > 5000: Treecode provides significant speedup
          - N > 50000: Treecode is essential for reasonable runtimes
    """

    method: Literal["DIRECT", "TREECODE"] = "DIRECT"
    """Velocity computation method: 'DIRECT' (O(N²)) or 'TREECODE' (O(N log N))."""

    theta: float = 0.3
    """Opening angle for Barnes-Hut treecode. Only used when method='TREECODE'.
      Smaller = more accurate but slower.
      - θ = 0.3: ~2% error, slower
      - θ = 0.5: ~5% error
      - θ = 0.7: ~15% error, faster"""

    multipole_order: Literal[1, 2, 3] = 1
    """Treecode far-field expansion order. 1 is the base monopole. 2 adds a
    circulation dipole moment per node, improving far-field gradient accuracy at
    extra per-node cost. 3 adds the quadrupole moment, allowing a larger theta
    at the same accuracy (usually the fastest accuracy/cost trade-off)."""

    sort_particle_targets: bool = False
    """Evaluate particle targets in Morton order for better GPU traversal
    coherence. Results are written back to original particle indices."""

    traversal_block_dim: int = 128
    """Taichi ``loop_config(block_dim=...)`` for tree traversal kernels.
    Set to 0 to leave Taichi's backend default untouched."""

    def __post_init__(self) -> None:
        method = self.method.upper()
        if method not in ("DIRECT", "TREECODE"):
            raise ValueError(f"velocity method must be DIRECT or TREECODE, got {self.method!r}")
        if not 0.0 < self.theta < 2.0:
            raise ValueError(f"velocity theta must be in (0, 2), got {self.theta!r}")
        if self.multipole_order not in (1, 2, 3):
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
        """Direct O(N²) pairwise velocity computation (exact).

        Returns
        -------
        VelocityConfig
            Configuration with ``method`` set to ``'DIRECT'``.

        Examples
        --------
        .. code-block:: python

            >>> velocity = VelocityConfig.direct()
            >>> velocity.method
            'DIRECT'
        """
        return VelocityConfig(method="DIRECT")

    @staticmethod
    def treecode(
        theta: float = 0.3,
        multipole_order: Literal[1, 2, 3] = 1,
        sort_particle_targets: bool = False,
        traversal_block_dim: int = 128,
    ) -> "VelocityConfig":
        """Barnes-Hut treecode O(N log N) velocity computation (approximate).

        Args:
              theta: Opening angle parameter (0.3-0.7). Smaller = more accurate.
              multipole_order: 1 monopole, 2 dipole, 3 quadrupole far-node expansion.
              sort_particle_targets: Evaluate particle targets in Morton order.
              traversal_block_dim: Taichi traversal kernel block size; 0 = backend default.

        Returns
        -------
        VelocityConfig
            Configuration with ``method`` set to ``'TREECODE'`` and the given
            opening angle.

        Examples
        --------
        .. code-block:: python

            >>> velocity = VelocityConfig.treecode()
            >>> velocity.theta
            0.3

            >>> velocity = VelocityConfig.treecode(theta=0.5)
            >>> velocity.theta
            0.5
        """
        return VelocityConfig(
            method="TREECODE",
            theta=theta,
            multipole_order=multipole_order,
            sort_particle_targets=sort_particle_targets,
            traversal_block_dim=traversal_block_dim,
        )


# =========================================================
# SOLVER CONFIGURATION DATACLASS
# =========================================================
