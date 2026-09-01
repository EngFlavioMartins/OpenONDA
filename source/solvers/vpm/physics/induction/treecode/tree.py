"""Private treecode workspace façade.

The mature LBVH implementation remains the traversal engine.  This façade
gives the induction package a stable ownership point while the low-level
Morton and traversal kernels remain in the acceleration backend.
"""

from __future__ import annotations


class TreecodeWorkspace:
    """Describe a cached LBVH workspace owned by a treecode evaluator."""

    def __init__(self, physics, count: int, theta: float):
        self.physics = physics
        self.count = int(count)
        self.theta = float(theta)

    def build(self, position, vortex_strength, core_radius) -> None:
        """Build the low-level hierarchy from one complete stage state."""
        self.physics._get_or_create_treecode(self.count, self.theta).build(
            position, vortex_strength, core_radius, self.count
        )


__all__ = ["TreecodeWorkspace"]
