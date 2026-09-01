"""Private treecode workspace façade for the induction package.

The LBVH implementation and its Morton/traversal kernels live beside the
treecode evaluator; this façade provides a small ownership point for callers
that need an explicit hierarchy build.
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
