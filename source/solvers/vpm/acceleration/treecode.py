"""
Barnes-Hut Treecode for VPM Velocity Computation.
=================================================
O(N log N) approximation for particle-particle interactions using octree
with multipole acceptance criterion (MAC).

This module provides a drop-in replacement for the O(N²) direct summation
velocity kernel, achieving significant speedup for large particle counts
while maintaining controllable accuracy via the opening angle θ.

Algorithm:
1. Build octree from particle position
2. Compute multipole moments (total vortex strength, centre of vorticity) bottom-up
3. For each target, traverse tree:
   - If node is "far" (size/distance < θ): use multipole approximation
   - Otherwise: recurse into children or direct-sum leaf particles

Accuracy: Relative error typically 1e-2 to 1e-4 depending on θ
- θ = 0.3: High accuracy (~1e-4), slower
- θ = 0.5: Balanced (~1e-3), good default
- θ = 0.7: Fast (~1e-2), acceptable for visualization
- θ = 1.0: Very fast, lower accuracy

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass
class OctreeNode:
    """
    Single octree node for Barnes-Hut algorithm.

    Attributes:
        centre: Node centre coordinates [3]
        half_size: Half the node width (node spans centre ± half_size)
        particles: List of particle indices contained in this node (leaf only)
        children: List of child nodes (interior only, up to 8)
        net_vortex_strength: Sum of particle vortex strengths for the multipole
        vortex_centroid: Vortex strength-weighted centroid Σ(x_i |Γ_i|) / Σ|Γ_i|
        avg_radius: Average particle radius in this node (for kernel smoothing)
    """

    centre: np.ndarray
    half_size: float
    particles: list[int] = field(default_factory=list)
    children: list["OctreeNode"] = field(default_factory=list)

    # Multipole moments
    net_vortex_strength: np.ndarray = None
    vortex_centroid: np.ndarray = None
    average_core_radius: float = 0.0

    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def is_empty(self) -> bool:
        """True if node contains no particles."""
        return len(self.particles) == 0 and self.is_leaf


class BarnesHutTreecode:
    """
    Barnes-Hut treecode for O(N log N) VPM velocity computation.

    This class builds an octree from vortex particles and uses the
    Multipole Acceptance Criterion (MAC) to approximate far-field
    interactions, reducing complexity from O(N²) to O(N log N).

    Example:
        >>> tree = BarnesHutTreecode(theta=0.5)
        >>> tree.build(position, vortex_strength, core_radius)
        >>> velocity = tree.compute_velocities(position, core_radius, freestream_velocity)

    Parameters:
        theta: Opening angle for MAC (default 0.5). Lower = more accurate.
        max_leaf_size: Maximum particles per leaf before subdivision (default 32).
        kernel_type: Regularization kernel ('gaussian' or 'winckelmans').
    """

    def __init__(
        self, theta: float = 0.5, max_leaf_size: int = 32, kernel_type: str = "winckelmans"
    ):
        self.theta = theta
        self.max_leaf_size = max_leaf_size
        self.kernel_type = kernel_type.lower()

        # Particle data references (set during build)
        self.position: np.ndarray | None = None
        self.vortex_strength: np.ndarray | None = None
        self.core_radius: np.ndarray | None = None

        # Tree root
        self.root: OctreeNode | None = None

        # Statistics
        self.n_particles_total = 0
        self.n_nodes = 0
        self.n_leaves = 0
        self.max_depth = 0
        self.build_time = 0.0

    def build(
        self, position: np.ndarray, vortex_strength: np.ndarray, core_radius: np.ndarray
    ) -> None:
        """
        Build octree from particle data.

        Args:
            position: Particle position [N, 3]
            vortex_strength: Particle vortex strength [N, 3] [m³/s]
            core_radius: Particle core radius [N] [m]
        """
        t_start = time.perf_counter()

        self.position = np.ascontiguousarray(position, dtype=np.float64)
        self.vortex_strength = np.ascontiguousarray(vortex_strength, dtype=np.float64)
        self.core_radius = np.ascontiguousarray(core_radius, dtype=np.float64)
        self.n_particles_total = len(position)

        # Reset statistics
        self.n_nodes = 0
        self.n_leaves = 0
        self.max_depth = 0

        # Compute bounding box with padding
        pmin = position.min(axis=0)
        pmax = position.max(axis=0)
        centre = 0.5 * (pmin + pmax)
        half_size = 0.5 * np.max(pmax - pmin) * 1.01  # 1% padding

        # Ensure minimum size to avoid numerical issues
        half_size = max(half_size, 1e-6)

        # Build tree recursively
        all_indices = list(range(self.n_particles_total))
        self.root = self._build_node(centre, half_size, all_indices, depth=0)

        # Compute multipole moments bottom-up
        self._compute_multipoles(self.root)

        self.build_time = time.perf_counter() - t_start

    def _build_node(
        self, centre: np.ndarray, half_size: float, indices: list[int], depth: int
    ) -> OctreeNode:
        """Recursively build octree node."""
        self.n_nodes += 1
        self.max_depth = max(self.max_depth, depth)

        node = OctreeNode(
            centre=centre.copy(), half_size=half_size, particles=indices.copy(), children=[]
        )

        # Leaf node if few enough particles or max depth reached
        if len(indices) <= self.max_leaf_size or depth > 20:
            self.n_leaves += 1
            return node

        # Subdivide into 8 children (octants)
        node.particles = []  # Interior nodes don't store particles directly
        child_half = half_size / 2

        # Compute which octant each particle belongs to
        particle_octants = self._compute_octants(centre, indices)

        for octant in range(8):
            # Get particles in this octant
            child_indices = [indices[i] for i, o in enumerate(particle_octants) if o == octant]

            if not child_indices:
                continue

            # Compute child centre
            offset = np.array(
                [
                    child_half if (octant & 1) else -child_half,
                    child_half if (octant & 2) else -child_half,
                    child_half if (octant & 4) else -child_half,
                ]
            )
            child_centre = centre + offset

            # Recursively build child
            child = self._build_node(child_centre, child_half, child_indices, depth + 1)
            node.children.append(child)

        return node

    def _compute_octants(self, centre: np.ndarray, indices: list[int]) -> list[int]:
        """Compute octant index for each particle."""
        octants = []
        for i in indices:
            pos = self.position[i]
            octant = 0
            if pos[0] >= centre[0]:
                octant |= 1
            if pos[1] >= centre[1]:
                octant |= 2
            if pos[2] >= centre[2]:
                octant |= 4
            octants.append(octant)
        return octants

    def _compute_leaf_multipoles(self, node: OctreeNode) -> None:
        """Compute multipole moments for a leaf node."""
        if not node.particles:
            node.net_vortex_strength = np.zeros(3)
            node.vortex_centroid = node.centre.copy()
            node.average_core_radius = 0.0
            return

        vortex_strengths_local = self.vortex_strength[node.particles]
        poss = self.position[node.particles]
        core_radii_local = self.core_radius[node.particles]

        node.net_vortex_strength = vortex_strengths_local.sum(axis=0)
        node.average_core_radius = core_radii_local.mean()

        mags = np.linalg.norm(vortex_strengths_local, axis=1, keepdims=True)
        total_mag = mags.sum()

        if total_mag > 1e-15:
            node.vortex_centroid = (poss * mags).sum(axis=0) / total_mag
        else:
            node.vortex_centroid = poss.mean(axis=0)

    def _compute_interior_multipoles(self, node: OctreeNode) -> None:
        """Aggregate multipole moments from children for an interior node."""
        net_vortex_strength = np.zeros(3)
        weighted_pos = np.zeros(3)
        total_weight = 0.0
        total_radius = 0.0
        n_particles_total = 0

        for child in node.children:
            net_vortex_strength += child.net_vortex_strength
            mag = np.linalg.norm(child.net_vortex_strength)
            weighted_pos += child.vortex_centroid * mag
            total_weight += mag

            n_child = len(child.particles) if child.is_leaf else 1
            total_radius += child.average_core_radius * n_child
            n_particles_total += n_child

        node.net_vortex_strength = net_vortex_strength
        node.average_core_radius = total_radius / max(n_particles_total, 1)

        if total_weight > 1e-15:
            node.vortex_centroid = weighted_pos / total_weight
        else:
            node.vortex_centroid = node.centre.copy()

    def _compute_multipoles(self, node: OctreeNode) -> None:
        """Compute multipole moments bottom-up."""
        if node.is_leaf:
            self._compute_leaf_multipoles(node)
        else:
            for child in node.children:
                self._compute_multipoles(child)
            self._compute_interior_multipoles(node)

    def compute_velocities(
        self,
        target_position: np.ndarray,
        target_core_radius: np.ndarray,
        freestream_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute velocity at target position using treecode.

        Args:
            target_position: Target position [M, 3]
            target_core_radius: Target core_radius for kernel smoothing [M]
            freestream_velocity: Freestream velocity [3] (optional)

        Returns:
            Velocities at targets [M, 3]
        """
        M = len(target_position)
        velocity = np.zeros((M, 3), dtype=np.float64)

        for i in range(M):
            velocity[i] = self._traverse(self.root, target_position[i], target_core_radius[i])

        if freestream_velocity is not None:
            velocity += freestream_velocity

        return velocity

    def _traverse(self, node: OctreeNode, target: np.ndarray, target_radius: float) -> np.ndarray:
        """
        Recursive tree traversal with MAC for single target.

        Uses the standard Barnes-Hut criterion: node_size / distance < θ
        """
        vel = np.zeros(3)

        if node is None:
            return vel

        # Vector from target to node's centre of vorticity
        r_vec = target - node.vortex_centroid
        r_mag = np.linalg.norm(r_vec)

        # Multipole Acceptance Criterion (MAC)
        # If node is "far enough", use multipole approximation
        if r_mag > 1e-10 and (2 * node.half_size / r_mag) < self.theta:
            # Use monopole approximation: treat entire node as single vortex
            sigma = 0.5 * (target_radius + node.average_core_radius)
            r_sigma = r_mag / sigma
            q_val = self._q_kernel(r_sigma)

            # Biot-Savart: u = -q(r/σ) * (r × Γ) / r³
            vel = -q_val * np.cross(r_vec, node.net_vortex_strength) / (r_mag**3)

        elif node.is_leaf:
            # Leaf node: direct summation over particles
            for j in node.particles:
                r_vec_j = target - self.position[j]
                r_mag_j = np.linalg.norm(r_vec_j)

                if r_mag_j > 1e-12:
                    sigma = 0.5 * (target_radius + self.core_radius[j])
                    r_sigma = r_mag_j / sigma
                    q_val = self._q_kernel(r_sigma)
                    vel -= q_val * np.cross(r_vec_j, self.vortex_strength[j]) / (r_mag_j**3)
        else:
            # Interior node: recurse into children
            for child in node.children:
                vel += self._traverse(child, target, target_radius)

        return vel

    def _q_kernel(self, r_sigma: float) -> float:
        """
        Regularization kernel q(r/σ) for velocity.

        The kernel smooths the 1/r² singularity of Biot-Savart.

        For Winckelmans kernel:
            q(ρ) = ρ³(ρ² + 2.5) / (ρ² + 1)^2.5 × (1/4π)

        where ρ = r/σ is the normalized distance.
        """
        ONE_OVER_FOUR_PI = 0.07957747154594767

        if self.kernel_type == "gaussian":
            # Gaussian kernel (Rosenhead-Moore)
            r2 = r_sigma * r_sigma
            return (1.0 - np.exp(-r2)) * ONE_OVER_FOUR_PI
        else:
            # Winckelmans high-order algebraic kernel (default)
            # q(ρ) = ρ³(ρ² + 2.5) / (ρ² + 1)^2.5 × (1/4π)
            r2 = r_sigma * r_sigma
            r3 = r_sigma * r_sigma * r_sigma
            return r3 * (r2 + 2.5) / (r2 + 1.0) ** 2.5 * ONE_OVER_FOUR_PI

    def info(self) -> str:
        """Return summary string of tree statistics."""
        return (
            f"BarnesHutTreecode:\n"
            f"  Particles: {self.n_particles_total}\n"
            f"  Nodes: {self.n_nodes} ({self.n_leaves} leaves)\n"
            f"  Max depth: {self.max_depth}\n"
            f"  Opening angle θ: {self.theta}\n"
            f"  Build time: {self.build_time * 1000:.2f} ms"
        )


# =========================================================
# Convenience functions for quick usage
# =========================================================


def compute_velocities_treecode(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    theta: float = 0.5,
    freestream_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute velocity using Barnes-Hut treecode (convenience function).

    Args:
        position: Particle position [N, 3]
        vortex_strength: Particle vortex strength [N, 3] [m³/s]
        core_radius: Particle core radius [N] [m]
        theta: Opening angle (default 0.5)
        freestream_velocity: Freestream velocity [3] (optional)

    Returns:
        Velocities [N, 3]
    """
    tree = BarnesHutTreecode(theta=theta)
    tree.build(position, vortex_strength, core_radius)
    return tree.compute_velocities(position, core_radius, freestream_velocity)


def benchmark_treecode(
    position: np.ndarray,
    vortex_strength: np.ndarray,
    core_radius: np.ndarray,
    direct_velocity: np.ndarray | None = None,
    theta_values: list[float] = None,
) -> dict:
    """
    Benchmark treecode accuracy and performance vs direct summation.

    Args:
        position: Particle position [N, 3]
        vortex_strength: Particle vortex strength [N, 3] [m³/s]
        core_radius: Particle core radius [N] [m]
        direct_velocity: Reference velocity from direct sum (optional)
        theta_values: List of θ values to test

    Returns:
        Dictionary with timing and error statistics
    """
    if theta_values is None:
        theta_values = [0.3, 0.5, 0.7, 1.0]
    results = {"N": len(position), "theta": [], "time": [], "error": []}

    v_ref_norm = 1.0
    if direct_velocity is not None:
        v_ref_norm = np.linalg.norm(direct_velocity)

    for theta in theta_values:
        t_start = time.perf_counter()
        tree = BarnesHutTreecode(theta=theta)
        tree.build(position, vortex_strength, core_radius)
        v_tree = tree.compute_velocities(position, core_radius)
        t_elapsed = time.perf_counter() - t_start

        if direct_velocity is not None:
            error = np.linalg.norm(v_tree - direct_velocity) / v_ref_norm
        else:
            error = float("nan")

        results["theta"].append(theta)
        results["time"].append(t_elapsed)
        results["error"].append(error)

        print(f"θ={theta:.2f}: time={t_elapsed:.3f}s, error={error:.2e}")

    return results
