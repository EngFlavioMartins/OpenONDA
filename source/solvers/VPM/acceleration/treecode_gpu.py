"""
Taichi GPU-Accelerated Barnes-Hut Treecode for VPM.
====================================================
Parallel octree construction and traversal using Taichi for GPU acceleration.

This module provides a GPU-optimized implementation of the Barnes-Hut
treecode algorithm, suitable for real-time VPM simulations with 10⁴-10⁶
particles.

The implementation uses a linear octree representation with Morton (Z-order)
encoding for efficient parallel construction and cache-friendly traversal.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import time

import numpy as np
import taichi as ti


@ti.data_oriented
class TaichiTreecode:
    """
    GPU-accelerated Barnes-Hut treecode using Taichi.

    This implementation uses a stack-based iterative tree traversal
    instead of recursion, which is more GPU-friendly.

    Parameters:
        max_particles: Maximum number of particles
        max_nodes: Maximum number of tree nodes (typically 2*max_particles)
        theta: Opening angle for MAC criterion
        max_leaf_size: Maximum particles per leaf node
    """

    def __init__(
        self,
        max_particles: int = 100000,
        max_nodes: int = 200000,
        theta: float = 0.5,
        max_leaf_size: int = 32,
        kernel_type: str = "WINCKELMANS",
    ):
        self.max_particles = max_particles
        self.max_nodes = max_nodes
        self.theta = theta
        self.max_leaf_size = max_leaf_size
        self.theta_sq = theta * theta  # Pre-compute for MAC
        self.kernel_type = kernel_type.upper()

        # ─────────────────────────────────────────────────────────────
        # PARTICLE DATA (copied from input)
        # ─────────────────────────────────────────────────────────────
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.circulations = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.radii = ti.field(dtype=ti.f32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # TREE STRUCTURE (Structure of Arrays for GPU efficiency)
        # ─────────────────────────────────────────────────────────────
        # Node properties
        self.node_center = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.node_half_size = ti.field(dtype=ti.f32, shape=max_nodes)

        # Multipole moments
        self.node_total_circ = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.node_com = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)  # Center of vorticity
        self.node_avg_radius = ti.field(dtype=ti.f32, shape=max_nodes)

        # Tree structure (8 children per node, -1 = no child)
        self.node_children = ti.field(dtype=ti.i32, shape=(max_nodes, 8))
        self.node_is_leaf = ti.field(dtype=ti.i32, shape=max_nodes)
        self.node_particle_start = ti.field(dtype=ti.i32, shape=max_nodes)
        self.node_particle_count = ti.field(dtype=ti.i32, shape=max_nodes)

        # Particle-to-leaf mapping (for direct sum in leaves)
        self.leaf_particles = ti.field(dtype=ti.i32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # OUTPUT VELOCITIES
        # ─────────────────────────────────────────────────────────────
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # OUTPUT VELOCITY GRADIENTS AND STRAIN RATES
        # ─────────────────────────────────────────────────────────────
        self.velocity_gradients = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.strain_rates = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # TARGET POINT FIELDS (for computing at arbitrary locations)
        # ─────────────────────────────────────────────────────────────
        self.max_targets = max_particles  # Can evaluate at up to max_particles targets
        self.target_positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.target_velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.target_velocity_gradients = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.n_targets = ti.field(dtype=ti.i32, shape=())
        self.kernel_type_id = ti.field(dtype=ti.i32, shape=())

        # ─────────────────────────────────────────────────────────────
        # TREE TRAVERSAL STACK (for iterative GPU traversal)
        # Per-thread fields allow @ti.func helpers to push/pop without
        # passing local arrays as template arguments.
        # ─────────────────────────────────────────────────────────────
        self.max_stack_depth = 48  # 7*D+1 ≤ 48 → safe for tree depth ≤ 6
        self.traversal_stack = ti.field(dtype=ti.i32, shape=(max_particles, 48))
        self.target_traversal_stack = ti.field(dtype=ti.i32, shape=(max_particles, 48))

        # ─────────────────────────────────────────────────────────────
        # COUNTERS
        # ─────────────────────────────────────────────────────────────
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_nodes = ti.field(dtype=ti.i32, shape=())

        # Background velocity
        self.u_inf = ti.Vector.field(3, dtype=ti.f32, shape=())

        # Statistics
        self.build_time = 0.0
        self.eval_time = 0.0
        self.grad_time = 0.0  # Time for velocity gradient computation

        # Cached temporary field for children transfer (avoids memory leak)
        self._temp_children = ti.field(dtype=ti.i32, shape=(max_nodes, 8))

        # Constants for gradient kernel
        self.MIN_R_SIGMA_GRADIENT = 0.5  # Skip overlapping cores
        self.DEFAULT_CUTOFF_RADIUS_FACTOR = 15.0  # Far-field cutoff

        self.set_kernel_type(self.kernel_type)

    def set_kernel_type(self, kernel_type: str) -> None:
        """Update the regularization kernel used by the treecode."""
        normalized = kernel_type.upper()
        if normalized == "GAUSSIAN":
            kernel_id = 0
        elif normalized == "WINCKELMANS":
            kernel_id = 1
        else:
            raise ValueError(f"Unsupported treecode kernel_type: {kernel_type}")

        self.kernel_type = normalized
        self.kernel_type_id[None] = kernel_id

    def build(self, positions: np.ndarray, circulations: np.ndarray, radii: np.ndarray) -> None:
        """
        Build octree from particle data.

        Note: Tree construction is done on CPU for simplicity.
        GPU construction is complex and typically not the bottleneck.

        Args:
            positions: Particle positions [N, 3] (N <= max_particles)
            circulations: Particle circulations [N, 3]
            radii: Particle radii [N]
        """
        t_start = time.perf_counter()

        N = len(positions)
        if self.max_particles < N:
            raise ValueError(
                f"Too many particles ({N}) for treecode capacity ({self.max_particles})"
            )

        self.n_particles[None] = N

        # Pad arrays to max_particles size for Taichi field compatibility
        # Taichi from_numpy requires exact shape match
        def pad_1d(arr):
            """Pad 1D array to max_particles."""
            padded = np.zeros(self.max_particles, dtype=np.float32)
            padded[:N] = arr.astype(np.float32)
            return padded

        def pad_2d(arr):
            """Pad 2D array to (max_particles, 3)."""
            padded = np.zeros((self.max_particles, 3), dtype=np.float32)
            padded[:N] = arr.astype(np.float32)
            return padded

        # Copy particle data to Taichi fields (padded)
        self.positions.from_numpy(pad_2d(positions))
        self.circulations.from_numpy(pad_2d(circulations))
        self.radii.from_numpy(pad_1d(radii))

        # Build tree on CPU (simpler, construction is O(N log N) anyway)
        self._build_tree_cpu(positions, circulations, radii)

        self.build_time = time.perf_counter() - t_start

    @staticmethod
    def _pad_to_max(arr: np.ndarray, max_size: int, fill: float = 0.0) -> np.ndarray:
        """Pad array to max_size along the first axis."""
        if len(arr) >= max_size:
            return arr[:max_size]
        pad_width = [(0, max_size - len(arr))] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad_width, constant_values=fill)

    @staticmethod
    def _classify_octant(pos: np.ndarray, center: np.ndarray) -> int:
        """Return the 0-7 octant index for *pos* relative to *center*."""
        octant = 0
        if pos[0] >= center[0]:
            octant |= 1
        if pos[1] >= center[1]:
            octant |= 2
        if pos[2] >= center[2]:
            octant |= 4
        return octant

    @staticmethod
    def _octant_offset(octant: int, child_half: float) -> np.ndarray:
        """Offset vector from parent center to the *octant* child center."""
        return np.array(
            [
                child_half if (octant & 1) else -child_half,
                child_half if (octant & 2) else -child_half,
                child_half if (octant & 4) else -child_half,
            ]
        )

    def _bfs_build_tree(self, positions: np.ndarray, N: int, center: np.ndarray, half_size: float):
        """BFS octree construction; returns node arrays."""
        node_centers: list = [center]
        node_half_sizes: list = [half_size]
        node_particles: list = [list(range(N))]
        node_children: list = [[-1] * 8]
        node_is_leaf: list = [0]
        queue: list = [0]
        node_idx = 1

        while queue:
            curr = queue.pop(0)
            particles = node_particles[curr]
            curr_center = node_centers[curr]
            curr_half = node_half_sizes[curr]

            if len(particles) <= self.max_leaf_size:
                node_is_leaf[curr] = 1
                continue

            child_half = curr_half / 2
            child_particles: list = [[] for _ in range(8)]
            for p in particles:
                child_particles[self._classify_octant(positions[p], curr_center)].append(p)

            for octant in range(8):
                if not child_particles[octant]:
                    continue
                child_idx = node_idx
                node_idx += 1
                node_centers.append(curr_center + self._octant_offset(octant, child_half))
                node_half_sizes.append(child_half)
                node_particles.append(child_particles[octant])
                node_children.append([-1] * 8)
                node_is_leaf.append(0)
                node_children[curr][octant] = child_idx
                queue.append(child_idx)

            node_particles[curr] = []

        return node_centers, node_half_sizes, node_particles, node_children, node_is_leaf

    @staticmethod
    def _compute_leaf_multipole(
        particles: list,
        positions: np.ndarray,
        circulations: np.ndarray,
        radii: np.ndarray,
        center: np.ndarray,
    ) -> tuple:
        """Compute multipole moment for a leaf node."""
        if not particles:
            return np.zeros(3, dtype=np.float32), center, 0.0
        circs = circulations[particles]
        poss = positions[particles]
        mags = np.linalg.norm(circs, axis=1, keepdims=True)
        total_mag = float(mags.sum())
        com = (poss * mags).sum(axis=0) / total_mag if total_mag > 1e-15 else poss.mean(axis=0)
        return circs.sum(axis=0), com, float(radii[particles].mean())

    @staticmethod
    def _compute_interior_multipole(
        i: int,
        node_children: list,
        node_total_circ: np.ndarray,
        node_com: np.ndarray,
        node_avg_radius: np.ndarray,
        node_centers: list,
    ) -> None:
        """Aggregate child multipoles into interior node i (in-place)."""
        total_circ = np.zeros(3)
        weighted_pos = np.zeros(3)
        total_weight = 0.0
        total_rad = 0.0
        n_child = 0
        for octant in range(8):
            child_idx = node_children[i][octant]
            if child_idx >= 0:
                total_circ += node_total_circ[child_idx]
                mag = float(np.linalg.norm(node_total_circ[child_idx]))
                weighted_pos += node_com[child_idx] * mag
                total_weight += mag
                total_rad += node_avg_radius[child_idx]
                n_child += 1
        node_total_circ[i] = total_circ
        node_avg_radius[i] = total_rad / max(n_child, 1)
        if total_weight > 1e-15:
            node_com[i] = weighted_pos / total_weight
        else:
            node_com[i] = node_centers[i]

    def _compute_node_multipoles(
        self,
        n_nodes: int,
        node_is_leaf: list,
        node_particles: list,
        node_children: list,
        node_centers: list,
        positions: np.ndarray,
        circulations: np.ndarray,
        radii: np.ndarray,
    ) -> tuple:
        """Bottom-up multipole computation for all nodes."""
        node_total_circ = np.zeros((n_nodes, 3), dtype=np.float32)
        node_com = np.zeros((n_nodes, 3), dtype=np.float32)
        node_avg_radius = np.zeros(n_nodes, dtype=np.float32)
        for i in range(n_nodes - 1, -1, -1):
            if node_is_leaf[i]:
                node_total_circ[i], node_com[i], node_avg_radius[i] = self._compute_leaf_multipole(
                    node_particles[i], positions, circulations, radii, node_centers[i]
                )
            else:
                self._compute_interior_multipole(
                    i, node_children, node_total_circ, node_com, node_avg_radius, node_centers
                )
        return node_total_circ, node_com, node_avg_radius

    def _transfer_to_gpu(
        self,
        n_nodes: int,
        node_centers: list,
        node_half_sizes: list,
        node_total_circ: np.ndarray,
        node_com: np.ndarray,
        node_avg_radius: np.ndarray,
        node_is_leaf: list,
        node_children: list,
    ) -> None:
        """Upload CPU node arrays to Taichi fields (padded to max_nodes)."""
        mn = self.max_nodes
        node_centers_arr = np.array(node_centers, dtype=np.float32)
        node_half_sizes_arr = np.array(node_half_sizes, dtype=np.float32)
        node_is_leaf_arr = np.array(node_is_leaf, dtype=np.int32)
        self.node_center.from_numpy(self._pad_to_max(node_centers_arr, mn))
        self.node_half_size.from_numpy(self._pad_to_max(node_half_sizes_arr, mn))
        self.node_total_circ.from_numpy(self._pad_to_max(node_total_circ, mn))
        self.node_com.from_numpy(self._pad_to_max(node_com, mn))
        self.node_avg_radius.from_numpy(self._pad_to_max(node_avg_radius, mn))
        self.node_is_leaf.from_numpy(self._pad_to_max(node_is_leaf_arr, mn))
        children_padded = np.full((mn, 8), -1, dtype=np.int32)
        children_array = np.array(node_children, dtype=np.int32)
        children_padded[:n_nodes] = children_array[:n_nodes]
        self._set_children_from_numpy(children_padded)

    def _pack_and_transfer_leaf_particles(
        self, n_nodes: int, node_is_leaf: list, node_particles: list
    ) -> None:
        """Pack leaf particle lists into a contiguous array and upload to GPU."""
        leaf_particles_list: list = []
        particle_starts = np.zeros(n_nodes, dtype=np.int32)
        particle_counts = np.zeros(n_nodes, dtype=np.int32)
        for i in range(n_nodes):
            if node_is_leaf[i] and node_particles[i]:
                particle_starts[i] = len(leaf_particles_list)
                particle_counts[i] = len(node_particles[i])
                leaf_particles_list.extend(node_particles[i])
        if leaf_particles_list:
            leaf_arr = np.array(leaf_particles_list, dtype=np.int32)
            self.leaf_particles.from_numpy(
                self._pad_to_max(leaf_arr, self.max_particles).astype(np.int32)
            )
        mn = self.max_nodes
        self.node_particle_start.from_numpy(self._pad_to_max(particle_starts, mn).astype(np.int32))
        self.node_particle_count.from_numpy(self._pad_to_max(particle_counts, mn).astype(np.int32))

    def _build_tree_cpu(
        self, positions: np.ndarray, circulations: np.ndarray, radii: np.ndarray
    ) -> None:
        """CPU-based tree construction with transfer to GPU fields."""
        N = len(positions)
        pmin = positions.min(axis=0)
        pmax = positions.max(axis=0)
        center = 0.5 * (pmin + pmax)
        half_size = max(0.5 * float(np.max(pmax - pmin)) * 1.01, 1e-6)

        node_centers, node_half_sizes, node_particles, node_children, node_is_leaf = (
            self._bfs_build_tree(positions, N, center, half_size)
        )

        n_nodes = len(node_centers)
        self.n_nodes[None] = n_nodes

        node_total_circ, node_com, node_avg_radius = self._compute_node_multipoles(
            n_nodes,
            node_is_leaf,
            node_particles,
            node_children,
            node_centers,
            positions,
            circulations,
            radii,
        )

        self._transfer_to_gpu(
            n_nodes,
            node_centers,
            node_half_sizes,
            node_total_circ,
            node_com,
            node_avg_radius,
            node_is_leaf,
            node_children,
        )

        self._pack_and_transfer_leaf_particles(n_nodes, node_is_leaf, node_particles)

    def _set_children_from_numpy(self, children: np.ndarray):
        """Transfer children array using a Taichi kernel for speed.

        Uses a cached temporary field to avoid memory leaks from repeated
        Taichi field allocations (Taichi fields cannot be garbage collected).
        """
        # Use the cached temporary field for the transfer (allocated once in __init__)
        self._temp_children.from_numpy(children)
        self._copy_children_kernel(self._temp_children)

    @ti.kernel
    def _copy_children_kernel(self, temp: ti.template()):
        """Copy children from temp field to node_children."""
        for i, j in ti.ndrange(self.max_nodes, 8):
            self.node_children[i, j] = temp[i, j]

    @ti.func
    def _erf_approx(self, x: ti.f32) -> ti.f32:
        """Fast erf approximation matching the direct Gaussian kernel."""
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.327591100

        sign = ti.cast(1.0, ti.f32)
        x_abs = x
        # Taichi does not allow return inside non-static if; use assignment
        if x < 0.0:
            sign = -1.0
            x_abs = -x

        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * ti.exp(-x_abs * x_abs))
        return sign * y

    @ti.func
    def q_kernel(self, r_sigma: ti.f32) -> ti.f32:
        """Regularization kernel q(ρ) matching the configured particle kernel."""
        ONE_OVER_FOUR_PI = ti.cast(0.07957747154594767, ti.f32)  # 1/(4π)
        result = ti.cast(0.0, ti.f32)
        if self.kernel_type_id[None] == 0:
            # Gaussian kernel
            two_over_sqrt_pi = ti.cast(1.1283791671, ti.f32)
            if r_sigma < 1e-4:
                result = (
                    (4.0 / (3.0 * ti.sqrt(ti.acos(-1.0) ** 3))) * (r_sigma**3) * ONE_OVER_FOUR_PI
                )
            else:
                erf_term = self._erf_approx(r_sigma)
                exp_term = two_over_sqrt_pi * r_sigma * ti.exp(-r_sigma * r_sigma)
                result = (erf_term - exp_term) * ONE_OVER_FOUR_PI
        else:
            # Winckelmans kernel
            r2 = r_sigma * r_sigma
            result = (
                r_sigma * r_sigma * r_sigma * (r2 + 2.5) / ti.pow(r2 + 1.0, 2.5) * ONE_OVER_FOUR_PI
            )
        return result

    @ti.func
    def zeta_kernel(self, r_sigma: ti.f32) -> ti.f32:
        """Vorticity kernel ζ(ρ) matching the configured particle kernel."""
        ONE_OVER_FOUR_PI = ti.cast(0.07957747154594767, ti.f32)  # 1/(4π)
        result = ti.cast(0.0, ti.f32)
        if self.kernel_type_id[None] == 0:
            # Gaussian kernel
            one_over_pi_15 = ti.cast(0.179587122125, ti.f32)
            result = one_over_pi_15 * ti.exp(-r_sigma * r_sigma)
        else:
            # Winckelmans kernel
            r2 = r_sigma * r_sigma
            result = 7.5 / ti.pow(r2 + 1.0, 3.5) * ONE_OVER_FOUR_PI
        return result

    @ti.func
    def skew(self, v: ti.template()) -> ti.Matrix:
        """Compute skew-symmetric matrix from vector.

        skew([a, b, c]) = [[0, -c, b], [c, 0, -a], [-b, a, 0]]
        """
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    @ti.func
    def _leaf_velocity_sum(
        self, node: int, target_pos: ti.template(), target_rad: ti.f32, self_idx: int
    ) -> ti.math.vec3:
        """Direct velocity summation over all particles in a leaf node."""
        vel = ti.Vector([0.0, 0.0, 0.0])
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            if j != self_idx:
                r_vec_j = target_pos - self.positions[j]
                r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
                if r_mag_j > 1e-10:
                    sigma = 0.5 * (target_rad + self.radii[j])
                    q_val = self.q_kernel(r_mag_j / sigma)
                    vel -= q_val * r_vec_j.cross(self.circulations[j]) / (r_mag_j**3)
        return vel

    @ti.func
    def _target_leaf_velocity_sum(self, node: int, target_pos: ti.template()) -> ti.math.vec3:
        """Direct velocity summation over all particles in a leaf for a target point."""
        vel = ti.Vector([0.0, 0.0, 0.0])
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            r_vec_j = target_pos - self.positions[j]
            r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
            if r_mag_j > 1e-10:
                sigma = self.radii[j]
                q_val = self.q_kernel(r_mag_j / sigma)
                vel -= q_val * r_vec_j.cross(self.circulations[j]) / (r_mag_j**3)
        return vel

    @ti.func
    def _leaf_gradient_sum(
        self,
        node: int,
        target_pos: ti.template(),
        target_rad: ti.f32,
        self_idx: int,
        min_r_sigma: ti.f32,
        max_r_sigma: ti.f32,
    ) -> ti.Matrix:
        """Direct velocity-gradient summation over all particles in a leaf node."""
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            if j != self_idx:
                r_vec_j = target_pos - self.positions[j]
                r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
                if r_mag_j > 1e-10:
                    sigma = 0.5 * (target_rad + self.radii[j])
                    r_sigma = r_mag_j / sigma
                    if r_sigma > min_r_sigma and r_sigma < max_r_sigma:
                        q_val = self.q_kernel(r_sigma)
                        zeta_val = self.zeta_kernel(r_sigma) / sigma**3
                        term1 = q_val / r_mag_j**3
                        term2 = 3.0 * q_val / r_mag_j**5 - zeta_val / r_mag_j**2
                        cross_j = r_vec_j.cross(self.circulations[j])
                        gradu += term1 * self.skew(
                            self.circulations[j]
                        ) + term2 * cross_j.outer_product(r_vec_j)
        return gradu

    @ti.func
    def _target_leaf_gradient_sum(
        self, node: int, target_pos: ti.template(), min_r_sigma: ti.f32
    ) -> ti.Matrix:
        """Direct velocity-gradient summation in a leaf node for a target point."""
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            r_vec_j = target_pos - self.positions[j]
            r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
            if r_mag_j > 1e-10:
                sigma = self.radii[j]
                r_sigma = r_mag_j / sigma
                if r_sigma > min_r_sigma:
                    q_val = self.q_kernel(r_sigma)
                    zeta_val = self.zeta_kernel(r_sigma) / sigma**3
                    term1 = q_val / r_mag_j**3
                    term2 = 3.0 * q_val / r_mag_j**5 - zeta_val / r_mag_j**2
                    cross_j = r_vec_j.cross(self.circulations[j])
                    gradu += term1 * self.skew(
                        self.circulations[j]
                    ) + term2 * cross_j.outer_product(r_vec_j)
        return gradu

    @ti.func
    def _push_children_particle(self, i: int, node: int, stack_ptr: int) -> int:
        """Push valid children of *node* onto the per-thread particle traversal stack."""
        for octant in ti.static(range(8)):
            child = self.node_children[node, octant]
            if child >= 0 and stack_ptr < self.max_stack_depth - 1:
                self.traversal_stack[i, stack_ptr] = child
                stack_ptr += 1
        return stack_ptr

    @ti.func
    def _push_children_target(self, i: int, node: int, stack_ptr: int) -> int:
        """Push valid children of *node* onto the per-thread target traversal stack."""
        for octant in ti.static(range(8)):
            child = self.node_children[node, octant]
            if child >= 0 and stack_ptr < self.max_stack_depth - 1:
                self.target_traversal_stack[i, stack_ptr] = child
                stack_ptr += 1
        return stack_ptr

    @ti.func
    def _traverse_particle_vel(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.math.vec3:
        """Iterative Barnes-Hut traversal returning velocity at particle i."""
        vel = ti.Vector([0.0, 0.0, 0.0])
        target_pos = self.positions[i]
        target_rad = self.radii[i]
        self.traversal_stack[i, 0] = 0
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = 0.5 * (target_rad + self.node_avg_radius[node])
                r_sigma = r_mag / sigma
                q_val = self.q_kernel(r_sigma)
                vel -= q_val * r_vec.cross(self.node_total_circ[node]) / (r_mag * r_mag * r_mag)
            elif self.node_is_leaf[node] == 1:
                vel += self._leaf_velocity_sum(node, target_pos, target_rad, i)
            else:
                stack_ptr = self._push_children_particle(i, node, stack_ptr)
        return vel

    @ti.func
    def _traverse_particle_grad(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.Matrix:
        """Iterative Barnes-Hut traversal returning velocity gradient at particle i."""
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        target_pos = self.positions[i]
        target_rad = self.radii[i]
        MIN_R_SIGMA = ti.cast(0.5, ti.f32)
        MAX_R_SIGMA = ti.cast(15.0, ti.f32)
        self.traversal_stack[i, 0] = 0
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = 0.5 * (target_rad + self.node_avg_radius[node])
                r_sigma = r_mag / sigma
                if r_sigma > MIN_R_SIGMA and r_sigma < MAX_R_SIGMA:
                    q_val = self.q_kernel(r_sigma)
                    zeta_val = self.zeta_kernel(r_sigma) / (sigma * sigma * sigma)
                    total_circ = self.node_total_circ[node]
                    term1 = q_val / (r_mag * r_mag * r_mag)
                    term2 = 3.0 * q_val / (r_mag * r_mag * r_mag * r_mag * r_mag) - zeta_val / (
                        r_mag * r_mag
                    )
                    gradu += term1 * self.skew(total_circ) + term2 * r_vec.cross(
                        total_circ
                    ).outer_product(r_vec)
            elif self.node_is_leaf[node] == 1:
                gradu += self._leaf_gradient_sum(
                    node, target_pos, target_rad, i, MIN_R_SIGMA, MAX_R_SIGMA
                )
            else:
                stack_ptr = self._push_children_particle(i, node, stack_ptr)
        return gradu

    @ti.func
    def _traverse_target_vel(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.math.vec3:
        """Iterative Barnes-Hut traversal returning velocity at target point i."""
        vel = ti.Vector([0.0, 0.0, 0.0])
        target_pos = self.target_positions[i]
        self.target_traversal_stack[i, 0] = 0
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.target_traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = self.node_avg_radius[node]
                r_sigma = r_mag / sigma
                q_val = self.q_kernel(r_sigma)
                vel -= q_val * r_vec.cross(self.node_total_circ[node]) / (r_mag * r_mag * r_mag)
            elif self.node_is_leaf[node] == 1:
                vel += self._target_leaf_velocity_sum(node, target_pos)
            else:
                stack_ptr = self._push_children_target(i, node, stack_ptr)
        return vel

    @ti.func
    def _traverse_target_grad(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.Matrix:
        """Iterative Barnes-Hut traversal returning velocity gradient at target point i."""
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        target_pos = self.target_positions[i]
        MIN_R_SIGMA = ti.cast(0.5, ti.f32)
        self.target_traversal_stack[i, 0] = 0
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.target_traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = self.node_avg_radius[node]
                r_sigma = r_mag / sigma
                if r_sigma > MIN_R_SIGMA:
                    q_val = self.q_kernel(r_sigma)
                    zeta_val = self.zeta_kernel(r_sigma) / (sigma * sigma * sigma)
                    total_circ = self.node_total_circ[node]
                    term1 = q_val / (r_mag * r_mag * r_mag)
                    term2 = 3.0 * q_val / (r_mag * r_mag * r_mag * r_mag * r_mag) - zeta_val / (
                        r_mag * r_mag
                    )
                    gradu += term1 * self.skew(total_circ) + term2 * r_vec.cross(
                        total_circ
                    ).outer_product(r_vec)
            elif self.node_is_leaf[node] == 1:
                gradu += self._target_leaf_gradient_sum(node, target_pos, MIN_R_SIGMA)
            else:
                stack_ptr = self._push_children_target(i, node, stack_ptr)
        return gradu

    @ti.kernel
    def compute_velocities_kernel(self, theta_sq: ti.f32):
        """
        GPU kernel for parallel velocity computation using treecode.

        Each thread computes velocity for one particle using iterative
        tree traversal with an explicit stack.
        """
        N = self.n_particles[None]
        n_nodes = self.n_nodes[None]
        for i in range(N):
            self.velocities[i] = (
                self._traverse_particle_vel(i, theta_sq, n_nodes) + self.u_inf[None]
            )

    @ti.kernel
    def compute_velocity_gradients_kernel(self, theta_sq: ti.f32):
        """
        GPU kernel for parallel velocity gradient computation using treecode.

        Computes ∇u at each particle location using the Barnes-Hut algorithm.
        The velocity gradient tensor is: (∇u)_ij = ∂u_i/∂x_j

        The multipole approximation uses the same tree structure as velocity,
        but applies the gradient kernel instead.

        Also computes strain rate tensor: S_ij = 0.5 * (∇u + (∇u)^T)
        """
        N = self.n_particles[None]
        n_nodes = self.n_nodes[None]
        for i in range(N):
            gradu = self._traverse_particle_grad(i, theta_sq, n_nodes)
            self.velocity_gradients[i] = gradu
            strain = ti.Matrix.zero(ti.f32, 3, 3)
            for p in ti.static(range(3)):
                for q in ti.static(range(3)):
                    strain[p, q] = 0.5 * (gradu[p, q] + gradu[q, p])
            self.strain_rates[i] = strain

    def compute_velocities(self, background_velocity: np.ndarray | None = None) -> np.ndarray:
        """
        Compute velocities using GPU-accelerated treecode.

        Args:
            background_velocity: Freestream velocity [3] (optional)

        Returns:
            Velocities [N, 3]
        """
        t_start = time.perf_counter()

        if background_velocity is not None:
            self.u_inf[None] = ti.Vector(background_velocity.astype(np.float32).tolist())
        else:
            self.u_inf[None] = ti.Vector([0.0, 0.0, 0.0])

        theta_sq = self.theta * self.theta
        self.compute_velocities_kernel(theta_sq)
        ti.sync()

        self.eval_time = time.perf_counter() - t_start

        N = self.n_particles[None]
        return self.velocities.to_numpy()[:N]

    def compute_velocity_gradients(self) -> tuple:
        """
        Compute velocity gradients and strain rates using GPU-accelerated treecode.

        Returns:
            tuple: (velocity_gradients [N, 3, 3], strain_rates [N, 3, 3])
        """
        t_start = time.perf_counter()

        theta_sq = self.theta * self.theta
        self.compute_velocity_gradients_kernel(theta_sq)
        ti.sync()

        self.grad_time = time.perf_counter() - t_start

        N = self.n_particles[None]
        grads = self.velocity_gradients.to_numpy()[:N]
        strains = self.strain_rates.to_numpy()[:N]
        return grads, strains

    @ti.kernel
    def compute_target_velocities_kernel(self, theta_sq: ti.f32, avg_radius: ti.f32):
        """
        GPU kernel for computing velocities at arbitrary target positions.

        Uses same tree traversal as particle velocities, but evaluates at
        target_positions instead of particle positions.

        Note: For target points (which have no intrinsic radius), we use only the
        particle/node radius for sigma, matching the direct kernel's behavior.
        """
        M = self.n_targets[None]
        n_nodes = self.n_nodes[None]
        for i in range(M):
            self.target_velocities[i] = (
                self._traverse_target_vel(i, theta_sq, n_nodes) + self.u_inf[None]
            )

    @ti.kernel
    def compute_target_velocity_gradients_kernel(self, theta_sq: ti.f32, avg_radius: ti.f32):
        """
        GPU kernel for computing velocity gradients at arbitrary target positions.

        Note: For target points (which have no intrinsic radius), we use only the
        particle/node radius for sigma, matching the direct kernel's behavior.

        The direct kernel does NOT have a far-field cutoff for target gradients,
        so we also omit MAX_R_SIGMA here to match. Only MIN_R_SIGMA (singularity
        avoidance) is used.
        """
        M = self.n_targets[None]
        n_nodes = self.n_nodes[None]
        for i in range(M):
            self.target_velocity_gradients[i] = self._traverse_target_grad(i, theta_sq, n_nodes)

    def compute_target_velocities(
        self, target_positions: np.ndarray, background_velocity: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute velocities at arbitrary target positions using treecode.

        Args:
            target_positions: Target coordinates [M, 3]
            background_velocity: Freestream velocity [3] (optional)

        Returns:
            np.ndarray: Velocities at targets [M, 3]
        """
        M = len(target_positions)
        if M == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if self.max_targets < M:
            raise ValueError(f"Too many targets: {M} > {self.max_targets}")

        # Set target positions
        self.n_targets[None] = M
        self.target_positions.from_numpy(target_positions.astype(np.float32))

        # Set background velocity
        if background_velocity is not None:
            self.u_inf[None] = ti.Vector(background_velocity.astype(np.float32).tolist())
        else:
            self.u_inf[None] = ti.Vector([0.0, 0.0, 0.0])

        # Use average particle radius for target evaluation
        avg_radius = float(self.radii.to_numpy()[: self.n_particles[None]].mean())

        theta_sq = self.theta * self.theta
        self.compute_target_velocities_kernel(theta_sq, avg_radius)
        ti.sync()

        return self.target_velocities.to_numpy()[:M]

    def compute_target_velocity_gradients(self, target_positions: np.ndarray) -> np.ndarray:
        """
        Compute velocity gradients at arbitrary target positions using treecode.

        Args:
            target_positions: Target coordinates [M, 3]

        Returns:
            np.ndarray: Velocity gradients at targets [M, 3, 3]
        """
        M = len(target_positions)
        if M == 0:
            return np.zeros((0, 3, 3), dtype=np.float32)

        if self.max_targets < M:
            raise ValueError(f"Too many targets: {M} > {self.max_targets}")

        # Set target positions
        self.n_targets[None] = M
        self.target_positions.from_numpy(target_positions.astype(np.float32))

        # Use average particle radius for target evaluation
        avg_radius = float(self.radii.to_numpy()[: self.n_particles[None]].mean())

        theta_sq = self.theta * self.theta
        self.compute_target_velocity_gradients_kernel(theta_sq, avg_radius)
        ti.sync()

        return self.target_velocity_gradients.to_numpy()[:M]

    def info(self) -> str:
        """Return summary string."""
        grad_info = f"\n  Grad time: {self.grad_time * 1000:.2f} ms" if self.grad_time > 0 else ""
        return (
            f"TaichiTreecode (GPU):\n"
            f"  Particles: {self.n_particles[None]}\n"
            f"  Nodes: {self.n_nodes[None]}\n"
            f"  Opening angle θ: {self.theta}\n"
            f"  Build time: {self.build_time * 1000:.2f} ms\n"
            f"  Eval time: {self.eval_time * 1000:.2f} ms{grad_info}"
        )


# =============================================================================
# Convenience function
# =============================================================================


def compute_velocities_treecode_gpu(
    positions: np.ndarray,
    circulations: np.ndarray,
    radii: np.ndarray,
    theta: float = 0.5,
    background_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute velocities using GPU-accelerated Barnes-Hut treecode.

    Args:
        positions: Particle positions [N, 3]
        circulations: Particle circulations [N, 3]
        radii: Particle core radii [N]
        theta: Opening angle (default 0.5)
        background_velocity: Freestream velocity [3] (optional)

    Returns:
        Velocities [N, 3]
    """
    N = len(positions)
    tree = TaichiTreecode(max_particles=N, max_nodes=2 * N, theta=theta)
    tree.build(positions, circulations, radii)
    return tree.compute_velocities(background_velocity)
