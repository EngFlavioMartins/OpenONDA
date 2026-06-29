"""
GPU spatial hash-grid neighbour search (TaichiNeighborSearch): bins particles
into cells and builds fixed-count neighbour lists for short-range interactions.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import math

import taichi as ti

# Import VPM constants
from ..config.constants import MAX_PARTICLES

@ti.data_oriented
class TaichiNeighborSearch:
    def __init__(
        self, domain_size: list = None, max_neighbors: int = 32, max_particles: int = MAX_PARTICLES
    ):
        """
        Initialize spatial hash grid for neighbor search.

        Args:
            domain_size: [x_size, y_size, z_size] of simulation domain, or None for auto-detection
            max_neighbors: Maximum neighbors per particle
        """
        # Default domain size if not provided
        if domain_size is None:
            domain_size = [10.0, 10.0, 10.0]  # Default 10x10x10 domain

        # Default cell size for the grid (will be adapted based on particle distribution)
        domain_extent = max(domain_size)
        self.cell_size = domain_extent / 32.0  # Start with reasonable grid resolution
        self.max_neighbors = max_neighbors
        self.max_particles_per_cell = 64  # Adjust based on particle density
        self.max_particles = max_particles

        # Grid resolution
        self.grid_res = [
            int(domain_size[0] / self.cell_size) + 1,
            int(domain_size[1] / self.cell_size) + 1,
            int(domain_size[2] / self.cell_size) + 1,
        ]

        # Domain bounds for clamping
        self.domain_min = ti.Vector([-domain_size[0] / 2, -domain_size[1] / 2, -domain_size[2] / 2])
        self.domain_max = ti.Vector([domain_size[0] / 2, domain_size[1] / 2, domain_size[2] / 2])

        # Hash grid data structures
        self.cell_count = ti.field(dtype=ti.i32, shape=self.grid_res)
        self.cell_list = ti.field(dtype=ti.i32, shape=(*self.grid_res, self.max_particles_per_cell))

        # Neighbor storage
        self.distances = ti.field(
            dtype=ti.f32, shape=(self.max_particles, max_neighbors)
        )  # Store distances
        self.neighbors = ti.field(
            dtype=ti.i32, shape=(self.max_particles, max_neighbors)
        )  # Resize as needed
        self.neighbor_counts = ti.field(dtype=ti.i32, shape=self.max_particles)

    @ti.func
    def get_cell_index(self, pos):
        """Convert position to grid cell index."""
        # Clamp position to domain bounds
        clamped_pos = ti.Vector(
            [
                ti.max(ti.min(pos[0], self.domain_max[0]), self.domain_min[0]),
                ti.max(ti.min(pos[1], self.domain_max[1]), self.domain_min[1]),
                ti.max(ti.min(pos[2], self.domain_max[2]), self.domain_min[2]),
            ]
        )

        # Convert to cell indices
        cell = ((clamped_pos - self.domain_min) / self.cell_size).cast(int)

        # Clamp to grid bounds
        cell[0] = ti.max(0, ti.min(cell[0], self.grid_res[0] - 1))
        cell[1] = ti.max(0, ti.min(cell[1], self.grid_res[1] - 1))
        cell[2] = ti.max(0, ti.min(cell[2], self.grid_res[2] - 1))

        return cell

    @ti.func
    def _is_cell_in_bounds(self, cell):
        """Check if grid cell is within domain bounds."""
        return (
            cell[0] >= 0
            and cell[0] < self.grid_res[0]
            and cell[1] >= 0
            and cell[1] < self.grid_res[1]
            and cell[2] >= 0
            and cell[2] < self.grid_res[2]
        )

    @ti.func
    def _process_cell_neighbors(
        self,
        i,
        pos_i,
        neighbor_cell,
        positions,
        search_radius,
        current_neighbor_count,
        store_distances: ti.template(),
    ):
        """Process all particles in a single grid cell and add to neighbor list if within radius."""
        neighbor_count = current_neighbor_count
        cell_particle_count = self.cell_count[neighbor_cell]

        # Clamp to max_particles_per_cell to avoid out-of-bounds access
        actual_count = ti.min(cell_particle_count, self.max_particles_per_cell)

        for j in range(actual_count):
            if neighbor_count >= self.max_neighbors:
                continue
            particle_id = self.cell_list[neighbor_cell[0], neighbor_cell[1], neighbor_cell[2], j]
            if particle_id == i or particle_id >= positions.shape[0]:
                continue
            r_vec = positions[particle_id] - pos_i
            r_mag = r_vec.norm()
            if r_mag < search_radius:
                self.neighbors[i, neighbor_count] = particle_id
                if ti.static(store_distances):
                    self.distances[i, neighbor_count] = r_mag
                neighbor_count += 1

        return neighbor_count

    @ti.func
    def _search_neighbor_cells(
        self, i, pos_i, cell_i, positions, search_radius, store_distances: ti.template()
    ):
        """Search all 27 neighboring cells for particle i."""
        neighbor_count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbor_cell = cell_i + ti.Vector([dx, dy, dz])

                    if self._is_cell_in_bounds(neighbor_cell):
                        neighbor_count = self._process_cell_neighbors(
                            i,
                            pos_i,
                            neighbor_cell,
                            positions,
                            search_radius,
                            neighbor_count,
                            store_distances,
                        )
        return neighbor_count

    @ti.kernel
    def clear_grid(self):
        """Clear the spatial hash grid."""
        for i, j, k in ti.ndrange(self.grid_res[0], self.grid_res[1], self.grid_res[2]):
            self.cell_count[i, j, k] = 0

    @ti.kernel
    def bin_particles(self, positions: ti.template(), N: ti.i32):  # type: ignore
        """Bin particles into spatial hash grid."""
        for i in range(N):
            cell = self.get_cell_index(positions[i])

            # Only add particle if there's space in the cell
            current_count = self.cell_count[cell]
            if current_count < self.max_particles_per_cell:
                # Atomically add particle to cell
                old_count = ti.atomic_add(self.cell_count[cell], 1)
                if old_count < self.max_particles_per_cell:
                    self.cell_list[cell[0], cell[1], cell[2], old_count] = i

    @ti.kernel
    def build_neighbors(self, positions: ti.template(), N: ti.i32, search_radius: ti.f32):  # type: ignore
        """Build neighbor lists using spatial hash grid with bounds checking."""
        for i in range(N):
            if i < self.neighbors.shape[0]:
                pos_i = positions[i]
                cell_i = self.get_cell_index(pos_i)
                self.neighbor_counts[i] = self._search_neighbor_cells(
                    i, pos_i, cell_i, positions, search_radius, False
                )

    @ti.kernel
    def build_neighbors_with_distances(
        self, positions: ti.template(), N: ti.i32, search_radius: ti.f32
    ):  # type: ignore
        """Build neighbor lists and distances using spatial hash grid with bounds checking."""
        for i in range(N):
            if i < self.neighbors.shape[0]:
                pos_i = positions[i]
                cell_i = self.get_cell_index(pos_i)
                self.neighbor_counts[i] = self._search_neighbor_cells(
                    i, pos_i, cell_i, positions, search_radius, True
                )

    # ---- + ----

    def resize_if_needed(self, N):
        """Dynamically resize neighbor storage if needed. Over-allocate by 10% for safety."""
        target_size = int(N * 1.5)  # Over-allocate by 50%
        if target_size > self.neighbors.shape[0]:
            print(
                f"(Info) Resizing neighbor storage from {self.neighbors.shape[0]} to {target_size} particles"
            )
            try:
                self.distances = ti.field(dtype=ti.f32, shape=(target_size, self.max_neighbors))
                self.neighbors = ti.field(dtype=ti.i32, shape=(target_size, self.max_neighbors))
                self.neighbor_counts = ti.field(dtype=ti.i32, shape=target_size)
                self.max_particles = target_size
                print(f"(Info) Neighbor storage resized successfully to {target_size} particles")
            except Exception as e:
                print(f"(error) Failed to resize neighbor storage: {e}")
                raise RuntimeError(
                    f"Cannot resize neighbor storage from {self.neighbors.shape[0]} to {target_size} particles: {e}"
                ) from e

    # ---- + ----

    def find_neighbors_fixed_count(self, positions, target_neighbors):
        """
        GPU-only fixed-count neighbor search.
        Returns Taichi fields for neighbors and neighbor counts.
        Args:
            positions: Taichi field of particle positions [N, 3]
            target_neighbors: Desired number of neighbors
        Returns:
            neighbors: Taichi field [N, max_neighbors]
            neighbor_counts: Taichi field [N]
        """
        N = positions.shape[0]
        self.resize_if_needed(N)
        # Estimate search radius for target neighbor count
        search_radius = ((3.0 * target_neighbors) / (4.0 * math.pi)) ** (1.0 / 3.0) * self.cell_size
        search_radius *= 1.5  # Safety margin
        self.clear_grid()
        self.bin_particles(positions, N)
        self.build_neighbors_with_distances(positions, N, search_radius)
        # Return Taichi fields directly (no numpy conversion)
        return self.neighbors, self.neighbor_counts

    def find_neighbors_for_filter(self, positions, target_neighbors, active_particles=None):
        """
        Get neighbors based on desired number of closest neighboring particles.

        Args:
            positions: Taichi field of particle positions
            target_neighbors: Desired number of closest neighbors
            active_particles: Number of active particles (optional, defaults to positions.shape[0])
        Returns:
            tuple: (neighbors, neighbor_counts) as Taichi fields
        """
        N = active_particles if active_particles is not None else positions.shape[0]
        self.resize_if_needed(N)
        # Estimate search radius for target neighbor count
        search_radius = ((3.0 * target_neighbors) / (4.0 * math.pi)) ** (1.0 / 3.0) * self.cell_size
        search_radius *= 1.5  # Safety margin
        self.clear_grid()
        self.bin_particles(positions, N)
        self.build_neighbors_with_distances(positions, N, search_radius)
        # Return Taichi fields directly (no numpy conversion)
        return self.neighbors, self.neighbor_counts

    def find_neighbors(self, positions, target_neighbors):
        """Alias for find_neighbors_fixed_count."""
        return self.find_neighbors_fixed_count(positions, target_neighbors)
