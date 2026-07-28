"""
Seeds vortex particles over flow fields and regions (ParticleDistributor).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
from scipy.spatial import cKDTree

# =========================================================


class ParticleDistributor:
    """
    A comprehensive class for generating and manipulating particle distributions
    in 3D space for computational fluid dynamics and particle methods.

    This class provides various distribution patterns (rectangular, hexagonal,
    cylindrical) and utilities for particle manipulation (removal, splitting).
    """

    EPSILON = 1e-10

    def __init__(self, default_radius: float = 1.5):
        """
        Initialize the ParticleDistributor class.

        Args:
              default_radius (float): Default particle radius. Defaults to 1.5.
        """
        self.default_radius = default_radius

    @staticmethod
    def compute_min_max(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, ...]:
        """
        Compute min and max for each axis.

        Args:
              x (np.ndarray): x-coordinates.
              y (np.ndarray): y-coordinates.
              z (np.ndarray): z-coordinates.

        Returns:
              Tuple[float, float, float, float, float, float]: xmin, xmax, ymin, ymax, zmin, zmax
        """
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)
        return xmin, xmax, ymin, ymax, zmin, zmax

    @staticmethod
    def gaussian(r: np.ndarray, omega_0: float, a: float) -> np.ndarray:
        """
        Gaussian function for fitting particle distributions.

        Args:
              r (np.ndarray): Radial distances.
              omega_0 (float): Peak amplitude.
              a (float): Characteristic length scale.

        Returns:
              np.ndarray: Gaussian values.
        """
        return omega_0 * np.exp(-(r**2) / a**2)

    @staticmethod
    def rectangular_distribution(
        domain_bounds: list, spacing: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a rectangular (box) distribution of particles.

        Args:
              domain_bounds (list): [xmin, xmax, ymin, ymax, zmin, zmax] bounds.
              spacing (float): Average distance between particles.
              radius (float, optional): Particle radius. Uses default if None.

        Returns:
              Tuple containing:
              - positions (np.ndarray): 3D coordinates of particles
              - volumes (np.ndarray): Volume associated with each particle
              - radii (np.ndarray): Radius of each particle
        """
        # particle radius is derived from spacing (single computation)
        radius = 2 * spacing

        numx = int(np.ceil((domain_bounds[1] - domain_bounds[0]) / spacing)) + 1
        numy = int(np.ceil((domain_bounds[3] - domain_bounds[2]) / spacing)) + 1
        numz = int(np.ceil((domain_bounds[5] - domain_bounds[4]) / spacing)) + 1

        # Ensure odd number of positions for symmetry
        if numx % 2 == 0:
            numx += 1
        if numy % 2 == 0:
            numy += 1
        if numz % 2 == 0:
            numz += 1

        x = np.linspace(domain_bounds[0], domain_bounds[1], num=numx)
        y = np.linspace(domain_bounds[2], domain_bounds[3], num=numy)
        z = np.linspace(domain_bounds[4], domain_bounds[5], num=numz)

        dx = x[1] - x[0] if len(x) > 1 else spacing
        dy = y[1] - y[0] if len(y) > 1 else spacing
        dz = z[1] - z[0] if len(z) > 1 else spacing

        positions = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
        volume_per_point = dx * dy * dz
        volumes = np.full(positions.shape[0], volume_per_point)
        radii = np.full(positions.shape[0], radius)

        return positions, volumes, radii

    @staticmethod
    def noisy_distribution(
        domain_bounds: list,
        spacing: float,
        noise_level: float = 0.3,
        seed: int | None = None,
        k_neighbors: int = 6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a noisy rectangular distribution of particles.

        Particles are initially placed on a regular grid, then randomly displaced
        within their grid cell. Radii are computed based on actual nearest neighbor
        distances after randomization to ensure proper kernel support.

        Args:
              domain_bounds (list): [xmin, xmax, ymin, ymax, zmin, zmax] bounds.
              spacing (float): Average distance between particles (grid spacing).
              noise_level (float): Noise amplitude as fraction of spacing.
                    Default is 0.3, meaning particles can move ±0.3*spacing/2
                    from their grid position. Valid range: [0.0, 1.0].
                    - 0.0: Regular grid (no noise)
                    - 1.0: Maximum noise (particles can move to cell boundaries)
              seed (int, optional): Random seed for reproducibility.
              k_neighbors (int): Number of nearest neighbors to consider for
                    radius computation. Default is 6 (typical for 3D grid).

        Returns:
              Tuple containing:
              - positions (np.ndarray): 3D coordinates of particles with noise
              - volumes (np.ndarray): Volume associated with each particle
              - radii (np.ndarray): Radius of each particle based on local spacing

        Example:
              >>> # 10 particles along 1m length with noise
              >>> # Each particle at 0.1m ± noise*0.1m
              >>> positions, vols, radii = noisy_distribution(
              ...     domain_bounds=[0, 1, 0, 1, 0, 1],
              ...     spacing=0.1,
              ...     noise_level=0.3
              ... )
        """
        # Validate noise level
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"noise_level must be between 0.0 and 1.0, got {noise_level}")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Calculate number of particles in each direction
        numx = int(np.ceil((domain_bounds[1] - domain_bounds[0]) / spacing)) + 1
        numy = int(np.ceil((domain_bounds[3] - domain_bounds[2]) / spacing)) + 1
        numz = int(np.ceil((domain_bounds[5] - domain_bounds[4]) / spacing)) + 1

        # Ensure odd number of positions for symmetry
        if numx % 2 == 0:
            numx += 1
        if numy % 2 == 0:
            numy += 1
        if numz % 2 == 0:
            numz += 1

        # Create regular grid
        x = np.linspace(domain_bounds[0], domain_bounds[1], num=numx)
        y = np.linspace(domain_bounds[2], domain_bounds[3], num=numy)
        z = np.linspace(domain_bounds[4], domain_bounds[5], num=numz)

        # Calculate actual grid spacing
        dx = x[1] - x[0] if len(x) > 1 else spacing
        dy = y[1] - y[0] if len(y) > 1 else spacing
        dz = z[1] - z[0] if len(z) > 1 else spacing

        # Create meshgrid for regular positions
        positions = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

        # Add noise to each position
        # Noise is uniformly distributed within ±noise_level * spacing/2
        # This ensures particles stay within their grid cell
        noise_x = (np.random.rand(positions.shape[0]) - 0.5) * noise_level * dx
        noise_y = (np.random.rand(positions.shape[0]) - 0.5) * noise_level * dy
        noise_z = (np.random.rand(positions.shape[0]) - 0.5) * noise_level * dz

        # Apply noise to positions
        positions[:, 0] += noise_x
        positions[:, 1] += noise_y
        positions[:, 2] += noise_z

        # Clamp positions to domain bounds to handle edge cases
        positions[:, 0] = np.clip(positions[:, 0], domain_bounds[0], domain_bounds[1])
        positions[:, 1] = np.clip(positions[:, 1], domain_bounds[2], domain_bounds[3])
        positions[:, 2] = np.clip(positions[:, 2], domain_bounds[4], domain_bounds[5])

        # Calculate volumes (still based on nominal grid spacing)
        volume_per_point = dx * dy * dz
        volumes = np.full(positions.shape[0], volume_per_point)

        # Compute radii based on actual nearest neighbor distances
        n_particles = positions.shape[0]
        radii = np.zeros(n_particles)

        # For large particle counts, use a cell-based approach
        # Divide domain into cells and only check nearby particles
        tree = cKDTree(positions)

        for i in range(n_particles):
            # Query k+1 neighbors (includes self)
            distances, _ = tree.query(positions[i], k=k_neighbors + 1)
            # Skip the first element (self at distance 0)
            k_nearest = distances[1:]

            # Set radius based on mean of k nearest neighbors
            mean_distance = np.mean(k_nearest)
            radii[i] = 1.5 * mean_distance  # 0.8 * mean_distance**0.5

        return positions, volumes, radii

    @staticmethod
    def rectangular_2d_distribution(
        domain_bounds: list,
        spacing: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a 2D rectangular distribution of particles (z=0)."""

        # Calculate number of points to best match the spacing argument
        numx = int(np.round((domain_bounds[1] - domain_bounds[0]) / spacing)) + 1
        numy = int(np.round((domain_bounds[3] - domain_bounds[2]) / spacing)) + 1

        # More direct approach - create coordinates directly
        x_coords = np.linspace(domain_bounds[0], domain_bounds[1], numx)
        y_coords = np.linspace(domain_bounds[2], domain_bounds[3], numy)

        xx, yy = np.meshgrid(x_coords, y_coords, indexing="ij")
        positions = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

        total_positions = len(positions)
        total_area = (domain_bounds[1] - domain_bounds[0]) * (domain_bounds[3] - domain_bounds[2])

        # compute particle radius from spacing
        radius = 2 * spacing
        volumes = np.full(total_positions, total_area / total_positions)
        radii = np.full(total_positions, radius)

        return positions, volumes, radii

    @staticmethod
    def hexagonal_distribution(
        domain_bounds: list | None = None,
        spacing: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a hexagonal lattice distribution of particles.

        Args:
              domain_bounds (list, optional): Domain bounds. Uses default if None.
              spacing (float): Lattice spacing.
              radius (float): Particle radius.

        Returns:
              Tuple containing positions, volumes, and radii arrays.
        """
        if domain_bounds is None:
            domain_bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

        x_min, x_max, y_min, y_max, z_min, z_max = domain_bounds

        x_range = np.arange(x_min, x_max + spacing / 2, spacing)
        y_range = np.arange(y_min, y_max + spacing / 2, spacing * np.sqrt(3) / 2)
        z_range = np.arange(z_min, z_max + spacing / 2, spacing)

        # Ensure at least one point if bounds are singular
        if len(x_range) == 0:
            x_range = np.array([x_min])
        if len(y_range) == 0:
            y_range = np.array([y_min])
        if len(z_range) == 0:
            z_range = np.array([z_min])

        positions_list: list[list[float]] = []
        for z in z_range:
            for i, y in enumerate(y_range):
                x_offset = spacing / 2 if i % 2 == 1 else 0
                for x in x_range:
                    positions_list.append([x + x_offset, y, z])

        positions = np.array(positions_list)

        # Determine effective volume/area
        # If one dimension is singular, it's 2D (Area); if two, it's 1D (Length).
        singular_dims = (x_min == x_max) + (y_min == y_max) + (z_min == z_max)

        if singular_dims == 0:
            volume = spacing**3 * np.sqrt(3) / 2
        elif singular_dims == 1:
            volume = spacing**2 * np.sqrt(3) / 2  # Area * 1.0 thickness
        else:
            volume = spacing  # Length * 1.0*1.0 area

        # compute particle radius from spacing
        radius = 2 * spacing
        volumes = np.full(positions.shape[0], volume)
        radii = np.full(positions.shape[0], radius)

        return positions, volumes, radii

    @staticmethod
    def cylindrical_distribution(
        cyl_radius: float = 1.0,
        height: float = 2.0,
        center: np.ndarray = np.array([0.0, 0.0, 0.0]),
        axis: str = "z",
        spacing: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a cylindrical distribution of particles.

        Args:
              radius (float): Cylinder radius.
              height (float): Cylinder height.
              center (np.ndarray): Cylinder center coordinates.
              axis (str): Cylinder axis ('x', 'y', or 'z').
              spacing (float): Average particle spacing.
              radius (float): Particle radius.

        Returns:
              Tuple containing positions, volumes, and radii arrays.
        """
        if axis not in ("x", "y", "z"):
            raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

        num_r = int(cyl_radius / spacing)
        num_z = int(height / spacing)
        num_t = int(np.pi * cyl_radius / spacing)

        dtheta = 2 * np.pi / num_t
        dz = height / num_z
        dr = cyl_radius / num_r

        r = np.linspace(dr, cyl_radius, num=num_r)
        z = np.linspace(-height / 2, height / 2, num=num_z + 1)
        theta = np.linspace(0, 2 * np.pi, num=num_t)[:-1]

        RR, ZZ, TT = np.meshgrid(r, z, theta, indexing="ij")
        X = RR * np.cos(TT)
        Y = RR * np.sin(TT)
        VOLS = (RR * dtheta) * dr * dz

        if axis == "x":
            pts = np.stack([ZZ + center[0], X + center[1], Y + center[2]], axis=-1)
            cax = np.stack(
                [z + center[0], np.full_like(z, center[1]), np.full_like(z, center[2])], axis=1
            )
        elif axis == "y":
            pts = np.stack([X + center[0], ZZ + center[1], Y + center[2]], axis=-1)
            cax = np.stack(
                [np.full_like(z, center[0]), z + center[1], np.full_like(z, center[2])], axis=1
            )
        else:
            pts = np.stack([X + center[0], Y + center[1], ZZ + center[2]], axis=-1)
            cax = np.stack(
                [np.full_like(z, center[0]), np.full_like(z, center[1]), z + center[2]], axis=1
            )

        positions = np.concatenate([pts.reshape(-1, 3), cax])
        center_vol = np.pi * (cyl_radius**2) * dz / len(z)
        volumes = np.concatenate([VOLS.reshape(-1), np.full(len(z), center_vol)])
        radii = np.full(positions.shape[0], 2 * spacing)

        return positions, volumes, radii

    def remove_weak_particles(
        self, psys, mode: str, weakest_percent: float, conserve_total_circulation: bool = False
    ) -> None:
        """
        Remove particles with low strength from the system.

        Args:
              psys: Particle system object.
              mode (str): 'absolute' or 'relative' threshold mode.
              weakest_percent (float): Threshold value (percentage or absolute).
              conserve_total_circulation (bool): Whether to conserve total circulation.
        """
        threshold = weakest_percent / 100
        particles_strength_mag = psys.get_particle_strength_magnitudes()
        total_strength_before = np.sum(particles_strength_mag)
        num_particles_before = len(psys)

        if len(particles_strength_mag) == 0:
            print("(warning) No particles available to evaluate.")
            return

        if mode == "absolute":
            weak_particles_list = particles_strength_mag < threshold
        elif mode == "relative":
            highest_strength = np.max(particles_strength_mag)
            if highest_strength == 0:
                print("(warning) All particle strengths are zero.")
                weak_particles_list = np.ones_like(particles_strength_mag, dtype=bool)
            else:
                weak_particles_list = particles_strength_mag / highest_strength < threshold
        else:
            print(f"(error) Mode '{mode}' not recognized. Use 'absolute' or 'relative'.")
            return

        weak_particles = np.where(weak_particles_list)[0]
        psys.remove_particles(weak_particles)

        if conserve_total_circulation:
            particles_strength_mag = psys.get_particle_strength_magnitudes()
            total_strength_after = np.sum(particles_strength_mag)
            particles_strength = psys.get_particle_strengths()
            correction = total_strength_before / (total_strength_after + self.EPSILON)

            for p, particle in enumerate(psys.particles):
                particle.update_state(strength=particles_strength[p] * correction)

        psys._cache_particle_arrays()
        print(f"\tRemoved {num_particles_before - len(psys)} particles from the system.")

    @staticmethod
    def get_highest_nonzero_indices(
        arr: np.ndarray, top_percentage: int | None = None, strength_threshold: float | None = None
    ) -> np.ndarray:
        """
        Get indices of highest non-zero entries in an array.

        Args:
              arr (np.ndarray): Input array.
              top_percentage (int, optional): Percentage of top entries to return.
              strength_threshold (float, optional): Minimum threshold value.

        Returns:
              np.ndarray: Indices of selected entries.
        """
        non_zero_indices = np.nonzero(arr)[0]
        selected_values = arr[non_zero_indices]
        sorted_indices = np.argsort(selected_values)[::-1]

        if top_percentage is not None:
            top_n = max(int(top_percentage * len(non_zero_indices) / 100), 1)
            highest_nonzero_indices = non_zero_indices[sorted_indices[:top_n]]
        else:
            if strength_threshold is not None:
                valid_indices = selected_values > strength_threshold
                selected_values = selected_values[valid_indices]
                non_zero_indices = non_zero_indices[valid_indices]
                sorted_indices = np.argsort(selected_values)[::-1]
            highest_nonzero_indices = non_zero_indices[sorted_indices]

        return highest_nonzero_indices
