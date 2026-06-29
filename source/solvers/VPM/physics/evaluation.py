"""
On-particle field evaluation (ParticleFieldEvaluation): caches Taichi fields and
computes velocity/vorticity at particle and target points.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti

# Import VPM constants
from ..config.constants import (
    DEFAULT_CUTOFF_RADIUS_FACTOR,
    EPSILON,
    MAX_PARTICLES,
)

@ti.data_oriented
class ParticleFieldEvaluation:
    """
    Manages all flow integral diagnostics for particle fields.

    This class provides GPU-accelerated computation of global flow quantities:
    - Energy metrics: kinetic energy and dissipation rates
    - Vorticity metrics: helicity, enstrophy, dissipation
    - Conservation quantities: total strength, impulses
    - Group analysis: centroids of circulation

    All computations use unbounded domain definitions and are optimized for GPU execution.
    Results are kept on GPU and only transferred to CPU when accessed.
    """

    def __init__(
        self,
        particles_kernel: str = "GAUSSIAN",
        max_particles: int = MAX_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
    ):
        """
        Initialize particle field physics diagnostics.

        Args:
            particles_kernel: Kernel type ("GAUSSIAN", "WINCKELMANS", "SUPER_GAUSSIAN")
            max_particles: Maximum number of particles for field allocation
            accumulator_dtype: Taichi dtype for accumulation (ti.f32 or ti.f64)
        """
        self.max_particles = max_particles
        self.particles_kernel = particles_kernel.upper()
        self.accumulator_dtype = accumulator_dtype

        # Initialize GPU fields for storing results
        self._initialize_result_fields()

        # Initialize time tracking for energy dissipation rate
        self._flow_time_history = []  # Store (time, energy) pairs
        self._max_history_length = (
            7  # Keep 7 points: poly-degree-3 regression gives implicit smoothing
        )

        # Define Taichi kernels
        self._define_taichi_kernels()

    def _initialize_result_fields(self):
        """Initialize Taichi fields for storing diagnostic results on GPU."""
        # Results struct for flow integrals kernel — named fields eliminate
        # the 'Field index not int32' warning from integer literal indexing.
        _F = self.accumulator_dtype
        FlowIntegralsStruct = ti.types.struct(
            energy=_F,
            helicity=_F,
            enstrophy=_F,
            dissipation=_F,
            str_mag=_F,
            gamma_x=_F,
            gamma_y=_F,
            gamma_z=_F,
            imp_x=_F,
            imp_y=_F,
            imp_z=_F,
            ang_x=_F,
            ang_y=_F,
            ang_z=_F,
        )
        self.total_quantities_results = FlowIntegralsStruct.field(shape=())

        # Separate fields for per-particle diagnostics
        self.particle_kinetic_energy = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.particle_helicity = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.particle_enstrophy = ti.field(dtype=ti.f32, shape=self.max_particles)

        # Cached centroid result fields (to avoid memory leak from repeated allocations)
        # CRITICAL: Taichi fields cannot be garbage collected, so we cache and reuse
        self._centroid_result = ti.field(dtype=ti.f32, shape=3)

    def _resize_fields(self, required_size: int):
        """Resize particle diagnostic fields if needed."""
        if required_size <= 0:
            return

        target_size = int(required_size)
        if target_size > self.max_particles:
            old_max = self.max_particles
            self.max_particles = target_size
            print(f"(Info) Resizing particle field physics from {old_max} to {self.max_particles}")

            # Re-initialize per-particle fields
            self.particle_kinetic_energy = ti.field(dtype=ti.f32, shape=self.max_particles)
            self.particle_helicity = ti.field(dtype=ti.f32, shape=self.max_particles)
            self.particle_enstrophy = ti.field(dtype=ti.f32, shape=self.max_particles)

    def __str__(self):
        """Return formatted string representation."""
        lines = []
        lines.append(f"  Kernel Type              : {self.particles_kernel}")
        lines.append("  Domain Type              : Unbounded (diagnostic computations)")
        lines.append(f"  Accumulator Precision    : {self.accumulator_dtype}")
        lines.append(f"  Max Particles (allocated): {self.max_particles:,}")
        lines.append(
            f"  Time History Length      : {len(self._flow_time_history)} / {self._max_history_length}"
        )
        return "\n".join(lines)

    # TAICHI KERNEL DEFINITIONS

    def _define_taichi_kernels(self):
        """Define Taichi kernels based on the selected kernel type."""

        # Import kernel creator function based on type
        if self.particles_kernel == "GAUSSIAN":
            from ..kernels.gaussian import create_gaussian_kernels

            kernel_dict = create_gaussian_kernels(self.accumulator_dtype)
        elif self.particles_kernel == "HIGH_ORDER_GAUSSIAN":
            from ..kernels.high_order_gaussian import create_high_order_gaussian_kernels

            kernel_dict = create_high_order_gaussian_kernels(self.accumulator_dtype)
        elif self.particles_kernel == "SUPER_GAUSSIAN":
            from ..kernels.super_gaussian import create_super_gaussian_kernels

            kernel_dict = create_super_gaussian_kernels(self.accumulator_dtype)
        elif self.particles_kernel == "WINCKELMANS":
            from ..kernels.winckelmans import create_winckelmans_kernels

            kernel_dict = create_winckelmans_kernels(self.accumulator_dtype)
        else:
            raise ValueError(f"Unknown particles_kernel: {self.particles_kernel}")

        # Extract kernel functions from dictionary
        kernel_functions = {
            "q_sigma": kernel_dict["q_"],
            "zeta_sigma": kernel_dict["zeta_"],
            "g_sigma": kernel_dict["g_"],
        }

        # Define all kernels
        self._define_flow_integral_kernels(kernel_functions, kernel_dict)
        self._define_per_particle_kernels(kernel_functions)
        self._define_centroid_kernels(kernel_functions)

    def _define_flow_integral_kernels(self, kernel_functions, kernel_dict):
        """Define kernels for computing total flow integrals."""

        g_ = kernel_functions["g_sigma"]
        q_ = kernel_functions["q_sigma"]
        zeta_ = kernel_functions["zeta_sigma"]

        # Get angular impulse correction constant from kernel
        angular_correction_func = kernel_dict["angular_impulse_correction_constant_"]

        @ti.kernel
        def compute_flow_integrals_kernel(
            positions: ti.template(),
            strengths: ti.template(),
            radii: ti.template(),
            viscosities_eff: ti.template(),
            results: ti.template(),
            num_particles: ti.i32,
        ):  # type: ignore
            """
            Compute all flow integrals in a single optimized GPU kernel.

            This combines energy, helicity, enstrophy, dissipation rates, strength,
            and impulses into one efficient computation using unbounded definitions.

            Angular impulse includes kernel correction per Winckelmans 1993:
            A = (1/3) Σ r × (r × Γ) - (2/9) C σ² Γ_total
            """
            N = num_particles

            # Initialize every field to zero explicitly
            results[None].energy = ti.cast(0.0, self.accumulator_dtype)
            results[None].helicity = ti.cast(0.0, self.accumulator_dtype)
            results[None].enstrophy = ti.cast(0.0, self.accumulator_dtype)
            results[None].dissipation = ti.cast(0.0, self.accumulator_dtype)
            results[None].str_mag = ti.cast(0.0, self.accumulator_dtype)
            results[None].gamma_x = ti.cast(0.0, self.accumulator_dtype)
            results[None].gamma_y = ti.cast(0.0, self.accumulator_dtype)
            results[None].gamma_z = ti.cast(0.0, self.accumulator_dtype)
            results[None].imp_x = ti.cast(0.0, self.accumulator_dtype)
            results[None].imp_y = ti.cast(0.0, self.accumulator_dtype)
            results[None].imp_z = ti.cast(0.0, self.accumulator_dtype)
            results[None].ang_x = ti.cast(0.0, self.accumulator_dtype)
            results[None].ang_y = ti.cast(0.0, self.accumulator_dtype)
            results[None].ang_z = ti.cast(0.0, self.accumulator_dtype)

            # Single loop for simple quantities (strength and impulses)
            for i in range(N):
                str_i = strengths[i]
                pos_i = positions[i]

                # Total strength magnitude: Σ|Γ| (sum of magnitudes, NOT |ΣΓ|)
                str_mag = str_i.norm()
                ti.atomic_add(results[None].str_mag, str_mag)

                # Total strength vector: ΣΓ (atomic for thread safety)
                ti.atomic_add(results[None].gamma_x, str_i[ti.static(0)])
                ti.atomic_add(results[None].gamma_y, str_i[ti.static(1)])
                ti.atomic_add(results[None].gamma_z, str_i[ti.static(2)])

                # Linear impulse: I = 0.5 * Σ (r × Γ)
                cross_product = pos_i.cross(str_i)
                ti.atomic_add(results[None].imp_x, cross_product[ti.static(0)] * 0.5)
                ti.atomic_add(results[None].imp_y, cross_product[ti.static(1)] * 0.5)
                ti.atomic_add(results[None].imp_z, cross_product[ti.static(2)] * 0.5)

                # Angular impulse (raw, without correction): (1/3) Σ (r × (r × Γ))
                r_cross_gamma = pos_i.cross(str_i)
                angular_contrib = pos_i.cross(r_cross_gamma)
                ti.atomic_add(results[None].ang_x, angular_contrib[ti.static(0)] * (1.0 / 3.0))
                ti.atomic_add(results[None].ang_y, angular_contrib[ti.static(1)] * (1.0 / 3.0))
                ti.atomic_add(results[None].ang_z, angular_contrib[ti.static(2)] * (1.0 / 3.0))

            # Apply kernel correction to angular impulse: subtract (2/9) C σ² Γ_total
            # Get kernel-specific correction constant
            C = angular_correction_func()

            # Compute weighted mean σ² (weighted by |Γ|)
            sigma_sq_weighted_sum = ti.cast(0.0, self.accumulator_dtype)
            total_circ_mag = results[None].str_mag  # Σ|Γ|

            for i in range(N):
                str_mag = strengths[i].norm()
                sigma_sq_weighted_sum += radii[i] ** 2 * str_mag

            # Mean σ² = Σ(σ²|Γ|) / Σ|Γ|
            sigma_sq_mean = sigma_sq_weighted_sum / (total_circ_mag + 1e-16)

            # Total circulation vector Γ_total = ΣΓ
            Gamma_total_x = results[None].gamma_x
            Gamma_total_y = results[None].gamma_y
            Gamma_total_z = results[None].gamma_z

            # Correction term: (2/9) C σ² Γ_total
            correction_factor = ti.cast((2.0 / 9.0) * C * sigma_sq_mean, self.accumulator_dtype)
            results[None].ang_x -= ti.cast(
                correction_factor * Gamma_total_x, self.accumulator_dtype
            )
            results[None].ang_y -= ti.cast(
                correction_factor * Gamma_total_y, self.accumulator_dtype
            )
            results[None].ang_z -= ti.cast(
                correction_factor * Gamma_total_z, self.accumulator_dtype
            )

            # Double loop for pairwise quantities (energy, helicity, enstrophy)
            for i in range(N):
                pos_i = positions[i]
                str_i = strengths[i]
                radii_i = radii[i]
                cutoff_radius = DEFAULT_CUTOFF_RADIUS_FACTOR * radii_i

                local_energy = ti.cast(0.0, self.accumulator_dtype)
                local_helicity = ti.cast(0.0, self.accumulator_dtype)
                local_enstrophy = ti.cast(0.0, self.accumulator_dtype)
                local_dissipation = ti.cast(0.0, self.accumulator_dtype)

                for j in range(N):
                    pos_j = positions[j]
                    str_j = strengths[j]
                    radii_j = radii[j]

                    # Unbounded domain: direct distance
                    r_ij = pos_i - pos_j
                    r_mag = r_ij.norm()

                    if r_mag > EPSILON and r_mag <= cutoff_radius:
                        sigma = 0.5 * (radii_i + radii_j)
                        r_sigma = r_mag / sigma

                        # Enstrophy = ∫|ω|² with ω = Σ Γ ζ_σ.  Two Gaussian blobs
                        # of width σ convolve to width σ√2, so the regularised
                        # enstrophy kernel must be evaluated at σ_ens = σ√2 (not
                        # σ).  Using σ here over-weights it by 2^{3/2} and breaks
                        # the dE/dt = −ν∫|ω|² balance.  Energy/helicity keep σ.
                        sigma_ens = sigma * 1.4142135623730951
                        zeta_val = zeta_(r_mag / sigma_ens) / sigma_ens**3
                        q_val = q_(r_sigma)
                        g_val = g_(r_sigma) / sigma

                        # Accumulate pairwise contributions (explicit cast avoids
                        # implicit f32↔f64 promotion warnings from Taichi JIT)
                        _acc = self.accumulator_dtype
                        local_energy += ti.cast(g_val * str_j.dot(str_i) * 0.5, _acc)
                        local_helicity += ti.cast(
                            q_val * r_ij.dot(str_i.cross(str_j)) / r_mag**3, _acc
                        )
                        pair_enstrophy = ti.cast(zeta_val * str_i.dot(str_j), _acc)
                        pair_nu = ti.cast(0.5 * (viscosities_eff[i] + viscosities_eff[j]), _acc)
                        local_enstrophy += pair_enstrophy
                        local_dissipation -= pair_nu * pair_enstrophy

                # Self-interaction (i == j, r = 0): dominant for the peaked
                # enstrophy kernel; the regularised ∫|ω|² (and −ν∫|ω|²) include
                # each blob's self-overlap, which the pairwise loop skips.  Uses
                # the same σ_ens = σ√2 convolution width as the pairwise term.
                sigma_ens_self = radii_i * 1.4142135623730951
                self_zeta = zeta_(ti.cast(0.0, ti.f32)) / (
                    sigma_ens_self * sigma_ens_self * sigma_ens_self
                )
                self_ens = ti.cast(self_zeta * str_i.dot(str_i), self.accumulator_dtype)
                local_enstrophy += self_ens
                local_dissipation -= ti.cast(viscosities_eff[i], self.accumulator_dtype) * self_ens

                # Atomic accumulation of local sums
                ti.atomic_add(results[None].energy, local_energy)
                ti.atomic_add(results[None].helicity, local_helicity)
                ti.atomic_add(results[None].enstrophy, local_enstrophy)
                ti.atomic_add(results[None].dissipation, local_dissipation)

        # Store kernel as instance method
        self.compute_flow_integrals_kernel = compute_flow_integrals_kernel

    def _define_kinetic_energy_kernel(self, g_):
        """Define and store the kinetic energy kernel using the provided g_sigma function."""

        @ti.kernel
        def compute_particles_kinetic_energy_kernel(
            positions: ti.template(),
            strengths: ti.template(),
            radii: ti.template(),
            kinetic_energy: ti.template(),
        ):  # type: ignore
            """Compute kinetic energy for each particle."""
            N = positions.shape[0]
            for i in range(N):
                energy_sum = ti.cast(0.0, ti.f32)
                str_i = strengths[i]
                pos_i = positions[i]

                for j in range(N):
                    pos_j = positions[j]
                    str_j = strengths[j]
                    radii_j = radii[j]
                    sigma = 0.5 * (radii[i] + radii_j)

                    r_ij = pos_i - pos_j
                    r_mag = ti.sqrt(r_ij.dot(r_ij))
                    cutoff_radius = DEFAULT_CUTOFF_RADIUS_FACTOR * radii_j

                    if r_mag > EPSILON and r_mag <= cutoff_radius:
                        r_sigma = r_mag / sigma
                        g_val = g_(r_sigma) / sigma
                        energy_sum += g_val * str_j.dot(str_i) * 0.5

                kinetic_energy[i] = energy_sum

        self.compute_particles_kinetic_energy_kernel = compute_particles_kinetic_energy_kernel

    def _define_helicity_kernel(self, q_):
        """Define and store the helicity kernel using the provided q_sigma function."""

        @ti.kernel
        def compute_particles_helicity_kernel(
            positions: ti.template(),
            strengths: ti.template(),
            radii: ti.template(),
            helicity: ti.template(),
        ):  # type: ignore
            """Compute helicity for each particle."""
            N = positions.shape[0]
            for i in range(N):
                hel = ti.cast(0.0, ti.f32)
                str_i = strengths[i]
                pos_i = positions[i]
                radii_i = radii[i]
                cutoff_radius = DEFAULT_CUTOFF_RADIUS_FACTOR * radii_i

                for j in range(N):
                    pos_j = positions[j]
                    str_j = strengths[j]
                    sigma = 0.5 * (radii_i + radii[j])

                    r_ij = pos_i - pos_j
                    r_mag = ti.sqrt(r_ij.dot(r_ij))

                    if r_mag > EPSILON and r_mag <= cutoff_radius:
                        r_sigma = r_mag / sigma
                        q_val = q_(r_sigma)
                        hel += q_val * r_ij.dot(str_i.cross(str_j)) / (r_mag**3)

                helicity[i] = hel

        self.compute_particles_helicity_kernel = compute_particles_helicity_kernel

    def _define_enstrophy_kernel(self, zeta_):
        """Define and store the enstrophy kernel using the provided zeta_sigma function."""

        @ti.kernel
        def compute_particles_enstrophy_kernel(
            positions: ti.template(),
            strengths: ti.template(),
            radii: ti.template(),
            enstrophy: ti.template(),
        ):  # type: ignore
            """Compute enstrophy for each particle."""
            N = positions.shape[0]
            for i in range(N):
                enstrophy_local = ti.cast(0.0, ti.f32)
                str_i = strengths[i]
                pos_i = positions[i]
                radii_i = radii[i]
                cutoff_radius = DEFAULT_CUTOFF_RADIUS_FACTOR * radii_i

                for j in range(N):
                    r_ij = pos_i - positions[j]
                    r_mag = ti.sqrt(r_ij.dot(r_ij))

                    if r_mag > EPSILON and r_mag <= cutoff_radius:
                        str_j = strengths[j]
                        sigma = 0.5 * (radii_i + radii[j])
                        r_sigma = r_mag / sigma
                        zeta_val = zeta_(r_sigma) / sigma**3
                        enstrophy_local += zeta_val * str_i.dot(str_j)

                enstrophy[i] = enstrophy_local

        self.compute_particles_enstrophy_kernel = compute_particles_enstrophy_kernel

    def _define_vorticity_reconstruction_kernel(self, zeta_):
        """Define the ω_h reconstruction kernel: ω_h(xᵢ) = Σⱼ ζ_σ(rᵢⱼ/σ)/σ³ · Γⱼ."""

        @ti.kernel
        def reconstruct_vorticity_kernel(
            positions: ti.template(),
            strengths: ti.template(),
            radii: ti.template(),
            out: ti.template(),
            num_particles: ti.i32,
        ):  # type: ignore
            for i in range(num_particles):
                acc = ti.Vector([0.0, 0.0, 0.0])
                pos_i = positions[i]
                radii_i = radii[i]
                cutoff_radius = DEFAULT_CUTOFF_RADIUS_FACTOR * radii_i
                for j in range(num_particles):
                    r_ij = pos_i - positions[j]
                    r_mag = r_ij.norm()
                    if r_mag <= cutoff_radius:
                        sigma = 0.5 * (radii_i + radii[j])
                        r_sigma = r_mag / sigma
                        zeta_val = zeta_(r_sigma) / sigma**3
                        acc += zeta_val * strengths[j]
                out[i] = acc

        self.reconstruct_vorticity_kernel = reconstruct_vorticity_kernel

    def _define_per_particle_kernels(self, kernel_functions):
        """Define kernels for computing per-particle diagnostics."""
        self._define_kinetic_energy_kernel(kernel_functions["g_sigma"])
        self._define_helicity_kernel(kernel_functions["q_sigma"])
        self._define_enstrophy_kernel(kernel_functions["zeta_sigma"])
        self._define_vorticity_reconstruction_kernel(kernel_functions["zeta_sigma"])

    def _define_group_centroid_kernel(self):
        """Define and store the group-centroid kernel."""

        @ti.kernel
        def compute_group_centroid_kernel(
            positions: ti.template(),
            strengths: ti.template(),
            group_ids: ti.template(),
            target_group: ti.i32,
            result: ti.template(),
        ):  # type: ignore
            """
            Compute centroid of circulation for a specific group.

            Centroid = Σ(r × Γ) / Σ|Γ| for particles in the group
            """
            weighted_pos = ti.Vector([0.0, 0.0, 0.0])
            total_strength_mag = 0.0

            N = positions.shape[0]
            for i in range(N):
                if group_ids[i] == target_group:
                    str_mag = strengths[i].norm()
                    if str_mag > EPSILON:
                        ti.atomic_add(weighted_pos[0], positions[i][0] * str_mag)
                        ti.atomic_add(weighted_pos[1], positions[i][1] * str_mag)
                        ti.atomic_add(weighted_pos[2], positions[i][2] * str_mag)
                        ti.atomic_add(total_strength_mag, str_mag)

            if total_strength_mag > EPSILON:
                result[0] = weighted_pos[0] / total_strength_mag
                result[1] = weighted_pos[1] / total_strength_mag
                result[2] = weighted_pos[2] / total_strength_mag
            else:
                result[0] = 0.0
                result[1] = 0.0
                result[2] = 0.0

        self.compute_group_centroid_kernel = compute_group_centroid_kernel

    def _define_global_centroid_kernel(self):
        """Define and store the global-centroid kernel."""

        @ti.kernel
        def compute_global_centroid_kernel(
            positions: ti.template(), strengths: ti.template(), result: ti.template()
        ):  # type: ignore
            """
            Compute centroid of circulation for the entire particle set.

            Centroid = Σ(r * |Γ|) / Σ|Γ| over all particles
            """
            weighted_pos = ti.Vector([0.0, 0.0, 0.0])
            total_strength_mag = 0.0

            N = positions.shape[0]
            for i in range(N):
                str_mag = strengths[i].norm()
                if str_mag > EPSILON:
                    ti.atomic_add(weighted_pos[0], positions[i][0] * str_mag)
                    ti.atomic_add(weighted_pos[1], positions[i][1] * str_mag)
                    ti.atomic_add(weighted_pos[2], positions[i][2] * str_mag)
                    ti.atomic_add(total_strength_mag, str_mag)

            if total_strength_mag > EPSILON:
                result[0] = weighted_pos[0] / total_strength_mag
                result[1] = weighted_pos[1] / total_strength_mag
                result[2] = weighted_pos[2] / total_strength_mag
            else:
                result[0] = 0.0
                result[1] = 0.0
                result[2] = 0.0

        self.compute_global_centroid_kernel = compute_global_centroid_kernel

    def _define_centroid_kernels(self, kernel_functions):
        """Define kernels for computing centroids of circulation."""
        self._define_group_centroid_kernel()
        self._define_global_centroid_kernel()

    # PUBLIC API METHODS

    @staticmethod
    def record_centroid_history(
        diagnostics_history: dict,
        positions: np.ndarray,
        circulation: np.ndarray,
    ) -> None:
        """Compute and append the circulation-weighted centroid to the diagnostics history.

        Args:
            diagnostics_history: Solver's ``_diagnostics_history`` dict (mutated in-place).
            positions: Particle positions array of shape (N, 3).
            circulation: Particle circulation array of shape (N, 3).
        """
        if "centroid" not in diagnostics_history:
            return
        try:
            if positions.size == 0 or circulation.size == 0:
                centroid = np.array([0.0, 0.0, 0.0])
            else:
                circ_mag = np.linalg.norm(circulation, axis=1)
                total_mag = circ_mag.sum()
                if total_mag > 0:
                    centroid = (positions * circ_mag[:, np.newaxis]).sum(axis=0) / total_mag
                else:
                    centroid = np.array([0.0, 0.0, 0.0])
            diagnostics_history["centroid"].append(tuple(centroid.tolist()))
        except Exception as exc:
            print(f"(Warning) Failed to compute circulation centroid: {exc}")

    def compute_centroid_of_circulation(self, particles) -> np.ndarray:
        """
        Compute the centroid of circulation for the entire particle set.

        Returns:
            np.ndarray: Centroid position [x, y, z], or zeros if no particles.
        """
        N = len(particles)
        if N == 0:
            return np.array([0.0, 0.0, 0.0])

        self._resize_fields(N)

        # Use cached result field (avoids memory leak from repeated Taichi allocations)
        # CRITICAL: Taichi fields cannot be garbage collected
        self._centroid_result.fill(0)

        # Call kernel
        self.compute_global_centroid_kernel(
            particles.position, particles.circulation, self._centroid_result
        )

        # Extract result
        centroid = self._centroid_result.to_numpy()
        return centroid

    def compute_flow_integrals(self, particles, flow_time: float, record_history: bool = True):
        """
        Compute all flow integral quantities in a single efficient GPU kernel call.

        This method computes and stores time history for energy dissipation rate calculation.

        Args:
            particles: Particles object containing positions, strengths, radii, viscosities
            flow_time: Current simulation time [s]
            record_history: Whether to append this sample to the kinetic-energy
                history used for finite-difference dE/dt diagnostics.

        Returns:
            dict: Dictionary containing all flow integral quantities:
                - 'kinetic_energy': Total kinetic energy [J]
                - 'helicity': Total helicity [m³/s²]
                - 'enstrophy': Total enstrophy [1/s²]
                - 'vorticity_dissipation_rate': Vorticity dissipation rate [1/s³]
                - 'kinetic_energy_dissipation_rate': Energy dissipation rate [J/s]
                - 'strength_magnitude': Total strength magnitude [1/s]
                - 'strength': Total strength vector [1/s]
                - 'linear_impulse': Linear impulse vector [m³/s]
                - 'angular_impulse': Angular impulse vector [m⁴/s]
        """
        N = len(particles)
        if N == 0:
            # Return zero values for empty particle system
            return self._get_zero_results()

        self._resize_fields(N)

        # Initialize results struct to zero before kernel call
        self.total_quantities_results.fill(0)

        # Call the combined kernel
        self.compute_flow_integrals_kernel(
            particles.position,
            particles.circulation,
            particles.radius,
            particles.viscosity_effective,
            self.total_quantities_results,
            N,
        )

        # Extract results from Taichi struct field
        r = self.total_quantities_results[None]
        kinetic_energy = float(r.energy)
        if record_history:
            self._update_energy_history(flow_time, kinetic_energy)

        # Compute kinetic energy dissipation rate using finite differences
        dE_dt = self._compute_energy_dissipation_rate()

        return {
            "kinetic_energy": kinetic_energy,
            "helicity": float(r.helicity),
            "enstrophy": float(r.enstrophy),
            "vorticity_dissipation_rate": float(r.dissipation),
            "kinetic_energy_dissipation_rate": dE_dt,
            "strength_magnitude": float(r.str_mag),
            "strength": np.array([float(r.gamma_x), float(r.gamma_y), float(r.gamma_z)]),
            "linear_impulse": np.array([float(r.imp_x), float(r.imp_y), float(r.imp_z)]),
            "angular_impulse": np.array([float(r.ang_x), float(r.ang_y), float(r.ang_z)]),
        }

    def compute_particles_kinetic_energy(self, particles) -> np.ndarray:
        """
        Compute kinetic energy for each particle.

        Args:
            particles: Particles object

        Returns:
            np.ndarray: Array of kinetic energy values [J] for each particle
        """
        N = len(particles)
        if N == 0:
            return np.array([])

        self._resize_fields(N)

        # Initialize field to zero before kernel call
        self.particle_kinetic_energy.fill(0.0)

        self.compute_particles_kinetic_energy_kernel(
            particles.position,
            particles.circulation,
            particles.radius,
            self.particle_kinetic_energy,
        )

        return self.particle_kinetic_energy.to_numpy()[:N]

    def compute_particles_helicity(self, particles) -> np.ndarray:
        """
        Compute helicity for each particle.

        Args:
            particles: Particles object

        Returns:
            np.ndarray: Array of helicity values [m/s²] for each particle
        """
        N = len(particles)
        if N == 0:
            return np.array([])

        self._resize_fields(N)

        # Initialize field to zero before kernel call
        self.particle_helicity.fill(0.0)

        self.compute_particles_helicity_kernel(
            particles.position, particles.circulation, particles.radius, self.particle_helicity
        )

        return self.particle_helicity.to_numpy()[:N]

    def compute_particles_enstrophy(self, particles) -> np.ndarray:
        """
        Compute enstrophy for each particle.

        Args:
            particles: Particles object

        Returns:
            np.ndarray: Array of enstrophy values [1/s²] for each particle
        """
        N = len(particles)
        if N == 0:
            return np.array([])

        self._resize_fields(N)

        # Initialize field to zero before kernel call
        self.particle_enstrophy.fill(0.0)

        self.compute_particles_enstrophy_kernel(
            particles.position, particles.circulation, particles.radius, self.particle_enstrophy
        )

        return self.particle_enstrophy.to_numpy()[:N]

    def reconstruct_vorticity(self, particles, out_field) -> None:
        """Write the kernel-reconstructed vorticity ω_h into a Taichi Vector field.

        Args:
            particles: Particles object with position, circulation, radius.
            out_field: ti.Vector.field(3, ...) to receive ω_h values (written in-place).
        """
        N = particles.number_of_particles
        if N == 0:
            return
        self.reconstruct_vorticity_kernel(
            particles.position, particles.circulation, particles.radius, out_field, N
        )

    def compute_centroids_of_circulation(self, particles) -> dict[int, np.ndarray]:
        """
        Compute centroids of circulation for each particle group.

        Args:
            particles: Particles object

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping group_id to centroid position [x, y, z]
        """
        N = len(particles)
        if N == 0:
            return {}

        # Check if particles have group_id (field name is singular, not "group_ids")
        if not hasattr(particles, "group_id"):
            return {}

        self._resize_fields(N)

        # Get unique group IDs (accessor is group_id_cpu(), not group_ids_cpu())
        group_ids_np = particles.group_id_cpu()
        unique_groups = np.unique(group_ids_np)

        centroids = {}
        for group_id in unique_groups:
            # Use cached result field (avoids memory leak from repeated Taichi allocations)
            # CRITICAL: Taichi fields cannot be garbage collected
            self._centroid_result.fill(0)

            # Compute centroid
            self.compute_group_centroid_kernel(
                particles.position,
                particles.circulation,
                particles.group_id,
                int(group_id),
                self._centroid_result,
            )

            # Extract result (copy before reusing field)
            centroids[int(group_id)] = self._centroid_result.to_numpy().copy()

        return centroids

    # ENERGY DISSIPATION RATE COMPUTATION

    def _update_energy_history(self, flow_time: float, kinetic_energy: float):
        """
        Update the time history of kinetic energy.

        Args:
            flow_time: Current simulation time [s]
            kinetic_energy: Current total kinetic energy [J]
        """
        # Replace the latest entry when callers request diagnostics multiple
        # times at the same physical time. This keeps dE/dt finite differences
        # well-posed and avoids zero-dt history pairs.
        if self._flow_time_history and abs(self._flow_time_history[-1][0] - flow_time) < 1e-12:
            self._flow_time_history[-1] = (flow_time, kinetic_energy)
            return

        # Add new entry
        self._flow_time_history.append((flow_time, kinetic_energy))

        # Keep only the last N entries
        if len(self._flow_time_history) > self._max_history_length:
            self._flow_time_history.pop(0)

    def _compute_energy_dissipation_rate(self) -> float:
        """
        Compute dE/dt by fitting a degree-3 polynomial through the energy history
        and evaluating its analytic derivative at the most recent time.

        This approach is correct for non-uniform time steps (uses the actual times,
        not just the last interval) and provides implicit least-squares smoothing
        when the history window is larger than the polynomial degree (n > 4).
        With the default history of 7 points the fit has 3 spare degrees of freedom
        that attenuate high-frequency noise in E(t) without biasing the trend.

        Falls back gracefully:
          n >= 4  →  degree-3 polynomial fit  (smoothed for n > 4)
          n == 3  →  degree-2 polynomial fit
          n == 2  →  degree-1 (linear) fit  =  simple forward difference

        Returns:
            float: Energy dissipation rate [J/s], or 0.0 if insufficient data
        """
        n = len(self._flow_time_history)

        if n < 2:
            return 0.0

        times = np.array([t for t, _ in self._flow_time_history])
        energies = np.array([E for _, E in self._flow_time_history])

        # Guard against degenerate (zero-range) time windows
        t_span = times[-1] - times[0]
        if t_span <= 0.0:
            return 0.0

        # Degree capped so the fit is never under-determined
        deg = min(3, n - 1)

        # np.polyfit uses the actual time values, so non-uniform spacing is handled
        # exactly.  When n > deg+1 the extra points act as regularisation.
        coeffs = np.polyfit(times, energies, deg)
        dcoeffs = np.polyder(coeffs)  # analytic derivative of the fitted poly
        return float(np.polyval(dcoeffs, times[-1]))

    def _get_zero_results(self) -> dict:
        """Return dictionary of zero values for empty particle system."""
        return {
            "kinetic_energy": 0.0,
            "helicity": 0.0,
            "enstrophy": 0.0,
            "vorticity_dissipation_rate": 0.0,
            "kinetic_energy_dissipation_rate": 0.0,
            "strength_magnitude": 0.0,
            "strength": np.array([0.0, 0.0, 0.0]),
            "linear_impulse": np.array([0.0, 0.0, 0.0]),
            "angular_impulse": np.array([0.0, 0.0, 0.0]),
        }

    def reset_energy_history(self):
        """Reset the energy time history (useful when restarting simulation)."""
        self._flow_time_history.clear()

# =========================================================
# PUBLIC API EXPORTS
# =========================================================

__all__ = ["ParticleFieldEvaluation"]
