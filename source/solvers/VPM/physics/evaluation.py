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

_HOST_TRANSFER_CHUNK_SIZE = 65536
_DIRECT_INTEGRAL_LIMIT = 50_000


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
        particle_kernel: str = "GAUSSIAN",
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
        self.particle_kernel = particle_kernel.upper()
        self.accumulator_dtype = accumulator_dtype

        # Initialize GPU fields for storing results
        self._initialize_result_fields()
        self._host_scalar_chunks = {}

        # Initialize time tracking for energy dissipation rate
        self._flow_time_history = []  # Store (time, energy) pairs
        # Retain a short audit trail.  dE/dt uses only the latest interval so
        # its sign is consistent with the two energy samples being reported.
        self._max_history_length = 7

        # Define Taichi kernels
        self._define_taichi_kernels()

    @ti.kernel
    def _extract_scalar_field_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        for i in range(n):
            dst[i] = src[start_idx + i]

    def _download_scalar_field(self, src, n: int) -> np.ndarray:
        if n == 0:
            return np.empty((0,), dtype=np.float32)
        key = id(src)
        if key not in self._host_scalar_chunks:
            self._host_scalar_chunks[key] = np.empty((_HOST_TRANSFER_CHUNK_SIZE,), dtype=np.float32)
        buf = self._host_scalar_chunks[key]
        out = np.empty((n,), dtype=np.float32)
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_scalar_field_prefix(src, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _initialize_result_fields(self):
        """Initialize Taichi fields for storing diagnostic results on GPU."""
        # Results struct for flow integrals kernel — named fields eliminate
        # the 'Field index not int32' warning from integer literal indexing.
        _F = self.accumulator_dtype
        FlowIntegralsStruct = ti.types.struct(
            energy=_F,
            helicity=_F,
            enstrophy=_F,
            enstrophy_test=_F,
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
        self._centroid_result = ti.field(dtype=self.accumulator_dtype, shape=3)

    def _resize_fields(self, required_size: int):
        """Validate that diagnostics fit the startup particle allocation."""
        if required_size <= 0:
            return
        if required_size > self.max_particles:
            raise ValueError(
                f"Diagnostics require {required_size} particles, but max_particles="
                f"{self.max_particles}. Increase VPMSetup.max_particles before "
                "constructing the solver."
            )

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
        if self.particle_kernel == "GAUSSIAN":
            from ..kernels.gaussian import create_gaussian_kernels

            kernel_dict = create_gaussian_kernels(self.accumulator_dtype)
        elif self.particle_kernel == "HIGH_ORDER_GAUSSIAN":
            from ..kernels.high_order_gaussian import create_high_order_gaussian_kernels

            kernel_dict = create_high_order_gaussian_kernels(self.accumulator_dtype)
        elif self.particle_kernel == "SUPER_GAUSSIAN":
            from ..kernels.super_gaussian import create_super_gaussian_kernels

            kernel_dict = create_super_gaussian_kernels(self.accumulator_dtype)
        elif self.particle_kernel == "WINCKELMANS":
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
            volumes: ti.template(),
            viscosities_eff: ti.template(),
            results: ti.template(),
            num_particles: ti.i32,
        ):  # type: ignore
            """
            Compute all flow integrals in a single optimized GPU kernel.

            This combines energy, helicity, enstrophy, dissipation rates, strength,
            and impulses into one efficient computation using unbounded definitions.

            Angular impulse includes the per-particle kernel correction:

                A = (1/3) Σ r_i × (r_i × Γ_i)
                    - (2/9) C Σ σ_i² Γ_i

            The correction must remain inside the sum when core radii vary.
            Replacing it by a mean radius times ``ΣΓ`` is only equivalent for
            uniform cores and gives a false angular-impulse drift as soon as
            core spreading changes individual radii.
            """
            N = num_particles

            # Initialize every field to zero explicitly
            results[None].energy = ti.cast(0.0, self.accumulator_dtype)
            results[None].helicity = ti.cast(0.0, self.accumulator_dtype)
            results[None].enstrophy = ti.cast(0.0, self.accumulator_dtype)
            results[None].enstrophy_test = ti.cast(0.0, self.accumulator_dtype)
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
                correction_factor = ti.cast(
                    (2.0 / 9.0) * angular_correction_func() * radii[i] ** 2,
                    self.accumulator_dtype,
                )
                corrected = angular_contrib * (1.0 / 3.0) - correction_factor * str_i
                ti.atomic_add(results[None].ang_x, corrected[ti.static(0)])
                ti.atomic_add(results[None].ang_y, corrected[ti.static(1)])
                ti.atomic_add(results[None].ang_z, corrected[ti.static(2)])

            # Double loop for pairwise quantities (energy, helicity, enstrophy)
            for i in range(N):
                pos_i = positions[i]
                str_i = strengths[i]
                radii_i = radii[i]

                local_energy = ti.cast(0.0, self.accumulator_dtype)
                local_helicity = ti.cast(0.0, self.accumulator_dtype)
                local_enstrophy = ti.cast(0.0, self.accumulator_dtype)
                local_enstrophy_test = ti.cast(0.0, self.accumulator_dtype)
                local_dissipation = ti.cast(0.0, self.accumulator_dtype)

                for j in range(N):
                    pos_j = positions[j]
                    str_j = strengths[j]
                    radii_j = radii[j]

                    # Unbounded domain: direct distance
                    r_ij = pos_i - pos_j
                    r_mag = r_ij.norm()
                    sigma = 0.5 * (radii_i + radii_j)
                    r_sigma = r_mag / sigma

                    if r_sigma <= DEFAULT_CUTOFF_RADIUS_FACTOR:
                        # Every quadratic field integral of the blob field is a
                        # *convolution* of two blobs, so its regularised kernel is
                        # evaluated at the convolved width
                        #
                        #     σ_e = sqrt(σ_i² + σ_j²)
                        #
                        # not at the pair mean σ = (σ_i+σ_j)/2.  This applies to
                        # E = ½∫ω·ψ, to H = ∫ω·u and to ∫|ω|² alike: each is
                        # ΣΣ Γ_i·Γ_j (k_σi * k_σj)(r_ij).  Using σ inflates the
                        # near field (+41 % on the single-blob self energy) and
                        # breaks the exact dE/dt = −ν∫|ω|² balance under core
                        # spreading.  With σ_e that identity closes to machine
                        # precision, including for unequal cores and per-particle
                        # ν_eff — which is what makes the LES energy budget
                        # auditable at all.
                        #
                        # ζ_σ * ζ_σ = ζ_{σ√2} is a *Gaussian* identity, so σ_e is
                        # exact for GAUSSIAN and an approximation for the
                        # algebraic kernels, whose double-mollified Green's
                        # function is not in their own family.  The pre-existing
                        # enstrophy kernel already assumed it for every kernel;
                        # this extends the same assumption to E and H so the
                        # three stay mutually consistent.  Deriving the true
                        # factor for WINCKELMANS et al. is open work.
                        sigma_e = ti.sqrt(radii_i * radii_i + radii_j * radii_j)
                        r_sigma_e = r_mag / sigma_e
                        zeta_val = zeta_(r_sigma_e) / sigma_e**3
                        g_val = g_(r_sigma_e) / sigma_e

                        # Accumulate pairwise contributions (explicit cast avoids
                        # implicit f32↔f64 promotion warnings from Taichi JIT)
                        _acc = self.accumulator_dtype
                        local_energy += ti.cast(g_val * str_j.dot(str_i) * 0.5, _acc)
                        if r_mag > EPSILON:
                            q_val = q_(r_sigma_e)
                            local_helicity += ti.cast(
                                q_val * r_ij.dot(str_i.cross(str_j)) / r_mag**3,
                                _acc,
                            )
                        # Test-filtered enstrophy: the same quadratic form at a
                        # widened width sqrt(sigma_i^2 + sigma_j^2 + Delta_test^2).
                        # Delta_test = 2 V^(1/3) matches the LES filter width in
                        # smagorinsky.py and remains a useful resolution diagnostic.
                        d_test = 2.0 * (volumes[i] ** (1.0 / 3.0) + volumes[j] ** (1.0 / 3.0)) * 0.5
                        sigma_t = ti.sqrt(sigma_e * sigma_e + d_test * d_test)
                        zeta_test = zeta_(r_mag / sigma_t) / sigma_t**3
                        pair_enstrophy = ti.cast(zeta_val * str_i.dot(str_j), _acc)
                        local_enstrophy_test += ti.cast(zeta_test * str_i.dot(str_j), _acc)
                        pair_nu = ti.cast(0.5 * (viscosities_eff[i] + viscosities_eff[j]), _acc)
                        local_enstrophy += pair_enstrophy
                        local_dissipation -= pair_nu * pair_enstrophy

                # Atomic accumulation of local sums
                ti.atomic_add(results[None].energy, local_energy)
                ti.atomic_add(results[None].helicity, local_helicity)
                ti.atomic_add(results[None].enstrophy, local_enstrophy)
                ti.atomic_add(results[None].enstrophy_test, local_enstrophy_test)
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
                    r_sigma = r_mag / sigma

                    if r_sigma <= DEFAULT_CUTOFF_RADIUS_FACTOR:
                        # Convolved pair width — see compute_flow_integrals_kernel.
                        sigma_e = ti.sqrt(radii[i] * radii[i] + radii_j * radii_j)
                        g_val = g_(r_mag / sigma_e) / sigma_e
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

                    r_ij = pos_i - pos_j
                    r_mag = ti.sqrt(r_ij.dot(r_ij))

                    if r_mag > EPSILON and r_mag <= cutoff_radius:
                        # Convolved pair width — see compute_flow_integrals_kernel.
                        sigma_e = ti.sqrt(radii_i * radii_i + radii[j] * radii[j])
                        q_val = q_(r_mag / sigma_e)
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

                for j in range(N):
                    r_ij = pos_i - positions[j]
                    r_mag = ti.sqrt(r_ij.dot(r_ij))
                    sigma = 0.5 * (radii_i + radii[j])
                    r_sigma = r_mag / sigma

                    if r_sigma <= DEFAULT_CUTOFF_RADIUS_FACTOR:
                        str_j = strengths[j]
                        # Convolved pair width — see compute_flow_integrals_kernel.
                        sigma_e = ti.sqrt(radii_i * radii_i + radii[j] * radii[j])
                        zeta_val = zeta_(r_mag / sigma_e) / sigma_e**3
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
                for j in range(num_particles):
                    r_ij = pos_i - positions[j]
                    r_mag = r_ij.norm()
                    # This is a pointwise field reconstruction, not a
                    # quadratic two-blob integral: source j contributes its
                    # own blob zeta_{sigma_j}(x_i-x_j).  A target/source mean
                    # radius gives the wrong field as soon as core spreading
                    # makes radii nonuniform and makes Gamma--omega alignment
                    # depend on the arbitrary target core.
                    sigma = radii[j]
                    if r_mag <= DEFAULT_CUTOFF_RADIUS_FACTOR * sigma:
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
            num_particles: ti.i32,
            target_group: ti.i32,
            result: ti.template(),
        ):  # type: ignore
            """
            Compute centroid of circulation for a specific group.

            Centroid = Σ(r × Γ) / Σ|Γ| for particles in the group
            """
            weighted_pos = ti.Vector([0.0, 0.0, 0.0])
            total_strength_mag = 0.0

            for i in range(num_particles):
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
            positions: ti.template(),
            strengths: ti.template(),
            num_particles: ti.i32,
            result: ti.template(),
        ):  # type: ignore
            """
            Compute centroid of circulation for the entire particle set.

            Centroid = Σ(r * |Γ|) / Σ|Γ| over all particles
            """
            weighted_pos = ti.Vector([0.0, 0.0, 0.0])
            total_strength_mag = 0.0

            for i in range(num_particles):
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
            particles.position,
            particles.vortex_strength,
            N,
            self._centroid_result,
        )

        # Extract result
        centroid = self._centroid_result.to_numpy()
        return centroid

    def compute_flow_integrals(self, particles, time: float, record_history: bool = True):
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
                - 'vortex_strength_magnitude': Total strength magnitude [1/s]
                - 'vortex_strength': Total strength vector [1/s]
                - 'linear_impulse': Linear impulse vector [m³/s]
                - 'angular_impulse': Angular impulse vector [m⁴/s]
        """
        N = len(particles)
        if N == 0:
            # Return zero values for empty particle system
            return self._get_zero_results()
        if N > _DIRECT_INTEGRAL_LIMIT and self.particle_kernel == "GAUSSIAN":
            return self._compute_fourier_flow_integrals(particles, time, record_history)

        self._resize_fields(N)

        # Initialize results struct to zero before kernel call
        self.total_quantities_results.fill(0)

        # Call the combined kernel
        self.compute_flow_integrals_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.volume,
            particles.effective_viscosity,
            self.total_quantities_results,
            N,
        )

        # Extract results from Taichi struct field
        r = self.total_quantities_results[None]
        kinetic_energy = float(r.energy)
        if record_history:
            self._update_energy_history(time, kinetic_energy)

        # Compute kinetic energy dissipation rate using finite differences
        dE_dt = self._compute_energy_dissipation_rate()

        return {
            "kinetic_energy": kinetic_energy,
            "helicity": float(r.helicity),
            "enstrophy": float(r.enstrophy),
            "enstrophy_test": float(r.enstrophy_test),
            "vorticity_dissipation_rate": float(r.dissipation),
            "kinetic_energy_dissipation_rate": dE_dt,
            "vortex_strength_magnitude": float(r.str_mag),
            "vortex_strength": np.array([float(r.gamma_x), float(r.gamma_y), float(r.gamma_z)]),
            "linear_impulse": np.array([float(r.imp_x), float(r.imp_y), float(r.imp_z)]),
            "angular_impulse": np.array([float(r.ang_x), float(r.ang_y), float(r.ang_z)]),
        }

    def _compute_fourier_flow_integrals(
        self,
        particles,
        time: float,
        record_history: bool,
    ) -> dict:
        from ..numerics.fourier_integrals import gaussian_fourier_integrals

        position = particles.position_cpu().astype(np.float64)
        circulation = particles.vortex_strength_cpu().astype(np.float64)
        radius = particles.core_radius_cpu().astype(np.float64)
        volume = particles.volume_cpu().astype(np.float64)
        viscosity = particles.effective_viscosity_cpu().astype(np.float64)
        spectral = gaussian_fourier_integrals(
            position,
            circulation,
            radius,
            volume,
            viscosity=viscosity,
        )
        if spectral.viscous_energy_dissipation is None:
            raise RuntimeError("Fourier flow diagnostics did not compute viscous dissipation")

        energy = spectral.energy
        if record_history:
            self._update_energy_history(time, energy)
        total = circulation.sum(axis=0, dtype=np.float64)
        impulse = 0.5 * np.cross(position, circulation).sum(axis=0, dtype=np.float64)
        angular = np.cross(position, np.cross(position, circulation)).sum(
            axis=0, dtype=np.float64
        ) / 3.0 - (1.0 / 3.0) * (radius[:, None] ** 2 * circulation).sum(axis=0)
        return {
            "kinetic_energy": energy,
            "helicity": spectral.helicity,
            "enstrophy": spectral.enstrophy,
            "enstrophy_test": spectral.enstrophy_test,
            "vorticity_dissipation_rate": spectral.viscous_energy_dissipation,
            "kinetic_energy_dissipation_rate": self._compute_energy_dissipation_rate(),
            "vortex_strength_magnitude": float(np.linalg.norm(circulation, axis=1).sum()),
            "vortex_strength": total,
            "linear_impulse": impulse,
            "angular_impulse": angular,
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
            particles.vortex_strength,
            particles.core_radius,
            self.particle_kinetic_energy,
        )

        return self._download_scalar_field(self.particle_kinetic_energy, N)

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
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            self.particle_helicity,
        )

        return self._download_scalar_field(self.particle_helicity, N)

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
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            self.particle_enstrophy,
        )

        return self._download_scalar_field(self.particle_enstrophy, N)

    def reconstruct_vorticity(self, particles, out_field) -> None:
        """Write the kernel-reconstructed vorticity ω_h into a Taichi Vector field.

        Args:
            particles: Particles object with position, circulation, radius.
            out_field: ti.Vector.field(3, ...) to receive ω_h values (written in-place).
        """
        N = particles.n_particles
        if N == 0:
            return
        self.reconstruct_vorticity_kernel(
            particles.position, particles.vortex_strength, particles.core_radius, out_field, N
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
                particles.vortex_strength,
                particles.group_id,
                N,
                int(group_id),
                self._centroid_result,
            )

            # Extract result (copy before reusing field)
            centroids[int(group_id)] = self._centroid_result.to_numpy().copy()

        return centroids

    # ENERGY DISSIPATION RATE COMPUTATION

    def _update_energy_history(self, time: float, kinetic_energy: float):
        """
        Update the time history of kinetic energy.

        Args:
            flow_time: Current simulation time [s]
            kinetic_energy: Current total kinetic energy [J]
        """
        # Replace the latest entry when callers request diagnostics multiple
        # times at the same physical time. This keeps dE/dt finite differences
        # well-posed and avoids zero-dt history pairs.
        if self._flow_time_history and abs(self._flow_time_history[-1][0] - time) < 1e-12:
            self._flow_time_history[-1] = (time, kinetic_energy)
            return

        # Add new entry
        self._flow_time_history.append((time, kinetic_energy))

        # Keep only the last N entries
        if len(self._flow_time_history) > self._max_history_length:
            self._flow_time_history.pop(0)

    def _compute_energy_dissipation_rate(self) -> float:
        """Return the energy change over the latest diagnostic interval.

        An endpoint derivative extrapolated from a higher-order polynomial can
        have the wrong sign even when every sampled energy decreases.  The
        backward secant is conservative, handles non-uniform output intervals,
        and makes the reported rate exactly consistent with the latest pair of
        diagnostic states.
        """
        if len(self._flow_time_history) < 2:
            return 0.0

        previous_time, previous_energy = self._flow_time_history[-2]
        current_time, current_energy = self._flow_time_history[-1]
        interval = current_time - previous_time
        if interval <= 0.0:
            return 0.0
        return float((current_energy - previous_energy) / interval)

    def _get_zero_results(self) -> dict:
        """Return dictionary of zero values for empty particle system."""
        return {
            "kinetic_energy": 0.0,
            "helicity": 0.0,
            "enstrophy": 0.0,
            "enstrophy_test": 0.0,
            "vorticity_dissipation_rate": 0.0,
            "kinetic_energy_dissipation_rate": 0.0,
            "vortex_strength_magnitude": 0.0,
            "vortex_strength": np.array([0.0, 0.0, 0.0]),
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
