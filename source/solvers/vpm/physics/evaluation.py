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
    MAX_N_PARTICLES,
)
from .events import NullPhysicsEventObserver, PhysicsEventObserver

_HOST_TRANSFER_CHUNK_SIZE = 65536
_DIRECT_INTEGRAL_PARTICLE_LIMIT = 10_000
_FOURIER_WARMUP_PARTICLE_LIMIT = 7_500
_FOURIER_GRID_EXTRA_CELLS = 8
_FOURIER_GRID_MIN_SLACK = 4


@ti.data_oriented
class ParticleFieldEvaluation:
    """
    Manages all flow integral diagnostics for particle fields.

    This class provides GPU-accelerated computation of global flow quantities:
    - Energy metrics: kinetic energy and dissipation rates
    - Vorticity metrics: helicity, enstrophy, dissipation
    - Conservation quantities: total strength, impulses
    - Group analysis: vortex_centroids of vortex strength

    All computations use unbounded domain definitions and are optimized for GPU execution.
    Results are kept on GPU and only transferred to CPU when accessed.
    """

    def __init__(
        self,
        particle_kernel: str = "GAUSSIAN",
        max_n_particles: int = MAX_N_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
        event_observer: PhysicsEventObserver | None = None,
    ):
        """
        Initialize particle field physics diagnostics.

        Args:
            particle_kernel: Kernel type ("GAUSSIAN", "WINCKELMANS", "SUPER_GAUSSIAN")
            max_n_particles: Maximum number of particles for field allocation
            accumulator_dtype: Taichi dtype for accumulation (ti.f32 or ti.f64)
        """
        self.max_n_particles = max_n_particles
        self.particle_kernel = particle_kernel.upper()
        self.accumulator_dtype = accumulator_dtype
        self._event_observer = event_observer or NullPhysicsEventObserver()

        # Initialize GPU fields for storing results
        self._initialize_result_fields()
        self._host_scalar_chunks = {}

        # Initialize time tracking for energy dissipation rate
        self._energy_history = []  # Store (time, energy, measurement) triples
        # Retain a short audit trail.  dE/dt uses only the latest interval so
        # its sign is consistent with the two energy samples being reported.
        self._max_history_length = 7
        # Large Gaussian clouds use a Fourier diagnostic.  Keep its lattice
        # spacing and particle-relative phase fixed between samples: rebuilding
        # a tight grid at every output makes the same particle field acquire a
        # different energy solely because the FFT box moved or changed shape.
        self._fourier_grid = None
        self._fourier_energy_offset = 0.0

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
            total_kinetic_energy=_F,
            total_helicity=_F,
            total_enstrophy=_F,
            test_filtered_enstrophy=_F,
            viscous_kinetic_energy_rate=_F,
            vortex_strength_magnitude_sum=_F,
            vortex_strength_x=_F,
            vortex_strength_y=_F,
            vortex_strength_z=_F,
            imp_x=_F,
            imp_y=_F,
            imp_z=_F,
            ang_x=_F,
            ang_y=_F,
            ang_z=_F,
        )
        self.total_quantities_results = FlowIntegralsStruct.field(shape=())

        # Separate fields for per-particle diagnostics
        self.particle_kinetic_energy = ti.field(dtype=ti.f32, shape=self.max_n_particles)
        self.particle_helicity = ti.field(dtype=ti.f32, shape=self.max_n_particles)
        self.particle_enstrophy = ti.field(dtype=ti.f32, shape=self.max_n_particles)

        # Cached vortex_centroid result fields (to avoid memory leak from repeated allocations)
        # CRITICAL: Taichi fields cannot be garbage collected, so we cache and reuse
        self._vortex_centroid_result = ti.field(dtype=self.accumulator_dtype, shape=3)

    def _resize_fields(self, required_size: int):
        """Validate that diagnostics fit the startup particle allocation."""
        if required_size <= 0:
            return
        if required_size > self.max_n_particles:
            raise ValueError(
                f"Diagnostics require {required_size} particles, but max_n_particles="
                f"{self.max_n_particles}. Increase Numerics.max_n_particles before "
                "constructing the solver."
            )

    def __str__(self):
        """Return formatted string representation."""
        lines = []
        lines.append(f"  Kernel Type              : {self.particle_kernel}")
        lines.append("  Domain Type              : Unbounded (diagnostic computations)")
        lines.append(f"  Accumulator Precision    : {self.accumulator_dtype}")
        lines.append(f"  Max Particles (allocated): {self.max_n_particles:,}")
        lines.append(
            f"  Time History Length      : {len(self._energy_history)} / {self._max_history_length}"
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
            raise ValueError(f"Unknown particle_kernel: {self.particle_kernel}")

        # Extract kernel functions from dictionary
        kernel_functions = {
            "q_sigma": kernel_dict["q_"],
            "zeta_sigma": kernel_dict["zeta_"],
            "g_sigma": kernel_dict["g_"],
        }

        # Define all kernels
        self._define_flow_integral_kernels(kernel_functions, kernel_dict)
        self._define_per_particle_kernels(kernel_functions)
        self._define_vortex_centroid_kernels(kernel_functions)

    def _define_flow_integral_kernels(self, kernel_functions, kernel_dict):
        """Define kernels for computing total flow integrals."""

        g_ = kernel_functions["g_sigma"]
        q_ = kernel_functions["q_sigma"]
        zeta_ = kernel_functions["zeta_sigma"]

        # Get angular impulse correction constant from kernel
        angular_correction_func = kernel_dict["angular_impulse_correction_constant_"]

        @ti.kernel
        def compute_flow_integrals_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            core_radius: ti.template(),
            particle_volume: ti.template(),
            viscosities_eff: ti.template(),
            results: ti.template(),
            n_particles_total: ti.i32,
        ):  # type: ignore
            """
            Compute all flow integrals in a single optimized GPU kernel.

            This combines energy, helicity, enstrophy, dissipation rates, strength,
            and impulses into one efficient computation using unbounded definitions.

            Angular impulse includes the per-particle kernel correction:

                A = (1/3) Σ r_i × (r_i × Γ_i)
                    - (2/9) C Σ σ_i² Γ_i

            The correction must remain inside the sum when core radii vary.
            Replacing it by a mean core radius times ``Σalpha`` is only equivalent for
            uniform cores and gives a false angular-impulse drift as soon as
            core spreading changes individual core_radius.
            """
            N = n_particles_total

            # Initialize every field to zero explicitly
            results[None].total_kinetic_energy = ti.cast(0.0, self.accumulator_dtype)
            results[None].total_helicity = ti.cast(0.0, self.accumulator_dtype)
            results[None].total_enstrophy = ti.cast(0.0, self.accumulator_dtype)
            results[None].test_filtered_enstrophy = ti.cast(0.0, self.accumulator_dtype)
            results[None].viscous_kinetic_energy_rate = ti.cast(0.0, self.accumulator_dtype)
            results[None].vortex_strength_magnitude_sum = ti.cast(0.0, self.accumulator_dtype)
            results[None].vortex_strength_x = ti.cast(0.0, self.accumulator_dtype)
            results[None].vortex_strength_y = ti.cast(0.0, self.accumulator_dtype)
            results[None].vortex_strength_z = ti.cast(0.0, self.accumulator_dtype)
            results[None].imp_x = ti.cast(0.0, self.accumulator_dtype)
            results[None].imp_y = ti.cast(0.0, self.accumulator_dtype)
            results[None].imp_z = ti.cast(0.0, self.accumulator_dtype)
            results[None].ang_x = ti.cast(0.0, self.accumulator_dtype)
            results[None].ang_y = ti.cast(0.0, self.accumulator_dtype)
            results[None].ang_z = ti.cast(0.0, self.accumulator_dtype)

            # Single loop for simple quantities (strength and impulses)
            for i in range(N):
                str_i = vortex_strength[i]
                pos_i = position[i]

                # Total strength magnitude: Σ|alpha| (sum of magnitudes, not |Σalpha|)
                str_mag = str_i.norm()
                ti.atomic_add(results[None].vortex_strength_magnitude_sum, str_mag)

                # Total vortex-strength vector: Σalpha (atomic for thread safety)
                ti.atomic_add(results[None].vortex_strength_x, str_i[ti.static(0)])
                ti.atomic_add(results[None].vortex_strength_y, str_i[ti.static(1)])
                ti.atomic_add(results[None].vortex_strength_z, str_i[ti.static(2)])

                # Linear impulse: I = 0.5 * Σ (r × Γ)
                cross_product = pos_i.cross(str_i)
                ti.atomic_add(results[None].imp_x, cross_product[ti.static(0)] * 0.5)
                ti.atomic_add(results[None].imp_y, cross_product[ti.static(1)] * 0.5)
                ti.atomic_add(results[None].imp_z, cross_product[ti.static(2)] * 0.5)

                # Angular impulse (raw, without correction): (1/3) Σ (r × (r × Γ))
                r_cross_vortex_strength = pos_i.cross(str_i)
                angular_contrib = pos_i.cross(r_cross_vortex_strength)
                correction_factor = ti.cast(
                    (2.0 / 9.0) * angular_correction_func() * core_radius[i] ** 2,
                    self.accumulator_dtype,
                )
                corrected = angular_contrib * (1.0 / 3.0) - correction_factor * str_i
                ti.atomic_add(results[None].ang_x, corrected[ti.static(0)])
                ti.atomic_add(results[None].ang_y, corrected[ti.static(1)])
                ti.atomic_add(results[None].ang_z, corrected[ti.static(2)])

            # Double loop for pairwise quantities (energy, helicity, enstrophy)
            for i in range(N):
                pos_i = position[i]
                str_i = vortex_strength[i]
                radii_i = core_radius[i]

                local_energy = ti.cast(0.0, self.accumulator_dtype)
                local_helicity = ti.cast(0.0, self.accumulator_dtype)
                local_enstrophy = ti.cast(0.0, self.accumulator_dtype)
                local_enstrophy_test = ti.cast(0.0, self.accumulator_dtype)
                local_dissipation = ti.cast(0.0, self.accumulator_dtype)

                for j in range(N):
                    pos_j = position[j]
                    str_j = vortex_strength[j]
                    radii_j = core_radius[j]

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
                        d_test = (
                            2.0
                            * (
                                particle_volume[i] ** (1.0 / 3.0)
                                + particle_volume[j] ** (1.0 / 3.0)
                            )
                            * 0.5
                        )
                        sigma_t = ti.sqrt(sigma_e * sigma_e + d_test * d_test)
                        zeta_test = zeta_(r_mag / sigma_t) / sigma_t**3
                        pair_enstrophy = ti.cast(zeta_val * str_i.dot(str_j), _acc)
                        local_enstrophy_test += ti.cast(zeta_test * str_i.dot(str_j), _acc)
                        pair_nu = ti.cast(0.5 * (viscosities_eff[i] + viscosities_eff[j]), _acc)
                        local_enstrophy += pair_enstrophy
                        local_dissipation -= pair_nu * pair_enstrophy

                # Atomic accumulation of local sums
                ti.atomic_add(results[None].total_kinetic_energy, local_energy)
                ti.atomic_add(results[None].total_helicity, local_helicity)
                ti.atomic_add(results[None].total_enstrophy, local_enstrophy)
                ti.atomic_add(results[None].test_filtered_enstrophy, local_enstrophy_test)
                ti.atomic_add(results[None].viscous_kinetic_energy_rate, local_dissipation)

        # Store kernel as instance method
        self.compute_flow_integrals_kernel = compute_flow_integrals_kernel

    def _define_kinetic_energy_kernel(self, g_):
        """Define and store the kinetic energy kernel using the provided g_sigma function."""

        @ti.kernel
        def compute_particles_kinetic_energy_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            core_radius: ti.template(),
            particle_kinetic_energy: ti.template(),
        ):  # type: ignore
            """Compute kinetic energy for each particle."""
            N = position.shape[0]
            for i in range(N):
                energy_sum = ti.cast(0.0, ti.f32)
                str_i = vortex_strength[i]
                pos_i = position[i]

                for j in range(N):
                    pos_j = position[j]
                    str_j = vortex_strength[j]
                    radii_j = core_radius[j]
                    sigma = 0.5 * (core_radius[i] + radii_j)

                    r_ij = pos_i - pos_j
                    r_mag = ti.sqrt(r_ij.dot(r_ij))
                    r_sigma = r_mag / sigma

                    if r_sigma <= DEFAULT_CUTOFF_RADIUS_FACTOR:
                        # Convolved pair width — see compute_flow_integrals_kernel.
                        sigma_e = ti.sqrt(core_radius[i] * core_radius[i] + radii_j * radii_j)
                        g_val = g_(r_mag / sigma_e) / sigma_e
                        energy_sum += g_val * str_j.dot(str_i) * 0.5

                particle_kinetic_energy[i] = energy_sum

        self.compute_particles_kinetic_energy_kernel = compute_particles_kinetic_energy_kernel

    def _define_helicity_kernel(self, q_):
        """Define and store the helicity kernel using the provided q_sigma function."""

        @ti.kernel
        def compute_particles_helicity_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            core_radius: ti.template(),
            particle_helicity: ti.template(),
        ):  # type: ignore
            """Compute helicity for each particle."""
            N = position.shape[0]
            for i in range(N):
                hel = ti.cast(0.0, ti.f32)
                str_i = vortex_strength[i]
                pos_i = position[i]
                radii_i = core_radius[i]
                cutoff_radius = DEFAULT_CUTOFF_RADIUS_FACTOR * radii_i

                for j in range(N):
                    pos_j = position[j]
                    str_j = vortex_strength[j]

                    r_ij = pos_i - pos_j
                    r_mag = ti.sqrt(r_ij.dot(r_ij))

                    if r_mag > EPSILON and r_mag <= cutoff_radius:
                        # Convolved pair width — see compute_flow_integrals_kernel.
                        sigma_e = ti.sqrt(radii_i * radii_i + core_radius[j] * core_radius[j])
                        q_val = q_(r_mag / sigma_e)
                        hel += q_val * r_ij.dot(str_i.cross(str_j)) / (r_mag**3)

                particle_helicity[i] = hel

        self.compute_particles_helicity_kernel = compute_particles_helicity_kernel

    def _define_enstrophy_kernel(self, zeta_):
        """Define and store the enstrophy kernel using the provided zeta_sigma function."""

        @ti.kernel
        def compute_particles_enstrophy_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            core_radius: ti.template(),
            particle_enstrophy: ti.template(),
        ):  # type: ignore
            """Compute enstrophy for each particle."""
            N = position.shape[0]
            for i in range(N):
                enstrophy_local = ti.cast(0.0, ti.f32)
                str_i = vortex_strength[i]
                pos_i = position[i]
                radii_i = core_radius[i]

                for j in range(N):
                    r_ij = pos_i - position[j]
                    r_mag = ti.sqrt(r_ij.dot(r_ij))
                    sigma = 0.5 * (radii_i + core_radius[j])
                    r_sigma = r_mag / sigma

                    if r_sigma <= DEFAULT_CUTOFF_RADIUS_FACTOR:
                        str_j = vortex_strength[j]
                        # Convolved pair width — see compute_flow_integrals_kernel.
                        sigma_e = ti.sqrt(radii_i * radii_i + core_radius[j] * core_radius[j])
                        zeta_val = zeta_(r_mag / sigma_e) / sigma_e**3
                        enstrophy_local += zeta_val * str_i.dot(str_j)

                particle_enstrophy[i] = enstrophy_local

        self.compute_particles_enstrophy_kernel = compute_particles_enstrophy_kernel

    def _define_vorticity_reconstruction_kernel(self, zeta_):
        """Define the ω_h reconstruction kernel: ω_h(xᵢ) = Σⱼ ζ_σ(rᵢⱼ/σ)/σ³ · Γⱼ."""

        @ti.kernel
        def reconstruct_vorticity_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            core_radius: ti.template(),
            out: ti.template(),
            n_particles_total: ti.i32,
        ):  # type: ignore
            for i in range(n_particles_total):
                acc = ti.Vector([0.0, 0.0, 0.0])
                pos_i = position[i]
                for j in range(n_particles_total):
                    r_ij = pos_i - position[j]
                    r_mag = r_ij.norm()
                    # This is a pointwise field reconstruction, not a
                    # quadratic two-blob integral: source j contributes its
                    # own blob zeta_{sigma_j}(x_i-x_j).  A target/source mean
                    # core radius gives the wrong field as soon as core spreading
                    # makes core_radius nonuniform and makes vortex_strength--omega alignment
                    # depend on the arbitrary target core.
                    sigma = core_radius[j]
                    if r_mag <= DEFAULT_CUTOFF_RADIUS_FACTOR * sigma:
                        r_sigma = r_mag / sigma
                        zeta_val = zeta_(r_sigma) / sigma**3
                        acc += zeta_val * vortex_strength[j]
                out[i] = acc

        self.reconstruct_vorticity_kernel = reconstruct_vorticity_kernel

    def _define_per_particle_kernels(self, kernel_functions):
        """Define kernels for computing per-particle diagnostics."""
        self._define_kinetic_energy_kernel(kernel_functions["g_sigma"])
        self._define_helicity_kernel(kernel_functions["q_sigma"])
        self._define_enstrophy_kernel(kernel_functions["zeta_sigma"])
        self._define_vorticity_reconstruction_kernel(kernel_functions["zeta_sigma"])

    def _define_group_vortex_centroid_kernel(self):
        """Define and store the group-vortex_centroid kernel."""

        @ti.kernel
        def compute_group_vortex_centroid_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            group_id: ti.template(),
            n_particles_total: ti.i32,
            target_group: ti.i32,
            result: ti.template(),
        ):  # type: ignore
            """
            Compute vortex_centroid of vortex strength for a specific group.

            Centroid = Σ(r × Γ) / Σ|Γ| for particles in the group
            """
            weighted_pos = ti.Vector([0.0, 0.0, 0.0])
            total_strength_mag = 0.0

            for i in range(n_particles_total):
                if group_id[i] == target_group:
                    str_mag = vortex_strength[i].norm()
                    if str_mag > EPSILON:
                        ti.atomic_add(weighted_pos[0], position[i][0] * str_mag)
                        ti.atomic_add(weighted_pos[1], position[i][1] * str_mag)
                        ti.atomic_add(weighted_pos[2], position[i][2] * str_mag)
                        ti.atomic_add(total_strength_mag, str_mag)

            if total_strength_mag > EPSILON:
                result[0] = weighted_pos[0] / total_strength_mag
                result[1] = weighted_pos[1] / total_strength_mag
                result[2] = weighted_pos[2] / total_strength_mag
            else:
                result[0] = 0.0
                result[1] = 0.0
                result[2] = 0.0

        self.compute_group_vortex_centroid_kernel = compute_group_vortex_centroid_kernel

    def _define_global_vortex_centroid_kernel(self):
        """Define and store the global-vortex_centroid kernel."""

        @ti.kernel
        def compute_global_vortex_centroid_kernel(
            position: ti.template(),
            vortex_strength: ti.template(),
            n_particles_total: ti.i32,
            result: ti.template(),
        ):  # type: ignore
            """
            Compute vortex_centroid of vortex strength for the entire particle set.

            Centroid = Σ(r * |Γ|) / Σ|Γ| over all particles
            """
            weighted_pos = ti.Vector([0.0, 0.0, 0.0])
            total_strength_mag = 0.0

            for i in range(n_particles_total):
                str_mag = vortex_strength[i].norm()
                if str_mag > EPSILON:
                    ti.atomic_add(weighted_pos[0], position[i][0] * str_mag)
                    ti.atomic_add(weighted_pos[1], position[i][1] * str_mag)
                    ti.atomic_add(weighted_pos[2], position[i][2] * str_mag)
                    ti.atomic_add(total_strength_mag, str_mag)

            if total_strength_mag > EPSILON:
                result[0] = weighted_pos[0] / total_strength_mag
                result[1] = weighted_pos[1] / total_strength_mag
                result[2] = weighted_pos[2] / total_strength_mag
            else:
                result[0] = 0.0
                result[1] = 0.0
                result[2] = 0.0

        self.compute_global_vortex_centroid_kernel = compute_global_vortex_centroid_kernel

    def _define_vortex_centroid_kernels(self, kernel_functions):
        """Define kernels for vortex-strength-magnitude-weighted vortex_centroids."""
        self._define_group_vortex_centroid_kernel()
        self._define_global_vortex_centroid_kernel()

    # PUBLIC API METHODS

    @staticmethod
    def record_vortex_centroid_history(
        diagnostics_history: dict,
        position: np.ndarray,
        vortex_strength: np.ndarray,
        event_observer: PhysicsEventObserver | None = None,
    ) -> None:
        """Append the vortex-strength-magnitude-weighted particle vortex_centroid.

        Args:
            diagnostics_history: Solver's ``_diagnostics_history`` dict (mutated in-place).
            position: Particle position array of shape (N, 3).
            vortex_strength: Particle vortex-strength array of shape (N, 3) [m³/s].
        """
        if "vortex_centroid" not in diagnostics_history:
            return
        try:
            if position.size == 0 or vortex_strength.size == 0:
                vortex_centroid = np.array([0.0, 0.0, 0.0])
            else:
                vortex_strength_magnitude = np.linalg.norm(vortex_strength, axis=1)
                total_magnitude = vortex_strength_magnitude.sum()
                if total_magnitude > 0:
                    vortex_centroid = (position * vortex_strength_magnitude[:, np.newaxis]).sum(
                        axis=0
                    ) / total_magnitude
                else:
                    vortex_centroid = np.array([0.0, 0.0, 0.0])
            diagnostics_history["vortex_centroid"].append(tuple(vortex_centroid.tolist()))
        except Exception as exc:
            (event_observer or NullPhysicsEventObserver()).warning(
                f"component=flow_diagnostics quantity=vortex_strength_centroid "
                f"status=evaluation_failed error={exc!r}"
            )

    def compute_vortex_centroid(self, particles) -> np.ndarray:
        """
        Compute the vortex-strength-magnitude-weighted particle vortex_centroid.

        Returns:
            np.ndarray: Centroid position [x, y, z], or zeros if no particles.
        """
        N = len(particles)
        if N == 0:
            return np.array([0.0, 0.0, 0.0])

        self._resize_fields(N)

        # Use cached result field (avoids memory leak from repeated Taichi allocations)
        # CRITICAL: Taichi fields cannot be garbage collected
        self._vortex_centroid_result.fill(0)

        # Call kernel
        self.compute_global_vortex_centroid_kernel(
            particles.position,
            particles.vortex_strength,
            N,
            self._vortex_centroid_result,
        )

        # Extract result
        vortex_centroid = self._vortex_centroid_result.to_numpy()
        return vortex_centroid

    def compute_flow_integrals(self, particles, time: float, record_history: bool = True):
        """
        Compute all flow integral quantities in a single efficient GPU kernel call.

        This method computes and stores time history for energy dissipation rate calculation.

        Args:
            particles: Particles object containing position, vortex_strength, core_radius, kinematic_viscosity
            time: Current simulation time [s]
            record_history: Whether to append this sample to the kinetic-energy
                history used for finite-difference dE/dt diagnostics.

        Returns:
            dict: Dictionary containing all flow integral quantities:
                - 'total_kinetic_energy': Total kinetic energy [J]
                - 'total_helicity': Total helicity [m³/s²]
                - 'total_enstrophy': Total enstrophy [1/s²]
                - 'viscous_kinetic_energy_rate': Viscous energy rate [J/s]
                - 'kinetic_energy_rate': Signed kinetic-energy rate [J/s]
                - 'vortex_strength_magnitude_sum': Sum of particle-strength norms [m³/s]
                - 'net_vortex_strength': Net strength vector [1/s]
                - 'linear_impulse': Linear impulse vector [m³/s]
                - 'angular_impulse': Angular impulse vector [m⁴/s]
        """
        N = len(particles)
        if N == 0:
            # Return zero values for empty particle system
            return self._get_zero_results()
        if _DIRECT_INTEGRAL_PARTICLE_LIMIT < N and self.particle_kernel == "GAUSSIAN":
            return self._compute_fourier_flow_integrals(particles, time, record_history)

        self._resize_fields(N)

        # Initialize results struct to zero before kernel call
        self.total_quantities_results.fill(0)

        # Call the combined kernel
        self.compute_flow_integrals_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.particle_volume,
            particles.effective_viscosity,
            self.total_quantities_results,
            N,
        )

        # Extract results from Taichi struct field
        r = self.total_quantities_results[None]
        total_kinetic_energy = float(r.total_kinetic_energy)
        if record_history:
            self._update_energy_history(time, total_kinetic_energy, "unbounded_energy")

        # Prime the scalable energy definition before the direct O(N^2)
        # crossover.  Calibrating both definitions on the same particle state
        # makes the first Fourier sample continuous with the direct history.
        if (
            record_history
            and self.particle_kernel == "GAUSSIAN"
            and N >= _FOURIER_WARMUP_PARTICLE_LIMIT
        ):
            self._prime_fourier_energy_tracker(particles, total_kinetic_energy)

        # Compute kinetic energy dissipation rate using finite differences
        dE_dt = self._compute_energy_dissipation_rate()

        return {
            "total_kinetic_energy": total_kinetic_energy,
            "total_helicity": float(r.total_helicity),
            "total_enstrophy": float(r.total_enstrophy),
            "test_filtered_enstrophy": float(r.test_filtered_enstrophy),
            "viscous_kinetic_energy_rate": float(r.viscous_kinetic_energy_rate),
            "kinetic_energy_rate": dE_dt,
            "kinetic_energy_rate_source": "direct_energy_backward_difference",
            "vortex_strength_magnitude_sum": float(r.vortex_strength_magnitude_sum),
            "net_vortex_strength": np.array(
                [float(r.vortex_strength_x), float(r.vortex_strength_y), float(r.vortex_strength_z)]
            ),
            "linear_impulse": np.array([float(r.imp_x), float(r.imp_y), float(r.imp_z)]),
            "angular_impulse": np.array([float(r.ang_x), float(r.ang_y), float(r.ang_z)]),
        }

    def _compute_fourier_flow_integrals(
        self,
        particles,
        time: float,
        record_history: bool,
    ) -> dict:
        position = particles.position_cpu().astype(np.float64)
        vortex_strength = particles.vortex_strength_cpu().astype(np.float64)
        core_radius = particles.core_radius_cpu().astype(np.float64)
        particle_volume = particles.particle_volume_cpu().astype(np.float64)
        effective_viscosity = particles.effective_viscosity_cpu().astype(np.float64)
        saved_grid = self._fourier_grid
        saved_offset = self._fourier_energy_offset
        spectral, continuity_preserved = self._fourier_integrals_on_persistent_grid(
            position,
            vortex_strength,
            core_radius,
            particle_volume,
            effective_viscosity,
        )
        evaluated_offset = self._fourier_energy_offset
        if not record_history:
            # Trial diagnostics used by regularisation must not alter the
            # accepted time-history definition.
            self._fourier_grid = saved_grid
            self._fourier_energy_offset = saved_offset
        if spectral.viscous_kinetic_energy_rate is None:
            raise RuntimeError("Fourier flow diagnostics did not compute viscous dissipation")

        rate_source = "fourier_energy_backward_difference"
        if record_history and not continuity_preserved and self._energy_history:
            # This is only needed if a cloud grows farther between output
            # samples than the reserved grid margin.  Anchor the new energy
            # definition with the physical viscous rate for that one interval;
            # subsequent samples again use measured backward differences.
            previous_time, previous_energy, _ = self._energy_history[-1]
            interval = time - previous_time
            if interval > 0.0:
                target_energy = previous_energy + spectral.viscous_kinetic_energy_rate * interval
                self._fourier_energy_offset = target_energy - spectral.total_kinetic_energy
                evaluated_offset = self._fourier_energy_offset
                rate_source = "fourier_transition_viscous_rate"

        total_kinetic_energy = spectral.total_kinetic_energy + evaluated_offset
        if record_history:
            self._update_energy_history(time, total_kinetic_energy, "unbounded_energy")
        dE_dt = self._compute_energy_dissipation_rate()
        total = vortex_strength.sum(axis=0, dtype=np.float64)
        impulse = 0.5 * np.cross(position, vortex_strength).sum(axis=0, dtype=np.float64)
        angular = np.cross(position, np.cross(position, vortex_strength)).sum(
            axis=0, dtype=np.float64
        ) / 3.0 - (1.0 / 3.0) * (core_radius[:, None] ** 2 * vortex_strength).sum(axis=0)
        return {
            "total_kinetic_energy": total_kinetic_energy,
            "total_helicity": spectral.total_helicity,
            "total_enstrophy": spectral.total_enstrophy,
            "test_filtered_enstrophy": spectral.test_filtered_enstrophy,
            "viscous_kinetic_energy_rate": spectral.viscous_kinetic_energy_rate,
            "kinetic_energy_rate": dE_dt,
            "kinetic_energy_rate_source": rate_source,
            "vortex_strength_magnitude_sum": float(np.linalg.norm(vortex_strength, axis=1).sum()),
            "net_vortex_strength": total,
            "linear_impulse": impulse,
            "angular_impulse": angular,
        }

    def _prime_fourier_energy_tracker(self, particles, direct_energy: float) -> None:
        """Align the scalable energy definition with a simultaneous direct value."""
        position = particles.position_cpu().astype(np.float64)
        vortex_strength = particles.vortex_strength_cpu().astype(np.float64)
        core_radius = particles.core_radius_cpu().astype(np.float64)
        particle_volume = particles.particle_volume_cpu().astype(np.float64)
        effective_viscosity = particles.effective_viscosity_cpu().astype(np.float64)
        spectral, _ = self._fourier_integrals_on_persistent_grid(
            position,
            vortex_strength,
            core_radius,
            particle_volume,
            effective_viscosity,
        )
        self._fourier_energy_offset = direct_energy - spectral.total_kinetic_energy

    @staticmethod
    def _fit_fourier_grid(position, spacing: float, shape=None, vortex_strength=None):
        """Return a translation-following grid centred around the particle support."""
        from ..numerics.fourier_integrals import CartesianGrid, _grid_for_particles

        required = _grid_for_particles(position, spacing)
        required_shape = np.asarray(required.shape, dtype=np.int64)
        if shape is None:
            fitted_shape = required_shape + _FOURIER_GRID_EXTRA_CELLS
        else:
            fitted_shape = np.asarray(shape, dtype=np.int64)
            if np.any(fitted_shape < required_shape):
                raise ValueError("Fourier diagnostic grid is too small for the particle support")
        if vortex_strength is None:
            centre = 0.5 * (position.min(axis=0) + position.max(axis=0))
        else:
            weights = np.linalg.norm(vortex_strength, axis=1)
            centre = (
                np.average(position, axis=0, weights=weights)
                if float(weights.sum()) > 0.0
                else position.mean(axis=0)
            )
        desired_origin = centre - 0.5 * (fitted_shape - 1) * spacing
        # Clamp the translating box only when the weighted centre would leave
        # too little room for the four-point M4 scatter stencil.
        lower_bound = position.max(axis=0) - (fitted_shape - 2.000001) * spacing
        upper_bound = position.min(axis=0) - spacing
        origin = np.minimum(np.maximum(desired_origin, lower_bound), upper_bound)
        return CartesianGrid(
            origin=origin.astype(np.float64),
            spacing=float(spacing),
            shape=tuple(int(value) for value in fitted_shape),
        )

    def _fourier_integrals_on_persistent_grid(
        self,
        position,
        vortex_strength,
        core_radius,
        particle_volume,
        effective_viscosity,
    ):
        """Evaluate on a reusable grid and preserve energy across grid growth."""
        from ..numerics.fourier_integrals import _grid_for_particles, gaussian_fourier_integrals

        if self._fourier_grid is None:
            spacing = float(np.median(np.cbrt(particle_volume)))
            self._fourier_grid = self._fit_fourier_grid(
                position, spacing, vortex_strength=vortex_strength
            )
            spectral = gaussian_fourier_integrals(
                position,
                vortex_strength,
                core_radius,
                particle_volume,
                effective_viscosity=effective_viscosity,
                grid=self._fourier_grid,
            )
            return spectral, False

        old_grid = self._fourier_grid
        required = _grid_for_particles(position, old_grid.spacing)
        required_shape = np.asarray(required.shape, dtype=np.int64)
        old_shape = np.asarray(old_grid.shape, dtype=np.int64)
        fits = bool(np.all(required_shape <= old_shape))
        needs_growth = (not fits) or bool(
            np.any(old_shape - required_shape < _FOURIER_GRID_MIN_SLACK)
        )

        old_spectral = None
        if fits:
            # Follow rigid cloud translation continuously, keeping particles at
            # the same sub-cell phase instead of injecting grid-crossing noise.
            old_grid = self._fit_fourier_grid(
                position,
                old_grid.spacing,
                old_shape,
                vortex_strength,
            )
            self._fourier_grid = old_grid
            old_spectral = gaussian_fourier_integrals(
                position,
                vortex_strength,
                core_radius,
                particle_volume,
                effective_viscosity=effective_viscosity,
                grid=old_grid,
            )
            if not needs_growth:
                return old_spectral, True

        grown_shape = np.maximum(
            required_shape + _FOURIER_GRID_EXTRA_CELLS,
            np.ceil(old_shape * 1.25).astype(np.int64),
        )
        new_grid = self._fit_fourier_grid(
            position,
            old_grid.spacing,
            grown_shape,
            vortex_strength,
        )
        new_spectral = gaussian_fourier_integrals(
            position,
            vortex_strength,
            core_radius,
            particle_volume,
            effective_viscosity=effective_viscosity,
            grid=new_grid,
        )
        self._fourier_grid = new_grid
        if old_spectral is None:
            return new_spectral, False

        corrected_old_energy = old_spectral.total_kinetic_energy + self._fourier_energy_offset
        self._fourier_energy_offset = corrected_old_energy - new_spectral.total_kinetic_energy
        return new_spectral, True

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
            particles: Particle container with position, vortex strength, and core radius.
            out_field: ti.Vector.field(3, ...) to receive ω_h values (written in-place).
        """
        N = particles.n_particles_total
        if N == 0:
            return
        self.reconstruct_vorticity_kernel(
            particles.position, particles.vortex_strength, particles.core_radius, out_field, N
        )

    def compute_vortex_centroids_by_group(self, particles) -> dict[int, np.ndarray]:
        """
        Compute vortex-strength-magnitude-weighted vortex_centroids by particle group.

        Args:
            particles: Particles object

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping group_id to vortex_centroid position [x, y, z]
        """
        N = len(particles)
        if N == 0:
            return {}

        # Check if particles have group_id (field name is singular, not "group_id")
        if not hasattr(particles, "group_id"):
            return {}

        self._resize_fields(N)

        # Get unique group IDs (accessor is group_id_cpu(), not group_ids_cpu())
        group_ids_np = particles.group_id_cpu()
        unique_groups = np.unique(group_ids_np)

        vortex_centroids = {}
        for group_id in unique_groups:
            # Use cached result field (avoids memory leak from repeated Taichi allocations)
            # CRITICAL: Taichi fields cannot be garbage collected
            self._vortex_centroid_result.fill(0)

            # Compute vortex_centroid
            self.compute_group_vortex_centroid_kernel(
                particles.position,
                particles.vortex_strength,
                particles.group_id,
                N,
                int(group_id),
                self._vortex_centroid_result,
            )

            # Extract result (copy before reusing field)
            vortex_centroids[int(group_id)] = self._vortex_centroid_result.to_numpy().copy()

        return vortex_centroids

    # ENERGY DISSIPATION RATE COMPUTATION

    def _update_energy_history(
        self,
        time: float,
        total_kinetic_energy: float,
        measurement: str = "direct",
    ):
        """
        Update the time history of kinetic energy.

        Args:
            time: Current simulation time [s]
            total_kinetic_energy: Current total kinetic energy [J]
            measurement: Energy definition shared by comparable samples.
        """
        # Replace the latest entry when callers request diagnostics multiple
        # times at the same physical time. This keeps dE/dt finite differences
        # well-posed and avoids zero-dt history pairs.
        if self._energy_history and abs(self._energy_history[-1][0] - time) < 1e-12:
            self._energy_history[-1] = (time, total_kinetic_energy, measurement)
            return

        # Add new entry
        self._energy_history.append((time, total_kinetic_energy, measurement))

        # Keep only the last N entries
        if len(self._energy_history) > self._max_history_length:
            self._energy_history.pop(0)

    def _compute_energy_dissipation_rate(self) -> float:
        """Return the energy change over the latest diagnostic interval.

        An endpoint derivative extrapolated from a higher-order polynomial can
        have the wrong sign even when every sampled energy decreases.  The
        backward secant is conservative, handles non-uniform output intervals,
        and makes the reported rate exactly consistent with the latest pair of
        diagnostic states. Direct and persistent-grid Fourier samples share the
        calibrated ``unbounded_energy`` definition.
        """
        if len(self._energy_history) < 2:
            return 0.0

        previous_time, previous_energy, previous_measurement = self._energy_history[-2]
        current_time, current_energy, current_measurement = self._energy_history[-1]
        if previous_measurement != "unbounded_energy" or current_measurement != "unbounded_energy":
            return float("nan")
        interval = current_time - previous_time
        if interval <= 0.0:
            return 0.0
        return float((current_energy - previous_energy) / interval)

    def _get_zero_results(self) -> dict:
        """Return dictionary of zero values for empty particle system."""
        return {
            "total_kinetic_energy": 0.0,
            "total_helicity": 0.0,
            "total_enstrophy": 0.0,
            "test_filtered_enstrophy": 0.0,
            "viscous_kinetic_energy_rate": 0.0,
            "kinetic_energy_rate": 0.0,
            "kinetic_energy_rate_source": "empty_particle_field",
            "vortex_strength_magnitude_sum": 0.0,
            "net_vortex_strength": np.array([0.0, 0.0, 0.0]),
            "linear_impulse": np.array([0.0, 0.0, 0.0]),
            "angular_impulse": np.array([0.0, 0.0, 0.0]),
        }

    def reset_energy_history(self):
        """Reset the energy time history (useful when restarting simulation)."""
        self._energy_history.clear()


# =========================================================
# PUBLIC API EXPORTS
# =========================================================

__all__ = ["ParticleFieldEvaluation"]
