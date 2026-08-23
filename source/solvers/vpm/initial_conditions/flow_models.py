"""
Analytic flow-field generators for VPM initialization and validation: Lamb-Oseen,
vortex ring, doublet, Taylor-Green, and isotropic turbulence.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np

# =========================================================


def lamb_oseen_vpm(
    kinematic_viscosity: float,
    mean_core_radius: float,
    position: np.ndarray,
    particle_volume: np.ndarray,
    vortex_centre_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
    circulation: float = np.pi,
    vortex_age: float = 1.0,
    perturbation_amplitude: float = 0.0,
    n_perturbation_modes: int = 12,
    is_anti_diffusion_enabled: bool = False,
):
    """Initialize Lamb-Oseen vortex field on particles.

    Parameters:
        kinematic_viscosity: Kinematic viscosity [m²/s]
        mean_core_radius: Average particle core radius [m]
        position: Particle position (N, 3)
        particle_volume: Particle volume (N,)
        vortex_centre_position: Vortex centre position [m]
        circulation: Total line-integral circulation [m²/s]
        vortex_age: Vortex age [s] (for viscous diffusion)
        perturbation_amplitude: Perturbation amplitude
        n_perturbation_modes: Number of perturbation modes
        is_anti_diffusion_enabled: Compensate for particle core diffusion

    Returns:
        velocity: (N, 3) ndarray
        kinematic_viscosity: (N,) ndarray
        vortex_strength: Particle alpha vectors, shape (N, 3) [m³/s]
    """
    n_particles_total = len(position)
    vorticity = np.zeros_like(position)
    velocity = np.zeros_like(position)

    # Handle inviscid case (kinematic_viscosity=0)
    if kinematic_viscosity <= 0.0:
        # Use particle radius as core size for inviscid vortex
        vortex_core_radius_squared = mean_core_radius**2
    else:
        # Adjust vortex_age for anti-diffusion
        # Gaussian particles use exp(-r^2/sigma^2), so their blob variance adds
        # an effective age shift sigma^2/(4*kinematic_viscosity) to the represented Lamb-Oseen core.
        anti_diffusion_time_shift = (
            (mean_core_radius**2 / (4.0 * kinematic_viscosity))
            if is_anti_diffusion_enabled
            else 0.0
        )
        if anti_diffusion_time_shift > vortex_age:
            raise ValueError(
                "Invalid parameters: vortex_age too small for anti-diffusion correction."
            )
        effective_vortex_age = vortex_age - anti_diffusion_time_shift
        # Compute core size squared from vortex_age
        vortex_core_radius_squared = 4.0 * kinematic_viscosity * effective_vortex_age

    # Add perturbation to the vortex centre_position
    position_z = position[:, 2]
    perturbation_phases = 2 * np.pi * np.random.rand(n_perturbation_modes)
    centreline_perturbation = np.zeros_like(position_z)
    for mode in range(1, n_perturbation_modes + 1):
        centreline_perturbation += np.cos(mode * position_z + perturbation_phases[mode - 1])
    centreline_perturbation /= np.sqrt(n_perturbation_modes)

    vortex_centre_x = vortex_centre_position[0] + perturbation_amplitude * centreline_perturbation
    vortex_centre_y = vortex_centre_position[1] + perturbation_amplitude * centreline_perturbation

    relative_x = position[:, 0] - vortex_centre_x
    relative_y = position[:, 1] - vortex_centre_y
    radial_distance = np.sqrt(relative_x**2 + relative_y**2)

    for i in range(n_particles_total):
        vorticity_profile = np.exp(-(radial_distance[i] ** 2) / vortex_core_radius_squared)
        vorticity[i, 2] = (circulation / (np.pi * vortex_core_radius_squared)) * vorticity_profile
        if radial_distance[i] > 1e-12:
            azimuthal_velocity = (circulation / (2 * np.pi * radial_distance[i])) * (
                1.0 - vorticity_profile
            )
            velocity[i, 0] = -azimuthal_velocity * relative_y[i] / radial_distance[i]
            velocity[i, 1] = azimuthal_velocity * relative_x[i] / radial_distance[i]

    kinematic_viscosity = np.full(n_particles_total, max(kinematic_viscosity, 0.0))
    vortex_strength = vorticity * particle_volume[:, None]

    return velocity, kinematic_viscosity, vortex_strength


# =========================================================


def vortex_ring_centreline(
    azimuth_angle: np.ndarray,
    ring_radius: float,
    widnall_amplitude: float = 0.0,
    seed: int = 42,
    n_widnall_modes: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reproducible perturbed ring radius and its azimuthal slope."""
    if n_widnall_modes < 1:
        raise ValueError("n_widnall_modes must be at least 1")

    rng = np.random.RandomState(seed)
    perturbation_phases = 2.0 * np.pi * rng.rand(n_widnall_modes)
    perturbation_shape = np.zeros_like(azimuth_angle, dtype=float)
    perturbation_slope = np.zeros_like(azimuth_angle, dtype=float)
    for mode in range(1, n_widnall_modes + 1):
        phase = mode * azimuth_angle + perturbation_phases[mode - 1]
        perturbation_shape += np.cos(phase)
        perturbation_slope -= mode * np.sin(phase)
    mode_normalization = np.sqrt(n_widnall_modes)
    centreline_radius = ring_radius * (
        1.0 + widnall_amplitude * perturbation_shape / mode_normalization
    )
    centreline_slope = ring_radius * widnall_amplitude * perturbation_slope / mode_normalization
    return centreline_radius, centreline_slope


def vortex_ring_vpm(
    kinematic_viscosity: float,
    ring_centre: np.ndarray,
    tube_circulation: float,
    ring_radius: float,
    ring_core_radius: float,
    mean_core_radius: float,
    position: np.ndarray,
    particle_volume: np.ndarray = None,
    widnall_amplitude: float = 0.0,
    seed: int = 42,
    n_widnall_modes: int = 24,
    is_anti_diffusion_enabled: bool = False,
    diffusivity_constant: float = 4.0,
    is_circulation_normalization_enabled: bool = False,
):
    """
    Initialize a vortex ring with an optional Widnall-type perturbation.

    The perturbation displaces the ring centreline and adds the radial
    vorticity component required for the field to remain divergence-free.
    In cylindrical coordinates about the x-axis,

        R_c(theta) = R (1 + widnall_amplitude g(theta))
        omega = W [e_theta + R_c'(theta) / rho e_rho]

    is the curl of an axial vector potential, so ``div(omega) = 0`` in the
    continuum.

    Returns:
    --------
    velocity : (N,3) ndarray
    kinematic_viscosity : (N,) ndarray
    vortex_strength : (N,3) ndarray

    ``diffusivity_constant`` must match the selected particle kernel's
    core-spreading law because the anti-diffusion shift is
    ``sigma_0² / (C_nu * kinematic_viscosity)``.  It is 4 for the Gaussian and
    Winckelmans--Leonard kernels.

    With ``is_circulation_normalization_enabled=True``, the discrete cross-sectional flux is
    rescaled to ``tube_circulation``.  This compensates only for finite quadrature
    and a deliberately truncated Gaussian tail; it does not alter the profile.
    """
    n_particles_total = len(position)
    if n_widnall_modes < 1:
        raise ValueError("n_widnall_modes must be at least 1")

    if diffusivity_constant <= 0.0:
        raise ValueError("diffusivity_constant must be positive")
    anti_diffusion_time_shift = (
        mean_core_radius**2 / (kinematic_viscosity * diffusivity_constant)
        if is_anti_diffusion_enabled
        else 0.0
    )

    physical_diffusion_time = ring_core_radius**2 / (4 * kinematic_viscosity)
    if anti_diffusion_time_shift > physical_diffusion_time:
        raise ValueError("Invalid parameters: represented core radius would be imaginary")
    represented_core_radius_squared = (
        4 * kinematic_viscosity * (physical_diffusion_time - anti_diffusion_time_shift)
    )

    relative_x = position[:, 0] - ring_centre[0]
    relative_y = position[:, 1] - ring_centre[1]
    relative_z = position[:, 2] - ring_centre[2]
    azimuth_angle = np.arctan2(relative_z, relative_y)

    radial_distance = np.sqrt(relative_y**2 + relative_z**2)
    centreline_radius, centreline_slope = vortex_ring_centreline(
        azimuth_angle,
        ring_radius,
        widnall_amplitude=widnall_amplitude,
        seed=seed,
        n_widnall_modes=n_widnall_modes,
    )
    core_distance = np.sqrt((radial_distance - centreline_radius) ** 2 + relative_x**2)

    vorticity_magnitude = (tube_circulation / (np.pi * represented_core_radius_squared)) * np.exp(
        -(core_distance**2) / represented_core_radius_squared
    )
    vorticity = np.zeros_like(position)
    radial_vorticity = np.zeros_like(vorticity_magnitude)
    away_from_axis = radial_distance > 1.0e-12
    radial_vorticity[away_from_axis] = (
        vorticity_magnitude[away_from_axis]
        * centreline_slope[away_from_axis]
        / radial_distance[away_from_axis]
    )
    vorticity[:, 1] = -vorticity_magnitude * np.sin(azimuth_angle) + radial_vorticity * np.cos(
        azimuth_angle
    )
    vorticity[:, 2] = vorticity_magnitude * np.cos(azimuth_angle) + radial_vorticity * np.sin(
        azimuth_angle
    )

    velocity = np.zeros_like(position)
    kinematic_viscosity = np.full(n_particles_total, kinematic_viscosity)
    if particle_volume is None:
        particle_volume = np.full(n_particles_total, 1.0)
    vortex_strength = vorticity * particle_volume[:, None]
    if is_circulation_normalization_enabled:
        tangent = np.zeros_like(position)
        tangent[away_from_axis, 1] = -relative_z[away_from_axis] / radial_distance[away_from_axis]
        tangent[away_from_axis, 2] = relative_y[away_from_axis] / radial_distance[away_from_axis]
        represented_circulation = np.sum(
            np.einsum("ij,ij->i", vortex_strength[away_from_axis], tangent[away_from_axis])
            / radial_distance[away_from_axis]
        ) / (2.0 * np.pi)
        if abs(represented_circulation) <= np.finfo(float).tiny:
            raise ValueError("cannot normalize a ring with zero represented circulation")
        vortex_strength *= tube_circulation / represented_circulation

    return velocity, kinematic_viscosity, vortex_strength


# =========================================================


def doublet_flow_vpm(
    kinematic_viscosity: float,
    centre_position: np.ndarray,
    doublet_direction: np.ndarray,
    doublet_strength: float,
    position: np.ndarray,
    particle_volume: np.ndarray = None,
):
    """
    Generate a vorticity field corresponding to a 3D doublet (vortex dipole) flow using particles.

    Returns:
    --------
    velocity : (N,3) ndarray
    kinematic_viscosity : (N,) ndarray
    vortex_strength : (N,3) ndarray
    """
    n_particles_total = position.shape[0]
    vorticity = np.zeros_like(position)
    velocity = np.zeros_like(position)

    direction_x = doublet_direction[0]
    direction_y = doublet_direction[1]
    direction_z = doublet_direction[2]
    direction_magnitude = np.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
    direction_x /= direction_magnitude
    direction_y /= direction_magnitude
    direction_z /= direction_magnitude

    for i in range(n_particles_total):
        relative_x = position[i, 0] - centre_position[0]
        relative_y = position[i, 1] - centre_position[1]
        relative_z = position[i, 2] - centre_position[2]

        distance_squared = relative_x**2 + relative_y**2 + relative_z**2
        distance_fifth_power = distance_squared**2.5 if distance_squared > 1e-12 else 1e12

        axial_projection = (
            direction_x * relative_x + direction_y * relative_y + direction_z * relative_z
        )
        vorticity_factor = -doublet_strength / (4 * np.pi * distance_fifth_power)

        vorticity[i, 0] = vorticity_factor * (
            distance_squared * direction_x - 3 * relative_x * axial_projection
        )
        vorticity[i, 1] = vorticity_factor * (
            distance_squared * direction_y - 3 * relative_y * axial_projection
        )
        vorticity[i, 2] = vorticity_factor * (
            distance_squared * direction_z - 3 * relative_z * axial_projection
        )

    kinematic_viscosity = np.full(n_particles_total, kinematic_viscosity)
    if particle_volume is None:
        particle_volume = np.full(n_particles_total, 1.0)
    vortex_strength = vorticity * particle_volume[:, None]

    return velocity, kinematic_viscosity, vortex_strength


# =========================================================


def taylor_green_vortex_vpm(
    kinematic_viscosity: float,
    box_size: float,
    mean_core_radius: float,
    position: np.ndarray,
    particle_volume: np.ndarray = None,
    time: float = 0.0,
):
    """
    Initialize Taylor-Green vortex flow field for periodic turbulence studies.

    The Taylor-Green vortex is a 3D periodic flow that serves as a canonical
    test case for studying turbulence transition and energy cascade.

    Args:
        kinematic_viscosity: Kinematic viscosity [m²/s]
        box_size: Periodic box size (domain is [0, box_size]³) [m]
        mean_core_radius: Particle core size [m]
        position: Particle position [m]
        particle_volume: Particle volume [m³] (optional)
        time: Initial flow time [s]. The ``decay_factor`` applied at
            ``time > 0`` is the single-mode linear (Stokes) decay of this
            initial condition, not the exact nonlinear 3D Taylor-Green
            solution -- the true field departs from a single Fourier mode as
            soon as the convective term acts. Use ``time=0`` for the
            canonical benchmark initial condition.

    Returns:
        velocity : (N,3) ndarray - Particle velocity [m/s]
        kinematic_viscosity : (N,) ndarray - Particle kinematic viscosity [m²/s]
        vortex_strength : (N,3) ndarray - Particle vortex strength [m³/s]
    """
    n_particles_total = len(position)
    vorticity = np.zeros_like(position)
    velocity = np.zeros_like(position)

    # Scale position to [0, 2π] for standard TG formulation
    wave_number = 2.0 * np.pi / box_size
    phase_x = position[:, 0] * wave_number
    phase_y = position[:, 1] * wave_number
    phase_z = position[:, 2] * wave_number

    # Time-dependent decay factor for viscous effects
    decay_factor = np.exp(-2.0 * kinematic_viscosity * time * wave_number**2) if time > 0 else 1.0

    # Taylor-Green vortex velocity field
    velocity[:, 0] = decay_factor * np.sin(phase_x) * np.cos(phase_y) * np.cos(phase_z)
    velocity[:, 1] = -decay_factor * np.cos(phase_x) * np.sin(phase_y) * np.cos(phase_z)
    velocity[:, 2] = 0.0

    # Taylor-Green vortex vorticity field: omega = curl(u)
    vorticity[:, 0] = (
        -decay_factor * np.cos(phase_x) * np.sin(phase_y) * np.sin(phase_z) * wave_number
    )
    vorticity[:, 1] = (
        -decay_factor * np.sin(phase_x) * np.cos(phase_y) * np.sin(phase_z) * wave_number
    )
    vorticity[:, 2] = (
        2.0 * decay_factor * np.sin(phase_x) * np.sin(phase_y) * np.cos(phase_z) * wave_number
    )

    # Set uniform kinematic_viscosity
    kinematic_viscosity = np.full(n_particles_total, kinematic_viscosity)

    # Convert vorticity to strength (Γ = ω * particle_volume)
    if particle_volume is None:
        particle_volume = np.full(n_particles_total, 1.0)
    vortex_strength = vorticity * particle_volume[:, None]

    return velocity, kinematic_viscosity, vortex_strength


# =========================================================


def isotropic_turbulence_vpm(
    kinematic_viscosity: float,
    box_size: float,
    kinetic_energy_spectrum_peak: float,
    turbulent_intensity: float,
    mean_core_radius: float,
    position: np.ndarray,
    particle_volume: np.ndarray = None,
    seed: int = 42,
    n_fourier_modes: int = 32,
):
    """
    Initialize isotropic turbulence using spectral synthesis for periodic domains.

    Generates a random turbulent velocity field with prescribed energy spectrum
    and statistical properties suitable for homogeneous isotropic turbulence studies.

    Args:
        kinematic_viscosity: Kinematic viscosity [m²/s]
        box_size: Periodic box size (domain is [0, box_size]³) [m]
        kinetic_energy_spectrum_peak: Wavenumber of kinetic-energy spectrum peak [1/m]
        turbulent_intensity: RMS velocity fluctuation magnitude [m/s]
        mean_core_radius: Particle core size [m]
        position: Particle position [m]
        particle_volume: Particle volume [m³] (optional)
        seed: Random seed for reproducibility
        n_fourier_modes: Number of Fourier modes per direction

    Returns:
        velocity : (N,3) ndarray - Particle velocity [m/s]
        kinematic_viscosity : (N,) ndarray - Particle kinematic viscosity [m²/s]
        vortex_strength : (N,3) ndarray - Particle vortex strength [m³/s]
    """
    n_particles_total = len(position)
    velocity = np.zeros_like(position)
    vorticity = np.zeros_like(position)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Scale position to [0, 2π] domain
    wave_number_scale = 2.0 * np.pi / box_size
    phase_x = position[:, 0] * wave_number_scale
    phase_y = position[:, 1] * wave_number_scale
    phase_z = position[:, 2] * wave_number_scale

    # Generate turbulent field using spectral synthesis
    for nx in range(1, n_fourier_modes + 1):
        for ny in range(1, n_fourier_modes + 1):
            for nz in range(1, n_fourier_modes + 1):
                # Wavenumber magnitude
                wave_number_magnitude = np.sqrt(nx**2 + ny**2 + nz**2) * wave_number_scale

                # Skip if wavenumber is too small
                if wave_number_magnitude < 1e-12:
                    continue

                # Energy spectrum (von Kármán-like)
                peak_wave_number = kinetic_energy_spectrum_peak
                energy_density = (wave_number_magnitude / peak_wave_number) ** 4 / (
                    (1 + (wave_number_magnitude / peak_wave_number) ** 2) ** (17 / 6)
                )
                amplitude = turbulent_intensity * np.sqrt(energy_density / wave_number_magnitude**2)

                # Random phases for each velocity component
                velocity_phase_1 = 2.0 * np.pi * np.random.random()
                velocity_phase_2 = 2.0 * np.pi * np.random.random()

                # Wave vector components
                wave_number_x = nx * wave_number_scale
                wave_number_y = ny * wave_number_scale
                wave_number_z = nz * wave_number_scale

                # Fourier mode arguments
                phase_argument = (
                    wave_number_x * phase_x + wave_number_y * phase_y + wave_number_z * phase_z
                )

                # Add velocity contributions (ensuring divergence-free field)
                # Project random amplitudes onto divergence-free subspace
                unit_wave_vector = (
                    np.array([wave_number_x, wave_number_y, wave_number_z]) / wave_number_magnitude
                )

                # Random velocity direction perpendicular to k
                random_direction_1 = np.array(
                    [np.cos(velocity_phase_1), np.sin(velocity_phase_1), 0]
                )
                random_direction_2 = np.array(
                    [0, np.cos(velocity_phase_2), np.sin(velocity_phase_2)]
                )

                # Gram-Schmidt orthogonalization
                transverse_direction_1 = (
                    random_direction_1
                    - np.dot(random_direction_1, unit_wave_vector) * unit_wave_vector
                )
                transverse_direction_1 = transverse_direction_1 / (
                    np.linalg.norm(transverse_direction_1) + 1e-12
                )

                transverse_direction_2 = (
                    random_direction_2
                    - np.dot(random_direction_2, unit_wave_vector) * unit_wave_vector
                    - np.dot(random_direction_2, transverse_direction_1) * transverse_direction_1
                )
                transverse_direction_2 = transverse_direction_2 / (
                    np.linalg.norm(transverse_direction_2) + 1e-12
                )

                # Add velocity contributions
                cosine_mode = np.cos(phase_argument + velocity_phase_1)
                sine_mode = np.sin(phase_argument + velocity_phase_2)

                velocity[:, 0] += amplitude * (
                    transverse_direction_1[0] * cosine_mode + transverse_direction_2[0] * sine_mode
                )
                velocity[:, 1] += amplitude * (
                    transverse_direction_1[1] * cosine_mode + transverse_direction_2[1] * sine_mode
                )
                velocity[:, 2] += amplitude * (
                    transverse_direction_1[2] * cosine_mode + transverse_direction_2[2] * sine_mode
                )

                # Compute vorticity from curl of velocity field
                vorticity[:, 0] += (
                    amplitude
                    * wave_number_magnitude
                    * (
                        (
                            transverse_direction_1[2] * wave_number_y
                            - transverse_direction_1[1] * wave_number_z
                        )
                        * (-np.sin(phase_argument + velocity_phase_1))
                        + (
                            transverse_direction_2[2] * wave_number_y
                            - transverse_direction_2[1] * wave_number_z
                        )
                        * (-np.cos(phase_argument + velocity_phase_2))
                    )
                )
                vorticity[:, 1] += (
                    amplitude
                    * wave_number_magnitude
                    * (
                        (
                            transverse_direction_1[0] * wave_number_z
                            - transverse_direction_1[2] * wave_number_x
                        )
                        * (-np.sin(phase_argument + velocity_phase_1))
                        + (
                            transverse_direction_2[0] * wave_number_z
                            - transverse_direction_2[2] * wave_number_x
                        )
                        * (-np.cos(phase_argument + velocity_phase_2))
                    )
                )
                vorticity[:, 2] += (
                    amplitude
                    * wave_number_magnitude
                    * (
                        (
                            transverse_direction_1[1] * wave_number_x
                            - transverse_direction_1[0] * wave_number_y
                        )
                        * (-np.sin(phase_argument + velocity_phase_1))
                        + (
                            transverse_direction_2[1] * wave_number_x
                            - transverse_direction_2[0] * wave_number_y
                        )
                        * (-np.cos(phase_argument + velocity_phase_2))
                    )
                )

    # Set uniform kinematic_viscosity
    kinematic_viscosity = np.full(n_particles_total, kinematic_viscosity)

    # Convert vorticity to strength
    if particle_volume is None:
        particle_volume = np.full(n_particles_total, 1.0)
    vortex_strength = vorticity * particle_volume[:, None]

    return velocity, kinematic_viscosity, vortex_strength
