"""
Analytic flow-field generators for VPM initialization and validation: Lamb-Oseen,
vortex ring, doublet, Taylor-Green, and isotropic turbulence.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np

# =========================================================

def LambOseenVPM(
    viscosity: float,
    avg_particle_radius: float,
    positions: np.ndarray,
    volumes: np.ndarray,
    vortex_center: np.ndarray = np.array([0.0, 0.0, 0.0]),
    vortex_strength: float = np.pi,
    vortex_time: float = 1.0,
    disturb_amp: float = 0.0,
    max_diturb_modes: int = 12,
    anti_diffuse_flag: bool = False,
):
    """Initialize Lamb-Oseen vortex field on particles.

    Parameters:
        viscosity: Kinematic viscosity [m²/s]
        avg_particle_radius: Average particle core radius [m]
        positions: Particle positions (N, 3)
        volumes: Particle volumes (N,)
        vortex_center: Vortex center position [m]
        vortex_strength: Total circulation [m²/s]
        vortex_time: Vortex age [s] (for viscous diffusion)
        disturb_amp: Perturbation amplitude
        max_diturb_modes: Number of perturbation modes
        anti_diffuse_flag: Compensate for particle core diffusion

    Returns:
        velocities: (N, 3) ndarray
        viscosities: (N,) ndarray
        strengths: (N, 3) ndarray
    """
    num_particles = len(positions)
    vorticities = np.zeros_like(positions)
    velocities = np.zeros_like(positions)

    # Handle inviscid case (viscosity=0)
    if viscosity <= 0.0:
        # Use particle radius as core size for inviscid vortex
        a_sq = avg_particle_radius**2
    else:
        # Adjust vortex_time for anti-diffusion
        # Gaussian particles use exp(-r^2/sigma^2), so their blob variance adds
        # an effective age shift sigma^2/(4*nu) to the represented Lamb-Oseen core.
        t_shift = (avg_particle_radius**2 / (4.0 * viscosity)) if anti_diffuse_flag else 0.0
        if t_shift > vortex_time:
            raise ValueError(
                "Invalid parameters: vortex_time too small for anti-diffusion correction."
            )
        vortex_time_eff = vortex_time - t_shift
        # Compute core size squared from vortex_time
        a_sq = 4.0 * viscosity * vortex_time_eff

    # Add perturbation to the vortex center
    z_vals = positions[:, 2]
    phases = 2 * np.pi * np.random.rand(max_diturb_modes)
    disturb = np.zeros_like(z_vals)
    for n in range(1, max_diturb_modes + 1):
        disturb += np.cos(n * z_vals + phases[n - 1])
    disturb /= np.sqrt(max_diturb_modes)

    perturbed_x = vortex_center[0] + disturb_amp * disturb
    perturbed_y = vortex_center[1] + disturb_amp * disturb

    dx = positions[:, 0] - perturbed_x
    dy = positions[:, 1] - perturbed_y
    r = np.sqrt(dx**2 + dy**2)

    for i in range(num_particles):
        exp_term = np.exp(-(r[i] ** 2) / a_sq)
        vorticities[i, 2] = (vortex_strength / (np.pi * a_sq)) * exp_term
        if r[i] > 1e-12:
            u_theta = (vortex_strength / (2 * np.pi * r[i])) * (1.0 - exp_term)
            velocities[i, 0] = -u_theta * dy[i] / r[i]
            velocities[i, 1] = u_theta * dx[i] / r[i]

    viscosities = np.full(num_particles, max(viscosity, 0.0))
    strengths = vorticities * volumes[:, None]

    return velocities, viscosities, strengths

# =========================================================

def VortexRingVPM(
    viscosity: float,
    ring_center: np.ndarray,
    ring_strength: float,
    ring_radius: float,
    ring_thickness: float,
    avg_particle_radius: float,
    positions: np.ndarray,
    volumes: np.ndarray = None,
    epsilon_W: float = 0.0,
    seed: int = 42,
    max_modes: int = 24,
    anti_diffuse_flag: bool = False,
):
    """
    Initialize vortex ring with optional Widnall instability perturbation.

    Returns:
    --------
    velocities : (N,3) ndarray
    viscosities : (N,) ndarray
    strengths : (N,3) ndarray
    """
    num_particles = len(positions)
    t_shift = avg_particle_radius**2 / (viscosity * 4.0) if anti_diffuse_flag else 0.0

    t0 = ring_thickness**2 / (4 * viscosity)
    if t_shift > t0:
        raise ValueError("Invalid parameters: actual_core_radius would be imaginary")
    actual_ring_thickness_sq = 4 * viscosity * (t0 - t_shift)

    X = positions[:, 0] - ring_center[0]
    Y = positions[:, 1] - ring_center[1]
    Z = positions[:, 2] - ring_center[2]
    theta = np.arctan2(Z, Y)

    np.random.seed(seed)
    phases = 2 * np.pi * np.random.rand(max_modes)
    g_theta = np.zeros_like(theta)
    for n in range(1, max_modes + 1):
        g_theta += np.cos(n * theta + phases[n - 1])
    g_theta /= np.sqrt(max_modes)

    radial_dist = np.sqrt(Y**2 + Z**2)
    radial_dist_perturbed = radial_dist * (1 + epsilon_W * g_theta)
    Y = radial_dist_perturbed * np.cos(theta)
    Z = radial_dist_perturbed * np.sin(theta)
    core_dist = np.sqrt((radial_dist_perturbed - ring_radius) ** 2 + X**2)

    omega_mag = (ring_strength / (np.pi * actual_ring_thickness_sq)) * np.exp(
        -(core_dist**2) / actual_ring_thickness_sq
    )
    vorticities = np.zeros_like(positions)
    vorticities[:, 1] = -omega_mag * np.sin(theta)
    vorticities[:, 2] = omega_mag * np.cos(theta)

    velocities = np.zeros_like(positions)
    viscosities = np.full(num_particles, viscosity)
    if volumes is None:
        volumes = np.full(num_particles, 1.0)
    strengths = vorticities * volumes[:, None]

    return velocities, viscosities, strengths

# =========================================================

def DoubletFlowVPM(
    viscosity: float,
    center: np.ndarray,
    direction: np.ndarray,
    kappa: float,
    positions: np.ndarray,
    volumes: np.ndarray = None,
):
    """
    Generate a vorticity field corresponding to a 3D doublet (vortex dipole) flow using particles.

    Returns:
    --------
    velocities : (N,3) ndarray
    viscosities : (N,) ndarray
    strengths : (N,3) ndarray
    """
    num_particles = positions.shape[0]
    vorticities = np.zeros_like(positions)
    velocities = np.zeros_like(positions)

    dx = direction[0]
    dy = direction[1]
    dz = direction[2]
    norm_dir = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm_dir
    dy /= norm_dir
    dz /= norm_dir

    for i in range(num_particles):
        x = positions[i, 0] - center[0]
        y = positions[i, 1] - center[1]
        z = positions[i, 2] - center[2]

        r2 = x**2 + y**2 + z**2
        r5 = r2**2.5 if r2 > 1e-12 else 1e12

        dot = dx * x + dy * y + dz * z
        factor = -kappa / (4 * np.pi * r5)

        vorticities[i, 0] = factor * (r2 * dx - 3 * x * dot)
        vorticities[i, 1] = factor * (r2 * dy - 3 * y * dot)
        vorticities[i, 2] = factor * (r2 * dz - 3 * z * dot)

    viscosities = np.full(num_particles, viscosity)
    if volumes is None:
        volumes = np.full(num_particles, 1.0)
    strengths = vorticities * volumes[:, None]

    return velocities, viscosities, strengths

# =========================================================

def TaylorGreenVortexVPM(
    viscosity: float,
    box_size: float,
    avg_particle_radius: float,
    positions: np.ndarray,
    volumes: np.ndarray = None,
    flow_time: float = 0.0,
):
    """
    Initialize Taylor-Green vortex flow field for periodic turbulence studies.

    The Taylor-Green vortex is a 3D periodic flow that serves as a canonical
    test case for studying turbulence transition and energy cascade.

    Args:
        viscosity: Kinematic viscosity [m²/s]
        box_size: Periodic box size (domain is [0, box_size]³) [m]
        avg_particle_radius: Particle core size [m]
        positions: Particle positions [m]
        volumes: Particle volumes [m³] (optional)
        flow_time: Initial flow time for decay calculation [s]

    Returns:
        velocities : (N,3) ndarray - Particle velocities [m/s]
        viscosities : (N,) ndarray - Particle viscosities [m²/s]
        strengths : (N,3) ndarray - Particle vorticity strengths [m³/s]
    """
    num_particles = len(positions)
    vorticities = np.zeros_like(positions)
    velocities = np.zeros_like(positions)

    # Scale positions to [0, 2π] for standard TG formulation
    scale = 2.0 * np.pi / box_size
    x = positions[:, 0] * scale
    y = positions[:, 1] * scale
    z = positions[:, 2] * scale

    # Time-dependent decay factor for viscous effects
    decay_factor = np.exp(-2.0 * viscosity * flow_time * scale**2) if flow_time > 0 else 1.0

    # Taylor-Green vortex velocity field
    velocities[:, 0] = decay_factor * np.sin(x) * np.cos(y) * np.cos(z)  # u
    velocities[:, 1] = -decay_factor * np.cos(x) * np.sin(y) * np.cos(z)  # v
    velocities[:, 2] = 0.0  # w

    # Taylor-Green vortex vorticity field
    vorticities[:, 0] = decay_factor * np.cos(x) * np.cos(y) * np.sin(z) * scale  # ωx
    vorticities[:, 1] = decay_factor * np.sin(x) * np.sin(y) * np.sin(z) * scale  # ωy
    vorticities[:, 2] = -2.0 * decay_factor * np.sin(x) * np.cos(y) * np.cos(z) * scale  # ωz

    # Set uniform viscosity
    viscosities = np.full(num_particles, viscosity)

    # Convert vorticity to strength (Γ = ω * volume)
    if volumes is None:
        volumes = np.full(num_particles, 1.0)
    strengths = vorticities * volumes[:, None]

    return velocities, viscosities, strengths

# =========================================================

def IsotropicTurbulenceVPM(
    viscosity: float,
    box_size: float,
    energy_spectrum_peak: float,
    turbulent_intensity: float,
    avg_particle_radius: float,
    positions: np.ndarray,
    volumes: np.ndarray = None,
    seed: int = 42,
    num_modes: int = 32,
):
    """
    Initialize isotropic turbulence using spectral synthesis for periodic domains.

    Generates a random turbulent velocity field with prescribed energy spectrum
    and statistical properties suitable for homogeneous isotropic turbulence studies.

    Args:
        viscosity: Kinematic viscosity [m²/s]
        box_size: Periodic box size (domain is [0, box_size]³) [m]
        energy_spectrum_peak: Wavenumber of energy spectrum peak [1/m]
        turbulent_intensity: RMS velocity fluctuation magnitude [m/s]
        avg_particle_radius: Particle core size [m]
        positions: Particle positions [m]
        volumes: Particle volumes [m³] (optional)
        seed: Random seed for reproducibility
        num_modes: Number of Fourier modes per direction

    Returns:
        velocities : (N,3) ndarray - Particle velocities [m/s]
        viscosities : (N,) ndarray - Particle viscosities [m²/s]
        strengths : (N,3) ndarray - Particle vorticity strengths [m³/s]
    """
    num_particles = len(positions)
    velocities = np.zeros_like(positions)
    vorticities = np.zeros_like(positions)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Scale positions to [0, 2π] domain
    scale = 2.0 * np.pi / box_size
    x = positions[:, 0] * scale
    y = positions[:, 1] * scale
    z = positions[:, 2] * scale

    # Generate turbulent field using spectral synthesis
    for nx in range(1, num_modes + 1):
        for ny in range(1, num_modes + 1):
            for nz in range(1, num_modes + 1):
                # Wavenumber magnitude
                k_mag = np.sqrt(nx**2 + ny**2 + nz**2) * scale

                # Skip if wavenumber is too small
                if k_mag < 1e-12:
                    continue

                # Energy spectrum (von Kármán-like)
                k_peak = energy_spectrum_peak
                energy_density = (k_mag / k_peak) ** 4 / ((1 + (k_mag / k_peak) ** 2) ** (17 / 6))
                amplitude = turbulent_intensity * np.sqrt(energy_density / k_mag**2)

                # Random phases for each velocity component
                phi_u = 2.0 * np.pi * np.random.random()
                phi_v = 2.0 * np.pi * np.random.random()
                2.0 * np.pi * np.random.random()

                # Wave vector components
                kx, ky, kz = nx * scale, ny * scale, nz * scale

                # Fourier mode arguments
                arg = kx * x + ky * y + kz * z

                # Add velocity contributions (ensuring divergence-free field)
                # Project random amplitudes onto divergence-free subspace
                k_hat = np.array([kx, ky, kz]) / k_mag

                # Random velocity direction perpendicular to k
                rand_dir1 = np.array([np.cos(phi_u), np.sin(phi_u), 0])
                rand_dir2 = np.array([0, np.cos(phi_v), np.sin(phi_v)])

                # Gram-Schmidt orthogonalization
                dir1 = rand_dir1 - np.dot(rand_dir1, k_hat) * k_hat
                dir1 = dir1 / (np.linalg.norm(dir1) + 1e-12)

                dir2 = rand_dir2 - np.dot(rand_dir2, k_hat) * k_hat - np.dot(rand_dir2, dir1) * dir1
                dir2 = dir2 / (np.linalg.norm(dir2) + 1e-12)

                # Add velocity contributions
                cos_arg = np.cos(arg + phi_u)
                sin_arg = np.sin(arg + phi_v)

                velocities[:, 0] += amplitude * (dir1[0] * cos_arg + dir2[0] * sin_arg)
                velocities[:, 1] += amplitude * (dir1[1] * cos_arg + dir2[1] * sin_arg)
                velocities[:, 2] += amplitude * (dir1[2] * cos_arg + dir2[2] * sin_arg)

                # Compute vorticity from curl of velocity field
                vorticities[:, 0] += (
                    amplitude
                    * k_mag
                    * (
                        (dir1[2] * ky - dir1[1] * kz) * (-np.sin(arg + phi_u))
                        + (dir2[2] * ky - dir2[1] * kz) * (-np.cos(arg + phi_v))
                    )
                )
                vorticities[:, 1] += (
                    amplitude
                    * k_mag
                    * (
                        (dir1[0] * kz - dir1[2] * kx) * (-np.sin(arg + phi_u))
                        + (dir2[0] * kz - dir2[2] * kx) * (-np.cos(arg + phi_v))
                    )
                )
                vorticities[:, 2] += (
                    amplitude
                    * k_mag
                    * (
                        (dir1[1] * kx - dir1[0] * ky) * (-np.sin(arg + phi_u))
                        + (dir2[1] * kx - dir2[0] * ky) * (-np.cos(arg + phi_v))
                    )
                )

    # Set uniform viscosity
    viscosities = np.full(num_particles, viscosity)

    # Convert vorticity to strength
    if volumes is None:
        volumes = np.full(num_particles, 1.0)
    strengths = vorticities * volumes[:, None]

    return velocities, viscosities, strengths
