import numpy as np
from numba import njit, prange

epsilon = 1e-8  # Regularization parameter

# ==========================================================
#
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_induced_velocity_at(
    target_positions: np.ndarray,
    particle_positions: np.ndarray,
    particle_strengths: np.ndarray,
    particle_radii: np.ndarray
) -> np.ndarray:
    """  
    Calculate induced velocity at specified target positions using a Regularized kernel.
    
    Parameters
    ----------
    particle_positions : np.ndarray
        Positions of source particles, shape (N, 3).
    particle_strengths : np.ndarray
        Strengths of source particles, shape (N, 3), units (m²/s).
    particle_radii : np.ndarray
        Radii of source particles, shape (N,), units (m).
    target_positions : np.ndarray
        Positions of target points where velocities are calculated, shape (M, 3).
    
    Returns
    -------
    np.ndarray
        Induced velocities at each target, shape (M, 3), units (m/s).
    """
    num_sources = len(particle_positions)
    num_targets = len(target_positions)
    one_over_four_pi = 1 / (4 * np.pi)
    
    velocities = np.zeros((num_targets, 3))
    
    for i in prange(num_targets):
        
        target_loc = target_positions[i]
        ux, uy, uz = 0.0, 0.0, 0.0

        for j in range(num_sources):
            
            r_ij = target_loc - particle_positions[j]
            r_ij_sq = np.dot(r_ij, r_ij)  # Efficient dot product for distance squared
                
            if r_ij_sq > epsilon:
                    
                sigma = particle_radii[j]
                sigma_sq = sigma * sigma
                rho_sq = r_ij_sq / sigma_sq

                # Compute the kernel term
                q_sigma_over_r3 = (rho_sq + 2.5) / ((rho_sq + 1)**2.5 * sigma * sigma_sq)

                # Compute the cross product (manual implementation for efficiency)
                cross_product = (
                    r_ij[1] * particle_strengths[j][2] - r_ij[2] * particle_strengths[j][1],
                    r_ij[2] * particle_strengths[j][0] - r_ij[0] * particle_strengths[j][2],
                    r_ij[0] * particle_strengths[j][1] - r_ij[1] * particle_strengths[j][0]
                )
                
                # Accumulate the velocity components
                ux -= q_sigma_over_r3 * cross_product[0]
                uy -= q_sigma_over_r3 * cross_product[1]
                uz -= q_sigma_over_r3 * cross_product[2]
            
        velocities[i] = ux, uy, uz
    
    return velocities * one_over_four_pi



# ==========================================================
#
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_selfinduced_velocity(
    particle_positions: np.ndarray,
    particle_strengths: np.ndarray,
    particle_radii: np.ndarray
) -> np.ndarray:
    """  
    Calculate self-induced velocity for each particle using a Regularized kernel.
    
    Parameters
    ----------
    particle_positions : np.ndarray
        Positions of particles, shape (N, 3).
    particle_strengths : np.ndarray
        Vorticity strengths of particles, shape (N, 3), units (m²/s).
    particle_radii : np.ndarray
        Radii of particles, shape (N,), units (m).
    
    Returns
    -------
    np.ndarray
        Self-induced velocities at each particle, shape (N, 3), units (m/s).
    """
    num_particles = len(particle_positions)
    one_over_four_pi = 1 / (4 * np.pi)
    velocities = np.zeros((num_particles, 3))

    # Precompute some constants outside the loop for efficiency
    for i in prange(num_particles):
        ux, uy, uz = 0.0, 0.0, 0.0
        target_loc = particle_positions[i]
        target_radius = particle_radii[i]

        # Precompute sigma for particle i (for better memory locality)
        sigma = target_radius  # The target radius is constant for each particle
        sigma_sq = sigma * sigma

        for j in range(num_particles):
            if i != j:
                r_ij = target_loc - particle_positions[j]
                r_ij_sq = np.dot(r_ij, r_ij)  # Efficient distance squared calculation

                # Precompute rho_sq
                rho_sq = r_ij_sq / sigma_sq

                # Kernel function
                q_sigma_over_r3 = (rho_sq + 2.5) / ((rho_sq + 1)**2.5 * sigma * sigma_sq)

                # Cross product (manual implementation for speed)
                cross_product = (
                    r_ij[1] * particle_strengths[j][2] - r_ij[2] * particle_strengths[j][1],
                    r_ij[2] * particle_strengths[j][0] - r_ij[0] * particle_strengths[j][2],
                    r_ij[0] * particle_strengths[j][1] - r_ij[1] * particle_strengths[j][0]
                )

                # Accumulate velocity components
                ux -= q_sigma_over_r3 * cross_product[0]
                uy -= q_sigma_over_r3 * cross_product[1]
                uz -= q_sigma_over_r3 * cross_product[2]

        velocities[i] = ux, uy, uz

    return velocities * one_over_four_pi



# ==========================================================
# 
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_velocity_gradient_field_at(
    target_positions: np.ndarray,
    particles_positions: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> np.ndarray:
    """
    Calculates the vorticity exchange of particles (DOmega_DT) based on stretching.

    Parameters:
    -----------
    target_positions: np.ndarray, shape (M, 3) - Target points.
    particles_positions: np.ndarray, shape (N, 3) - Positions of particles.
    particles_strengths: np.ndarray, shape (N, 3) - Vorticity strengths of particles (m²/s).
    particles_radii: np.ndarray, shape (N,) - Radii of particles (m).

    Returns:
    --------
    np.ndarray, shape (M, 3) - Vortex stretching contributions to vorticity exchange (m²/s).
    """
    num_targets = len(target_positions)
    num_sources = len(particles_positions)
    dGamma_dt = np.zeros((num_targets, 3), dtype=np.float64)
    one_over_four_pi = 1 / (4 * np.pi)

    # Loop over each target particle in parallel
    for i in prange(num_targets):
        
        target_loc = target_positions[i]

        for j in prange(num_sources):

                # Precompute repetitive terms
                r_ij = target_loc - particles_positions[j]
                r_ij_sq = np.dot(r_ij, r_ij)  # Efficient distance squared calculation
                
                if r_ij_sq > epsilon:
                    
                    
                    sigma = particles_radii[j]
                    sigma_sq = sigma * sigma
                    sigma_cb = sigma_sq * sigma
                    rho_sq_plus_one = r_ij_sq / sigma_sq + 1.0

                    # Kernel factors
                    factor1 = (rho_sq_plus_one + 1.5) / (sigma_cb * rho_sq_plus_one ** 2.5)
                    factor2 = 3.0 * (rho_sq_plus_one + 2.5) / (sigma_cb * sigma_sq * rho_sq_plus_one ** 3.5)

                    # Cross product
                    cross_product = np.cross(r_ij, particles_strengths[j])

                    # Update dGamma_dt for each target
                    dGamma_dt[i] += factor1 * particles_strengths[j] + factor2 * cross_product * r_ij

    return dGamma_dt * one_over_four_pi


# ==========================================================
# CONTINUE FROM HERE
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_vorticity_field_at(
    target_positions: np.ndarray,
    particles_position: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> np.ndarray:
    """
    Calculate the induced vorticity at each target position using a regularized kernel.

    Parameters:
    -----------
    target_positions: np.ndarray, shape (M, 3) - Target points.
    particles_position: np.ndarray, shape (N, 3) - Source positions.
    particles_strengths: np.ndarray, shape (N, 3) - Strengths of sources (m²/s).
    particles_radii: np.ndarray, shape (N,) - Radii of sources (m).

    Returns:
    --------
    np.ndarray, shape (M, 3) - Induced vorticities at each target (m/s).
    """
    num_particles, num_targets = len(particles_position), len(target_positions)
    one_over_four_pi = 1 / (4 * np.pi)
    vorticities = np.zeros((num_targets, 3), dtype=np.float64)

    for i in prange(num_targets):
        vorticity, target_loc = np.zeros(3, dtype=np.float64), target_positions[i]

        for j in prange(num_particles):
            
            source_loc = particles_position[j]
            source_radius = particles_radii[j]
            source_strength = particles_strengths[j]
            r_ij = target_loc - source_loc
            r_ij_sq = np.dot(r_ij, r_ij)
            
            if r_ij_sq > epsilon:
                
                rho2 = r_ij_sq / (source_radius ** 2)

                source_radius3 = source_radius ** 3
                xi_sigma = 7.5 / ((rho2 + 1) ** 3.5 * source_radius3)
                q_sigma_over_r3 = (rho2 + 2.5) / ((rho2 + 1) ** 2.5 * source_radius3)

                term1 = xi_sigma - q_sigma_over_r3
                term2 = (3.0 * q_sigma_over_r3 - xi_sigma) * np.dot(r_ij, source_strength) / r_ij_sq

                vorticity += term1 * source_strength + term2 * r_ij

        vorticities[i] = vorticity * one_over_four_pi

    return vorticities


# ==========================================================
#
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_strength_gradients(
    particles_position: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> np.ndarray:
    """
    Calculates the vorticity exchange of particles (DOmega_DT) based on stretching.

    Parameters:
    -----------
    particles_position: np.ndarray, shape (N, 3) - Positions of particles.
    particles_strengths: np.ndarray, shape (N, 3) - Vorticity strengths of particles (m²/s).
    particles_radii: np.ndarray, shape (N,) - Radii of particles (m).

    Returns:
    --------
    np.ndarray, shape (N, 3) - Vortex stretching contributions to vorticity exchange (m²/s).
    """
    num_particles = len(particles_position)
    dGamma_dt = np.zeros((num_particles, 3), dtype=np.float64)
    one_over_four_pi = 1 / (4 * np.pi)

    for i in prange(num_particles):
        radius_i, position_i, strength_i = particles_radii[i], particles_position[i], particles_strengths[i]

        for j in prange(num_particles):
            if j!=i:
                radius_j, position_j, strength_j = particles_radii[j], particles_position[j], particles_strengths[j]
                r_ij = position_i - position_j
                distance_sq = np.dot(r_ij, r_ij)
                sigma = (radius_i + radius_j) * 0.5
                sigma_cb = sigma ** 3
                rho_sq_plus_one = distance_sq / sigma ** 2 + 1

                factor1 = (rho_sq_plus_one + 1.5) / (sigma_cb * rho_sq_plus_one ** 2.5)
                factor2 = 3.0 * (rho_sq_plus_one + 2.5) / (sigma_cb * sigma ** 2 * rho_sq_plus_one ** 3.5)

                strength_product1 = np.cross(strength_i, strength_j)
                strength_product2 = np.cross(r_ij, strength_j)
                strength_product3 = np.dot(strength_i, strength_product2)

                dGamma_dt[i] += factor1 * strength_product1 + factor2 * strength_product3 * r_ij

    return dGamma_dt * one_over_four_pi


# ==========================================================
#
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_strength_gradients_PSE(
    particles_positions: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray,
    particles_viscosities: np.ndarray
) -> np.ndarray:
    """
    Calculates vorticity exchange with stretching and viscous diffusion.

    Parameters:
    -----------
    particles_positions: np.ndarray, shape (N, 3) - Positions of particles.
    particles_strengths: np.ndarray, shape (N, 3) - Vorticity strengths of particles (m²/s).
    particles_radii: np.ndarray, shape (N,) - Radii of particles (m).
    particles_nu: np.ndarray, shape (N,) - Kinematic viscosities (m²/s).

    Returns:
    --------
    np.ndarray, shape (N, 3) - Vortex stretching and viscous diffusion contributions.
    """
    num_particles = len(particles_positions)
    dGamma_dt = np.zeros((num_particles, 3), dtype=np.float64)
    one_over_four_pi = 1 / (4 * np.pi)

    for i in prange(num_particles):
        radius_i   = particles_radii[i]
        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        viscosity_i = particles_viscosities[i]

        acc_dGamma_dt_x = 0.0
        acc_dGamma_dt_y = 0.0
        acc_dGamma_dt_z = 0.0

        for j in range(num_particles):
            if j!=i:
                radius_j    = particles_radii[j]
                position_j  = particles_positions[j]
                strength_j  = particles_strengths[j]
                viscosity_j = particles_viscosities[j]

                viscosity = (viscosity_i + viscosity_j) * 0.5
                sigma = (radius_i + radius_j) * 0.5

                r_ij = position_i - position_j
                distance_sq = np.dot(r_ij, r_ij)
                
                sigma_sq = sigma * sigma
                sigma_cb = sigma_sq * sigma

                rho_sq_plus_one = (distance_sq / sigma_sq) + 1.0 + epsilon

                factor3 = 105.0 * viscosity / (sigma_sq * sigma_cb * rho_sq_plus_one**4.5)

                acc_dGamma_dt_x += factor3 * sigma_cb * (strength_j[0] - strength_i[0])
                acc_dGamma_dt_y += factor3 * sigma_cb * (strength_j[1] - strength_i[1])
                acc_dGamma_dt_z += factor3 * sigma_cb * (strength_j[2] - strength_i[2])
        
        dGamma_dt[i,0] = acc_dGamma_dt_x
        dGamma_dt[i,1] = acc_dGamma_dt_y
        dGamma_dt[i,2] = acc_dGamma_dt_z


    return dGamma_dt * one_over_four_pi


# ==========================================================
#
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_total_kinetic_energy(
    particles_positions: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> float:

    num_particles = len(particles_positions)
    E = 0.0

    for i in prange(num_particles):

        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i   = particles_radii[i]

        for j in range(num_particles):
            if i != j:
                position_j = particles_positions[j]
                strength_j = particles_strengths[j]
                radius_j   = particles_radii[j]

                r_ij = position_i - position_j
                r_ij_norm = (r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2])**0.5
                
                r_ij_unit =  r_ij / r_ij_norm

                sigma = (radius_i + radius_j) * 0.5
                
                rho = r_ij_norm / sigma
                rho_sq_plus_one = rho*rho + 1.0 + epsilon

                term_1 = 2. * rho / rho_sq_plus_one**0.5
                term_2 = rho*rho*rho / rho_sq_plus_one**1.5
                
                product_1 = np.dot(strength_i, strength_j)
                product_2 = np.dot(r_ij_unit, strength_i)
                product_3 = np.dot(r_ij_unit, strength_j) 

                E += ( term_1*product_1 + term_2 * ( product_2 * product_3 - product_1 ) ) / r_ij_norm

    return E / (16 * np.pi)


@njit(parallel=True, fastmath=True)
def get_total_helicity(
    particles_positions: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> float:

    num_particles = len(particles_positions)
    H = 0.0

    for i in prange(num_particles):

        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i   = particles_radii[i]

        for j in range(num_particles):
            if i != j:
                position_j = particles_positions[j]
                strength_j = particles_strengths[j]
                radius_j   = particles_radii[j]

                r_ij = position_i - position_j
                r_ij_mag = (r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2])**0.5
                
                sigma = (radius_i + radius_j) * 0.5
                rho = r_ij_mag / sigma
                rho_sq = rho*rho

                term = (rho_sq + 2.5) / (sigma*sigma*sigma * (rho_sq + 1.0)**2.5)

                product_1 = np.cross(strength_i, strength_j)
                product_2 = np.dot(r_ij, product_1)
 
                H += product_2 * term

    return H / (4. * np.pi)

# ==========================================================
#
# ==========================================================
@njit(parallel=True, fastmath=True)
def get_total_enstrophy_long(
    particles_positions: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> float:

    num_particles = len(particles_positions)
    Ens = 0.0

    for i in prange(num_particles):

        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i   = particles_radii[i]

        for j in range(num_particles):
            position_j = particles_positions[j]
            strength_j = particles_strengths[j]
            radius_j   = particles_radii[j]

            r_ij = position_i - position_j
            r_ij_norm = (r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2])**0.5
            
            sigma = (radius_i + radius_j) * 0.5
            sigma_cb = sigma*sigma*sigma
            
            rho = r_ij_norm / sigma
            rho_sq = rho*rho
            rho_sq_plus_one = rho_sq + 1.0 + epsilon

            term_1 = (5.0 - rho_sq * (rho_sq + 3.5)) / rho_sq_plus_one**3.5
            term_2 = ((rho_sq * (rho_sq + 4.5) + 3.5)) / rho_sq_plus_one**4.5
            
            product_1 = np.dot(strength_i, strength_j)
            product_2 = np.dot(r_ij, strength_i)
            product_3 = np.dot(r_ij, strength_j) 
        
            Ens += ( term_1*product_1 + 3 * term_2 * product_2 * product_3 ) / sigma_cb

    return Ens / (4 * np.pi)



@njit(parallel=True, fastmath=True)
def get_total_enstrophy(
    particles_positions: np.ndarray,
    particles_strengths: np.ndarray,
    particles_radii: np.ndarray
) -> float:

    num_particles = len(particles_positions)
    Ens = 0.0

    for i in prange(num_particles):

        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i   = particles_radii[i]

        for j in range(num_particles):
            position_j = particles_positions[j]
            strength_j = particles_strengths[j]
            radius_j   = particles_radii[j]

            r_ij = position_i - position_j
            r_ij_sq = (r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2])
            
            sigma = (radius_i + radius_j) * 0.5
            sigma_sq = sigma*sigma
            
            term_1 = (r_ij_sq + 3.2*sigma_sq) / (r_ij_sq + sigma_sq)**1.5

            product_1 = np.dot(strength_i, strength_j)

            Ens += term_1*product_1

    return Ens / (8 * np.pi)