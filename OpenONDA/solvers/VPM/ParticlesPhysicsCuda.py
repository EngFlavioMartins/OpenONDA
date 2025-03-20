import numpy as np
from numba import cuda, float64
from numpy import float64

threads_per_block = 512
epsilon = 1e-8  # Regularization parameter

# ==========================================================
# Induced velocity calculation
# ==========================================================
def get_induced_velocity_at(target_positions, particles_positions, particles_strengths, particles_radii):

    blocks_per_grid = (len(target_positions) + threads_per_block - 1) // threads_per_block

    # Allocate GPU memory
    velocities_gpu = cuda.device_array_like(target_positions)

    # Batch data transfer
    target_positions_gpu = cuda.to_device(target_positions)
    particles_positions_gpu = cuda.to_device(particles_positions)
    particles_strengths_gpu = cuda.to_device(particles_strengths)
    particles_radii_gpu = cuda.to_device(particles_radii)

    # Launch kernel
    get_induced_velocity_at_kernel[blocks_per_grid, threads_per_block](
        target_positions_gpu, 
        particles_positions_gpu, 
        particles_strengths_gpu, 
        particles_radii_gpu, 
        velocities_gpu
    )

    # Copy result back to host (implicit synchronization)
    return velocities_gpu.copy_to_host()


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:])")
def get_induced_velocity_at_kernel(target_positions: np.float64, particles_positions: np.float64, particles_strengths: np.float64, particles_radii: np.float64, target_velocities: np.float64):
    """
    GPU-accelerated function to compute induced velocity for each target position due to source particles.
    """

    i = cuda.grid(1)
    num_targets   = target_positions.shape[0]
    num_particles = particles_positions.shape[0]
    one_over_four_pi = 1 / (4 * np.pi)
    
    # Local storage
    cross_product = cuda.local.array(3, dtype=np.float64)
    target_loc = cuda.local.array(3, dtype=np.float64)  
    source_loc = cuda.local.array(3, dtype=np.float64)  
    r_ij       = cuda.local.array(3, dtype=np.float64)

    if i < num_targets:
        ux, uy, uz = 0.0, 0.0, 0.0

        # Copy target particle's position to local storage
        target_loc[0] = target_positions[i, 0]
        target_loc[1] = target_positions[i, 1] 
        target_loc[2] = target_positions[i, 2]

        # Loop over each source particle
        for j in range(num_particles):
            # Copy source particle position
            source_loc[0] = particles_positions[j, 0]
            source_loc[1] = particles_positions[j, 1]
            source_loc[2] = particles_positions[j, 2]
            source_strength = particles_strengths[j]
            sigma = particles_radii[j]

            # Compute relative position r_ij
            r_ij[0] = target_loc[0] - source_loc[0]
            r_ij[1] = target_loc[1] - source_loc[1]
            r_ij[2] = target_loc[2] - source_loc[2]
            
            r_ij_sq = r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]
            
            if r_ij_sq > epsilon:

                sigma_sq = sigma*sigma
                rho2 = r_ij_sq / sigma_sq
                q_sigma_over_r3 = (rho2 + 2.5) / ((rho2 + 1)**2.5 * sigma_sq*sigma)

                cross_product[0] = r_ij[1] * source_strength[2] - r_ij[2] * source_strength[1]
                cross_product[1] = r_ij[2] * source_strength[0] - r_ij[0] * source_strength[2]
                cross_product[2] = r_ij[0] * source_strength[1] - r_ij[1] * source_strength[0]

                ux -= q_sigma_over_r3 * cross_product[0]
                uy -= q_sigma_over_r3 * cross_product[1]
                uz -= q_sigma_over_r3 * cross_product[2]

        # Write result to global memory
        target_velocities[i, 0] = ux * one_over_four_pi
        target_velocities[i, 1] = uy * one_over_four_pi
        target_velocities[i, 2] = uz * one_over_four_pi


# ==========================================================
#
# ==========================================================
def get_selfinduced_velocity(particles_positions: np.float64, particles_strengths: np.float64, particles_radii: np.float64):
    particles_velocities = np.zeros_like(particles_positions, dtype=np.float64)

    blocks_per_grid = (len(particles_positions) + threads_per_block - 1) // threads_per_block

    # Allocate GPU memory
    velocities_gpu = cuda.device_array_like(particles_positions)
    
    # Batch data transfer
    particles_positions_gpu = cuda.to_device(particles_positions)
    particles_strengths_gpu = cuda.to_device(particles_strengths)
    particles_radii_gpu = cuda.to_device(particles_radii)

    # Run GPU kernel to compute half-step velocities
    get_selfinduced_velocity_kernel[blocks_per_grid, threads_per_block](
        particles_positions_gpu, 
        particles_strengths_gpu, 
        particles_radii_gpu, 
        velocities_gpu
    )
    
    # Copy results back to host
    particles_velocities = velocities_gpu.copy_to_host()

    return particles_velocities



# ===================== + ===================== #
       
@cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64[:,:])")
def get_selfinduced_velocity_kernel(particles_positions: np.float64, particles_strengths: np.float64, particles_radii: np.float64, velocities: np.float64):
    """
    GPU-accelerated function to compute self-induced velocities for each particle due to all other particles.
    """

    i = cuda.grid(1)
    num_particles = particles_positions.shape[0]
    one_over_four_pi = 1 / (4 * np.pi)
    
    # Local storage
    target_loc = cuda.local.array(3, dtype=np.float64)  
    source_loc = cuda.local.array(3, dtype=np.float64)  
    r_ij       = cuda.local.array(3, dtype=np.float64)
    cross_product = cuda.local.array(3, dtype=np.float64)

    if i < num_particles:
        ux, uy, uz = 0.0, 0.0, 0.0

        # Copy target particle's position to local storage
        target_loc[0] = particles_positions[i, 0]
        target_loc[1] = particles_positions[i, 1] 
        target_loc[2] = particles_positions[i, 2]
        target_radius = particles_radii[i]

        # Loop over each source particle
        for j in range(num_particles):
            if j != i:
                # Copy source particle position
                source_loc[0] = particles_positions[j, 0]
                source_loc[1] = particles_positions[j, 1]
                source_loc[2] = particles_positions[j, 2]
                source_strength = particles_strengths[j]
                sigma = (target_radius + particles_radii[j]) / 2

                # Compute relative position r_ij
                r_ij[0] = target_loc[0] - source_loc[0]
                r_ij[1] = target_loc[1] - source_loc[1]
                r_ij[2] = target_loc[2] - source_loc[2]

                r_ij_sq = r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]

                sigma_sq = sigma*sigma
                rho_sq = r_ij_sq / sigma_sq
                q_sigma_over_r3 = (rho_sq + 2.5) / ((rho_sq + 1)**2.5 * sigma*sigma_sq)

                # Calculate cross product and induced velocity components
                cross_product[0] = r_ij[1] * source_strength[2] - r_ij[2] * source_strength[1]
                cross_product[1] = r_ij[2] * source_strength[0] - r_ij[0] * source_strength[2]
                cross_product[2] = r_ij[0] * source_strength[1] - r_ij[1] * source_strength[0]

                ux -=  q_sigma_over_r3 * cross_product[0]
                uy -=  q_sigma_over_r3 * cross_product[1]
                uz -=  q_sigma_over_r3 * cross_product[2]

        # Write result to global memory
        velocities[i, 0] = one_over_four_pi * ux
        velocities[i, 1] = one_over_four_pi * uy
        velocities[i, 2] = one_over_four_pi * uz
        
        
        
# ==========================================================
#
# ==========================================================
def get_vorticity_field_at(target_positions, particle_positions, particle_strengths, particle_radii):
    num_particles = particle_positions.shape[0]
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block

    # Allocate GPU memory
    target_positions_gpu = cuda.to_device(target_positions)
    particle_positions_gpu = cuda.to_device(particle_positions)
    particle_strengths_gpu = cuda.to_device(particle_strengths)
    particle_radii_gpu = cuda.to_device(particle_radii)
    vorticities_gpu = cuda.device_array_like(particle_strengths)

    # Launch CUDA kernel
    get_vorticity_field_at_kernel_simplified[blocks_per_grid, threads_per_block](
        target_positions_gpu,
        particle_positions_gpu,
        particle_strengths_gpu,
        particle_radii_gpu,
        vorticities_gpu
    )

    # Copy result back to host
    return vorticities_gpu.copy_to_host()


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:])")
def get_vorticity_field_at_kernel(target_positions: np.float64, particle_positions: np.float64, particle_strengths: np.float64, particle_radii: np.float64, vorticities: np.float64):
    """
    Calculate induced vorticity at each target position using a Regularized kernel.
    Parameters:
        target_positions : np.ndarray, shape (M, 3)
        particle_positions : np.ndarray, shape (N, 3)
        particle_strengths : np.ndarray, shape (N, 3)
        particle_radii : np.ndarray, shape (N,)
        vorticities : np.ndarray, output, shape (M, 3)
    """
    i = cuda.grid(1)  # Global thread index (1D)
    num_targets = target_positions.shape[0]
    num_particles = particle_positions.shape[0]
    one_over_four_pi = 1 / (4 * np.pi)

    target_loc = cuda.local.array(3, dtype=float64)
    source_loc = cuda.local.array(3, dtype=float64)
    r_ij = cuda.local.array(3, dtype=float64)

    if i < num_targets:
        target_loc[0] = target_positions[i, 0]
        target_loc[1] = target_positions[i, 1]
        target_loc[2] = target_positions[i, 2]

        vorticity = cuda.local.array(3, dtype=np.float64)
        vorticity[0] = vorticity[1] = vorticity[2] = 0.0

        for j in range(num_particles):
            source_loc[0] = particle_positions[j, 0]
            source_loc[1] = particle_positions[j, 1]
            source_loc[2] = particle_positions[j, 2]
            sigma = particle_radii[j]
            source_strength = particle_strengths[j]

            r_ij[0] = target_loc[0] - source_loc[0]
            r_ij[1] = target_loc[1] - source_loc[1]
            r_ij[2] = target_loc[2] - source_loc[2]

            r_ij_sq = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]

            if r_ij_sq > epsilon:
                
                rho_sq = r_ij_sq / (sigma * sigma)
                sigma_cb = sigma * sigma * sigma
                xi_sigma = 7.5 / ((rho_sq + 1) ** 3.5 * sigma_cb)
                q_sigma_over_r3 = (rho_sq + 2.5) / ((rho_sq + 1) ** 2.5 * sigma_cb)

                term1 = xi_sigma - q_sigma_over_r3
                term2 = (3.0 * q_sigma_over_r3 - xi_sigma) * (
                    (r_ij[0] * source_strength[0] +
                     r_ij[1] * source_strength[1] +
                     r_ij[2] * source_strength[2]) / r_ij_sq
                )

                vorticity[0] += term1 * source_strength[0] + term2 * r_ij[0]
                vorticity[1] += term1 * source_strength[1] + term2 * r_ij[1]
                vorticity[2] += term1 * source_strength[2] + term2 * r_ij[2]

        # Store the result
        vorticities[i, 0] = vorticity[0] * one_over_four_pi
        vorticities[i, 1] = vorticity[1] * one_over_four_pi
        vorticities[i, 2] = vorticity[2] * one_over_four_pi


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:])")
def get_vorticity_field_at_kernel_simplified(target_positions: np.float64, particle_positions: np.float64, particle_strengths: np.float64, particle_radii: np.float64, vorticities: np.float64):
    """
    Calculate induced vorticity at each target position using a Regularized kernel.
    Parameters:
        target_positions : np.ndarray, shape (M, 3)
        particle_positions : np.ndarray, shape (N, 3)
        particle_strengths : np.ndarray, shape (N, 3)
        particle_radii : np.ndarray, shape (N,)
        vorticities : np.ndarray, output, shape (M, 3)
    """
    i = cuda.grid(1)  # Global thread index (1D)
    num_targets = target_positions.shape[0]
    num_particles = particle_positions.shape[0]
    one_over_four_pi = 1 / (4 * np.pi)

    target_loc = cuda.local.array(3, dtype=float64)
    source_loc = cuda.local.array(3, dtype=float64)
    r_ij = cuda.local.array(3, dtype=float64)

    if i < num_targets:
        target_loc[0] = target_positions[i, 0]
        target_loc[1] = target_positions[i, 1]
        target_loc[2] = target_positions[i, 2]

        vorticity = cuda.local.array(3, dtype=np.float64)
        vorticity[0] = vorticity[1] = vorticity[2] = 0.0

        for j in range(num_particles):
            source_loc[0] = particle_positions[j, 0]
            source_loc[1] = particle_positions[j, 1]
            source_loc[2] = particle_positions[j, 2]
            sigma = particle_radii[j]
            source_strength = particle_strengths[j]

            r_ij[0] = target_loc[0] - source_loc[0]
            r_ij[1] = target_loc[1] - source_loc[1]
            r_ij[2] = target_loc[2] - source_loc[2]

            r_ij_sq = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]
            
            if r_ij_sq > epsilon:
                
                rho_sq = r_ij_sq / (sigma * sigma) 
                
                sigma_cb = sigma * sigma * sigma
                xi_sigma = 7.5 / ((rho_sq + 1) ** 3.5 * sigma_cb)

                vorticity[0] += xi_sigma * source_strength[0] 
                vorticity[1] += xi_sigma * source_strength[1] 
                vorticity[2] += xi_sigma * source_strength[2] 

        # Store the result
        vorticities[i, 0] = vorticity[0] * one_over_four_pi
        vorticities[i, 1] = vorticity[1] * one_over_four_pi
        vorticities[i, 2] = vorticity[2] * one_over_four_pi

# ==========================================================
#
# ==========================================================
def get_strength_gradients(particle_positions, particle_strengths, particle_radii):
    num_particles = particle_positions.shape[0]
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block

    # Allocate GPU memory
    particle_positions_gpu = cuda.to_device(particle_positions)
    particle_strengths_gpu = cuda.to_device(particle_strengths)
    particle_radii_gpu = cuda.to_device(particle_radii)
    dGamma_dt_gpu = cuda.device_array_like(particle_strengths)

    # Launch CUDA kernel
    get_strength_gradients_kernel[blocks_per_grid, threads_per_block](
        particle_positions_gpu,
        particle_strengths_gpu,
        particle_radii_gpu,
        dGamma_dt_gpu
    )

    # Copy result back to host
    return dGamma_dt_gpu.copy_to_host()


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64[:,:])")
def get_strength_gradients_kernel(particles_position: np.float64, particles_strengths: np.float64, particles_radii: np.float64, dGamma_dt: np.float64):
    """
    CUDA kernel to calculate the vorticity exchange of particles (DOmega_DT).
    """
    one_over_four_pi = 1 / (4 * np.pi)
    num_particles = particles_position.shape[0] 
    r_ij = cuda.local.array(3, dtype=np.float64)
    strength_product1 = cuda.local.array(3, dtype=np.float64)
    strength_product2 = cuda.local.array(3, dtype=np.float64)

    i = cuda.grid(1)

    if i < num_particles:  # Target particles
        radius_i = particles_radii[i]
        position_i = particles_position[i]
        strength_i = particles_strengths[i]

        # Zero accumulation for each thread's result
        acc_dGamma_dt_x, acc_dGamma_dt_y, acc_dGamma_dt_z = 0.0, 0.0, 0.0

        for j in range(num_particles):  # Source particles
            if i != j:
                radius_j = particles_radii[j]
                position_j = particles_position[j]
                strength_j = particles_strengths[j]

                # Compute relative position vector r_ij = position_i - position_j
                r_ij[0] = position_i[0] - position_j[0]
                r_ij[1] = position_i[1] - position_j[1]
                r_ij[2] = position_i[2] - position_j[2]

                # Calculate squared distance and regularized parameters
                r_ij_norm_sq = r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]
                sigma = (radius_i + radius_j) * 0.5
                sigma_sq = sigma * sigma
                sigma_cb = sigma_sq * sigma

                rho_sq_plus_one = r_ij_norm_sq / sigma_sq + 1.0

                # Precompute factors for efficiency
                factor1 = (rho_sq_plus_one + 1.5) / (sigma_cb * rho_sq_plus_one**2.5)
                factor2 = 3.0 * (rho_sq_plus_one + 2.5) / (sigma_cb * sigma_sq * rho_sq_plus_one**3.5)

                # Compute strength_product1 = cross(strength_i, strength_j)
                strength_product1[0] = strength_i[1] * strength_j[2] - strength_i[2] * strength_j[1]
                strength_product1[1] = strength_i[2] * strength_j[0] - strength_i[0] * strength_j[2]
                strength_product1[2] = strength_i[0] * strength_j[1] - strength_i[1] * strength_j[0]

                # Compute strength_product2 = cross(r_ij, strength_j)
                strength_product2[0] = r_ij[1] * strength_j[2] - r_ij[2] * strength_j[1]
                strength_product2[1] = r_ij[2] * strength_j[0] - r_ij[0] * strength_j[2]
                strength_product2[2] = r_ij[0] * strength_j[1] - r_ij[1] * strength_j[0]

                # Compute strength_product3 = dot(strength_i, strength_product2)
                strength_product3 = (
                    strength_i[0] * strength_product2[0] +
                    strength_i[1] * strength_product2[1] +
                    strength_i[2] * strength_product2[2]
                )


                # Accumulate the results for each component
                acc_dGamma_dt_x += (factor1 * strength_product1[0] + factor2 * strength_product3 * r_ij[0]) * one_over_four_pi
                acc_dGamma_dt_y += (factor1 * strength_product1[1] + factor2 * strength_product3 * r_ij[1]) * one_over_four_pi
                acc_dGamma_dt_z += (factor1 * strength_product1[2] + factor2 * strength_product3 * r_ij[2]) * one_over_four_pi

        # Store the accumulated gradient in the global memory
        dGamma_dt[i, 0] = acc_dGamma_dt_x
        dGamma_dt[i, 1] = acc_dGamma_dt_y
        dGamma_dt[i, 2] = acc_dGamma_dt_z



# ==========================================================
#
# ==========================================================
def get_strength_gradients_PSE(positions, strengths, radii, viscosities):
    num_particles = positions.shape[0]
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block

    # Allocate GPU memory
    positions_gpu = cuda.to_device(positions)
    strengths_gpu = cuda.to_device(strengths)
    radii_gpu = cuda.to_device(radii)
    viscosities_gpu = cuda.to_device(viscosities)
    dGamma_dt_gpu = cuda.device_array_like(strengths)

    # Launch CUDA kernel
    get_strength_gradients_PSE_kernel[blocks_per_grid, threads_per_block](
        positions_gpu,
        strengths_gpu,
        radii_gpu,
        viscosities_gpu,
        dGamma_dt_gpu
    )

    # Copy result back to host
    return dGamma_dt_gpu.copy_to_host()


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64[:], float64[:,:])")
def get_strength_gradients_PSE_kernel(particles_position: np.float64, particles_strengths: np.float64, particles_radii: np.float64, particles_viscosities: np.float64, dGamma_dt: np.float64):
    """
    CUDA kernel to calculate the vorticity exchange of particles (DOmega_DT).
    """
    one_over_four_pi = 1 / (4 * np.pi)
    num_particles = particles_position.shape[0] 
    r_ij = cuda.local.array(3, dtype=np.float64)

    i = cuda.grid(1)

    if i < num_particles:  # Target particles
        radius_i = particles_radii[i]
        position_i = particles_position[i]
        strength_i = particles_strengths[i]
        viscosity_i =  particles_viscosities[i]

        # Zero accumulation for each thread's result
        acc_dGamma_dt_x = 0.0
        acc_dGamma_dt_y = 0.0 
        acc_dGamma_dt_z = 0.0

        for j in range(num_particles):  # Source particles
            if i != j:
                radius_j = particles_radii[j]
                position_j = particles_position[j]
                strength_j = particles_strengths[j]

                # Compute relative position vector r_ij = position_i - position_j
                r_ij[0] = position_i[0] - position_j[0]
                r_ij[1] = position_i[1] - position_j[1]
                r_ij[2] = position_i[2] - position_j[2]

                # Calculate squared distance and regularized parameters
                distance_sq = r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]
                sigma = (radius_i + radius_j) * 0.5
                sigma_sq = sigma * sigma
                sigma_cb = sigma_sq * sigma
                sigma_qu = sigma_sq * sigma_cb

                rho_sq_plus_one = distance_sq / sigma_sq + 1.0

                factor3 = 105.0 * viscosity_i / (sigma_qu * rho_sq_plus_one ** 4.5)

                # Accumulate the results for each component
                acc_dGamma_dt_x += factor3 * sigma_cb * (strength_j[0] - strength_i[0]) * one_over_four_pi
                acc_dGamma_dt_y += factor3 * sigma_cb * (strength_j[1] - strength_i[1]) * one_over_four_pi
                acc_dGamma_dt_z += factor3 * sigma_cb * (strength_j[2] - strength_i[2]) * one_over_four_pi

        # Store the accumulated gradient in the global memory
        dGamma_dt[i, 0] = acc_dGamma_dt_x
        dGamma_dt[i, 1] = acc_dGamma_dt_y
        dGamma_dt[i, 2] = acc_dGamma_dt_z



# ==========================================================
#
# ==========================================================
def get_total_kinetic_energy(particles_positions, particles_strengths, particles_radii):
    num_particles = len(particles_positions)
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    particles_positions_gpu = cuda.to_device(particles_positions)
    particles_strengths_gpu = cuda.to_device(particles_strengths)
    particles_radii_gpu = cuda.to_device(particles_radii)
    E_device = cuda.device_array(1, dtype=np.float64)  # Accumulator for energy

    # Launch CUDA kernel
    get_total_kinetic_energy_kernel[blocks_per_grid, threads_per_block](
        particles_positions_gpu,
        particles_strengths_gpu,
        particles_radii_gpu,
        E_device
    )

    # Copy result back to host
    E_host = E_device.copy_to_host()[0]

    return E_host


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64[:])")
def get_total_kinetic_energy_kernel(particles_positions, particles_strengths, particles_radii, result):
    # Thread's index
    i = cuda.grid(1)

    # Shared memory for partial results
    shared_energy = cuda.shared.array(threads_per_block, dtype=float64)  # Adjust for block size
    local_energy = 0.0  # Local energy accumulator for this thread
    r_ij = cuda.local.array(3, dtype=np.float64)
    r_ij_unit = cuda.local.array(3, dtype=float64)

    num_particles = particles_positions.shape[0]
    
    # Initialize local energy accumulator for this thread
    local_energy = 0.0
    
    if i < len(particles_positions):
        num_particles   = len(particles_positions)
        position_i      = particles_positions[i]
        strength_i      = particles_strengths[i]
        radius_i        = particles_radii[i]

        for j in range(num_particles):
            if j != i:
                position_j  = particles_positions[j]
                strength_j  = particles_strengths[j]
                radius_j    = particles_radii[j]

                # Calculate r_ij and r_ij_norm
                r_ij[0] = position_i[0] - position_j[0]
                r_ij[1] = position_i[1] - position_j[1]
                r_ij[2] = position_i[2] - position_j[2]
                r_ij_norm = (r_ij[0]*r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2])**0.5

                # Normalize r_ij
                r_ij_unit[0] = r_ij[0] / r_ij_norm
                r_ij_unit[1] = r_ij[1] / r_ij_norm
                r_ij_unit[2] = r_ij[2] / r_ij_norm

                sigma = (radius_i + radius_j) * 0.5
                rho = r_ij_norm / sigma
                rho_sq_plus_one = rho**2 + 1.0

                term_1 = 2. * rho / (rho_sq_plus_one)**0.5
                term_2 = rho**3 / rho_sq_plus_one**1.5

                # Compute product_1 = dot(strength_i, strength_j)
                product_1 = (
                    strength_i[0] * strength_j[0] +
                    strength_i[1] * strength_j[1] +
                    strength_i[2] * strength_j[2]
                )

                # Compute product_2 = dot(r_ij_unit, strength_i)
                product_2 = (
                    r_ij_unit[0] * strength_i[0] +
                    r_ij_unit[1] * strength_i[1] +
                    r_ij_unit[2] * strength_i[2]
                )

                # Compute product_3 = dot(r_ij_unit, strength_j)
                product_3 = (
                    r_ij_unit[0] * strength_j[0] +
                    r_ij_unit[1] * strength_j[1] +
                    r_ij_unit[2] * strength_j[2]
                )


                local_energy += (term_1 * product_1 + term_2 * (product_2 * product_3 - product_1)) / r_ij_norm
            
    # Store thread's result in shared memory
    shared_energy[cuda.threadIdx.x] = local_energy
    cuda.syncthreads()

    # Perform block-level reduction
    if cuda.threadIdx.x == 0:
        block_energy = 0.0
        for k in range(cuda.blockDim.x):
            block_energy += shared_energy[k]
        cuda.atomic.add(result, 0, block_energy / (16 * np.pi))

# ==========================================================
#
# ==========================================================
def get_total_helicity(particles_positions, particles_strengths, particles_radii):
    num_particles = particles_positions.shape[0]
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    particles_positions_gpu = cuda.to_device(particles_positions)
    particles_strengths_gpu = cuda.to_device(particles_strengths)
    particles_radii_gpu = cuda.to_device(particles_radii)
    H_device = cuda.device_array(1, dtype=np.float64)  # Accumulator for helicity

    # Launch CUDA kernel
    get_total_helicity_kernel[blocks_per_grid, threads_per_block](
        particles_positions_gpu,
        particles_strengths_gpu,
        particles_radii_gpu,
        H_device
    )

    # Copy result back to host
    H_host = H_device.copy_to_host()[0]

    return H_host / (4 * np.pi)


# ===================== + ===================== #

@cuda.jit
def get_total_helicity_kernel(
    particles_positions, particles_strengths, particles_radii, result
):
    """
    CUDA kernel to compute total helicity for a particle system.

    Parameters:
    -----------
    particles_positions: np.ndarray, shape (N, 3)
        Positions of particles.
    particles_strengths: np.ndarray, shape (N, 3)
        Vorticity strengths of particles.
    particles_radii: np.ndarray, shape (N,)
        Radii of particles.
    helicity_accumulator: np.ndarray, shape (1,)
        Accumulator for total helicity (shared across threads).
    """
    # Thread's index
    i = cuda.grid(1)

    # Shared memory for partial results
    shared_helicity = cuda.shared.array(threads_per_block, dtype=float64)  # Adjust for block size
    local_helicity = 0.0  # Local energy accumulator for this thread
    product_1 = cuda.local.array(3, dtype=np.float64)
    r_ij = cuda.local.array(3, dtype=np.float64)

    num_particles = particles_positions.shape[0]

    if i < num_particles:
        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i = particles_radii[i]

        for j in range(num_particles):
            if i != j:
                position_j = particles_positions[j]
                strength_j = particles_strengths[j]
                radius_j = particles_radii[j]

                # Calculate relative position and parameters
                r_ij = cuda.local.array(3, dtype=np.float64)
                r_ij[0] = position_i[0] - position_j[0]
                r_ij[1] = position_i[1] - position_j[1]
                r_ij[2] = position_i[2] - position_j[2]

                sigma = 0.5 * (radius_i + radius_j)
                rho = cuda.local.array(3, dtype=np.float64)
                rho[0] = r_ij[0] / sigma
                rho[1] = r_ij[1] / sigma
                rho[2] = r_ij[2] / sigma

                rho_sq = rho[0]*rho[0] + rho[1]*rho[1] + rho[2]*rho[2]
                term = (rho_sq + 2.5) / (sigma*sigma*sigma * (rho_sq + 1.0)**2.5)

                product_1[0] = strength_i[1] * strength_j[2] - strength_i[2] * strength_j[1]
                product_1[1] = strength_i[2] * strength_j[0] - strength_i[0] * strength_j[2]
                product_1[2] = strength_i[0] * strength_j[1] - strength_i[1] * strength_j[0]

                # Dot product r_ij · product_1
                product_2 = (
                    r_ij[0] * product_1[0]
                    + r_ij[1] * product_1[1]
                    + r_ij[2] * product_1[2]
                )

                local_helicity += product_2 * term

    # Store thread's result in shared memory
    shared_helicity[cuda.threadIdx.x] = local_helicity
    cuda.syncthreads()

    # Perform block-level reduction
    if cuda.threadIdx.x == 0:
        block_helicity = 0.0
        for k in range(cuda.blockDim.x):
            block_helicity += shared_helicity[k]
        cuda.atomic.add(result, 0, block_helicity / (4 * np.pi))
        
        
# ==========================================================
#
# ==========================================================
def get_total_enstrophy(particles_positions, particles_strengths, particles_radii):
    num_particles = len(particles_positions)
    blocks_per_grid = (num_particles + threads_per_block - 1) // threads_per_block

    # Allocate device memory
    particles_positions_gpu = cuda.to_device(particles_positions)
    particles_strengths_gpu = cuda.to_device(particles_strengths)
    particles_radii_gpu = cuda.to_device(particles_radii)
    Ens_device = cuda.device_array(1, dtype=np.float64)  # Accumulator for enstrophy

    # Launch CUDA kernel
    get_total_enstrophy_kernel_simple[blocks_per_grid, threads_per_block](
        particles_positions_gpu,
        particles_strengths_gpu,
        particles_radii_gpu,
        Ens_device
    )

    # Copy result back to host
    Ens_host = Ens_device.copy_to_host()[0]

    return Ens_host / (4 * np.pi)


# ===================== + ===================== #

@cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64[:])")
def get_total_enstrophy_kernel_long(particles_positions, particles_strengths, particles_radii, result):
    # Calculate the thread's unique index
    i = cuda.grid(1)
    
    # Shared memory for partial results
    shared_ens  = cuda.shared.array(threads_per_block, dtype=float64)  # Adjust for block size
    r_ij        = cuda.local.array(3, dtype=float64)
    r_ij_unit   = cuda.local.array(3, dtype=float64)
    r_ij        = cuda.local.array(3, dtype=np.float64)
    local_ens   = 0.0 # Local enstrophy accumulator for this thread
    
    num_particles = particles_positions.shape[0]
    
    if i < len(particles_positions):
        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i = particles_radii[i]

        for j in range(num_particles):
            position_j = particles_positions[j]
            strength_j = particles_strengths[j]
            radius_j = particles_radii[j]

            # Compute relative position vector r_ij = position_i - position_j
            r_ij[0] = position_i[0] - position_j[0]
            r_ij[1] = position_i[1] - position_j[1]
            r_ij[2] = position_i[2] - position_j[2]
            r_ij_norm_sq = r_ij[0]* r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]
            
            r_ij_norm = r_ij_norm_sq**0.5

            # Normalize r_ij
            r_ij_unit[0] = r_ij[0] / r_ij_norm
            r_ij_unit[1] = r_ij[1] / r_ij_norm
            r_ij_unit[2] = r_ij[2] / r_ij_norm

            sigma = (radius_i + radius_j) * 0.5
            sigma_cb = sigma*sigma*sigma
            
            rho = r_ij_norm / sigma
            rho_sq = rho * rho
            rho_sq_plus_one = rho ** 2 + 1.0

            term_1 = (5.0 - rho_sq * (rho_sq + 3.5)) / rho_sq_plus_one**3.5
            term_2 = 3.0 * ((rho_sq * (rho_sq + 4.5) + 3.5)) / rho_sq_plus_one**4.5

            # Compute product_1 = dot(strength_i, strength_j)
            product_1 = (
                strength_i[0] * strength_j[0] +
                strength_i[1] * strength_j[1] +
                strength_i[2] * strength_j[2]
            )

            # Compute product_2 = dot(r_ij, strength_i)
            product_2 = (
                r_ij[0] * strength_i[0] +
                r_ij[1] * strength_i[1] +
                r_ij[2] * strength_i[2]
            )

            # Compute product_3 = dot(r_ij, strength_j)
            product_3 = (
                r_ij[0] * strength_j[0] +
                r_ij[1] * strength_j[1] +
                r_ij[2] * strength_j[2]
            )


            local_ens += (term_1 * product_1 + term_2 * product_2 * product_3) / sigma_cb
                
    # Store thread's result in shared memory
    shared_ens[cuda.threadIdx.x] = local_ens
    cuda.syncthreads()

    # Perform block-level reduction
    if cuda.threadIdx.x == 0:
        block_ens = 0.0
        for k in range(cuda.blockDim.x):
            block_ens += shared_ens[k]
        cuda.atomic.add(result, 0, block_ens / (4 * np.pi) )
        
        
# ===================== + ===================== #  
        
@cuda.jit("void(float64[:,:], float64[:,:], float64[:], float64[:])")
def get_total_enstrophy_kernel_simple(particles_positions, particles_strengths, particles_radii, result):
    # Calculate the thread's unique index
    i = cuda.grid(1)
    
    # Shared memory for partial results
    shared_ens  = cuda.shared.array(threads_per_block, dtype=float64)  # Adjust for block size
    r_ij        = cuda.local.array(3, dtype=float64)
    r_ij_unit   = cuda.local.array(3, dtype=float64)
    r_ij        = cuda.local.array(3, dtype=np.float64)
    local_ens   = 0.0 # Local enstrophy accumulator for this thread
    
    num_particles = particles_positions.shape[0]
    
    if i < len(particles_positions):
        position_i = particles_positions[i]
        strength_i = particles_strengths[i]
        radius_i = particles_radii[i]

        for j in range(num_particles):
            position_j = particles_positions[j]
            strength_j = particles_strengths[j]
            radius_j = particles_radii[j]

            # Compute relative position vector r_ij = position_i - position_j
            r_ij[0] = position_i[0] - position_j[0]
            r_ij[1] = position_i[1] - position_j[1]
            r_ij[2] = position_i[2] - position_j[2]
            r_ij_sq = r_ij[0]* r_ij[0] + r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2]

            sigma = (radius_i + radius_j) * 0.5
            sigma_sq = sigma*sigma

            term_1 = (r_ij_sq + 3.2*sigma_sq) / (r_ij_sq + sigma_sq)**1.5

            # Compute product_1 = dot(strength_i, strength_j)
            product_1 = (
                strength_i[0] * strength_j[0] +
                strength_i[1] * strength_j[1] +
                strength_i[2] * strength_j[2]
            )

            local_ens += term_1 * product_1
                
    # Store thread's result in shared memory
    shared_ens[cuda.threadIdx.x] = local_ens
    cuda.syncthreads()

    # Perform block-level reduction
    if cuda.threadIdx.x == 0:
        block_ens = 0.0
        for k in range(cuda.blockDim.x):
            block_ens += shared_ens[k]
        cuda.atomic.add(result, 0, block_ens / (8 * np.pi) )