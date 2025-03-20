from numba import njit, prange
import numpy as np

# Define Gaussian function for fitting
def gaussian(r, omega_0, a):
    return omega_0 * np.exp(-(r**2) / a**2)

epsilon = 1e-9

# ============================================================= # 
# Scripts to apply analytical flow solutions to particle fields:
# ============================================================= # 
@njit(parallel=True, fastmath=True)
def lamb_oseen_vpm(
    particle_positions: np.ndarray, 
    volumes: np.ndarray, 
    particle_radius: float, 
    particle_viscosity: float,
    vortex_center: np.ndarray,  
    vortex_strength: float,
    core_radius: float,
    epsilon_W: float = 0.0,   # Amplitude of perturbation
    phase_W: float = 0.0,     # Phase of the perturbation
    N_W: float = 12           # Wavenumber of perturbation, most amplified mode of the Widnall instability
): 
      """
      Create a Lamb-Oseen vortex with sinusoidal perturbation in vortex_center's y- and z-coordinates.

      Parameters:
            particle_positions (ndarray): Positions of particles, shape (N, 3).
            volumes (ndarray): Particle volumes, shape (N,).
            particle_radius (float): Smoothing radius for particles.
            particle_viscosity (float): Kinematic viscosity.
            vortex_center (ndarray): Initial center of the vortex, shape (3,).
            vortex_strength (float): Total circulation of the vortex (Gamma).
            core_radius (float): Core radius of the vortex.
            epsilon_W (float): Amplitude of sinusoidal perturbation.
            phase_W (float): Phase of the sinusoidal perturbation.
            N_W (float): Wavenumber of the sinusoidal perturbation.

      Returns:
            velocities (ndarray): Velocities of particles, shape (N, 3).
            strengths (ndarray): Vorticity field (omega) of particles, shape (N, 3).
            radii (ndarray): Smoothing radii of particles, shape (N,).
            viscosities (ndarray): Viscosity of particles, shape (N,).
      """
      num_particles = len(particle_positions)

      # Initialize vorticity strengths array
      strengths = np.zeros_like(particle_positions, dtype=np.float64)

      # Perturb the vortex center in x and y as sinusoidal functions of z
      perturbed_x = vortex_center[0] + epsilon_W * np.sin(phase_W + N_W * particle_positions[:, 2])
      perturbed_y = vortex_center[1] + epsilon_W * np.cos(phase_W + N_W * particle_positions[:, 2])

      # Compute distances from vortex center
      x = particle_positions[:, 0] - perturbed_x
      y = particle_positions[:, 1] - perturbed_y
      z = particle_positions[:, 2] - vortex_center[2]  # Distance along the z-axis

      # Radial distance from the vortex core in the x-y plane
      r_magnitude = np.sqrt(x**2 + y**2)


      # Set particle core radii
      radii = particle_radius * np.ones(num_particles, dtype=np.float64)

      # Loop over particles to compute vorticity field
      for i in prange(num_particles):
            # Theoretical vorticity profile at particle position
            omega_z = (vortex_strength / (np.pi * core_radius**2)) * np.exp(-r_magnitude[i]**2 / core_radius**2)
            
            # Scale vorticity by particle volume to maintain consistency
            strengths[i, 2] = omega_z * volumes[i]

      # Compute the initial velocities induced by the vortex ring (placeholder)
      velocities = np.zeros_like(strengths, dtype=np.float64)

      # Set particle viscosity array
      viscosities = particle_viscosity * np.ones(num_particles, dtype=np.float64)

      return velocities, strengths, radii, viscosities



@njit(parallel=True, fastmath=True)
def vortex_ring_vpm(
    particle_positions: np.ndarray, 
    volumes: np.ndarray, 
    particle_radius: float, 
    particle_viscosity: float,
    ring_center: np.ndarray, 
    ring_radius: float, 
    ring_strength: float, 
    ring_thickness: float,
    epsilon_W: float = 0.0,   
    phase_W: float = 0.0,     
    N_W: float = 12           
):  
      """
      Compute the total vorticity induced by a vortex ring in 3D space with
      a sinusoidal perturbation displacing the vortex core axis in the azimuthal direction.
      
      Parameters:
            particle_positions: (N, 3) array of particle positions.
            volumes: (N,) array of particle volumes.
            particle_radius: Scalar radius for particles.
            particle_viscosity: Scalar viscosity for particles.
            ring_center: (3,) array representing the vortex ring center.
            ring_radius: Scalar radius of the vortex ring.
            ring_strength: Scalar circulation strength of the ring.
            ring_thickness: Scalar thickness of the vortex ring.
            epsilon_W, phase_W, N_W: Parameters controlling Widnall instability.
      
      Returns:
            velocities: (N, 3) array of initial velocities (placeholder here).
            strengths: (N, 3) array of vorticity strengths.
            radii: (N,) array of particle radii.
            viscosities: (N,) array of particle viscosities.
      """
      # Shift particle positions relative to ring center
      X_shift = particle_positions[:, 0] - ring_center[0]
      Y_shift = particle_positions[:, 1] - ring_center[1]
      Z_shift = particle_positions[:, 2] - ring_center[2]

      # Toroidal coordinates and perturbation
      theta = np.arctan2(Z_shift, Y_shift)
      delta_Y = epsilon_W * np.cos(phase_W + N_W * theta)
      delta_Z = epsilon_W * np.sin(phase_W + N_W * theta)
      Y_shift -= delta_Y
      Z_shift -= delta_Z

      # Gaussian vorticity profile
      r_yz = np.sqrt(Y_shift**2 + Z_shift**2)
      radial_distance_to_core = np.abs(r_yz - ring_radius)
      vorticity_magnitude = (ring_strength / (np.pi * ring_thickness**2)) * np.exp(-radial_distance_to_core**2 / ring_thickness**2)
      x_decay = np.exp(-X_shift**2 / ring_thickness**2)

      # Tangential vorticity
      omega_x = np.zeros_like(X_shift)
      omega_y = -vorticity_magnitude * np.sin(theta) * x_decay
      omega_z = vorticity_magnitude * np.cos(theta) * x_decay

      # Circulation and correction
      strengths = np.empty_like(particle_positions, dtype=np.float64)
      strengths[:, 0] = omega_x * volumes
      strengths[:, 1] = omega_y * volumes
      strengths[:, 2] = omega_z * volumes
      
      Gamma_total = np.sum(np.sqrt(np.sum(strengths**2, axis=1)))
      Gamma_theo = 2 * np.pi * ring_radius * ring_strength
      strengths *= np.abs(Gamma_theo / Gamma_total)

      # Placeholder velocities
      velocities = np.zeros_like(strengths, dtype=np.float64)
      radii = np.full(len(particle_positions), particle_radius, dtype=np.float64)
      viscosities = np.full(len(particle_positions), particle_viscosity, dtype=np.float64)

      return velocities, strengths, radii, viscosities



# ==================================================
# Create turbulent velocity field:
# ==================================================

def isotropic_turbulence_vpm(box_size: float, particle_distance: float, particle_radius: float):

      # ==================================
      # Get the particles positions:
      # ==================================
      x = np.arange(-box_size/2, box_size/2, step=particle_distance)
      y = np.arange(-box_size/2, box_size/2, step=particle_distance)
      z = np.arange(-box_size/2, box_size/2, step=particle_distance)

      xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

      particle_positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

      # ==================================
      # Get the velocity field:
      # ==================================
      # Define parameters. N is the number of vortex in a line of the 3D domain
      N = len(x)  # Number of grid points in each dimension
      L = box_size  # Domain size
      #k_min = 2 * np.pi / L      # Minimum wavenumber
      k_max = np.pi * N / L      # Largest wavenumber
      k_diss = k_max / 3    # Dissipation wavenumber (adjust this as needed)

      # Define wavenumbers (kx, ky, kz)
      kx = np.fft.fftfreq(N, L/N) * 2 * np.pi
      ky = np.fft.fftfreq(N, L/N) * 2 * np.pi
      kz = np.fft.fftfreq(N, L/N) * 2 * np.pi
      KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

      # Compute the wavenumber magnitude |k|
      K = np.sqrt(KX**2 + KY**2 + KZ**2)

      # Define the energy spectrum E(k) ~ k^-5/3 with exponential cutoff
      E_k = K**(-5/3) * np.exp(-(K / k_diss)**2)

      # Handle K=0 separately to avoid division by zero
      E_k[K == 0] = 0

      # Generate random phases and amplitudes for each wavenumber
      np.random.seed(0)  # For reproducibility
      random_phases = np.exp(2j * np.pi * np.random.rand(N, N, N))


      # Generate random amplitudes for the x, y, z components of the velocity field
      u_hat_x = np.sqrt(E_k) * random_phases
      u_hat_y = np.sqrt(E_k) * random_phases
      u_hat_z = np.sqrt(E_k) * random_phases

      # Ensure incompressibility: u_hat_x, u_hat_y, u_hat_z should be perpendicular to k
      # Project onto the plane perpendicular to the wavenumber vector
      u_hat_dot_k = (KX * u_hat_x + KY * u_hat_y + KZ * u_hat_z) / (K + 1e-10)  # Avoid div by zero
      u_hat_x -= u_hat_dot_k * KX / (K + 1e-10)  # Subtract the component along kx
      u_hat_y -= u_hat_dot_k * KY / (K + 1e-10)  # Subtract the component along ky
      u_hat_z -= u_hat_dot_k * KZ / (K + 1e-10)  # Subtract the component along kz

      # Inverse FFT to obtain the velocity field in real space
      u_x = np.real(np.fft.ifftn(u_hat_x))
      u_y = np.real(np.fft.ifftn(u_hat_y))
      u_z = np.real(np.fft.ifftn(u_hat_z))


      particles_velocities =  np.zeros_like(particle_positions)

      particles_velocities[:,0] = u_x.reshape(-1)
      particles_velocities[:,1] = u_y.reshape(-1)
      particles_velocities[:,2] = u_z.reshape(-1)

      velocity_field = np.zeros((N,N,N,3))
      velocity_field[:,:,:,0] = u_x
      velocity_field[:,:,:,1] = u_y
      velocity_field[:,:,:,2] = u_z

      # ==================================
      # Get the vorticity field:
      # ==================================
      vorticity_field = np.zeros((N, N, N, 3))

      # Central difference for interior points
      vorticity_field[..., 0] = (
            np.roll(velocity_field[..., 2], -1, axis=1) - np.roll(velocity_field[..., 2], 1, axis=1) ) / (2 * particle_distance) 
      - ( np.roll(velocity_field[..., 1], -1, axis=2) - np.roll(velocity_field[..., 1], 1, axis=2) ) / (2 * particle_distance)

      vorticity_field[..., 1] = (
            np.roll(velocity_field[..., 0], -1, axis=2) - np.roll(velocity_field[..., 0], 1, axis=2) ) / (2 * particle_distance) 
      - ( np.roll(velocity_field[..., 2], -1, axis=0) - np.roll(velocity_field[..., 2], 1, axis=0) ) / (2 * particle_distance)

      vorticity_field[..., 2] = (
            np.roll(velocity_field[..., 1], -1, axis=0) - np.roll(velocity_field[..., 1], 1, axis=0) ) / (2 * particle_distance) 
      - ( np.roll(velocity_field[..., 0], -1, axis=1) - np.roll(velocity_field[..., 0], 1, axis=1) ) / (2 * particle_distance)

      # Compute vortex strengths (magnitude of vorticity)
      particles_strengths = np.reshape(vorticity_field,(N*N*N, 3))*(4/3)*particle_distance**3

      # ==================================
      # Get the particles radii:
      # ==================================
      particles_radii = particle_radius * np.ones_like(particle_positions[:,0])

      return particle_positions, particles_velocities, particles_strengths, particles_radii






