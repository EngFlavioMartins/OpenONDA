# ========================================================== # 
# Importing modules
# ========================================================== # 
import importlib
import importlib.util
import numpy as np
import pyvista as pv
from numba import cuda
from openONDA.solvers.VPM.Particle import Particle
from numba import njit

np.bool = np.bool_
    
    
# Helper function for manual norm computation
@njit(parallel=True, fastmath=True)
def compute_norm(arr: np.ndarray, axis=None):
      return np.sqrt(np.sum(arr*arr, axis=axis))


# Small value
epsilon = 1e-9

# ========================================================== #
# Definition of ParticleSystem class:
# ========================================================== #
class ParticleSystem:
      """
      Manages particles and calculates fluid dynamics properties.

      Attributes:
      ----------
      particles : list of Particle
            Collection of Particle objects in the system.
      dt : float
            Time step size for the simulation in seconds.
      nu : float
            Kinematic viscosity of the fluid in m²/s.
      time_integration_method : str
            Time integration method, either "Euler", "RK2" or "RK3" (Runge-Kutta methods).
      processing_unit : str
            Specifies "CPU" or "GPU" for computation.
      flow_model : str
            Specifies flow model: DNS, LES, or pseudo2D.
      monitor_variables : list of str
            Variables to monitor during simulation, e.g., ['Circulation', 'Linear Impulse', 'Angular Impulse'].
      """

      def __init__(
            self,
            time_step_size: float = 1.0, 
            time_integration_method: str = 'RK2', 
            processing_unit: str = 'CPU', 
            flow_model: str = 'DNS', 
            viscous_scheme: str = 'CoreSpreading', 
            monitor_variables: list[str] = None, 
            time: float = 0.0, 
            time_step: int = 0, 
            backup_frequency: int = 1, 
            backup_filename: str ="particle_data",
            relax_strength_solution: bool = False):
            """
            Initialize the simulation parameters and dynamically load the appropriate physics module.
            """
            if monitor_variables is None:
                  monitor_variables = ['Circulation', 
                                       'Linear Impulse', 
                                       'Angular Impulse', 
                                       'Kinetic Energy', 
                                       'Enstrophy', 
                                       'Helicity']
            
            # Input validation
            self._validate_float(time_step_size, "time_step_size")
            self._validate_integration_method(time_integration_method)
            self._validate_flow_model(flow_model)
            self._validate_viscous_scheme(viscous_scheme)
            self._validate_processing_unit(processing_unit)

            # Simulation parameters
            self.particles          = []
            self.dt                 = time_step_size
            self.flow_time          = time
            self.time_step          = time_step
            self.flow_model         = flow_model
            self.viscous_scheme     = viscous_scheme
            self.monitor_variables  = monitor_variables
            self.backup_filename    = backup_filename
            self.backup_frequency   = backup_frequency
            self.time_integration_method  = time_integration_method 
            self.processing_unit          = processing_unit
            self.Cnu                      = 0.02958 # reference LES fitler constant
            self.relax_strength_solution  = relax_strength_solution
            
            
            # Dynamically load the physics module based on the processing unit
            # Attempt to get the current CUDA device if available
            if self.processing_unit == "GPU":
                  print("=" * 60)
                  print(f"Checking if CUDA-enabled GPU device is available...")
                  print("-" * 60)
                  try:
                        # Attempt to access CUDA device context
                        device = cuda.current_context().device
                        print(f"- CUDA device found: {device.name}")
                        self.processing_unit = "GPU"  # Set to GPU explicitly
                  except cuda.CudaSupportError:
                        print("- No CUDA device detected. Skipping GPU-specific imports.")
                        self.processing_unit = "CPU"  # Force to use CPU if no CUDA device is found

                  print(f"- Using processing unit: {self.processing_unit}")  # Check the assigned processing unit
                  print("=" * 60 + "\n")

            self._load_physics_module()
            
            
            if self.flow_model == 'Potential':
                  self.flow_model_description = "Potential flow"
                  if self.viscous_scheme is not None:
                        raise ValueError("For 'Potential' flow model, 'viscous_scheme' must be None.")

            elif self.flow_model == 'pseudo2D':
                  self.flow_model_description = "(ν)(∇²)ω"
                  if self.viscous_scheme not in ["CoreSpreading", "PSE"]:
                        raise ValueError("For 'pseudo2D' flow model, 'viscous_scheme' must be 'CoreSpreading' or 'PSE'.")

            elif self.flow_model == 'LES':
                  self.flow_model_description = "(ω.∇)u + (ν+νt)(∇²)ω"
                  if self.viscous_scheme not in ["CoreSpreading", "PSE"]:
                        raise ValueError("For 'LES' flow model, 'viscous_scheme' must be 'CoreSpreading' or 'PSE'.")

            elif self.flow_model == 'DNS':
                  self.flow_model_description = "(ω.∇)u + (ν)(∇²)ω"
                  if self.viscous_scheme not in ["CoreSpreading", "PSE"]:
                        raise ValueError("For 'DNS' flow model, 'viscous_scheme' must be 'CoreSpreading' or 'PSE'.")
            else:
                  raise ValueError(f"Unsupported flow model: {self.flow_model}")
            
      # ===================== + ===================== #       

      def _load_physics_module(self):
            """
            Dynamically imports the physics module based on the selected processing unit.
            """
            if self.processing_unit == "CPU":
                  module_name = "openONDA.solvers.VPM.ParticlesPhysics" 
            elif self.processing_unit == "GPU":
                  module_name = "openONDA.solvers.VPM.ParticlesPhysicsCuda"
            else:
                  raise ValueError(f"Unknown processing unit: {self.processing_unit}")

            print(f"Loading physics module: {module_name}")
            physics_module = importlib.import_module(module_name)
            self.physics = physics_module  # Store the module reference



      def _validate_float(self, value, name):
            if not isinstance(value, (float, int, np.float64)):
                  raise TypeError(f"{name} must be a float or int, got {type(value).__name__}")


      def _validate_integration_method(self, method):
            valid_methods = {"Euler", "RK2", "RK3", "RK4"}
            if method not in valid_methods:
                  raise ValueError(f"Invalid time integration method '{method}'. Choose from {valid_methods}.")


      def _validate_viscous_scheme(self, scheme):
            valid_schemes = {"PSE", "CoreSpreading", None}
            if scheme not in valid_schemes:
                  raise ValueError(f"Invalid viscous scheme '{scheme}'. Choose from {valid_schemes}.")


      def _validate_processing_unit(self, unit):
            valid_units = {"CPU", "GPU"}
            if unit not in valid_units:
                  raise ValueError(f"Invalid processing unit '{unit}'. Choose from {valid_units}.")
            
            
      def _validate_flow_model(self, method):
            valid_methods = {"Potential", "pseudo2D", "DNS", "LES"}
            if method not in valid_methods:
                  raise ValueError(f"Invalid time integration method '{method}'. Choose from {valid_methods}.")


      # System Information
      def __len__(self):
            return len(self.particles)


      def __getitem__(self, index):
            return self.particles[index]


      def __iter__(self):
            return iter(self.particles)


      def __str__(self):
            return (
                  f"\n{'=' * 60}\n"
                  f"Particle System with {len(self.particles)} particles\n"
                  f"{'-' * 60}\n"
                  f"Starting from step:  {self.time_step}\n"
                  f"Simulation Time:     {self.flow_time:.3E} s\n"
                  f"Time Step Size:      {self.dt:.3E} s\n"
                  f"Viscous scheme:      {self.viscous_scheme}\n"
                  f"Total strength:      {self.get_total_strength_magnitude():.3E} m³/s\n"
                  f"Integration Method:  {self.time_integration_method}\n"
                  f"Processing Unit:     {self.processing_unit}\n"
                  f"Flow Model:          {self.flow_model}\n"
                  f"Monitor Variables:   {', '.join(self.monitor_variables)}\n"
                  f"Backup Frequency:    Every {self.backup_frequency} time steps\n"
                  f"Backup Filename:     {self.backup_filename}\n"
                  f"{'=' * 60}"
            )

      def get_particle_positions(self):
            return np.array([particle.position for particle in self.particles])
      
      def get_particle_group_id(self):
            return np.array([particle.group_id for particle in self.particles])


      def get_particle_strengths(self):
            return np.array([particle.strength for particle in self.particles])


      def get_particle_velocities(self):
            return np.array([particle.velocity for particle in self.particles])


      def get_particle_radii(self):
            return np.array([particle.radius for particle in self.particles])


      def get_particle_viscosities(self):
            return np.array([particle.viscosity for particle in self.particles])
      
      
      def get_particle_viscosities_t(self):
            return np.array([particle.viscosity_t for particle in self.particles])
      
      
      def get_particle_viscosities_eff(self):
            return np.array([particle.viscosity_eff for particle in self.particles])
      

      def get_particle_strength_magnitudes(self) -> np.ndarray:
            return np.array([particle.strength_magnitude for particle in self.particles])

      
      def get_particle_vorticities(self) -> np.ndarray:
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii = self.get_particle_radii()

            vorticities = self.physics.get_vorticity_field_at(positions, positions, strengths, radii)
            
            return vorticities


      def get_total_linear_impulse(self) -> np.ndarray:
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            return 0.5 * np.sum(np.cross(positions, strengths), axis=0)


      def get_total_angular_impulse(self) -> np.ndarray:
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            I = np.cross(positions, strengths)
            return (1 / 3) * np.sum(np.cross(positions, I), axis=0)
      

      def get_total_strength(self) -> float:
            strengths = np.array([particle.strength for particle in self.particles])
            return np.sum(strengths, axis=0)
      
      
      def get_total_strength_magnitude(self) -> float:
            strengths = np.array([particle.strength_magnitude for particle in self.particles])
            return np.sum(strengths)
      
      
      def get_total_enstrophy(self) -> float:
            
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii     = self.get_particle_radii()
            
            total_enstrophy = self.physics.get_total_enstrophy(positions, strengths, radii)
            
            return total_enstrophy

      def get_total_kinetic_energy(self) -> float:
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii = self.get_particle_radii()

            kinetic_energy =self.physics.get_total_kinetic_energy(positions, strengths, radii)
            
            return kinetic_energy
      
      
      def get_total_helicity(self) -> float:
            """
            Compute the total helicity of the system.

            Returns:
            --------
            total_helicity : float
                  The total helicity of the system.
            """
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii     = self.get_particle_radii()        # Shape (N,)

            # Compute dot product for each particle
            total_helicity = self.physics.get_total_helicity(positions, strengths, radii)
            return total_helicity

      
      # ========================================================== # 
      # Calculated induced fields:
      # ========================================================== # 
      def get_vorticity_field_at(self, grid_positions: np.ndarray):
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii = self.get_particle_radii()

            vorticity_field = self.physics.get_vorticity_field_at(grid_positions, positions, strengths, radii)
            
            return vorticity_field


      def get_induced_velocity_at(self, grid_positions):
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii = self.get_particle_radii()

            velocity_field = self.physics.get_induced_velocity_at(grid_positions, positions, strengths, radii)
            
            return velocity_field
      

      # ========================================================== # 
      # Diagnostic Methods
      # ========================================================== # 
      def log_diagnostics(self):
            """Log and display flow diagnostic information based on monitored variables and write to a CSV file."""
            print("\n" + "=" * 60)
            print(f"{'Flow Diagnostics':^5}")
            print("-" * 60)
            print(f"Flow time: {self.flow_time:0.3E} s")
            print(f"Time step: {self.time_step}\n")

            if 'Circulation' in self.monitor_variables:
                  circulation = self.get_total_strength_magnitude()
                  print(f"{'Circulation':<20}: {circulation:0.3E} m³/s")
                  
            if 'Kinetic energy' in self.monitor_variables:
                  total_Ek = self.get_total_kinetic_energy()
                  print(f"{'Kinetic energy':<20}: {total_Ek:0.3E} m⁵/s²")
                  
            if 'Linear impulse' in self.monitor_variables:
                  total_I = self.get_total_linear_impulse()
                  print(f"{'Linear Impulse':<20}: ({total_I[0]:0.3E}, {total_I[1]:0.3E}, {total_I[2]:0.3E}) m⁴/s")
                  
            if 'Angular impulse' in self.monitor_variables:
                  total_A = self.get_total_angular_impulse()
                  print(f"{'Angular Impulse':<20}: ({total_A[0]:0.3E}, {total_A[1]:0.3E}, {total_A[2]:0.3E}) m⁵/s")
                  
            if 'Enstrophy' in self.monitor_variables:
                  total_Ens = self.get_total_enstrophy()
                  print(f"{'Total enstrophy':<20}: {total_Ens:0.3E} m³/s²")
                  
            if 'Helicity' in self.monitor_variables:
                  total_H = self.get_total_helicity()
                  print(f"{'Total helicity':<20}: {total_H:0.3E} m²/s²")
                  
            print("=" * 60 + "\n")
            
            
      
      # ========================================================== # 
      # Particle Initialization and Backup Methods
      # ========================================================== # 
      def add_particle(self, particle: Particle):
            """Add a new particle to the system."""
            self.particles.append(particle)
            

      def remove_particles(self, particle_indices: list):
            """Remove particles based on their list indices."""
            for index in sorted(set(particle_indices), reverse=True):
                  if 0 <= index < len(self.particles):
                        self.particles.pop(index)
            print(f"(info) Removed {len(particle_indices)} particles")


      def remove_all_particles(self):
            """Remove all particles from the system."""
            self.particles.clear()
            print(f"(info) Removed all particles from particle system.")


      def print_particle_information(self, index=None):
            """
            Display information about the particle field.
            
            If an index is provided, display information for that specific particle.
            """
            print("\nParticle system information ({} particles):".format(len(self)))
            if index is None:
                  for particle in self.particles: print(particle)
            elif isinstance(index, int):
                  print(self.particles[index])


      def add_particle_field(self, positions: np.ndarray, velocities: np.ndarray, strengths: np.ndarray, radii: np.ndarray, viscosities: np.ndarray, viscosities_t: np.ndarray = None, group_id: int = None):
            """
            Initialize particle system from user-provided numpy arrays.

            Arguments:
            ----------
            positions : np.ndarray
                  Array of particle positions (N x 3).
            velocities : np.ndarray
                  Array of particle velocities (N x 3).
            strengths : np.ndarray
                  Array of particle strengths (N x 3).
            radii : np.ndarray
                  Array of particle radii (N x 1).
            viscosities : np.ndarray
                  Array of particle viscosities (N x 1).
            viscosities_t : np.ndarray, optional
                  Array of particle turbulence viscosities (N x 1). Defaults to zeros.
            group_id : int or np.ndarray, optional
                  Group ID for particles. If None, defaults to zeros. If an integer, it will be applied to all particles.
            """
            # Validate input array dimensions
            N = positions.shape[0]
            if positions.shape[1] != 3 or velocities.shape[1] != 3 or strengths.shape[1] != 3:
                  raise ValueError("Positions, velocities, and strengths must be of shape (N x 3).")
            if radii.shape[0] != N or viscosities.shape[0] != N:
                  raise ValueError("Radii and viscosities must have the same number of elements as positions.")

            # Default values for optional arguments
            if viscosities_t is None:
                  viscosities_t = np.zeros_like(viscosities, dtype=np.float64)
                  
            if group_id is None:
                  group_id = np.zeros(N, dtype=np.int64)
            elif isinstance(group_id, int):
                  group_id = np.full(N, group_id, dtype=np.int64)

            # Add particles to the system
            for pos, vel, strg, rad, vis, vis_t, g_id in zip(positions, velocities, strengths, radii, viscosities, viscosities_t, group_id):
                  self.add_particle(Particle(position=pos, velocity=vel, strength=strg, radius=rad, viscosity=vis, viscosity_t=vis_t, group_id=g_id))

                  


      def load_particle_field_from_backup(self, filename: str):
            """
            Import particle data from a VTP file.

            Arguments:
            ----------
            filename : str
                  The name of the VTP file to read.

            Returns:
            --------
            Positions, velocities, strengths, and radii as numpy arrays.
            """
            point_cloud = pv.read(filename)
            
            positions  = np.array(point_cloud.points, dtype=np.float64)
            velocities = np.array(point_cloud.point_data['Velocity'], dtype=np.float64)
            strengths  = np.array(point_cloud.point_data['Strength'], dtype=np.float64)
            radii      = np.array(point_cloud.point_data['Radius'], dtype=np.float64)
            group_id   = np.array(point_cloud.point_data['Group_ID'], dtype=np.int64)
            viscosities = np.array(point_cloud.point_data['Viscosity'], dtype=np.float64)
            viscosities_t = np.array(point_cloud.point_data['Viscosity_t'], dtype=np.float64)
            
            for pos, vel, strg, rad, vis, vis_t, g_id in zip(positions, velocities, strengths, radii, viscosities, viscosities_t, group_id):
                  self.add_particle(Particle(position=pos, velocity=vel, strength=strg, radius=rad, viscosity=vis, viscosity_t=vis_t, group_id=g_id))

      def backup_particle_field(self):
            """
            Export particle data to a VTK file.

            A filename pattern "backup_<step>.vtp" is used for each backup.
            """
            points      = self.get_particle_positions()
            velocities  = self.get_particle_velocities()
            strengths   = self.get_particle_strengths()
            radii       = self.get_particle_radii()
            group_id    = self.get_particle_group_id()
            viscosities = self.get_particle_viscosities()
            viscosities_t = self.get_particle_viscosities_t()

            point_cloud = pv.PolyData(points)
            point_cloud.point_data['Velocity']  = velocities
            point_cloud.point_data['Strength']  = strengths
            point_cloud.point_data['Radius']    = radii
            point_cloud.point_data['Group_ID']  = group_id
            point_cloud.point_data['Viscosity'] = viscosities
            point_cloud.point_data['Viscosity_t'] = viscosities_t

            backup_filename = f"{self.backup_filename}_{str(self.time_step).zfill(4)}.vtp"
            point_cloud.save(backup_filename)
            print(f"• Particle data exported to {backup_filename}")

            
      # ========================================================== # 
      # Methods status of the particle system:
      # ========================================================== # 
      def update_state(self):
            """ 
            Perform the time integration of particle positions. This method advances the particles 
            based on their velocities and the chosen time integration method (Euler or RK2).
            """
            print("• Advecting particles and updating time-step.")
            positions = self.get_particle_positions()
            velocities = self.get_particle_velocities()

            # Ensure time_integration_method is cleaned of any extra spaces
            self.time_integration_method = self.time_integration_method.strip()

            # Select the integration method based on the time_integration_method
            integration_methods = {
                  'Euler': self._euler_update_position,
                  'RK2': self._rk2_update_position,
                  'RK3': self._rk3_update_position,
                  'RK4': self._rk4_update_position
            }

            if self.time_integration_method in integration_methods:
                  new_positions = integration_methods[self.time_integration_method](positions, velocities)
            else:
                  raise ValueError(f"Unknown time integration method: {self.time_integration_method}")

            # Backup the data at the user-prescribed time-steps:
            if self.time_step % self.backup_frequency == 0:
                  self.backup_particle_field()
                  self.log_diagnostics()

            # Update time:
            self.flow_time += self.dt
            self.time_step += 1
            
            print(f"\n>>> Flow time: {self.flow_time:0.3E} s, Time-step: {self.time_step:d}\n")

            # Update particle positions
            for p, particle in enumerate(self.particles):
                  particle.update_state(position=new_positions[p])


      def _euler_update_position(self, positions, velocities_1):
            """ 
            Update particle positions using the Euler integration method.

            Arguments:
            ----------
            positions : numpy.ndarray
                  Array of particle positions (N-by-3).
            velocities : numpy.ndarray
                  Array of particle velocities (N-by-3).

            Returns:
            --------
            numpy.ndarray
                  Updated particle positions (N-by-3).
            """
            print("• Used Euler scheme for advection.")
            return positions + velocities_1 * self.dt


      def _rk2_update_position(self, positions, velocities_1):
            """ 
            Update particle positions using the 2nd order Runge-Kutta (RK2) integration method.

            Arguments:
            ----------
            positions : numpy.ndarray
                  Array of particle positions (N-by-3).
            velocities_1 : numpy.ndarray
                  Array of particle velocities (N-by-3).

            Returns:
            --------
            numpy.ndarray
                  Updated particle positions (N-by-3).
            """
            # Get particle data
            strengths = self.get_particle_strengths()
            radii     = self.get_particle_radii()
  
            # Stage 1: Compute intermediate positions
            positions_1 = positions + velocities_1 * (self.dt * 0.5)

            # Stage 2: Compute velocities at the intermediate step
            velocities_2 = self.physics.get_selfinduced_velocity(positions_1, strengths, radii)

            # Final update
            positions_next = positions + velocities_2 * self.dt
            
            print("• Used second-order Runge-Kutta scheme for advection.")

            return positions_next

      
      def _rk3_update_position(self, positions, velocities_1):
            """ 
            Update particle positions using the 3rd order Runge-Kutta method with midpoint integration.

            Arguments:
            ----------
            positions : numpy.ndarray
                  Array of particle positions (N-by-3).
            velocities : numpy.ndarray
                  Array of particle velocities (N-by-3).

            Returns:
            --------
            numpy.ndarray
                  Updated particle positions (N-by-3).
            """
            strengths = self.get_particle_strengths()
            radii = self.get_particle_radii()

            # Stage 1: Initial update (standard RK3 step)
            positions_1 = positions + velocities_1 * (self.dt / 2)

            # Stage 2: Compute velocities at the half-step (midpoint update)
            velocities_2 = self.physics.get_selfinduced_velocity(positions_1, strengths, radii)
            positions_2 = positions + (self.dt / 2) * velocities_2

            # Stage 3: Compute velocities for the final update (using positions_2)
            velocities_3 = self.physics.get_selfinduced_velocity(positions_2, strengths, radii)

            # Final update using RK3 weights
            positions_next = positions + self.dt * velocities_3
            
            print("• Used third-order Runge-Kutta scheme for advection.")

            return positions_next
      
      def _rk4_update_position(self, positions, velocities_1):
            """ 
            Update particle positions using the 4th-order Runge-Kutta (RK4) method.

            Arguments:
            ----------
            positions : numpy.ndarray
                  Array of particle positions (N-by-3).
            velocities_1 : numpy.ndarray
                  Array of particle velocities (N-by-3) at the initial positions.

            Returns:
            --------
            numpy.ndarray
                  Updated particle positions (N-by-3).
            """
            strengths = self.get_particle_strengths()
            radii = self.get_particle_radii()

            # Stage 1: Initial velocities
            k1 = velocities_1

            # Stage 2: Compute velocities at intermediate step (positions + dt/2 * k1)
            positions_1 = positions + (self.dt / 2) * k1
            k2 = self.physics.get_selfinduced_velocity(positions_1, strengths, radii)

            # Stage 3: Compute velocities at another intermediate step (positions + dt/2 * k2)
            positions_2 = positions + (self.dt / 2) * k2
            k3 = self.physics.get_selfinduced_velocity(positions_2, strengths, radii)

            # Stage 4: Compute velocities at final step (positions + dt * k3)
            positions_3 = positions + self.dt * k3
            k4 = self.physics.get_selfinduced_velocity(positions_3, strengths, radii)

            # Combine the stages to compute the next positions
            positions_next = positions + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            print("• Used fourth-order Runge-Kutta scheme for advection.")

            return positions_next


      def update_dt(self, time_step_size: float): 
            self.dt = time_step_size


      def update_velocities(self):
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii     = self.get_particle_radii()

            velocities = self.physics.get_selfinduced_velocity(positions, strengths, radii)

            # Update the particles state:
            for p, particle in enumerate(self.particles): 
                  particle.update_state(velocity=velocities[p])

            print(f"• Updated particles' velocities")


      def update_strengths(self):
            """
            Update the strength of the particles using a third-order backward scheme (BDF3).
            """

            # If the flow is potential, ignore everything else.
            if self.flow_model == "Potential":
                  print("• Update_strengths() ignored because flow model is set to potential flow.")
                  return

            print(f"• Calculating the particles' strengths ::: {self.flow_model_description}")
            
            # Pre-compute particle data
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii     = self.get_particle_radii()

            # Update viscosity if LES model
            if self.flow_model == "LES":
                  
                  vorticity = self.get_particle_vorticities()
                  enstrophy = self.get_total_enstrophy()
                  
                  self._update_LES_filter_constant(enstrophy)
                  self._update_LES_particles_viscosity(vorticity, radii)


            # Initialize previous strengths if not already defined
            if not hasattr(self, 'previous_strengths'):
                  self.previous_strengths = np.zeros_like(strengths)
            if not hasattr(self, 'previous_previous_strengths'):
                  self.previous_previous_strengths = np.zeros_like(strengths)
            if not hasattr(self, 'previous_previous_previous_strengths'):
                  self.previous_previous_previous_strengths = np.zeros_like(strengths)


            if self.flow_model in {"LES", "DNS"}:
                  # Handle first time steps separately
                  if self.time_step == 0:
                        # Euler scheme for the first time step
                        dGamma_dt = self.physics.get_strength_gradients(positions, strengths, radii)
                        strengths += self.dt * dGamma_dt
                        print("\t• Used Euler scheme for the first time step for vortex stretching.")

                  elif self.time_step == 1:
                        # Second-order backward scheme (BDF2) for the second time step
                        dGamma_dt = self.physics.get_strength_gradients(positions, strengths, radii)
                        strengths = (4 * strengths - self.previous_strengths) / 3 + (2 / 3) * self.dt * dGamma_dt
                        print("\t• Used second-order backward difference scheme for the second time step for vortex stretching.")

                  elif self.time_step == 2:
                        # Third-order backward difference scheme (BDF3) for the third time step
                        dGamma_dt = self.physics.get_strength_gradients(positions, strengths, radii)
                        strengths = (
                              (18 * strengths - 9 * self.previous_strengths + 2 * self.previous_previous_strengths) / 11
                              + (6 / 11) * self.dt * dGamma_dt
                        )
                        print("\t• Used third-order backward difference scheme for the third time step for vortex stretching.")

                  else:
                        # Fourth-order backward difference scheme (BDF4)
                        dGamma_dt = self.physics.get_strength_gradients(positions, strengths, radii)
                        strengths = (
                              (48 * strengths - 36 * self.previous_strengths + 16 * self.previous_previous_strengths
                              - 3 * self.previous_previous_previous_strengths) / 25
                              + (12 / 25) * self.dt * dGamma_dt
                        )
                        print("\t• Used fourth-order backward difference scheme for vortex stretching.")

                  # Store the current and previous strengths for the next iteration
                  if self.time_step >= 3:
                        self.previous_previous_previous_strengths = self.previous_previous_strengths.copy()
                  if self.time_step >= 2:
                        self.previous_previous_strengths = self.previous_strengths.copy()
                  self.previous_strengths = strengths.copy()

                  # Update each particle's strength
                  for p, particle in enumerate(self.particles):
                        particle.update_state(strength=strengths[p])

                        
            # ===============================
            # Handle viscous diffusion via PSE
            # ===============================
            if self.viscous_scheme == 'PSE':

                  viscosities = self.get_particle_viscosities_eff()

                  if self.time_step == 0:
                        # Euler scheme for the first time step
                        dGamma_dt = self.physics.get_strength_gradients_PSE(positions, strengths, radii, viscosities)
                        
                        strengths += self.dt * dGamma_dt
                        print("\t• Used Euler scheme for viscous diffusion via PSE.")

                  elif self.time_step == 1:
                        # Second-order backward scheme (BDF2)
                        dGamma_dt = self.physics.get_strength_gradients_PSE(positions, strengths, radii, viscosities)
                        
                        strengths = (4 * strengths - self.previous_strengths) / 3 + (2 / 3) * self.dt * dGamma_dt
                        print("\t• Used second-order backward difference scheme for viscous diffusion via PSE.")

                  elif self.time_step == 2:
                        # Third-order backward difference scheme (BDF3)
                        dGamma_dt = self.physics.get_strength_gradients_PSE(positions, strengths, radii, viscosities)
                        
                        strengths = (
                              (18 * strengths - 9 * self.previous_strengths + 2 * self.previous_previous_strengths) / 11
                              + (6 / 11) * self.dt * dGamma_dt
                        )
                        print("\t• Used third-order backward difference scheme for viscous diffusion via PSE.")

                  else:
                        # Fourth-order backward difference scheme (BDF4)
                        dGamma_dt = self.physics.get_strength_gradients_PSE(positions, strengths, radii, viscosities)

                        strengths = (
                              (48 * strengths - 36 * self.previous_strengths + 16 * self.previous_previous_strengths
                              - 3 * self.previous_previous_previous_strengths) / 25
                              + (12 / 25) * self.dt * dGamma_dt
                        )
                        print("\t• Used fourth-order backward difference scheme for viscous diffusion via PSE.")

                  # Store the current and previous strengths for the next iteration
                  if self.time_step >= 3:
                        self.previous_previous_previous_strengths = self.previous_previous_strengths.copy()
                  if self.time_step >= 2:
                        self.previous_previous_strengths = self.previous_strengths.copy()
                  self.previous_strengths = strengths.copy()

                  # Update each particle's strength
                  for p, particle in enumerate(self.particles):
                        particle.update_state(strength=strengths[p])
                        

            # ===============================
            # Handle viscous diffusion via CS
            # ===============================
            elif self.viscous_scheme == 'CoreSpreading':
                  
                  viscosity_eff = self.get_particle_viscosities_eff()
                  
                  C = 5.34 # it should be 4, but I believe there is an issue when translating from the Gaussian kernel to the high algebraic.

                  new_radius = np.sqrt(radii**2 + C * viscosity_eff * self.dt)
                  
                  for p, particle in enumerate(self.particles):
                        particle.update_state(radius=new_radius[p])
                  
                  print("\t• Performed viscous diffusion via CS.")
                  
            # ===============================
            # Solution relaxation:
            # ===============================
            if self.relax_strength_solution:
                  alpha = 0.95

                  # Get vorticities and strengths
                  vorticity = self.get_particle_vorticities()  # n-by-3
                  strengths = self.get_particle_strengths()    # n-by-3

                  # Compute norms
                  vorticity_norm = compute_norm(vorticity, axis=1)  # n-by-1
                  strengths_norm = compute_norm(strengths, axis=1)  # n-by-1

                  # Normalize and scale
                  strengths = strengths * alpha + (1 - alpha) * (vorticity / vorticity_norm) * strengths_norm

                  # Update each particle's strength
                  for p, particle in enumerate(self.particles):
                        particle.update_state(strength=strengths[p])

                        
                        
      def _update_LES_filter_constant(self, enstrophy: float):
            """
            Dynamically adjusts the LES filter constant (Cnu) based on evolving flow properties using vorticity magnitudes.
            """
            alpha = 0.1     # Smoothing factor for reference updates
            
            # Compute root-mean-square (RMS) vorticity magnitude
            enstrophy_rms = np.sqrt(enstrophy)

            # Initialize or update the reference vorticity RMS
            if not hasattr(self, "enstrophy_ref"):
                  self.enstrophy_ref = enstrophy_rms

            # Smooth the reference value for stability
            self.enstrophy_ref = alpha * enstrophy_rms + (1 - alpha) * self.enstrophy_ref

            # Update LES filter constant (Cnu) dynamically
            self.Cnu *= enstrophy_rms / (self.enstrophy_ref + epsilon)
            
            # Cap Cnu to its maximum value found in the literature:
            self.Cnu = max(epsilon, min(self.Cnu, 0.0761))

            print(f"\t• Updated LES filter constant: Cν = {self.Cnu:0.3E} (dynamic adjustment)")


            


                  
      def _update_LES_particles_viscosity(self, vorticity: np.ndarray, radii: np.ndarray):
            """
            Update the viscosity of each particle based on their enstrophy and LES constant (Cnu).
            This represents the LES subgrid-scale (SGS) viscosity model.

            Arguments:
            ----------
            vorticity : np.ndarray
                  Array of vorticity vectors for each particle.
            radii : np.ndarray
                  Array of particle radii.

            Returns:
            --------
            None
            """
            beta_0 = np.pi / 4
            
            # Normalize each vorticity vector
            vorticity_magnitudes = compute_norm(vorticity, axis=1)  # Efficient row-wise norm
            vorticity_unit = vorticity / vorticity_magnitudes[:, None]  # Normalize each vorticity vector

            # Normalize average vorticity
            vorticity_avg = np.mean(vorticity, axis=0)
            vorticity_avg_magnitude = compute_norm(vorticity_avg, axis=0)  # Single vector norm
            vorticity_avg_unit = vorticity_avg / vorticity_avg_magnitude

            # Dot product for cosine of the angle
            cos_beta_m = np.dot(vorticity_unit, vorticity_avg_unit)
            beta_m = np.arccos(np.clip(cos_beta_m, -1.0, 1.0))  # Ensure values are within [-1, 1] for arccos

            # Assign psi_p based on beta_m and beta_0 threshold
            psi_p = np.where((beta_0 <= beta_m) & (beta_m <= np.pi - beta_0), 1.0, 0.0)
            
            # Compute viscosity update
            viscosity_t = psi_p * (self.Cnu * radii)**2 * vorticity_magnitudes  # Efficient use of precomputed magnitudes

            # Determine maximum viscosity
            viscosity_t_max = np.max(viscosity_t)
            
            if viscosity_t_max > epsilon:

                  # Update viscosity for each particle
                  for p, particle in enumerate(self.particles):
                        particle.update_state(viscosity_t=viscosity_t[p])

                  print(f"\t• Updated particles viscosity for LES model. max(νt) = {viscosity_t_max:0.3E} m²/s")




      # ========================================================== # 
      # Method for controlling the particle field:
      # ========================================================== # 
      def remove_weak_particles(self, mode: str, threshold: float, conserve_total_circulation=False):
            """ 
            Removes particles from the system whose strength is below a given threshold. The method can 
            operate in either 'absolute' or 'relative' mode, where in 'absolute' mode particles with strength 
            below a specified value are removed, and in 'relative' mode, particles with strength below a 
            percentage of the maximum strength are removed. The total circulation of the system can optionally 
            be conserved after removal by scaling the strengths of the remaining particles.

            Arguments:
            ----------
            mode : str
                  The mode of strength comparison. Options are:
                  - 'absolute': Uses a direct strength threshold.
                  - 'relative': Uses a relative threshold based on the maximum strength.
                  
            threshold : float
                  The strength threshold for removal. In 'absolute' mode, it is the direct strength value 
                  (in m³/s). In 'relative' mode, it is the fraction of the maximum strength.
                  
            conserve_total_circulation : bool, optional
                  Whether to scale the remaining particles' strengths to conserve the total circulation 
                  (i.e., total strength) of the system. Default is False.

            Returns:
            --------
            None
            """
            # Array with particle system strengths:
            particles_strength_mag = self.get_particle_strength_magnitudes()
            total_strength_before = np.sum(particles_strength_mag)
            

            # Check if there are particles and valid threshold
            if len(particles_strength_mag) == 0:
                  print("(warning) No particles available to evaluate.")
                  return
            
            if mode == 'absolute':
                  weak_particles_list = (particles_strength_mag < threshold)

            elif mode == 'relative':
                  highest_strength = np.max(particles_strength_mag)
                  
                  if highest_strength == 0:
                        print("(warning) All particle strengths are zero.")
                        weak_particles_list = np.ones_like(particles_strength_mag, dtype=bool)  # Remove all particles
                  else:
                        weak_particles_list = (particles_strength_mag / highest_strength < threshold)

            else:
                  print(f"(error) Mode '{mode}' not recognized. Use 'absolute' or 'relative'.")
                  return
            
            # Convert boolean array into array of indexes of Trues and remove particles:
            weak_particles = np.where(weak_particles_list)[0]
            self.remove_particles(weak_particles)

            if conserve_total_circulation:

                  particles_strength_mag = self.get_particle_strength_magnitudes()
                  total_strength_after   = np.sum(particles_strength_mag)

                  parricles_strength = self.get_particle_strengths()

                  correction = total_strength_before / total_strength_after

                  for p, particle in enumerate(self.particles): 
                        particle.update_state(strength=parricles_strength[p] * correction )


      def subdivide_strong_particles(self, top_percentage=None, strength_threshold=None):
            """ 
            Splits the strongest particles into two smaller particles, either based on the top percentage 
            or a strength threshold. The method will split particles with strength above the specified 
            threshold or from the top percentage of particles by strength. The strength of the split particles 
            is halved, and their positions are offset slightly based on the particle's radius and strength.

            Arguments:
            ----------
            top_percentage : float, optional
                  Percentage (between 0 and 1) of particles to split. This value is ignored if 
                  `strength_threshold` is provided.
                  
            strength_threshold : float, optional
                  The strength above which particles must be split into two. This value is ignored if 
                  `top_percentage` is provided.

            Returns:
            --------
            None
            """
            if (top_percentage is None) and (strength_threshold is None):
                  raise ValueError("Provide either top_percentage or strength_threshold.")

            # Array with particle system strengths:
            particles_strength_mag = self.get_particles_strength_magnitude()

            # Selection based on highest values:
            if strength_threshold is not None:
                  strongest_particles_indexes = get_highest_nonzero_indices(particles_strength_mag, strength_threshold=strength_threshold)

            # Selection based on percentage
            elif top_percentage is not None:
                  strongest_particles_indexes = get_highest_nonzero_indices(particles_strength_mag, top_percentage=top_percentage)

            # Return the top_n indexes
            strong_particles_list = [self.particles[p] for p in strongest_particles_indexes]

            for particle in strong_particles_list:
                  half_strength = particle.strength / 2
                  dx = particle.radius * particle.strength / particle.strength_magnitude

                  particleA_position = particle.position + dx * 0.25
                  particleB_position = particle.position - dx * 0.25

                  particleA = Particle(position=particleA_position, velocity=np.array([0, 0, 0]), strength=half_strength)
                  particleB = Particle(position=particleB_position, velocity=np.array([0, 0, 0]), strength=half_strength)

                  self.add_particle(particleA)
                  self.add_particle(particleB)

            self.remove_particles(strongest_particles_indexes)

            print("• Subdivided strongest particles")
            print(self)
            
            
      def remesh_solution(self, flow2D: bool = False):
            positions = self.get_particle_positions()
            strengths = self.get_particle_strengths()
            radii   = self.get_particle_radii()
            
            
            xmin, xmax = min(positions[:,0]), max(positions[:,0])
            ymin, ymax = min(positions[:,0]), max(positions[:,0])
            zmin, zmax = min(positions[:,0]), max(positions[:,0])
            
            spacing = (xmax-xmin) / len(positions)
            
            x = np.arange(xmin, xmax, step=spacing)
            y = np.arange(ymin, ymax, step=spacing)
            z = np.arange(zmin, zmax, step=spacing)

            # Create a grid of points
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

            # Calculate the volume of each element
            volume = spacing**3
            volumes = np.full(points.shape[0], volume)
            
            # Interpolate strengths from
            

@njit
def get_highest_nonzero_indices(arr, top_percentage=None, strength_threshold=None):
      """
      Returns the indices of the top N highest non-zero entries in the array.

      Parameters:
      -----------
      arr : np.ndarray
            The input array to search for highest non-zero entries.
      top_percentage : int, optional
            The percentage of top non-zero entries to return.
      strength_threshold : float, optional
            Threshold to select entries above a certain value.

      Returns:
      --------
      highest_nonzero_indices : np.ndarray
            Array of indices corresponding to the highest non-zero entries.
      """
      
      # Filter out zero entries
      non_zero_indices = np.nonzero(arr)[0]
      selected_values = arr[non_zero_indices]

      # Sort indices of non-zero values by descending order of values
      sorted_indices = np.argsort(selected_values)[::-1]

      if top_percentage is not None:
            # Calculate top_n based on the percentage
            top_n = max(int(top_percentage * len(non_zero_indices) / 100), 1)
            highest_nonzero_indices = non_zero_indices[sorted_indices[:top_n]]
      
      else:
            # Filter by strength threshold if provided
            if strength_threshold is not None:
                  valid_indices = selected_values > strength_threshold
                  selected_values = selected_values[valid_indices]
                  non_zero_indices = non_zero_indices[valid_indices]
                  sorted_indices = np.argsort(selected_values)[::-1]

            # If no top_percentage is provided, return all values above the strength threshold
            highest_nonzero_indices = non_zero_indices[sorted_indices]

      return highest_nonzero_indices