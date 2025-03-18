import numpy as np

# ========================================================== # 
# Definition of Particle class:
# ========================================================== # 

class Particle:
    """
    Represents a particle in a fluid flow simulation.

    Arguments:
    - position   : Optional 3x1 numpy array of floats representing the particle's 3D coordinates.
    - velocity   : Optional 3x1 numpy array of floats for the particle's velocity components.
    - strength   : Optional 3x1 numpy array of floats representing the circulation vector components.
    - radius     : Optional float representing the particle's radius (in meters).
    - viscosity  : Optional float, representing the particle's viscosity.
    - viscosity_t: Optional float, representing the sub-grid-scale viscosity.
    - domain_size: Optional float, defining the range for randomly generated x, y, z coordinates (default is 2.0).

    Attributes:
    - position   : 3x1 numpy array with the particle's coordinates [x, y, z] (in meters).
    - velocity   : 3x1 numpy array with the particle's velocity components [ux, uy, uz] (in m/s).
    - strength   : 3x1 numpy array with the particle's strength components [gx, gy, gz] (in m³/s).
    - radius     : Float, radius of the particle (in meters).
    - volume     : Float, calculated volume of the particle (in m³).
    - viscosity  : Float, representing particle viscosity (m²/s).
    - viscosity_t: Float, representing sub-grid-scale viscosity (m²/s).
    - viscosity_eff: Float, effective viscosity (m²/s).
    - velocity_magnitude: Float, magnitude of the particle's velocity.
    - strength_magnitude: Float, magnitude of the particle's strength.
    """

    def __init__(self, position: np.array=None, 
                 velocity: np.array=None, 
                 strength: np.array=None, 
                 radius: float=None, 
                 viscosity: float=None, 
                 viscosity_t: float=None,
                 domain_size: float=1.0,
                 group_id: int = 0):

        # Initialize position
        self.position = np.array(position, dtype=np.float64) if position is not None else domain_size * np.random.random(3) - 1.0

        # Initialize velocity
        self.velocity = np.array(velocity if velocity is not None else np.zeros(3), dtype=np.float64)
        self.velocity_magnitude = np.linalg.norm(self.velocity)

        # Initialize strength
        self.strength = np.array(strength if strength is not None else np.zeros(3), dtype=np.float64)
        self.strength_magnitude = np.linalg.norm(self.strength)

        # Initialize radius
        self.radius = float(radius) if radius is not None else 0.01

        # Initialize kinematic viscosity and sub-grid-scale viscosity:
        self.viscosity = float(viscosity) if viscosity is not None else 0.0
        self.viscosity_t = float(viscosity_t) if viscosity_t is not None else 0.0

        # Effective viscosity is the sum of kinematic viscosity and sub-grid-scale viscosity:
        self.viscosity_eff = self.viscosity + self.viscosity_t
        
        # ID of the particle group. This helps keeping track of particle groups
        self.group_id = group_id


    def update_state(self, position: np.array = None, 
                     velocity: np.array = None, 
                     strength: np.array = None, 
                     radius: float = None, 
                     viscosity: float = None, 
                     viscosity_t: float = None):
        """ 
        Update the particle's state.
        """

        if position is not None:
            assert position.size == 3, "Position array must have 3 elements."
            self.position = np.array(position, dtype=np.float64)

        if velocity is not None:
            assert velocity.size == 3, "Velocity array must have 3 elements."
            self.velocity = np.array(velocity, dtype=np.float64)
            self.velocity_magnitude = np.linalg.norm(self.velocity)

        if strength is not None:
            assert strength.size == 3, "Strength array must have 3 elements."
            self.strength = np.array(strength, dtype=np.float64)
            self.strength_magnitude = np.linalg.norm(self.strength)

        if radius is not None:
            assert isinstance(radius, float), "Radius must be a float."
            self.radius = radius

        if viscosity is not None:
            assert isinstance(viscosity, float), "Viscosity must be a float."
            self.viscosity = viscosity

        if viscosity_t is not None:
            assert isinstance(viscosity_t, float), "Subgrid-scale viscosity must be a float."
            self.viscosity_t = viscosity_t

        # Update effective viscosity if necessary
        if viscosity is not None or viscosity_t is not None:
            self.viscosity_eff = self.viscosity + self.viscosity_t

    def __str__(self):
        return (f"Location:({self.position[0]:.2E},{self.position[1]:.2E},{self.position[2]:.2E}) m, "
                f"Velocity:({self.velocity[0]:.2E},{self.velocity[1]:.2E},{self.velocity[2]:.2E}) m/s, "
                f"Strength:({self.strength[0]:.2E},{self.strength[1]:.2E},{self.strength[2]:.2E}) m³/s, "
                f"Radius: {self.radius:.2E} m, "
                f"Viscosity: {self.viscosity:.2E} m²/s, "
                f"Subgrid-scale viscosity: {self.viscosity_t:.2E} m²/s")

