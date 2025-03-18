from numba import njit
import numpy as np

# Define Gaussian function for fitting
def gaussian(r, omega_0, a):
    return omega_0 * np.exp(-(r**2) / a**2)

epsilon = 1e-9

@njit(parallel=True, fastmath=True)
def compute_min_max(x, y, z):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    
    return xmin, xmax, ymin, ymax, zmin, zmax

# ============================================================= # 
# Useful scripts to distribuite particles:
# ============================================================= # 
def get_rectangular_point_distribuition(box_limits: np.ndarray, spacing: float):
      """ 
      Homogeneously distribute particles in a box-like region, with each point associated with a volume element.

      Arguments:
      ----------
      box_limits: Array of float
            [xmin, xmax, ymin, ymax, zmin, zmax] coordinates of the particles.
      spacing: float
            The average distance between the particles in the box.

      Returns:
      --------
      points : np.ndarray
            The 3D coordinates of the points in the box.
      volumes : np.ndarray
            The volume associated with each point.
      """
      # Compute the number of points along each axis, including boundaries
      numx = int(np.ceil((box_limits[1] - box_limits[0]) / spacing)) + 1
      numy = int(np.ceil((box_limits[3] - box_limits[2]) / spacing)) + 1
      numz = int(np.ceil((box_limits[5] - box_limits[4]) / spacing)) + 1

      # Ensure an odd number of points
      if numx % 2 == 0:
            numx += 1
      if numy % 2 == 0:
            numy += 1
      if numz % 2 == 0:
            numz += 1
      
      x = np.linspace(box_limits[0], box_limits[1], num=numx)
      y = np.linspace(box_limits[2], box_limits[3], num=numy)
      z = np.linspace(box_limits[4], box_limits[5], num=numz)

      # Create a grid of points
      xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
      points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

      # Calculate the total box volume
      total_box_volume = (box_limits[1] - box_limits[0]) * (box_limits[3] - box_limits[2]) * (box_limits[5] - box_limits[4])

      # Assign an equal volume to each point, ensuring total volume matches
      total_points = points.shape[0]
      volume_per_point = total_box_volume / total_points
      volumes = np.full(total_points, volume_per_point)

      return points, volumes


def get_2D_rectangular_point_distribuition(box_limits: np.ndarray, spacing: float):
      """ 
      Homogeneously distribute particles in a box-like region, with each point associated with a volume element.

      Arguments:
      ----------
      box_limits: Array of float
            [xmin, xmax, ymin, ymax, zmin, zmax] coordinates of the particles.
      spacing: float
            The average distance between the particles in the box.

      Returns:
      --------
      points : np.ndarray
            The 3D coordinates of the points in the box.
      volumes : np.ndarray
            The volume associated with each point.
      """
      # Compute the number of points along each axis, including boundaries
      numx = int(np.ceil((box_limits[1] - box_limits[0]) / spacing)) + 1
      numy = int(np.ceil((box_limits[3] - box_limits[2]) / spacing)) + 1


      # Ensure an odd number of points
      if numx % 2 == 0:
            numx += 1
      if numy % 2 == 0:
            numy += 1
      
      x = np.linspace(box_limits[0], box_limits[1], num=numx)
      y = np.linspace(box_limits[2], box_limits[3], num=numy)

      # Create a grid of points
      xx, yy = np.meshgrid(x, y, indexing='ij')
      zz = np.zeros_like(xx)  # Z-coordinates are all zero
      points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

      # Calculate the total box volume
      total_box_volume = (box_limits[1] - box_limits[0]) * (box_limits[3] - box_limits[2])

      # Assign an equal volume to each point, ensuring total volume matches
      total_points = points.shape[0]
      volume_per_point = total_box_volume / total_points
      volumes = np.full(total_points, volume_per_point)

      return points, volumes


@njit
def get_hexagonal_point_distribution(box_limits: np.ndarray, spacing: float):
      """ 
      Homogeneously distribute particles in a box-like region with a hexagonal pattern in the xy-plane, 
      with each point associated with a volume element.

      Arguments:
      ----------
      box_limits: Array of float
            [xmin, xmax, ymin, ymax, zmin, zmax] coordinates of the particles
      spacing: float
            The average distance between particles in the xy-plane

      Returns:
      --------
      points : np.ndarray
            The 3D coordinates of the points in the box.
      volumes : np.ndarray
            The volume associated with each point.
      """
      # Generate x and y coordinates for hexagonal lattice in the xy-plane
      x_min, x_max, y_min, y_max, z_min, z_max = box_limits
      x_range = np.arange(x_min, x_max + spacing, spacing)
      y_range = np.arange(y_min, y_max + spacing * np.sqrt(3) / 2, spacing * np.sqrt(3) / 2)
      z_range = np.arange(z_min, z_max + spacing, spacing)

      points = []
      for z in z_range:
            for i, y in enumerate(y_range):
                  x_offset = spacing / 2 if i % 2 == 1 else 0  # Offset every other row
                  for x in x_range:
                        points.append([x + x_offset, y, z])

      # Convert to numpy array
      points = np.array(points)

      # Calculate the volume of each element
      volume = spacing**3 * np.sqrt(3) / 2  # Volume for hexagonal spacing in 3D
      volumes = np.full(points.shape[0], volume)

      return points, volumes



@njit(parallel=True, fastmath=True)
def get_cylindrical_point_distribuition(particle_distance: float, radius=1.0, height=2.0, center=np.array([0,0,0]), axis='x'):
      """
      Generate a regular grid of points inside a cylinder, with each point associated with a volume element.

      Parameters:
      -----------
      particle_distance : float
            The average distance between the particles, m
      radius : float
            The radius of the cylinder, m
      height : float
            The height of the cylinder, m
      center: 3-by-1 numpy array
            Describes the center of the domain, m
      axis : str
            The axis along which the cylinder is aligned ('x', 'y', or 'z')

      Returns:
      --------
      points : np.ndarray
            The 3D coordinates of the points in the cylinder.
      volumes : np.ndarray
            The volume associated with each point.
      """

      num_points_radius = int(radius/particle_distance)
      num_points_height = int(height/particle_distance)
      num_points_theta  = int(np.pi * radius/particle_distance)

      
      # Initialize lists to hold the points and their volumes
      points = []
      volumes = []

      # Differential elements
      dtheta = 2 * np.pi / (num_points_theta)
      dz = height / (num_points_height)
      dr = radius/num_points_radius
      rmin = dr
      
      
      r = np.linspace(rmin, radius, num=num_points_radius)
      z = np.linspace(-height/2, height/2, num=num_points_height+1)
      theta = np.linspace(0, 2*np.pi, num=num_points_theta)
      theta = theta[:-1]

      # Convert the grid to Cartesian coordinates and calculate volumes
      for ri in r:
            for zi in z:
                  for thetai in theta:
                        x = ri * np.cos(thetai) 
                        y = ri * np.sin(thetai)
                        
                        # Assign Cartesian coordinates based on axis
                        if axis == 'x':
                              point = [zi+center[0], x+center[1], y+center[2]] 
                        elif axis == 'y':
                              point = [x+center[0], zi+center[1], y+center[2]] 
                        elif axis == 'z':
                              point = [x+center[0], y+center[1], zi+center[2]] 
                        else:
                              raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

                        # Calculate the differential volume element
                        volume = (ri * dtheta) * dr * dz
                        points.append(point)
                        volumes.append(volume)
                        
                        
      # Add points along the central axis (r = 0)
      for zi in z:
            if axis == 'x':
                  point = [zi + center[0], center[1], center[2]]
            elif axis == 'y':
                  point = [center[0], zi + center[1], center[2]]
            elif axis == 'z':
                  point = [center[0], center[1], zi + center[2]]
            else:
                  raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

            # The volume for axis points includes the entire cylinder slice at that z
            volume = np.pi * (radius**2) * dz / len(z)
            points.append(point)
            volumes.append(volume)


      return np.array(points), np.array(volumes)



# ============================================================= # 
# Scripts for sorting and selecting values:
# ============================================================= # 

