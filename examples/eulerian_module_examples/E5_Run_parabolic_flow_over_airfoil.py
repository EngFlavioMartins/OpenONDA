import os
import time
import numpy as np

from OpenONDA.solvers.FVM import fvmModule as fvm
from OpenONDA.utilities import (
    set_initial_condition,
    set_eulerian_module,
    set_boundary_conditions
)

def parabolic_velocity_with_airfoil_pressure(points, U0=10, p0=1, p1=3, sigma_factor=0.2):
    """
    Compute a parabolic velocity profile in the xy-plane with a bell-shaped airfoil-like pressure profile.

    - Velocity is always in the x-direction (u), while v and w remain zero.
    - Velocity varies along the y-axis, being 0 at y_min and U0 at y_max.
    - Pressure follows a Gaussian-like distribution with the minimum at the trailing edge (y_TE).

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        The coordinates of the points (cell centers or boundary faces).
    U0 : float, optional
        The reference (maximum) velocity (default is 10).
    p0 : float, optional
        The baseline pressure level (default is 1).
    p1 : float, optional
        The baseline pressure drop amplitude (default is 3).
    sigma_factor : float, optional
        Controls the width of the low-pressure region (default is 0.2).

    Returns
    -------
    u, v, w, p : ndarrays of shape (N,)
        The velocity components (u, v, w) and a pressure field (p).
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be a 2D array with shape (N,3)")

    # Extract y-coordinates
    y = points[:, 1]

    # Find min and max y-coordinates
    y_min = np.min(y)
    y_max = np.max(y)

    if np.isclose(y_min, y_max):
        raise ValueError("All points have the same y-coordinate; no velocity variation possible.")

    # Define the trailing edge position in y
    y_TE = (y_max + y_min) / 2  # Midpoint as an approximation of trailing edge line

    # Normalize y to [0, 1] range
    y_normalized = (y - y_min) / (y_max - y_min)

    # Parabolic velocity profile: U = U0 * (1 - (1 - y_normalized)^2)
    u = U0 * (1 - (1 - y_normalized) ** 2)

    # Other velocity components are zero
    v = np.zeros_like(y)
    w = np.zeros_like(y)

    # Gaussian-like pressure profile
    sigma = sigma_factor * (y_max - y_min)  # Width of the low-pressure region
    p = p0 - p1 * np.exp(-((y - y_TE) ** 2) / (2 * sigma ** 2))

    return u, v, w, p



# ============================
# 🚀 Initialization of the Solver
# ============================

# Get current working directory
current_dir = os.getcwd()
    
# Paths to the OpenFOAM simulation cases
openfoam_case_dir = os.path.join(current_dir, "E5_parabolic_flow_over_airfoil")

# Prepare the OpenFOAM simulation environments
set_eulerian_module(current_dir, openfoam_case_dir)

# Initialize the Eulerian PIMPLE solvers
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir}"])

# ============================
# 📊 Mesh and Boundary Setup
# ============================
cell_centers    = solver.get_cell_center_coordinates()
boundary_faces  = solver.get_boundary_face_center_coordinates()

# ============================
# 🔄 Get info from donor
# ============================
#u0, v0, w0, p0 = parabolic_velocity_with_airfoil_pressure(cell_centers, U0=10, p0=-3, p1=6)

x = cell_centers[:,0]
u0, v0, w0, p0 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

u0_boundary, v0_boundary, w0_boundary, p0_boundary = parabolic_velocity_with_airfoil_pressure(boundary_faces, U0=10, p0=-3, p1=6)


# Set internal and boundary fields to the values obtained from the parabolic velocity profile routine:
set_initial_condition(openfoam_case_dir, u0, v0, w0, p0, u0_boundary, v0_boundary, w0_boundary, p0_boundary)

for time_step in range(600):
    
    solver.set_dirichlet_velocity_boundary_condition(u0_boundary, v0_boundary, w0_boundary, patchName="numericalBoundary")
    solver.set_dirichlet_pressure_boundary_condition(p0_boundary, patchName="numericalBoundary")
    
    solver.evolve()
    time.sleep(0.1)