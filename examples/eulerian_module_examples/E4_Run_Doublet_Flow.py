import os
import time
import numpy as np

from OpenONDA.solvers.FVM import fvmModule as fvm
from OpenONDA.utilities import (
    set_initial_condition,
    set_eulerian_module,
    set_boundary_conditions,
    doublet_flow_model
)

# ============================
# 🚀 Initialization of the Solver
# ============================

# Get current working directory
current_dir = os.getcwd()

# Paths to the OpenFOAM simulation cases
openfoam_case_dir_A = os.path.join(current_dir, "E4_doublet_flow", "domainA")
openfoam_case_dir_B = os.path.join(current_dir, "E4_doublet_flow", "domainB")

# Prepare the OpenFOAM simulation environments
set_eulerian_module(current_dir, openfoam_case_dir_A)
set_eulerian_module(current_dir, openfoam_case_dir_B)

# Initialize the Eulerian PIMPLE solvers
solver_A = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir_A}"])
solver_B = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir_B}"])

# ============================
# 📊 Mesh and Boundary Setup
# ============================

# Get the number of boundary faces
num_boundary_faces_A = solver_A.get_number_of_boundary_faces(patchName="numericalBoundary")
num_boundary_faces_B = solver_B.get_number_of_boundary_faces(patchName="numericalBoundary")

# Get the coordinates of the boundary face centers
boundary_face_centers_A = solver_A.get_boundary_face_center_coordinates(patchName="numericalBoundary")
boundary_face_centers_B = solver_B.get_boundary_face_center_coordinates(patchName="numericalBoundary")

# Get number of cells and their centers
num_cells_A = solver_A.get_number_of_cells()
num_cells_B = solver_B.get_number_of_cells()

cell_centers_A = solver_A.get_cell_center_coordinates()
cell_centers_B = solver_B.get_cell_center_coordinates()

# ============================
# ⚙️ Simulation Parameters
# ============================

U_inf = np.array([0.0, 0.0, 10.0], dtype=np.float64)  # Background velocity
dipole_radius = 1.0  # Sphere radius producing the dipole, m
dipole_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
dipole_center = np.array([0.0, 0.0, -1.51], dtype=np.float64)  # m
dipole_strength = 2 * np.pi * np.linalg.norm(U_inf) * dipole_radius ** 3  # m^4/s
n_time_steps = 20
dt = 1.0e-3  # s

print(f"(INFORMATION) Dipole strength: {dipole_strength:8.2f} m**4/s")

# Create doublet flow model
doublet_flow = doublet_flow_model(dipole_center, dipole_direction, dipole_strength)

# ============================
# 🌊 Initial Conditions
# ============================

# Initial conditions for both domains
vx0_A, vy0_A, vz0_A, p0_A = doublet_flow.get_induced_velocity(cell_centers_A, U_inf)
vx0_B, vy0_B, vz0_B, p0_B = doublet_flow.get_induced_velocity(cell_centers_B, U_inf)

# Boundary conditions
vx_boundary_A, vy_boundary_A, vz_boundary_A, p_boundary_A = doublet_flow.get_induced_velocity(boundary_face_centers_A, U_inf)
vx_boundary_B, vy_boundary_B, vz_boundary_B, p_boundary_B = doublet_flow.get_induced_velocity(boundary_face_centers_B, U_inf)

# Apply initial conditions
set_initial_condition(openfoam_case_dir_A, vx0_A, vy0_A, vz0_A, p0_A, vx_boundary_A, vy_boundary_A, vz_boundary_A, p_boundary_A)
set_initial_condition(openfoam_case_dir_B, vx0_B, vy0_B, vz0_B, p0_B, vx_boundary_B, vy_boundary_B, vz_boundary_B, p_boundary_B)

# Re-initialize solvers with custom boundary conditions
solver_A = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir_A}"])
solver_B = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir_B}"])

solver_A.set_time_step(dt)
solver_B.set_time_step(dt)

# ============================
# 🔄 Flow Simulation Loop
# ============================

for time_step in range(n_time_steps):
    # Apply Doublet Flow boundary conditions
    vx_boundary_A, vy_boundary_A, vz_boundary_A, p_boundary_A = doublet_flow.get_induced_velocity(boundary_face_centers_A, U_inf)
    vx_boundary_B, vy_boundary_B, vz_boundary_B, p_boundary_B = doublet_flow.get_induced_velocity(boundary_face_centers_B, U_inf)

    # Set boundary conditions
    set_boundary_conditions(solver_A, vx_boundary_A, vy_boundary_A, vz_boundary_A, p_boundary_A)
    set_boundary_conditions(solver_B, vx_boundary_B, vy_boundary_B, vz_boundary_B, p_boundary_B)

    # Evolve solutions
    solver_A.evolve()
    time.sleep(0.1)  # Ensure proper solver communication

    solver_B.evolve()
    time.sleep(0.1)  # Ensure proper solver communication

    # Update boundary face center coordinates
    boundary_face_centers_A = solver_A.get_boundary_face_center_coordinates(patchName="numericalBoundary")
    boundary_face_centers_B = solver_B.get_boundary_face_center_coordinates(patchName="numericalBoundary")
