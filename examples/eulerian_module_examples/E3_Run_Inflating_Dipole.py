import os
import time
import numpy as np

from openONDA.solvers.FVM import fvmModule as fvm
from openONDA.utilities import (
    set_initial_condition,
    set_eulerian_module,
    set_boundary_conditions,
    inflating_dipole_model
)

# ==========================
# 🚀 Initialization of the Solver
# ==========================

# Define directories
current_dir = os.getcwd()  # Current working directory
openfoam_case_dir = os.path.join(current_dir, "E3_inflating_dipole")  # OpenFOAM case directory

# Set up the OpenFOAM simulation environment
set_eulerian_module(current_dir, openfoam_case_dir)

# Initialize the Eulerian PIMPLE solver
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", openfoam_case_dir])

# Retrieve mesh information
num_boundary_faces = solver.get_number_of_boundary_faces(patchName="numericalBoundary")
boundary_face_centers = solver.get_boundary_face_center_coordinates(patchName="numericalBoundary")
num_cells = solver.get_number_of_cells()
cell_centers = solver.get_cell_center_coordinates()

# ==========================
# ⚙️ Simulation Parameters
# ==========================

dt = 2.5e-3  # Time step (s)
kinematic_viscosity = 1.0e-2  # Kinematic viscosity (m^2/s)
free_stream_velocity = np.array([0.50, 0, 0], dtype=np.float64)  # Free-stream velocity (m/s)
dipole_center = np.array([-1.5, 0, 0], dtype=np.float64)  # Dipole center (m)
dipole_char_velocity = 0.5  # Dipole characteristic velocity (m/s)
dipole_char_length = 0.5  # Dipole characteristic length (m)
n_time_steps = 20  # Number of time steps

# ==========================
# 🌪️ Dipole Initialization
# ==========================

# Create inflating dipole model
inflating_dipole = inflating_dipole_model(dipole_center, dipole_char_velocity, dipole_char_length, kinematic_viscosity)

# Set initial conditions for velocity and pressure
vx0, vy0, vz0, p0 = inflating_dipole.get_induced_velocity(cell_centers, free_stream_velocity)

# Calculate boundary conditions
vx_boundary, vy_boundary, vz_boundary, p_boundary = inflating_dipole.get_induced_velocity(
    boundary_face_centers, free_stream_velocity
)

# Apply initial conditions
set_initial_condition(
    openfoam_case_dir, vx0, vy0, vz0, p0,
    vx_boundary, vy_boundary, vz_boundary, p_boundary
)

# Re-initialize the solver with custom boundary conditions
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", openfoam_case_dir])
solver.set_time_step(dt)

# ==========================
# 🔄 Time-Stepping Loop
# ==========================

for time_step in range(n_time_steps):
    # Update boundary conditions with the dipole's current state
    vx_boundary, vy_boundary, vz_boundary, p_boundary = inflating_dipole.get_induced_velocity(
        boundary_face_centers, free_stream_velocity
    )
    set_boundary_conditions(solver, vx_boundary, vy_boundary, vz_boundary, p_boundary)

    # Evolve the flow solution and mesh
    solver.evolve()

    # Update mesh coordinates after evolution
    boundary_face_centers = solver.get_boundary_face_center_coordinates(patchName="numericalBoundary")

    # Update dipole's dynamic state
    inflating_dipole.update_dipole_state(dt, free_stream_velocity)

    # Ensure smooth communication between solvers
    time.sleep(0.1)