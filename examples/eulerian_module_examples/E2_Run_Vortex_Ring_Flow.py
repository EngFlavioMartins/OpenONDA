import os
import time
import numpy as np

from openONDA.solvers.FVM import fvmModule as fvm
from openONDA.utilities import (
    set_initial_condition,
    set_eulerian_module,
    set_boundary_conditions,
    vortex_ring_model
)

# ============================
# 🚀 Initialization of the Solver
# ============================

# Get the current working directory
current_dir = os.getcwd()

# Path to the OpenFOAM simulation case
openfoam_case_dir = os.path.join(current_dir, "E2_vortex_ring_flow")

# Prepare the OpenFOAM simulation environment
set_eulerian_module(current_dir, openfoam_case_dir)

# Initialize the Eulerian PIMPLE solver
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir}"])

# ============================
# 📊 Mesh and Simulation Setup
# ============================

# Get mesh properties
num_boundary_faces = solver.get_number_of_boundary_faces(patchName="numericalBoundary")
boundary_face_centers = solver.get_boundary_face_center_coordinates(patchName="numericalBoundary")

num_cells = solver.get_number_of_cells()
cell_centers = solver.get_cell_center_coordinates()

# ============================
# ⚙️ Simulation Parameters
# ============================

# Vortex characteristics
kinematic_viscosity = 1e-3  # m^2/s
stroke_time = 1.0           # s
vortex_strength = 2 * np.pi / 5  # m^2/s
vortex_center = np.array([-2.8, 0.0, 0.0], dtype=np.float64)  # m
vortex_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # m/s
vortex_radius = 1.5         # m
free_stream_velocity = np.array([0.0, 0.0, 0.0])  # m/s

# Time-stepping parameters
num_time_steps = 20
delta_t = 1.0e-1  # s

# Create vortex ring model
vortex_ring = vortex_ring_model(vortex_strength, kinematic_viscosity, stroke_time, vortex_center, vortex_velocity, vortex_radius)

# ============================
# 🌊 Initial Conditions
# ============================

# Compute induced velocity at cell centers and boundary faces
vx0, vy0, vz0, p0 = vortex_ring.get_induced_velocity(cell_centers)
vx_boundary, vy_boundary, vz_boundary, p_boundary = vortex_ring.get_induced_velocity(boundary_face_centers)

# Set initial conditions for the simulation
set_initial_condition(
    openfoam_case_dir, vx0, vy0, vz0, p0,
    vx_boundary, vy_boundary, vz_boundary, p_boundary
)

# Re-initialize the solver to apply custom boundary conditions
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir}"])

# Set the simulation time step
solver.set_time_step(delta_t)

# ============================
# 🔄 Time Integration Loop
# ============================

for step in range(num_time_steps):
    # Apply Lamb–Oseen vortex profile at boundaries
    vx_boundary, vy_boundary, vz_boundary, p_boundary = vortex_ring.get_induced_velocity(boundary_face_centers)

    # Apply updated boundary conditions
    set_boundary_conditions(solver, vx_boundary, vy_boundary, vz_boundary, p_boundary)

    # Evolve the solution and mesh
    solver.evolve()

    # Update the vortex ring's state (position, velocity, etc.)
    vortex_ring.update_vortex_ring_state(delta_t, free_stream_velocity)

    # Allow for proper solver communication
    time.sleep(0.1)