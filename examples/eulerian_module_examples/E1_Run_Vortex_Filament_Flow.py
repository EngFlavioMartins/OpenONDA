import os
import time
import numpy as np

from openONDA.solvers.FVM import fvmModule as fvm
from openONDA.utilities import (
    set_initial_condition,
    set_eulerian_module,
    set_boundary_conditions,
    vortex_filament_model
)


# ============================
# 🚀 Initialization of the Solver
# ============================

# Get the current working directory
current_dir = os.getcwd()

# Path to the OpenFOAM simulation case
openfoam_case_dir = os.path.join(current_dir, "E1_vortex_filament_flow")

# Prepare the OpenFOAM simulation environment
set_eulerian_module(current_dir, openfoam_case_dir)

# Initialize the Eulerian PIMPLE solver
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir}"])

# ============================
# 📊 Mesh and Geometry Information
# ============================

# Retrieve the number of boundary faces for the specified patch
num_boundary_faces = solver.get_number_of_boundary_faces()

# Get the coordinates of boundary face centers
boundary_face_centers = solver.get_boundary_face_center_coordinates()

# Retrieve the total number of cells in the mesh
num_cells = solver.get_number_of_cells()

# Get the coordinates of the cell centers
cell_centers = solver.get_cell_center_coordinates()

# ============================
# ⚙️ Simulation Parameters
# ============================

num_time_steps = 20             # Total number of simulation steps
time_step_size = 2.5e-3         # Time step size (s)
kinematic_viscosity = 1.0e-2    # Kinematic viscosity (m^2/s)
initial_time = 1.0              # Initial vortex time (s)

# Vortex properties
vortex_center = np.array([1.2, 0, 0], dtype=np.float64)  # Vortex center coordinates (m)
vortex_strength = 2 * np.pi  # Vortex circulation strength (m^2/s)

# Create a vortex filament object to induce velocities
vortex_filament = vortex_filament_model(vortex_center, vortex_strength, kinematic_viscosity, initial_time)

# Compute initial velocity and pressure fields
vx0, vy0, vz0, p0 = vortex_filament.get_induced_velocity(cell_centers)

# Compute boundary conditions
vx_boundary, vy_boundary, vz_boundary, p_boundary = vortex_filament.get_induced_velocity(boundary_face_centers)

# ============================
# 📥 Set Initial Conditions
# ============================

# This block sets custom initial conditions for OpenFOAM
set_initial_condition(
    openfoam_case_dir, vx0, vy0, vz0, p0,
    vx_boundary, vy_boundary, vz_boundary, p_boundary
)

# Re-initialize the solver after setting boundary conditions
solver = fvm.pyFoamSolver(["pimpleStepperFoam", "-case", f"{openfoam_case_dir}"])

# Set the time step size for the simulation
solver.set_time_step(time_step_size)

# ============================
# 🔄 Simulation Loop
# ============================

# Performance timers
phiflow_runtime = 0.0
openfoam_runtime = 0.0

for step in range(num_time_steps):
    # === pHyFlow Block ===
    start_time = time.time()

    # Apply vortex-induced velocity to boundary faces
    vx_boundary, vy_boundary, vz_boundary, p_boundary = vortex_filament.get_induced_velocity(boundary_face_centers)

    # Advance the vortex model in time
    vortex_filament.update_vortex_time(time_step_size)

    # Apply boundary conditions to the solver
    set_boundary_conditions(solver, vx_boundary, vy_boundary, vz_boundary, p_boundary)

    # Update boundary face center coordinates after mesh evolution
    boundary_face_centers = solver.get_boundary_face_center_coordinates(patchName="numericalBoundary")

    phiflow_runtime += time.time() - start_time

    # === OpenFOAM Block ===
    start_time = time.time()

    # Evolve the flow solution and mesh
    solver.evolve()

    openfoam_runtime += time.time() - start_time

    # Small delay to ensure proper communication between solvers
    time.sleep(0.1)

# ============================
# 📋 Summary of Simulation Performance
# ============================

summary = (
    f"openONDA runtime: {phiflow_runtime:.3e} s\n"
    f"OpenFOAM runtime: {openfoam_runtime:.3e} s\n"
)

print(summary)