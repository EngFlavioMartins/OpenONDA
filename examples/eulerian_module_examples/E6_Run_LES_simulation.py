import os
import time
import numpy as np

from OpenONDA.solvers.FVM import fvmModule as fvm
from OpenONDA.utilities import (
    set_initial_condition,
    set_eulerian_module,
    set_boundary_conditions
)

# ============================
# 🚀 Initialization of the Solver
# ============================

# Get current working directory
current_dir = os.getcwd()
    
# Paths to the OpenFOAM simulation cases
openfoam_case_dir = os.path.join(current_dir, "E6_LES_simulation")

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
# 🔄 Set initial and b.c.
# ============================
x = cell_centers[:,0]
u0, v0, w0, p0 = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

umax = 0.1 # m/s

xb = boundary_faces[:,0]
u0_boundary, v0_boundary, w0_boundary, p0_boundary = umax*np.ones_like(xb), umax*np.ones_like(xb), umax*np.ones_like(xb), np.zeros_like(xb)

set_initial_condition(openfoam_case_dir, u0, v0, w0, p0, u0_boundary, v0_boundary, w0_boundary, p0_boundary)


# ============================
# 🔄 Run simulation
# ============================
for time_step in range(600):
    
    solver.set_dirichlet_velocity_boundary_condition(u0_boundary, v0_boundary, w0_boundary, patchName="numericalBoundary")
    solver.set_dirichlet_pressure_boundary_condition(p0_boundary, patchName="numericalBoundary")
    
    solver.evolve()
    time.sleep(0.1)