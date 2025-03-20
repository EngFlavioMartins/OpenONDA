import numpy as np
import sys
import os

# Import all relevant moduli:
from openONDA.solvers.VPM import vpmModule as vpm

import openONDA.utilities.vpm_flow_models   as vpm_fm
import openONDA.utilities.vpm_solver_helper as vpm_sh
from   openONDA.utilities.scripts_helper import remove_files

# ==================================================
# Set up some basic IO
# ==================================================
filesnames      = "Lamb_Oseen_Vortex"
case_directory  = "./E1_Lamb_Oseen_Vortex/Raw"
backup_filename = os.path.join(case_directory, filesnames)

# Ensure the directory exists
os.makedirs(os.path.dirname(backup_filename), exist_ok=True)

log_file_path = './E1_Lamb_Oseen_Vortex/Lamb_Oseen_Vortex.out'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Redirect stdout and stderr to a log file
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

remove_files(case_directory, filesnames)

# ==================================================
# Properties of the vortex filament model
# ==================================================
core_radius       = 1.0
vortex_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
vortex_strength   = np.pi                       # m²/s, vortex strength

# ==================================================
# Particle Distribution Setup
# ==================================================
particle_distance  = 0.25 * core_radius    # m >>> 0.2
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = np.pi*1e-3   # m²/s, kinematic viscosity
time_step_size     = 3 * particle_distance**2/vortex_strength  # s
n_time_steps       = int( 100 / vortex_strength / time_step_size) ### >>> 10
backup_frequency   = 5

# ==================================================
# Properties of the Particle Distribution Region
# ==================================================
box_domain = np.array([-3., 3., -3., 3., -10., 10.]) * core_radius

# Get the coordinates of regularly-distribuited 
positions, volumes = vpm_sh.get_hexagonal_point_distribution(box_domain, spacing=particle_distance)

# ==================================================
# Compute Particle Velocities, Strengths, Radii,
# viscosities and add them to the particle system:
# ==================================================
velocities, strengths, radii, viscosities = vpm_fm.lamb_oseen_vpm(
    positions, volumes, particle_radius, particle_viscosity, vortex_center, vortex_strength, core_radius)

# ==================================================
# Initialize the particle system
# ==================================================
particle_system = vpm.ParticleSystem(
    flow_model='LES',
    time_step_size=time_step_size,
    time_integration_method='RK2', 
    viscous_scheme='CoreSpreading',
    processing_unit='GPU',
    monitor_variables = ['Circulation', 'Kinetic energy'],
    backup_filename=backup_filename,
    backup_frequency=backup_frequency
)

particle_system.add_particle_field(positions, velocities, strengths, radii, viscosities)

particle_system.remove_weak_particles(mode='relative', threshold=1e-1, conserve_total_circulation=True)

print(particle_system)

# ==================================================
# Perform the Simulation Over n Time Steps
# ==================================================
for t in range(n_time_steps):
    particle_system.update_velocities()
    particle_system.update_strengths()
    particle_system.update_state()