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
filesnames      = "Vortex_Ring"
case_directory  = "./E2_Vortex_Ring/Raw"
backup_filename = os.path.join(case_directory, filesnames)

# Ensure the directory exists
os.makedirs(os.path.dirname(backup_filename), exist_ok=True)

log_file_path = './E2_Vortex_Ring/Vortex_Ring.out'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Redirect stdout and stderr to a log file
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

remove_files(case_directory, filesnames)

# ==================================================
# Properties of the Vortex Ring
# ==================================================
ring_center   = np.array([0.0, 0.0, 0.0]) # m, center of the vortex
ring_radius   = 1.0              # m, radius of the vortex ring
ring_strength = 1.0              # m²/s, vortex strength
ring_thickness = 0.2*ring_radius # m, thickness of the vortex ring

# ==================================================
# Particle Distribution Setup
# ==================================================
Re = 750                                   # Reynolds number
particle_distance  = 0.25*ring_thickness    # m
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = ring_strength/Re       # m²/s, kinematic viscosity
time_step_size     = 5 * particle_distance**2/ring_strength  # s
n_time_steps       = int( 20*ring_radius**2 / ring_strength / time_step_size)

# ==================================================
# Properties of the Particle Distribution Region
# ==================================================
box_domain=[-2*ring_thickness,  2*ring_thickness, 
            -2*ring_radius,     2*ring_radius, 
            -2*ring_radius,     2*ring_radius]

# Get the coordinates of regularly-distribuited 
# points and their associated volumes.
positions, volumes = vpm_sh.get_hexagonal_point_distribution(box_domain, spacing=particle_distance)


# ==================================================
# Compute Particle Velocities, Strengths, Radii,
# viscosities and add them to the particle system:
# ==================================================
r_corr_factor = 1.28839 # use this for dr/thickness = 0.25

velocities, strengths, radii, viscosities = vpm_fm.vortex_ring_vpm(
    positions, volumes, particle_radius, particle_viscosity, ring_center, ring_radius, ring_strength, ring_thickness/r_corr_factor
)

# ==================================================
# Initialize the particle system
# ==================================================
particle_system = vpm.ParticleSystem(
    flow_model='Potential',
    time_step_size=time_step_size,
    time_integration_method='RK2', 
    viscous_scheme=None,
    processing_unit='GPU',
    monitor_variables = ['Circulation', 'Linear impulse'],
    backup_filename=backup_filename,
    backup_frequency=25
)

'''
# ==================================================
These are some other options for other flow models:
# ==================================================

particle_system = vpm.ParticleSystem(
    flow_model='pseudo2D',
    time_step_size=time_step_size,
    time_integration_method='RK2', 
    viscous_scheme="CoreSpreading",
    processing_unit='GPU',
    monitor_variables = ['Circulation', 'Kinetic energy'],
    backup_filename=backup_filename,
    backup_frequency=25
)

particle_system = vpm.ParticleSystem(
    flow_model='DNS',
    time_step_size=time_step_size,
    time_integration_method='RK2', 
    viscous_scheme="CoreSpreading",
    processing_unit='GPU',
    monitor_variables = ['Circulation', 'Kinetic energy'],
    backup_filename=backup_filename,
    backup_frequency=25
)

particle_system = vpm.ParticleSystem(
    flow_model='LES', # other options: '
    time_step_size=time_step_size,
    time_integration_method='RK2', 
    viscous_scheme="CoreSpreading",
    processing_unit='GPU',
    monitor_variables = ['Circulation', 'Kinetic energy'],
    backup_filename=backup_filename,
    backup_frequency=25
)

'''

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