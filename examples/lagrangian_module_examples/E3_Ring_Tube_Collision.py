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
filesnames      = "Ring_Tube_Collision"
case_directory  = "./E3_Ring_Tube_Collision/Raw"
backup_filename = os.path.join(case_directory, filesnames)

# Ensure the directory exists
os.makedirs(os.path.dirname(backup_filename), exist_ok=True)

log_file_path = './E3_Ring_Tube_Collision/Ring_Tube_Collision.out'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Redirect stdout and stderr to a log file
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

remove_files(case_directory, filesnames)

# ==================================================
# Properties of the Vortex Ring and tube vortex
# ==================================================
ring_radius     = 1.0              # m, radius of the vortex ring
ring_strength   = 1.0              # m²/s, vortex strength
ring_thickness  = 0.275*ring_radius # m, thickness of the vortex ring
ring_center     = np.array([-1.0*ring_radius, 0.0, 0.0])   # m, center of the vortex ring
epsilon_W       = 0.02*ring_radius    # Amplitude of perturbation, m

vortex_radius   = 0.5*ring_thickness
vortex_center   = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
vortex_strength = ring_strength               # m²/s, vortex strength


# ==================================================
# Particle Distribution Setup
# ==================================================
Re = 5000                                   # Reynolds number
particle_distance  = 0.2*ring_thickness   # m
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = ring_strength/Re      # m²/s, kinematic viscosity
time_step_size     = 5 * particle_distance**2/ring_strength  # s
n_time_steps       = int( 75*ring_radius**2 / ring_strength / time_step_size)


# ==================================================
# Properties of the Particle Distribution Region
# ==================================================
box_domain1=[-2*ring_thickness + ring_center[0], 
              2*ring_thickness + ring_center[0], 
             -2*ring_radius,     2*ring_radius, 
             -2*ring_radius,     2*ring_radius]

box_domain2 = np.array([-2., 2., -2., 2., -25., 25.]) * vortex_radius

positions1, volumes1 = vpm_sh.get_rectangular_point_distribuition(box_domain1, spacing=particle_distance)

positions2, volumes2 = vpm_sh.get_rectangular_point_distribuition(box_domain2, spacing=particle_distance)


# ==================================================
# Compute Particle Velocities, Strengths, Radii,
# viscosities and add them to the particle system:
# ==================================================
velocities1, strengths1, radii1, viscosities1 = vpm_fm.vortex_ring_vpm(
    positions1, volumes1, particle_radius, particle_viscosity, ring_center, ring_radius, ring_strength, ring_thickness, epsilon_W
)

velocities2, strengths2, radii2, viscosities2 = vpm_fm.lamb_oseen_vpm(
    positions2, volumes2, particle_radius, particle_viscosity, vortex_center, vortex_strength, vortex_radius)


# ==================================================
# Initialize the particle system
# ==================================================
particle_system = vpm.ParticleSystem(
    time_step=0,
    flow_model='LES',
    time_step_size=time_step_size,
    time_integration_method='RK2', 
    viscous_scheme='CoreSpreading',
    processing_unit='GPU',
    monitor_variables = ['Circulation', 'Kinetic energy'],
    backup_filename=backup_filename,
    backup_frequency=10
)

particle_system.add_particle_field(positions1, velocities1, strengths1, radii1, viscosities1, group_id=0)

particle_system.add_particle_field(positions2, velocities2, strengths2, radii2, viscosities2, group_id=1)

particle_system.remove_weak_particles(mode='relative', threshold=1e-1, conserve_total_circulation=True)

print(particle_system)

# ==================================================
# Perform the Simulation Over n Time Steps
# ==================================================
for t in range(n_time_steps):
    particle_system.update_velocities()
    particle_system.update_strengths()
    particle_system.update_state()