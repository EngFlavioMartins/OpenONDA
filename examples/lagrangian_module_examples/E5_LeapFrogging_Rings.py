import numpy as np
import sys
import os

# Import all relevant moduli:
from OpenONDA.solvers.VPM import vpmModule as vpm

import OpenONDA.utilities.vpm_flow_models   as vpm_fm
import OpenONDA.utilities.vpm_solver_helper as vpm_sh
from   OpenONDA.utilities.scripts_helper import remove_files

# ==================================================
# Set up some basic IO
# ==================================================
filesnames      = "Vortex_Ring"
case_directory  = "./E5_LeapFrogging_Rings/Raw"
backup_filename = os.path.join(case_directory, filesnames)

# Ensure the directory exists
os.makedirs(os.path.dirname(backup_filename), exist_ok=True)

log_file_path = './E5_LeapFrogging_Rings/LeapFrogging_Rings.out'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Redirect stdout and stderr to a log file
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

remove_files(case_directory, filesnames)


# ==================================================
# Properties of the Vortex Ring
# ==================================================
ring_radius1     = 1.0              # m, radius of the vortex ring
ring_strength1   = 1.0              # m²/s, vortex strength
ring_thickness1  = 0.1*ring_radius1 # m, thickness of the vortex ring
ring_center1     = np.array([-0.5*ring_radius1, 0.0, 0.0])   # m, center of the vortex ring
epsilon_W1       = 0.04*ring_radius1    # Amplitude of perturbation, m
phase_W1         = 0.0                  # Phase of the perturbation, rad

ring_radius2     = 1.0
ring_strength2   = 1.0     
ring_thickness2  = 0.1*ring_radius2
ring_center2     = np.array([0.5*ring_radius2, 0.0, 0.0])
epsilon_W2       = 0.04*ring_radius2 
phase_W2         = np.pi / 12


# ==================================================
# Particle Distribution Setup
# ==================================================
Re = 3000                                   # Reynolds number
particle_distance  = 0.25*ring_thickness1   # m
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = ring_strength1/Re      # m²/s, kinematic viscosity
time_step_size     = 5 * particle_distance**2/ring_strength1  # s
n_time_steps       = int( 40*ring_radius1**2 / ring_strength1 / time_step_size)


# ==================================================
# Properties of the Particle Distribution Region
# ==================================================
box_domain1=[-2*ring_thickness1+ring_center1[0], 
             2*ring_thickness1+ring_center1[0], 
             -2*ring_radius1,     2*ring_radius1, 
             -2*ring_radius1,     2*ring_radius1]

box_domain2=[-2*ring_thickness2+ring_center2[0],  
             2*ring_thickness2+ring_center2[0], 
             -2*ring_radius2,     2*ring_radius2, 
             -2*ring_radius2,     2*ring_radius2]

positions1, volumes1 = vpm_sh.get_rectangular_point_distribuition(box_domain1, spacing=particle_distance)

positions2, volumes2 = vpm_sh.get_rectangular_point_distribuition(box_domain2, spacing=particle_distance)

# ==================================================
# Compute Particle Velocities, Strengths, Radii,
# viscosities and add them to the particle system:
# ==================================================
r_corr_factor = 1.28839 # use this for dr/thickness = 0.25

velocities1, strengths1, radii1, viscosities1 = vpm_fm.vortex_ring_vpm(
    positions1, volumes1, particle_radius, particle_viscosity, ring_center1, ring_radius1, ring_strength1, ring_thickness1/r_corr_factor, epsilon_W1, phase_W1
)

velocities2, strengths2, radii2, viscosities2 = vpm_fm.vortex_ring_vpm(
    positions2, volumes2, particle_radius, particle_viscosity, ring_center2, ring_radius2, ring_strength2, ring_thickness2/r_corr_factor, epsilon_W2, phase_W2
)

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